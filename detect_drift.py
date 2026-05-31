"""Detect data and prediction-distribution drift in the beans image classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance
from src.config import load_config
from src.drift import (
    detect_data_drift,
    detect_prediction_drift,
    extract_embeddings,
    load_reference,
    save_reference,
)
from src.model import load_model_and_processor
from src.utils import get_label_mappings, set_seed, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = ["configs/base.yaml", "configs/data.yaml", "configs/model.yaml"]


def _apply_noise(images: list[Image.Image], std: float) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    out = []
    for img in images:
        arr = np.array(img).astype(np.float32)
        arr += rng.normal(0, std * 255, arr.shape)
        out.append(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
    return out


def _apply_brightness(images: list[Image.Image], factor: float) -> list[Image.Image]:
    return [ImageEnhance.Brightness(img).enhance(factor) for img in images]


def _predict(model, processor, images, device, batch_size=32) -> np.ndarray:
    model.eval()
    preds: list[int] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    return np.array(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift detection for the beans classifier.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--current-split", default="validation")
    parser.add_argument(
        "--load-reference",
        metavar="PATH",
        help="Load pre-computed reference .npz instead of recomputing it.",
    )
    parser.add_argument(
        "--save-reference",
        metavar="PATH",
        help="Save reference embeddings/predictions to a .npz file.",
    )
    parser.add_argument(
        "--artificial-shift",
        choices=["none", "noise", "brightness"],
        default="none",
        help="Perturb the current split to simulate drift (useful for validating the detector).",
    )
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--brightness-factor", type=float, default=0.3)
    parser.add_argument("--output", default="drift_report.json")
    args = parser.parse_args()

    cfg = load_config(*args.configs)
    base_cfg = cfg.get("project", {})
    setup_logging(base_cfg.get("log_level", "INFO"))
    set_seed(base_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_cfg = cfg["dataset"]
    logger.info("Loading dataset '%s'…", data_cfg["name"])
    ds = load_dataset(data_cfg["name"], cache_dir=data_cfg.get("cache_dir", "data/raw"))
    label2id, id2label = get_label_mappings(ds)
    label_names = [id2label[i] for i in sorted(id2label)]

    model_cfg = cfg["model"]
    model, processor = load_model_and_processor(model_cfg, label2id, id2label)
    model = model.to(device)

    # Current data — optionally perturbed
    cur_split = ds[args.current_split]
    cur_images: list[Image.Image] = [ex[data_cfg["image_column"]] for ex in cur_split]
    logger.info("Current split '%s': %d samples.", args.current_split, len(cur_images))

    if args.artificial_shift == "noise":
        logger.info("Applying Gaussian noise (std=%.2f) to simulate data drift.", args.noise_std)
        cur_images = _apply_noise(cur_images, args.noise_std)
    elif args.artificial_shift == "brightness":
        logger.info(
            "Applying brightness factor %.2f to simulate data drift.", args.brightness_factor
        )
        cur_images = _apply_brightness(cur_images, args.brightness_factor)

    logger.info("Extracting current embeddings…")
    cur_emb = extract_embeddings(model, processor, cur_images, device=device)
    logger.info("Running predictions on current data…")
    cur_preds = _predict(model, processor, cur_images, device=device)

    # Reference data
    if args.load_reference:
        ref = load_reference(args.load_reference)
        ref_emb, ref_preds = ref["embeddings"], ref["predictions"]
        ref_label = args.load_reference
    else:
        ref_split = ds[args.reference_split]
        ref_images: list[Image.Image] = [ex[data_cfg["image_column"]] for ex in ref_split]
        logger.info(
            "Reference split '%s': %d samples. Extracting embeddings…",
            args.reference_split,
            len(ref_images),
        )
        ref_emb = extract_embeddings(model, processor, ref_images, device=device)
        logger.info("Running predictions on reference data…")
        ref_preds = _predict(model, processor, ref_images, device=device)
        ref_label = args.reference_split

    if args.save_reference:
        save_reference(ref_emb, ref_preds, args.save_reference)

    logger.info("Running data drift detection (KS test on embedding dims)…")
    data_drift = detect_data_drift(ref_emb, cur_emb)

    logger.info("Running prediction distribution drift detection (KL divergence)…")
    pred_drift = detect_prediction_drift(ref_preds, cur_preds, label_names)

    report = {
        "artificial_shift": args.artificial_shift,
        "reference": ref_label,
        "current_split": args.current_split,
        "reference_samples": int(len(ref_emb)),
        "current_samples": int(len(cur_emb)),
        "data_drift": data_drift,
        "prediction_drift": pred_drift,
        "overall_drift_detected": data_drift["is_drifted"] or pred_drift["is_drifted"],
    }

    # Write local JSON
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Log to MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", base_cfg.get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(base_cfg.get("experiment_name", "image-classifier"))

    with mlflow.start_run(run_name="drift-check"):
        mlflow.set_tags(
            {
                "run_type": "drift-check",
                "artificial_shift": args.artificial_shift,
                "reference": ref_label,
                "current_split": args.current_split,
                "jenkins_build_number": os.getenv("JENKINS_BUILD_NUMBER", "local"),
            }
        )
        mlflow.log_params(
            {
                "model_name": model_cfg["name"],
                "dataset": data_cfg["name"],
                "reference_samples": int(len(ref_emb)),
                "current_samples": int(len(cur_emb)),
                "noise_std": args.noise_std if args.artificial_shift == "noise" else None,
                "brightness_factor": (
                    args.brightness_factor if args.artificial_shift == "brightness" else None
                ),
            }
        )
        mlflow.log_metrics(
            {
                "data_drift_fraction": data_drift["drift_fraction"],
                "data_drift_detected": int(data_drift["is_drifted"]),
                "data_drift_mean_p_value": data_drift["mean_p_value"],
                "data_drift_min_p_value": data_drift["min_p_value"],
                "data_drift_n_drifted_dims": data_drift["n_drifted_dims"],
                "pred_drift_kl_divergence": pred_drift["kl_divergence"],
                "pred_drift_detected": int(pred_drift["is_drifted"]),
                "overall_drift_detected": int(report["overall_drift_detected"]),
            }
        )
        mlflow.log_artifact(args.output, artifact_path="drift")
        logger.info(
            "Drift results logged to MLflow run under experiment '%s'.",
            base_cfg.get("experiment_name"),
        )

    if report["overall_drift_detected"]:
        logger.warning("DRIFT DETECTED — consider retraining or investigating the data shift.")
    else:
        logger.info("No significant drift detected.")


if __name__ == "__main__":
    main()
