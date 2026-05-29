"""Training entry point.

Usage::

    python train.py
    python train.py --config configs/base.yaml configs/data.yaml configs/model.yaml configs/training.yaml

The script loads all YAML configs, sets up MLflow experiment tracking, preprocesses
the dataset, and launches a HuggingFace Trainer run.  After training it logs:

- Lineage tags (git commit, Docker image, Jenkins build number)
- Model artifact (transformers flavor, enabling model registry)
- Model card YAML summarising provenance and train metrics
- ``models/artifacts/run_id.txt`` so downstream pipeline stages can resume the run
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.model import load_model_and_processor
from src.train import compute_metrics, get_training_args
from src.utils import ensure_dir, get_label_mappings, set_seed, setup_logging
from transformers import Trainer

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ViT image classifier")
    parser.add_argument(
        "--config",
        nargs="+",
        default=DEFAULT_CONFIGS,
        metavar="PATH",
        help="One or more YAML config files (merged left-to-right)",
    )
    return parser.parse_args()


def _build_model_card(cfg: dict, metrics: dict, run_id: str) -> dict:
    """Assemble a minimal model card capturing provenance and train metrics."""
    base = cfg.get("project", {})
    model = cfg.get("model", {})
    data = cfg.get("dataset", {})
    train = cfg.get("training", {})
    return {
        "model_name": model.get("name"),
        "dataset": data.get("name"),
        "task": "image-classification",
        "mlflow_run_id": run_id,
        "git_commit": os.getenv("GIT_COMMIT_HASH", "unknown"),
        "docker_image": os.getenv("DOCKER_IMAGE_TAG", "local"),
        "jenkins_build": os.getenv("JENKINS_BUILD_NUMBER", "local"),
        "hyperparameters": {
            "num_epochs": train.get("num_epochs", 3),
            "learning_rate": train.get("learning_rate", 2e-5),
            "train_batch_size": train.get("per_device_train_batch_size", 16),
            "seed": base.get("seed", 42),
        },
        "train_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    base_cfg = cfg.get("project", {})
    data_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    setup_logging(base_cfg.get("log_level", "INFO"))
    set_seed(base_cfg.get("seed", 42))

    output_dir = Path(training_cfg["output_dir"])
    ensure_dir(output_dir)
    ensure_dir(data_cfg.get("cache_dir", "data/raw"))
    ensure_dir(model_cfg.get("cache_dir", "models/hf_cache"))

    # Respect MLFLOW_TRACKING_URI env var so Jenkins can point at the cluster server
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", base_cfg.get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(base_cfg.get("experiment_name", "image-classifier"))

    with mlflow.start_run() as run:
        # --- Lineage tags (populated by the Jenkins pipeline via env vars) ---
        mlflow.set_tags(
            {
                "jenkins_build_number": os.getenv("JENKINS_BUILD_NUMBER", "local"),
                "docker_image": os.getenv("DOCKER_IMAGE_TAG", "local"),
                "git_commit": os.getenv("GIT_COMMIT_HASH", "unknown"),
            }
        )

        mlflow.log_params(
            {
                "model_name": model_cfg["name"],
                "dataset": data_cfg["name"],
                "seed": base_cfg.get("seed", 42),
                "num_epochs": training_cfg.get("num_epochs", 3),
                "learning_rate": training_cfg.get("learning_rate", 2e-5),
                "train_batch_size": training_cfg.get("per_device_train_batch_size", 16),
            }
        )

        # Persist run ID so evaluate / register / deploy stages can resume this run
        run_id_path = output_dir / "run_id.txt"
        run_id_path.write_text(run.info.run_id)
        logger.info("MLflow run ID: %s", run.info.run_id)

        # --- Data ---
        raw_ds = load_image_dataset(data_cfg)
        label2id, id2label = get_label_mappings(raw_ds)
        logger.info("Label mapping: %s", id2label)

        # --- Model ---
        model, processor = load_model_and_processor(model_cfg, label2id, id2label)

        # --- Preprocessing ---
        processed_ds = preprocess_dataset(raw_ds, processor, data_cfg)
        processed_ds.set_format("torch", columns=["pixel_values", "labels"])

        # --- Trainer (MLflowCallback uses the active run for step-level metrics) ---
        training_args = get_training_args(training_cfg)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_ds["train"],
            eval_dataset=processed_ds.get("validation"),
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

        logger.info("Starting training …")
        train_result = trainer.train()
        logger.info("Training finished. Metrics: %s", train_result.metrics)

        # --- Save model to disk ---
        logger.info("Saving model to '%s' …", output_dir)
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))

        # --- Log model to MLflow (transformers flavor enables model registry) ---
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "image_processor": processor},
            artifact_path="model",
            task="image-classification",
        )
        logger.info("Model artifact logged to MLflow.")

        # --- Model card ---
        card = _build_model_card(cfg, train_result.metrics, run.info.run_id)
        card_path = output_dir / "model_card.yaml"
        with open(card_path, "w") as fh:
            yaml.dump(card, fh, default_flow_style=False)
        mlflow.log_artifact(str(card_path), artifact_path="model_card")
        logger.info("Model card logged.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
