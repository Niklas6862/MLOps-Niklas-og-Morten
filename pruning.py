from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import mlflow
import torch
from src.compress import apply_pruning, benchmark, evaluate_accuracy, make_pruning_permanent
from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.utils import set_seed, setup_logging
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]

SWEEP_AMOUNTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply L1-unstructured magnitude pruning to a trained model")
    parser.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--model-dir", default="models/artifacts")
    parser.add_argument("--output-dir", default=None, help="Where to save the pruned model")
    parser.add_argument(
        "--prune-amount",
        type=float,
        default=0.3,
        metavar="FRAC",
        help="Fraction of Linear weights to prune (default: 0.3)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep over multiple prune amounts instead of a single run",
    )
    parser.add_argument("--benchmark-batches", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="models/artifacts/pruning_report.json")
    return parser.parse_args()


def _make_test_loader(cfg: dict, processor: AutoImageProcessor, batch_size: int) -> DataLoader:
    raw = load_image_dataset(cfg["dataset"])
    processed = preprocess_dataset(raw, processor, cfg["dataset"])
    processed.set_format("torch", columns=["pixel_values", "labels"])
    return DataLoader(processed["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    setup_logging(cfg.get("project", {}).get("log_level", "INFO"))
    set_seed(cfg.get("project", {}).get("seed", 42))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else model_dir.parent / (model_dir.name + "_pruned")
    )

    processor = AutoImageProcessor.from_pretrained(model_dir)
    batch_size = cfg.get("training", {}).get("per_device_eval_batch_size", 32)
    test_loader = _make_test_loader(cfg, processor, batch_size)

    def _fresh_model() -> AutoModelForImageClassification:
        m = AutoModelForImageClassification.from_pretrained(model_dir)
        m.eval()
        return m

    logger.info("Benchmarking baseline …")
    base_model = _fresh_model()
    base_acc = evaluate_accuracy(base_model, test_loader, device)
    base_lat = benchmark(base_model, test_loader, device, n_batches=args.benchmark_batches)
    logger.info("Baseline: acc=%.4f  throughput=%.1f fps", base_acc, base_lat.get("throughput_fps", 0))
    del base_model

    report: dict = {
        "model_dir": str(model_dir),
        "baseline": {"accuracy": round(base_acc, 4), **base_lat},
    }

    if args.sweep:
        sweep = []
        for amount in SWEEP_AMOUNTS:
            m = _fresh_model()
            stats = apply_pruning(m, amount)
            make_pruning_permanent(m)
            acc = evaluate_accuracy(m, test_loader, device)
            entry = {"prune_amount": amount, "accuracy": round(acc, 4), "accuracy_drop": round(base_acc - acc, 4), **stats}
            sweep.append(entry)
            logger.info("amount=%.0f%%  acc=%.4f  drop=%.4f", amount * 100, acc, base_acc - acc)
            del m
        report["sweep"] = sweep
    else:
        model = _fresh_model()
        prune_stats = apply_pruning(model, args.prune_amount)
        make_pruning_permanent(model)

        prune_acc = evaluate_accuracy(model, test_loader, device)
        prune_lat = benchmark(model, test_loader, device, n_batches=args.benchmark_batches)
        speedup = (
            round(prune_lat["throughput_fps"] / base_lat["throughput_fps"], 2)
            if base_lat.get("throughput_fps") and prune_lat.get("throughput_fps")
            else None
        )
        logger.info(
            "Pruned: acc=%.4f (drop=%.4f)  throughput=%.1f fps%s",
            prune_acc,
            base_acc - prune_acc,
            prune_lat.get("throughput_fps", 0),
            f"  speedup={speedup:.2f}x" if speedup else "",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        run_id_file = model_dir / "run_id.txt"
        if run_id_file.exists():
            shutil.copy(run_id_file, output_dir / "run_id.txt")
        logger.info("Pruned model saved to %s", output_dir)

        report["prune_amount"] = args.prune_amount
        report["output_dir"] = str(output_dir)
        report["pruned"] = {
            "accuracy": round(prune_acc, 4),
            "accuracy_drop": round(base_acc - prune_acc, 4),
            **prune_lat,
            **prune_stats,
            **({"speedup_x": speedup} if speedup else {}),
        }

    # MLflow
    experiment_name = cfg.get("project", {}).get("experiment_name", "image-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_id_file = model_dir / "run_id.txt"
    parent_run_id = run_id_file.read_text().strip() if run_id_file.exists() else None

    with mlflow.start_run(run_name="pruning-sweep" if args.sweep else "pruning"):
        mlflow.set_tags(
            {
                "run_type": "pruning",
                **({"training_run_id": parent_run_id} if parent_run_id else {}),
            }
        )
        mlflow.log_params({"prune_amount": args.prune_amount if not args.sweep else "sweep"})
        for k, v in report["baseline"].items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"baseline_{k}", float(v))
        if not args.sweep:
            for k, v in report["pruned"].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"pruned_{k}", float(v))

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Pruning report saved to %s", report_path)


if __name__ == "__main__":
    main()
