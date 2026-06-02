from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from src.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pruning degree vs accuracy from a sweep report"
    )
    parser.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--report", default="models/artifacts/pruning_report.json")
    parser.add_argument("--output", default="models/artifacts/pruning_curve.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    with open(args.report) as f:
        report = json.load(f)

    sweep = report.get("sweep", [])
    if not sweep:
        logger.error("No sweep data found in %s", args.report)
        return

    baseline_acc = report["baseline"]["accuracy"]
    amounts = [e["prune_amount"] for e in sweep]
    accuracies = [e["accuracy"] for e in sweep]
    drops = [e["accuracy_drop"] for e in sweep]

    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(baseline_acc, color="gray", linestyle="--", label=f"Baseline ({baseline_acc:.3f})")
    ax.plot([a * 100 for a in amounts], accuracies, marker="o", label="Pruned accuracy")
    ax.set_xlabel("Pruning amount (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pruning Degree vs Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    logger.info("Pruning curve saved to %s", args.output)

    experiment_name = cfg.get("project", {}).get("experiment_name", "image-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="pruning-curve-plot") as run:
        mlflow.set_tag("run_type", "pruning_analysis")
        for amount, acc, drop in zip(amounts, accuracies, drops):  # noqa: B905
            step = int(amount * 100)
            mlflow.log_metric("pruned_accuracy", acc, step=step)
            mlflow.log_metric("accuracy_drop", drop, step=step)
        mlflow.log_artifact(args.output)
        mlflow.log_artifact(args.report)
        logger.info("Pruning analysis logged as MLflow run %s", run.info.run_id)


if __name__ == "__main__":
    main()
