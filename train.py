"""Training entry point.

Usage::

    python train.py
    python train.py --config configs/base.yaml configs/data.yaml configs/model.yaml configs/training.yaml

The script loads all YAML configs, sets up MLflow experiment tracking, preprocesses
the dataset, and launches a HuggingFace Trainer run.
"""
from __future__ import annotations

import argparse
import logging

import mlflow
from transformers import Trainer

from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.model import load_model_and_processor
from src.train import compute_metrics, get_training_args
from src.utils import ensure_dir, get_label_mappings, set_seed, setup_logging

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


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    base_cfg = cfg.get("project", {})
    data_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    setup_logging(base_cfg.get("log_level", "INFO"))
    set_seed(base_cfg.get("seed", 42))

    ensure_dir(training_cfg["output_dir"])
    ensure_dir(data_cfg.get("cache_dir", "data/raw"))
    ensure_dir(model_cfg.get("cache_dir", "models/hf_cache"))

    # MLflow experiment tracking
    mlflow.set_tracking_uri(base_cfg.get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment(base_cfg.get("experiment_name", "image-classifier"))
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

    # Data
    raw_ds = load_image_dataset(data_cfg)
    label2id, id2label = get_label_mappings(raw_ds)
    logger.info("Label mapping: %s", id2label)

    # Model
    model, processor = load_model_and_processor(model_cfg, label2id, id2label)

    # Preprocessing
    processed_ds = preprocess_dataset(raw_ds, processor, data_cfg)
    processed_ds.set_format("torch", columns=["pixel_values", "labels"])

    # Trainer
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

    # Save artifacts
    logger.info("Saving model to '%s' …", training_cfg["output_dir"])
    trainer.save_model(training_cfg["output_dir"])
    processor.save_pretrained(training_cfg["output_dir"])

    # Log final metrics to MLflow
    mlflow.log_metrics(
        {k: v for k, v in train_result.metrics.items() if isinstance(v, (int, float))}
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
