"""Evaluation entry point.

Usage::

    python evaluate.py
    python evaluate.py --model-dir models/artifacts --split test --output models/artifacts/eval_results.json

Loads a fine-tuned model from disk, runs it over a dataset split, and saves
detailed per-class metrics to JSON.

Note: this file is named ``evaluate.py`` intentionally as a CLI entry point.
All imports from the ``src`` package use ``src.eval`` to avoid shadowing any
installed package named ``evaluate``.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

from src.config import load_config
from src.data import collate_fn, load_image_dataset, preprocess_dataset
from src.eval import compute_detailed_metrics, save_results
from src.train import compute_metrics
from src.utils import get_label_mappings, set_seed, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned image classifier")
    parser.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS, metavar="PATH")
    parser.add_argument("--model-dir", default="models/artifacts", help="Saved model directory")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument(
        "--output",
        default="models/artifacts/eval_results.json",
        help="JSON file for results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    base_cfg = cfg.get("project", {})
    data_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})

    setup_logging(base_cfg.get("log_level", "INFO"))
    set_seed(base_cfg.get("seed", 42))

    # Load model
    logger.info("Loading model from '%s' …", args.model_dir)
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)

    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}

    # Load and preprocess data
    raw_ds = load_image_dataset(data_cfg)
    processed_ds = preprocess_dataset(raw_ds, processor, data_cfg)
    processed_ds.set_format("torch", columns=["pixel_values", "labels"])

    split_ds = processed_ds.get(args.split)
    if split_ds is None:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(processed_ds.keys())}")

    eval_args = TrainingArguments(
        output_dir=training_cfg.get("output_dir", "models/artifacts"),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 32),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=split_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # Run evaluation
    logger.info("Evaluating on split '%s' …", args.split)
    hf_results = trainer.evaluate()
    logger.info("HuggingFace metrics: %s", hf_results)

    # Collect all predictions for detailed metrics
    predictions_output = trainer.predict(split_ds)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    detailed = compute_detailed_metrics(logits, np.array(labels), id2label)
    save_results(detailed, args.output)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
