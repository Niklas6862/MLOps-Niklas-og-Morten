"""Training argument builder and metric computation."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import TrainingArguments


def get_training_args(training_cfg: dict[str, Any]) -> TrainingArguments:
    """Convert the training YAML section into a HuggingFace TrainingArguments object."""
    return TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg.get("num_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 32),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        eval_strategy=training_cfg.get("eval_strategy", "epoch"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "accuracy"),
        greater_is_better=training_cfg.get("greater_is_better", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        fp16=training_cfg.get("fp16", False),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 2),
        report_to=training_cfg.get("report_to", "mlflow"),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        push_to_hub=training_cfg.get("push_to_hub", False),
        remove_unused_columns=False,
    )


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute accuracy from model logits and ground-truth labels.

    Compatible with the HuggingFace Trainer ``compute_metrics`` signature.
    Avoids importing the ``evaluate`` package to prevent name shadowing with
    the root-level ``evaluate.py`` entry point.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}
