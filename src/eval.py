"""Evaluation helpers: per-class metrics and results serialisation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_detailed_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
) -> dict[str, Any]:
    """Return accuracy + per-class precision/recall/f1 without heavy deps.

    Uses only numpy so there is no dependency on the ``evaluate`` package
    (which would conflict with ``evaluate.py`` at the project root).
    """
    predictions = np.argmax(logits, axis=-1)
    n_classes = len(id2label)
    accuracy = float((predictions == labels).mean())

    per_class: dict[str, dict[str, float]] = {}
    for cls_id, cls_name in id2label.items():
        tp = int(((predictions == cls_id) & (labels == cls_id)).sum())
        fp = int(((predictions == cls_id) & (labels != cls_id)).sum())
        fn = int(((predictions != cls_id) & (labels == cls_id)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int((labels == cls_id).sum()),
        }

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "num_classes": n_classes,
        "num_samples": len(labels),
    }


def save_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Serialise evaluation results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results saved to %s", path)
