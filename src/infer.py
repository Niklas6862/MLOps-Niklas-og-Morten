"""Core inference logic (reusable from scripts or notebooks)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)


def load_pipeline(
    model_dir: str | Path,
) -> tuple[AutoModelForImageClassification, AutoImageProcessor]:
    """Load a fine-tuned model and its processor from *model_dir*."""
    logger.info("Loading fine-tuned model from '%s' …", model_dir)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()
    return model, processor


def predict_image(
    image_path: str | Path,
    model: AutoModelForImageClassification,
    processor: AutoImageProcessor,
    top_k: int = 3,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run a single-image forward pass and return top-k predictions.

    Returns a list of dicts, each with ``label`` (str) and ``score`` (float).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    k = min(top_k, probs.shape[0])
    top_probs, top_ids = torch.topk(probs, k)

    return [
        {
            "label": model.config.id2label[int(idx)],
            "score": round(float(prob), 4),
        }
        for prob, idx in zip(top_probs, top_ids)
    ]
