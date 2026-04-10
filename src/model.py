"""Model and processor loading helpers."""
from __future__ import annotations

import logging
from typing import Any

from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_cfg: dict[str, Any],
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> tuple[AutoModelForImageClassification, AutoImageProcessor]:
    """Load a pre-trained HuggingFace vision model and its image processor.

    The classification head is re-initialised to match *label2id* / *id2label*.
    ``ignore_mismatched_sizes=True`` suppresses the weight mismatch warning for
    the classifier head.
    """
    name = model_cfg["name"]
    cache = model_cfg.get("cache_dir", "models/hf_cache")
    n_labels = model_cfg.get("num_labels", len(label2id))

    logger.info("Loading processor from '%s' …", name)
    processor = AutoImageProcessor.from_pretrained(name, cache_dir=cache)

    logger.info("Loading model '%s' with %d labels …", name, n_labels)
    model = AutoModelForImageClassification.from_pretrained(
        name,
        num_labels=n_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=model_cfg.get("ignore_mismatched_sizes", True),
        cache_dir=cache,
    )
    return model, processor
