"""Shared utilities: seeding, logging, path helpers, label mappings."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from datasets import DatasetDict


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist; return a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_label_mappings(dataset: "DatasetDict") -> tuple[dict[str, int], dict[int, str]]:
    """Extract label2id / id2label from a HuggingFace DatasetDict.

    Assumes the label column uses a ClassLabel feature.
    """
    features = dataset["train"].features
    # Try common label column names
    for col in ("labels", "label"):
        if col in features and hasattr(features[col], "names"):
            names: list[str] = features[col].names
            id2label = {i: name for i, name in enumerate(names)}
            label2id = {name: i for i, name in id2label.items()}
            return label2id, id2label
    raise ValueError(
        "Could not find a ClassLabel feature in the dataset. "
        "Check 'image_column' and 'label_column' in configs/data.yaml."
    )
