"""Dataset loading and image preprocessing for HuggingFace image classification."""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torchvision.transforms as T
from datasets import DatasetDict, load_dataset
from transformers import AutoImageProcessor

logger = logging.getLogger(__name__)


def load_image_dataset(data_cfg: dict[str, Any]) -> DatasetDict:
    """Download (or load from cache) a HuggingFace image dataset."""
    name = data_cfg["name"]
    logger.info("Loading dataset '%s' …", name)
    ds: DatasetDict = load_dataset(name, cache_dir=data_cfg.get("cache_dir", "data/raw"))
    logger.info("Splits available: %s", list(ds.keys()))
    return ds


def _build_transform(processor: AutoImageProcessor, is_train: bool) -> Callable:
    """Return a torchvision transform pipeline for train or eval."""
    h = processor.size.get("height", 224)
    mean = processor.image_mean
    std = processor.image_std

    if is_train:
        return T.Compose(
            [
                T.RandomResizedCrop(h),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    return T.Compose(
        [
            T.Resize(h),
            T.CenterCrop(h),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def preprocess_dataset(
    dataset: DatasetDict,
    processor: AutoImageProcessor,
    data_cfg: dict[str, Any],
) -> DatasetDict:
    """Apply image transforms to every split and return a new DatasetDict.

    The processed dataset will have columns:
        - ``pixel_values``: torch.Tensor of shape (C, H, W)
        - ``labels``: int
    """
    image_col = data_cfg.get("image_column", "image")
    label_col = data_cfg.get("label_column", "labels")

    train_tf = _build_transform(processor, is_train=True)
    eval_tf = _build_transform(processor, is_train=False)

    def _apply(tf: Callable) -> Callable:
        def transform_fn(batch: dict) -> dict:
            batch["pixel_values"] = [
                tf(img.convert("RGB")) for img in batch[image_col]
            ]
            return batch

        return transform_fn

    split_map = {
        "train": _apply(train_tf),
        "validation": _apply(eval_tf),
        "test": _apply(eval_tf),
    }

    processed: dict[str, Any] = {}
    for split, fn in split_map.items():
        if split not in dataset:
            continue
        logger.info("Preprocessing split '%s' …", split)
        processed[split] = dataset[split].map(
            fn,
            remove_columns=[image_col],
            batched=True,
            batch_size=32,
            desc=f"Preprocessing {split}",
        )
        # Rename label column to the standard 'labels' if needed
        if label_col != "labels" and label_col in processed[split].column_names:
            processed[split] = processed[split].rename_column(label_col, "labels")

    return DatasetDict(processed)


def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
    """Collate a list of dataset examples into a batched dict."""
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}
