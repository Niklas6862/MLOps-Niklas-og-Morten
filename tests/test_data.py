"""Tests for data utilities: collate_fn, transforms, and shared utils."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image


def test_collate_fn_shapes() -> None:
    """collate_fn should stack pixel_values and stack labels correctly."""
    from src.data import collate_fn

    examples = [
        {"pixel_values": torch.randn(3, 224, 224), "labels": 0},
        {"pixel_values": torch.randn(3, 224, 224), "labels": 2},
    ]
    batch = collate_fn(examples)
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    assert batch["labels"].tolist() == [0, 2]
    assert batch["labels"].dtype == torch.long


def test_collate_fn_single_example() -> None:
    from src.data import collate_fn

    examples = [{"pixel_values": torch.zeros(3, 224, 224), "labels": 1}]
    batch = collate_fn(examples)
    assert batch["pixel_values"].shape == (1, 3, 224, 224)
    assert batch["labels"].item() == 1


def test_set_seed_reproducibility() -> None:
    """Same seed should produce identical random tensors."""
    from src.utils import set_seed

    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_set_seed_different_seeds() -> None:
    """Different seeds should produce different tensors (with overwhelming probability)."""
    from src.utils import set_seed

    set_seed(0)
    a = torch.randn(100)
    set_seed(1)
    b = torch.randn(100)
    assert not torch.allclose(a, b)


def test_ensure_dir_creates_nested(tmp_path) -> None:
    from src.utils import ensure_dir

    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.exists()
    assert result.is_dir()


def test_ensure_dir_idempotent(tmp_path) -> None:
    from src.utils import ensure_dir

    target = tmp_path / "existing"
    ensure_dir(target)
    ensure_dir(target)  # Should not raise
    assert target.exists()


def test_build_transform_returns_callable() -> None:
    """_build_transform should return something callable that processes a PIL image."""
    from unittest.mock import MagicMock

    from src.data import _build_transform

    mock_processor = MagicMock()
    mock_processor.size = {"height": 224}
    mock_processor.image_mean = [0.5, 0.5, 0.5]
    mock_processor.image_std = [0.5, 0.5, 0.5]

    for is_train in (True, False):
        tf = _build_transform(mock_processor, is_train=is_train)
        img = Image.new("RGB", (256, 256), color=(128, 64, 32))
        result = tf(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
