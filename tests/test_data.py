from __future__ import annotations

import torch
from PIL import Image


def test_collate_fn_shapes() -> None:
    from src.data import collate_fn

    examples = [
        {"pixel_values": torch.randn(3, 224, 224), "labels": 0},
        {"pixel_values": torch.randn(3, 224, 224), "labels": 2},
    ]
    batch = collate_fn(examples)
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    assert batch["labels"].tolist() == [0, 2]


def test_set_seed_reproducibility() -> None:
    from src.utils import set_seed

    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_build_transform_returns_callable() -> None:
    from unittest.mock import MagicMock

    from src.data import _build_transform

    mock_processor = MagicMock()
    mock_processor.size = {"height": 224}
    mock_processor.image_mean = [0.5, 0.5, 0.5]
    mock_processor.image_std = [0.5, 0.5, 0.5]

    tf = _build_transform(mock_processor, is_train=False)
    result = tf(Image.new("RGB", (256, 256)))
    assert result.shape == (3, 224, 224)
