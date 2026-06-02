from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


def test_compute_metrics_perfect() -> None:
    from src.train import compute_metrics

    logits = np.array([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
    labels = np.array([0, 1, 2])
    result = compute_metrics((logits, labels))
    assert result["accuracy"] == pytest.approx(1.0)


def test_predict_image_returns_sorted_scores(tmp_path: Path) -> None:
    from src.infer import predict_image

    img_path = tmp_path / "test.png"
    Image.new("RGB", (224, 224), color=(100, 150, 200)).save(img_path)

    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "angular_leaf_spot", 1: "bean_rust", 2: "healthy"}
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[3.0, 1.0, 0.5]])
    mock_model.return_value = mock_outputs

    results = predict_image(img_path, mock_model, mock_processor, top_k=3)

    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
