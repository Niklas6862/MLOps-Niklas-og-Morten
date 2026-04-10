"""Tests for model utilities, inference logic, and metric computation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


def test_compute_metrics_perfect() -> None:
    """compute_metrics should return accuracy=1.0 for perfect predictions."""
    from src.train import compute_metrics

    logits = np.array([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
    labels = np.array([0, 1, 2])
    result = compute_metrics((logits, labels))
    assert "accuracy" in result
    assert result["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_zero() -> None:
    """compute_metrics should return accuracy=0.0 for all-wrong predictions."""
    from src.train import compute_metrics

    logits = np.array([[0.1, 2.0, 0.1], [0.1, 0.1, 2.0], [2.0, 0.1, 0.1]])
    labels = np.array([0, 1, 2])
    result = compute_metrics((logits, labels))
    assert result["accuracy"] == pytest.approx(0.0)


def test_compute_detailed_metrics_structure() -> None:
    """compute_detailed_metrics should include accuracy and per-class keys."""
    from src.eval import compute_detailed_metrics

    logits = np.array([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
    labels = np.array([0, 1, 2])
    id2label = {0: "cat", 1: "dog", 2: "bird"}
    result = compute_detailed_metrics(logits, labels, id2label)

    assert "accuracy" in result
    assert "per_class" in result
    assert set(result["per_class"].keys()) == {"cat", "dog", "bird"}
    for cls_metrics in result["per_class"].values():
        assert "precision" in cls_metrics
        assert "recall" in cls_metrics
        assert "f1" in cls_metrics


def test_predict_image_returns_sorted_scores(tmp_path: Path) -> None:
    """predict_image should return top-k results sorted by score descending."""
    from src.infer import predict_image

    # Create a dummy image
    img_path = tmp_path / "test.png"
    Image.new("RGB", (224, 224), color=(100, 150, 200)).save(img_path)

    # Mock model and processor
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "angular_leaf_spot", 1: "bean_rust", 2: "healthy"}
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[3.0, 1.0, 0.5]])
    mock_model.return_value = mock_outputs

    results = predict_image(img_path, mock_model, mock_processor, top_k=3)

    assert len(results) == 3
    assert all("label" in r and "score" in r for r in results)
    # Scores should be in descending order
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_predict_image_top_k_clipped(tmp_path: Path) -> None:
    """top_k larger than num_classes should be silently clipped."""
    from src.infer import predict_image

    img_path = tmp_path / "test.png"
    Image.new("RGB", (224, 224)).save(img_path)

    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "a", 1: "b"}
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[1.0, 0.5]])
    mock_model.return_value = mock_outputs

    results = predict_image(img_path, mock_model, mock_processor, top_k=99)
    assert len(results) == 2  # capped at 2 classes


def test_save_results_creates_json(tmp_path: Path) -> None:
    from src.eval import save_results

    data = {"accuracy": 0.95, "per_class": {"cat": {"f1": 0.94}}}
    output = tmp_path / "subdir" / "results.json"
    save_results(data, output)

    assert output.exists()
    import json

    loaded = json.loads(output.read_text())
    assert loaded["accuracy"] == 0.95
