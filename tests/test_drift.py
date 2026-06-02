from __future__ import annotations

from pathlib import Path

import numpy as np


def test_detect_data_drift_identical() -> None:
    from src.drift import detect_data_drift

    embeddings = np.random.default_rng(0).random((50, 128))
    result = detect_data_drift(embeddings, embeddings)
    assert result["is_drifted"] is False


def test_detect_data_drift_different() -> None:
    from src.drift import detect_data_drift

    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, (100, 128))
    cur = rng.normal(5, 1, (100, 128))  # clearly shifted distribution
    result = detect_data_drift(ref, cur)
    assert result["is_drifted"] is True
    assert "drift_fraction" in result


def test_detect_prediction_drift_identical() -> None:
    from src.drift import detect_prediction_drift

    preds = np.array([0, 1, 2, 0, 1, 2])
    result = detect_prediction_drift(preds, preds, ["a", "b", "c"])
    assert result["kl_divergence"] == 0.0
    assert result["is_drifted"] is False


def test_save_and_load_reference(tmp_path: Path) -> None:
    from src.drift import load_reference, save_reference

    embeddings = np.ones((10, 4))
    predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    path = tmp_path / "ref.npz"

    save_reference(embeddings, predictions, path)
    loaded = load_reference(path)

    np.testing.assert_array_equal(loaded["embeddings"], embeddings)
    np.testing.assert_array_equal(loaded["predictions"], predictions)
