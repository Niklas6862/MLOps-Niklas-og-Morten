"""Embedding-based data drift and prediction distribution drift detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

_N_TEST_DIMS = 100


def extract_embeddings(
    model: Any,
    processor: Any,
    images: list,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return CLS-token embeddings from the last ViT hidden layer; shape (n, hidden_dim)."""
    model.eval()
    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)
        logger.debug("Embedded batch %d–%d.", start, start + len(batch))
    return np.concatenate(all_embeddings, axis=0)


def save_reference(embeddings: np.ndarray, predictions: np.ndarray, path: str | Path) -> None:
    """Save reference embeddings and predictions to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, predictions=predictions)
    logger.info("Saved reference → '%s' (%d samples).", path, len(embeddings))


def load_reference(path: str | Path) -> dict[str, np.ndarray]:
    """Load a reference distribution saved by save_reference."""
    data = np.load(path)
    logger.info("Loaded reference from '%s' (%d samples).", path, len(data["embeddings"]))
    return {"embeddings": data["embeddings"], "predictions": data["predictions"]}


def detect_data_drift(
    ref_embeddings: np.ndarray,
    cur_embeddings: np.ndarray,
    threshold: float = 0.05,
    n_test_dims: int = _N_TEST_DIMS,
    drift_fraction_threshold: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """KS test on randomly sampled embedding dims with Bonferroni correction.

    Flags drift when > drift_fraction_threshold fraction of tested dims reject H0.
    """
    rng = np.random.default_rng(seed)
    n_dims = ref_embeddings.shape[1]
    n_test = min(n_dims, n_test_dims)
    sampled_dims = rng.choice(n_dims, size=n_test, replace=False)

    corrected_threshold = threshold / n_test
    p_values: list[float] = []
    for dim in sampled_dims:
        _, p = ks_2samp(ref_embeddings[:, dim], cur_embeddings[:, dim])
        p_values.append(float(p))

    n_drifted = sum(p < corrected_threshold for p in p_values)
    drift_fraction = n_drifted / n_test

    return {
        "is_drifted": drift_fraction > drift_fraction_threshold,
        "drift_fraction": round(drift_fraction, 4),
        "n_drifted_dims": n_drifted,
        "n_tested_dims": n_test,
        "mean_p_value": round(float(np.mean(p_values)), 6),
        "min_p_value": round(float(np.min(p_values)), 6),
        "bonferroni_threshold": round(corrected_threshold, 8),
    }


def detect_prediction_drift(
    ref_predictions: np.ndarray,
    cur_predictions: np.ndarray,
    label_names: list[str],
    kl_threshold: float = 0.10,
) -> dict[str, Any]:
    """KL divergence between reference and current predicted class distributions."""
    n_classes = len(label_names)
    eps = 1e-10
    ref_dist = np.bincount(ref_predictions, minlength=n_classes) / len(ref_predictions)
    cur_dist = np.bincount(cur_predictions, minlength=n_classes) / len(cur_predictions)
    kl_div = float(np.sum(ref_dist * np.log((ref_dist + eps) / (cur_dist + eps))))

    return {
        "is_drifted": kl_div > kl_threshold,
        "kl_divergence": round(kl_div, 6),
        "kl_threshold": kl_threshold,
        "reference_distribution": {
            label_names[i]: round(float(ref_dist[i]), 4) for i in range(n_classes)
        },
        "current_distribution": {
            label_names[i]: round(float(cur_dist[i]), 4) for i in range(n_classes)
        },
    }
