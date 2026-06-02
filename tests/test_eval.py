from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_compute_detailed_metrics_perfect() -> None:
    from src.eval import compute_detailed_metrics

    logits = np.array([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
    labels = np.array([0, 1, 2])
    id2label = {0: "cat", 1: "dog", 2: "bird"}
    result = compute_detailed_metrics(logits, labels, id2label)

    assert result["accuracy"] == 1.0
    assert set(result["per_class"].keys()) == {"cat", "dog", "bird"}
    for cls in result["per_class"].values():
        assert cls["f1"] == 1.0


def test_save_results_creates_json(tmp_path: Path) -> None:
    from src.eval import save_results

    data = {"accuracy": 0.95, "per_class": {"cat": {"f1": 0.94}}}
    output = tmp_path / "subdir" / "results.json"
    save_results(data, output)

    assert output.exists()
    loaded = json.loads(output.read_text())
    assert loaded["accuracy"] == 0.95
