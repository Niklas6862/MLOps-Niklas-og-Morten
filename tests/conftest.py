"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def tmp_base_config(tmp_path: Path) -> Path:
    cfg = {
        "project": {
            "name": "test-project",
            "seed": 42,
            "output_dir": str(tmp_path / "output"),
            "mlflow_tracking_uri": str(tmp_path / "mlruns"),
            "experiment_name": "test-exp",
            "log_level": "WARNING",
        }
    }
    p = tmp_path / "base.yaml"
    p.write_text(yaml.dump(cfg))
    return p


@pytest.fixture()
def tmp_data_config(tmp_path: Path) -> Path:
    cfg = {
        "dataset": {
            "name": "beans",
            "split_train": "train",
            "split_val": "validation",
            "split_test": "test",
            "image_column": "image",
            "label_column": "labels",
            "cache_dir": str(tmp_path / "data"),
        }
    }
    p = tmp_path / "data.yaml"
    p.write_text(yaml.dump(cfg))
    return p


@pytest.fixture()
def tmp_model_config(tmp_path: Path) -> Path:
    cfg = {
        "model": {
            "name": "google/vit-base-patch16-224",
            "num_labels": 3,
            "ignore_mismatched_sizes": True,
            "cache_dir": str(tmp_path / "hf_cache"),
        }
    }
    p = tmp_path / "model.yaml"
    p.write_text(yaml.dump(cfg))
    return p


@pytest.fixture()
def tmp_training_config(tmp_path: Path) -> Path:
    cfg = {
        "training": {
            "output_dir": str(tmp_path / "output"),
            "num_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "learning_rate": 2e-5,
            "report_to": "none",
            "save_total_limit": 1,
        }
    }
    p = tmp_path / "training.yaml"
    p.write_text(yaml.dump(cfg))
    return p
