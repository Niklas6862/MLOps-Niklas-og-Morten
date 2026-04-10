"""Tests for src/config.py — config loading and merging."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import deep_merge, load_config, load_yaml


def test_load_yaml_basic(tmp_path: Path) -> None:
    data = {"key": "value", "nested": {"a": 1, "b": 2}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    assert load_yaml(p) == data


def test_load_yaml_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    result = load_yaml(p)
    assert result == {}


def test_load_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "does_not_exist.yaml")


def test_deep_merge_flat() -> None:
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested() -> None:
    base = {"outer": {"x": 1, "y": 2}, "top": "keep"}
    override = {"outer": {"y": 99, "z": 0}}
    result = deep_merge(base, override)
    assert result["outer"]["x"] == 1      # untouched
    assert result["outer"]["y"] == 99     # overridden
    assert result["outer"]["z"] == 0      # added
    assert result["top"] == "keep"        # untouched top-level key


def test_load_config_single(tmp_path: Path) -> None:
    cfg = {"project": {"name": "test", "seed": 42}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    loaded = load_config(p)
    assert loaded["project"]["seed"] == 42


def test_load_config_merge_multiple(tmp_path: Path) -> None:
    cfg1 = {"project": {"name": "base", "seed": 42}}
    cfg2 = {"project": {"seed": 99, "new_key": "hello"}}
    p1 = tmp_path / "cfg1.yaml"
    p2 = tmp_path / "cfg2.yaml"
    p1.write_text(yaml.dump(cfg1))
    p2.write_text(yaml.dump(cfg2))

    merged = load_config(p1, p2)
    assert merged["project"]["name"] == "base"   # from cfg1
    assert merged["project"]["seed"] == 99       # overridden by cfg2
    assert merged["project"]["new_key"] == "hello"  # added by cfg2


def test_load_config_all_sections(
    tmp_base_config: Path,
    tmp_data_config: Path,
    tmp_model_config: Path,
    tmp_training_config: Path,
) -> None:
    cfg = load_config(
        tmp_base_config,
        tmp_data_config,
        tmp_model_config,
        tmp_training_config,
    )
    assert "project" in cfg
    assert "dataset" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert cfg["project"]["seed"] == 42
    assert cfg["dataset"]["name"] == "beans"
    assert cfg["model"]["num_labels"] == 3
