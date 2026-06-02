from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.config import deep_merge, load_config, load_yaml


def test_load_yaml_basic(tmp_path: Path) -> None:
    data = {"key": "value", "nested": {"a": 1}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    assert load_yaml(p) == data


def test_load_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "does_not_exist.yaml")


def test_deep_merge_nested() -> None:
    base = {"outer": {"x": 1, "y": 2}}
    override = {"outer": {"y": 99, "z": 0}}
    result = deep_merge(base, override)
    assert result["outer"] == {"x": 1, "y": 99, "z": 0}


def test_load_config_merge_multiple(tmp_path: Path) -> None:
    p1 = tmp_path / "cfg1.yaml"
    p2 = tmp_path / "cfg2.yaml"
    p1.write_text(yaml.dump({"project": {"name": "base", "seed": 42}}))
    p2.write_text(yaml.dump({"project": {"seed": 99}}))
    merged = load_config(p1, p2)
    assert merged["project"]["name"] == "base"
    assert merged["project"]["seed"] == 99
