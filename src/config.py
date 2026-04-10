"""Configuration loading and merging utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (modifies base in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(*config_paths: str | Path) -> dict[str, Any]:
    """Merge one or more YAML config files left-to-right.

    Later files override earlier ones for overlapping keys.

    Example::

        cfg = load_config("configs/base.yaml", "configs/training.yaml")
    """
    merged: dict[str, Any] = {}
    for path in config_paths:
        _deep_merge(merged, load_yaml(path))
    return merged


# Re-export _deep_merge under a public name for tests
deep_merge = _deep_merge
