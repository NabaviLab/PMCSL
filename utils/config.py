from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            deepcopy(config),
            handle,
            sort_keys=False,
            default_flow_style=False,
        )


def get_nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Read a nested key using dot notation.

    Example:
        get_nested(cfg, "training.learning_rate")
    """
    current: Any = config

    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current
