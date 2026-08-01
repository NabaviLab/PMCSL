from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(
    payload: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=indent,
            ensure_ascii=False,
            allow_nan=True,
        )


def read_yaml(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            sort_keys=False,
        )


def save_dataframe(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> None:
    """Save a DataFrame according to the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame.to_csv(path, index=index)
    elif suffix in {".xlsx", ".xls"}:
        frame.to_excel(path, index=index)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=index)
    else:
        raise ValueError(
            f"Unsupported dataframe output format: {path.suffix}"
        )
