from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def flatten_dict(
    mapping: Mapping[str, Any],
    *,
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten a nested mapping for logging."""
    items: dict[str, Any] = {}

    for key, value in mapping.items():
        composed = (
            f"{parent_key}{separator}{key}"
            if parent_key
            else str(key)
        )

        if isinstance(value, Mapping):
            items.update(
                flatten_dict(
                    value,
                    parent_key=composed,
                    separator=separator,
                )
            )
        else:
            items[composed] = value

    return items


def count_trainable_parameters(model) -> int:
    """Count trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
