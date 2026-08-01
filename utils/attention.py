from __future__ import annotations

import numpy as np
import torch


def aggregate_attention_heads(
    attention: torch.Tensor,
    *,
    method: str = "mean",
) -> torch.Tensor:
    """
    Aggregate multi-head cross-attention.

    Expected shape:
        [B, H, N_visual, N_text]
    """
    if attention.ndim != 4:
        raise ValueError(
            f"Expected [B,H,N_visual,N_text], got {tuple(attention.shape)}"
        )

    if method == "mean":
        return attention.mean(dim=1)

    if method == "max":
        return attention.max(dim=1).values

    raise ValueError("method must be 'mean' or 'max'.")


def visual_token_importance(
    attention: torch.Tensor,
    *,
    head_reduction: str = "mean",
    text_reduction: str = "mean",
) -> torch.Tensor:
    """
    Convert cross-attention into one importance score per visual token.

    Returns:
        [B, N_visual]
    """
    reduced = aggregate_attention_heads(
        attention,
        method=head_reduction,
    )

    if text_reduction == "mean":
        return reduced.mean(dim=-1)

    if text_reduction == "max":
        return reduced.max(dim=-1).values

    raise ValueError("text_reduction must be 'mean' or 'max'.")


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = values.min()
    maximum = values.max()

    if maximum <= minimum:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)
