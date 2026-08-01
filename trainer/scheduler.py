from __future__ import annotations

import math

import torch


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.01,
):
    """
    Cosine learning-rate schedule with optional linear warmup.
    """

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max((epoch + 1) / warmup_epochs, 1e-8)

        progress = (
            epoch - warmup_epochs
        ) / max(total_epochs - warmup_epochs - 1, 1)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )
