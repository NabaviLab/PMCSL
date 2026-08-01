from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss for four-class Gleason grading.

    Optional class weights can be supplied to address class imbalance.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits [B,C], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"Expected targets [B], got {tuple(targets.shape)}")

        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
