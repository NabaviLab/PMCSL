from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .classification import ClassificationLoss
from .regularization import l2_parameter_penalty


@dataclass
class LossOutput:
    total: torch.Tensor
    classification: torch.Tensor
    regularization: torch.Tensor


class TotalLoss(nn.Module):
    """
    Complete objective used for model optimization.

    L = L_CE + lambda * ||Theta||_2^2
    """

    def __init__(
        self,
        *,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        l2_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.classification_loss = ClassificationLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        self.l2_lambda = float(l2_lambda)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
    ) -> LossOutput:
        classification = self.classification_loss(logits, targets)

        if self.l2_lambda > 0.0:
            regularization = l2_parameter_penalty(model)
        else:
            regularization = torch.zeros(
                (),
                device=classification.device,
                dtype=classification.dtype,
            )

        total = classification + self.l2_lambda * regularization

        return LossOutput(
            total=total,
            classification=classification.detach(),
            regularization=regularization.detach(),
        )
