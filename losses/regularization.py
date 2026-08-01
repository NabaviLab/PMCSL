from __future__ import annotations

import torch
from torch import nn


def l2_parameter_penalty(
    model: nn.Module,
    *,
    exclude_bias: bool = True,
    exclude_norm: bool = True,
) -> torch.Tensor:
    """
    Compute the squared L2 norm of trainable parameters.

    This explicit term corresponds to:
        lambda * ||Theta||_2^2

    When AdamW with nonzero weight_decay is used, set the explicit
    regularization coefficient to zero to avoid double regularization.
    """
    device = next(model.parameters()).device
    penalty = torch.zeros((), device=device)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if exclude_bias and name.endswith("bias"):
            continue
        if exclude_norm and ("norm" in name.lower() or parameter.ndim == 1):
            continue

        penalty = penalty + parameter.pow(2).sum()

    return penalty
