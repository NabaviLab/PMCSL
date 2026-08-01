from __future__ import annotations

import torch
from torch import nn


@torch.inference_mode()
def predict_slide(
    model: nn.Module,
    multi_scale_features: dict[str, torch.Tensor],
    *,
    text_tokens: torch.Tensor,
    text_attention_mask: torch.Tensor | None = None,
    device: torch.device,
) -> dict:
    """
    Run inference for one slide or one padded mini-batch.
    """
    model.eval()

    features = {
        scale: tensor.to(device)
        for scale, tensor in multi_scale_features.items()
    }
    text_tokens = text_tokens.to(device)

    if text_attention_mask is not None:
        text_attention_mask = text_attention_mask.to(device)

    output = model(
        multi_scale_features=features,
        text_tokens=text_tokens,
        text_attention_mask=text_attention_mask,
    )

    probabilities = output["probabilities"]
    predicted_class = probabilities.argmax(dim=1)

    return {
        "predicted_class": predicted_class.cpu(),
        "probabilities": probabilities.cpu(),
        "slide_embedding": output["slide_embedding"].cpu(),
        "grade_prototypes": output["grade_prototypes"].cpu(),
        "irm_indices": {
            key: value.cpu()
            for key, value in output["irm_indices"].items()
        },
        "cross_attention_maps": {
            key: value.cpu()
            for key, value in output["cross_attention_maps"].items()
        },
    }
