from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from tqdm import tqdm

from .metrics import compute_classification_metrics


def move_feature_dict(
    features: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        scale: tensor.to(device, non_blocking=True)
        for scale, tensor in features.items()
    }


class Evaluator:
    """
    Validation/test evaluator for slide-level predictions.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        use_amp: bool = True,
    ) -> None:
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")

    @torch.inference_mode()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: Iterable,
        *,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
        desc: str = "Evaluating",
    ) -> dict:
        model.eval()

        targets: list[int] = []
        probabilities: list[list[float]] = []
        slide_ids: list[str] = []

        text_tokens = text_tokens.to(self.device)
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(self.device)

        for batch in tqdm(dataloader, desc=desc, leave=False):
            features = move_feature_dict(
                batch["features"],
                self.device,
            )
            labels = batch["label"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                output = model(
                    multi_scale_features=features,
                    text_tokens=text_tokens,
                    text_attention_mask=text_attention_mask,
                )

            targets.extend(labels.cpu().tolist())
            probabilities.extend(
                output["probabilities"].cpu().tolist()
            )

            batch_slide_ids = batch.get("slide_id")
            if isinstance(batch_slide_ids, str):
                slide_ids.append(batch_slide_ids)
            elif batch_slide_ids is not None:
                slide_ids.extend(list(batch_slide_ids))

        metrics = compute_classification_metrics(
            targets,
            probabilities,
        )
        metrics["num_slides"] = len(targets)
        metrics["slide_ids"] = slide_ids
        metrics["targets"] = targets
        metrics["probabilities"] = probabilities

        return metrics
