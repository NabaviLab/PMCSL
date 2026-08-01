from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class SemanticPrototypeClassifier(nn.Module):
    """Cosine similarity between slide embedding z_v and grade prototypes z_c."""
    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = False) -> None:
        super().__init__()
        log_scale = torch.tensor(1.0 / temperature).log()
        if learnable_temperature:
            self.logit_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("logit_scale", log_scale)

    def forward(self, slide_embedding: torch.Tensor, prototypes: torch.Tensor):
        slide_embedding = F.normalize(slide_embedding, p=2, dim=-1)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        logits = self.logit_scale.exp().clamp(max=100.0) * slide_embedding @ prototypes.t()
        return logits, torch.softmax(logits, dim=-1)
