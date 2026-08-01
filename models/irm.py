from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .initialization import initialize_module


@dataclass
class IRMOutput:
    features: torch.Tensor
    scores: torch.Tensor
    indices: torch.Tensor


class IterativeRefinementModule(nn.Module):
    """Iteratively score patches, prune low-information tokens, and retain Top-M."""
    def __init__(self, dim: int, top_m: int = 256, iterations: int = 3,
                 keep_ratio: float = 0.65, dropout: float = 0.1) -> None:
        super().__init__()
        if top_m < 1 or iterations < 1 or not 0 < keep_ratio <= 1:
            raise ValueError("Invalid IRM hyperparameters.")
        self.top_m = top_m
        self.iterations = iterations
        self.keep_ratio = keep_ratio
        hidden = max(dim // 2, 1)
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.apply(initialize_module)

    def forward(self, x: torch.Tensor) -> IRMOutput:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,N,D], got {tuple(x.shape)}")
        b, n, d = x.shape
        current = x
        indices = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
        for _ in range(self.iterations):
            scores = self.scorer(current).squeeze(-1)
            if current.size(1) <= self.top_m:
                break
            keep = min(current.size(1), max(self.top_m, int(round(current.size(1) * self.keep_ratio))))
            _, pos = torch.topk(scores, keep, dim=1, sorted=False)
            current = torch.gather(current, 1, pos.unsqueeze(-1).expand(-1, -1, d))
            indices = torch.gather(indices, 1, pos)
        scores = self.scorer(current).squeeze(-1)
        k = min(self.top_m, current.size(1))
        top_scores, pos = torch.topk(scores, k, dim=1, sorted=True)
        features = torch.gather(current, 1, pos.unsqueeze(-1).expand(-1, -1, d))
        indices = torch.gather(indices, 1, pos)
        return IRMOutput(features, top_scores, indices)
