from __future__ import annotations
import torch
from torch import nn
from .initialization import initialize_module


class VisualProjection(nn.Module):
    """Project patch embeddings P^(s): [B,N,Dv] -> E^(s): [B,N,Dh]."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.apply(initialize_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,N,D], got {tuple(x.shape)}")
        return self.net(x)


class TextProjection(nn.Module):
    """Project T_c: [C,L,Dt] -> G_c: [C,L,Dh]."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.apply(initialize_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [C,L,D], got {tuple(x.shape)}")
        return self.net(x)
