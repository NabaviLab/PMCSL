from __future__ import annotations
import torch
from torch import nn
from .initialization import initialize_module


class ScaleSpecificCrossAttention(nn.Module):
    """Visual queries with concatenated grade-text keys/values."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.apply(initialize_module)

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor,
                text_padding_mask: torch.Tensor | None = None):
        attended, weights = self.attn(
            query=visual_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(visual_tokens + attended)
        x = self.norm2(x + self.ffn(x))
        return x, weights
