from __future__ import annotations
import torch
from torch import nn
from .initialization import initialize_module


class HierarchicalMultiScaleTransformer(nn.Module):
    """Fuse H^5, H^10, H^20 with a learnable CLS token."""
    def __init__(self, dim: int, num_heads: int, depth: int = 2,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 num_scales: int = 3) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.scale_embeddings = nn.Parameter(torch.zeros(1, num_scales, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.apply(initialize_module)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.scale_embeddings, std=0.02)

    def forward(self, scale_tokens: list[torch.Tensor]):
        if len(scale_tokens) != self.num_scales:
            raise ValueError("Scale count mismatch.")
        b = scale_tokens[0].size(0)
        tokens = [x + self.scale_embeddings[:, i:i+1] for i, x in enumerate(scale_tokens)]
        h = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(b, -1, -1)
        h0 = torch.cat([cls, h], dim=1)
        h_l = self.norm(self.encoder(h0))
        z_v = h_l[:, 0, :]
        return z_v, h_l
