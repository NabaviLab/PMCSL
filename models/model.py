from __future__ import annotations
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from .projection import VisualProjection, TextProjection
from .irm import IterativeRefinementModule
from .cross_attention import ScaleSpecificCrossAttention
from .hierarchical_transformer import HierarchicalMultiScaleTransformer
from .semantic_classifier import SemanticPrototypeClassifier


class ProstateSemanticMIL(nn.Module):
    """Complete pathology-aware multi-scale cross-modal grading model."""
    def __init__(self, *, visual_input_dim: int, text_input_dim: int,
                 hidden_dim: int = 128, scales=("5x", "10x", "20x"),
                 top_m: int = 256, irm_iterations: int = 3,
                 irm_keep_ratio: float = 0.65, num_heads: int = 4,
                 cross_attention_dropout: float = 0.1,
                 fusion_depth: int = 2, fusion_dropout: float = 0.1,
                 temperature: float = 0.07) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.visual_projection = VisualProjection(visual_input_dim, hidden_dim)
        self.text_projection = TextProjection(text_input_dim, hidden_dim)
        self.irm = nn.ModuleDict({
            s: IterativeRefinementModule(hidden_dim, top_m, irm_iterations, irm_keep_ratio)
            for s in self.scales
        })
        self.cross_attention = nn.ModuleDict({
            s: ScaleSpecificCrossAttention(hidden_dim, num_heads, cross_attention_dropout)
            for s in self.scales
        })
        self.fusion = HierarchicalMultiScaleTransformer(
            hidden_dim, num_heads, fusion_depth, dropout=fusion_dropout,
            num_scales=len(self.scales)
        )
        self.classifier = SemanticPrototypeClassifier(temperature)

    def _prepare_text(self, text_tokens: torch.Tensor,
                      text_attention_mask: torch.Tensor | None,
                      batch_size: int):
        g_c = self.text_projection(text_tokens)  # [C,L,Dh]
        if text_attention_mask is None:
            z_c = g_c.mean(dim=1)
            padding_mask = None
        else:
            mask = text_attention_mask.to(g_c.dtype).unsqueeze(-1)
            z_c = (g_c * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            padding_mask = (~text_attention_mask.bool()).flatten().unsqueeze(0).expand(batch_size, -1)
        z_c = F.normalize(z_c, p=2, dim=-1)
        g = g_c.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
        return g, z_c, padding_mask

    def forward(self, multi_scale_features: Dict[str, torch.Tensor],
                text_tokens: torch.Tensor,
                text_attention_mask: torch.Tensor | None = None):
        b = multi_scale_features[self.scales[0]].size(0)
        g, z_c, text_padding_mask = self._prepare_text(text_tokens, text_attention_mask, b)
        semantic_tokens = []
        irm_scores, irm_indices, attn_maps = {}, {}, {}
        for scale in self.scales:
            x = self.visual_projection(multi_scale_features[scale])
            refined = self.irm[scale](x)
            h_s, attn = self.cross_attention[scale](refined.features, g, text_padding_mask)
            semantic_tokens.append(h_s)
            irm_scores[scale] = refined.scores
            irm_indices[scale] = refined.indices
            attn_maps[scale] = attn
        z_v, encoded = self.fusion(semantic_tokens)
        logits, probabilities = self.classifier(z_v, z_c)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "slide_embedding": z_v,
            "grade_prototypes": z_c,
            "encoded_tokens": encoded,
            "irm_scores": irm_scores,
            "irm_indices": irm_indices,
            "cross_attention_maps": attn_maps,
        }
