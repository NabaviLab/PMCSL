from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class EncodedPrompts:
    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor


class FrozenTransformerTextEncoder(nn.Module):
    """Frozen text encoder that returns token-level hidden states [C,L,D_t]."""

    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def encode(self, descriptions, max_length=128, device="cuda"):
        self.model.to(device)
        inputs = self.tokenizer(
            list(descriptions),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = self.model(**inputs)
        return EncodedPrompts(
            token_embeddings=output.last_hidden_state.cpu(),
            attention_mask=inputs["attention_mask"].cpu(),
        )
