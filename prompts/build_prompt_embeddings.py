from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prompts.frozen_text_encoder import FrozenTransformerTextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptions",
        default="data/prompts/gleason_descriptions.json",
    )
    parser.add_argument(
        "--encoder",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument(
        "--output",
        default="data/prompts/prompt_tokens.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    descriptions = json.loads(Path(args.descriptions).read_text(encoding="utf-8"))
    grades = ["Grade1", "Grade2", "Grade3", "Grade4"]
    ordered = [descriptions[grade] for grade in grades]

    encoder = FrozenTransformerTextEncoder(args.encoder)
    encoded = encoder.encode(
        ordered,
        max_length=args.max_length,
        device=args.device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "grade_names": grades,
            "tokens": encoded.token_embeddings,
            "attention_mask": encoded.attention_mask,
            "encoder_name": args.encoder,
        },
        output,
    )
    print(f"Saved prompt tokens to {output}")


if __name__ == "__main__":
    main()
