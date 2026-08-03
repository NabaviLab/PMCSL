from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from build import (
    build_model,
    load_prompt_payload,
)
from trainer.checkpoint import load_checkpoint
from trainer.inference import predict_slide
from utils import (
    build_logger,
    load_config,
    resolve_device,
    write_json,
)
from utils.attention import (
    aggregate_attention_heads,
    visual_token_importance,
)
from utils.constants import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ProstateSemanticMIL inference for one WSI feature bag."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Resolved YAML configuration used during training.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to one slide-level multi-scale .pt feature file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/inference",
        help="Directory for JSON and attention tensors.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override.",
    )
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save raw and aggregated attention tensors.",
    )
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "project": ("device",),
        "data": ("prompt_file", "scales"),
        "model": ("visual_input_dim", "hidden_dim"),
    }

    missing: list[str] = []
    for section, keys in required.items():
        if section not in config:
            missing.extend(f"{section}.{key}" for key in keys)
            continue
        for key in keys:
            if key not in config[section]:
                missing.append(f"{section}.{key}")

    if missing:
        raise KeyError(
            "Missing configuration fields: " + ", ".join(sorted(missing))
        )


def load_multi_scale_features(
    path: str | Path,
    scales: list[str] | tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """
    Load one slide feature bag.

    Supported file layouts:
        {
            "5x": Tensor[N5,D],
            "10x": Tensor[N10,D],
            "20x": Tensor[N20,D],
        }

    or:
        {
            "features": {
                "5x": ...,
                "10x": ...,
                "20x": ...,
            }
        }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    payload = payload.get("features", payload)

    features: dict[str, torch.Tensor] = {}
    for scale in scales:
        if scale not in payload:
            raise KeyError(
                f"Feature file does not contain required scale '{scale}'."
            )

        tensor = payload[scale].float()
        if tensor.ndim != 2:
            raise ValueError(
                f"{scale} features must have shape [N,D], "
                f"received {tuple(tensor.shape)}."
            )

        # Add the mini-batch dimension expected by the model.
        features[scale] = tensor.unsqueeze(0)

    return features


def serialize_probabilities(
    probabilities: torch.Tensor,
) -> dict[str, float]:
    row = probabilities[0].tolist()
    return {
        class_name: float(row[index])
        for index, class_name in enumerate(CLASS_NAMES)
    }


def save_attention_outputs(
    result: dict,
    output_dir: Path,
) -> dict[str, str]:
    """
    Save raw cross-attention and derived visual-token importance scores.

    Cross-attention shape:
        [B,H,N_visual,N_text]
    """
    attention_dir = output_dir / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}

    for scale, attention in result["cross_attention_maps"].items():
        raw_path = attention_dir / f"{scale}_cross_attention.pt"
        torch.save(attention, raw_path)

        aggregated = aggregate_attention_heads(
            attention,
            method="mean",
        )
        importance = visual_token_importance(
            attention,
            head_reduction="mean",
            text_reduction="mean",
        )

        summary_path = attention_dir / f"{scale}_attention_summary.pt"
        torch.save(
            {
                "aggregated_attention": aggregated,
                "visual_token_importance": importance,
                "selected_patch_indices": result["irm_indices"][scale],
            },
            summary_path,
        )

        saved_paths[f"{scale}_raw"] = str(raw_path)
        saved_paths[f"{scale}_summary"] = str(summary_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    if args.device is not None:
        config["project"]["device"] = args.device

    device = resolve_device(config["project"].get("device"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(
        "ProstateSemanticMIL.infer",
        log_file=output_dir / "infer.log",
    )
    logger.info("Feature file: %s", args.features)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Device: %s", device)

    text_tokens, text_attention_mask = load_prompt_payload(
        config["data"]["prompt_file"],
        device,
    )

    model = build_model(
        config,
        text_input_dim=int(text_tokens.size(-1)),
    ).to(device)

    checkpoint = load_checkpoint(
        args.checkpoint,
        model=model,
        map_location=device,
        strict=True,
    )
    logger.info(
        "Loaded checkpoint epoch: %s",
        checkpoint.get("epoch", "unknown"),
    )

    multi_scale_features = load_multi_scale_features(
        args.features,
        config["data"]["scales"],
    )

    result = predict_slide(
        model,
        multi_scale_features,
        text_tokens=text_tokens,
        text_attention_mask=text_attention_mask,
        device=device,
    )

    predicted_index = int(result["predicted_class"][0])
    probabilities = serialize_probabilities(
        result["probabilities"]
    )

    output_payload: dict[str, Any] = {
        "feature_file": str(Path(args.features).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "predicted_index": predicted_index,
        "predicted_grade": CLASS_NAMES[predicted_index],
        "probabilities": probabilities,
        "selected_patch_counts": {
            scale: int(indices.shape[1])
            for scale, indices in result["irm_indices"].items()
        },
    }

    if args.save_attention:
        output_payload["attention_files"] = save_attention_outputs(
            result,
            output_dir,
        )

    write_json(
        output_payload,
        output_dir / "prediction.json",
    )

    logger.info(
        "Predicted grade: %s",
        output_payload["predicted_grade"],
    )
    logger.info(
        "Probabilities: %s",
        probabilities,
    )
    logger.info("Saved inference output to %s", output_dir)


if __name__ == "__main__":
    main()
