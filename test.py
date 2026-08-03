from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from build import (
    build_dataloader,
    build_model,
    load_prompt_payload,
)
from trainer import Evaluator
from trainer.checkpoint import load_checkpoint
from utils import (
    build_logger,
    load_config,
    resolve_device,
    write_json,
)
from utils.constants import CLASS_NAMES
from utils.visualization import (
    plot_confusion_matrix,
    plot_multiclass_roc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ProstateSemanticMIL checkpoint."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Resolved YAML configuration used during training.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best.pt or another trained checkpoint.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory for metrics, predictions, and figures.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g., cuda:0 or cpu.",
    )
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "project": ("device",),
        "data": (
            "metadata_csv",
            "feature_root",
            "prompt_file",
            "scales",
            "fold",
        ),
        "model": (
            "visual_input_dim",
            "hidden_dim",
        ),
        "training": ("amp",),
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


def build_prediction_frame(result: dict) -> pd.DataFrame:
    probabilities = result["probabilities"]
    predictions = [
        max(range(len(row)), key=row.__getitem__)
        for row in probabilities
    ]

    payload: dict[str, list] = {
        "slide_id": result["slide_ids"],
        "target": result["targets"],
        "prediction": predictions,
        "target_name": [
            CLASS_NAMES[index] for index in result["targets"]
        ],
        "prediction_name": [
            CLASS_NAMES[index] for index in predictions
        ],
    }

    for class_index, class_name in enumerate(CLASS_NAMES):
        payload[f"prob_{class_name}"] = [
            row[class_index] for row in probabilities
        ]

    return pd.DataFrame(payload)


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
        "ProstateSemanticMIL.test",
        log_file=output_dir / "test.log",
    )
    logger.info("Evaluating split: %s", args.split)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Device: %s", device)

    # Load frozen grade-specific token embeddings [C,L,D_t].
    text_tokens, text_attention_mask = load_prompt_payload(
        config["data"]["prompt_file"],
        device,
    )

    dataloader = build_dataloader(config, split=args.split)

    # Reconstruct the same architecture used during training.
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
        "Loaded checkpoint from epoch %s",
        checkpoint.get("epoch", "unknown"),
    )

    evaluator = Evaluator(
        device=device,
        use_amp=bool(config["training"].get("amp", True)),
    )
    result = evaluator.evaluate(
        model,
        dataloader,
        text_tokens=text_tokens,
        text_attention_mask=text_attention_mask,
        desc=f"Evaluating {args.split}",
    )

    prediction_frame = build_prediction_frame(result)
    prediction_frame.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    summary = {
        key: value
        for key, value in result.items()
        if key not in {
            "slide_ids",
            "targets",
            "probabilities",
        }
    }
    write_json(
        summary,
        output_dir / "metrics.json",
    )

    plot_confusion_matrix(
        result["confusion_matrix"],
        CLASS_NAMES,
        output_dir / "confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        result["confusion_matrix"],
        CLASS_NAMES,
        output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_multiclass_roc(
        result["targets"],
        result["probabilities"],
        CLASS_NAMES,
        output_dir / "roc_curves.png",
    )

    logger.info(
        "ACC=%.4f | F1=%.4f | AUC=%.4f | Kappa=%.4f",
        float(summary["accuracy"]),
        float(summary["macro_f1"]),
        float(summary["macro_auc_ovr"]),
        float(summary["cohen_kappa"]),
    )
    logger.info("Saved evaluation outputs to %s", output_dir)


if __name__ == "__main__":
    main()
