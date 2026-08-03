from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from build import (
    build_dataloader,
    build_model,
    build_optimization,
    load_prompt_payload,
)
from trainer import Trainer
from trainer.checkpoint import load_checkpoint
from utils import (
    build_logger,
    load_config,
    resolve_device,
    save_config,
    seed_everything,
)
from utils.device import describe_device
from utils.misc import count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ProstateSemanticMIL."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    required = [
        ("project", "seed"),
        ("project", "output_dir"),
        ("project", "experiment_name"),
        ("data", "metadata_csv"),
        ("data", "feature_root"),
        ("data", "prompt_file"),
        ("data", "scales"),
        ("data", "fold"),
        ("model", "visual_input_dim"),
        ("model", "hidden_dim"),
        ("training", "epochs"),
        ("training", "learning_rate"),
        ("training", "weight_decay"),
    ]
    missing = [
        f"{section}.{key}"
        for section, key in required
        if section not in config or key not in config[section]
    ]
    if missing:
        raise KeyError(
            "Missing configuration fields: " + ", ".join(missing)
        )


def apply_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.fold is not None:
        config["data"]["fold"] = args.fold
    if args.experiment_name is not None:
        config["project"]["experiment_name"] = args.experiment_name
    if args.device is not None:
        config["project"]["device"] = args.device
    if args.resume is not None:
        config["training"]["resume"] = args.resume
    return config


def build_output_directory(config: dict[str, Any]) -> Path:
    output_dir = (
        Path(config["project"]["output_dir"])
        / str(config["project"]["experiment_name"])
        / f"fold_{int(config['data']['fold'])}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_run_metadata(
    output_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
) -> None:
    metadata = {
        "device": describe_device(device),
        "trainable_parameters": count_trainable_parameters(model),
        "fold": int(config["data"]["fold"]),
        "scales": list(config["data"]["scales"]),
    }
    with (output_dir / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()

    # 1. Configuration and reproducibility
    config = apply_overrides(load_config(args.config), args)
    validate_config(config)
    seed_everything(
        int(config["project"]["seed"]),
        deterministic=True,
    )
    device = resolve_device(config["project"].get("device"))

    # 2. Output directory and logger
    output_dir = build_output_directory(config)
    save_config(config, output_dir / "resolved_config.yaml")
    logger = build_logger(
        "ProstateSemanticMIL.train",
        log_file=output_dir / "train.log",
    )
    logger.info("Using device: %s", device)
    logger.info("Training fold: %s", config["data"]["fold"])

    # 3. Frozen grade-specific token embeddings [C, L, D_t]
    text_tokens, text_attention_mask = load_prompt_payload(
        config["data"]["prompt_file"],
        device,
    )
    logger.info("Prompt token shape: %s", tuple(text_tokens.shape))

    # 4. Variable-length multi-scale WSI feature bags
    train_loader = build_dataloader(config, split="train")
    val_loader = build_dataloader(config, split="val")
    logger.info("Training slides: %d", len(train_loader.dataset))
    logger.info("Validation slides: %d", len(val_loader.dataset))

    # 5. IRM -> cross-attention -> hierarchical transformer -> cosine classifier
    model = build_model(
        config,
        text_input_dim=int(text_tokens.size(-1)),
    ).to(device)
    logger.info(
        "Trainable parameters: %d",
        count_trainable_parameters(model),
    )
    save_run_metadata(output_dir, config, device, model)

    # 6. AdamW, cosine scheduler, and CE + optional explicit L2
    optimizer, scheduler, loss_fn = build_optimization(
        model, config
    )

    # 7. Training engine
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        epochs=int(config["training"]["epochs"]),
        grad_clip_norm=float(
            config["training"].get("grad_clip_norm", 5.0)
        ),
        use_amp=bool(config["training"].get("amp", True)),
        early_stopping_patience=int(
            config["training"].get(
                "early_stopping_patience", 15
            )
        ),
        monitor=str(
            config["training"].get("monitor", "macro_f1")
        ),
        monitor_mode=str(
            config["training"].get("monitor_mode", "max")
        ),
        config=config,
    )

    # 8. Optional resume
    start_epoch = 1
    resume_path = config["training"].get("resume")
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=trainer.scaler,
            map_location=device,
            strict=True,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from epoch %d", start_epoch - 1)

    # 9. Training
    result = trainer.fit(
        train_loader,
        val_loader,
        text_tokens=text_tokens,
        text_attention_mask=text_attention_mask,
        start_epoch=start_epoch,
    )

    logger.info("Training completed.")
    logger.info("Best metrics: %s", result["best_metrics"])
    logger.info("Best checkpoint: %s", output_dir / "best.pt")


if __name__ == "__main__":
    main()
