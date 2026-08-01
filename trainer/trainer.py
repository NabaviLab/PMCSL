from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from losses import TotalLoss
from .checkpoint import save_checkpoint
from .early_stopping import EarlyStopping
from .evaluator import Evaluator, move_feature_dict


class Trainer:
    """
    Complete training loop for the proposed multimodal grading framework.

    Features:
        - mixed precision
        - gradient clipping
        - validation
        - cosine scheduler support
        - checkpointing
        - early stopping
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: TotalLoss,
        device: torch.device,
        output_dir: str | Path,
        scheduler: Any | None = None,
        epochs: int = 100,
        grad_clip_norm: float = 5.0,
        use_amp: bool = True,
        early_stopping_patience: int = 15,
        monitor: str = "macro_f1",
        monitor_mode: str = "max",
        config: dict | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler = scheduler
        self.epochs = int(epochs)
        self.grad_clip_norm = float(grad_clip_norm)
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.monitor = monitor
        self.config = config

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode=monitor_mode,
        )

        self.evaluator = Evaluator(
            device=device,
            use_amp=use_amp,
        )

    def _train_one_epoch(
        self,
        dataloader,
        *,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None,
        epoch: int,
    ) -> dict:
        self.model.train()

        text_tokens = text_tokens.to(self.device)
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(self.device)

        running_total = 0.0
        running_ce = 0.0
        running_reg = 0.0
        num_batches = 0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch:03d}",
            leave=False,
        )

        for batch in progress:
            features = move_feature_dict(
                batch["features"],
                self.device,
            )
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                output = self.model(
                    multi_scale_features=features,
                    text_tokens=text_tokens,
                    text_attention_mask=text_attention_mask,
                )
                loss_output = self.loss_fn(
                    output["logits"],
                    labels,
                    self.model,
                )

            self.scaler.scale(loss_output.total).backward()
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_total += float(loss_output.total.detach())
            running_ce += float(loss_output.classification)
            running_reg += float(loss_output.regularization)
            num_batches += 1

            progress.set_postfix(
                loss=f"{running_total / num_batches:.4f}"
            )

        denominator = max(num_batches, 1)

        return {
            "train_loss": running_total / denominator,
            "train_ce": running_ce / denominator,
            "train_regularization": running_reg / denominator,
        }

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
        start_epoch: int = 1,
    ) -> dict:
        best_metrics: dict = {}
        history: list[dict] = []

        for epoch in range(start_epoch, self.epochs + 1):
            train_metrics = self._train_one_epoch(
                train_loader,
                text_tokens=text_tokens,
                text_attention_mask=text_attention_mask,
                epoch=epoch,
            )

            val_metrics = self.evaluator.evaluate(
                self.model,
                val_loader,
                text_tokens=text_tokens,
                text_attention_mask=text_attention_mask,
                desc="Validation",
            )

            epoch_metrics = {
                "epoch": epoch,
                **train_metrics,
                **{
                    key: value
                    for key, value in val_metrics.items()
                    if key not in {
                        "slide_ids",
                        "targets",
                        "probabilities",
                    }
                },
            }
            history.append(epoch_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            monitor_value = float(val_metrics[self.monitor])
            improved = (
                self.early_stopping.best_value is None
                or self.early_stopping._is_improvement(monitor_value)
            )

            save_checkpoint(
                self.output_dir / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                metrics=epoch_metrics,
                config=self.config,
            )

            if improved:
                best_metrics = epoch_metrics
                save_checkpoint(
                    self.output_dir / "best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    config=self.config,
                )

            should_stop = self.early_stopping.step(
                monitor_value
            )

            print(
                f"Epoch {epoch:03d} | "
                f"loss={train_metrics['train_loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['macro_f1']:.4f} | "
                f"val_auc={val_metrics['macro_auc_ovr']:.4f}"
            )

            if should_stop:
                print("Early stopping triggered.")
                break

        return {
            "best_metrics": best_metrics,
            "history": history,
        }
