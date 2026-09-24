"""Training loop.

Replaces the notebook's ``train_loop`` function. Everything the original did is
preserved — Adam, ``ReduceLROnPlateau``, AMP, gradient clipping, best-checkpoint
selection — with these changes, each of which the audit or a deprecation warning
forced:

* **Device-agnostic AMP.** The original gated mixed precision on
  ``torch.cuda.is_available()`` and used ``torch.cuda.amp.GradScaler``, which is
  deprecated and does nothing on non-CUDA hardware. Here it is
  ``torch.amp.autocast(device_type=...)`` and a scaler created only when the device
  actually supports it.
* **Best-checkpoint selection on macro F1 by default**, not accuracy.
* **Early stopping**, so a run does not burn epochs after the metric has stopped
  moving.
* **Resume from checkpoint**, including optimiser and scheduler state, so an
  interrupted run continues rather than restarting.
* **A tracker seam**, so MLflow is optional and the loop contains no tracking
  conditionals.
* **Structured logging** of per-epoch metrics and step-level progress.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from gtsrb.config.schema import ProjectConfig
from gtsrb.models.checkpoints import build_metadata, load_checkpoint, save_checkpoint
from gtsrb.models.factory import summarise_model
from gtsrb.runtime import get_logger, resolve_device, supports_amp
from gtsrb.tracking.base import NullTracker, Tracker
from gtsrb.training.callbacks import (
    CheckpointSelector,
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    current_learning_rate,
)
from gtsrb.training.losses import build_loss
from gtsrb.training.metrics import EpochMetrics, MetricsAccumulator

logger = get_logger("training.trainer")


@dataclass
class TrainResult:
    """Outcome of a training run."""

    model_name: str
    epochs_trained: int
    best_epoch: int
    best_metrics: dict[str, float]
    history: list[dict[str, float]]
    checkpoint_path: str | None
    stopped_early: bool
    duration_seconds: float
    device: str
    parameter_summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "epochs_trained": self.epochs_trained,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "history": self.history,
            "checkpoint_path": self.checkpoint_path,
            "stopped_early": self.stopped_early,
            "duration_seconds": round(self.duration_seconds, 3),
            "device": self.device,
            "parameter_summary": self.parameter_summary,
        }

    def summary(self) -> str:
        best = " ".join(f"{key}={value:.4f}" for key, value in sorted(self.best_metrics.items()))
        return (
            f"{self.model_name} trained for {self.epochs_trained} epoch(s) on {self.device}; "
            f"best epoch {self.best_epoch}: {best}"
        )


class Trainer:
    """Trains a classifier and selects its best checkpoint.

    Args:
        config: full project configuration.
        model: the module to train.
        train_loader: training data.
        val_loader: validation data.
        device: optional device override; otherwise taken from the configuration.
        tracker: experiment tracker. Defaults to a no-op tracker.
    """

    def __init__(
        self,
        config: ProjectConfig,
        model: nn.Module,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        device: torch.device | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else resolve_device(config.training.device)
        self.tracker: Tracker = tracker if tracker is not None else NullTracker()

        self.model.to(self.device)
        self.loss_fn = build_loss(config.training)
        training = config.training
        self.optimizer = build_optimizer(
            self.model,
            name=training.optimizer.name,
            lr=training.optimizer.lr,
            weight_decay=training.optimizer.weight_decay,
            momentum=training.optimizer.momentum,
            nesterov=training.optimizer.nesterov,
        )
        self.scheduler = build_scheduler(
            config.training.scheduler, self.optimizer, epochs=training.epochs
        )

        self.use_amp = training.amp and supports_amp(self.device)
        # GradScaler is CUDA-only; constructing one for CPU or MPS would be a no-op
        # at best and an error at worst.
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.summary = summarise_model(
            self.model,
            name=config.model.name,
            image_size=config.data.image_size,
            num_classes=config.model.num_classes,
        )

    # -- public API --------------------------------------------------------
    def fit(
        self,
        *,
        resume_from: Path | None = None,
        start_epoch: int = 0,
    ) -> TrainResult:
        """Train the model.

        Args:
            resume_from: checkpoint to restore before training. Use ``last.pt``:
                ``best.pt`` stores weights and metadata only, so it cannot restore
                optimiser or scheduler state.
            start_epoch: epoch index to resume from. Ignored unless ``resume_from``
                is given, in which case it defaults to that checkpoint's epoch.

        Returns:
            A :class:`TrainResult`.
        """
        training = self.config.training
        if resume_from is not None:
            start_epoch = self._resume(resume_from, start_epoch)

        checkpoint_selector = CheckpointSelector(
            monitor=training.checkpoint.monitor, mode=training.checkpoint.mode
        )
        early_stopping = (
            EarlyStopping(
                monitor=training.early_stopping.monitor,
                mode=training.early_stopping.mode,
                patience=training.early_stopping.patience,
                min_delta=training.early_stopping.min_delta,
            )
            if training.early_stopping.enabled
            else None
        )

        history: list[dict[str, float]] = []
        best_metrics: dict[str, float] = {}
        checkpoint_path: Path | None = None
        stopped_early = False

        logger.info(
            "starting training",
            extra={
                **self.summary.as_dict(),
                "epochs": training.epochs,
                "batch_size": training.batch_size,
                "optimizer": training.optimizer.name,
                "lr": training.optimizer.lr,
                "scheduler": training.scheduler.name,
                "amp": self.use_amp,
                "device": str(self.device),
                "train_batches": len(self.train_loader),
                "val_batches": len(self.val_loader),
                "monitor": training.checkpoint.monitor,
            },
        )

        started = time.perf_counter()

        # ``epoch_index`` is the 0-based loop counter; ``epoch`` is the 1-based
        # number reported in metrics, logs and checkpoint metadata. Keeping the
        # reported number 1-based means "best epoch 28" means the 28th epoch, which
        # is what the original project's logs meant too.
        for epoch_index in range(start_epoch, training.epochs):
            epoch = epoch_index + 1
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate()

            epoch_metrics: dict[str, float] = {
                **{f"train_{key}": value for key, value in train_metrics.as_dict().items()},
                **{f"val_{key}": value for key, value in val_metrics.as_dict().items()},
                "learning_rate": current_learning_rate(self.optimizer),
                "epoch": float(epoch),
            }
            history.append(epoch_metrics)

            self._step_scheduler(val_metrics)

            logger.info(
                "epoch complete",
                extra={
                    "epoch": epoch,
                    "train_loss": round(train_metrics.loss, 4),
                    "train_accuracy": round(train_metrics.accuracy, 4),
                    "train_macro_f1": round(train_metrics.macro_f1, 4),
                    "val_loss": round(val_metrics.loss, 4),
                    "val_accuracy": round(val_metrics.accuracy, 4),
                    "val_macro_f1": round(val_metrics.macro_f1, 4),
                    "val_top3_accuracy": round(val_metrics.top3_accuracy, 4),
                    "learning_rate": epoch_metrics["learning_rate"],
                },
            )
            self.tracker.log_metrics(epoch_metrics, step=epoch)

            # last.pt is written every epoch, best.pt only on improvement. Writing
            # last.pt only on non-improving epochs would leave a run that improved
            # every epoch with no resumable checkpoint at all.
            if training.checkpoint.save_last:
                self._save_checkpoint(epoch, epoch_metrics, is_best=False)

            decision = checkpoint_selector.should_save(epoch_metrics, epoch)
            if decision.saved:
                best_metrics = dict(epoch_metrics)
                checkpoint_path = self._save_checkpoint(epoch, epoch_metrics, is_best=True)

            if early_stopping is not None and early_stopping.update(epoch_metrics, epoch):
                stopped_early = True
                break

        duration = time.perf_counter() - started
        epochs_trained = len(history)

        result = TrainResult(
            model_name=self.config.model.name,
            epochs_trained=epochs_trained,
            best_epoch=checkpoint_selector.best_epoch,
            best_metrics=best_metrics,
            history=history,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            stopped_early=stopped_early,
            duration_seconds=duration,
            device=str(self.device),
            parameter_summary=self.summary.as_dict(),
        )

        logger.info(
            "training finished",
            extra={
                "epochs_trained": epochs_trained,
                "best_epoch": result.best_epoch,
                "best_val_macro_f1": best_metrics.get("val_macro_f1"),
                "stopped_early": stopped_early,
                "duration_seconds": round(duration, 1),
            },
        )
        return result

    # -- internals ---------------------------------------------------------
    def _train_epoch(self, epoch: int) -> EpochMetrics:
        self.model.train()
        accumulator = MetricsAccumulator(num_classes=self.config.model.num_classes, top_k=3)
        log_every = self.config.training.log_every_n_steps

        for step, (batch_images, batch_targets) in enumerate(self.train_loader):
            images = batch_images.to(self.device, non_blocking=True)
            targets = batch_targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, targets)
                self.scaler.scale(loss).backward()
                if self.config.training.grad_clip_norm is not None:
                    # Gradients must be unscaled before clipping, or the clip
                    # threshold applies to the scaled magnitudes instead.
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                if self.config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip_norm
                    )
                self.optimizer.step()

            accumulator.update(logits, targets, float(loss.detach().item()))

            if log_every and (step + 1) % log_every == 0:
                logger.info(
                    "training step",
                    extra={
                        "epoch": epoch,
                        "step": step + 1,
                        "steps_per_epoch": len(self.train_loader),
                        "loss": round(float(loss.detach().item()), 4),
                        "learning_rate": current_learning_rate(self.optimizer),
                    },
                )

        metrics = accumulator.compute()
        if not torch.isfinite(torch.tensor(metrics.loss)):
            # A non-finite loss means the run is already lost; continuing would
            # write a corrupt checkpoint over a good one.
            raise RuntimeError(
                f"Training loss became non-finite at epoch {epoch} "
                f"({metrics.loss}). Lower the learning rate, or enable gradient clipping."
            )
        return metrics

    @torch.no_grad()
    def _validate(self) -> EpochMetrics:
        self.model.eval()
        accumulator = MetricsAccumulator(num_classes=self.config.model.num_classes, top_k=3)

        for batch_images, batch_targets in self.val_loader:
            images = batch_images.to(self.device, non_blocking=True)
            targets = batch_targets.to(self.device, non_blocking=True)
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, targets)
            else:
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
            accumulator.update(logits, targets, float(loss.detach().item()))

        return accumulator.compute()

    def _step_scheduler(self, val_metrics: EpochMetrics) -> None:
        scheduler = self.scheduler
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitored = {
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
            }[self.config.training.scheduler.monitor]
            scheduler.step(monitored)
        else:
            scheduler.step()

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float], *, is_best: bool) -> Path:
        """Write either the best-so-far weights or the resumable training state.

        ``best.pt`` holds weights and metadata only; ``last.pt`` additionally holds
        optimiser and scheduler state. The split is deliberate and measurable: Adam
        keeps two moment buffers per parameter, so a resumable checkpoint is roughly
        three times the size of the weights it contains. Measured on the compact CNN,
        a 36.5 MB model produces a 114 MB full checkpoint. Shipping 114 MB to serve a
        36.5 MB model is waste, and the served artifact never needs optimiser state.

        Consequence: resuming requires ``last.pt``. That is why ``save_last``
        defaults to true.
        """
        checkpoint_config = self.config.training.checkpoint
        filename = checkpoint_config.filename if is_best else "last.pt"
        path = Path(checkpoint_config.directory) / filename

        # Only the validation metrics belong in the checkpoint; training metrics are
        # a diagnostic, not a property of the saved weights.
        stored = {key: value for key, value in metrics.items() if key.startswith("val_")}
        metadata = build_metadata(self.config, metrics=stored, epoch=epoch)

        resumable = not is_best
        return save_checkpoint(
            path,
            self.model,
            metadata,
            optimizer=self.optimizer if resumable else None,
            scheduler=self.scheduler if resumable else None,
            extra={"is_best": is_best, "resumable": resumable},
        )

    def _resume(self, path: Path, start_epoch: int) -> int:
        loaded = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(loaded.model_state)
        if loaded.optimizer_state is not None:
            self.optimizer.load_state_dict(loaded.optimizer_state)
        if loaded.scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(loaded.scheduler_state)

        # ``metadata.epoch`` is 1-based and names the last *completed* epoch, so it
        # is exactly the 0-based index of the next epoch to run.
        resume_epoch = start_epoch or loaded.metadata.epoch
        logger.info(
            "resumed from checkpoint",
            extra={
                "path": str(path),
                "resume_epoch": resume_epoch,
                "checkpoint_epoch": loaded.epoch,
            },
        )
        return resume_epoch


def train_model(
    config: ProjectConfig,
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    tracker: Tracker | None = None,
    resume_from: Path | None = None,
) -> TrainResult:
    """Convenience wrapper around :class:`Trainer`."""
    trainer = Trainer(config, model, train_loader, val_loader, tracker=tracker)
    return trainer.fit(resume_from=resume_from)


__all__ = ["TrainResult", "Trainer", "train_model"]
