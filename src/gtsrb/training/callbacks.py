"""Training callbacks: schedules, early stopping and checkpoint selection.

These were inline ``if`` statements in the original notebook — "save if validation
accuracy improved", "stop never". Extracting them makes each policy independently
testable, which matters because they decide which weights you end up shipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau, StepLR

from gtsrb.config.schema import SchedulerConfig
from gtsrb.runtime import get_logger

logger = get_logger("training.callbacks")

#: Metrics that a schedule or stopping rule is allowed to watch.
MONITORED_METRICS = ("val_loss", "val_accuracy", "val_macro_f1")


def build_scheduler(
    config: SchedulerConfig,
    optimizer: Optimizer,
    *,
    epochs: int,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Build the learning-rate schedule, or ``None`` for a constant rate.

    ``ReduceLROnPlateau`` keeps parity with the original run. ``cosine`` and
    ``step`` are included because the original report itself names them as the
    obvious next thing to try, and having them configurable makes that a one-line
    experiment rather than a rewrite.

    Raises:
        ValueError: if the schedule needs a step size that is not positive.
    """
    if config.name == "none":
        return None
    if config.name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min" if config.monitor == "val_loss" else "max",
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )
    if config.name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=config.min_lr)
    if config.name == "step":
        if config.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {config.step_size}")
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    raise ValueError(f"Unsupported scheduler {config.name!r}")


def current_learning_rate(optimizer: Optimizer) -> float:
    """Read the first parameter group's learning rate."""
    return float(optimizer.param_groups[0]["lr"])


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    The original run always trained the full 30 epochs. The recorded curves show
    that was wasteful: the CNN's best validation epoch was 28 but it had plateaued
    by about epoch 10, and the MLP was still improving at epoch 30 — two opposite
    failures that one patience setting cannot fix, but which a per-run policy at
    least makes visible and cheap.

    Args:
        monitor: metric key to watch.
        mode: ``"max"`` for metrics where higher is better.
        patience: epochs without improvement to tolerate.
        min_delta: improvement smaller than this does not count.
    """

    def __init__(
        self,
        *,
        monitor: str = "val_macro_f1",
        mode: str = "max",
        patience: int = 8,
        min_delta: float = 1e-4,
    ) -> None:
        if monitor not in MONITORED_METRICS:
            raise ValueError(f"Cannot monitor {monitor!r}; expected one of {MONITORED_METRICS}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.best_epoch: int = 0
        self.epochs_without_improvement: int = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, metrics: dict[str, float], epoch: int) -> bool:
        """Fold in one epoch's metrics.

        Returns:
            ``True`` when training should stop.
        """
        value = metrics[self.monitor]
        if self._is_improvement(value):
            self.best = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            logger.info(
                "early stopping triggered",
                extra={
                    "monitor": self.monitor,
                    "best": self.best,
                    "best_epoch": self.best_epoch,
                    "patience": self.patience,
                },
            )
            return True
        return False


@dataclass
class CheckpointDecision:
    """Whether a checkpoint was written, and why."""

    saved: bool
    reason: str
    metric: float | None = None


class CheckpointSelector:
    """Decide which epoch's weights are "best".

    Defaults to ``val_macro_f1`` rather than accuracy. Accuracy is dominated by the
    common classes; the audit showed a model can be at 98.8% accuracy with a class
    at 54% recall. Selecting on macro F1 selects the model that is actually better
    across all 43 classes. Accuracy is still reported, so nothing is hidden.
    """

    def __init__(
        self,
        *,
        monitor: str = "val_macro_f1",
        mode: str = "max",
    ) -> None:
        if monitor not in MONITORED_METRICS:
            raise ValueError(f"Cannot monitor {monitor!r}; expected one of {MONITORED_METRICS}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.best: float | None = None
        self.best_epoch: int = 0

    def should_save(self, metrics: dict[str, float], epoch: int) -> CheckpointDecision:
        value = metrics[self.monitor]
        improved = self.best is None or (
            value > self.best if self.mode == "max" else value < self.best
        )
        if improved:
            self.best = value
            self.best_epoch = epoch
            return CheckpointDecision(saved=True, reason="improved", metric=value)
        return CheckpointDecision(saved=False, reason="no improvement", metric=value)


def build_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> Optimizer:
    """Build the optimiser.

    ``adam`` is the default for parity with the original run. ``adamw`` is offered
    because it decouples weight decay from the adaptive step, which is the correct
    behaviour when weight decay is meant as regularisation rather than as L2 in the
    gradient — the two differ for exactly the adaptive optimisers used here.

    Raises:
        ValueError: if the optimiser name is unknown.
    """
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
    raise ValueError(f"Unsupported optimizer {name!r}; expected adam, adamw or sgd")


def scheduler_metadata(scheduler: LRScheduler | ReduceLROnPlateau | None) -> dict[str, Any]:
    """Describe a scheduler for logging."""
    if scheduler is None:
        return {"name": "none"}
    return {"name": type(scheduler).__name__}


__all__ = [
    "MONITORED_METRICS",
    "CheckpointDecision",
    "CheckpointSelector",
    "EarlyStopping",
    "build_optimizer",
    "build_scheduler",
    "current_learning_rate",
    "scheduler_metadata",
]
