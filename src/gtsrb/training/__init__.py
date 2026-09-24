"""Training: loop, losses, metrics and callbacks.

Entry point is :func:`gtsrb.training.trainer.train_model`, or the CLI at
``python -m gtsrb.cli.train``.
"""

from __future__ import annotations

from gtsrb.training.callbacks import (
    MONITORED_METRICS,
    CheckpointDecision,
    CheckpointSelector,
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    current_learning_rate,
)
from gtsrb.training.losses import build_loss, softmax_probabilities
from gtsrb.training.metrics import (
    EpochMetrics,
    MetricsAccumulator,
    confusion_counts,
    expected_calibration_error,
    per_class_metrics,
)
from gtsrb.training.trainer import Trainer, TrainResult, train_model

__all__ = [
    "MONITORED_METRICS",
    "CheckpointDecision",
    "CheckpointSelector",
    "EarlyStopping",
    "EpochMetrics",
    "MetricsAccumulator",
    "TrainResult",
    "Trainer",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "confusion_counts",
    "current_learning_rate",
    "expected_calibration_error",
    "per_class_metrics",
    "softmax_probabilities",
    "train_model",
]
