"""Classification metrics computed during training and evaluation.

Accuracy alone is not enough here, and this is not a theoretical point: the
recorded CNN reached 98.84% accuracy while getting class 27 (Pedestrians) right
only 54% of the time. A single number hid a class that the model had effectively
not learned.

So every epoch reports accuracy **and** macro-averaged precision, recall and F1,
plus top-3 accuracy. Macro averaging weights all 43 classes equally, which is what
makes the rare classes visible; top-3 accuracy separates "confidently wrong" from
"narrowly missed", which matters when the target is a driver warning.

Predictions are collected for the whole split rather than accumulated per batch.
The test split is 12,630 integers — negligible memory — and computing macro F1
from a running average would be simply wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics for one pass over a dataset."""

    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    top3_accuracy: float
    num_samples: int

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    def summary(self) -> str:
        return (
            f"loss={self.loss:.4f} acc={self.accuracy:.4f} "
            f"macro_f1={self.macro_f1:.4f} top3={self.top3_accuracy:.4f}"
        )


class MetricsAccumulator:
    """Collect per-batch logits and targets, then compute :class:`EpochMetrics`.

    Args:
        num_classes: label space size; used to bound the confusion counts and to
            decide which top-k values are meaningful.
        top_k: ``k`` for top-k accuracy. Capped at ``num_classes``.
    """

    def __init__(self, *, num_classes: int = 43, top_k: int = 3) -> None:
        self.num_classes = num_classes
        self.top_k = min(top_k, num_classes)
        self._loss_sum = 0.0
        self._num_samples = 0
        self._predictions: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._top_k_correct = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        """Fold one batch in.

        Args:
            logits: ``(N, num_classes)`` model output, before softmax.
            targets: ``(N,)`` ground-truth class ids.
            loss: the batch's *mean* loss, as returned by the criterion.

        Raises:
            ValueError: if the batch shapes disagree, which would otherwise
                silently mis-attribute predictions to labels.
        """
        if logits.dim() != 2:
            raise ValueError(f"Expected 2D logits (N, C), got shape {tuple(logits.shape)}")
        if targets.dim() != 1:
            raise ValueError(f"Expected 1D targets (N,), got shape {tuple(targets.shape)}")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {logits.shape[0]} logits against {targets.shape[0]} targets"
            )

        batch_size = int(targets.shape[0])
        self._loss_sum += float(loss) * batch_size
        self._num_samples += batch_size

        detached = logits.detach()
        self._predictions.append(detached.argmax(dim=1).cpu().numpy())
        self._targets.append(targets.detach().cpu().numpy())

        if self.top_k > 1 and detached.shape[1] >= self.top_k:
            top = detached.topk(self.top_k, dim=1).indices
            self._top_k_correct += int(
                (top == targets.detach().unsqueeze(1)).any(dim=1).sum().item()
            )

    def compute(self) -> EpochMetrics:
        """Compute the epoch summary.

        Raises:
            RuntimeError: if nothing was accumulated. Returning zeros here would
                make an empty validation pass look like a total failure.
        """
        if self._num_samples == 0:
            raise RuntimeError("No batches were accumulated; cannot compute metrics")

        predictions = np.concatenate(self._predictions)
        targets = np.concatenate(self._targets)

        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            labels=list(range(self.num_classes)),
            average="macro",
            zero_division=0,
        )

        return EpochMetrics(
            loss=self._loss_sum / self._num_samples,
            accuracy=float((predictions == targets).mean()),
            macro_precision=float(precision),
            macro_recall=float(recall),
            macro_f1=float(f1),
            top3_accuracy=(self._top_k_correct / self._num_samples if self.top_k > 1 else 0.0),
            num_samples=self._num_samples,
        )

    @property
    def num_samples(self) -> int:
        return self._num_samples


def confusion_counts(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Confusion matrix as integer counts.

    Row-normalised percentages are easier to read for a 43-class problem, but
    counts are what the numbers are, so counts are what gets returned and
    normalisation is left to the plotting layer.

    Raises:
        ValueError: if any label falls outside the label space.
    """
    for name, array in (("predictions", predictions), ("targets", targets)):
        if array.size and (array.min() < 0 or array.max() >= num_classes):
            raise ValueError(
                f"{name} contains labels outside [0, {num_classes}): "
                f"min={array.min()}, max={array.max()}"
            )

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> list[dict[str, Any]]:
    """Per-class precision, recall, F1 and support.

    This is the table that makes the error analysis possible without re-running
    inference, and the one that surfaced class 27 in the audit.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    return [
        {
            "class_id": class_id,
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in range(num_classes)
    ]


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    num_bins: int = 15,
) -> float:
    """Expected calibration error of the predicted-class confidences.

    For a safety-facing classifier it is not enough to be right; the confidence has
    to mean something. ECE is the average gap between stated confidence and observed
    accuracy across confidence bins: 0 is perfectly calibrated, and a positive value
    means the model is over-confident.

    Args:
        confidences: predicted probability of the predicted class, in ``[0, 1]``.
        correct: 1.0 where the prediction was right, else 0.0.
        num_bins: number of equally spaced confidence bins.

    Returns:
        The weighted mean absolute gap. ``0.0`` for empty input.
    """
    if confidences.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    # Right edge inclusive so a confidence of exactly 1.0 lands in the last bin.
    bin_ids = np.clip(np.digitize(confidences, edges[1:-1], right=False), 0, num_bins - 1)

    total = confidences.size
    error = 0.0
    for bin_id in range(num_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(confidences[mask].mean()) - float(correct[mask].mean()))
        error += (count / total) * gap
    return error


__all__ = [
    "EpochMetrics",
    "MetricsAccumulator",
    "confusion_counts",
    "expected_calibration_error",
    "per_class_metrics",
]
