"""Classification metrics for evaluation.

Builds on :mod:`gtsrb.training.metrics` rather than reimplementing it, and adds
what evaluation needs beyond a training epoch:

* **category-level metrics** — precision/recall/F1 grouped by sign family
  (speed limit, warning, mandatory, ...). A per-class table for 43 classes is
  accurate but hard to reason about; the family view answers "does the model
  confuse signs *within* a family or *across* families?", which is a different and
  more serious question.
* **confidence statistics** — mean confidence when right, mean confidence when
  wrong, and the expected calibration error. For a driver-warning system,
  confidently wrong is worse than uncertain.
* **top-k accuracy** — how often the right class is in the top 5, which separates
  "did not learn it" from "narrowly missed it".

A single accuracy number is not enough and this is not hypothetical: the recorded
run reported 98.84% while class 27 (Pedestrians) was at 54% recall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from gtsrb.config.labels import CLASS_CATEGORY_NAMES, NUM_CLASSES, class_category
from gtsrb.training.metrics import (
    confusion_counts,
    expected_calibration_error,
    per_class_metrics,
)


@dataclass(frozen=True)
class ConfidenceStatistics:
    """How confident the model is, split by whether it was right."""

    mean_confidence: float
    mean_confidence_correct: float
    mean_confidence_incorrect: float
    expected_calibration_error: float
    num_samples: int

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class CategoryMetrics:
    """Metrics for one sign family."""

    category: str
    support: int
    correct: int
    precision: float
    recall: float
    f1: float

    @property
    def accuracy(self) -> float:
        """Share of this family's samples classified into the correct family."""
        return self.correct / self.support if self.support else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "support": self.support,
            "correct": self.correct,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


@dataclass
class EvaluationMetrics:
    """Everything computed from one pass over a labelled split."""

    split: str
    num_samples: int
    loss: float
    accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    confidence: ConfidenceStatistics
    per_class: list[dict[str, Any]] = field(default_factory=list)
    per_category: list[CategoryMetrics] = field(default_factory=list)

    def headline(self) -> dict[str, float]:
        """The scalar metrics worth reporting in a table or a README."""
        return {
            "accuracy": self.accuracy,
            "top3_accuracy": self.top3_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "loss": self.loss,
            "expected_calibration_error": self.confidence.expected_calibration_error,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "num_samples": self.num_samples,
            **self.headline(),
            "confidence": self.confidence.as_dict(),
            "per_class": self.per_class,
            "per_category": [row.as_dict() for row in self.per_category],
        }

    def summary(self) -> str:
        return (
            f"{self.split} ({self.num_samples} samples): "
            f"accuracy={self.accuracy:.4f} top3={self.top3_accuracy:.4f} "
            f"macro_f1={self.macro_f1:.4f} ECE={self.confidence.expected_calibration_error:.4f}"
        )


def category_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> list[CategoryMetrics]:
    """Aggregate precision/recall/F1 by sign family.

    Computed by collapsing the 43 classes into their families and treating family
    membership as the label. Macro-averaged over families present in ``targets``.
    """
    if predictions.size == 0:
        return []

    present = [
        name
        for name in CLASS_CATEGORY_NAMES
        if any(class_category(int(t)) == name for t in targets)
    ]
    category_index = {name: index for index, name in enumerate(present)}

    mapped_predictions = np.array([category_index[class_category(int(p))] for p in predictions])
    mapped_targets = np.array([category_index[class_category(int(t))] for t in targets])

    precision, recall, f1, support = precision_recall_fscore_support(
        mapped_targets,
        mapped_predictions,
        labels=list(range(len(present))),
        average=None,
        zero_division=0,
    )

    rows: list[CategoryMetrics] = []
    for index, name in enumerate(present):
        correct = int(((mapped_predictions == index) & (mapped_targets == index)).sum())
        rows.append(
            CategoryMetrics(
                category=name,
                support=int(support[index]),
                correct=correct,
                precision=float(precision[index]),
                recall=float(recall[index]),
                f1=float(f1[index]),
            )
        )
    return rows


def category_confusion(confusion: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Collapse a 43x43 class confusion matrix into a family-level matrix.

    Returns:
        ``(matrix, category_names)`` where ``matrix[i, j]`` counts samples of
        family ``i`` predicted as family ``j``.
    """
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            f"Expected a {NUM_CLASSES}x{NUM_CLASSES} confusion matrix, got {confusion.shape}"
        )

    names = list(CLASS_CATEGORY_NAMES)
    index = {name: position for position, name in enumerate(names)}
    collapsed = np.zeros((len(names), len(names)), dtype=np.int64)
    for true_class in range(NUM_CLASSES):
        for predicted_class in range(NUM_CLASSES):
            count = int(confusion[true_class, predicted_class])
            if count:
                collapsed[
                    index[class_category(true_class)], index[class_category(predicted_class)]
                ] += count
    return collapsed, names


def confidence_statistics(logits: torch.Tensor, targets: torch.Tensor) -> ConfidenceStatistics:
    """Summarise predicted-class confidence and calibration.

    Args:
        logits: ``(N, num_classes)`` model output.
        targets: ``(N,)`` ground-truth class ids.
    """
    if logits.shape[0] == 0:
        return ConfidenceStatistics(0.0, 0.0, 0.0, 0.0, 0)

    probabilities = torch.softmax(logits, dim=-1)
    confidence, predicted = probabilities.max(dim=1)
    correct = (predicted == targets).float()

    confidences = confidence.detach().cpu().numpy()
    correctness = correct.detach().cpu().numpy()

    was_correct = correctness > 0.5
    return ConfidenceStatistics(
        mean_confidence=float(confidences.mean()),
        mean_confidence_correct=float(confidences[was_correct].mean())
        if was_correct.any()
        else 0.0,
        mean_confidence_incorrect=(
            float(confidences[~was_correct].mean()) if (~was_correct).any() else 0.0
        ),
        expected_calibration_error=expected_calibration_error(confidences, correctness),
        num_samples=int(confidences.size),
    )


def compute_evaluation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    split: str,
    loss: float,
    num_classes: int = NUM_CLASSES,
) -> tuple[EvaluationMetrics, np.ndarray]:
    """Compute every metric from a complete pass over a split.

    Predictions and targets are held in memory rather than accumulated per batch.
    For GTSRB that is 12,630 integers — trivial — and a running average cannot
    compute macro F1 or a confusion matrix correctly.

    Args:
        logits: ``(N, num_classes)`` concatenated model output for the whole split.
        targets: ``(N,)`` ground-truth class ids.
        split: name of the split, for reporting.
        loss: mean loss over the split.
        num_classes: label space size.

    Returns:
        ``(metrics, confusion)``.
    """
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"logits and targets disagree on sample count: {logits.shape[0]} vs {targets.shape[0]}"
        )

    predictions = logits.argmax(dim=1).cpu().numpy()
    target_array = targets.cpu().numpy()

    # ``labels=range(num_classes)`` is not optional. Without it, sklearn averages
    # only over the classes that appear in y_true or y_pred, so a class the model
    # never predicts at all would silently drop out of the macro average instead of
    # contributing zero. That would also make evaluation macro metrics disagree with
    # the training macro metrics, which do pass an explicit label range.
    label_space = list(range(num_classes))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target_array, predictions, labels=label_space, average="macro", zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        target_array, predictions, labels=label_space, average="micro", zero_division=0
    )

    confusion = confusion_counts(predictions, target_array, num_classes)
    accuracy = float((predictions == target_array).mean())

    def top_k(k: int) -> float:
        if k > num_classes:
            return accuracy
        top = logits.topk(k, dim=1).indices
        return float((top == targets.unsqueeze(1)).any(dim=1).float().mean())

    metrics = EvaluationMetrics(
        split=split,
        num_samples=int(targets.shape[0]),
        loss=float(loss),
        accuracy=accuracy,
        top3_accuracy=top_k(3),
        top5_accuracy=top_k(5),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        micro_precision=float(micro_precision),
        micro_recall=float(micro_recall),
        micro_f1=float(micro_f1),
        confidence=confidence_statistics(logits, targets),
        per_class=per_class_metrics(predictions, target_array, num_classes),
        per_category=category_metrics(predictions, target_array),
    )
    return metrics, confusion


class BatchMetricsAccumulator:
    """Accumulate loss alongside concatenated logits and targets.

    Thin wrapper over :class:`~gtsrb.training.metrics.MetricsAccumulator` plus
    prediction retention, so evaluation produces the full arrays that a confusion
    matrix and error analysis need.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self._loss_sum = 0.0
        self._num_samples = 0
        self._logits: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        batch_size = int(targets.shape[0])
        self._loss_sum += float(loss) * batch_size
        self._num_samples += batch_size
        self._logits.append(logits.detach().cpu())
        self._targets.append(targets.detach().cpu())

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def mean_loss(self) -> float:
        return self._loss_sum / self._num_samples if self._num_samples else 0.0

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, targets)`` for the whole split."""
        if not self._logits:
            raise RuntimeError("No batches were accumulated")
        return torch.cat(self._logits, dim=0), torch.cat(self._targets, dim=0)


def compare_evaluations(results: dict[str, EvaluationMetrics]) -> list[dict[str, Any]]:
    """Build a comparison table across models, as the original project did.

    Returns one row per metric with a column per model, in percentage points —
    the format the original notebook's final table used, so the refactored numbers
    are directly comparable with the recorded ones.
    """
    if not results:
        return []

    rows: list[dict[str, Any]] = []
    for metric in (
        "accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
    ):
        row: dict[str, Any] = {"metric": metric}
        for name, result in results.items():
            row[name] = round(result.headline()[metric] * 100, 2)
        rows.append(row)
    return rows


__all__ = [
    "BatchMetricsAccumulator",
    "CategoryMetrics",
    "ConfidenceStatistics",
    "EvaluationMetrics",
    "category_confusion",
    "category_metrics",
    "compare_evaluations",
    "compute_evaluation_metrics",
    "confidence_statistics",
    "confusion_counts",
    "expected_calibration_error",
    "per_class_metrics",
]
