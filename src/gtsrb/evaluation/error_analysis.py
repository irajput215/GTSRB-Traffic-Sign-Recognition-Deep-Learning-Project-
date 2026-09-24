"""Error analysis.

Turns a confusion matrix into the questions an engineer would actually ask:

1. **Which classes does the model fail on?** Ranked by recall, not by error count.
   A class with 30 samples that is 50% wrong matters more than a class with 1500
   samples that is 2% wrong, and a count-based ranking buries it.
2. **What does it confuse them with?** Expressed as a share of the true class, so a
   confusion affecting a rare class is not hidden by one affecting a common class.
3. **Are the errors within a sign family or across families?** Confusing two speed
   limits is fine-grained; confusing a speed limit with a warning triangle is a
   different kind of failure.
4. **Which errors is the model confident about?** Confidently wrong is the
   dangerous case.
5. **Are the classes it fails on rare?** If the failures concentrate in the
   long tail, the fix is data or sampling, not architecture.

This module produces the numbers; :mod:`gtsrb.evaluation.visualization` produces the
figures. Both are driven by :mod:`gtsrb.evaluation.evaluate`.

The original project produced a confusion matrix, a per-class gallery and ten
random misclassified images. Those are a good start, but "here are ten random
failures" is not an analysis — it does not say which classes are affected, how
badly, or what to do about it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from gtsrb.config.labels import NUM_CLASSES, class_category, class_name, class_short_name

#: An off-diagonal cell must hold at least this share of its true class to be
#: reported. Below it, the "confusion" is one or two images and reporting it would
#: be noise dressed up as a finding.
DEFAULT_CONFUSION_THRESHOLD = 0.005


@dataclass(frozen=True)
class ConfusedPair:
    """A true class that is frequently predicted as another class."""

    true_class_id: int
    predicted_class_id: int
    count: int
    share_of_true_class: float
    same_family: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "true_class_name": class_name(self.true_class_id),
            "predicted_class_name": class_name(self.predicted_class_id),
            "true_class_short": class_short_name(self.true_class_id),
            "predicted_class_short": class_short_name(self.predicted_class_id),
            "true_family": class_category(self.true_class_id),
            "predicted_family": class_category(self.predicted_class_id),
        }


@dataclass(frozen=True)
class ClassPerformance:
    """How the model does on one class, with the context needed to judge it."""

    class_id: int
    support: int
    correct: int
    recall: float
    precision: float
    f1: float
    top_confusion: int | None
    top_confusion_count: int
    split: str

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "class_name": class_name(self.class_id),
            "class_short": class_short_name(self.class_id),
            "family": class_category(self.class_id),
            "top_confusion_name": (
                class_name(self.top_confusion) if self.top_confusion is not None else None
            ),
            "error_count": self.support - self.correct,
        }


@dataclass(frozen=True)
class ErrorAnalysis:
    """The complete error analysis for one evaluation."""

    split: str
    num_samples: int
    accuracy: float
    macro_f1: float
    num_errors: int
    worst_classes: list[ClassPerformance] = field(default_factory=list)
    top_confusions: list[ConfusedPair] = field(default_factory=list)
    within_family_error_share: float = 0.0
    across_family_error_share: float = 0.0
    high_confidence_errors: int = 0
    high_confidence_error_share: float = 0.0
    rare_class_error_share: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "num_errors": self.num_errors,
            "worst_classes": [row.as_dict() for row in self.worst_classes],
            "top_confusions": [row.as_dict() for row in self.top_confusions],
            "within_family_error_share": self.within_family_error_share,
            "across_family_error_share": self.across_family_error_share,
            "high_confidence_errors": self.high_confidence_errors,
            "high_confidence_error_share": self.high_confidence_error_share,
            "rare_class_error_share": self.rare_class_error_share,
        }

    def to_markdown(self) -> str:
        """Render the analysis as Markdown for the evaluation report."""
        lines: list[str] = []
        lines.append(f"# Error analysis — {self.split} split")
        lines.append("")
        lines.append(
            f"{self.num_errors} errors out of {self.num_samples} samples "
            f"(accuracy {self.accuracy * 100:.2f}%, macro F1 {self.macro_f1:.4f})."
        )
        lines.append("")

        lines.append("## Worst classes by recall")
        lines.append("")
        header = (
            "| Class | Name | Family | Support | Correct | Recall | Prec | F1 "
            "| Most confused with |"
        )
        lines.append(header)
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in self.worst_classes:
            confused_with = "—"
            if row.top_confusion is not None:
                confused_with = (
                    f"{row.top_confusion} {class_short_name(row.top_confusion)} "
                    f"({row.top_confusion_count})"
                )
            lines.append(
                f"| {row.class_id} | {class_name(row.class_id)} | {class_category(row.class_id)} "
                f"| {row.support} | {row.correct} | {row.recall * 100:.1f}% "
                f"| {row.precision:.3f} | {row.f1:.3f} | {confused_with} |"
            )
        lines.append("")

        lines.append("## Most frequent confusions")
        lines.append("")
        lines.append("| True | Predicted | Count | Share of true class | Same family |")
        lines.append("| --- | --- | ---: | ---: | :---: |")
        if self.top_confusions:
            for pair in self.top_confusions:
                lines.append(
                    f"| {pair.true_class_id} {class_short_name(pair.true_class_id)} "
                    f"| {pair.predicted_class_id} {class_short_name(pair.predicted_class_id)} "
                    f"| {pair.count} | {pair.share_of_true_class * 100:.1f}% "
                    f"| {'yes' if pair.same_family else 'no'} |"
                )
        else:
            lines.append("| — | — | 0 | 0% | — |")
        lines.append("")

        lines.append("## What the errors look like")
        lines.append("")
        lines.append(
            f"- **Within-family errors**: {self.within_family_error_share * 100:.1f}% of errors. "
            "These are fine-grained mistakes between visually similar signs."
        )
        lines.append(
            f"- **Across-family errors**: {self.across_family_error_share * 100:.1f}% of errors. "
            "These cross sign categories and are the more serious kind."
        )
        lines.append(
            f"- **Confident errors**: {self.high_confidence_errors} errors "
            f"({self.high_confidence_error_share * 100:.1f}% of all errors) were made with "
            "top-1 confidence above 0.9."
        )
        lines.append(
            f"- **Rare-class errors**: {self.rare_class_error_share * 100:.1f}% of errors fall in "
            "classes that are in the smallest third by training support."
        )
        lines.append("")
        return "\n".join(lines)


def top_confusions(
    confusion: np.ndarray,
    *,
    max_pairs: int = 15,
    threshold: float = DEFAULT_CONFUSION_THRESHOLD,
) -> list[ConfusedPair]:
    """Rank off-diagonal confusions by their share of the true class.

    Args:
        confusion: ``(43, 43)`` integer counts, ``[true, predicted]``.
        max_pairs: how many pairs to return.
        threshold: minimum share of the true class to report.

    Raises:
        ValueError: if the matrix is not square and of the expected size.
    """
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(f"Expected a {NUM_CLASSES}x{NUM_CLASSES} matrix, got {confusion.shape}")

    row_totals = confusion.sum(axis=1)
    pairs: list[ConfusedPair] = []
    for true_class in range(NUM_CLASSES):
        total = int(row_totals[true_class])
        if total == 0:
            continue
        for predicted_class in range(NUM_CLASSES):
            if predicted_class == true_class:
                continue
            count = int(confusion[true_class, predicted_class])
            if count == 0:
                continue
            share = count / total
            if share < threshold:
                continue
            pairs.append(
                ConfusedPair(
                    true_class_id=true_class,
                    predicted_class_id=predicted_class,
                    count=count,
                    share_of_true_class=share,
                    same_family=class_category(true_class) == class_category(predicted_class),
                )
            )
    pairs.sort(key=lambda pair: (-pair.share_of_true_class, -pair.count))
    return pairs[:max_pairs]


def worst_classes(
    per_class: list[dict[str, Any]],
    confusion: np.ndarray,
    *,
    split: str,
    max_classes: int = 10,
    max_recall: float = 1.0,
) -> list[ClassPerformance]:
    """Rank classes by recall, worst first.

    Args:
        per_class: rows from :func:`gtsrb.training.metrics.per_class_metrics`.
        confusion: the confusion matrix, used to name each class's main confusion.
        split: split name, carried into the report.
        max_classes: how many rows to return.
        max_recall: only include classes at or below this recall.

    Returns:
        One :class:`ClassPerformance` per selected class.
    """
    rows: list[ClassPerformance] = []
    for entry in per_class:
        class_id = int(entry["class_id"])
        support = int(entry["support"])
        if support == 0:
            continue
        if float(entry["recall"]) > max_recall:
            continue
        row = confusion[class_id].copy()
        row[class_id] = 0
        top_confusion = int(row.argmax()) if row.max() > 0 else None
        rows.append(
            ClassPerformance(
                class_id=class_id,
                support=support,
                correct=int(confusion[class_id, class_id]),
                recall=float(entry["recall"]),
                precision=float(entry["precision"]),
                f1=float(entry["f1"]),
                top_confusion=top_confusion if row.max() > 0 else None,
                top_confusion_count=int(row.max()) if row.max() > 0 else 0,
                split=split,
            )
        )
    rows.sort(key=lambda item: (item.recall, -item.support))
    return rows[:max_classes]


def within_and_across_family_errors(confusion: np.ndarray) -> tuple[int, int]:
    """Count errors that stay inside a sign family and errors that cross families.

    A within-family error is an off-diagonal cell connecting two classes that share
    a family; everything else off-diagonal crosses families. Reported separately
    because the two mean different things: "confused a 60 sign with an 80 sign" and
    "confused a 60 sign with a warning triangle" are not the same failure.

    Returns:
        ``(within_family_errors, across_family_errors)``.
    """
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(f"Expected a {NUM_CLASSES}x{NUM_CLASSES} matrix, got {confusion.shape}")

    within = 0
    across = 0
    for true_class in range(NUM_CLASSES):
        true_family = class_category(true_class)
        for predicted_class in range(NUM_CLASSES):
            if predicted_class == true_class:
                continue
            count = int(confusion[true_class, predicted_class])
            if count == 0:
                continue
            if class_category(predicted_class) == true_family:
                within += count
            else:
                across += count
    return within, across


def family_split(confusion: np.ndarray) -> tuple[float, float]:
    """Share of errors that stay within a sign family versus crossing families.

    Returns:
        ``(within_family_share, across_family_share)`` as fractions of all errors.
        ``(0.0, 0.0)`` when there are no errors.
    """
    within, across = within_and_across_family_errors(confusion)
    total = within + across
    if total == 0:
        return 0.0, 0.0
    return within / total, across / total


def analyze_errors(
    confusion: np.ndarray,
    per_class: list[dict[str, Any]],
    *,
    split: str,
    accuracy: float,
    macro_f1: float,
    confidences: np.ndarray | None = None,
    correct: np.ndarray | None = None,
    class_support: dict[int, int] | None = None,
    confidence_threshold: float = 0.9,
    max_classes: int = 10,
    max_pairs: int = 15,
    rare_fraction: float = 1 / 3,
) -> ErrorAnalysis:
    """Build the full error analysis.

    Args:
        confusion: ``(43, 43)`` counts, ``[true, predicted]``.
        per_class: rows from :func:`gtsrb.training.metrics.per_class_metrics`.
        split: split name.
        accuracy: overall accuracy, carried into the report.
        macro_f1: macro F1, carried into the report.
        confidences: predicted-class confidence per sample. Enables the
            confident-error statistic.
        correct: 1.0 where the prediction was right. Enables the confident-error
            statistic.
        class_support: training-set support per class, used to decide which classes
            are "rare". Defaults to the test-split support from ``confusion``.
        confidence_threshold: confidence above which an error counts as confident.
        max_classes: how many classes to include in the worst-classes table.
        max_pairs: how many confusions to include.
        rare_fraction: fraction of classes, by ascending support, treated as rare.

    Returns:
        An :class:`ErrorAnalysis`.
    """
    num_samples = int(confusion.sum())
    num_errors = num_samples - int(confusion.diagonal().sum())
    within, across = within_and_across_family_errors(confusion)
    total_errors = within + across

    high_confidence_errors = 0
    if confidences is not None and correct is not None and confidences.size:
        high_confidence_errors = int(
            ((confidences >= confidence_threshold) & (correct < 0.5)).sum()
        )

    support = class_support or {
        int(row["class_id"]): int(row["support"]) for row in per_class if int(row["support"]) > 0
    }
    rare_class_error_share = 0.0
    if support and num_errors:
        ordered = sorted(support.items(), key=lambda item: item[1])
        rare_count = max(1, int(len(ordered) * rare_fraction))
        rare_classes = {class_id for class_id, _ in ordered[:rare_count]}
        rare_errors = sum(
            int(confusion[class_id].sum() - confusion[class_id, class_id])
            for class_id in rare_classes
            if class_id < NUM_CLASSES
        )
        rare_class_error_share = rare_errors / num_errors

    return ErrorAnalysis(
        split=split,
        num_samples=num_samples,
        accuracy=accuracy,
        macro_f1=macro_f1,
        num_errors=num_errors,
        worst_classes=worst_classes(per_class, confusion, split=split, max_classes=max_classes),
        top_confusions=top_confusions(confusion, max_pairs=max_pairs),
        within_family_error_share=(within / total_errors) if total_errors else 0.0,
        across_family_error_share=(across / total_errors) if total_errors else 0.0,
        high_confidence_errors=high_confidence_errors,
        high_confidence_error_share=(high_confidence_errors / num_errors) if num_errors else 0.0,
        rare_class_error_share=rare_class_error_share,
    )


__all__ = [
    "DEFAULT_CONFUSION_THRESHOLD",
    "ClassPerformance",
    "ConfusedPair",
    "ErrorAnalysis",
    "analyze_errors",
    "family_split",
    "top_confusions",
    "within_and_across_family_errors",
    "worst_classes",
]
