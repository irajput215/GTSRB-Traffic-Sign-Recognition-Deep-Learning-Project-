"""Evaluation plots.

Every figure is a function of arrays, not of a model or a dataset, so each can be
unit-tested with synthetic input and none of them can accidentally hold a
reference to training state.

The matplotlib backend is forced to ``Agg`` at import. Evaluation runs on servers
and in CI where there is no display, and matplotlib's default backend selection is
the usual cause of a run dying at the last step after an hour of training.

Figures are saved to disk and the path returned; nothing calls ``plt.show()``,
because in a non-interactive context that is a no-op at best.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from gtsrb.config.labels import CLASS_SHORT_NAMES, NUM_CLASSES, class_short_name
from gtsrb.data.preprocessing import to_display_array
from gtsrb.training.metrics import confusion_counts

FIGURE_DPI = 130


def _save(figure: plt.Figure, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_class_distribution(
    counts: dict[int, int],
    output_path: Path,
    *,
    title: str = "Class distribution",
    log_scale: bool = False,
) -> Path:
    """Bar chart of per-class sample counts.

    GTSRB is long-tailed — 150 samples for the rarest class against 1500 for the
    most common — so this is the plot that motivates the balancing strategy.
    """
    figure, axes = plt.subplots(figsize=(12, 4.5))
    classes = sorted(counts)
    axes.bar(classes, [counts[c] for c in classes], color="steelblue")
    axes.set_xlabel("Class id")
    axes.set_ylabel("Samples (log scale)" if log_scale else "Samples")
    axes.set_title(title)
    axes.set_xticks(range(0, NUM_CLASSES, 2))
    if log_scale:
        axes.set_yscale("log")
    axes.grid(axis="y", linestyle="--", alpha=0.4)
    return _save(figure, output_path)


def plot_confusion_matrix(
    confusion: np.ndarray,
    output_path: Path,
    *,
    normalize: bool = True,
    title: str | None = None,
    annotate_threshold: float = 1.0,
) -> Path:
    """Row-normalised confusion matrix.

    Row normalisation is the right choice for 43 imbalanced classes: raw counts are
    dominated by the common classes, and the interesting question is "of the
    samples whose true class was X, where did they go?".

    Args:
        confusion: ``(43, 43)`` integer counts, ``[true, predicted]``.
        output_path: where to write the PNG.
        normalize: divide each row by its sum and show percentages.
        title: figure title.
        annotate_threshold: only annotate cells at or above this percentage. A
            fully annotated 43x43 grid is unreadable.
    """
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(f"Expected a {NUM_CLASSES}x{NUM_CLASSES} matrix, got {confusion.shape}")

    display = confusion.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = (
            np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0) * 100.0
        )

    figure, axes = plt.subplots(figsize=(12, 10.5))
    image = axes.imshow(display, interpolation="nearest", cmap="viridis")
    figure.colorbar(image, ax=axes, label="% of true class" if normalize else "count")
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    axes.set_title(
        title
        or ("Confusion matrix (row-normalised, %)" if normalize else "Confusion matrix (counts)")
    )
    ticks = np.arange(NUM_CLASSES)
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)
    axes.set_xticklabels([class_short_name(int(i)) for i in ticks], rotation=90, fontsize=6)
    axes.set_yticklabels([class_short_name(int(i)) for i in ticks], fontsize=6)

    ceiling = display.max() if display.size else 0.0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = display[row, column]
            if value >= annotate_threshold:
                axes.text(
                    column,
                    row,
                    f"{value:.0f}" if normalize else str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if value > ceiling * 0.6 else "black",
                )
    return _save(figure, output_path)


def plot_per_class_recall(
    per_class: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Per-class recall",
    highlight_threshold: float = 0.95,
) -> Path:
    """Horizontal bar chart of recall per class, worst first.

    Sorted ascending so the problem classes are at the top. Classes below
    ``highlight_threshold`` are coloured differently: on the recorded run this is
    what makes class 27 (Pedestrians, 54%) impossible to miss.
    """
    ordered = sorted(per_class, key=lambda row: row["recall"])
    labels = [f"{row['class_id']:2d} {class_short_name(int(row['class_id']))}" for row in ordered]
    values = [row["recall"] * 100 for row in ordered]
    colors = ["firebrick" if value < highlight_threshold * 100 else "steelblue" for value in values]

    figure, axes = plt.subplots(figsize=(9, 10))
    axes.barh(labels, values, color=colors)
    axes.set_xlim(0, 105)
    axes.set_xlabel("Recall (%)")
    axes.set_title(title)
    axes.axvline(highlight_threshold * 100, color="black", linestyle="--", linewidth=0.8)
    axes.grid(axis="x", linestyle="--", alpha=0.4)
    axes.tick_params(axis="y", labelsize=7)
    return _save(figure, output_path)


def plot_top_confusions(
    pairs: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    top_n: int = 15,
    title: str = "Most frequent confusions",
) -> Path:
    """Bar chart of the largest off-diagonal confusions.

    Expressed as a share of the true class rather than a raw count, so a confusion
    affecting a rare class is not hidden by one affecting a common class.
    """
    selected = list(pairs)[:top_n]
    if not selected:
        selected = []

    labels = [
        f"{row['true_class_id']} {class_short_name(int(row['true_class_id']))}"
        f"  ->  {row['predicted_class_id']} {class_short_name(int(row['predicted_class_id']))}"
        for row in selected
    ]
    values = [row["share_of_true_class"] * 100 for row in selected]

    figure, axes = plt.subplots(figsize=(10, max(3, 0.45 * len(selected) + 1.5)))
    if selected:
        axes.barh(labels[::-1], values[::-1], color="darkorange")
        axes.set_xlabel("Share of the true class (%)")
    else:
        axes.text(0.5, 0.5, "No confusions above threshold", ha="center", va="center")
        axes.axis("off")
    axes.set_title(title)
    axes.grid(axis="x", linestyle="--", alpha=0.4)
    axes.tick_params(axis="y", labelsize=7)
    return _save(figure, output_path)


def plot_category_confusion(
    collapsed: np.ndarray,
    categories: Sequence[str],
    output_path: Path,
    *,
    title: str = "Confusion between sign families",
) -> Path:
    """Row-normalised confusion matrix collapsed to sign families.

    Answers a question the 43x43 matrix cannot: are errors within a family (an
    inherently hard, fine-grained mistake) or across families (a qualitatively
    different failure)?
    """
    row_sums = collapsed.sum(axis=1, keepdims=True)
    display = (
        np.divide(
            collapsed, row_sums, out=np.zeros_like(collapsed, dtype=float), where=row_sums > 0
        )
        * 100.0
    )

    figure, axes = plt.subplots(figsize=(7, 6))
    image = axes.imshow(display, interpolation="nearest", cmap="magma", vmin=0, vmax=100)
    figure.colorbar(image, ax=axes, label="% of true family")
    axes.set_xticks(range(len(categories)))
    axes.set_yticks(range(len(categories)))
    axes.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    axes.set_yticklabels(categories, fontsize=8)
    axes.set_xlabel("Predicted family")
    axes.set_ylabel("True family")
    axes.set_title(title)
    for row in range(len(categories)):
        for column in range(len(categories)):
            value = display[row, column]
            if value >= 0.5:
                axes.text(
                    column,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < 60 else "black",
                )
    return _save(figure, output_path)


def plot_training_curves(
    history: Sequence[dict[str, float]],
    output_path: Path,
    *,
    title: str = "Training history",
) -> Path:
    """Loss and accuracy curves for train and validation, side by side.

    Expects the per-epoch dicts produced by the trainer, with ``train_*`` and
    ``val_*`` keys.
    """
    if not history:
        raise ValueError("Cannot plot an empty training history")

    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="validation")
    if any("val_macro_f1" in row for row in history):
        axes[1].plot(
            epochs,
            [row.get("val_macro_f1", float("nan")) for row in history],
            label="validation macro F1",
            linestyle=":",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy")
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].legend()

    figure.suptitle(title)
    return _save(figure, output_path)


def plot_confidence_analysis(
    confidences: np.ndarray,
    correct: np.ndarray,
    output_path: Path,
    *,
    num_bins: int = 15,
    title: str = "Confidence and calibration",
) -> Path:
    """Confidence histogram plus a reliability diagram.

    The histogram shows whether errors are confident or hesitant. The reliability
    diagram compares stated confidence against observed accuracy per bin; the gap is
    the expected calibration error. For a safety-facing classifier an over-confident
    model is more dangerous than an inaccurate one, because downstream logic trusts
    the score.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    is_correct = correct > 0.5
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    axes[0].hist(confidences[is_correct], bins=bins, alpha=0.75, label="correct", color="steelblue")
    axes[0].hist(
        confidences[~is_correct], bins=bins, alpha=0.75, label="incorrect", color="firebrick"
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Predicted-class confidence")
    axes[0].set_ylabel("Samples (log scale)")
    axes[0].set_title("Confidence distribution")
    axes[0].legend()
    axes[0].grid(linestyle="--", alpha=0.4)

    bin_ids = np.clip(np.digitize(confidences, bins[1:-1]), 0, num_bins - 1)
    centres: list[float] = []
    observed: list[float] = []
    for bin_id in range(num_bins):
        mask = bin_ids == bin_id
        if mask.any():
            centres.append(float(confidences[mask].mean()))
            observed.append(float(is_correct[mask].mean()))

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8, label="perfect")
    if centres:
        axes[1].plot(centres, observed, marker="o", color="darkorange", label="observed")
    axes[1].set_xlabel("Mean confidence in bin")
    axes[1].set_ylabel("Observed accuracy")
    axes[1].set_title("Reliability diagram")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(linestyle="--", alpha=0.4)

    figure.suptitle(title)
    return _save(figure, output_path)


def _render_prediction_grid(
    images: torch.Tensor,
    rows_data: Sequence[tuple[int, int, float, float]],
    output_path: Path,
    *,
    title: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    columns: int = 5,
) -> Path:
    """Shared renderer for the correct/incorrect prediction galleries.

    ``rows_data`` holds ``(index, true_class, predicted_class, confidence)``.
    """
    count = len(rows_data)
    if count == 0:
        figure, single_axis = plt.subplots(figsize=(6, 2))
        single_axis.text(0.5, 0.5, "Nothing to display", ha="center", va="center")
        single_axis.axis("off")
        return _save(figure, output_path)

    columns = min(columns, count)
    rows = (count + columns - 1) // columns
    figure, grid = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.3 * rows), squeeze=False)

    for position, (index, true_class, predicted_class, confidence) in enumerate(rows_data):
        axis = grid[position // columns][position % columns]
        axis.imshow(to_display_array(images[index], mean, std))
        is_correct = true_class == predicted_class
        axis.set_title(
            f"true {true_class} {class_short_name(int(true_class))}\n"
            f"pred {predicted_class} {class_short_name(int(predicted_class))}\n"
            f"p={confidence:.3f}",
            fontsize=7,
            color="darkgreen" if is_correct else "firebrick",
        )
        axis.axis("off")

    for position in range(count, rows * columns):
        grid[position // columns][position % columns].axis("off")

    figure.suptitle(title, fontsize=11)
    return _save(figure, output_path)


def plot_misclassified(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    output_path: Path,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    max_images: int = 10,
    min_confidence: float = 0.0,
    title: str = "Misclassified examples",
) -> Path:
    """Gallery of incorrect predictions, most confident first.

    Sorting by confidence puts the model's *confident* mistakes first. Those are
    the dangerous ones; low-confidence errors on a blurred image are expected.
    """
    target_array = targets.cpu().numpy()
    prediction_array = predictions.cpu().numpy()
    confidence_array = confidences.cpu().numpy()

    wrong = np.where(target_array != prediction_array)[0]
    if min_confidence > 0:
        wrong = wrong[confidence_array[wrong] >= min_confidence]
    wrong = wrong[np.argsort(-confidence_array[wrong])][:max_images]

    return _render_prediction_grid(
        images,
        [
            (int(i), int(target_array[i]), int(prediction_array[i]), float(confidence_array[i]))
            for i in wrong
        ],
        output_path,
        title=title,
        mean=mean,
        std=std,
    )


def plot_correct_predictions(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    output_path: Path,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    max_images: int = 10,
    title: str = "Correct predictions",
) -> Path:
    """Gallery of correct predictions, most confident first.

    Included so the README shows what the model gets right, not only where it
    fails. A gallery of only failures misrepresents a 98%-accurate model.
    """
    target_array = targets.cpu().numpy()
    prediction_array = predictions.cpu().numpy()
    confidence_array = confidences.cpu().numpy()

    right = np.where(target_array == prediction_array)[0]
    right = right[np.argsort(-confidence_array[right])][:max_images]

    return _render_prediction_grid(
        images,
        [
            (int(i), int(target_array[i]), int(prediction_array[i]), float(confidence_array[i]))
            for i in right
        ],
        output_path,
        title=title,
        mean=mean,
        std=std,
    )


def plot_misclassified_by_class(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    output_path: Path,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    class_ids: Sequence[int] | None = None,
    max_classes: int = 8,
    title: str = "Errors for the worst classes",
) -> Path:
    """One misclassified example per poorly-performing class.

    Complements the overall error gallery: it shows *which classes* the errors
    come from rather than which individual images the model was most sure about.
    """
    target_array = targets.cpu().numpy()
    prediction_array = predictions.cpu().numpy()
    confidence_array = confidences.cpu().numpy()

    if class_ids is None:
        confusion = confusion_counts(prediction_array, target_array, NUM_CLASSES)
        row_totals = np.maximum(confusion.sum(axis=1), 1)
        recall = confusion.diagonal() / row_totals
        # Only classes that actually have errors; a class at 100% recall has none.
        with_errors = [
            int(c) for c in np.argsort(recall) if confusion[c].sum() - confusion[c, c] > 0
        ]
        class_ids = with_errors[:max_classes]

    selected: list[tuple[int, int, int, float]] = []
    for class_id in list(class_ids)[:max_classes]:
        candidates = np.where((target_array == class_id) & (prediction_array != class_id))[0]
        if candidates.size == 0:
            continue
        best = candidates[np.argmax(confidence_array[candidates])]
        selected.append(
            (int(best), class_id, int(prediction_array[best]), float(confidence_array[best]))
        )

    return _render_prediction_grid(images, selected, output_path, title=title, mean=mean, std=std)


def plot_per_class_gallery(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    output_path: Path,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    title: str = "One sample per class with true and correctly predicted counts",
) -> Path:
    """Grid with a representative image per class and its correct/total count.

    Recreates the original project's per-class gallery while adding the class name,
    which the original only had as a numeric id.
    """
    target_array = targets.cpu().numpy()
    prediction_array = predictions.cpu().numpy()
    confusion = confusion_counts(prediction_array, target_array, NUM_CLASSES)
    totals = confusion.sum(axis=1)
    correct = confusion.diagonal()

    columns = 9
    rows = (NUM_CLASSES + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(1.65 * columns, 1.95 * rows), squeeze=False)

    for class_id in range(NUM_CLASSES):
        axis = axes[class_id // columns][class_id % columns]
        candidates = np.where(target_array == class_id)[0]
        if candidates.size == 0:
            axis.axis("off")
            continue
        axis.imshow(to_display_array(images[int(candidates[0])], mean, std))
        name = CLASS_SHORT_NAMES[class_id]
        axis.set_title(
            f"{class_id} {name}\n{correct[class_id]}/{totals[class_id]}",
            fontsize=5.5,
        )
        axis.axis("off")

    for position in range(NUM_CLASSES, rows * columns):
        axes[position // columns][position % columns].axis("off")

    figure.suptitle(title, fontsize=11)
    return _save(figure, output_path)


def plot_samples_per_class(
    images: torch.Tensor,
    targets: torch.Tensor,
    output_path: Path,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    columns: int = 10,
    rows: int = 5,
    title: str = "Sample images",
) -> Path:
    """Plain grid of sample images with their labels."""
    limit = min(columns * rows, int(images.shape[0]))
    figure, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.85 * rows), squeeze=False)
    for position in range(rows * columns):
        axis = axes[position // columns][position % columns]
        if position >= limit:
            axis.axis("off")
            continue
        axis.imshow(to_display_array(images[position], mean, std))
        class_id = int(targets[position])
        axis.set_title(f"{class_id} {class_short_name(class_id)}", fontsize=5.5)
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    return _save(figure, output_path)


__all__ = [
    "FIGURE_DPI",
    "plot_category_confusion",
    "plot_class_distribution",
    "plot_confidence_analysis",
    "plot_confusion_matrix",
    "plot_correct_predictions",
    "plot_misclassified",
    "plot_misclassified_by_class",
    "plot_per_class_gallery",
    "plot_per_class_recall",
    "plot_samples_per_class",
    "plot_top_confusions",
    "plot_training_curves",
]
