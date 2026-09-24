"""Evaluation runner.

One function evaluates a checkpoint on a labelled split and writes a complete,
reproducible artifact set:

```text
artifacts/evaluation/<model>/
├── metrics.json              scalar metrics, per-class and per-category tables
├── per_class.csv             per-class precision/recall/F1/support
├── confusion_matrix.csv      raw counts, readable without this code
├── confusion_matrix.npy      raw counts, for scripts
├── error_analysis.json       structured error analysis
├── error_analysis.md         the same analysis as a written report
├── predictions.csv           per-sample prediction, confidence and correctness
└── figures/
    ├── confusion_matrix.png
    ├── category_confusion.png
    ├── per_class_recall.png
    ├── top_confusions.png
    ├── confidence_analysis.png
    ├── correct_predictions.png
    ├── misclassified.png
    ├── misclassified_by_class.png
    └── per_class_gallery.png
```

`predictions.csv` exists so the analysis can be re-run or extended without a GPU:
it is the full per-sample record, 12,630 rows.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from gtsrb.config.labels import NUM_CLASSES, class_name
from gtsrb.config.schema import ProjectConfig
from gtsrb.data import build_dataloaders
from gtsrb.data.dataset import TransformSubset
from gtsrb.evaluation.error_analysis import ErrorAnalysis, analyze_errors
from gtsrb.evaluation.metrics import (
    BatchMetricsAccumulator,
    EvaluationMetrics,
    category_confusion,
    compute_evaluation_metrics,
)
from gtsrb.evaluation.visualization import (
    plot_category_confusion,
    plot_confidence_analysis,
    plot_confusion_matrix,
    plot_correct_predictions,
    plot_misclassified,
    plot_misclassified_by_class,
    plot_per_class_gallery,
    plot_per_class_recall,
    plot_top_confusions,
)
from gtsrb.models import CheckpointMetadata, load_model_from_checkpoint
from gtsrb.runtime import get_logger, resolve_device
from gtsrb.training.losses import build_loss

logger = get_logger("evaluation.evaluate")

SplitName = str


@dataclass
class EvaluationResult:
    """Everything one evaluation produced."""

    checkpoint_path: str
    split: str
    metrics: EvaluationMetrics
    confusion: np.ndarray
    error_analysis: ErrorAnalysis
    artifacts: list[Path] = field(default_factory=list)
    metadata: CheckpointMetadata | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint_path,
            "split": self.split,
            "metrics": self.metrics.as_dict(),
            "error_analysis": self.error_analysis.as_dict(),
            "artifacts": [str(path) for path in self.artifacts],
        }

    def summary(self) -> str:
        return (
            f"{self.checkpoint_path} on {self.split}: {self.metrics.summary()}; "
            f"{self.error_analysis.num_errors} errors, "
            f"{len(self.error_analysis.top_confusions)} confusions above threshold"
        )


@dataclass(frozen=True)
class PredictionSet:
    """Model output for a whole split, kept together with the inputs."""

    logits: torch.Tensor
    targets: torch.Tensor
    images: torch.Tensor
    loss: float

    @property
    def predictions(self) -> torch.Tensor:
        return self.logits.argmax(dim=1)

    @property
    def confidences(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1).max(dim=1).values

    @property
    def correct(self) -> torch.Tensor:
        return (self.predictions == self.targets).float()


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    criterion: torch.nn.Module,
    keep_images: bool = True,
) -> PredictionSet:
    """Run the model over a loader and return logits, targets and (optionally) inputs.

    ``keep_images`` retains the decoded, normalised inputs so prediction galleries can
    be rendered without a second pass over the dataset. For the 12,630-image test
    split at 64x64 that is about 40 MB of float32, which is acceptable; set it to
    ``False`` for a memory-constrained or very large evaluation.
    """
    model.eval()
    accumulator = BatchMetricsAccumulator(num_classes=NUM_CLASSES)
    images_buffer: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_images, batch_targets in loader:
            images = batch_images.to(device, non_blocking=True)
            targets = batch_targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            accumulator.update(logits, targets, float(loss.detach().item()))
            if keep_images:
                images_buffer.append(images.detach().cpu())

    logits_all, targets_all = accumulator.stacked()
    images_all = (
        torch.cat(images_buffer, dim=0)
        if keep_images and images_buffer
        else torch.empty(0, dtype=torch.float32)
    )
    return PredictionSet(
        logits=logits_all,
        targets=targets_all,
        images=images_all,
        loss=accumulator.mean_loss,
    )


def select_loader(config: ProjectConfig, split: str) -> tuple[DataLoader, TransformSubset]:
    """Build the dataloaders and pick the requested split.

    Raises:
        ValueError: if ``split`` is not ``train``, ``val`` or ``test``.
    """
    bundle = build_dataloaders(config, download=False)
    if split == "train":
        return bundle.train, bundle.train_dataset
    if split == "val":
        return bundle.val, bundle.val_dataset
    if split == "test":
        return bundle.test, bundle.test_dataset
    raise ValueError(f"Unknown split {split!r}; expected train, val or test")


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_predictions(
    predictions: PredictionSet,
    path: Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    prediction_array = predictions.predictions.cpu().numpy()
    target_array = predictions.targets.cpu().numpy()
    confidence_array = predictions.confidences.cpu().numpy()
    probabilities = torch.softmax(predictions.logits, dim=-1).cpu().numpy()

    for index in range(prediction_array.size):
        true_class = int(target_array[index])
        predicted_class = int(prediction_array[index])
        top_k = np.argsort(-probabilities[index])[:3]
        rows.append(
            {
                "index": index,
                "true_class_id": true_class,
                "true_class": class_name(true_class),
                "predicted_class_id": predicted_class,
                "predicted_class": class_name(predicted_class),
                "confidence": round(float(confidence_array[index]), 6),
                "correct": bool(true_class == predicted_class),
                "top2_class_id": int(top_k[1]),
                "top3_class_id": int(top_k[2]),
            }
        )
    return _write_csv(rows, path)


def _category_counts(confusion: np.ndarray) -> list[dict[str, Any]]:
    collapsed, names = category_confusion(confusion)
    rows: list[dict[str, Any]] = []
    for true_index, true_name in enumerate(names):
        for predicted_index, predicted_name in enumerate(names):
            count = int(collapsed[true_index, predicted_index])
            if count:
                rows.append(
                    {
                        "true_family": true_name,
                        "predicted_family": predicted_name,
                        "count": count,
                        "same_family": true_name == predicted_name,
                    }
                )
    return rows


def render_figures(
    predictions: PredictionSet,
    confusion: np.ndarray,
    metrics: EvaluationMetrics,
    error_analysis: ErrorAnalysis,
    output_dir: Path,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> list[Path]:
    """Render every evaluation figure and return the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    figures.append(
        plot_confusion_matrix(
            confusion,
            output_dir / "confusion_matrix.png",
            title=f"Confusion matrix (row-normalised %) — {metrics.split} split",
        )
    )

    collapsed, categories = category_confusion(confusion)
    figures.append(
        plot_category_confusion(collapsed, categories, output_dir / "category_confusion.png")
    )

    figures.append(plot_per_class_recall(metrics.per_class, output_dir / "per_class_recall.png"))

    figures.append(
        plot_top_confusions(
            [pair.as_dict() for pair in error_analysis.top_confusions],
            output_dir / "top_confusions.png",
        )
    )

    confidences = predictions.confidences.cpu().numpy()
    correct = predictions.correct.cpu().numpy()
    figures.append(
        plot_confidence_analysis(confidences, correct, output_dir / "confidence_analysis.png")
    )

    if predictions.images.numel() > 0:
        common: dict[str, Any] = {"mean": mean, "std": std}
        figures.append(
            plot_correct_predictions(
                predictions.images,
                predictions.targets,
                predictions.predictions,
                predictions.confidences,
                output_dir / "correct_predictions.png",
                **common,
            )
        )
        figures.append(
            plot_misclassified(
                predictions.images,
                predictions.targets,
                predictions.predictions,
                predictions.confidences,
                output_dir / "misclassified.png",
                **common,
            )
        )
        figures.append(
            plot_misclassified(
                predictions.images,
                predictions.targets,
                predictions.predictions,
                predictions.confidences,
                output_dir / "misclassified_confident.png",
                min_confidence=0.9,
                title="Misclassified with confidence above 0.9",
                **common,
            )
        )
        figures.append(
            plot_misclassified_by_class(
                predictions.images,
                predictions.targets,
                predictions.predictions,
                predictions.confidences,
                output_dir / "misclassified_by_class.png",
                **common,
            )
        )
        figures.append(
            plot_per_class_gallery(
                predictions.images,
                predictions.targets,
                predictions.predictions,
                output_dir / "per_class_gallery.png",
                **common,
            )
        )

    return figures


def evaluate_checkpoint(
    config: ProjectConfig,
    checkpoint_path: Path | None = None,
    *,
    split: str = "test",
    output_dir: Path | None = None,
    render: bool = True,
    keep_images: bool | None = None,
) -> EvaluationResult:
    """Evaluate a checkpoint on a labelled split and persist the artifacts.

    Args:
        config: project configuration. ``data`` and ``inference`` are read; the
            model is rebuilt from the checkpoint's own stored config.
        checkpoint_path: checkpoint to evaluate. Defaults to
            ``config.inference.checkpoint_path``.
        split: ``train``, ``val`` or ``test``.
        output_dir: where artifacts are written. Defaults to
            ``<output_dir>/evaluation/<model name>``.
        render: whether to produce figures. Skipping them makes a metrics-only run
            much faster.
        keep_images: retain inputs for the galleries. Defaults to the value of
            ``render``, since the galleries are the only consumer.

    Returns:
        An :class:`EvaluationResult`.

    Raises:
        CheckpointError: if the checkpoint cannot be read (from
            ``load_model_from_checkpoint``).
    """
    checkpoint = Path(checkpoint_path or config.inference.checkpoint_path)
    device = resolve_device(config.inference.device)
    model, metadata = load_model_from_checkpoint(checkpoint, device=device)

    logger.info(
        "evaluating checkpoint",
        extra={
            "checkpoint": str(checkpoint),
            "split": split,
            "device": str(device),
            "model": metadata.model_name,
            "trained_epoch": metadata.epoch,
        },
    )

    loader, dataset = select_loader(config, split)
    criterion = build_loss(config.training)
    wants_images = render if keep_images is None else keep_images

    predictions = collect_predictions(
        model, loader, device=device, criterion=criterion, keep_images=wants_images
    )

    metrics, confusion = compute_evaluation_metrics(
        predictions.logits,
        predictions.targets,
        split=split,
        loss=predictions.loss,
        num_classes=metadata.num_classes,
    )

    error_analysis = analyze_errors(
        confusion,
        metrics.per_class,
        split=split,
        accuracy=metrics.accuracy,
        macro_f1=metrics.macro_f1,
        confidences=predictions.confidences.cpu().numpy(),
        correct=predictions.correct.cpu().numpy(),
    )

    target_dir = output_dir or Path(config.output_dir) / "evaluation" / metadata.model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = [
        _write_json(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_metadata": metadata.as_dict(),
                "dataset": {
                    "split": split,
                    "num_samples": len(dataset),
                    "image_size": metadata.image_size,
                },
                "metrics": metrics.as_dict(),
            },
            target_dir / "metrics.json",
        ),
        _write_csv(metrics.per_class, target_dir / "per_class.csv"),
        _write_json(error_analysis.as_dict(), target_dir / "error_analysis.json"),
        _write_csv(
            [pair.as_dict() for pair in error_analysis.top_confusions],
            target_dir / "top_confusions.csv",
        ),
        _write_csv(
            [row.as_dict() for row in metrics.per_category], target_dir / "per_category.csv"
        ),
        _write_csv(_category_counts(confusion), target_dir / "category_confusion.csv"),
        _write_predictions(predictions, target_dir / "predictions.csv"),
    ]

    confusion_path = target_dir / "confusion_matrix.npy"
    np.save(confusion_path, confusion)
    artifacts.append(confusion_path)
    confusion_rows: list[dict[str, Any]] = []
    for true_class in range(confusion.shape[0]):
        row = {"true_class_id": true_class, "true_class": class_name(true_class)}
        row.update({f"pred_{c}": int(confusion[true_class, c]) for c in range(confusion.shape[1])})
        confusion_rows.append(row)
    artifacts.append(_write_csv(confusion_rows, target_dir / "confusion_matrix.csv"))

    report = error_analysis.to_markdown()
    report_path = target_dir / "error_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    artifacts.append(report_path)

    if render:
        artifacts.extend(
            render_figures(
                predictions,
                confusion,
                metrics,
                error_analysis,
                target_dir / "figures",
                mean=metadata.normalize_mean,
                std=metadata.normalize_std,
            )
        )

    logger.info(
        "evaluation complete",
        extra={
            "split": split,
            "accuracy": round(metrics.accuracy, 4),
            "macro_f1": round(metrics.macro_f1, 4),
            "top3_accuracy": round(metrics.top3_accuracy, 4),
            "errors": error_analysis.num_errors,
            "output_dir": str(target_dir),
            "artifacts": len(artifacts),
        },
    )

    return EvaluationResult(
        checkpoint_path=str(checkpoint),
        split=split,
        metrics=metrics,
        confusion=confusion,
        error_analysis=error_analysis,
        artifacts=artifacts,
        metadata=metadata,
    )


def load_predictions_csv(path: Path) -> list[dict[str, Any]]:
    """Read a ``predictions.csv`` back into rows.

    Provided so the error analysis can be re-run from a saved evaluation without a
    GPU, and so the file format is exercised by a test.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def confusion_from_predictions_csv(rows: list[dict[str, Any]]) -> np.ndarray:
    """Rebuild a confusion matrix from ``predictions.csv`` rows."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for row in rows:
        true_class = int(row["true_class_id"])
        predicted_class = int(row["predicted_class_id"])
        if 0 <= true_class < NUM_CLASSES and 0 <= predicted_class < NUM_CLASSES:
            matrix[true_class, predicted_class] += 1
    return matrix


__all__ = [
    "EvaluationResult",
    "PredictionSet",
    "collect_predictions",
    "confusion_from_predictions_csv",
    "evaluate_checkpoint",
    "load_predictions_csv",
    "render_figures",
    "select_loader",
]
