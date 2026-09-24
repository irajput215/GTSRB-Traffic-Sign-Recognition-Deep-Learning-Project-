"""Evaluation: metrics, error analysis and figures.

``evaluate_checkpoint`` is the entry point; it loads a checkpoint, runs inference
over a split, computes the metrics, analyses the errors and writes a complete
artifact set.

The original project computed accuracy plus macro/micro precision, recall and F1,
drew a confusion matrix and a per-class gallery, and displayed ten random
misclassified images. All of that is preserved here, and extended with the
questions a count-based view cannot answer: which classes fail, what they are
confused with, whether errors cross sign families, how confident the errors are,
and whether failures concentrate in the rare classes.
"""

from __future__ import annotations

from gtsrb.evaluation.error_analysis import (
    DEFAULT_CONFUSION_THRESHOLD,
    ClassPerformance,
    ConfusedPair,
    ErrorAnalysis,
    analyze_errors,
    family_split,
    top_confusions,
    within_and_across_family_errors,
    worst_classes,
)
from gtsrb.evaluation.evaluate import (
    EvaluationResult,
    PredictionSet,
    collect_predictions,
    confusion_from_predictions_csv,
    evaluate_checkpoint,
    load_predictions_csv,
    render_figures,
    select_loader,
)
from gtsrb.evaluation.metrics import (
    BatchMetricsAccumulator,
    CategoryMetrics,
    ConfidenceStatistics,
    EvaluationMetrics,
    category_confusion,
    category_metrics,
    compare_evaluations,
    compute_evaluation_metrics,
    confidence_statistics,
)
from gtsrb.evaluation.visualization import (
    plot_category_confusion,
    plot_class_distribution,
    plot_confidence_analysis,
    plot_confusion_matrix,
    plot_correct_predictions,
    plot_misclassified,
    plot_misclassified_by_class,
    plot_per_class_gallery,
    plot_per_class_recall,
    plot_samples_per_class,
    plot_top_confusions,
    plot_training_curves,
)

__all__ = [
    "DEFAULT_CONFUSION_THRESHOLD",
    "BatchMetricsAccumulator",
    "CategoryMetrics",
    "ClassPerformance",
    "ConfidenceStatistics",
    "ConfusedPair",
    "ErrorAnalysis",
    "EvaluationMetrics",
    "EvaluationResult",
    "PredictionSet",
    "analyze_errors",
    "category_confusion",
    "category_metrics",
    "collect_predictions",
    "compare_evaluations",
    "compute_evaluation_metrics",
    "confidence_statistics",
    "confusion_from_predictions_csv",
    "evaluate_checkpoint",
    "family_split",
    "load_predictions_csv",
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
    "render_figures",
    "select_loader",
    "top_confusions",
    "within_and_across_family_errors",
    "worst_classes",
]
