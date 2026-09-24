"""Unit tests for evaluation metrics, error analysis, figures and the runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from gtsrb.config.schema import ProjectConfig
from gtsrb.data.dataset import TransformSubset
from gtsrb.evaluation import (
    BatchMetricsAccumulator,
    ErrorAnalysis,
    analyze_errors,
    category_confusion,
    category_metrics,
    collect_predictions,
    compare_evaluations,
    compute_evaluation_metrics,
    confidence_statistics,
    confusion_from_predictions_csv,
    evaluate_checkpoint,
    family_split,
    load_predictions_csv,
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
    top_confusions,
    within_and_across_family_errors,
    worst_classes,
)
from gtsrb.models import CheckpointError, build_metadata, build_model, save_checkpoint
from gtsrb.training.metrics import per_class_metrics
from tests.conftest import make_sign_image

pytestmark = pytest.mark.unit

#: (images, targets, predictions, confidences) for the gallery tests.
GalleryBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

#: (config, checkpoint path) for the evaluation-runner tests.
TrainedCheckpoint = tuple[ProjectConfig, Path]


def make_logits(
    targets: list[int],
    *,
    num_classes: int = 43,
    correct: list[bool] | None = None,
    confidence: float = 8.0,
    wrong_class: int | None = None,
) -> torch.Tensor:
    """Build logits that are deliberately right or wrong for each target.

    ``confidence`` is the logit gap between the top class and the rest, so it maps
    monotonically onto softmax confidence.
    """
    flags = correct if correct is not None else [True] * len(targets)
    logits = torch.zeros(len(targets), num_classes)
    for row, (target, is_correct) in enumerate(zip(targets, flags, strict=True)):
        predicted = (
            target
            if is_correct
            else (wrong_class if wrong_class is not None else (target + 1) % num_classes)
        )
        logits[row, predicted] = confidence
    return logits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestComputeEvaluationMetrics:
    def test_perfect_predictions(self) -> None:
        targets = torch.tensor([0, 1, 2, 3] * 5)
        logits = make_logits(targets.tolist())
        metrics, confusion = compute_evaluation_metrics(logits, targets, split="test", loss=0.01)
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.top3_accuracy == pytest.approx(1.0)
        assert metrics.num_samples == 20
        assert confusion.diagonal().sum() == 20

    def test_reports_every_metric_in_the_headline(self) -> None:
        targets = torch.tensor([0, 0, 1, 1])
        logits = make_logits(targets.tolist(), correct=[True, False, True, True])
        metrics, _ = compute_evaluation_metrics(logits, targets, split="val", loss=0.5)
        headline = metrics.headline()
        for key in (
            "accuracy",
            "top3_accuracy",
            "top5_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "loss",
            "expected_calibration_error",
        ):
            assert key in headline
            assert isinstance(headline[key], float)

    def test_macro_f1_is_lower_than_accuracy_when_a_class_fails(self) -> None:
        """The pattern the audit found: high accuracy, one class at zero recall.

        Macro averaging over all 43 classes means a single failed class costs
        roughly 1/43 of macro recall but only its share of accuracy - which is
        exactly why accuracy hid the failure.
        """
        targets = torch.tensor([0] * 90 + [1] * 10)
        logits = make_logits(targets.tolist(), correct=[True] * 90 + [False] * 10, wrong_class=0)
        metrics, _ = compute_evaluation_metrics(logits, targets, split="test", loss=1.0)
        assert metrics.accuracy == pytest.approx(0.9)
        assert metrics.macro_recall < metrics.accuracy
        assert metrics.macro_recall < 0.1

    def test_sample_count_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="disagree on sample count"):
            compute_evaluation_metrics(
                torch.randn(4, 43), torch.tensor([0, 1]), split="test", loss=0.0
            )

    def test_top_k_equals_accuracy_at_the_label_space_size(self) -> None:
        targets = torch.tensor([0, 1])
        logits = make_logits(targets.tolist())
        metrics, _ = compute_evaluation_metrics(
            logits, targets, split="test", loss=0.0, num_classes=2
        )
        assert metrics.top3_accuracy == pytest.approx(metrics.accuracy)

    def test_top5_is_at_least_top3_is_at_least_accuracy(self) -> None:
        torch.manual_seed(0)
        targets = torch.randint(0, 43, (200,))
        logits = torch.randn(200, 43)
        metrics, _ = compute_evaluation_metrics(logits, targets, split="test", loss=1.0)
        assert metrics.accuracy <= metrics.top3_accuracy + 1e-9
        assert metrics.top3_accuracy <= metrics.top5_accuracy + 1e-9

    def test_summary_is_readable(self) -> None:
        targets = torch.tensor([0, 1])
        metrics, _ = compute_evaluation_metrics(
            make_logits(targets.tolist()), targets, split="test", loss=0.0
        )
        assert "test" in metrics.summary()
        assert "accuracy=" in metrics.summary()

    def test_as_dict_is_json_serialisable(self) -> None:
        targets = torch.tensor([0, 1, 2])
        metrics, _ = compute_evaluation_metrics(
            make_logits(targets.tolist()), targets, split="test", loss=0.0
        )
        assert json.loads(json.dumps(metrics.as_dict()))["split"] == "test"


class TestConfidenceStatistics:
    def test_confident_and_correct(self) -> None:
        targets = torch.tensor([0, 1])
        stats = confidence_statistics(make_logits(targets.tolist(), confidence=10.0), targets)
        assert stats.mean_confidence > 0.99
        assert stats.mean_confidence_incorrect == 0.0
        assert stats.num_samples == 2

    def test_confident_and_wrong_is_flagged_by_calibration(self) -> None:
        targets = torch.tensor([0, 1, 2, 3])
        logits = make_logits(targets.tolist(), correct=[False] * 4, confidence=10.0)
        stats = confidence_statistics(logits, targets)
        assert stats.mean_confidence_incorrect > 0.99
        assert stats.expected_calibration_error > 0.9

    def test_empty_input_is_handled(self) -> None:
        stats = confidence_statistics(torch.empty(0, 43), torch.empty(0, dtype=torch.long))
        assert stats.num_samples == 0


class TestCategoryMetrics:
    def test_all_families_present_are_reported(self) -> None:
        # 0 is a speed limit, 18 is a warning sign.
        predictions = np.array([0, 18])
        targets = np.array([0, 18])
        rows = category_metrics(predictions, targets)
        names = {row.category for row in rows}
        assert names == {"speed_limit", "warning"}

    def test_accuracy_within_a_family(self) -> None:
        predictions = np.array([0, 1, 2])
        targets = np.array([0, 1, 2])
        rows = {row.category: row for row in category_metrics(predictions, targets)}
        assert rows["speed_limit"].accuracy == pytest.approx(1.0)

    def test_empty_input_returns_no_rows(self) -> None:
        assert category_metrics(np.array([]), np.array([])) == []

    def test_category_confusion_is_square_and_conserves_counts(self) -> None:
        confusion = np.zeros((43, 43), dtype=np.int64)
        confusion[0, 1] = 5  # both speed limits: within family
        confusion[0, 18] = 3  # speed limit -> warning: across families
        collapsed, names = category_confusion(confusion)
        assert collapsed.shape == (len(names), len(names))
        assert collapsed.sum() == 8
        speed = names.index("speed_limit")
        warning = names.index("warning")
        assert collapsed[speed, speed] == 5
        assert collapsed[speed, warning] == 3

    def test_wrong_matrix_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Expected a 43x43"):
            category_confusion(np.zeros((10, 10), dtype=np.int64))


class TestBatchMetricsAccumulator:
    def test_weighted_mean_loss(self) -> None:
        accumulator = BatchMetricsAccumulator()
        accumulator.update(torch.randn(3, 43), torch.tensor([0, 1, 2]), loss=1.0)
        accumulator.update(torch.randn(1, 43), torch.tensor([0]), loss=5.0)
        assert accumulator.mean_loss == pytest.approx(2.0)
        assert accumulator.num_samples == 4

    def test_stacked_concatenates(self) -> None:
        accumulator = BatchMetricsAccumulator()
        accumulator.update(torch.randn(3, 43), torch.tensor([0, 1, 2]), loss=1.0)
        accumulator.update(torch.randn(2, 43), torch.tensor([3, 4]), loss=1.0)
        logits, targets = accumulator.stacked()
        assert logits.shape == (5, 43)
        assert targets.shape == (5,)

    def test_empty_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No batches"):
            BatchMetricsAccumulator().stacked()


class TestCompareEvaluations:
    def test_builds_a_percentage_table(self) -> None:
        targets = torch.tensor([0, 1, 2])
        logits = make_logits(targets.tolist())
        a, _ = compute_evaluation_metrics(logits, targets, split="test", loss=0.1)
        table = compare_evaluations({"CNN": a, "MLP": a})
        assert len(table) == 9
        assert table[0]["metric"] == "accuracy"
        assert table[0]["CNN"] == pytest.approx(100.0)
        assert table[0]["MLP"] == pytest.approx(100.0)

    def test_empty_input(self) -> None:
        assert compare_evaluations({}) == []


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def synthetic_confusion() -> np.ndarray:
    """A matrix with the shape of the recorded run's failures.

    Class 27 (Pedestrians) is heavily confused with 23 (Slippery road) and 30
    (Beware of ice/snow) — all warning triangles. Class 6 (End of speed limit 80) is
    confused with 5 (Speed limit 80) — within the speed-limit family. Class 12
    (Priority road) is confused with 15 (No vehicles) — across families.
    """
    confusion = np.zeros((43, 43), dtype=np.int64)
    for class_id in range(43):
        confusion[class_id, class_id] = 100
    confusion[27, 27] = 54
    confusion[27, 23] = 30
    confusion[27, 30] = 17
    confusion[6, 6] = 92
    confusion[6, 5] = 8
    confusion[12, 12] = 96
    confusion[12, 15] = 4
    return confusion


class TestTopConfusions:
    def test_ranks_by_share_of_true_class(self) -> None:
        pairs = top_confusions(synthetic_confusion())
        assert pairs[0].true_class_id == 27
        assert pairs[0].predicted_class_id == 23
        assert pairs[0].share_of_true_class == pytest.approx(30 / 101)
        assert pairs[0].count == 30

    def test_marks_whether_the_confusion_stays_in_family(self) -> None:
        pairs = {
            (p.true_class_id, p.predicted_class_id): p
            for p in top_confusions(synthetic_confusion())
        }
        assert pairs[(6, 5)].same_family is True  # both speed limits
        assert pairs[(27, 23)].same_family is True  # both warning signs
        assert pairs[(12, 15)].same_family is False  # priority vs prohibition

    def test_threshold_filters_trivial_confusions(self) -> None:
        confusion = synthetic_confusion()
        confusion[0, 1] = 1  # 1/101 ~ 1%, below the 0.5% default? it is above; use max_pairs
        pairs = top_confusions(confusion, threshold=0.05)
        assert all(pair.share_of_true_class >= 0.05 for pair in pairs)

    def test_max_pairs_is_respected(self) -> None:
        assert len(top_confusions(synthetic_confusion(), max_pairs=2)) == 2

    def test_wrong_matrix_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Expected a 43x43"):
            top_confusions(np.zeros((5, 5), dtype=np.int64))

    def test_pair_dict_includes_names_and_families(self) -> None:
        payload = top_confusions(synthetic_confusion())[0].as_dict()
        assert payload["true_class_name"] == "Pedestrians"
        assert payload["predicted_class_name"] == "Slippery road"
        assert payload["true_family"] == "warning"


class TestFamilyErrors:
    def test_counts_within_and_across(self) -> None:
        within, across = within_and_across_family_errors(synthetic_confusion())
        assert within == 30 + 17 + 8  # 27->23, 27->30, 6->5
        assert across == 4  # 12->15

    def test_shares_sum_to_one(self) -> None:
        within_share, across_share = family_split(synthetic_confusion())
        assert within_share + across_share == pytest.approx(1.0)

    def test_no_errors_gives_zero_shares(self) -> None:
        diagonal = np.zeros((43, 43), dtype=np.int64)
        np.fill_diagonal(diagonal, 5)
        assert family_split(diagonal) == (0.0, 0.0)

    def test_wrong_matrix_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Expected a 43x43"):
            within_and_across_family_errors(np.zeros((3, 3), dtype=np.int64))


def per_class_from_confusion(confusion: np.ndarray) -> list[dict[str, object]]:
    """Per-class rows consistent with ``confusion``.

    Derived directly from the matrix rather than via a fake prediction array, which
    keeps the two consistent by construction.
    """
    rows: list[dict[str, object]] = []
    for class_id in range(43):
        support = int(confusion[class_id].sum())
        correct = int(confusion[class_id, class_id])
        rows.append(
            {
                "class_id": class_id,
                "support": support,
                "precision": 0.0,
                "recall": correct / support if support else 0.0,
                "f1": 0.0,
            }
        )
    return rows


class TestWorstClasses:
    def test_ranks_by_recall_worst_first(self) -> None:
        confusion = synthetic_confusion()
        rows = worst_classes(per_class_from_confusion(confusion), confusion, split="test")
        assert rows[0].class_id == 27
        assert rows[0].recall == pytest.approx(54 / 101)

    def test_names_the_main_confusion(self) -> None:
        confusion = synthetic_confusion()
        rows = {
            row.class_id: row
            for row in worst_classes(per_class_from_confusion(confusion), confusion, split="test")
        }
        assert rows[27].top_confusion == 23
        assert rows[27].top_confusion_count == 30

    def test_max_recall_filter(self) -> None:
        confusion = synthetic_confusion()
        rows = worst_classes(
            per_class_from_confusion(confusion), confusion, split="test", max_recall=0.95
        )
        assert all(row.recall <= 0.95 for row in rows)

    def test_row_dict_has_names_and_error_count(self) -> None:
        confusion = synthetic_confusion()
        payload = worst_classes(per_class_from_confusion(confusion), confusion, split="test")[
            0
        ].as_dict()
        assert payload["class_name"] == "Pedestrians"
        assert payload["error_count"] == 101 - 54


class TestAnalyzeErrors:
    def _analysis(self, **kwargs: object) -> ErrorAnalysis:
        confusion = synthetic_confusion()
        per_class = per_class_from_confusion(confusion)
        defaults: dict[str, object] = {
            "split": "test",
            "accuracy": float(confusion.diagonal().sum() / confusion.sum()),
            "macro_f1": 0.98,
        }
        defaults.update(kwargs)
        return analyze_errors(confusion, per_class, **defaults)  # type: ignore[arg-type]

    def test_reports_error_totals(self) -> None:
        analysis = self._analysis()
        assert analysis.num_errors == 30 + 17 + 8 + 4
        assert len(analysis.worst_classes) > 0
        assert len(analysis.top_confusions) > 0

    def test_shares_are_fractions(self) -> None:
        analysis = self._analysis()
        assert (
            analysis.within_family_error_share + analysis.across_family_error_share
            == pytest.approx(1.0)
        )
        assert 0.0 <= analysis.rare_class_error_share <= 1.0

    def test_confident_errors_need_confidences(self) -> None:
        without = self._analysis()
        assert without.high_confidence_errors == 0
        with_conf = self._analysis(
            confidences=np.full(43, 0.99),
            correct=np.array([1.0] * 39 + [0.0] * 4),
        )
        assert with_conf.high_confidence_errors > 0
        assert with_conf.high_confidence_error_share > 0

    def test_markdown_report_contains_the_key_findings(self) -> None:
        report = self._analysis().to_markdown()
        assert "# Error analysis" in report
        assert "Worst classes by recall" in report
        assert "Most frequent confusions" in report
        assert "Pedestrians" in report
        assert "Slippery road" in report

    def test_as_dict_is_json_serialisable(self) -> None:
        assert json.loads(json.dumps(self._analysis().as_dict()))["split"] == "test"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _assert_png(path: Path) -> None:
    assert path.exists(), f"{path} was not written"
    assert path.stat().st_size > 1000, f"{path} looks empty"
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100


class TestFigures:
    def test_confusion_matrix(self, tmp_path: Path) -> None:
        _assert_png(plot_confusion_matrix(synthetic_confusion(), tmp_path / "cm.png"))

    def test_confusion_matrix_counts(self, tmp_path: Path) -> None:
        _assert_png(
            plot_confusion_matrix(
                synthetic_confusion(), tmp_path / "cm_counts.png", normalize=False
            )
        )

    def test_confusion_matrix_rejects_wrong_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Expected a 43x43"):
            plot_confusion_matrix(np.zeros((4, 4), dtype=np.int64), tmp_path / "bad.png")

    def test_category_confusion(self, tmp_path: Path) -> None:
        collapsed, names = category_confusion(synthetic_confusion())
        _assert_png(plot_category_confusion(collapsed, names, tmp_path / "cat.png"))

    def test_per_class_recall(self, tmp_path: Path) -> None:
        rows = per_class_metrics(np.arange(43), np.arange(43), 43)
        _assert_png(plot_per_class_recall(rows, tmp_path / "recall.png"))

    def test_top_confusions(self, tmp_path: Path) -> None:
        pairs = [pair.as_dict() for pair in top_confusions(synthetic_confusion())]
        _assert_png(plot_top_confusions(pairs, tmp_path / "conf.png"))

    def test_top_confusions_with_no_data(self, tmp_path: Path) -> None:
        _assert_png(plot_top_confusions([], tmp_path / "empty.png"))

    def test_class_distribution(self, tmp_path: Path) -> None:
        _assert_png(
            plot_class_distribution({c: 150 + c * 10 for c in range(43)}, tmp_path / "dist.png")
        )

    def test_training_curves(self, tmp_path: Path) -> None:
        history = [
            {
                "epoch": float(e),
                "train_loss": 2.0 / (e + 1),
                "val_loss": 2.2 / (e + 1),
                "train_accuracy": 0.5 + 0.02 * e,
                "val_accuracy": 0.48 + 0.02 * e,
                "val_macro_f1": 0.45 + 0.02 * e,
            }
            for e in range(1, 11)
        ]
        _assert_png(plot_training_curves(history, tmp_path / "curves.png"))

    def test_training_curves_rejects_empty_history(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty training history"):
            plot_training_curves([], tmp_path / "none.png")

    def test_confidence_analysis(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        confidences = rng.uniform(0.2, 1.0, 500)
        correct = (rng.uniform(size=500) < confidences).astype(float)
        _assert_png(plot_confidence_analysis(confidences, correct, tmp_path / "confidence.png"))


class TestPredictionGalleries:
    @pytest.fixture
    def batch(self) -> GalleryBatch:
        raw = torch.stack(
            [
                torch.from_numpy(np.asarray(make_sign_image(32, class_id=c), dtype=np.float32))
                .permute(2, 0, 1)
                .contiguous()
                / 255.0
                for c in range(10)
            ]
        )
        # to_display_array expects normalised input, so apply the real normalisation.
        mean = torch.tensor([0.3403, 0.3121, 0.3214]).view(3, 1, 1)
        std = torch.tensor([0.2724, 0.2608, 0.2669]).view(3, 1, 1)
        normalized = (raw - mean) / std
        targets = torch.arange(10)
        predictions = torch.arange(10)
        predictions[3] = 7
        predictions[5] = 6
        confidences = torch.linspace(0.99, 0.5, 10)
        return normalized, targets, predictions, confidences

    def test_correct_predictions(self, tmp_path: Path, batch: GalleryBatch) -> None:
        images, targets, predictions, confidences = batch
        _assert_png(
            plot_correct_predictions(
                images,
                targets,
                predictions,
                confidences,
                tmp_path / "correct.png",
                mean=(0.3403, 0.3121, 0.3214),
                std=(0.2724, 0.2608, 0.2669),
            )
        )

    def test_misclassified(self, tmp_path: Path, batch: GalleryBatch) -> None:
        images, targets, predictions, confidences = batch
        _assert_png(
            plot_misclassified(
                images,
                targets,
                predictions,
                confidences,
                tmp_path / "wrong.png",
                mean=(0.3403, 0.3121, 0.3214),
                std=(0.2724, 0.2608, 0.2669),
            )
        )

    def test_misclassified_with_no_errors_renders_a_placeholder(self, tmp_path: Path) -> None:
        images = torch.randn(3, 3, 32, 32)
        targets = torch.tensor([0, 1, 2])
        _assert_png(
            plot_misclassified(
                images,
                targets,
                targets.clone(),
                torch.ones(3),
                tmp_path / "none.png",
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
        )

    def test_misclassified_by_class(self, tmp_path: Path, batch: GalleryBatch) -> None:
        images, targets, predictions, confidences = batch
        _assert_png(
            plot_misclassified_by_class(
                images,
                targets,
                predictions,
                confidences,
                tmp_path / "byclass.png",
                mean=(0.3403, 0.3121, 0.3214),
                std=(0.2724, 0.2608, 0.2669),
            )
        )

    def test_per_class_gallery(self, tmp_path: Path, batch: GalleryBatch) -> None:
        images, targets, predictions, _ = batch
        _assert_png(
            plot_per_class_gallery(
                images,
                targets,
                predictions,
                tmp_path / "gallery.png",
                mean=(0.3403, 0.3121, 0.3214),
                std=(0.2724, 0.2608, 0.2669),
            )
        )

    def test_samples_per_class(self, tmp_path: Path, batch: GalleryBatch) -> None:
        images, targets, _, _ = batch
        _assert_png(
            plot_samples_per_class(
                images,
                targets,
                tmp_path / "samples.png",
                mean=(0.3403, 0.3121, 0.3214),
                std=(0.2724, 0.2608, 0.2669),
                columns=5,
                rows=2,
            )
        )


# ---------------------------------------------------------------------------
# Prediction collection and the runner
# ---------------------------------------------------------------------------
class SignDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: torch.Tensor, targets: torch.Tensor) -> None:
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.targets[index]


def make_loader(
    num_samples: int = 43, image_size: int = 32
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    total = max(num_samples, 43)
    targets = torch.arange(total) % 43
    images = torch.randn(total, 3, image_size, image_size)
    return DataLoader(SignDataset(images, targets), batch_size=8)


class TestCollectPredictions:
    def test_returns_logits_targets_and_images(self) -> None:
        model = build_model(
            ProjectConfig.model_validate({"model": {"name": "compact_cnn"}}).model, image_size=32
        )
        result = collect_predictions(
            model,
            make_loader(),
            device=torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
        )
        assert result.logits.shape[1] == 43
        assert result.images.shape[0] == result.targets.shape[0]
        assert result.predictions.shape[0] == result.targets.shape[0]
        assert result.confidences.min() >= 0.0
        assert result.confidences.max() <= 1.0

    def test_images_can_be_discarded(self) -> None:
        model = build_model(
            ProjectConfig.model_validate({"model": {"name": "compact_cnn"}}).model, image_size=32
        )
        result = collect_predictions(
            model,
            make_loader(),
            device=torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
            keep_images=False,
        )
        assert result.images.numel() == 0

    def test_leaves_the_model_in_eval_mode(self) -> None:
        model = build_model(
            ProjectConfig.model_validate({"model": {"name": "compact_cnn"}}).model, image_size=32
        )
        model.train()
        collect_predictions(
            model, make_loader(), device=torch.device("cpu"), criterion=torch.nn.CrossEntropyLoss()
        )
        assert model.training is False


class TestEvaluateCheckpoint:
    @pytest.fixture
    def trained_checkpoint(self, tmp_path: Path) -> TrainedCheckpoint:
        config = ProjectConfig.model_validate(
            {
                "output_dir": str(tmp_path / "artifacts"),
                "data": {"image_size": 32},
                "model": {"name": "compact_cnn"},
                "inference": {"device": "cpu", "checkpoint_path": str(tmp_path / "best.pt")},
            }
        )
        model = build_model(config.model, image_size=32)
        save_checkpoint(
            Path(config.inference.checkpoint_path), model, build_metadata(config, epoch=3)
        )
        return config, Path(config.inference.checkpoint_path)

    @pytest.fixture(autouse=True)
    def _patch_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace the real data pipeline with a synthetic loader.

        Evaluation has to run without the GTSRB download, and a test that needs a
        300 MB dataset is a test nobody runs.
        """
        loader = make_loader(129)
        dataset = loader.dataset
        assert isinstance(dataset, SignDataset)
        subset = TransformSubset(
            dataset,  # type: ignore[arg-type]
            list(range(len(dataset))),
            # Never invoked: the loader below is built from the raw dataset, and the
            # subset only exists so evaluate_checkpoint can report a dataset size.
            ToTensor(),
        )

        def fake_select_loader(
            _config: ProjectConfig, split: str
        ) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], TransformSubset]:
            if split not in ("train", "val", "test"):
                raise ValueError(f"Unknown split {split!r}; expected train, val or test")
            return loader, subset

        monkeypatch.setattr("gtsrb.evaluation.evaluate.select_loader", fake_select_loader)

    def test_produces_metrics_and_confusion(self, trained_checkpoint: TrainedCheckpoint) -> None:
        config, checkpoint = trained_checkpoint
        result = evaluate_checkpoint(config, checkpoint, split="test", render=False)
        assert result.metrics.num_samples == 129
        assert 0.0 <= result.metrics.accuracy <= 1.0
        assert result.confusion.shape == (43, 43)
        assert result.confusion.sum() == 129

    def test_writes_the_expected_artifacts(self, trained_checkpoint: TrainedCheckpoint) -> None:
        config, checkpoint = trained_checkpoint
        result = evaluate_checkpoint(config, checkpoint, split="test", render=False)
        names = {path.name for path in result.artifacts}
        assert {
            "metrics.json",
            "per_class.csv",
            "confusion_matrix.npy",
            "error_analysis.json",
            "error_analysis.md",
            "predictions.csv",
            "confusion_matrix.csv",
            "per_category.csv",
            "top_confusions.csv",
        } <= names
        for path in result.artifacts:
            assert path.exists()
            assert path.stat().st_size > 0

    def test_metrics_json_is_well_formed(self, trained_checkpoint: TrainedCheckpoint) -> None:
        config, checkpoint = trained_checkpoint
        result = evaluate_checkpoint(config, checkpoint, split="test", render=False)
        payload = json.loads((Path(result.artifacts[0]).parent / "metrics.json").read_text())
        assert payload["dataset"]["split"] == "test"
        assert payload["dataset"]["num_samples"] == 129
        assert "macro_f1" in payload["metrics"]
        assert len(payload["metrics"]["per_class"]) == 43

    def test_renders_figures_when_asked(self, trained_checkpoint: TrainedCheckpoint) -> None:
        config, checkpoint = trained_checkpoint
        result = evaluate_checkpoint(config, checkpoint, split="test", render=True)
        figures = [path for path in result.artifacts if path.parent.name == "figures"]
        assert figures, "no figures were written"
        expected = {
            "confusion_matrix.png",
            "category_confusion.png",
            "per_class_recall.png",
            "top_confusions.png",
            "confidence_analysis.png",
            "correct_predictions.png",
            "misclassified.png",
            "misclassified_by_class.png",
            "per_class_gallery.png",
        }
        assert expected <= {path.name for path in figures}

    def test_output_directory_defaults_to_the_model_name(
        self, trained_checkpoint: TrainedCheckpoint
    ) -> None:
        config, checkpoint = trained_checkpoint
        result = evaluate_checkpoint(config, checkpoint, split="test", render=False)
        assert "evaluation" in str(result.artifacts[0])
        assert "compact_cnn" in str(result.artifacts[0])

    def test_result_is_json_serialisable(self, trained_checkpoint: TrainedCheckpoint) -> None:
        config, checkpoint = trained_checkpoint
        payload = evaluate_checkpoint(config, checkpoint, split="test", render=False).as_dict()
        assert json.loads(json.dumps(payload, default=str))["split"] == "test"

    def test_summary_mentions_the_checkpoint_and_split(
        self, trained_checkpoint: TrainedCheckpoint
    ) -> None:
        config, checkpoint = trained_checkpoint
        summary = evaluate_checkpoint(config, checkpoint, split="test", render=False).summary()
        assert "test" in summary
        assert "accuracy=" in summary

    def test_predictions_csv_round_trips(self, trained_checkpoint: TrainedCheckpoint) -> None:
        config, checkpoint = trained_checkpoint
        result = evaluate_checkpoint(config, checkpoint, split="test", render=False)
        predictions_path = next(path for path in result.artifacts if path.name == "predictions.csv")
        rows = load_predictions_csv(predictions_path)
        assert len(rows) == 129
        rebuilt = confusion_from_predictions_csv(rows)
        assert np.array_equal(rebuilt, result.confusion)

    def test_missing_predictions_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Predictions file not found"):
            load_predictions_csv(tmp_path / "absent.csv")

    def test_invalid_split_is_rejected(self, trained_checkpoint: TrainedCheckpoint) -> None:
        config, checkpoint = trained_checkpoint
        with pytest.raises(ValueError, match="Unknown split"):
            evaluate_checkpoint(config, checkpoint, split="holdout", render=False)

    def test_missing_checkpoint_is_reported(self, trained_checkpoint: TrainedCheckpoint) -> None:
        config, _ = trained_checkpoint
        with pytest.raises(CheckpointError, match="Checkpoint not found"):
            evaluate_checkpoint(config, Path("/nonexistent/best.pt"), split="test", render=False)
