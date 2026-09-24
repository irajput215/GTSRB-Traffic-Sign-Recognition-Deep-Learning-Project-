"""Unit tests for the training loop, metrics and callbacks."""

from __future__ import annotations

import importlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from gtsrb.config.schema import ProjectConfig
from gtsrb.models import build_metadata, load_checkpoint, save_checkpoint
from gtsrb.tracking import NullTracker, build_tracker
from gtsrb.training import (
    CheckpointSelector,
    EarlyStopping,
    EpochMetrics,
    MetricsAccumulator,
    Trainer,
    build_loss,
    build_optimizer,
    build_scheduler,
    confusion_counts,
    current_learning_rate,
    expected_calibration_error,
    per_class_metrics,
)
from gtsrb.training.callbacks import CheckpointDecision

pytestmark = pytest.mark.unit

#: (config, model, train_loader, val_loader)
TrainerSetup = tuple[
    ProjectConfig,
    torch.nn.Module,
    "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
    "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetricsAccumulator:
    def test_perfect_predictions(self) -> None:
        accumulator = MetricsAccumulator(num_classes=3)
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = torch.tensor([0, 1])
        accumulator.update(logits, targets, loss=0.1)
        metrics = accumulator.compute()
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.loss == pytest.approx(0.1)
        assert metrics.num_samples == 2
        # Class 2 has no samples, so its F1 is 0 and macro F1 is 2/3. This is the
        # correct behaviour: macro averaging over all 43 classes is exactly what
        # makes a class the model never predicts visible.
        assert metrics.macro_f1 == pytest.approx(2 / 3)

    def test_macro_f1_is_perfect_when_every_class_is_present(self) -> None:
        accumulator = MetricsAccumulator(num_classes=2)
        accumulator.update(torch.tensor([[10.0, 0.0], [0.0, 10.0]]), torch.tensor([0, 1]), loss=0.1)
        assert accumulator.compute().macro_f1 == pytest.approx(1.0)

    def test_loss_is_weighted_by_batch_size(self) -> None:
        accumulator = MetricsAccumulator(num_classes=2)
        accumulator.update(torch.randn(3, 2), torch.tensor([0, 1, 0]), loss=1.0)
        accumulator.update(torch.randn(1, 2), torch.tensor([0]), loss=5.0)
        assert accumulator.compute().loss == pytest.approx((3 * 1.0 + 1 * 5.0) / 4)

    def test_accuracy_reflects_errors(self) -> None:
        accumulator = MetricsAccumulator(num_classes=3)
        accumulator.update(
            torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
            torch.tensor([0, 2]),
            loss=0.5,
        )
        assert accumulator.compute().accuracy == pytest.approx(0.5)

    def test_macro_f1_is_lower_than_accuracy_when_one_class_fails(self) -> None:
        """The exact pattern the audit found: one class at zero recall."""
        accumulator = MetricsAccumulator(num_classes=3)
        # Class 0 and 1 predicted correctly; every class-2 sample predicted as 1.
        logits = torch.tensor(
            [[10.0, 0.0, 0.0]] * 5 + [[0.0, 10.0, 0.0]] * 5 + [[0.0, 10.0, 0.0]] * 5
        )
        targets = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
        accumulator.update(logits, targets, loss=1.0)
        metrics = accumulator.compute()
        assert metrics.accuracy == pytest.approx(10 / 15)
        assert metrics.macro_recall == pytest.approx((1.0 + 1.0 + 0.0) / 3)
        assert metrics.macro_f1 < metrics.accuracy

    def test_top3_accuracy_is_at_least_accuracy(self) -> None:
        accumulator = MetricsAccumulator(num_classes=5, top_k=3)
        logits = torch.randn(20, 5)
        targets = torch.randint(0, 5, (20,))
        accumulator.update(logits, targets, loss=1.0)
        metrics = accumulator.compute()
        assert metrics.top3_accuracy >= metrics.accuracy - 1e-9

    def test_top_k_is_capped_at_the_label_space(self) -> None:
        assert MetricsAccumulator(num_classes=2, top_k=5).top_k == 2

    def test_empty_accumulator_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No batches"):
            MetricsAccumulator(num_classes=3).compute()

    def test_batch_size_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Batch size mismatch"):
            MetricsAccumulator(num_classes=3).update(torch.randn(4, 3), torch.tensor([0, 1]), 1.0)

    def test_wrong_logit_rank_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="2D logits"):
            MetricsAccumulator(num_classes=3).update(
                torch.randn(4, 3, 1), torch.tensor([0] * 4), 1.0
            )

    def test_wrong_target_rank_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="1D targets"):
            MetricsAccumulator(num_classes=3).update(torch.randn(4, 3), torch.randn(4, 1), 1.0)

    def test_summary_is_readable(self) -> None:
        accumulator = MetricsAccumulator(num_classes=3)
        accumulator.update(torch.randn(2, 3), torch.tensor([0, 1]), 1.0)
        assert "acc=" in accumulator.compute().summary()

    def test_as_dict_returns_floats(self) -> None:
        accumulator = MetricsAccumulator(num_classes=3)
        accumulator.update(torch.randn(2, 3), torch.tensor([0, 1]), 1.0)
        assert all(isinstance(value, float) for value in accumulator.compute().as_dict().values())


class TestConfusionCounts:
    def test_counts_are_placed_on_the_right_cells(self) -> None:
        predictions = np.array([0, 0, 1, 2])
        targets = np.array([0, 1, 1, 2])
        matrix = confusion_counts(predictions, targets, num_classes=3)
        assert matrix[0, 0] == 1
        assert matrix[1, 0] == 1
        assert matrix[1, 1] == 1
        assert matrix[2, 2] == 1
        assert matrix.sum() == 4

    def test_out_of_range_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            confusion_counts(np.array([5]), np.array([0]), num_classes=3)
        with pytest.raises(ValueError, match="outside"):
            confusion_counts(np.array([0]), np.array([-1]), num_classes=3)

    def test_empty_input_yields_a_zero_matrix(self) -> None:
        matrix = confusion_counts(np.array([], dtype=np.int64), np.array([], dtype=np.int64), 3)
        assert matrix.shape == (3, 3)
        assert matrix.sum() == 0


class TestPerClassMetrics:
    def test_reports_every_class_including_absent_ones(self) -> None:
        rows = per_class_metrics(np.array([0, 0]), np.array([0, 0]), num_classes=3)
        assert len(rows) == 3
        assert rows[0]["recall"] == pytest.approx(1.0)
        assert rows[2]["support"] == 0
        assert rows[2]["recall"] == pytest.approx(0.0)

    def test_identifies_a_class_with_zero_recall(self) -> None:
        predictions = np.array([0, 0, 0])
        targets = np.array([0, 1, 2])
        rows = per_class_metrics(predictions, targets, num_classes=3)
        assert rows[0]["recall"] == pytest.approx(1.0)
        assert rows[1]["recall"] == pytest.approx(0.0)
        assert rows[2]["recall"] == pytest.approx(0.0)


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_is_zero(self) -> None:
        confidences = np.full(100, 0.5)
        correct = np.array([1.0] * 50 + [0.0] * 50)
        assert expected_calibration_error(confidences, correct, num_bins=10) == pytest.approx(0.0)

    def test_overconfidence_is_positive(self) -> None:
        confidences = np.full(100, 0.99)
        correct = np.zeros(100)
        assert expected_calibration_error(confidences, correct, num_bins=15) > 0.9

    def test_empty_input_is_zero(self) -> None:
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0

    def test_confidence_of_one_lands_in_the_last_bin(self) -> None:
        assert expected_calibration_error(
            np.array([1.0]), np.array([1.0]), num_bins=5
        ) == pytest.approx(0.0)


class TestEpochMetrics:
    def test_as_dict_and_summary(self) -> None:
        metrics = EpochMetrics(
            loss=0.5,
            accuracy=0.9,
            macro_precision=0.9,
            macro_recall=0.9,
            macro_f1=0.9,
            top3_accuracy=0.99,
            num_samples=100,
        )
        assert metrics.as_dict()["accuracy"] == 0.9
        assert "loss=0.5000" in metrics.summary()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class TestEarlyStopping:
    def test_first_epoch_is_always_an_improvement(self) -> None:
        assert (
            EarlyStopping(monitor="val_macro_f1", patience=3).update({"val_macro_f1": 0.5}, 0)
            is False
        )

    def test_stops_after_patience_epochs_without_improvement(self) -> None:
        stopping = EarlyStopping(monitor="val_macro_f1", mode="max", patience=2)
        stopping.update({"val_macro_f1": 0.9}, 0)
        assert stopping.update({"val_macro_f1": 0.8}, 1) is False
        assert stopping.update({"val_macro_f1": 0.8}, 2) is True

    def test_improvement_resets_patience(self) -> None:
        stopping = EarlyStopping(monitor="val_macro_f1", mode="max", patience=2)
        stopping.update({"val_macro_f1": 0.5}, 0)
        stopping.update({"val_macro_f1": 0.4}, 1)
        stopping.update({"val_macro_f1": 0.9}, 2)
        assert stopping.epochs_without_improvement == 0
        assert stopping.best == pytest.approx(0.9)

    def test_min_mode_for_loss(self) -> None:
        stopping = EarlyStopping(monitor="val_loss", mode="min", patience=1)
        stopping.update({"val_loss": 1.0}, 0)
        assert stopping.update({"val_loss": 1.1}, 1) is True
        assert stopping.update({"val_loss": 0.5}, 2) is False

    def test_min_delta_filters_noise(self) -> None:
        stopping = EarlyStopping(monitor="val_macro_f1", mode="max", patience=1, min_delta=0.01)
        stopping.update({"val_macro_f1": 0.90}, 0)
        assert stopping.update({"val_macro_f1": 0.905}, 1) is True

    def test_unknown_metric_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Cannot monitor"):
            EarlyStopping(monitor="f1_score")

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="sideways")

    def test_missing_metric_key_raises(self) -> None:
        with pytest.raises(KeyError):
            EarlyStopping(monitor="val_macro_f1").update({"val_accuracy": 0.9}, 0)


class TestCheckpointSelector:
    def test_selects_on_macro_f1_not_accuracy(self) -> None:
        """The deliberate deviation from the original run's choice of accuracy."""
        selector = CheckpointSelector(monitor="val_macro_f1", mode="max")
        assert selector.should_save({"val_accuracy": 0.99, "val_macro_f1": 0.50}, 0).saved
        # Higher accuracy, lower macro F1: must not be selected.
        decision = selector.should_save({"val_accuracy": 0.995, "val_macro_f1": 0.49}, 1)
        assert decision.saved is False
        assert decision.reason == "no improvement"

    def test_first_epoch_is_selected(self) -> None:
        decision = CheckpointSelector().should_save({"val_macro_f1": 0.1}, 0)
        assert decision.saved is True
        assert decision.metric == pytest.approx(0.1)

    def test_min_mode_for_loss(self) -> None:
        selector = CheckpointSelector(monitor="val_loss", mode="min")
        assert selector.should_save({"val_loss": 1.0}, 0).saved
        assert selector.should_save({"val_loss": 0.5}, 1).saved
        assert selector.should_save({"val_loss": 0.7}, 2).saved is False

    def test_best_epoch_is_tracked(self) -> None:
        selector = CheckpointSelector()
        selector.should_save({"val_macro_f1": 0.5}, 0)
        selector.should_save({"val_macro_f1": 0.9}, 3)
        selector.should_save({"val_macro_f1": 0.1}, 4)
        assert selector.best_epoch == 3
        assert selector.best == pytest.approx(0.9)

    def test_decision_is_a_dataclass(self) -> None:
        assert isinstance(
            CheckpointSelector().should_save({"val_macro_f1": 1.0}, 0), CheckpointDecision
        )


class TestOptimizerAndScheduler:
    def test_builds_each_optimizer(self, tiny_model: torch.nn.Module) -> None:
        for name in ("adam", "adamw", "sgd"):
            optimizer = build_optimizer(tiny_model, name=name, lr=1e-3, weight_decay=0.0)
            assert current_learning_rate(optimizer) == pytest.approx(1e-3)

    def test_unknown_optimizer_is_rejected(self, tiny_model: torch.nn.Module) -> None:
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            build_optimizer(tiny_model, name="rmsprop", lr=1e-3, weight_decay=0.0)

    def test_none_scheduler_returns_none(self, tiny_model: torch.nn.Module) -> None:
        optimizer = build_optimizer(tiny_model, name="adam", lr=1e-3, weight_decay=0.0)
        config = ProjectConfig.model_validate({"training": {"scheduler": {"name": "none"}}})
        assert build_scheduler(config.training.scheduler, optimizer, epochs=5) is None

    def test_plateau_scheduler_halves_the_rate(self, tiny_model: torch.nn.Module) -> None:
        optimizer = build_optimizer(tiny_model, name="adam", lr=1e-2, weight_decay=0.0)
        config = ProjectConfig.model_validate(
            {"training": {"scheduler": {"name": "reduce_on_plateau", "factor": 0.5, "patience": 0}}}
        )
        scheduler = build_scheduler(config.training.scheduler, optimizer, epochs=5)
        assert isinstance(scheduler, ReduceLROnPlateau)
        # The first step establishes the baseline; the plateau is detected on the
        # next, non-improving step.
        scheduler.step(1.0)
        scheduler.step(1.0)
        assert current_learning_rate(optimizer) == pytest.approx(5e-3)

    def test_cosine_scheduler_decays(self, tiny_model: torch.nn.Module) -> None:
        optimizer = build_optimizer(tiny_model, name="adam", lr=1e-2, weight_decay=0.0)
        config = ProjectConfig.model_validate({"training": {"scheduler": {"name": "cosine"}}})
        scheduler = build_scheduler(config.training.scheduler, optimizer, epochs=4)
        assert scheduler is not None
        for _ in range(4):
            optimizer.step()  # torch warns if the scheduler is stepped first
            scheduler.step()
        assert current_learning_rate(optimizer) < 1e-2

    def test_step_scheduler_decays_by_gamma(self, tiny_model: torch.nn.Module) -> None:
        optimizer = build_optimizer(tiny_model, name="adam", lr=1e-2, weight_decay=0.0)
        config = ProjectConfig.model_validate(
            {"training": {"scheduler": {"name": "step", "step_size": 1, "gamma": 0.1}}}
        )
        scheduler = build_scheduler(config.training.scheduler, optimizer, epochs=5)
        assert scheduler is not None
        optimizer.step()
        scheduler.step()
        assert current_learning_rate(optimizer) == pytest.approx(1e-3)

    def test_unsupported_scheduler_is_rejected(self, tiny_model: torch.nn.Module) -> None:
        optimizer = build_optimizer(tiny_model, name="adam", lr=1e-3, weight_decay=0.0)
        config = ProjectConfig.model_validate({"training": {"scheduler": {"name": "none"}}})
        broken = config.training.scheduler.model_copy(update={"name": "warmup"})
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            build_scheduler(broken, optimizer, epochs=5)


class TestLoss:
    def test_default_has_no_label_smoothing(self) -> None:
        config = ProjectConfig()
        assert build_loss(config.training).label_smoothing == pytest.approx(0.0)

    def test_label_smoothing_is_applied(self) -> None:
        config = ProjectConfig.model_validate({"training": {"label_smoothing": 0.1}})
        assert build_loss(config.training).label_smoothing == pytest.approx(0.1)

    def test_unsupported_loss_is_rejected(self) -> None:
        config = ProjectConfig()
        broken = config.training.model_copy(update={"loss": "focal"})
        with pytest.raises(ValueError, match="Unsupported loss"):
            build_loss(broken)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SyntheticSigns(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A typed in-memory dataset.

    ``TensorDataset`` is typed as yielding ``tuple[Tensor, ...]``, which does not
    describe "an image and its label" and therefore does not typecheck against the
    trainer. This does.
    """

    def __init__(self, images: torch.Tensor, targets: torch.Tensor) -> None:
        if images.shape[0] != targets.shape[0]:
            raise ValueError("images and targets must have the same length")
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.targets[index]


def make_dataset(
    num_samples: int, image_size: int, num_classes: int, seed: int = 0
) -> SyntheticSigns:
    """A linearly separable synthetic dataset, so a few epochs must reduce the loss.

    Each class gets a distinct constant offset added to its images, which even a
    tiny model can separate in a handful of steps. That makes it possible to assert
    that training *does* something rather than merely that it runs.
    """
    generator = torch.Generator().manual_seed(seed)
    targets = torch.arange(num_samples) % num_classes
    images = torch.randn(num_samples, 3, image_size, image_size, generator=generator) * 0.1
    for class_id in range(num_classes):
        images[targets == class_id] += (class_id + 1) * 0.5
    return SyntheticSigns(images, targets)


class FlattenClassifier(torch.nn.Module):
    """Minimal model for trainer tests: adaptive pool, flatten, linear."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.fc(self.pool(x).flatten(1))
        return logits


@pytest.fixture
def trainer_setup(tmp_path: Path) -> TrainerSetup:
    config = ProjectConfig.model_validate(
        {
            "data": {"image_size": 16},
            "training": {
                "epochs": 4,
                "batch_size": 8,
                "device": "cpu",
                "amp": False,
                "log_every_n_steps": 100,
                "seed": 0,
                "checkpoint": {
                    "directory": str(tmp_path / "checkpoints"),
                    "filename": "best.pt",
                },
                "early_stopping": {"enabled": False},
            },
        }
    )
    model = FlattenClassifier(num_classes=43)
    train = DataLoader(make_dataset(86, 16, 43), batch_size=8, shuffle=True)
    val = DataLoader(make_dataset(43, 16, 43, seed=1), batch_size=8, shuffle=False)
    return config, model, train, val


class TestTrainerRun:
    def test_fit_returns_a_complete_result(self, trainer_setup: TrainerSetup) -> None:
        config, model, train, val = trainer_setup
        trainer = Trainer(config, model, train, val)
        result = trainer.fit()
        assert result.epochs_trained == 4
        assert result.model_name == "compact_cnn"
        assert result.device == "cpu"
        assert len(result.history) == 4
        assert 0.0 <= result.best_metrics["val_accuracy"] <= 1.0
        assert result.checkpoint_path is not None
        assert Path(result.checkpoint_path).exists()
        assert result.duration_seconds > 0

    def test_history_contains_both_splits_and_the_learning_rate(
        self, trainer_setup: TrainerSetup
    ) -> None:
        config, model, train, val = trainer_setup
        history = Trainer(config, model, train, val).fit().history
        for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_macro_f1"):
            assert key in history[0]
        assert "learning_rate" in history[0]

    def test_training_reduces_the_loss(self, trainer_setup: TrainerSetup) -> None:
        config, model, train, val = trainer_setup
        result = Trainer(config, model, train, val).fit()
        assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]

    def test_best_checkpoint_is_weights_only(self, trainer_setup: TrainerSetup) -> None:
        """best.pt is the served artifact, so it must not carry optimiser state."""
        config, model, train, val = trainer_setup
        result = Trainer(config, model, train, val).fit()
        assert result.checkpoint_path is not None
        loaded = load_checkpoint(Path(result.checkpoint_path))
        assert loaded.metadata.model_name == "compact_cnn"
        assert loaded.metadata.num_classes == 43
        assert "val_macro_f1" in loaded.metadata.metrics
        assert loaded.optimizer_state is None
        assert loaded.extra["resumable"] is False

    def test_last_checkpoint_is_resumable(self, trainer_setup: TrainerSetup) -> None:
        """last.pt exists so an interrupted run can continue with full state."""
        config, model, train, val = trainer_setup
        Trainer(config, model, train, val).fit()
        last = load_checkpoint(Path(config.training.checkpoint.directory) / "last.pt")
        assert last.optimizer_state is not None
        assert last.extra["resumable"] is True

    def test_weights_only_checkpoint_omits_the_optimiser_payload(
        self, trainer_setup: TrainerSetup
    ) -> None:
        """Asserted on the payload keys, not on bytes: file-size overhead dominates
        for a tiny test model, but the invariant holds at any scale."""
        config, model, train, val = trainer_setup
        result = Trainer(config, model, train, val).fit()
        assert result.checkpoint_path is not None
        directory = Path(config.training.checkpoint.directory)
        best_payload = torch.load(result.checkpoint_path, weights_only=False)
        last_payload = torch.load(directory / "last.pt", weights_only=False)
        assert "optimizer_state" not in best_payload
        assert "optimizer_state" in last_payload

    def test_last_checkpoint_is_written_every_epoch(self, trainer_setup: TrainerSetup) -> None:
        """Even when every epoch improves, the run must stay resumable."""
        config, model, train, val = trainer_setup
        Trainer(config, model, train, val).fit()
        assert (Path(config.training.checkpoint.directory) / "last.pt").exists()

    def test_last_checkpoint_written_when_save_last_is_disabled_only_never(
        self, tmp_path: Path
    ) -> None:
        config = ProjectConfig.model_validate(
            {
                "data": {"image_size": 16},
                "training": {
                    "epochs": 1,
                    "batch_size": 8,
                    "device": "cpu",
                    "amp": False,
                    "log_every_n_steps": 100,
                    "early_stopping": {"enabled": False},
                    "checkpoint": {
                        "directory": str(tmp_path / "ckpt"),
                        "filename": "best.pt",
                        "save_last": False,
                    },
                },
            }
        )
        train = DataLoader(make_dataset(43, 16, 43), batch_size=8)
        val = DataLoader(make_dataset(43, 16, 43, seed=1), batch_size=8)
        Trainer(config, FlattenClassifier(43), train, val).fit()
        assert (tmp_path / "ckpt" / "best.pt").exists()
        assert not (tmp_path / "ckpt" / "last.pt").exists()

    def test_result_is_json_serialisable(self, trainer_setup: TrainerSetup) -> None:
        config, model, train, val = trainer_setup
        payload = Trainer(config, model, train, val).fit().as_dict()
        assert json.loads(json.dumps(payload, default=str))["epochs_trained"] == 4

    def test_summary_mentions_the_model_and_device(self, trainer_setup: TrainerSetup) -> None:
        config, model, train, val = trainer_setup
        assert "cpu" in Trainer(config, model, train, val).fit().summary()

    def test_parameter_summary_is_recorded(self, trainer_setup: TrainerSetup) -> None:
        config, model, train, val = trainer_setup
        summary = Trainer(config, model, train, val).fit().parameter_summary
        assert summary["total_parameters"] > 0
        assert summary["image_size"] == 16


class TestTrainerResume:
    def test_resume_continues_from_the_stored_epoch(self, tmp_path: Path) -> None:
        config = ProjectConfig.model_validate(
            {
                "data": {"image_size": 16},
                "training": {
                    "epochs": 2,
                    "batch_size": 8,
                    "device": "cpu",
                    "amp": False,
                    "log_every_n_steps": 100,
                    "checkpoint": {"directory": str(tmp_path / "ckpt"), "filename": "best.pt"},
                    "early_stopping": {"enabled": False},
                },
            }
        )
        train = DataLoader(make_dataset(86, 16, 43), batch_size=8)
        val = DataLoader(make_dataset(43, 16, 43, seed=1), batch_size=8)

        Trainer(config, FlattenClassifier(43), train, val).fit()
        last = Path(config.training.checkpoint.directory) / "last.pt"
        checkpoint_epoch = load_checkpoint(last).epoch

        longer = config.model_copy(deep=True).model_copy(
            update={"training": config.training.model_copy(update={"epochs": 6})}
        )
        resumed = Trainer(longer, FlattenClassifier(43), train, val).fit(resume_from=last)
        # Training resumes after the checkpoint's epoch rather than restarting.
        assert resumed.epochs_trained == 6 - checkpoint_epoch
        assert resumed.history[0]["epoch"] == pytest.approx(checkpoint_epoch + 1)

    def test_resume_restores_weights_into_the_model(self, tmp_path: Path) -> None:
        config = ProjectConfig.model_validate(
            {
                "data": {"image_size": 16},
                "training": {
                    "epochs": 1,
                    "batch_size": 8,
                    "device": "cpu",
                    "amp": False,
                    "log_every_n_steps": 100,
                    "checkpoint": {"directory": str(tmp_path / "ckpt"), "filename": "best.pt"},
                    "early_stopping": {"enabled": False},
                },
            }
        )
        train = DataLoader(make_dataset(86, 16, 43), batch_size=8)
        val = DataLoader(make_dataset(43, 16, 43, seed=1), batch_size=8)

        trained = FlattenClassifier(43)
        Trainer(config, trained, train, val).fit()
        last = Path(config.training.checkpoint.directory) / "last.pt"

        fresh = FlattenClassifier(43)
        Trainer(config, fresh, train, val).fit(resume_from=last)
        for original, restored in zip(trained.parameters(), fresh.parameters(), strict=True):
            assert torch.allclose(original, restored)


class TestTrainerEarlyStopping:
    def test_stops_before_the_epoch_limit(self, tmp_path: Path) -> None:
        config = ProjectConfig.model_validate(
            {
                "data": {"image_size": 16},
                "training": {
                    "epochs": 50,
                    "batch_size": 8,
                    "device": "cpu",
                    "amp": False,
                    "log_every_n_steps": 100,
                    "early_stopping": {
                        "enabled": True,
                        "monitor": "val_macro_f1",
                        "mode": "max",
                        "patience": 1,
                        "min_delta": 0.5,
                    },
                    "checkpoint": {"directory": str(tmp_path / "ckpt")},
                },
            }
        )
        train = DataLoader(make_dataset(86, 16, 43), batch_size=8)
        val = DataLoader(make_dataset(43, 16, 43, seed=1), batch_size=8)
        result = Trainer(config, FlattenClassifier(43), train, val).fit()
        assert result.stopped_early is True
        assert result.epochs_trained < 50


class TestTrainerFailures:
    def test_non_finite_loss_is_reported(self, tmp_path: Path) -> None:
        class Exploding(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # A parameter is required: Adam rejects an empty parameter list.
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                logits: torch.Tensor = torch.full((x.shape[0], 43), float("nan")) * self.scale
                return logits

        config = ProjectConfig.model_validate(
            {
                "data": {"image_size": 16},
                "training": {
                    "epochs": 1,
                    "batch_size": 8,
                    "device": "cpu",
                    "amp": False,
                    "log_every_n_steps": 100,
                },
            }
        )
        train = DataLoader(make_dataset(16, 16, 43), batch_size=8)
        val = DataLoader(make_dataset(16, 16, 43, seed=1), batch_size=8)
        with pytest.raises(RuntimeError, match="non-finite"):
            Trainer(config, Exploding(), train, val).fit()

    def test_amp_is_disabled_on_cpu(self, trainer_setup: TrainerSetup) -> None:
        config, model, train, val = trainer_setup
        config = config.model_copy(
            update={"training": config.training.model_copy(update={"amp": True})}
        )
        assert Trainer(config, model, train, val).use_amp is False


class TestTrackerSeam:
    def test_null_tracker_records_nothing_and_is_safe(self) -> None:
        tracker = NullTracker()
        tracker.log_params({"a": 1})
        tracker.log_metrics({"b": 2.0}, step=0)
        tracker.finish()
        tracker.finish()  # must be idempotent

    def test_trainer_calls_the_tracker(self, trainer_setup: TrainerSetup) -> None:
        config, model, train, val = trainer_setup
        calls: list[str] = []

        class SpyTracker:
            def log_params(self, params: object) -> None:
                calls.append("params")

            def log_metrics(self, metrics: object, step: int | None = None) -> None:
                calls.append("metrics")

            def log_artifact(self, path: object, artifact_path: str | None = None) -> None:
                calls.append("artifact")

            def log_dict(self, payload: object, filename: str) -> None:
                calls.append("dict")

            def log_model(self, model: object, metadata: object) -> None:
                calls.append("model")

            def finish(self) -> None:
                calls.append("finish")

        Trainer(config, model, train, val, tracker=SpyTracker()).fit()
        assert calls.count("metrics") == 4
        assert "params" not in calls  # the CLI logs params, not the trainer

    def test_build_tracker_returns_null_when_disabled(self) -> None:
        assert isinstance(build_tracker(ProjectConfig()), NullTracker)

    def test_build_tracker_errors_clearly_when_mlflow_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabled tracking with no MLflow installed must not silently no-op."""
        config = ProjectConfig.model_validate({"tracking": {"enabled": True}})
        real_import = importlib.import_module

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "gtsrb.tracking.mlflow_tracker":
                raise ImportError("simulated missing MLflow")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with pytest.raises(RuntimeError, match="could not be imported"):
            build_tracker(config)

    def test_build_tracker_errors_when_the_class_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = ProjectConfig.model_validate({"tracking": {"enabled": True}})
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda _name: types.ModuleType("gtsrb.tracking.mlflow_tracker"),
        )
        with pytest.raises(RuntimeError, match=r"defines no MlflowTracker"):
            build_tracker(config)


class TestCheckpointMetadataFromTrainer:
    def test_metadata_only_carries_validation_metrics(self, tmp_path: Path) -> None:
        config = ProjectConfig.model_validate(
            {
                "data": {"image_size": 16},
                "training": {
                    "epochs": 1,
                    "batch_size": 8,
                    "device": "cpu",
                    "amp": False,
                    "log_every_n_steps": 100,
                    "early_stopping": {"enabled": False},
                    "checkpoint": {"directory": str(tmp_path / "ckpt"), "filename": "best.pt"},
                },
            }
        )
        train = DataLoader(make_dataset(43, 16, 43), batch_size=8)
        val = DataLoader(make_dataset(43, 16, 43, seed=1), batch_size=8)
        result = Trainer(config, FlattenClassifier(43), train, val).fit()
        assert result.checkpoint_path is not None
        stored = load_checkpoint(Path(result.checkpoint_path)).metadata.metrics
        assert all(key.startswith("val_") for key in stored)

    def test_metadata_from_config_is_complete(self, trainer_setup: TrainerSetup) -> None:
        config, *_ = trainer_setup
        metadata = build_metadata(config, metrics={"val_macro_f1": 0.5}, epoch=2)
        assert metadata.config["data"]["image_size"] == 16


class TestSaveCheckpointHelper:
    def test_direct_helper_round_trip(self, tmp_path: Path, tiny_model: torch.nn.Module) -> None:
        config = ProjectConfig.model_validate({"data": {"image_size": 32}})
        path = tmp_path / "direct.pt"
        save_checkpoint(path, tiny_model, build_metadata(config, epoch=5))
        assert load_checkpoint(path).epoch == 5
