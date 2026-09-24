"""Integration tests for MLflow experiment tracking.

These run against a real MLflow **local file store** in a temporary directory, so
they exercise the actual MLflow code path without a server or network access. They
are marked ``integration`` because each run touches the filesystem and MLflow's
model serialisation is not instant.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from gtsrb.config.schema import ProjectConfig
from gtsrb.models import build_metadata, build_model
from gtsrb.tracking import NullTracker, build_tracker
from gtsrb.tracking.mlflow_tracker import MlflowTracker

pytestmark = [pytest.mark.integration, pytest.mark.slow]

mlflow = pytest.importorskip("mlflow", reason="MLflow is an optional extra")


@pytest.fixture
def tracking_config(tmp_path: Path) -> ProjectConfig:
    """A configuration whose tracking URI is a throwaway local SQLite database."""
    return ProjectConfig.model_validate(
        {
            "output_dir": str(tmp_path / "artifacts"),
            "data": {"image_size": 32},
            "model": {"name": "compact_cnn"},
            "tracking": {
                "enabled": True,
                "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                "experiment_name": "pytest-gtsrb",
                "run_name": "unit-run",
                "log_model": True,
                "register_model": False,
            },
        }
    )


@pytest.fixture(autouse=True)
def _isolate_mlflow_runs() -> Iterator[None]:
    """Make sure a failed test cannot leave an MLflow run open for the next one."""
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()


def start_tracker(config: ProjectConfig) -> MlflowTracker:
    """Start a concrete MLflow tracker.

    ``build_tracker`` returns the ``Tracker`` protocol, which deliberately does not
    expose ``run_id`` (a no-op tracker has no run). Tests that need the run id
    construct the concrete type, which is also what lets mypy check the calls.
    """
    return MlflowTracker(config)


class TestBuildTracker:
    def test_disabled_tracking_returns_a_null_tracker(self) -> None:
        assert isinstance(build_tracker(ProjectConfig()), NullTracker)

    def test_enabled_tracking_returns_the_mlflow_tracker(
        self, tracking_config: ProjectConfig
    ) -> None:
        tracker = start_tracker(tracking_config)
        try:
            assert isinstance(tracker, MlflowTracker)
            assert tracker.run_id
        finally:
            tracker.finish()


class TestMlflowTrackerLifecycle:
    def test_starting_a_run_creates_an_active_run(self, tracking_config: ProjectConfig) -> None:
        tracker = start_tracker(tracking_config)
        try:
            assert mlflow.active_run() is not None
            assert tracker.run_id == mlflow.active_run().info.run_id
        finally:
            tracker.finish()

    def test_finish_is_idempotent(self, tracking_config: ProjectConfig) -> None:
        tracker = start_tracker(tracking_config)
        tracker.finish()
        tracker.finish()
        assert mlflow.active_run() is None

    def test_context_manager_closes_the_run(self, tracking_config: ProjectConfig) -> None:
        with MlflowTracker(tracking_config):
            assert mlflow.active_run() is not None
        assert mlflow.active_run() is None

    def test_run_name_and_tags_are_recorded(self, tracking_config: ProjectConfig) -> None:
        tracker = start_tracker(tracking_config)
        try:
            run = mlflow.get_run(tracker.run_id)
            assert run.data.tags["mlflow.runName"] == "unit-run"
            assert run.data.tags["model"] == "compact_cnn"
        finally:
            tracker.finish()


class TestMlflowTrackerPayloads:
    def test_params_are_recorded(self, tracking_config: ProjectConfig) -> None:
        tracker = start_tracker(tracking_config)
        try:
            tracker.log_params(
                {"training.epochs": 30, "model.name": "compact_cnn", "nested": [1, 2]}
            )
            params = mlflow.get_run(tracker.run_id).data.params
            assert params["training.epochs"] == "30"
            assert params["model.name"] == "compact_cnn"
            assert params["nested"] == "[1, 2]"
        finally:
            tracker.finish()

    def test_long_params_are_truncated_rather_than_raising(
        self, tracking_config: ProjectConfig
    ) -> None:
        tracker = start_tracker(tracking_config)
        try:
            tracker.log_params({"long": "x" * 2000})
            assert len(mlflow.get_run(tracker.run_id).data.params["long"]) == 500
        finally:
            tracker.finish()

    def test_metrics_are_recorded_with_steps(self, tracking_config: ProjectConfig) -> None:
        tracker = start_tracker(tracking_config)
        try:
            for epoch in range(3):
                tracker.log_metrics({"val_accuracy": 0.5 + epoch * 0.1}, step=epoch)
            history = mlflow.get_run(tracker.run_id).data.metrics
            assert history["val_accuracy"] == pytest.approx(0.7)
        finally:
            tracker.finish()

    def test_non_finite_metrics_are_skipped(self, tracking_config: ProjectConfig) -> None:
        """MLflow rejects NaN and infinity; a divergence must not crash tracking."""
        tracker = start_tracker(tracking_config)
        try:
            tracker.log_metrics({"bad": float("nan"), "worse": float("inf"), "good": 1.0})
            metrics = mlflow.get_run(tracker.run_id).data.metrics
            assert "good" in metrics
            assert "bad" not in metrics
            assert "worse" not in metrics
        finally:
            tracker.finish()

    def test_dict_is_written_and_logged_as_an_artifact(
        self, tracking_config: ProjectConfig
    ) -> None:
        tracker = start_tracker(tracking_config)
        try:
            tracker.log_dict({"model": "compact_cnn", "epochs": 1}, "resolved_config.json")
            artifacts = mlflow.artifacts.list_artifacts(run_id=tracker.run_id)
            assert any(a.path == "resolved_config.json" for a in artifacts)
        finally:
            tracker.finish()

    def test_missing_artifact_is_skipped_without_raising(
        self, tracking_config: ProjectConfig, tmp_path: Path
    ) -> None:
        tracker = start_tracker(tracking_config)
        try:
            tracker.log_artifact(tmp_path / "absent.json")
        finally:
            tracker.finish()


class TestMlflowModelLogging:
    def test_model_is_logged_with_a_signature(self, tracking_config: ProjectConfig) -> None:
        tracker = start_tracker(tracking_config)
        try:
            model = build_model(tracking_config.model, image_size=32)
            metadata = build_metadata(tracking_config, metrics={"val_macro_f1": 0.9}, epoch=1)
            tracker.log_model(model, metadata)
            model_dir = mlflow.artifacts.list_artifacts(
                run_id=tracker.run_id, artifact_path="model"
            )
            assert any(a.path.endswith("MLmodel") for a in model_dir)
        finally:
            tracker.finish()

    def test_logged_model_leaves_the_caller_model_mode_unchanged(
        self, tracking_config: ProjectConfig
    ) -> None:
        """log_model switches to eval mode internally; it must switch back."""
        tracker = start_tracker(tracking_config)
        try:
            model = build_model(tracking_config.model, image_size=32)
            model.train()
            tracker.log_model(model, build_metadata(tracking_config))
            assert model.training is True
        finally:
            tracker.finish()

    def test_model_logging_can_be_disabled(self, tracking_config: ProjectConfig) -> None:
        config = tracking_config.model_copy(
            update={"tracking": tracking_config.tracking.model_copy(update={"log_model": False})}
        )
        tracker = start_tracker(config)
        try:
            tracker.log_model(build_model(config.model, image_size=32), build_metadata(config))
            artifacts = mlflow.artifacts.list_artifacts(run_id=tracker.run_id)
            assert not any(a.path == "model" for a in artifacts)
        finally:
            tracker.finish()

    def test_register_model_creates_a_registry_entry(self, tracking_config: ProjectConfig) -> None:
        """The full lifecycle: experiment -> best model -> registered model."""
        config = tracking_config.model_copy(
            update={
                "tracking": tracking_config.tracking.model_copy(
                    update={
                        "register_model": True,
                        "registered_model_name": "pytest-gtsrb-classifier",
                    }
                )
            }
        )
        tracker = start_tracker(config)
        try:
            tracker.log_model(build_model(config.model, image_size=32), build_metadata(config))
            client = mlflow.MlflowClient()
            versions = client.search_model_versions("name='pytest-gtsrb-classifier'")
            assert len(versions) >= 1
            assert versions[0].name == "pytest-gtsrb-classifier"
        finally:
            tracker.finish()

    def test_log_model_rejects_a_non_module(self, tracking_config: ProjectConfig) -> None:
        tracker = start_tracker(tracking_config)
        try:
            with pytest.raises(TypeError, match=r"expects an nn\.Module"):
                tracker.log_model(object(), build_metadata(tracking_config))
        finally:
            tracker.finish()

    def test_logged_model_can_be_loaded_back_and_predicts(
        self, tracking_config: ProjectConfig
    ) -> None:
        """Round-trip: what the registry stores must actually serve predictions."""
        tracker = start_tracker(tracking_config)
        try:
            model = build_model(tracking_config.model, image_size=32)
            tracker.log_model(model, build_metadata(tracking_config))
            run_id = tracker.run_id
        finally:
            tracker.finish()

        model_uri = f"runs:/{run_id}/model"
        loaded = mlflow.pytorch.load_model(model_uri)
        # No .eval() call: the pt2 format stores a traced graph rather than a live
        # nn.Module, and the graph was traced from an eval-mode model, so dropout
        # and batch-norm are already in inference mode. Traced modules raise
        # NotImplementedError on .eval() precisely because there is nothing to
        # switch.
        with torch.no_grad():
            output = loaded(torch.zeros(1, 3, 32, 32))
        assert output.shape == (1, 43)

    def test_registry_model_loads_via_the_registered_uri(
        self, tracking_config: ProjectConfig
    ) -> None:
        """The registered-model path, which is what a deployment would consume."""
        config = tracking_config.model_copy(
            update={
                "tracking": tracking_config.tracking.model_copy(
                    update={
                        "register_model": True,
                        "registered_model_name": "pytest-gtsrb-servable",
                    }
                )
            }
        )
        tracker = start_tracker(config)
        try:
            tracker.log_model(build_model(config.model, image_size=32), build_metadata(config))
        finally:
            tracker.finish()

        client = mlflow.MlflowClient()
        versions = client.search_model_versions("name='pytest-gtsrb-servable'")
        assert versions
        loaded = mlflow.pytorch.load_model(f"models:/{versions[0].name}/{versions[0].version}")
        with torch.no_grad():
            assert loaded(torch.zeros(2, 3, 32, 32)).shape == (2, 43)
