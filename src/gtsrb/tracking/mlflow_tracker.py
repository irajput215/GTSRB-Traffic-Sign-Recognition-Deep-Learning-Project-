"""MLflow-backed experiment tracking.

Design points that matter more than the API calls:

* **Local by default.** ``tracking.tracking_uri`` defaults to
  ``sqlite:///mlflow.db``, so ``--set tracking.enabled=true`` works with no server,
  no network and no account. SQLite rather than the legacy ``file:`` store because
  MLflow 3.x put the filesystem backend into maintenance mode and raises on it
  unless ``MLFLOW_ALLOW_FILE_STORE=true`` is set. Pointing it at an HTTP URI is a
  one-line change.
* **The run is opened when the tracker is built**, not on the first metric, so
  parameters and the resolved config are recorded even if training later fails.
  A failed run is more informative than a missing one.
* **The model is logged with a signature.** The registry entry then documents the
  input tensor shape and output shape, which is what makes an MLflow model
  self-describing rather than an opaque artifact.
* **Registration is opt-in.** ``tracking.register_model`` promotes the run's model
  to the model registry under ``tracking.registered_model_name``. Off by default,
  because a registry entry is a statement that a model is worth promoting.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import torch
from mlflow.models import infer_signature

from gtsrb.config.schema import ProjectConfig
from gtsrb.runtime import get_logger

logger = get_logger("tracking.mlflow")


class MlflowTracker:
    """Records a training run in MLflow.

    Args:
        config: full project configuration; ``config.tracking`` drives the
            behaviour and the rest is logged as parameters.
        artifact_root: where the resolved configuration is written before being
            logged as an artifact. Defaults to the checkpoint directory's parent.
    """

    def __init__(self, config: ProjectConfig, *, artifact_root: Path | None = None) -> None:
        self.config = config
        self.tracking = config.tracking
        self._finished = False

        mlflow.set_tracking_uri(self.tracking.tracking_uri)
        mlflow.set_experiment(self.tracking.experiment_name)

        tags: dict[str, str] = {"model": config.model.name, **self.tracking.tags}
        run = mlflow.start_run(
            run_name=self.tracking.run_name or config.run_name,
            tags=tags,
            description="GTSRB traffic-sign classification",
        )
        self.run_id: str = run.info.run_id

        self._artifact_root = artifact_root or Path(config.output_dir) / "runs" / self.run_id
        self._artifact_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "started MLflow run",
            extra={
                "run_id": self.run_id,
                "experiment": self.tracking.experiment_name,
                "tracking_uri": self.tracking.tracking_uri,
            },
        )

    # -- Tracker protocol --------------------------------------------------
    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record parameters, flattening nested values to dotted keys.

        MLflow parameter values must be scalars and are capped at 500 characters,
        so long values are truncated rather than raising.
        """
        flat: dict[str, str] = {}
        for key, value in params.items():
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            flat[key] = text[:500]
        mlflow.log_params(flat)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record metrics for a step.

        Non-finite values are skipped: MLflow rejects NaN and infinity, and a
        divergence that produced them is already reported by the trainer.
        """
        finite = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and torch.isfinite(torch.tensor(float(value)))
        }
        if finite:
            mlflow.log_metrics(finite, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Record a file or directory."""
        path = Path(path)
        if not path.exists():
            logger.warning("artifact does not exist, skipping", extra={"path": str(path)})
            return
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, payload: Mapping[str, Any], filename: str) -> None:
        """Record a JSON-serialisable mapping as an artifact."""
        if not self.tracking.log_artifacts:
            return
        target = self._artifact_root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(target))

    def log_model(self, model: Any, metadata: Any) -> None:
        """Record the trained model, registering it when configured to.

        The signature is inferred from a single example input so the registry entry
        carries the input and output contract.

        Serialisation uses the ``pickle`` format rather than MLflow 3.x's default
        ``pt2`` traced graph, and that is a deliberate, measured choice. The traced
        graph is exported against a fixed input example, so it accepts exactly the
        batch size it was traced with: a model logged from a batch of 1 rejects a
        batch of 2 with ``Guard failed: x.size()[0] == 1``, and calling ``.eval()``
        on the result raises ``NotImplementedError`` because a traced graph has no
        training state left to switch. A registry artifact that only serves batch
        size 1 is not a useful artifact.

        ``pickle`` stores the live ``nn.Module``, so the registered model accepts
        any batch size and behaves like the model that was trained. The cost is
        losing the traced graph's inference speed-up, which this project does not
        depend on: the inference service loads the project's own checkpoint rather
        than an MLflow artifact, precisely because a checkpoint also carries the
        class labels, input geometry and normalisation statistics.
        """
        if not self.tracking.log_model:
            return
        if not isinstance(model, torch.nn.Module):  # pragma: no cover - guards a caller bug
            raise TypeError(f"log_model expects an nn.Module, got {type(model).__name__}")

        image_size = int(self.config.data.image_size)
        was_training = model.training
        model.eval()
        try:
            example = torch.zeros(1, 3, image_size, image_size)
            with torch.no_grad():
                output = model(example)
            signature = infer_signature(example.numpy(), output.numpy())
        finally:
            # Leave the caller's model in the state they had it.
            model.train(was_training)

        registered_name = (
            self.tracking.registered_model_name if self.tracking.register_model else None
        )
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            signature=signature,
            input_example=example.numpy(),
            registered_model_name=registered_name,
            serialization_format="pickle",
            metadata={
                "model_name": getattr(metadata, "model_name", self.config.model.name),
                "num_classes": str(self.config.model.num_classes),
                "image_size": str(image_size),
                "class_names": json.dumps(list(getattr(metadata, "class_names", []))),
            },
        )

        logger.info(
            "logged model to MLflow",
            extra={
                "registered_as": registered_name,
                "run_id": self.run_id,
                "serialization_format": "pickle",
            },
        )

    def finish(self) -> None:
        """End the run. Safe to call more than once."""
        if self._finished:
            return
        mlflow.end_run()
        self._finished = True
        logger.info("finished MLflow run", extra={"run_id": self.run_id})

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> MlflowTracker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finish()


__all__ = ["MlflowTracker"]
