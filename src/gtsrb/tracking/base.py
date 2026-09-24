"""Experiment-tracking seam.

The training loop talks to a :class:`Tracker`, not to MLflow. That keeps one
concern in one place and, more importantly, means tracking is genuinely optional:
the default :class:`NullTracker` does nothing, so the project trains and evaluates
with no tracking server, no MLflow install and no network access.

MLflow is a large dependency. It is declared as an optional extra
(``uv sync --extra tracking``) and imported lazily inside
``gtsrb.tracking.mlflow_tracker`` so that a training-only environment never pays
for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    """Minimal experiment-tracking interface used by the trainer."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record run configuration. Called once, before training starts."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record metrics for a step (an epoch, here)."""
        ...

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Record a file produced by the run."""
        ...

    def log_dict(self, payload: Mapping[str, Any], filename: str) -> None:
        """Record a JSON-serialisable mapping as a named artifact."""
        ...

    def log_model(self, model: Any, metadata: Any) -> None:
        """Record the trained model artifact, and register it if configured."""
        ...

    def finish(self) -> None:
        """Close the run. Must be safe to call twice."""
        ...


class NullTracker:
    """A tracker that records nothing.

    Used when ``tracking.enabled`` is false. Implemented as a real object rather
    than ``None`` so the trainer has no conditional paths around tracking.
    """

    def log_params(self, params: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        return None

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        return None

    def log_dict(self, payload: Mapping[str, Any], filename: str) -> None:
        return None

    def log_model(self, model: Any, metadata: Any) -> None:
        return None

    def finish(self) -> None:
        return None


__all__ = ["NullTracker", "Tracker"]
