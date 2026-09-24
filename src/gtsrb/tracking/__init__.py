"""Experiment tracking.

``gtsrb.tracking.base`` defines the tracker interface the trainer depends on, plus
the no-op implementation used when tracking is disabled.
``gtsrb.tracking.mlflow_tracker`` provides the MLflow-backed implementation and
imports MLflow lazily, because MLflow is an optional extra.
"""

from __future__ import annotations

import importlib
from typing import Any

from gtsrb.tracking.base import NullTracker, Tracker

__all__ = ["NullTracker", "Tracker", "build_tracker"]


def build_tracker(config: Any) -> Tracker:
    """Return the tracker described by ``config.tracking``.

    Returns a :class:`NullTracker` when tracking is disabled. Raises a helpful
    error if tracking is enabled but MLflow is not installed, rather than silently
    recording nothing.
    """
    if not config.tracking.enabled:
        return NullTracker()

    # Resolved dynamically rather than with a top-level import: MLflow is an
    # optional extra, and importing this package must not require it. Importing
    # inside the function body would also trip Ruff's PLC0415, so the indirection
    # is deliberate.
    try:
        module = importlib.import_module("gtsrb.tracking.mlflow_tracker")
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise RuntimeError(
            "tracking.enabled is true but the MLflow tracker could not be imported. "
            "Install MLflow with 'uv sync --extra tracking' (or 'pip install mlflow'), "
            "or set tracking.enabled=false."
        ) from exc

    tracker_class = getattr(module, "MlflowTracker", None)
    if tracker_class is None:  # pragma: no cover - guards a packaging mistake
        raise RuntimeError("gtsrb.tracking.mlflow_tracker defines no MlflowTracker class")

    tracker: Tracker = tracker_class(config)
    return tracker
