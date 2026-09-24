"""GTSRB traffic sign recognition: data, models, training, evaluation and serving.

The package is layered so that each stage can be used, tested and reasoned about
on its own:

``gtsrb.config``
    Validated configuration objects and the canonical class-label definitions.
``gtsrb.data``
    Dataset access, splits, preprocessing, augmentation and integrity checks.
``gtsrb.models``
    Model architectures and the factory that builds them from configuration.
``gtsrb.training``
    Training loop, losses, running metrics and callbacks.
``gtsrb.tracking``
    Optional MLflow experiment tracking and model registration.
``gtsrb.evaluation``
    Classification metrics, confusion analysis, error analysis and plots.
``gtsrb.inference``
    Checkpoint-backed single and batch prediction.
``gtsrb.api``
    FastAPI service that exposes inference over HTTP.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
