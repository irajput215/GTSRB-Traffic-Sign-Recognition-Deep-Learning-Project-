"""Model architectures and the factory that builds them.

Three architectures, matching the original project's controlled comparison:

* :class:`~gtsrb.models.compact_cnn.CompactCNN` — from-scratch CNN, the default
* :class:`~gtsrb.models.transfer.ResNet50Classifier` — ImageNet transfer learning
* :class:`~gtsrb.models.mlp.MLPClassifier` — fully-connected baseline

``build_model`` is the only constructor the rest of the project uses.
"""

from __future__ import annotations

from gtsrb.models.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    CheckpointMetadata,
    LoadedCheckpoint,
    build_metadata,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from gtsrb.models.compact_cnn import CompactCNN
from gtsrb.models.factory import (
    MODEL_BUILDERS,
    ModelSummary,
    build_model,
    count_parameters,
    summarise_model,
)
from gtsrb.models.mlp import MLPClassifier
from gtsrb.models.transfer import RESNET50_BLOCKS, ResNet50Classifier

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "MODEL_BUILDERS",
    "RESNET50_BLOCKS",
    "CheckpointError",
    "CheckpointMetadata",
    "CompactCNN",
    "LoadedCheckpoint",
    "MLPClassifier",
    "ModelSummary",
    "ResNet50Classifier",
    "build_metadata",
    "build_model",
    "count_parameters",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "save_checkpoint",
    "summarise_model",
]
