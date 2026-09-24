"""Data pipeline: access, splitting, preprocessing, augmentation and validation.

Flow:

    raw GTSRB  ->  validate  ->  stratified split (persisted manifest)
               ->  per-split transforms  ->  class-balanced sampling
               ->  DataLoader

``build_dataloaders`` is the entry point; everything else here is a piece of it
that is also useful on its own (splitting for an audit, statistics for a report,
transforms for the inference path).
"""

from __future__ import annotations

from gtsrb.data.augmentation import build_augmentation, build_train_transform
from gtsrb.data.dataset import (
    GTSRBSplitName,
    LabelledDataset,
    TransformSubset,
    load_raw_split,
    read_labels,
)
from gtsrb.data.loaders import ClassWeights, DataBundle, build_dataloaders, compute_class_weights
from gtsrb.data.preprocessing import (
    build_eval_transform,
    build_transform,
    channel_statistics,
    denormalize,
    to_display_array,
)
from gtsrb.data.splits import (
    SplitError,
    SplitManifest,
    build_manifest,
    compute_split,
    get_or_create_manifest,
    verify_manifest,
)
from gtsrb.data.validation import (
    DatasetStatistics,
    DatasetValidationError,
    compare_statistics,
    estimate_channel_statistics,
    label_distribution,
    validate_dataset,
)

__all__ = [
    "ClassWeights",
    "DataBundle",
    "DatasetStatistics",
    "DatasetValidationError",
    "GTSRBSplitName",
    "LabelledDataset",
    "SplitError",
    "SplitManifest",
    "TransformSubset",
    "build_augmentation",
    "build_dataloaders",
    "build_eval_transform",
    "build_manifest",
    "build_train_transform",
    "build_transform",
    "channel_statistics",
    "compare_statistics",
    "compute_class_weights",
    "compute_split",
    "denormalize",
    "estimate_channel_statistics",
    "get_or_create_manifest",
    "label_distribution",
    "load_raw_split",
    "read_labels",
    "to_display_array",
    "validate_dataset",
    "verify_manifest",
]
