"""DataLoader construction.

One function builds every loader the project uses, so the training and evaluation
paths cannot end up with different batch sizes, worker counts or seeding. The
returned :class:`DataBundle` carries the split manifest and the fitted class
weights alongside the loaders, so a caller never has to recompute them (and
therefore cannot compute them differently).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from gtsrb.config.schema import ProjectConfig
from gtsrb.data.augmentation import build_train_transform
from gtsrb.data.dataset import TransformSubset, load_raw_split, read_labels
from gtsrb.data.preprocessing import build_transform
from gtsrb.data.splits import SplitManifest, get_or_create_manifest
from gtsrb.runtime import get_logger, make_generator, seed_worker

logger = get_logger("data.loaders")


@dataclass(frozen=True)
class ClassWeights:
    """Per-class sampling weights for the training split.

    ``power=0`` gives every sample weight 1.0. ``power=1`` makes the expected
    number of draws per class equal, which is the fully class-balanced case.
    """

    per_class: dict[int, float]
    power: float

    def as_tensor(self, labels: Sequence[int]) -> torch.Tensor:
        """Expand per-class weights into a per-sample weight tensor."""
        return torch.tensor([self.per_class[label] for label in labels], dtype=torch.double)

    def effective_counts(self, labels: Sequence[int], num_samples: int) -> dict[int, float]:
        """Expected number of draws per class if ``num_samples`` are sampled.

        Reported in the run log, which is how the class balance achieved at
        training time is verified rather than assumed.
        """
        weights = self.as_tensor(labels)
        total = float(weights.sum())
        if total == 0:
            return {}
        counts: dict[int, float] = {}
        for label, weight in zip(labels, weights.tolist(), strict=True):
            counts[label] = counts.get(label, 0.0) + num_samples * weight / total
        return dict(sorted(counts.items()))


def compute_class_weights(labels: Sequence[int], power: float) -> ClassWeights:
    """Compute inverse-frequency class weights raised to ``power``.

    Args:
        labels: labels of the training split.
        power: 0 is uniform sampling, 1 is fully class-balanced.

    Raises:
        ValueError: if ``labels`` is empty or ``power`` is outside ``[0, 1]``.
    """
    if not labels:
        raise ValueError("Cannot compute class weights from an empty label sequence")
    if not 0.0 <= power <= 1.0:
        raise ValueError(f"power must be in [0, 1], got {power}")

    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    per_class = {label: (1.0 / count) ** power for label, count in counts.items()}
    # Normalise so the largest weight is 1.0. Only the ratio matters to the
    # sampler, but a normalised table is readable in a log line or a report.
    largest = max(per_class.values())
    if largest > 0:
        per_class = {label: weight / largest for label, weight in per_class.items()}
    return ClassWeights(per_class=per_class, power=power)


@dataclass(frozen=True)
class DataBundle:
    """Loaders plus everything needed to describe how they were built."""

    train: DataLoader[tuple[torch.Tensor, int]]
    val: DataLoader[tuple[torch.Tensor, int]]
    test: DataLoader[tuple[torch.Tensor, int]]
    manifest: SplitManifest
    class_weights: ClassWeights | None
    train_dataset: TransformSubset
    val_dataset: TransformSubset
    test_dataset: TransformSubset
    num_classes: int

    def metadata(self) -> dict[str, Any]:
        """JSON-serialisable description of the data pipeline.

        Logged with every run and stored in every checkpoint, so a checkpoint can
        be traced back to the exact split and sampling policy that produced it.
        """
        return {
            "num_classes": self.num_classes,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "split_seed": self.manifest.seed,
            "val_fraction": self.manifest.val_fraction,
            "balance_power": self.class_weights.power if self.class_weights else 0.0,
        }


def _loader_kwargs(config: ProjectConfig) -> dict[str, Any]:
    data = config.data
    kwargs: dict[str, Any] = {
        "num_workers": data.num_workers,
        "pin_memory": data.pin_memory,
        "worker_init_fn": seed_worker,
    }
    if data.num_workers > 0:
        kwargs["persistent_workers"] = data.persistent_workers
    return kwargs


def build_dataloaders(
    config: ProjectConfig,
    *,
    download: bool = True,
    force_split: bool = False,
) -> DataBundle:
    """Build the training, validation and test loaders.

    The split manifest is loaded if present and re-created if not. Either way it is
    verified against the dataset before use, so a manifest left over from a
    different dataset revision fails loudly instead of quietly mis-splitting.

    Args:
        config: full project configuration.
        download: allow downloading GTSRB when it is not present.
        force_split: recompute and overwrite the split manifest.

    Returns:
        A :class:`DataBundle`.
    """
    data = config.data
    train_transform = build_train_transform(data)
    eval_transform = build_transform(data)

    raw_train = load_raw_split(data.root, "train", download=download)
    labels = read_labels(raw_train)

    manifest = get_or_create_manifest(labels, data, force=force_split)

    train_subset = TransformSubset(
        raw_train,
        manifest.train_indices,
        train_transform,
        num_classes=data.num_classes,
    )
    val_subset = TransformSubset(
        raw_train,
        manifest.val_indices,
        eval_transform,
        num_classes=data.num_classes,
    )

    raw_test = load_raw_split(data.root, "test", download=download)
    test_subset = TransformSubset(
        raw_test,
        list(range(len(raw_test))),
        eval_transform,
        num_classes=data.num_classes,
    )

    class_weights: ClassWeights | None = None
    if data.balance.mode == "weighted":
        class_weights = compute_class_weights(train_subset.labels, data.balance.power)

    generator = make_generator(config.training.seed)
    loader_kwargs = _loader_kwargs(config)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.training.batch_size,
        shuffle=class_weights is None,
        sampler=(
            WeightedRandomSampler(
                weights=class_weights.as_tensor(train_subset.labels).tolist(),
                num_samples=len(train_subset),
                replacement=True,
                generator=generator,
            )
            if class_weights is not None
            else None
        ),
        generator=generator,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    bundle = DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        manifest=manifest,
        class_weights=class_weights,
        train_dataset=train_subset,
        val_dataset=val_subset,
        test_dataset=test_subset,
        num_classes=data.num_classes,
    )

    log_extra: dict[str, Any] = {
        **bundle.metadata(),
        "image_size": data.image_size,
        "batch_size": config.training.batch_size,
        "balance_mode": data.balance.mode,
    }
    if class_weights is not None:
        effective = class_weights.effective_counts(train_subset.labels, len(train_subset))
        log_extra["effective_class_counts_min"] = round(min(effective.values()), 1)
        log_extra["effective_class_counts_max"] = round(max(effective.values()), 1)
    logger.info("built data pipeline", extra=log_extra)

    return bundle


__all__ = [
    "ClassWeights",
    "DataBundle",
    "build_dataloaders",
    "compute_class_weights",
]
