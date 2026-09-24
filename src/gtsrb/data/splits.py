"""Train/validation splitting and split manifests.

The original project split the training data with
``train_test_split(..., test_size=0.2, stratify=labels, random_state=43)``, which
is a sound choice. What it did not do was *record* the result, so "we used the
same split" was a claim nobody could check.

This module keeps the same partition and writes it down. The manifest is the
contract: it stores the exact indices, the label of each index, and the seed and
fraction that produced it. ``load_manifest`` re-derives the labels and refuses to
return a manifest whose labels no longer match the dataset, which is what stops a
stale manifest from silently mis-splitting a changed dataset.

Leakage discipline, stated once so it is checkable:

    split first  ->  balance the training part  ->  augment the training part

No augmented view and no resampled copy of a validation image can reach the
training set, because the validation indices are chosen before either happens and
the validation pipeline applies no augmentation at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from gtsrb.config.schema import DataConfig
from gtsrb.runtime import get_logger

logger = get_logger("data.splits")

#: Bumped when the manifest layout changes in an incompatible way.
MANIFEST_VERSION = 1


class SplitError(RuntimeError):
    """Raised when a split cannot be created or a manifest is not usable."""


@dataclass(frozen=True)
class SplitManifest:
    """An auditable record of how the training data was partitioned."""

    version: int
    seed: int
    val_fraction: float
    num_samples: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_labels: tuple[int, ...]
    val_labels: tuple[int, ...]
    created_at: str

    @property
    def num_train(self) -> int:
        return len(self.train_indices)

    @property
    def num_val(self) -> int:
        return len(self.val_indices)

    def class_counts(self, split: str) -> dict[int, int]:
        """Per-class counts for ``"train"`` or ``"val"``."""
        if split == "train":
            labels = self.train_labels
        elif split == "val":
            labels = self.val_labels
        else:
            raise ValueError(f"Unknown split {split!r}; expected 'train' or 'val'")
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "num_samples": self.num_samples,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "created_at": self.created_at,
            "train_indices": list(self.train_indices),
            "val_indices": list(self.val_indices),
            "train_labels": list(self.train_labels),
            "val_labels": list(self.val_labels),
        }

    def save(self, path: Path) -> None:
        """Write the manifest as JSON, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info(
            "wrote split manifest",
            extra={"path": str(path), "train": self.num_train, "val": self.num_val},
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitManifest:
        """Rebuild a manifest from parsed JSON.

        Raises:
            SplitError: if required keys are missing or the layout version is
                unknown. Guessing here would defeat the purpose of the manifest.
        """
        required = {
            "version",
            "seed",
            "val_fraction",
            "num_samples",
            "train_indices",
            "val_indices",
            "train_labels",
            "val_labels",
        }
        missing = required - payload.keys()
        if missing:
            raise SplitError(f"Split manifest is missing keys: {sorted(missing)}")
        if payload["version"] != MANIFEST_VERSION:
            raise SplitError(
                f"Split manifest version {payload['version']} is not supported "
                f"(expected {MANIFEST_VERSION}); delete it and re-create the split"
            )
        return cls(
            version=int(payload["version"]),
            seed=int(payload["seed"]),
            val_fraction=float(payload["val_fraction"]),
            num_samples=int(payload["num_samples"]),
            train_indices=tuple(int(i) for i in payload["train_indices"]),
            val_indices=tuple(int(i) for i in payload["val_indices"]),
            train_labels=tuple(int(label) for label in payload["train_labels"]),
            val_labels=tuple(int(label) for label in payload["val_labels"]),
            created_at=str(payload.get("created_at", "unknown")),
        )

    @classmethod
    def load(cls, path: Path) -> SplitManifest:
        """Read a manifest from disk.

        Raises:
            SplitError: if the file does not exist or is not valid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise SplitError(f"Split manifest not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SplitError(f"Split manifest {path} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SplitError(f"Split manifest {path} must contain a JSON object")
        return cls.from_dict(payload)


def compute_split(
    labels: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Stratified train/validation index split.

    Positions are split, not the images themselves, so the caller keeps full
    control over which transform each side receives.

    Args:
        labels: label of every sample, in dataset order.
        val_fraction: fraction of samples reserved for validation.
        seed: random seed; the split is a deterministic function of this.

    Returns:
        ``(train_indices, val_indices)``, both sorted ascending so the manifest is
        stable and diffable.

    Raises:
        SplitError: if there is not enough data per class to stratify.
    """
    if not labels:
        raise SplitError("Cannot split an empty dataset")

    indices = list(range(len(labels)))
    # A class needs at least two members for stratified splitting to place one on
    # each side. GTSRB's rarest class has 150, but synthetic fixtures may not.
    rarest = min(labels.count(label) for label in set(labels))
    if rarest < 2:
        raise SplitError(
            f"Every class needs at least 2 samples to stratify; the rarest class has {rarest}. "
            "Collect more data for that class, or split without stratification."
        )

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )
    return sorted(train_indices), sorted(val_indices)


def build_manifest(
    labels: list[int],
    config: DataConfig,
) -> SplitManifest:
    """Create a split manifest for ``labels`` under ``config``'s split policy."""
    train_indices, val_indices = compute_split(labels, config.val_fraction, config.split_seed)
    return SplitManifest(
        version=MANIFEST_VERSION,
        seed=config.split_seed,
        val_fraction=config.val_fraction,
        num_samples=len(labels),
        train_indices=tuple(train_indices),
        val_indices=tuple(val_indices),
        train_labels=tuple(labels[i] for i in train_indices),
        val_labels=tuple(labels[i] for i in val_indices),
        created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
    )


def verify_manifest(manifest: SplitManifest, labels: list[int], config: DataConfig) -> None:
    """Check a manifest still describes this dataset and this split policy.

    This is the guard against the failure mode a persisted split introduces: a
    manifest that was correct for a previous dataset revision, silently applied to
    a new one.

    Raises:
        SplitError: if the manifest does not match.
    """
    if manifest.num_samples != len(labels):
        raise SplitError(
            f"Split manifest describes {manifest.num_samples} samples but the dataset has "
            f"{len(labels)}. Delete {config.split_manifest} and re-create the split."
        )
    if manifest.seed != config.split_seed:
        raise SplitError(
            f"Split manifest was built with seed {manifest.seed} but the config specifies "
            f"{config.split_seed}. Delete {config.split_manifest} and re-create the split."
        )
    if abs(manifest.val_fraction - config.val_fraction) > 1e-9:
        raise SplitError(
            f"Split manifest used val_fraction={manifest.val_fraction} but the config specifies "
            f"{config.val_fraction}. Delete {config.split_manifest} and re-create the split."
        )

    covered = set(manifest.train_indices) | set(manifest.val_indices)
    if covered != set(range(len(labels))):
        raise SplitError(
            "Split manifest indices do not cover the dataset exactly once; the manifest is corrupt"
        )
    overlap = set(manifest.train_indices) & set(manifest.val_indices)
    if overlap:
        raise SplitError(
            f"Split manifest has {len(overlap)} index/indices in both train and validation, "
            "which is a data leak"
        )

    for name, indices, recorded in (
        ("train", manifest.train_indices, manifest.train_labels),
        ("val", manifest.val_indices, manifest.val_labels),
    ):
        mismatched = [
            (position, index, expected, labels[index])
            for position, (index, expected) in enumerate(zip(indices, recorded, strict=True))
            if labels[index] != expected
        ]
        if mismatched:
            position, index, expected, actual = mismatched[0]
            raise SplitError(
                f"Split manifest label mismatch in {name} at position {position} "
                f"(dataset index {index}): manifest says {expected}, dataset says {actual}. "
                "The dataset changed underneath the manifest; re-create the split."
            )


def get_or_create_manifest(
    labels: list[int],
    config: DataConfig,
    *,
    force: bool = False,
) -> SplitManifest:
    """Load the manifest for this config, creating it if absent.

    Args:
        labels: label of every sample in the training split, in dataset order.
        config: provides ``split_manifest``, ``split_seed`` and ``val_fraction``.
        force: ignore any existing manifest and recompute.

    Returns:
        A manifest that is guaranteed to describe ``labels``.
    """
    path = Path(config.split_manifest)
    if path.exists() and not force:
        manifest = SplitManifest.load(path)
        verify_manifest(manifest, labels, config)
        return manifest

    manifest = build_manifest(labels, config)
    verify_manifest(manifest, labels, config)
    manifest.save(path)
    return manifest


__all__ = [
    "MANIFEST_VERSION",
    "SplitError",
    "SplitManifest",
    "build_manifest",
    "compute_split",
    "get_or_create_manifest",
    "verify_manifest",
]
