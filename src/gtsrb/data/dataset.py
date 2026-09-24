"""Dataset access.

Wraps ``torchvision.datasets.GTSRB`` so that the rest of the project never
touches it directly. Three things this layer is responsible for:

1. **Explicit indexing.** Every split is a ``TransformSubset`` over the raw
   dataset with an explicit index list. That is what makes the split auditable:
   the indices are computed once and persisted, not recomputed from a seed and
   hoped to be identical.
2. **Per-split transforms.** The base dataset always yields PIL images; the
   transform is applied by the subset. Applying the augmentation transform to the
   base dataset and then selecting a subset is how augmented views of validation
   images end up in the training set.
3. **Worker seeding.** ``worker_init_fn`` is required for reproducible
   augmentation, because worker processes do not inherit the parent's NumPy or
   Python RNG state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import GTSRB

from gtsrb.runtime import get_logger, seed_worker

logger = get_logger("data.dataset")

GTSRBSplitName = Literal["train", "test"]

#: Transform signature: a PIL image in, a tensor out.
ImageTransform = Callable[[Image.Image], torch.Tensor]


class LabelledDataset(Protocol):
    """Minimal structural type for anything that pairs an image with a label."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Image.Image, int]: ...


class TransformSubset(Dataset[tuple[torch.Tensor, int]]):
    """A subset of ``dataset`` at explicit ``indices``, with an optional transform.

    The transform is required rather than optional: a subset that yields PIL
    images and one that yields tensors are different things, and leaving the
    choice implicit is how a training pipeline ends up feeding untransformed
    images to a model. Use ``load_raw_split`` for the untransformed case.

    Args:
        dataset: the base dataset, expected to return ``(PIL image, label)``.
        indices: positions in ``dataset`` that make up this split.
        transform: applied to the PIL image on access.
        validate_labels: check that every label is a valid class id on
            construction. Cheap, and it turns a silently wrong label into a
            startup failure.
        num_classes: only used when ``validate_labels`` is true.
    """

    def __init__(
        self,
        dataset: LabelledDataset,
        indices: Sequence[int],
        transform: ImageTransform,
        *,
        validate_labels: bool = True,
        num_classes: int = 43,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self._all_labels = read_labels(dataset)
        if validate_labels:
            self._check_labels(num_classes)

    def _check_labels(self, num_classes: int) -> None:
        invalid = [
            (index, self._all_labels[index])
            for index in self.indices
            if not 0 <= self._all_labels[index] < num_classes
        ]
        if invalid:
            preview = ", ".join(f"index {i} -> label {label}" for i, label in invalid[:5])
            raise ValueError(
                f"{len(invalid)} sample(s) have a label outside [0, {num_classes}): {preview}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), int(label)

    @property
    def labels(self) -> list[int]:
        """The label of every sample in this subset, in subset order.

        Cheap enough to call when building a sampler: see ``read_labels`` for why
        this does not decode images.
        """
        return [self._all_labels[index] for index in self.indices]

    def label_distribution(self) -> dict[int, int]:
        """Count samples per class in this subset."""
        counts: dict[int, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))


def read_labels(dataset: LabelledDataset) -> list[int]:
    """Read every label, without decoding images when the dataset allows it.

    ``torchvision``'s ``GTSRB`` decodes a PNM file inside ``__getitem__`` before
    returning the label, so the obvious ``[dataset[i][1] for i in range(len(dataset))]``
    decodes all 26,640 images just to read 26,640 integers. That is roughly 30
    seconds of pure I/O on every call, and this function is called whenever a split
    manifest is verified, which is every run.

    Prefers a metadata attribute when one is available and falls back to
    ``__getitem__`` otherwise, so it is correct for any ``LabelledDataset`` and
    merely slow for the ones that expose no metadata.

    Args:
        dataset: the dataset to read labels from.

    Returns:
        One label per sample, in ``__getitem__`` order.
    """
    # torchvision's GTSRB keeps ``(path, target)`` tuples in ``_samples``.
    samples = getattr(dataset, "_samples", None)
    if (
        isinstance(samples, list)
        and samples
        and isinstance(samples[0], tuple)
        and len(samples[0]) == 2
    ):
        return [int(target) for _, target in samples]

    # Several torchvision datasets (and this project's fixtures) expose ``targets``.
    targets = getattr(dataset, "targets", None)
    if isinstance(targets, list) and len(targets) == len(dataset):
        return [int(target) for target in targets]

    return [int(dataset[index][1]) for index in range(len(dataset))]


def load_raw_split(
    root: Path,
    split: GTSRBSplitName,
    *,
    download: bool = True,
) -> LabelledDataset:
    """Load a GTSRB split with no transform applied.

    No transform is applied here on purpose: the base dataset must stay
    transform-free so that the same underlying images can back both the training
    and the validation split with different pipelines.

    Args:
        root: directory that contains (or will contain) the ``gtsrb`` folder.
        split: ``"train"`` or ``"test"``.
        download: allow downloading the archive. Set ``False`` in CI so a missing
            dataset fails fast instead of pulling several hundred megabytes.

    Returns:
        The dataset, yielding ``(PIL image, label)``.

    Note:
        ``torchvision``'s ``train`` split contains **26,640** images, whereas the
        official GTSRB training partition contains **39,209**. The difference is a
        property of torchvision's packaging, not of this code. See
        ``docs/PROJECT_AUDIT.md``; any comparison with published GTSRB numbers has
        to carry that caveat.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not download and not _dataset_present(root):
        raise FileNotFoundError(
            f"GTSRB data not found under {root} and download=False. "
            "Run 'make data' (or scripts/prepare_data.py) first."
        )
    dataset = GTSRB(root=str(root), split=split, download=download)
    logger.info(
        "loaded GTSRB split",
        extra={"split": split, "samples": len(dataset), "root": str(root)},
    )
    # ``torchvision.datasets.GTSRB`` returns ``(PIL image, int)``, which is exactly
    # ``LabelledDataset``. The cast records that, and ``validate_dataset`` checks
    # the image type at runtime rather than trusting the annotation.
    return cast("LabelledDataset", dataset)


def _dataset_present(root: Path) -> bool:
    """Return whether a previously downloaded GTSRB tree looks usable."""
    candidates = (
        root / "gtsrb" / "GTSRB" / "Training",
        root / "gtsrb" / "GTSRB" / "Final_Test",
    )
    return any(path.exists() for path in candidates)


__all__ = [
    "GTSRBSplitName",
    "ImageTransform",
    "LabelledDataset",
    "TransformSubset",
    "load_raw_split",
    "read_labels",
    "seed_worker",
]
