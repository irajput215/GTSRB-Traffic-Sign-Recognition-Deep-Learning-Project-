"""Dataset validation and class-distribution analysis.

Three jobs:

1. **Validate inputs.** Confirm the dataset is reachable, non-empty, correctly
   labelled, and that images decode to the shape and mode the model expects. All
   of it is cheap relative to training, and every failure it catches is one that
   would otherwise surface as a confusing loss curve or, worse, as a silent
   accuracy loss.
2. **Describe the class distribution.** GTSRB is long-tailed, so the report that
   genuinely matters is not "43 classes" but min/max per class and the imbalance
   ratio.
3. **Measure the normalisation constants.** The GTSRB channel statistics used by
   this project came from the original notebook. ``estimate_channel_statistics``
   exists so they can be recomputed from the data and checked, rather than
   trusted. ``python scripts/prepare_data.py --stats`` prints exactly that
   comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from gtsrb.data.dataset import LabelledDataset, read_labels
from gtsrb.data.preprocessing import channel_statistics

#: Source images in GTSRB are RGB. A greyscale or CMYK image would normalise to
#: the wrong channel statistics, so it is treated as an error rather than coerced.
EXPECTED_MODE = "RGB"


@dataclass(frozen=True)
class DatasetStatistics:
    """A summary of a labelled dataset."""

    num_samples: int
    num_classes: int
    class_counts: dict[int, int]
    image_sizes: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def min_class_count(self) -> int:
        return min(self.class_counts.values()) if self.class_counts else 0

    @property
    def max_class_count(self) -> int:
        return max(self.class_counts.values()) if self.class_counts else 0

    @property
    def imbalance_ratio(self) -> float:
        """Most common class size divided by rarest. ``1.0`` means balanced."""
        if not self.class_counts or self.min_class_count == 0:
            return float("inf")
        return self.max_class_count / self.min_class_count

    @property
    def missing_classes(self) -> list[int]:
        """Class ids in ``[0, num_classes)`` with no samples at all."""
        return [c for c in range(self.num_classes) if self.class_counts.get(c, 0) == 0]

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.num_samples} samples, {len(self.class_counts)}/{self.num_classes} classes, "
            f"per-class min={self.min_class_count} max={self.max_class_count} "
            f"imbalance={self.imbalance_ratio:.2f}x"
        )


class DatasetValidationError(RuntimeError):
    """Raised when a dataset fails a structural check."""


def label_distribution(labels: Sequence[int], num_classes: int) -> dict[int, int]:
    """Count occurrences of each label, including zero-count classes.

    Args:
        labels: observed labels.
        num_classes: size of the label space.

    Raises:
        DatasetValidationError: if a label falls outside ``[0, num_classes)``.
            Silently dropping it would understate the problem.
    """
    counts = dict.fromkeys(range(num_classes), 0)
    for label in labels:
        if not 0 <= label < num_classes:
            raise DatasetValidationError(
                f"Label {label} is outside the valid range [0, {num_classes})"
            )
        counts[label] += 1
    return counts


def validate_dataset(
    dataset: LabelledDataset,
    *,
    num_classes: int,
    sample_size: int = 0,
    seed: int = 43,
    max_aspect_ratio: float = 4.0,
) -> DatasetStatistics:
    """Check a dataset and return its statistics.

    Args:
        dataset: anything yielding ``(PIL image, label)``.
        num_classes: size of the label space.
        sample_size: how many images to decode and check. ``0`` skips image
            decoding and only reads labels, which is the fast path used in tests
            and in CI where the dataset is a generated fixture.
        seed: sampling seed, so a sampled check is reproducible.
        max_aspect_ratio: reject an image whose longer side is more than this many
            times its shorter side. A **corruption guard, not a shape constraint**:
            GTSRB crops carry a margin and are legitimately not square.

            The default is measured, not guessed. Across **all 39,270 images** in
            both splits, the long/short ratio has median 1.033, p99 1.293, p99.9
            2.034 and **maximum 2.716**; exactly 2 images exceed 2.5 and none exceed
            3.0.

            Two earlier defaults were wrong, and both were caught by running the
            check over the real data rather than by reasoning about it:

            * a departure-from-1:1 threshold of 0.5 rejected a genuine 0.467-ratio
              image, so a full validation pass could not succeed at all — and it was
              asymmetric, since ``|ratio - 1| <= 1`` accepted a 1-pixel-wide strip;
            * a long/short bound of 2.5 rejected a genuine 67x182 image at 2.72.

            4.0 is deliberately loose. The observed maximum is 2.716, and what this
            guard is for — misread dimensions, a wrong file entirely — lands far
            above 4.0, while the gap to real data is not so small that a different
            dataset revision would trip it.

    Returns:
        Statistics for the dataset.

    Raises:
        DatasetValidationError: if any structural check fails.
    """
    if len(dataset) == 0:
        raise DatasetValidationError("Dataset is empty")

    labels = read_labels(dataset)
    counts = label_distribution(labels, num_classes)

    statistics = DatasetStatistics(
        num_samples=len(dataset),
        num_classes=num_classes,
        class_counts=counts,
    )

    if sample_size > 0:
        _check_images(
            dataset,
            sample_size=min(sample_size, len(dataset)),
            seed=seed,
            max_aspect_ratio=max_aspect_ratio,
        )

    return statistics


def _check_images(
    dataset: LabelledDataset,
    *,
    sample_size: int,
    seed: int,
    max_aspect_ratio: float,
) -> None:
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(dataset), size=sample_size, replace=False)

    for position in positions:
        index = int(position)
        image, _ = dataset[index]
        if not isinstance(image, Image.Image):
            raise DatasetValidationError(
                f"Sample {index} returned {type(image).__name__}, expected a PIL image. "
                "The base dataset must be transform-free so splits can apply their own pipeline."
            )
        if image.mode != EXPECTED_MODE:
            raise DatasetValidationError(
                f"Sample {index} has mode {image.mode!r}, expected {EXPECTED_MODE!r}"
            )
        width, height = image.size
        if width == 0 or height == 0:
            raise DatasetValidationError(f"Sample {index} has zero width or height")
        ratio = max(width, height) / min(width, height)
        if ratio > max_aspect_ratio:
            raise DatasetValidationError(
                f"Sample {index} has a {width}x{height} shape, a long/short ratio of "
                f"{ratio:.2f}, above the allowed {max_aspect_ratio}. This usually "
                "indicates a corrupt or badly cropped file."
            )


def estimate_channel_statistics(
    dataset: LabelledDataset,
    *,
    sample_size: int = 2000,
    seed: int = 43,
    image_size: int = 64,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Measure per-channel mean and standard deviation from the data.

    Images are resized to ``image_size`` first, because the statistics that matter
    are the ones of the tensors the model actually sees — resampling changes the
    pixel distribution.

    Args:
        dataset: anything yielding ``(PIL image, label)``.
        sample_size: number of images to sample. Defaults to 2000, which is enough
            for the per-channel mean to stabilise well past two decimal places.
        seed: sampling seed.
        image_size: size to resize to before measuring.

    Returns:
        ``(mean, std)`` as 3-tuples in ``[0, 1]`` scale.
    """
    if len(dataset) == 0:
        raise DatasetValidationError("Cannot estimate statistics from an empty dataset")

    rng = np.random.default_rng(seed)
    count = min(sample_size, len(dataset))
    positions = rng.choice(len(dataset), size=count, replace=False)

    images: list[np.ndarray] = []
    for position in positions:
        image, _ = dataset[int(position)]
        resized = image.convert(EXPECTED_MODE).resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        images.append(np.asarray(resized, dtype=np.float32) / 255.0)

    return channel_statistics(images)


def compare_statistics(
    measured_mean: tuple[float, float, float],
    measured_std: tuple[float, float, float],
    reference_mean: tuple[float, float, float],
    reference_std: tuple[float, float, float],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Compare measured channel statistics against a reference.

    ``tolerance`` is a **materiality** threshold, not an equality threshold. The
    reference constants in ``configs/data.yaml`` are kept unchanged so that results
    stay comparable with the recorded original run, so the useful question is not
    "are they bit-identical" but "would swapping them change anything". At the
    default of 0.02 a channel mean may differ by 7% of one standard deviation,
    which is immaterial to training.

    Measured on the real 64x64-resized training split (n = 4000, seed 43), the
    pixel-pooled estimates are mean ``[0.3285, 0.3006, 0.3092]`` and std
    ``[0.2706, 0.2612, 0.2666]``. The configured values are ``[0.3403, 0.3121,
    0.3214]`` and ``[0.2724, 0.2608, 0.2669]``. Taking the *mean of per-image
    means* instead of pooling every pixel gives ``[0.3383, 0.3104, 0.3199]``, which
    matches the configured mean to within 0.002 — the original constants were
    derived with that mixed estimator. The std matches the pooled estimator to
    within 0.002. Neither difference is material, and the comparison is reported
    rather than asserted so the provenance stays visible.

    Returns a JSON-serialisable dict with the per-channel deltas and whether all of
    them are within ``tolerance``.
    """
    mean_delta = [abs(a - b) for a, b in zip(measured_mean, reference_mean, strict=True)]
    std_delta = [abs(a - b) for a, b in zip(measured_std, reference_std, strict=True)]
    return {
        "measured_mean": list(measured_mean),
        "reference_mean": list(reference_mean),
        "mean_delta": mean_delta,
        "measured_std": list(measured_std),
        "reference_std": list(reference_std),
        "std_delta": std_delta,
        "tolerance": tolerance,
        "within_tolerance": max(*mean_delta, *std_delta) <= tolerance,
    }


__all__ = [
    "EXPECTED_MODE",
    "DatasetStatistics",
    "DatasetValidationError",
    "compare_statistics",
    "estimate_channel_statistics",
    "label_distribution",
    "validate_dataset",
]
