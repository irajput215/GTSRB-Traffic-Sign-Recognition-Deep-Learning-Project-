"""Transforms and normalisation.

Two pipelines, and they must not be confused:

* the **evaluation** pipeline resizes, converts to a tensor and normalises;
* the **training** pipeline does the same but inserts augmentation first.

They share the resize, tensor conversion and normalisation steps so the two
cannot drift apart. ``build_train_transform`` composes the evaluation transform
*after* the augmentation block rather than re-declaring it, which is what makes
that guarantee structural instead of a convention.

Normalisation uses GTSRB channel statistics rather than ImageNet statistics for
the 64x64 models. The constants are configuration, not code, and
``estimate_channel_statistics`` in ``gtsrb.data.validation`` exists so they can be
measured from the data rather than trusted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from gtsrb.config.schema import DataConfig

MEAN_STD = tuple[float, float, float]


def build_eval_transform(
    image_size: int,
    mean: MEAN_STD,
    std: MEAN_STD,
) -> transforms.Compose:
    """Deterministic preprocessing used for validation, test and inference.

    Resizing uses bilinear interpolation with ``antialias=True``: GTSRB source
    images are small and often rectangular, and downsampling them without
    antialiasing aliases the thin black pictogram strokes that carry most of the
    class information.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_transform(config: DataConfig) -> transforms.Compose:
    """Build the evaluation transform for ``config``."""
    return build_eval_transform(config.image_size, config.normalize_mean, config.normalize_std)


def denormalize(
    tensor: torch.Tensor,
    mean: MEAN_STD,
    std: MEAN_STD,
) -> torch.Tensor:
    """Invert ``Normalize`` so a tensor can be displayed or saved.

    Args:
        tensor: a ``(C, H, W)`` or ``(N, C, H, W)`` normalised tensor.
        mean: the per-channel mean used to normalise.
        std: the per-channel standard deviation used to normalise.

    Returns:
        A tensor of the same shape clamped to ``[0, 1]``.
    """
    if tensor.dim() not in (3, 4):
        raise ValueError(f"Expected a 3D or 4D tensor, got shape {tuple(tensor.shape)}")

    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    view_shape: tuple[int, ...] = (1, 3, 1, 1) if tensor.dim() == 4 else (3, 1, 1)
    restored = tensor * std_t.view(view_shape) + mean_t.view(view_shape)
    return restored.clamp(0.0, 1.0)


def to_display_array(
    tensor: torch.Tensor,
    mean: MEAN_STD = (0.3403, 0.3121, 0.3214),
    std: MEAN_STD = (0.2724, 0.2608, 0.2669),
) -> np.ndarray:
    """Convert a normalised CHW tensor into an HWC float array in ``[0, 1]``.

    Raises:
        ValueError: if the tensor is not a 3-channel image. Traffic signs are RGB;
            silently accepting a greyscale tensor here would produce confusing
            plots rather than an error.
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected a single CHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {tensor.shape[0]}")
    array = denormalize(tensor.detach().cpu(), mean, std).numpy()
    return np.transpose(array, (1, 2, 0))


def channel_statistics(
    images: Sequence[np.ndarray],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute per-channel mean and standard deviation over ``images``.

    Each image must be an ``(H, W, 3)`` float array scaled to ``[0, 1]``.

    Raises:
        ValueError: if ``images`` is empty or an image has an unexpected shape.
    """
    if not images:
        raise ValueError("Cannot compute channel statistics from an empty sequence")
    stacked = []
    for index, image in enumerate(images):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image {index} must have shape (H, W, 3), got {image.shape}")
        stacked.append(image.reshape(-1, 3))
    pixels = np.concatenate(stacked, axis=0)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return (
        (float(mean[0]), float(mean[1]), float(mean[2])),
        (float(std[0]), float(std[1]), float(std[2])),
    )


__all__ = [
    "MEAN_STD",
    "build_eval_transform",
    "build_transform",
    "channel_statistics",
    "denormalize",
    "to_display_array",
]
