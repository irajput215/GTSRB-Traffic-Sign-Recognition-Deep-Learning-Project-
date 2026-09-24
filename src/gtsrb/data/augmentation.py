"""Training-time augmentation.

Why augment at all here: GTSRB images are cropped from video, so the same sign
appears at slightly different scales, angles, exposures and crops. Without
augmentation the model can memorise the specific framing of the training images.
The dataset is also long-tailed — the rarest class has 150 training images against
1500 for the most common — so minority classes benefit disproportionately from
every extra view.

Which transforms, and why these:

* **rotation / translation / shear / scale** — a sign's position, orientation and
  apparent size in the crop vary with camera geometry. Rotation stays small
  (15 degrees by default): traffic signs are mounted upright, so large angles
  generate images the model will never see and for which the label stops being
  plausible.
* **brightness / contrast** — the original data has strong exposure variation; the
  recorded misclassified gallery is dominated by washed-out and under-exposed
  signs.
* **random erasing** — simulates partial occlusion. Off by default so the default
  profile stays equivalent to the recorded run.

Deliberately **not** included: horizontal and vertical flips. Mirroring a sign
changes its meaning — "Turn left ahead" becomes "Turn right ahead", "Keep right"
becomes "Keep left" — so a flip-augmented training set teaches the model the wrong
label for a substantial minority of the 43 classes. This is the most common
mistake in traffic-sign augmentation, and it is why ``RandomHorizontalFlip`` does
not appear below.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torchvision import transforms

from gtsrb.config.schema import AugmentationConfig, DataConfig
from gtsrb.data.preprocessing import build_eval_transform

Transform = Callable[[Any], Any]


def build_augmentation(config: AugmentationConfig) -> transforms.Compose:
    """Build the geometry and colour augmentation block.

    The block consumes and produces PIL images, so it composes *before* the
    evaluation transform.
    """
    if not config.enabled:
        return transforms.Compose([])

    ops: list[Transform] = []
    scaling = config.scale_min != 1.0 or config.scale_max != 1.0

    if config.rotation_degrees > 0 or config.translate > 0 or config.shear_degrees > 0 or scaling:
        ops.append(
            transforms.RandomAffine(
                degrees=config.rotation_degrees,
                translate=(config.translate, config.translate) if config.translate else None,
                scale=(config.scale_min, config.scale_max) if scaling else None,
                shear=config.shear_degrees if config.shear_degrees else None,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
        )

    if config.brightness > 0 or config.contrast > 0 or config.saturation > 0 or config.hue > 0:
        ops.append(
            transforms.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
                hue=config.hue,
            )
        )

    return transforms.Compose(ops)


def build_train_transform(config: DataConfig) -> transforms.Compose:
    """Build the full training pipeline.

    Augmentation first, then *exactly* the evaluation transform. Composing the
    evaluation transform instead of repeating its steps is what guarantees the
    train and eval pipelines share resizing and normalisation: each is defined
    once, so they cannot drift apart.
    """
    steps: list[Transform] = [
        *build_augmentation(config.augmentation).transforms,
        *build_eval_transform(
            config.image_size, config.normalize_mean, config.normalize_std
        ).transforms,
    ]

    if config.augmentation.random_erasing_p > 0:
        # RandomErasing operates on the normalised tensor, so it goes last.
        steps.append(
            transforms.RandomErasing(
                p=config.augmentation.random_erasing_p,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0.0,
            )
        )

    return transforms.Compose(steps)


__all__ = ["build_augmentation", "build_train_transform"]
