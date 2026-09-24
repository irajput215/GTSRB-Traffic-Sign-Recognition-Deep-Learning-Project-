"""Shared pytest fixtures.

Fixtures here are deliberately lightweight: they build synthetic traffic-sign-like
images and tiny models rather than touching the GTSRB dataset. The dataset is a
several-hundred-megabyte download, so a test suite that needs it is a test suite
nobody runs. Anything that genuinely needs real data is marked ``slow`` and lives
in the integration tests.
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gtsrb.config.schema import ProjectConfig


@pytest.fixture
def project_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    return project_root / "configs"


@pytest.fixture
def base_config() -> ProjectConfig:
    """A small, fast configuration suitable for tests."""
    return ProjectConfig.model_validate(
        {
            "output_dir": "artifacts",
            "data": {"image_size": 32, "num_workers": 0, "split_seed": 7},
            "training": {"epochs": 2, "batch_size": 4, "device": "cpu", "amp": False},
            "inference": {"top_k": 3},
        }
    )


def make_sign_image(
    size: int = 32,
    *,
    class_id: int = 14,
    background: tuple[int, int, int] = (60, 60, 60),
) -> Image.Image:
    """Draw a crude synthetic traffic sign.

    The point is not realism — it is that the image has structure the model can
    overfit to, so a smoke-training test can show the loss actually moving. Each
    class id gets a distinct hue so that a synthetic dataset is learnable.
    """
    array = np.full((size, size, 3), background, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    centre = size / 2
    radius = size * 0.35
    inside = (yy - centre) ** 2 + (xx - centre) ** 2 <= radius**2

    hue = np.array(
        [
            (class_id * 37) % 256,
            (class_id * 91) % 256,
            (class_id * 53) % 256,
        ],
        dtype=np.uint8,
    )
    array[inside] = hue

    # A deterministic inner mark so different classes differ structurally too.
    mark = radius * 0.5
    inner = (np.abs(yy - centre) <= mark) & (np.abs(xx - centre) <= mark)
    array[inside & inner] = (250, 250, 250)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def sign_image_factory() -> Callable[..., Image.Image]:
    """Return the synthetic sign factory as a fixture."""
    return make_sign_image


@pytest.fixture
def sample_image() -> Image.Image:
    return make_sign_image(size=64, class_id=14)


@pytest.fixture
def image_bytes(sample_image: Image.Image) -> bytes:
    """PNG bytes of a valid sign image, as an upload would arrive."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def synthetic_image_dir(tmp_path: Path) -> Path:
    """A tiny on-disk image tree laid out like GTSRB's ``Train/<class>/*.ppm``.

    Used to exercise the dataset/split code without the real download.
    """
    rng = random.Random(1234)
    for class_id in range(43):
        class_dir = tmp_path / f"{class_id:05d}"
        class_dir.mkdir(parents=True)
        # Deliberately uneven, so class-balance logic has something to fix.
        count = rng.randint(4, 10)
        for index in range(count):
            image = make_sign_image(size=32, class_id=class_id)
            image.save(class_dir / f"{index:05d}.ppm")
    return tmp_path


@pytest.fixture
def tiny_model() -> torch.nn.Module:
    """A very small CNN with the right output shape for GTSRB."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 43),
    )


@pytest.fixture
def torch_seed() -> int:
    return 1234
