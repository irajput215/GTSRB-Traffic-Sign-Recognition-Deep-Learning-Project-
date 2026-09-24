"""Input validation and preprocessing for inference.

The single most important property of this module is that it preprocesses exactly
the way the model was trained. Getting that wrong degrades accuracy silently — there
is no crash, just worse predictions — so the transform is built from the
**checkpoint's recorded** image size and normalisation statistics rather than from
the current configuration. A checkpoint trained at 64x64 with GTSRB statistics is
preprocessed at 64x64 with GTSRB statistics even if ``configs/data.yaml`` has since
changed.

Why Pillow and not OpenCV: Pillow is already a project dependency (torchvision
requires it), it decodes every format an API is likely to receive, and it handles
EXIF orientation and truncated files. OpenCV would add a large compiled dependency
to solve problems this project does not have. See ``docs/DECISIONS.md``.

Validation is deliberate about what it rejects:

* **Size**, before decoding, so a 2 GB upload cannot exhaust memory.
* **Decodability**, by actually decoding — never by trusting a filename or a
  ``Content-Type`` header, both of which the client controls.
* **Mode**, converted to RGB rather than rejected, because a greyscale or CMYK
  upload is a legitimate user action and converting is unambiguous.
* **Minimum dimension**, so a 3x3 pixel image is reported as unusable instead of
  being stretched into a confident prediction.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from gtsrb.data.preprocessing import build_eval_transform
from gtsrb.runtime import get_logger

logger = get_logger("inference.preprocessing")

#: Refuse images larger than this in either dimension. GTSRB signs are small
#: crops; anything this large is either a mistake or an attempt to exhaust memory.
MAX_DIMENSION = 8000

#: Guard against decompression bombs. Pillow raises ``DecompressionBombError`` above
#: twice this value and warns above it.
MAX_PIXELS = 32_000_000


class InvalidImageError(ValueError):
    """Raised when an input cannot be treated as a traffic-sign image."""


@dataclass(frozen=True)
class PreparedImage:
    """A validated, model-ready tensor plus the facts needed to report on it."""

    tensor: torch.Tensor
    original_size: tuple[int, int]
    original_mode: str

    @property
    def width(self) -> int:
        return self.original_size[0]

    @property
    def height(self) -> int:
        return self.original_size[1]


class ImagePreprocessor:
    """Validates and transforms images for a specific checkpoint.

    Args:
        image_size: the square input size the model expects.
        mean: per-channel normalisation mean the model was trained with.
        std: per-channel normalisation standard deviation.
        max_upload_bytes: reject payloads larger than this before decoding.
        min_dimension: reject images smaller than this in either dimension.

    Raises:
        ValueError: if ``image_size`` is not positive or ``std`` contains a zero.
    """

    def __init__(
        self,
        *,
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        max_upload_bytes: int = 5_000_000,
        min_dimension: int = 8,
    ) -> None:
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if any(value <= 0 for value in std):
            raise ValueError(f"std must be strictly positive, got {std}")
        if min_dimension <= 0:
            raise ValueError(f"min_dimension must be positive, got {min_dimension}")

        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.max_upload_bytes = max_upload_bytes
        self.min_dimension = min_dimension
        # The same builder the evaluation pipeline uses, so the two cannot diverge.
        self._transform = build_eval_transform(image_size, mean, std)

    def prepare(self, image: Image.Image) -> PreparedImage:
        """Validate a PIL image and convert it to a normalised CHW tensor.

        Raises:
            InvalidImageError: if the image is unusable.
        """
        if not isinstance(image, Image.Image):
            raise InvalidImageError(f"Expected a PIL image, got {type(image).__name__}")

        width, height = image.size
        if width < self.min_dimension or height < self.min_dimension:
            raise InvalidImageError(
                f"Image is {width}x{height}; both dimensions must be at least "
                f"{self.min_dimension} pixels. A sign this small cannot be classified "
                "meaningfully."
            )
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            raise InvalidImageError(
                f"Image is {width}x{height}; neither dimension may exceed {MAX_DIMENSION} pixels"
            )

        mode = image.mode
        try:
            # ``convert`` both normalises the mode and forces a full decode, so a
            # truncated file that Pillow opened lazily fails here rather than
            # producing a half-grey tensor.
            converted = image.convert("RGB") if mode != "RGB" else image
            converted.load()
        except (OSError, ValueError) as exc:
            raise InvalidImageError(f"Image could not be decoded: {exc}") from exc

        tensor = self._transform(converted)
        if not isinstance(tensor, torch.Tensor):  # pragma: no cover - transform contract
            raise InvalidImageError("Preprocessing did not produce a tensor")
        return PreparedImage(tensor=tensor, original_size=(width, height), original_mode=mode)

    def prepare_bytes(self, payload: bytes) -> PreparedImage:
        """Validate and prepare raw bytes, as received from an HTTP upload.

        Raises:
            InvalidImageError: if the payload is empty, too large, not an image, or
                otherwise unusable.
        """
        if not payload:
            raise InvalidImageError("Uploaded payload is empty")
        if len(payload) > self.max_upload_bytes:
            raise InvalidImageError(
                f"Uploaded payload is {len(payload)} bytes, which exceeds the "
                f"{self.max_upload_bytes}-byte limit"
            )
        return self.prepare(self.decode(payload))

    def decode(self, payload: bytes) -> Image.Image:
        """Decode bytes into a PIL image.

        Raises:
            InvalidImageError: if the bytes are not a decodable image.
        """
        # Set the bomb guard per call rather than globally so the limit is explicit
        # here and does not depend on import order.
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        try:
            with Image.open(io.BytesIO(payload)) as image:
                # ``load`` inside the context manager: the file object is closed on
                # exit, and a lazily-decoded image would fail afterwards.
                image.load()
                return image.copy()
        except Image.DecompressionBombError as exc:
            raise InvalidImageError(
                f"Image is too large to process safely (limit {MAX_PIXELS} pixels)"
            ) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(f"Could not decode image: {exc}") from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit

    def prepare_path(self, path: Path) -> PreparedImage:
        """Validate and prepare an image file.

        Raises:
            InvalidImageError: if the file is missing or unusable.
        """
        path = Path(path)
        if not path.exists():
            raise InvalidImageError(f"Image file not found: {path}")
        return self.prepare_bytes(path.read_bytes())

    def batch(self, prepared: list[PreparedImage]) -> torch.Tensor:
        """Stack prepared images into a single ``(N, C, H, W)`` batch.

        Raises:
            InvalidImageError: if ``prepared`` is empty.
        """
        if not prepared:
            raise InvalidImageError("Cannot build a batch from an empty list")
        return torch.stack([item.tensor for item in prepared], dim=0)


__all__ = [
    "MAX_DIMENSION",
    "MAX_PIXELS",
    "ImagePreprocessor",
    "InvalidImageError",
    "PreparedImage",
]
