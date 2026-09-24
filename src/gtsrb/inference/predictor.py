"""Checkpoint-backed prediction.

The clean interface the rest of the system uses for inference:

```python
predictor = Predictor.from_config(config)
prediction = predictor.predict(pil_image)
prediction.predicted.class_name   # "Stop"
prediction.predicted.confidence   # 0.98
prediction.top_k                  # ranked alternatives
```

Everything the predictor needs comes from the checkpoint: architecture, weights,
input size, normalisation statistics and the class-label ordering. There is no
separate "serving configuration" that could drift from the training configuration,
which is the failure mode that makes a served model quietly worse than the model
that was evaluated.

The predictor owns the model in ``eval`` mode and wraps inference in
``torch.inference_mode()``. ``inference_mode`` is stricter than ``no_grad``: it also
skips version-counter bookkeeping on tensors, which is a real saving on a service
handling concurrent requests, and it makes accidental autograd use an error rather
than a silent slowdown.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from gtsrb.config.labels import class_category, class_name, class_short_name
from gtsrb.config.schema import ProjectConfig
from gtsrb.inference.preprocessing import ImagePreprocessor, InvalidImageError, PreparedImage
from gtsrb.models import CheckpointMetadata, build_model, load_checkpoint
from gtsrb.runtime import get_logger, resolve_device

logger = get_logger("inference.predictor")


@dataclass(frozen=True)
class ClassScore:
    """One class and the model's confidence in it."""

    class_id: int
    confidence: float

    @property
    def class_name(self) -> str:
        return class_name(self.class_id)

    @property
    def short_name(self) -> str:
        return class_short_name(self.class_id)

    @property
    def family(self) -> str:
        return class_category(self.class_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "family": self.family,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class Prediction:
    """A complete prediction for one image."""

    predicted: ClassScore
    top_k: tuple[ClassScore, ...]
    latency_ms: float
    original_size: tuple[int, int]
    model_name: str
    checkpoint_version: str

    @property
    def class_id(self) -> int:
        return self.predicted.class_id

    @property
    def class_name(self) -> str:
        return self.predicted.class_name

    @property
    def confidence(self) -> float:
        return self.predicted.confidence

    def as_dict(self, *, include_all_scores: bool = False) -> dict[str, Any]:
        """Serialise for an API response.

        ``top_predictions`` always contains the full ranked list, as the task's
        response contract requires; the top entry is repeated as the headline fields
        so a client that only wants the answer does not have to index into a list.
        """
        payload: dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "family": self.predicted.family,
            "confidence": round(self.confidence, 6),
            "top_predictions": [score.as_dict() for score in self.top_k],
            "latency_ms": round(self.latency_ms, 3),
            "model_name": self.model_name,
            "model_version": self.checkpoint_version,
        }
        if include_all_scores:
            payload["original_size"] = list(self.original_size)
        return payload


def _to_class_scores(probabilities: torch.Tensor, top_k: int) -> tuple[ClassScore, ...]:
    """Rank the top ``top_k`` classes by probability.

    ``torch.topk`` returns ``(values, indices)`` in that order; zipping them the
    other way round silently swaps the class id for the probability.
    """
    count = min(top_k, int(probabilities.numel()))
    values, indices = probabilities.topk(count)
    return tuple(
        ClassScore(class_id=int(index), confidence=float(value))
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    )


class Predictor:
    """Loads a checkpoint and serves predictions.

    Args:
        config: project configuration. ``inference`` supplies the checkpoint path,
            device and ``top_k``; ``data`` is only used to build the dataloaders for
            batch workflows elsewhere and is not required here.
        checkpoint_path: overrides ``config.inference.checkpoint_path``.
        device: overrides ``config.inference.device``.

    Raises:
        CheckpointError: if the checkpoint is missing or incompatible.
    """

    def __init__(
        self,
        config: ProjectConfig,
        checkpoint_path: Path | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.checkpoint_path = Path(checkpoint_path or config.inference.checkpoint_path)
        self.device = resolve_device(device or config.inference.device)

        loaded = load_checkpoint(self.checkpoint_path, map_location="cpu")
        self.metadata: CheckpointMetadata = loaded.metadata

        # Rebuild from the checkpoint's own config, not the caller's, so serving
        # cannot diverge from training.
        checkpoint_config = ProjectConfig.model_validate(self.metadata.config)
        model = build_model(checkpoint_config.model, image_size=self.metadata.image_size)
        model.load_state_dict(loaded.model_state, strict=True)
        model.to(self.device)
        model.eval()
        self.model = model

        self.preprocessor = ImagePreprocessor(
            image_size=self.metadata.image_size,
            mean=self.metadata.normalize_mean,
            std=self.metadata.normalize_std,
            max_upload_bytes=config.inference.max_upload_bytes,
        )
        self.top_k = min(config.inference.top_k, self.metadata.num_classes)

        logger.info(
            "loaded model for inference",
            extra={
                "checkpoint": str(self.checkpoint_path),
                "model_name": self.metadata.model_name,
                "image_size": self.metadata.image_size,
                "num_classes": self.metadata.num_classes,
                "trained_epoch": self.metadata.epoch,
                "device": str(self.device),
                "top_k": self.top_k,
            },
        )

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: ProjectConfig,
        checkpoint_path: Path | None = None,
        *,
        device: str | None = None,
    ) -> Predictor:
        """Build a predictor from configuration. Equivalent to the constructor."""
        return cls(config, checkpoint_path, device=device)

    # -- introspection -----------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.metadata.model_name

    @property
    def num_classes(self) -> int:
        return self.metadata.num_classes

    @property
    def image_size(self) -> int:
        return self.metadata.image_size

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.metadata.class_names

    @property
    def trained_epoch(self) -> int:
        return self.metadata.epoch

    @property
    def train_metrics(self) -> dict[str, float]:
        return dict(self.metadata.metrics)

    def info(self) -> dict[str, Any]:
        """A description of the served model, for a ``/model-info`` endpoint or a log."""
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "model_name": self.model_name,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_format_version": self.metadata.format_version,
            "created_at": self.metadata.created_at,
            "trained_epoch": self.trained_epoch,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "normalize_mean": list(self.metadata.normalize_mean),
            "normalize_std": list(self.metadata.normalize_std),
            "device": str(self.device),
            "parameters": parameter_count,
            "validation_metrics": self.train_metrics,
            "git_commit": self.metadata.git_commit,
            "library_versions": self.metadata.library_versions,
        }

    # -- warm-up -----------------------------------------------------------
    @torch.inference_mode()
    def warmup(self) -> float:
        """Run one forward pass so the first real request is not the slow one.

        Returns the elapsed milliseconds. Called on API startup: lazy kernel
        compilation and memory allocation on the first request otherwise show up as
        a latency outlier in any monitoring.
        """
        started = time.perf_counter()
        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        self.model(dummy)
        if self.device.type == "cuda":  # pragma: no cover - no CUDA in CI
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("model warmed up", extra={"latency_ms": round(elapsed_ms, 3)})
        return elapsed_ms

    # -- prediction --------------------------------------------------------
    @torch.inference_mode()
    def predict_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for a ``(N, C, H, W)`` batch."""
        if batch.dim() != 4:
            raise ValueError(f"Expected a 4D batch (N, C, H, W), got shape {tuple(batch.shape)}")
        if batch.shape[1] != 3:
            raise ValueError(f"Expected 3-channel input, got {batch.shape[1]} channels")
        logits = self.model(batch.to(self.device))
        return torch.softmax(logits.float(), dim=-1)

    def predict(self, image: Image.Image) -> Prediction:
        """Predict a single PIL image.

        Raises:
            InvalidImageError: if the image is unusable.
        """
        return self.predict_batch([image])[0]

    def predict_bytes(self, payload: bytes) -> Prediction:
        """Predict a single image supplied as raw bytes.

        Raises:
            InvalidImageError: if the payload is not a usable image.
        """
        return self.predict_batch_bytes([payload])[0]

    def predict_path(self, path: Path) -> Prediction:
        """Predict a single image file.

        Raises:
            InvalidImageError: if the file is missing or unusable.
        """
        return self.predict_batch_paths([path])[0]

    def predict_array(self, array: np.ndarray) -> Prediction:
        """Predict a single image supplied as an ``(H, W, 3)`` uint8 or float array.

        Raises:
            InvalidImageError: if the array has an unexpected shape or dtype.
        """
        if array.ndim != 3 or array.shape[2] != 3:
            raise InvalidImageError(f"Expected an (H, W, 3) array, got shape {array.shape}")
        if array.dtype not in (np.uint8, np.float32, np.float64):
            raise InvalidImageError(
                f"Expected uint8 or float pixels, got dtype {array.dtype}. "
                "Float arrays are interpreted as 0-255 and scaled; pass uint8 to avoid ambiguity."
            )
        return self.predict(Image.fromarray(array))

    def predict_batch(self, images: Sequence[Image.Image]) -> list[Prediction]:
        """Predict a batch of PIL images in a single forward pass.

        Raises:
            InvalidImageError: if ``images`` is empty or any image is unusable.
        """
        if not images:
            raise InvalidImageError("Cannot predict an empty batch")
        prepared = [self.preprocessor.prepare(image) for image in images]
        return self._run(prepared)

    def predict_batch_bytes(self, payloads: Sequence[bytes]) -> list[Prediction]:
        """Predict a batch of byte payloads in a single forward pass."""
        if not payloads:
            raise InvalidImageError("Cannot predict an empty batch")
        prepared = [self.preprocessor.prepare_bytes(payload) for payload in payloads]
        return self._run(prepared)

    def predict_batch_paths(self, paths: Sequence[Path]) -> list[Prediction]:
        """Predict a batch of image files in a single forward pass."""
        if not paths:
            raise InvalidImageError("Cannot predict an empty batch")
        prepared = [self.preprocessor.prepare_path(path) for path in paths]
        return self._run(prepared)

    def _run(self, prepared: list[PreparedImage]) -> list[Prediction]:
        batch = self.preprocessor.batch(prepared)
        started = time.perf_counter()
        probabilities = self.predict_tensor(batch)
        elapsed_ms = (time.perf_counter() - started) * 1000
        per_image_ms = elapsed_ms / len(prepared)

        results: list[Prediction] = []
        for index, item in enumerate(prepared):
            scores = _to_class_scores(probabilities[index].cpu(), self.top_k)
            results.append(
                Prediction(
                    predicted=scores[0],
                    top_k=scores,
                    latency_ms=per_image_ms,
                    original_size=item.original_size,
                    model_name=self.model_name,
                    checkpoint_version=f"epoch-{self.trained_epoch}",
                )
            )

        logger.info(
            "prediction served",
            extra={
                "batch_size": len(prepared),
                "latency_ms": round(per_image_ms, 3),
                "top_class_id": results[0].class_id,
                "top_confidence": round(results[0].confidence, 4),
                "device": str(self.device),
                "model_name": self.model_name,
            },
        )
        return results


def load_predictor(
    config: ProjectConfig,
    checkpoint_path: Path | None = None,
    *,
    device: str | None = None,
) -> Predictor:
    """Convenience wrapper around :class:`Predictor`."""
    return Predictor(config, checkpoint_path, device=device)


__all__ = [
    "ClassScore",
    "InvalidImageError",
    "Prediction",
    "Predictor",
    "load_predictor",
]
