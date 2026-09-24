"""Model factory.

One function turns a validated :class:`ModelConfig` into a ``nn.Module``. Nothing
else in the codebase constructs a model, which is what guarantees the training
path, the evaluation path and the inference service all build the same thing from
the same settings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from gtsrb.config.labels import NUM_CLASSES
from gtsrb.config.schema import CompactCNNConfig, MLPConfig, ModelConfig, ModelName, ResNet50Config
from gtsrb.models.compact_cnn import CompactCNN
from gtsrb.models.mlp import MLPClassifier
from gtsrb.models.transfer import ResNet50Classifier

ModelBuilder = Callable[[ModelConfig, int, int], nn.Module]


def _build_compact_cnn(config: ModelConfig, num_classes: int, image_size: int) -> nn.Module:
    architecture: CompactCNNConfig = config.compact_cnn
    return CompactCNN(architecture, num_classes=num_classes, image_size=image_size)


def _build_mlp(config: ModelConfig, num_classes: int, image_size: int) -> nn.Module:
    architecture: MLPConfig = config.mlp
    return MLPClassifier(architecture, num_classes=num_classes, image_size=image_size)


def _build_resnet50(config: ModelConfig, num_classes: int, image_size: int) -> nn.Module:
    architecture: ResNet50Config = config.resnet50
    return ResNet50Classifier(architecture, num_classes=num_classes, image_size=image_size)


#: Registered architectures. Keys are the values accepted by ``ModelConfig.name``.
MODEL_BUILDERS: dict[ModelName, ModelBuilder] = {
    "compact_cnn": _build_compact_cnn,
    "mlp": _build_mlp,
    "resnet50": _build_resnet50,
}


def build_model(config: ModelConfig, *, image_size: int) -> nn.Module:
    """Instantiate the architecture named by ``config.name``.

    Args:
        config: model configuration, including the per-architecture settings.
        image_size: expected square input size. Needed because both the CNN and the
            MLP size their first dense layer from it.

    Returns:
        An untrained ``nn.Module``.

    Raises:
        ValueError: if ``config.name`` is not registered, or ``num_classes`` does
            not match the label space.
    """
    builder = MODEL_BUILDERS.get(config.name)
    if builder is None:
        raise ValueError(
            f"Unknown model {config.name!r}; registered architectures: {sorted(MODEL_BUILDERS)}"
        )

    if config.num_classes != NUM_CLASSES:
        raise ValueError(
            f"ModelConfig.num_classes is {config.num_classes} but the label space has "
            f"{NUM_CLASSES} classes"
        )

    model = builder(config, config.num_classes, image_size)
    if not isinstance(model, nn.Module):  # pragma: no cover - guards a builder bug
        raise TypeError(f"Builder for {config.name!r} returned {type(model).__name__}")
    return model


@dataclass(frozen=True)
class ModelSummary:
    """Size and parameter counts for a built model."""

    name: str
    total_parameters: int
    trainable_parameters: int
    size_mb: float
    image_size: int
    num_classes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "size_mb": round(self.size_mb, 3),
            "image_size": self.image_size,
            "num_classes": self.num_classes,
        }

    def describe(self) -> str:
        return (
            f"{self.name}: {self.total_parameters:,} parameters "
            f"({self.trainable_parameters:,} trainable), {self.size_mb:.1f} MB, "
            f"{self.image_size}x{self.image_size} input, {self.num_classes} classes"
        )


def summarise_model(
    model: nn.Module, *, name: str, image_size: int, num_classes: int
) -> ModelSummary:
    """Count parameters and estimate the in-memory size of ``model``.

    The size is the sum of parameter and buffer byte counts — what has to be moved
    to a device — not the on-disk checkpoint size, which is usually smaller.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    bytes_total = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    ) + sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    return ModelSummary(
        name=name,
        total_parameters=total,
        trainable_parameters=trainable,
        size_mb=bytes_total / (1024 * 1024),
        image_size=image_size,
        num_classes=num_classes,
    )


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


__all__ = [
    "MODEL_BUILDERS",
    "ModelBuilder",
    "ModelSummary",
    "build_model",
    "count_parameters",
    "summarise_model",
]
