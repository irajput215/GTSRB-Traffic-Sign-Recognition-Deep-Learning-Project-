"""Transfer learning from ImageNet-pretrained backbones.

The original project's ResNet-50 branch reached 96.97% while training only
``layer4`` and the new head. That is a real result, but it under-uses the model: at
224x224 with 26,640 training images, freezing three of four stages leaves most of
the network's capacity unadapted to a domain — cropped, tightly-framed signs — that
is visibly different from ImageNet.

This module keeps the original behaviour as the default (``freeze_backbone=True``,
``trainable_blocks=("layer4",)``) so the recorded comparison stays reproducible, and
exposes two changes that are known to help and cost nothing to configure:

* **more than one trainable stage** — ``trainable_blocks=("layer3", "layer4")``
  matches what the written report *claimed* the original code did (the code
  unfroze only ``layer4``; see ``docs/PROJECT_AUDIT.md``);
* **a choice of backbone activation** — ``gelu`` for the head, matching the report.

The head is a small MLP with batch norm, because the backbone's 2048-dimensional
feature vector is far wider than the 43-way problem needs and a single linear layer
on top of it overfits almost immediately on this dataset size.

Only ResNet-50 is implemented. Adding more backbones would be easy but nothing in
this project needs them, and an unexercised code path is a liability rather than a
feature.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from gtsrb.config.schema import ResNet50Config

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

#: Conv/BN stages of ResNet-50, in forward order. Used for freezing decisions.
RESNET50_BLOCKS: tuple[str, ...] = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


def _resnet50_backbone(weights: str) -> tuple[nn.Module, int]:
    """Build a ResNet-50 backbone and report its feature width.

    ``weights="none"`` builds randomly-initialised weights, which exists so the
    transfer-learning claim can be tested against a from-scratch control rather
    than asserted.
    """
    if weights == "imagenet":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        backbone = models.resnet50(weights=None)
    in_features = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return backbone, in_features


class ResNet50Classifier(nn.Module):
    """ImageNet-pretrained ResNet-50 with a custom classification head.

    Args:
        config: weight source, freezing policy and head geometry.
        num_classes: output logits.
        image_size: recorded for checkpoint metadata; a ResNet-50 accepts any size
            divisible by 32.
    """

    def __init__(
        self,
        config: ResNet50Config,
        *,
        num_classes: int = 43,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.image_size = image_size

        self.backbone, in_features = _resnet50_backbone(config.weights)

        self.freeze_all()
        if config.freeze_backbone:
            self.unfreeze_blocks(config.trainable_blocks)
        else:
            self.unfreeze_all()

        activation = _ACTIVATIONS[config.head_activation]
        self.classifier = nn.Sequential(
            nn.Linear(in_features, config.head_hidden),
            nn.BatchNorm1d(config.head_hidden),
            activation(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden, num_classes),
        )

    # -- freezing policy ---------------------------------------------------
    def freeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def unfreeze_blocks(self, blocks: tuple[str, ...]) -> None:
        """Allow gradients through the named stages, and everything after them.

        Unfreezing ``layer4`` while leaving ``layer3`` frozen would train a stage
        whose inputs never change, so the stages after each named block are
        unfrozen too. ``conv1``/``bn1`` unfreeze only themselves: they are the
        first stage and have nothing before them.
        """
        for name in blocks:
            if name not in RESNET50_BLOCKS:
                raise ValueError(
                    f"Unknown ResNet-50 block {name!r}; expected one of {RESNET50_BLOCKS}"
                )
            index = RESNET50_BLOCKS.index(name)
            for following in RESNET50_BLOCKS[index:]:
                module = getattr(self.backbone, following)
                for parameter in module.parameters():
                    parameter.requires_grad = True

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``(N, num_classes)``."""
        logits: torch.Tensor = self.classifier(self.backbone(x))
        return logits


__all__ = ["RESNET50_BLOCKS", "ResNet50Classifier"]
