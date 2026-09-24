"""The from-scratch convolutional network.

This is the architecture from the original project, preserved deliberately. It was
the best-performing model in the recorded run (98.84% test accuracy against 96.97%
for the fine-tuned ResNet-50) and it is much cheaper to serve than the alternatives:
64x64 input, no dependency on ImageNet weights, and **9,557,483 parameters** (36.5 MB
in memory) as measured by ``summarise_model``.

For scale, the MLP baseline in this same project has **27,940,907** parameters
(106.6 MB) and scores 65.84%. The baseline carries three times the parameters and
still loses by 33 accuracy points, which is the cleanest possible demonstration that
parameter count is not what a convolutional prior buys you here. See
``docs/DECISIONS.md`` for why this architecture is the default.

Structure, which matches the original code exactly:

    block i: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> MaxPool2 -> Dropout(0.25)
    channels 3 -> 64 -> 128 -> 256, spatial 64 -> 32 -> 16 -> 8
    head:     Flatten -> Linear(256*8*8, 512) -> ReLU -> Dropout(0.5) -> Linear(512, 43)

One documented discrepancy: the written report claimed the head used an adaptive
average pooling layer producing a 6x6 map, but the code used three 2x2 max pools
producing 8x8 followed by a flatten. The code is what ran, so the code is what is
preserved here. This is recorded in ``docs/PROJECT_AUDIT.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from gtsrb.config.schema import CompactCNNConfig


class CompactCNN(nn.Module):
    """Three-block CNN for 64x64 traffic-sign images.

    Args:
        config: channel widths, head width and dropout rates.
        num_classes: output logits.
        image_size: expected square input size. Used to size the flattened feature
            vector; must be divisible by 8 because the network downsamples three
            times by a factor of two.
    """

    #: Spatial reduction factor: three ``MaxPool2d(2)`` stages.
    _DOWNSAMPLE = 8

    def __init__(
        self,
        config: CompactCNNConfig,
        *,
        num_classes: int = 43,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size % self._DOWNSAMPLE != 0:
            raise ValueError(
                f"image_size must be divisible by {self._DOWNSAMPLE} for this architecture, "
                f"got {image_size}"
            )

        self.config = config
        self.num_classes = num_classes
        self.image_size = image_size

        first, second, third = config.channels
        self.features = nn.Sequential(
            self._block(3, first, config.conv_dropout),
            self._block(first, second, config.conv_dropout),
            self._block(second, third, config.conv_dropout),
        )

        pooled = image_size // self._DOWNSAMPLE
        self.flattened_features = third * pooled * pooled
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_features, config.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden, num_classes),
        )

        self._initialise_weights()

    @staticmethod
    def _block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

    def _initialise_weights(self) -> None:
        """Kaiming-initialise convolutions, as is standard for ReLU networks.

        The original relied on PyTorch's defaults. Explicit initialisation is
        cheap, makes the run easier to reproduce across torch versions, and costs
        nothing at inference.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:  # pragma: no cover - bias is disabled above
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``(N, num_classes)``."""
        logits: torch.Tensor = self.classifier(self.features(x))
        return logits


__all__ = ["CompactCNN"]
