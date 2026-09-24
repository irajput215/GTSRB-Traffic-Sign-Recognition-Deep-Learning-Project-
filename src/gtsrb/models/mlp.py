"""Fully-connected baseline.

Kept for one reason: the original project's headline comparison was that a small
task-specific CNN beat this by 33 accuracy points (98.84% against 65.84%). Without
the baseline that comparison has nothing to anchor it, and "the CNN works" would be
an assertion rather than a measurement.

It is not a strawman and it is not misconfigured. It is a competent MLP — four
hidden layers, batch norm, dropout — and it fails for a specific, explainable
reason: flattening a 3x64x64 image into a 12,288-vector discards all spatial
structure, so the model has to relearn translation invariance from scratch, which
40k images cannot pay for.
"""

from __future__ import annotations

import torch
from torch import nn

from gtsrb.config.schema import MLPConfig


class MLPClassifier(nn.Module):
    """Fully-connected classifier over flattened images.

    Args:
        config: hidden widths and a matching dropout rate per hidden layer.
        num_classes: output logits.
        image_size: expected square input size; sets the flattened input width.
        in_channels: input channels.
    """

    def __init__(
        self,
        config: MLPConfig,
        *,
        num_classes: int = 43,
        image_size: int = 64,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.image_size = image_size
        self.in_channels = in_channels

        layers: list[nn.Module] = [nn.Flatten()]
        previous = in_channels * image_size * image_size
        for hidden, dropout in zip(config.hidden_sizes, config.dropouts, strict=True):
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``(N, num_classes)``."""
        logits: torch.Tensor = self.net(x)
        return logits


__all__ = ["MLPClassifier"]
