"""Loss functions.

Cross-entropy, with optional label smoothing.

Label smoothing is worth having available for this dataset specifically: GTSRB
contains genuine label noise at the margins — a handful of images are partially
occluded or so blurred that the "correct" class is arguable. Hard one-hot targets
push the model to be maximally confident on those too. Smoothing caps the target
confidence and is a cheap regulariser. It defaults to ``0.0`` so the default
training configuration reproduces the recorded run exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from gtsrb.config.schema import TrainingConfig


def build_loss(config: TrainingConfig) -> nn.Module:
    """Build the training criterion.

    Raises:
        ValueError: if the configured loss is not implemented. Silently falling back
            to cross-entropy would hide a configuration mistake.
    """
    if config.loss != "cross_entropy":
        raise ValueError(f"Unsupported loss {config.loss!r}; only 'cross_entropy' is implemented")
    return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


def softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to class probabilities along the last dimension.

    Kept separate from the model so that models return logits. ``CrossEntropyLoss``
    expects raw logits, and applying softmax inside a model is a classic source of
    double-softmax bugs at inference time.
    """
    return torch.softmax(logits, dim=-1)


__all__ = ["build_loss", "softmax_probabilities"]
