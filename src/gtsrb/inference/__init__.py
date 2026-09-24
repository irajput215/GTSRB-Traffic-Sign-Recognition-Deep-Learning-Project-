"""Inference: validation, preprocessing and prediction.

``Predictor`` is the single entry point. It loads a checkpoint and exposes single
and batch prediction over PIL images, raw bytes, file paths and NumPy arrays.

Everything the predictor needs comes from the checkpoint — architecture, weights,
input size, normalisation statistics and the class-label ordering — so there is no
separate serving configuration that could drift from the training configuration.

```python
from gtsrb.config import load_config
from gtsrb.inference import Predictor

predictor = Predictor.from_config(load_config([...]))
prediction = predictor.predict(image)
prediction.class_name, prediction.confidence
```
"""

from __future__ import annotations

from gtsrb.inference.predictor import (
    ClassScore,
    Prediction,
    Predictor,
    load_predictor,
)
from gtsrb.inference.preprocessing import (
    MAX_DIMENSION,
    MAX_PIXELS,
    ImagePreprocessor,
    InvalidImageError,
    PreparedImage,
)

__all__ = [
    "MAX_DIMENSION",
    "MAX_PIXELS",
    "ClassScore",
    "ImagePreprocessor",
    "InvalidImageError",
    "Prediction",
    "Predictor",
    "PreparedImage",
    "load_predictor",
]
