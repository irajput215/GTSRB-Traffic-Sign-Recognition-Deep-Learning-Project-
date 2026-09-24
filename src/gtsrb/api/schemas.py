"""Request and response schemas.

These are the API's contract, and they are the reason the OpenAPI documentation is
worth reading: every field carries a description, every response has an example,
and the shapes are enforced rather than described.

Two rules are applied throughout:

* **No internal detail leaks.** Error responses carry a machine-readable ``error``
  code, a human-readable ``detail`` and a ``request_id`` to correlate with the
  server log. They never carry a stack trace, a file path or a library message.
* **Confidence is a probability.** It is validated into ``[0, 1]`` on the way out,
  so a wiring mistake cannot emit a raw logit or softmax on unmapped logits.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gtsrb.config.labels import NUM_CLASSES


class ClassScoreResponse(BaseModel):
    """One candidate class and the model's confidence in it."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "class_id": 14,
                "class_name": "Stop",
                "family": "priority",
                "confidence": 0.98,
            }
        }
    )

    class_id: int = Field(..., ge=0, lt=NUM_CLASSES, description="GTSRB class id, 0-42.")
    class_name: str = Field(..., description="Official GTSRB class name.")
    family: str = Field(
        ...,
        description="Sign family: speed_limit, prohibition, priority, warning, mandatory or other.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of this class.")

    @field_validator("confidence")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be a finite number")
        return value


class PredictionResponse(BaseModel):
    """A classification result for one uploaded image."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "class_id": 14,
                "class_name": "Stop",
                "family": "priority",
                "confidence": 0.98,
                "top_predictions": [
                    {
                        "class_id": 14,
                        "class_name": "Stop",
                        "family": "priority",
                        "confidence": 0.98,
                    },
                    {
                        "class_id": 13,
                        "class_name": "Yield",
                        "family": "priority",
                        "confidence": 0.011,
                    },
                ],
                "latency_ms": 7.42,
                "model_name": "compact_cnn",
                "model_version": "epoch-28",
                "original_size": [64, 64],
            }
        }
    )

    class_id: int = Field(..., ge=0, lt=NUM_CLASSES, description="Predicted GTSRB class id.")
    class_name: str = Field(..., description="Predicted class name.")
    family: str = Field(..., description="Predicted sign family.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the predicted class.")
    top_predictions: list[ClassScoreResponse] = Field(
        ..., min_length=1, description="Ranked alternatives, most likely first."
    )
    latency_ms: float = Field(..., ge=0.0, description="Model inference time, excluding upload.")
    model_name: str = Field(..., description="Architecture that produced the prediction.")
    model_version: str = Field(
        ..., description="Checkpoint version, derived from its training epoch."
    )
    original_size: list[int] | None = Field(
        default=None, description="Width and height of the uploaded image before resizing."
    )

    @field_validator("confidence")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be a finite number")
        return value


class HealthResponse(BaseModel):
    """Liveness. Answers "is the process up?", never "is it useful?"."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "ok", "uptime_seconds": 12.5, "version": "1.0.0"}}
    )

    status: str = Field(..., description="Always 'ok' when the process can respond.")
    uptime_seconds: float = Field(..., ge=0.0, description="Seconds since the service started.")
    version: str = Field(..., description="Service version.")


class ReadyResponse(BaseModel):
    """Readiness. Answers "can this instance serve a prediction right now?"."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ready",
                "model_loaded": True,
                "model_name": "compact_cnn",
                "device": "cpu",
                "checkpoint": "artifacts/checkpoints/best.pt",
            }
        }
    )

    status: str = Field(..., description="'ready' or 'not_ready'.")
    model_loaded: bool = Field(..., description="Whether a checkpoint is loaded and warmed up.")
    model_name: str | None = Field(default=None, description="Loaded architecture, if any.")
    device: str | None = Field(default=None, description="Device the model is serving on.")
    checkpoint: str | None = Field(default=None, description="Checkpoint path in use.")


class ModelInfoResponse(BaseModel):
    """Everything needed to describe the served model in a report or a log."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "compact_cnn",
                "checkpoint": "artifacts/checkpoints/best.pt",
                "checkpoint_format_version": 1,
                "created_at": "2026-09-24T10:00:00+00:00",
                "trained_epoch": 28,
                "num_classes": 43,
                "image_size": 64,
                "device": "cpu",
                "parameters": 9557483,
                "validation_metrics": {"val_macro_f1": 0.98},
                "git_commit": None,
            }
        }
    )

    model_name: str
    checkpoint: str
    checkpoint_format_version: int
    created_at: str
    trained_epoch: int
    num_classes: int
    image_size: int
    device: str
    parameters: int
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    normalize_mean: list[float] | None = None
    normalize_std: list[float] | None = None
    git_commit: str | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """A failure, described without exposing anything internal."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "invalid_image",
                "detail": "Could not decode image: cannot identify image file",
                "request_id": "3f9c1b2e-4d5a-4c7b-8e9f-0a1b2c3d4e5f",
            }
        }
    )

    error: str = Field(..., description="Stable machine-readable error code.")
    detail: str = Field(..., description="Human-readable explanation, safe to show a user.")
    request_id: str | None = Field(
        default=None, description="Correlates this response with the server log entry."
    )


class MetricsSummaryResponse(BaseModel):
    """A JSON view of the service counters, for debugging without Prometheus."""

    requests_total: dict[str, int] = Field(default_factory=dict)
    predictions_total: dict[str, int] = Field(default_factory=dict)
    errors_total: int = 0
    prediction_latency_ms: dict[str, float] = Field(default_factory=dict)


def error_payload(code: str, detail: str, request_id: str | None = None) -> dict[str, Any]:
    """Build the standard error body.

    Kept as a function so every error path returns the same shape, including the
    ones raised by FastAPI's own handlers.
    """
    return {"error": code, "detail": detail, "request_id": request_id}


__all__ = [
    "ClassScoreResponse",
    "ErrorResponse",
    "HealthResponse",
    "MetricsSummaryResponse",
    "ModelInfoResponse",
    "PredictionResponse",
    "ReadyResponse",
    "error_payload",
]
