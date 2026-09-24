"""FastAPI inference service.

``create_app`` is the entry point; ``gtsrb.cli.serve`` runs it under uvicorn.

    from gtsrb.api import create_app
    app = create_app()

Endpoints:

* ``POST /predict`` — classify one uploaded image
* ``GET  /health`` — liveness
* ``GET  /ready`` — readiness (model loaded)
* ``GET  /model-info`` — describe the served model
* ``GET  /metrics`` — Prometheus exposition
* ``GET  /metrics-summary`` — the same counters as JSON
* ``GET  /docs`` — OpenAPI / Swagger UI
"""

from __future__ import annotations

from gtsrb.api.dependencies import AppState, get_predictor, get_request_id, get_state
from gtsrb.api.main import build_app_from_env, create_app, load_state
from gtsrb.api.metrics import Metrics
from gtsrb.api.schemas import (
    ClassScoreResponse,
    ErrorResponse,
    HealthResponse,
    MetricsSummaryResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
    error_payload,
)

__all__ = [
    "AppState",
    "ClassScoreResponse",
    "ErrorResponse",
    "HealthResponse",
    "Metrics",
    "MetricsSummaryResponse",
    "ModelInfoResponse",
    "PredictionResponse",
    "ReadyResponse",
    "build_app_from_env",
    "create_app",
    "error_payload",
    "get_predictor",
    "get_request_id",
    "get_state",
    "load_state",
]
