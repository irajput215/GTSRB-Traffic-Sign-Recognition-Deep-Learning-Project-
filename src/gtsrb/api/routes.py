"""HTTP routes.

Four endpoints, each answering one question:

* ``GET /health`` — is this process alive?
* ``GET /ready`` — can this instance serve a prediction right now?
* ``POST /predict`` — classify one uploaded image.
* ``GET /model-info`` — which model is being served, and how good was it?

``/health`` and ``/ready`` are separate on purpose. Liveness must not depend on the
model: if ``/health`` failed whenever the model was missing, an orchestrator would
restart a healthy process forever. Readiness is where model state belongs.

Errors are handled at the boundary and rendered through :func:`error_payload`, so
every failure — including the ones FastAPI raises itself — has the same body shape
and none of them expose a stack trace or an internal path.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response

from gtsrb import __version__
from gtsrb.api.dependencies import AppState, get_predictor, get_request_id, get_state
from gtsrb.api.schemas import (
    HealthResponse,
    MetricsSummaryResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
    error_payload,
)
from gtsrb.inference import InvalidImageError
from gtsrb.runtime import get_logger

logger = get_logger("api.routes")

router = APIRouter()

#: Documented error responses, attached to the /predict route so they appear in the
#: OpenAPI schema rather than only in prose.
PREDICT_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {
        "description": "The uploaded file could not be decoded as an image.",
        "content": {
            "application/json": {
                "example": {
                    "error": "invalid_image",
                    "detail": "Could not decode image: cannot identify image file",
                    "request_id": "3f9c1b2e-4d5a-4c7b-8e9f-0a1b2c3d4e5f",
                }
            }
        },
    },
    413: {
        "description": "The uploaded file exceeds the configured size limit.",
        "content": {
            "application/json": {
                "example": {
                    "error": "payload_too_large",
                    "detail": (
                        "Uploaded payload is 9000000 bytes, which exceeds the 5000000-byte limit"
                    ),
                    "request_id": "3f9c1b2e-4d5a-4c7b-8e9f-0a1b2c3d4e5f",
                }
            }
        },
    },
    422: {
        "description": "The request did not include a file field named 'file'.",
        "content": {
            "application/json": {
                "example": {
                    "error": "missing_file",
                    "detail": "Request must include a file field named 'file'",
                    "request_id": "3f9c1b2e-4d5a-4c7b-8e9f-0a1b2c3d4e5f",
                }
            }
        },
    },
    503: {
        "description": "The model is not loaded yet.",
        "content": {
            "application/json": {
                "example": {
                    "error": "model_unavailable",
                    "detail": "Model is not loaded yet",
                    "request_id": "3f9c1b2e-4d5a-4c7b-8e9f-0a1b2c3d4e5f",
                }
            }
        },
    },
}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 whenever the process is running. Deliberately independent of "
        "the model: a liveness probe that fails when the model is unavailable causes "
        "an orchestrator to restart a healthy process indefinitely."
    ),
    tags=["operations"],
)
def health(request: Request) -> HealthResponse:
    state: AppState = get_state(request)
    return HealthResponse(
        status="ok",
        uptime_seconds=round(state.uptime_seconds, 3),
        version=__version__,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        503: {
            "description": "The model is still loading, or failed to load.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "not_ready",
                        "model_loaded": False,
                        "model_name": None,
                        "device": None,
                        "checkpoint": None,
                    }
                }
            },
        }
    },
    summary="Readiness probe",
    description=(
        "Returns 200 once a checkpoint is loaded and warmed up, and 503 otherwise. "
        "This is the probe a load balancer should use before routing traffic here."
    ),
    tags=["operations"],
)
def ready(request: Request) -> Response:
    state: AppState = get_state(request)
    predictor = state.predictor
    payload = ReadyResponse(
        status="ready" if predictor is not None else "not_ready",
        model_loaded=predictor is not None,
        model_name=predictor.model_name if predictor else None,
        device=str(predictor.device) if predictor else None,
        checkpoint=str(predictor.checkpoint_path) if predictor else None,
    )
    return JSONResponse(
        content=payload.model_dump(),
        status_code=status.HTTP_200_OK
        if predictor is not None
        else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses=PREDICT_ERROR_RESPONSES,
    summary="Classify a traffic sign",
    description=(
        "Accepts one image as multipart form data in a field named `file` and returns "
        "the predicted GTSRB class with a ranked list of alternatives.\n\n"
        "The image is validated by decoding it, not by trusting its filename or "
        "`Content-Type`. It is never written to disk or logged."
    ),
    tags=["inference"],
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file (PNG, JPEG, PPM, BMP, WebP or TIFF)."),
) -> Response:
    predictor = get_predictor(request)
    request_id = get_request_id(request)
    state: AppState = get_state(request)
    metrics = state.metrics

    payload = await file.read()
    started = time.perf_counter()
    try:
        prediction = predictor.predict_bytes(payload)
    except InvalidImageError as exc:
        detail = str(exc)
        # Distinguish "too large" from "not an image" so a client can react
        # differently: one is a retry with a smaller file, the other is a bug.
        code = "payload_too_large" if "exceeds the" in detail else "invalid_image"
        status_code = (
            status.HTTP_413_CONTENT_TOO_LARGE
            if code == "payload_too_large"
            else status.HTTP_400_BAD_REQUEST
        )
        # The metrics middleware counts the error exactly once, using this kind.
        request.state.error_code = code
        logger.info(
            "prediction rejected",
            extra={
                "request_id": request_id,
                "error": code,
                "filename_provided": bool(file.filename),
            },
        )
        return JSONResponse(
            content=error_payload(code, detail, request_id), status_code=status_code
        )
    finally:
        await file.close()

    total_ms = (time.perf_counter() - started) * 1000
    if metrics is not None:
        metrics.observe_latency(prediction.latency_ms)
        metrics.observe_prediction(prediction.class_id)

    response_payload = prediction.as_dict(include_all_scores=True)
    logger.info(
        "prediction served",
        extra={
            "request_id": request_id,
            "class_id": prediction.class_id,
            "class_name": prediction.class_name,
            "confidence": round(prediction.confidence, 4),
            "inference_ms": round(prediction.latency_ms, 3),
            "total_ms": round(total_ms, 3),
            # Only image-derived facts that are the service's own output.
            "original_size": list(prediction.original_size),
        },
    )
    return JSONResponse(content=response_payload, status_code=status.HTTP_200_OK)


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses={503: {"description": "The model is not loaded yet."}},
    summary="Describe the served model",
    description=(
        "Returns the architecture, checkpoint provenance, input geometry and the "
        "validation metrics recorded when the checkpoint was written. Useful for "
        "confirming which model a deployment is actually running."
    ),
    tags=["operations"],
)
def model_info(request: Request) -> ModelInfoResponse:
    state: AppState = get_state(request)
    predictor = state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded yet"
        )
    return ModelInfoResponse.model_validate(predictor.info())


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description=(
        "Prometheus exposition format. Exposes request counts, prediction latency, "
        "error counts and the predicted-class distribution. Returns a comment-only "
        "body when metrics are disabled by configuration."
    ),
    response_class=Response,
    tags=["operations"],
)
def prometheus_metrics(request: Request) -> Response:
    state: AppState = get_state(request)
    metrics = state.metrics
    if metrics is None:
        return Response(content=b"# metrics are disabled\n", media_type="text/plain")
    return Response(content=metrics.render(), media_type=metrics.content_type)


@router.get(
    "/metrics-summary",
    response_model=MetricsSummaryResponse,
    summary="Service counters as JSON",
    description="A readable view of the same counters, for inspection without Prometheus.",
    tags=["operations"],
)
def metrics_summary(request: Request) -> MetricsSummaryResponse:
    state: AppState = get_state(request)
    metrics = state.metrics
    if metrics is None:
        return MetricsSummaryResponse()
    return MetricsSummaryResponse.model_validate(metrics.summary())


__all__ = ["PREDICT_ERROR_RESPONSES", "router"]
