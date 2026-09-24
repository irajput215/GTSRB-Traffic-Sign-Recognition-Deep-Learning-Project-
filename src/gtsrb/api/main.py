"""FastAPI application factory.

``create_app`` builds the service; nothing at module scope builds one, so importing
this module never loads a model. That matters for tests, for tooling that imports
the app to read its schema, and for `uvicorn --factory`.

Startup and shutdown are handled by a lifespan context manager:

* **startup** loads the checkpoint from ``config.inference.checkpoint_path`` and
  warms it up. A failure is recorded on the app state and logged, but it does *not*
  prevent the process from starting: ``/health`` keeps returning 200 and ``/ready``
  returns 503, which is the behaviour an orchestrator needs. Crashing on startup
  would produce a restart loop and no diagnosable endpoint.
* **shutdown** drops the model reference so memory is released promptly rather than
  at process exit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from gtsrb.api.dependencies import AppState, get_request_id
from gtsrb.api.observability import add_request_logging, install_metrics
from gtsrb.api.routes import router
from gtsrb.api.schemas import error_payload
from gtsrb.config import ProjectConfig, load_config
from gtsrb.inference import Predictor
from gtsrb.runtime import configure_logging, get_logger

logger = get_logger("api.main")

DEFAULT_CONFIG_LAYERS: tuple[str, ...] = (
    "data.yaml",
    "model.yaml",
    "train.yaml",
    "api.yaml",
)

#: HTTP status to stable error code, used by the generic handler.
_STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    503: "model_unavailable",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model on startup and release it on shutdown."""
    state: AppState = app.state.gtsrb
    predictor: Predictor | None = getattr(app.state, "injected_predictor", None)

    logger.info(
        "starting inference service",
        extra={
            "checkpoint": str(state.config.inference.checkpoint_path),
            "device": state.config.inference.device,
            "warmup": state.config.api.warmup_on_startup,
            "metrics_enabled": state.config.api.metrics_enabled,
        },
    )

    try:
        if predictor is not None:
            state.load_predictor(predictor)
        elif state.config.api.warmup_on_startup:
            state.load_predictor()
        else:
            # Load without warming up, for an environment where startup latency
            # matters more than first-request latency.
            state.predictor = Predictor.from_config(state.config)
            state.warmup_ms = None
        logger.info(
            "model ready",
            extra={
                "model_name": state.predictor.model_name if state.predictor else None,
                "warmup_ms": state.warmup_ms,
                "uptime_seconds": round(state.uptime_seconds, 3),
            },
        )
    except Exception as exc:
        # Do not re-raise: /health must stay up so the failure is observable.
        state.startup_error = str(exc)
        logger.error("model could not be loaded; /ready will report 503: %s", exc)

    yield

    logger.info("shutting down inference service", extra={"uptime_seconds": state.uptime_seconds})
    state.clear()


def create_app(
    config: ProjectConfig | None = None,
    *,
    predictor: Predictor | None = None,
    config_paths: list[Path] | None = None,
    overrides: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config: a resolved configuration. When omitted, one is loaded from
            ``config_paths`` (default: the four shipped YAML layers).
        predictor: an already-loaded model. Injected by tests and by callers that
            want to share one model between apps; the lifespan then skips loading.
        config_paths: config layers to load when ``config`` is not given.
        overrides: ``key=value`` config overrides.

    Returns:
        A configured ``FastAPI`` instance.
    """
    resolved = config or load_config(config_paths or _default_config_paths(), overrides)

    app = FastAPI(
        title=resolved.api.title,
        version="1.0.0",
        description=(
            "Traffic-sign classification for the German Traffic Sign Recognition "
            "Benchmark (GTSRB), 43 classes.\n\n"
            "Served by a PyTorch model loaded from a self-describing checkpoint: the "
            "architecture, input size, normalisation statistics and class labels all "
            "come from the checkpoint, so serving cannot drift from training.\n\n"
            "Uploaded images are validated by decoding them, are never written to "
            "disk, and are never included in logs."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.gtsrb = AppState(resolved)
    app.state.injected_predictor = predictor

    add_request_logging(app)
    install_metrics(app, app.state.gtsrb)
    app.include_router(router)

    _install_error_handlers(app)
    return app


def _default_config_paths() -> list[Path]:
    """Locate the shipped config layers relative to the installed package."""
    candidates = [
        Path.cwd() / "configs",
        Path(__file__).resolve().parents[3] / "configs",
    ]
    for directory in candidates:
        if (directory / "data.yaml").exists():
            return [directory / name for name in DEFAULT_CONFIG_LAYERS]
    # Fall back to schema defaults rather than raising: a container image may ship
    # only the package, and env vars can supply the paths.
    return []


def _install_error_handlers(app: FastAPI) -> None:
    """Register handlers that render every failure through ``error_payload``.

    Without these, FastAPI's own handlers would return a different body shape for
    validation errors, and an unhandled exception would return a bare 500. Neither
    exposes a stack trace by default, but a consistent shape is what lets a client
    parse errors without special-casing.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request.state.error_code = "validation_error"
        return JSONResponse(
            content=error_payload(
                "validation_error", _format_validation_detail(exc), _safe_request_id(request)
            ),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "http_error")
        request.state.error_code = code
        detail = exc.detail if isinstance(exc.detail, str) else "Request could not be processed"
        return JSONResponse(
            content=error_payload(code, detail, _safe_request_id(request)),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = _safe_request_id(request)
        # The traceback goes to the log, not to the client.
        logger.error(
            "unhandled error",
            exc_info=exc,
            extra={"request_id": request_id, "path": request.url.path, "error": str(exc)},
        )
        request.state.error_code = "internal_error"
        return JSONResponse(
            content=error_payload(
                "internal_error",
                "An internal error occurred. Quote the request id when reporting it.",
                request_id,
            ),
            status_code=500,
        )


def _format_validation_detail(exc: RequestValidationError) -> str:
    """Summarise a validation failure without echoing the submitted values.

    Echoing input is how a request body ends up in a log or an error message; the
    field locations and the reason are enough to fix a malformed request.
    """
    problems = []
    for error in exc.errors()[:5]:
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        problems.append(f"{location or 'request'}: {error.get('msg', 'invalid')}")
    return "; ".join(problems) or "Request validation failed"


def _safe_request_id(request: Request) -> str | None:
    """Return the request id if the middleware has already run."""
    try:
        return get_request_id(request)
    except Exception:  # pragma: no cover - only during very early failures
        return None


def build_app_from_env() -> FastAPI:
    """Build the app from the environment, for ``uvicorn --factory``.

    Configure logging here rather than at import time, so the process that starts
    the server controls its own logging.
    """
    config = load_config(_default_config_paths())
    configure_logging(config.api.log_level)
    return create_app(config)


def load_state(app: FastAPI) -> AppState:
    """Return the app state (convenience for tests and tooling)."""
    return get_state_from_app(app)


def get_state_from_app(app: FastAPI) -> AppState:
    """Return the app state without needing a request."""
    state: Any = getattr(app.state, "gtsrb", None)
    if not isinstance(state, AppState):  # pragma: no cover - guards a wiring mistake
        raise RuntimeError("Application was not built by create_app()")
    return state


__all__ = ["DEFAULT_CONFIG_LAYERS", "build_app_from_env", "create_app", "lifespan", "load_state"]
