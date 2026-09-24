"""Application state and dependency wiring.

The predictor is expensive to build and must be shared across requests, so it
lives on ``app.state``. It is deliberately **not** a module-level global: a global
would make the app untestable without a real checkpoint and would silently share
one model between two differently-configured apps.

Access goes through :func:`get_predictor`, which raises a 503 rather than a 500 when
the model is not available. That distinction matters to a load balancer: "not ready
yet" should be retried or routed elsewhere, "broken request" should not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request, status

from gtsrb.config.schema import ProjectConfig
from gtsrb.inference import Predictor
from gtsrb.runtime import get_logger

logger = get_logger("api.dependencies")


class AppState:
    """Mutable runtime state held on the FastAPI application.

    Attributes:
        config: the resolved project configuration.
        predictor: the loaded model, or ``None`` until startup completes.
        started_at: monotonic timestamp used for the uptime report.
        startup_error: why loading failed, if it did. Retained so ``/ready`` can
            explain itself instead of just returning 503.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.predictor: Predictor | None = None
        self.started_at: float = time.monotonic()
        self.startup_error: str | None = None
        self.warmup_ms: float | None = None
        # Set by gtsrb.api.observability.install_metrics. Typed as Any because
        # importing Metrics here would create a cycle: metrics imports this module
        # for the AppState it reports on.
        self.metrics: Any = None

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def is_ready(self) -> bool:
        return self.predictor is not None

    def load_predictor(self, predictor: Predictor | None = None) -> Predictor:
        """Attach a predictor, warming it up unless one was injected.

        An injected predictor is assumed to be warm, which keeps tests fast and lets
        a caller share a model across apps.
        """
        if predictor is not None:
            self.predictor = predictor
            self.startup_error = None
            logger.info("using an injected predictor", extra={"model": predictor.model_name})
            return predictor

        try:
            loaded = Predictor.from_config(self.config)
            self.warmup_ms = loaded.warmup()
            self.predictor = loaded
            self.startup_error = None
            return loaded
        except Exception as exc:
            self.startup_error = str(exc)
            logger.error("model failed to load: %s", exc)
            raise

    def clear(self) -> None:
        self.predictor = None


@dataclass
class RequestContext:
    """Per-request data threaded through the middleware and the handlers."""

    request_id: str
    client: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def get_state(request: Request) -> AppState:
    """Return the application state.

    Raises:
        RuntimeError: if the app was not built by :func:`gtsrb.api.main.create_app`.
    """
    state = getattr(request.app.state, "gtsrb", None)
    if not isinstance(state, AppState):  # pragma: no cover - guards a wiring mistake
        raise RuntimeError("Application state is missing; build the app with create_app()")
    return state


def get_predictor(request: Request) -> Predictor:
    """Return the loaded predictor.

    Raises:
        HTTPException: 503 when the model is not loaded. A 503 tells a caller to
            retry or route elsewhere; a 500 would suggest the request was bad.
    """
    state = get_state(request)
    if state.predictor is None:
        detail = "Model is not loaded yet"
        if state.startup_error:
            detail = f"Model failed to load: {state.startup_error}"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return state.predictor


def get_request_id(request: Request) -> str:
    """Return the request id assigned by the logging middleware."""
    context: RequestContext | None = getattr(request.state, "context", None)
    if context is not None:
        return context.request_id
    return request.headers.get("x-request-id", "unknown")


__all__ = [
    "AppState",
    "RequestContext",
    "get_predictor",
    "get_request_id",
    "get_state",
]
