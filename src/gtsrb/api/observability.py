"""Observability: request logging and Prometheus metrics.

**Logging** records what is needed to operate the service — method, path, status,
duration, request id — and nothing about the request body. Uploaded images are user
content: they are not logged, not written to disk and not included in error
messages. The only image-derived facts recorded anywhere are the predicted class and
its confidence, which are the service's own output.

**Metrics** live in :mod:`gtsrb.api.metrics`. Anything more elaborate would be
monitoring infrastructure this project does not have, and would not be exercised.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from gtsrb.api.dependencies import AppState, RequestContext
from gtsrb.api.metrics import Metrics
from gtsrb.runtime import get_logger

#: Fallback error kind for a status code, used when a handler did not set a more
#: specific one on ``request.state.error_code``.
_STATUS_TO_KIND = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    500: "internal_error",
    503: "model_unavailable",
}

logger = get_logger("api.request")


def add_request_logging(app: FastAPI) -> None:
    """Attach the request-logging middleware."""

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Honour an inbound correlation id so a request can be traced across
        # services, and mint one otherwise.
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.context = RequestContext(
            request_id=request_id,
            client=request.client.host if request.client else None,
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            # The exception handler will render the response; this records the fact
            # that the request failed and how long it took to fail.
            logger.exception(
                "request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(elapsed_ms, 3),
                },
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed_ms, 3),
                "client": request.state.context.client,
            },
        )
        return response


def add_metrics_middleware(app: FastAPI, metrics: Metrics) -> None:
    """Attach the middleware that feeds the request counter."""

    @app.middleware("http")
    async def record_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # An unhandled exception propagates straight past this middleware to
        # Starlette's ServerErrorMiddleware, so the status-code path below never runs
        # for it. Catching and re-raising is the only way a 500 gets counted at all.
        try:
            response = await call_next(request)
        except Exception:
            metrics.observe_request(request.url.path, request.method, 500)
            metrics.observe_error("internal_error")
            raise

        # Use the route template rather than the raw path so /predict does not
        # become one series per request.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        metrics.observe_request(endpoint, request.method, response.status_code)

        if response.status_code >= 400:
            # Exactly one error per failed request. Handlers set the most specific
            # kind they know on ``request.state.error_code``; otherwise it is derived
            # from the status. Counting here *and* in the handler would double-count
            # every rejection the route produced itself.
            kind = getattr(request.state, "error_code", None) or _STATUS_TO_KIND.get(
                response.status_code, "http_error"
            )
            metrics.observe_error(kind)
        return response


def install_metrics(app: FastAPI, state: AppState) -> Metrics:
    """Create the metrics registry and attach its middleware."""
    metrics = Metrics(enabled=state.config.api.metrics_enabled)
    state.metrics = metrics
    if metrics.enabled:
        add_metrics_middleware(app, metrics)
    return metrics


__all__ = ["Metrics", "add_metrics_middleware", "add_request_logging", "install_metrics"]
