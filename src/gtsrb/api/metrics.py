"""Prometheus metrics.

Four instruments, chosen because each answers an operational question:

* ``gtsrb_requests_total`` — request rate, by endpoint and status.
* ``gtsrb_prediction_latency_ms`` — the distribution that decides whether the
  service meets a latency budget.
* ``gtsrb_errors_total`` — error rate, by kind.
* ``gtsrb_predictions_total`` — predicted-class distribution, which is how drift
  shows up first: a class that suddenly stops appearing is a signal long before
  accuracy can be measured.

A single :class:`Metrics` instance is attached to the app rather than using the
module-level default registry, so two apps in one process (as in the tests) do not
register the same metric twice and raise.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


class Metrics:
    """Prometheus instruments, created lazily so the API works without the extra.

    A single instance is attached to the app rather than using the module-level
    default registry, so two apps in one process (as in the tests) do not register
    the same metric twice and raise.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._registry = None
        self._requests = None
        self._latency = None
        self._errors = None
        self._predictions = None
        self._counts: dict[str, int] = {}
        self._class_counts: dict[int, int] = {}
        self._error_count = 0
        self._latency_samples: list[float] = []

        if not self.enabled:
            return

        self._registry = CollectorRegistry()
        self._requests = Counter(
            "gtsrb_requests_total",
            "HTTP requests handled, by endpoint and status code.",
            ["endpoint", "method", "status"],
            registry=self._registry,
        )
        self._latency = Histogram(
            "gtsrb_prediction_latency_ms",
            "Model inference latency in milliseconds.",
            buckets=(1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000),
            registry=self._registry,
        )
        self._errors = Counter(
            "gtsrb_errors_total",
            "Failed requests, by error kind.",
            ["kind"],
            registry=self._registry,
        )
        self._predictions = Counter(
            "gtsrb_predictions_total",
            "Predictions served, by predicted class id.",
            ["class_id"],
            registry=self._registry,
        )

    # -- recording ---------------------------------------------------------
    def observe_request(self, endpoint: str, method: str, status_code: int) -> None:
        key = f"{method} {endpoint}:{status_code}"
        self._counts[key] = self._counts.get(key, 0) + 1
        if self._requests is not None:
            self._requests.labels(endpoint=endpoint, method=method, status=str(status_code)).inc()

    def observe_latency(self, latency_ms: float) -> None:
        self._latency_samples.append(latency_ms)
        if self._latency is not None:
            self._latency.observe(latency_ms)

    def observe_error(self, kind: str) -> None:
        self._error_count += 1
        if self._errors is not None:
            self._errors.labels(kind=kind).inc()

    def observe_prediction(self, class_id: int) -> None:
        self._class_counts[class_id] = self._class_counts.get(class_id, 0) + 1
        if self._predictions is not None:
            self._predictions.labels(class_id=str(class_id)).inc()

    # -- reporting ---------------------------------------------------------
    def render(self) -> bytes:
        """Render the Prometheus exposition format."""
        if self._registry is None:
            return b"# metrics are disabled\n"
        return generate_latest(self._registry)

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST

    def summary(self) -> dict[str, object]:
        """A JSON-friendly view of the counters, for quick inspection."""
        samples = sorted(self._latency_samples)
        latency: dict[str, float] = {}
        if samples:
            latency = {
                "count": float(len(samples)),
                "mean": sum(samples) / len(samples),
                "min": samples[0],
                "max": samples[-1],
                "p50": samples[len(samples) // 2],
                "p95": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
            }
        return {
            "requests_total": dict(self._counts),
            "predictions_total": {str(k): v for k, v in sorted(self._class_counts.items())},
            "errors_total": self._error_count,
            "prediction_latency_ms": latency,
        }
