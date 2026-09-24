"""Integration tests for the FastAPI inference service.

These run the real application through FastAPI's test client — real routing, real
middleware, real error handlers, real multipart parsing — with a predictor injected
so no GTSRB download is needed. The only thing that is not real is the model
weights, which are randomly initialised; every assertion is about the service's
behaviour, not about accuracy.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from gtsrb.config.schema import ProjectConfig
from gtsrb.models import build_metadata, build_model, save_checkpoint

pytestmark = [pytest.mark.integration]

fastapi = pytest.importorskip("fastapi", reason="FastAPI is an optional extra")
from fastapi.testclient import TestClient  # noqa: E402

from gtsrb.api import create_app  # noqa: E402
from gtsrb.api.dependencies import AppState  # noqa: E402
from gtsrb.inference import Predictor  # noqa: E402
from tests.conftest import make_sign_image  # noqa: E402


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "output_dir": str(tmp_path / "artifacts"),
            "data": {"image_size": 32},
            "model": {"name": "compact_cnn"},
            "inference": {
                "checkpoint_path": str(tmp_path / "best.pt"),
                "device": "cpu",
                "top_k": 5,
                "max_upload_bytes": 2_000_000,
            },
            "api": {"metrics_enabled": True, "warmup_on_startup": True, "title": "Test API"},
        }
    )


@pytest.fixture
def checkpoint(config: ProjectConfig) -> Path:
    model = build_model(config.model, image_size=config.data.image_size)
    path = Path(config.inference.checkpoint_path)
    save_checkpoint(
        path,
        model,
        build_metadata(config, metrics={"val_macro_f1": 0.9, "val_accuracy": 0.95}, epoch=5),
    )
    return path


@pytest.fixture
def predictor(config: ProjectConfig, checkpoint: Path) -> Predictor:
    return Predictor.from_config(config, checkpoint)


@pytest.fixture
def client(config: ProjectConfig, predictor: Predictor) -> Iterator[TestClient]:
    app = create_app(config, predictor=predictor)
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health and readiness
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_is_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["uptime_seconds"] >= 0.0
        assert payload["version"]

    def test_health_does_not_require_the_model(self, config: ProjectConfig) -> None:
        """Liveness must not depend on the model, or a healthy process gets restarted."""
        app = create_app(config)
        state: AppState = app.state.gtsrb
        state.predictor = None
        with TestClient(app) as test_client:
            assert test_client.get("/health").status_code == 200


class TestReadiness:
    def test_ready_when_the_model_is_loaded(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ready"
        assert payload["model_loaded"] is True
        assert payload["model_name"] == "compact_cnn"
        assert payload["device"] == "cpu"

    def test_not_ready_when_the_model_is_absent(self, config: ProjectConfig) -> None:
        app = create_app(config)
        state: AppState = app.state.gtsrb
        state.predictor = None
        state.startup_error = "simulated load failure"
        with TestClient(app) as test_client:
            response = test_client.get("/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "not_ready"

    def test_predict_is_503_when_not_ready(self, config: ProjectConfig) -> None:
        """503, not 500: the caller should retry, not assume a bad request."""
        app = create_app(config)
        app.state.gtsrb.predictor = None
        with TestClient(app) as test_client:
            response = test_client.post(
                "/predict",
                files={"file": ("sign.png", png_bytes(make_sign_image(32)), "image/png")},
            )
            assert response.status_code == 503
            assert response.json()["error"] == "model_unavailable"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class TestPredict:
    def test_returns_the_documented_contract(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("sign.png", png_bytes(make_sign_image(64, class_id=14)), "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        for key in (
            "class_id",
            "class_name",
            "family",
            "confidence",
            "top_predictions",
            "latency_ms",
            "model_name",
            "model_version",
        ):
            assert key in payload, key
        assert 0 <= payload["class_id"] < 43
        assert payload["class_name"]
        assert 0.0 <= payload["confidence"] <= 1.0

    def test_top_predictions_are_ranked(self, client: TestClient) -> None:
        payload = client.post(
            "/predict", files={"file": ("sign.png", png_bytes(make_sign_image(64)), "image/png")}
        ).json()
        assert len(payload["top_predictions"]) == 5
        confidences = [entry["confidence"] for entry in payload["top_predictions"]]
        assert confidences == sorted(confidences, reverse=True)
        assert payload["top_predictions"][0]["class_id"] == payload["class_id"]
        for entry in payload["top_predictions"]:
            assert set(entry) == {"class_id", "class_name", "family", "confidence"}

    def test_accepts_jpeg(self, client: TestClient) -> None:
        buffer = io.BytesIO()
        make_sign_image(64).save(buffer, format="JPEG")
        response = client.post(
            "/predict", files={"file": ("sign.jpg", buffer.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200

    def test_original_size_is_reported(self, client: TestClient) -> None:
        payload = client.post(
            "/predict", files={"file": ("sign.png", png_bytes(make_sign_image(96)), "image/png")}
        ).json()
        assert payload["original_size"] == [96, 96]

    def test_request_id_header_is_returned(self, client: TestClient) -> None:
        response = client.post(
            "/predict", files={"file": ("sign.png", png_bytes(make_sign_image(64)), "image/png")}
        )
        assert response.headers.get("x-request-id")

    def test_inbound_request_id_is_echoed(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("sign.png", png_bytes(make_sign_image(64)), "image/png")},
            headers={"x-request-id": "test-correlation-id"},
        )
        assert response.headers["x-request-id"] == "test-correlation-id"

    def test_does_not_log_the_filename(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="gtsrb.api.request"):
            client.post(
                "/predict",
                files={
                    "file": ("secret-path/sign.png", png_bytes(make_sign_image(64)), "image/png")
                },
            )
        assert "secret-path" not in caplog.text


class TestPredictErrors:
    def test_non_image_is_400(self, client: TestClient) -> None:
        response = client.post(
            "/predict", files={"file": ("notes.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        payload = response.json()
        assert payload["error"] == "invalid_image"
        assert payload["request_id"]

    def test_empty_file_is_400(self, client: TestClient) -> None:
        response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
        assert response.status_code == 400

    def test_oversized_file_is_413(self, config: ProjectConfig, checkpoint: Path) -> None:
        # The byte limit is baked into the preprocessor when the predictor is built,
        # so the predictor has to be created under the tighter limit for this to bind.
        # In production the app builds the predictor from the same config at startup,
        # so the two always agree.
        small_limit = config.model_copy(
            update={"inference": config.inference.model_copy(update={"max_upload_bytes": 100})}
        )
        small_predictor = Predictor.from_config(small_limit, checkpoint)
        app = create_app(small_limit, predictor=small_predictor)
        with TestClient(app) as test_client:
            response = test_client.post(
                "/predict", files={"file": ("big.png", b"x" * 500, "image/png")}
            )
        assert response.status_code == 413
        assert response.json()["error"] == "payload_too_large"

    def test_tiny_image_is_400(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("tiny.png", png_bytes(Image.new("RGB", (3, 3))), "image/png")},
        )
        assert response.status_code == 400
        assert "at least" in response.json()["detail"]

    def test_missing_file_field_is_422(self, client: TestClient) -> None:
        response = client.post("/predict")
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"

    def test_error_bodies_have_a_consistent_shape(self, client: TestClient) -> None:
        """Every failure, wherever it is raised, has the same parseable body."""
        failures = (
            client.post("/predict", files={"file": ("x.txt", b"nope", "text/plain")}),
            client.post("/predict"),
            client.get("/unknown-route"),
            client.get("/predict"),
        )
        for response in failures:
            assert response.status_code >= 400
            payload = response.json()
            assert set(payload) >= {"error", "detail"}, response.status_code
            assert isinstance(payload["error"], str) and payload["error"]

    def test_get_on_predict_is_405(self, client: TestClient) -> None:
        response = client.get("/predict")
        assert response.status_code == 405
        assert response.json()["error"] == "method_not_allowed"

    def test_unknown_route_is_404(self, client: TestClient) -> None:
        response = client.get("/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"] == "not_found"

    def test_internal_errors_do_not_leak_details(
        self, config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash must return a clean body, with the traceback only in the log."""

        class Exploding:
            model_name = "boom"
            device = "cpu"
            checkpoint_path = Path("/tmp/boom.pt")

            def predict_bytes(self, payload: bytes) -> None:
                raise RuntimeError("internal detail that must not reach the client")

        app = create_app(config)
        app.state.gtsrb.predictor = Exploding()
        # A 500 raised inside a route must not escape the client as a traceback.
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post(
                "/predict",
                files={"file": ("sign.png", png_bytes(make_sign_image(64)), "image/png")},
            )
        assert response.status_code == 500
        payload = response.json()
        assert payload["error"] == "internal_error"
        assert "internal detail" not in json.dumps(payload)
        assert "Traceback" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# Model info and metrics
# ---------------------------------------------------------------------------
class TestModelInfo:
    def test_describes_the_served_model(self, client: TestClient) -> None:
        payload = client.get("/model-info").json()
        assert payload["model_name"] == "compact_cnn"
        assert payload["num_classes"] == 43
        assert payload["image_size"] == 32
        assert payload["trained_epoch"] == 5
        assert payload["parameters"] > 0
        assert payload["validation_metrics"]["val_macro_f1"] == pytest.approx(0.9)

    def test_is_503_without_a_model(self, config: ProjectConfig) -> None:
        app = create_app(config)
        app.state.gtsrb.predictor = None
        with TestClient(app) as test_client:
            assert test_client.get("/model-info").status_code == 503


class TestMetrics:
    def test_prometheus_endpoint_exposes_the_instruments(self, client: TestClient) -> None:
        client.post(
            "/predict", files={"file": ("sign.png", png_bytes(make_sign_image(64)), "image/png")}
        )
        body = client.get("/metrics").text
        assert "gtsrb_requests_total" in body
        assert "gtsrb_prediction_latency_ms" in body

    def test_prediction_is_counted_by_class(self, client: TestClient) -> None:
        payload = client.post(
            "/predict", files={"file": ("sign.png", png_bytes(make_sign_image(64)), "image/png")}
        ).json()
        summary = client.get("/metrics-summary").json()
        assert str(payload["class_id"]) in summary["predictions_total"]
        assert summary["prediction_latency_ms"]["count"] >= 1
        assert summary["requests_total"]

    def test_each_failed_request_is_counted_exactly_once(self, client: TestClient) -> None:
        """A rejection produced by the route must not also be counted by the middleware."""
        before = client.get("/metrics-summary").json()["errors_total"]
        client.post("/predict", files={"file": ("x.txt", b"nope", "text/plain")})
        after = client.get("/metrics-summary").json()["errors_total"]
        assert after - before == 1

    def test_error_kind_is_the_specific_one(self, client: TestClient) -> None:
        client.post("/predict", files={"file": ("x.txt", b"nope", "text/plain")})
        body = client.get("/metrics").text
        # The route knows more than the status code does, and that is what is labelled.
        assert 'kind="invalid_image"' in body
        assert 'kind="client_error"' not in body

    def test_framework_errors_are_labelled_from_the_status(self, client: TestClient) -> None:
        client.get("/unknown-route")
        assert 'kind="not_found"' in client.get("/metrics").text

    def test_unhandled_errors_are_counted_as_internal(self, config: ProjectConfig) -> None:
        class Exploding:
            model_name = "boom"
            device = "cpu"
            checkpoint_path = Path("/tmp/boom.pt")

            def predict_bytes(self, payload: bytes) -> None:
                raise RuntimeError("boom")

        app = create_app(config)
        app.state.gtsrb.predictor = Exploding()
        with TestClient(app, raise_server_exceptions=False) as test_client:
            test_client.post(
                "/predict",
                files={"file": ("sign.png", png_bytes(make_sign_image(64)), "image/png")},
            )
            assert 'kind="internal_error"' in test_client.get("/metrics").text

    def test_metrics_can_be_disabled(self, config: ProjectConfig, predictor: Predictor) -> None:
        disabled = config.model_copy(
            update={"api": config.api.model_copy(update={"metrics_enabled": False})}
        )
        app = create_app(disabled, predictor=predictor)
        with TestClient(app) as test_client:
            response = test_client.get("/metrics")
            assert response.status_code == 200
            assert "disabled" in response.text


# ---------------------------------------------------------------------------
# Documentation and app metadata
# ---------------------------------------------------------------------------
class TestOpenApi:
    def test_schema_is_served(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "Test API"
        assert set(schema["paths"]) >= {
            "/predict",
            "/health",
            "/ready",
            "/model-info",
            "/metrics",
        }

    def test_predict_is_documented_with_responses_and_examples(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        predict = schema["paths"]["/predict"]["post"]
        assert predict["summary"]
        assert predict["description"]
        assert set(predict["responses"]) >= {"200", "400", "413", "422", "503"}
        assert predict["responses"]["400"]["content"]["application/json"]["example"]["error"]
        assert "requestBody" in predict

    def test_response_schema_has_field_descriptions(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        prediction = schema["components"]["schemas"]["PredictionResponse"]
        assert prediction["properties"]["confidence"]["description"]
        assert prediction["properties"]["top_predictions"]["description"]
        assert prediction["example"]["class_name"]

    def test_swagger_ui_is_available(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200


class TestAppFactory:
    def test_app_can_be_built_without_a_model(self, config: ProjectConfig) -> None:
        """Importing or building the app must not require a checkpoint."""
        app = create_app(config)
        assert isinstance(app.state.gtsrb, AppState)

    def test_lifespan_survives_a_missing_checkpoint(self, config: ProjectConfig) -> None:
        """A missing model must leave /health up and /ready at 503, not crash."""
        missing = config.model_copy(
            update={
                "inference": config.inference.model_copy(
                    update={"checkpoint_path": Path("/nonexistent/best.pt")}
                )
            }
        )
        app = create_app(missing)
        with TestClient(app) as test_client:
            assert test_client.get("/health").status_code == 200
            ready = test_client.get("/ready")
            assert ready.status_code == 503
            assert app.state.gtsrb.startup_error
