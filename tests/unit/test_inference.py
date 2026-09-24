"""Unit tests for the inference pipeline: validation, preprocessing and prediction."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gtsrb.config.schema import ProjectConfig
from gtsrb.inference import (
    MAX_DIMENSION,
    ImagePreprocessor,
    InvalidImageError,
    Predictor,
)
from gtsrb.inference.predictor import ClassScore
from gtsrb.models import CheckpointError, build_metadata, build_model, save_checkpoint
from tests.conftest import make_sign_image

pytestmark = pytest.mark.unit

MEAN = (0.3403, 0.3121, 0.3214)
STD = (0.2724, 0.2608, 0.2669)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class TestImagePreprocessor:
    def test_output_shape_and_dtype(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        prepared = preprocessor.prepare(make_sign_image(64))
        assert prepared.tensor.shape == (3, 32, 32)
        assert prepared.tensor.dtype == torch.float32

    def test_records_the_original_size_and_mode(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        prepared = preprocessor.prepare(make_sign_image(48))
        assert prepared.original_size == (48, 48)
        assert prepared.original_mode == "RGB"
        assert prepared.width == 48
        assert prepared.height == 48

    def test_normalisation_uses_the_given_statistics(self) -> None:
        """The defect this guards against is silent accuracy loss, not a crash."""
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        prepared = preprocessor.prepare(make_sign_image(32, background=(0, 0, 0)))
        # A pure-black image normalises to -mean/std.
        assert float(prepared.tensor[0, 0, 0]) == pytest.approx(-MEAN[0] / STD[0], rel=1e-4)

    def test_greyscale_is_converted_not_rejected(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        greyscale = make_sign_image(48).convert("L")
        prepared = preprocessor.prepare(greyscale)
        assert prepared.tensor.shape == (3, 32, 32)
        assert prepared.original_mode == "L"

    def test_non_square_images_are_resized(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        prepared = preprocessor.prepare(Image.new("RGB", (200, 100)))
        assert prepared.tensor.shape == (3, 32, 32)
        assert prepared.original_size == (200, 100)

    def test_too_small_images_are_rejected(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD, min_dimension=8)
        with pytest.raises(InvalidImageError, match="at least 8 pixels"):
            preprocessor.prepare(Image.new("RGB", (4, 4)))

    def test_oversized_images_are_rejected(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        with pytest.raises(InvalidImageError, match=f"exceed {MAX_DIMENSION}"):
            preprocessor.prepare(Image.new("RGB", (MAX_DIMENSION + 1, 16)))

    def test_non_image_input_is_rejected(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        with pytest.raises(InvalidImageError, match="Expected a PIL image"):
            preprocessor.prepare("not an image")  # type: ignore[arg-type]

    def test_constructor_validates_its_arguments(self) -> None:
        with pytest.raises(ValueError, match="image_size must be positive"):
            ImagePreprocessor(image_size=0, mean=MEAN, std=STD)
        with pytest.raises(ValueError, match="std must be strictly positive"):
            ImagePreprocessor(image_size=32, mean=MEAN, std=(0.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="min_dimension must be positive"):
            ImagePreprocessor(image_size=32, mean=MEAN, std=STD, min_dimension=0)


class TestPrepareBytes:
    def test_valid_png_round_trip(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        prepared = preprocessor.prepare_bytes(png_bytes(make_sign_image(64)))
        assert prepared.tensor.shape == (3, 32, 32)
        assert prepared.original_size == (64, 64)

    def test_empty_payload_is_rejected(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        with pytest.raises(InvalidImageError, match="empty"):
            preprocessor.prepare_bytes(b"")

    def test_oversized_payload_is_rejected_before_decoding(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD, max_upload_bytes=100)
        with pytest.raises(InvalidImageError, match="exceeds the 100-byte limit"):
            preprocessor.prepare_bytes(b"x" * 200)

    def test_non_image_bytes_are_rejected(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        with pytest.raises(InvalidImageError, match="Could not decode image"):
            preprocessor.prepare_bytes(b"this is not an image")

    def test_truncated_png_is_rejected(self) -> None:
        """A lazily-opened truncated file must not produce a half-decoded tensor."""
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        payload = png_bytes(make_sign_image(64))
        with pytest.raises(InvalidImageError, match="Could not decode image"):
            preprocessor.prepare_bytes(payload[: len(payload) // 3])

    def test_decompression_bomb_is_rejected(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        # A tiny PNG that declares an enormous canvas.
        huge = Image.new("RGB", (20000, 20000))
        payload = png_bytes(huge)
        with pytest.raises(InvalidImageError, match="too large to process safely"):
            preprocessor.prepare_bytes(payload)

    def test_prepare_path_reads_a_file(self, tmp_path: Path) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        path = tmp_path / "sign.png"
        make_sign_image(48).save(path)
        assert preprocessor.prepare_path(path).tensor.shape == (3, 32, 32)

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        with pytest.raises(InvalidImageError, match="Image file not found"):
            preprocessor.prepare_path(tmp_path / "absent.png")

    def test_batch_stacks_prepared_images(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        prepared = [preprocessor.prepare(make_sign_image(48)) for _ in range(3)]
        batch = preprocessor.batch(prepared)
        assert batch.shape == (3, 3, 32, 32)

    def test_empty_batch_is_rejected(self) -> None:
        preprocessor = ImagePreprocessor(image_size=32, mean=MEAN, std=STD)
        with pytest.raises(InvalidImageError, match="empty list"):
            preprocessor.batch([])


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    config = ProjectConfig.model_validate(
        {"data": {"image_size": 32}, "model": {"name": "compact_cnn"}}
    )
    model = build_model(config.model, image_size=32)
    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        model,
        build_metadata(config, metrics={"val_macro_f1": 0.9, "val_accuracy": 0.95}, epoch=7),
    )
    return path


@pytest.fixture
def predictor(checkpoint_path: Path, tmp_path: Path) -> Predictor:
    config = ProjectConfig.model_validate(
        {
            "output_dir": str(tmp_path),
            "data": {"image_size": 32},
            "inference": {"checkpoint_path": str(checkpoint_path), "device": "cpu", "top_k": 5},
        }
    )
    return Predictor.from_config(config)


class TestPredictorSetup:
    def test_loads_the_checkpoint_and_exposes_its_metadata(
        self, predictor: Predictor, checkpoint_path: Path
    ) -> None:
        assert predictor.model_name == "compact_cnn"
        assert predictor.num_classes == 43
        assert predictor.image_size == 32
        assert predictor.trained_epoch == 7
        assert len(predictor.class_names) == 43
        assert predictor.checkpoint_path == checkpoint_path

    def test_uses_the_checkpoints_image_size_not_the_configs(
        self, checkpoint_path: Path, tmp_path: Path
    ) -> None:
        """Serving must not diverge from training because a config changed."""
        config = ProjectConfig.model_validate(
            {
                "output_dir": str(tmp_path),
                "data": {"image_size": 224},  # deliberately different
                "inference": {"checkpoint_path": str(checkpoint_path), "device": "cpu"},
            }
        )
        predictor = Predictor.from_config(config)
        assert predictor.image_size == 32
        assert predictor.predict(make_sign_image(64)).class_id < 43

    def test_uses_the_checkpoints_normalisation_statistics(
        self, checkpoint_path: Path, tmp_path: Path
    ) -> None:
        config = ProjectConfig.model_validate(
            {
                "output_dir": str(tmp_path),
                "data": {"image_size": 32, "normalize_mean": [0.5, 0.5, 0.5]},
                "inference": {"checkpoint_path": str(checkpoint_path), "device": "cpu"},
            }
        )
        predictor = Predictor.from_config(config)
        assert predictor.preprocessor.mean == (0.3403, 0.3121, 0.3214)

    def test_model_is_in_eval_mode(self, predictor: Predictor) -> None:
        assert predictor.model.training is False

    def test_top_k_is_capped_at_the_label_space(
        self, checkpoint_path: Path, tmp_path: Path
    ) -> None:
        config = ProjectConfig.model_validate(
            {
                "output_dir": str(tmp_path),
                "data": {"image_size": 32},
                "inference": {"checkpoint_path": str(checkpoint_path), "top_k": 43},
            }
        )
        assert Predictor.from_config(config).top_k == 43

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        config = ProjectConfig.model_validate(
            {"inference": {"checkpoint_path": str(tmp_path / "absent.pt")}}
        )
        with pytest.raises(CheckpointError, match="Checkpoint not found"):
            Predictor.from_config(config)

    def test_info_describes_the_served_model(self, predictor: Predictor) -> None:
        info = predictor.info()
        assert info["model_name"] == "compact_cnn"
        assert info["num_classes"] == 43
        assert info["image_size"] == 32
        assert info["trained_epoch"] == 7
        assert info["parameters"] > 0
        assert info["validation_metrics"]["val_macro_f1"] == pytest.approx(0.9)
        assert isinstance(info["library_versions"], dict)

    def test_warmup_runs_a_forward_pass(self, predictor: Predictor) -> None:
        assert predictor.warmup() >= 0.0


class TestPredictorPrediction:
    def test_predicts_a_probability_distribution(self, predictor: Predictor) -> None:
        prediction = predictor.predict(make_sign_image(64, class_id=14))
        assert 0.0 <= prediction.confidence <= 1.0
        assert 0 <= prediction.class_id < 43
        assert prediction.class_name
        assert prediction.predicted.family

    def test_top_k_is_ranked_and_normalised(self, predictor: Predictor) -> None:
        prediction = predictor.predict(make_sign_image(64, class_id=14))
        assert len(prediction.top_k) == 5
        confidences = [score.confidence for score in prediction.top_k]
        assert confidences == sorted(confidences, reverse=True)
        assert sum(confidences) <= 1.0 + 1e-6
        assert prediction.predicted == prediction.top_k[0]

    def test_all_43_classes_sum_to_one(self, predictor: Predictor) -> None:
        config = predictor.config.model_copy(
            update={"inference": predictor.config.inference.model_copy(update={"top_k": 43})}
        )
        full = Predictor(config, predictor.checkpoint_path)
        total = sum(score.confidence for score in full.predict(make_sign_image(64)).top_k)
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_latency_is_recorded(self, predictor: Predictor) -> None:
        assert predictor.predict(make_sign_image(64)).latency_ms >= 0.0

    def test_predict_bytes_matches_predict(self, predictor: Predictor) -> None:
        image = make_sign_image(64, class_id=3)
        from_image = predictor.predict(image)
        from_bytes = predictor.predict_bytes(png_bytes(image))
        assert from_image.class_id == from_bytes.class_id
        assert from_image.confidence == pytest.approx(from_bytes.confidence, abs=1e-6)

    def test_predict_path(self, predictor: Predictor, tmp_path: Path) -> None:
        path = tmp_path / "sign.png"
        make_sign_image(48).save(path)
        assert predictor.predict_path(path).class_id < 43

    def test_predict_array_uint8(self, predictor: Predictor) -> None:
        array = np.asarray(make_sign_image(64, class_id=14), dtype=np.uint8)
        assert predictor.predict_array(array).class_id < 43

    def test_predict_array_rejects_wrong_shape(self, predictor: Predictor) -> None:
        with pytest.raises(InvalidImageError, match=r"\(H, W, 3\)"):
            predictor.predict_array(np.zeros((64, 64), dtype=np.uint8))

    def test_predict_array_rejects_wrong_dtype(self, predictor: Predictor) -> None:
        with pytest.raises(InvalidImageError, match="Expected uint8 or float pixels"):
            predictor.predict_array(np.zeros((64, 64, 3), dtype=np.int32))

    def test_invalid_image_is_rejected(self, predictor: Predictor) -> None:
        with pytest.raises(InvalidImageError):
            predictor.predict(Image.new("RGB", (2, 2)))


class TestPredictorBatch:
    def test_batch_matches_single_predictions(self, predictor: Predictor) -> None:
        images = [make_sign_image(64, class_id=c) for c in range(6)]
        single = [predictor.predict(image).class_id for image in images]
        batched = [prediction.class_id for prediction in predictor.predict_batch(images)]
        assert single == batched

    def test_batch_is_a_single_forward_pass(self, predictor: Predictor) -> None:
        calls = 0
        original = predictor.predict_tensor

        def counting_predict_tensor(batch: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return original(batch)

        predictor.predict_tensor = counting_predict_tensor  # type: ignore[method-assign]
        predictor.predict_batch([make_sign_image(64) for _ in range(8)])
        assert calls == 1

    def test_batch_reports_per_image_latency(self, predictor: Predictor) -> None:
        predictions = predictor.predict_batch([make_sign_image(64) for _ in range(4)])
        assert all(prediction.latency_ms >= 0.0 for prediction in predictions)

    def test_empty_batch_is_rejected(self, predictor: Predictor) -> None:
        with pytest.raises(InvalidImageError, match="empty batch"):
            predictor.predict_batch([])

    def test_batch_bytes_and_paths(self, predictor: Predictor, tmp_path: Path) -> None:
        images = [make_sign_image(48, class_id=c) for c in range(3)]
        from_bytes = predictor.predict_batch_bytes([png_bytes(image) for image in images])
        paths = []
        for index, image in enumerate(images):
            path = tmp_path / f"sign_{index}.png"
            image.save(path)
            paths.append(path)
        from_paths = predictor.predict_batch_paths(paths)
        assert len(from_bytes) == len(from_paths) == 3

    def test_predict_tensor_validates_shape(self, predictor: Predictor) -> None:
        with pytest.raises(ValueError, match="4D batch"):
            predictor.predict_tensor(torch.randn(3, 32, 32))
        with pytest.raises(ValueError, match="3-channel"):
            predictor.predict_tensor(torch.randn(1, 1, 32, 32))


class TestPredictionSerialisation:
    def test_as_dict_matches_the_documented_response_contract(self, predictor: Predictor) -> None:
        payload = predictor.predict(make_sign_image(64, class_id=14)).as_dict()
        for key in (
            "class_id",
            "class_name",
            "confidence",
            "top_predictions",
            "model_name",
            "model_version",
            "latency_ms",
        ):
            assert key in payload
        assert isinstance(payload["top_predictions"], list)
        assert set(payload["top_predictions"][0]) == {
            "class_id",
            "class_name",
            "family",
            "confidence",
        }
        assert payload["class_id"] == payload["top_predictions"][0]["class_id"]

    def test_confidence_is_rounded_for_transport(self, predictor: Predictor) -> None:
        payload = predictor.predict(make_sign_image(64)).as_dict()
        assert len(str(payload["confidence"]).split(".")[-1]) <= 6

    def test_include_all_scores_adds_the_original_size(self, predictor: Predictor) -> None:
        prediction = predictor.predict(make_sign_image(64))
        assert "original_size" not in prediction.as_dict()
        assert prediction.as_dict(include_all_scores=True)["original_size"] == [64, 64]

    def test_class_score_helpers(self) -> None:
        score = ClassScore(class_id=14, confidence=0.98)
        assert score.class_name == "Stop"
        assert score.short_name == "Stop"
        assert score.family == "priority"
        assert score.as_dict()["confidence"] == 0.98
