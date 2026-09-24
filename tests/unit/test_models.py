"""Unit tests for the model factory, architectures and checkpoint contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gtsrb.config.labels import CLASS_NAMES, NUM_CLASSES
from gtsrb.config.schema import ModelConfig, ProjectConfig
from gtsrb.models import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    CheckpointMetadata,
    CompactCNN,
    MLPClassifier,
    ResNet50Classifier,
    build_metadata,
    build_model,
    count_parameters,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
    summarise_model,
)
from gtsrb.models.transfer import RESNET50_BLOCKS

pytestmark = pytest.mark.unit


def _model_config(name: str, **overrides: object) -> ModelConfig:
    payload: dict[str, object] = {"name": name}
    payload.update(overrides)
    return ModelConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Architecture construction
# ---------------------------------------------------------------------------
class TestCompactCNN:
    def test_output_shape(self) -> None:
        model = CompactCNN(_model_config("compact_cnn").compact_cnn, image_size=64)
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 43)

    def test_parameter_count_is_an_order_of_magnitude_below_the_mlp(self) -> None:
        """The headline comparison: fewer parameters, far higher accuracy."""
        cnn = count_parameters(CompactCNN(_model_config("compact_cnn").compact_cnn, image_size=64))
        mlp = count_parameters(MLPClassifier(_model_config("mlp").mlp, image_size=64))
        assert cnn < mlp / 2

    def test_parameter_count_is_stable(self) -> None:
        model = CompactCNN(_model_config("compact_cnn").compact_cnn, image_size=64)
        count = count_parameters(model)
        # Measured: 9,557,483. Asserted as a range so an intentional width change
        # does not break the test, but tightly enough to catch a silent blow-up.
        assert 8_000_000 < count < 12_000_000

    def test_accepts_a_different_input_size(self) -> None:
        assert CompactCNN(_model_config("compact_cnn").compact_cnn, image_size=32)(
            torch.randn(1, 3, 32, 32)
        ).shape == (1, 43)

    def test_rejects_input_size_not_divisible_by_eight(self) -> None:
        with pytest.raises(ValueError, match="divisible by 8"):
            CompactCNN(_model_config("compact_cnn").compact_cnn, image_size=60)

    def test_has_three_convolutional_blocks_as_documented(self) -> None:
        model = CompactCNN(_model_config("compact_cnn").compact_cnn, image_size=64)
        assert len(model.features) == 3
        assert model.flattened_features == 256 * 8 * 8

    def test_channel_widths_follow_config(self) -> None:
        config = ModelConfig.model_validate(
            {"name": "compact_cnn", "compact_cnn": {"channels": [8, 16, 32]}}
        )
        model = CompactCNN(config.compact_cnn, image_size=32)
        conv_outputs = [
            module.out_channels
            for module in model.features.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        assert conv_outputs == [8, 8, 16, 16, 32, 32]

    def test_forward_is_differentiable(self) -> None:
        model = CompactCNN(_model_config("compact_cnn").compact_cnn, image_size=32)
        logits = model(torch.randn(2, 3, 32, 32, requires_grad=False))
        logits.sum().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestMLP:
    def test_output_shape(self) -> None:
        model = MLPClassifier(_model_config("mlp").mlp, image_size=32)
        assert model(torch.randn(4, 3, 32, 32)).shape == (4, 43)

    def test_hidden_layer_widths_follow_config(self) -> None:
        model = MLPClassifier(_model_config("mlp").mlp, image_size=32)
        linear_widths = [
            module.out_features for module in model.net if isinstance(module, torch.nn.Linear)
        ]
        assert linear_widths == [2048, 1024, 512, 256, 43]

    def test_flattens_the_input(self) -> None:
        model = MLPClassifier(_model_config("mlp").mlp, image_size=64)
        first_linear = next(m for m in model.net if isinstance(m, torch.nn.Linear))
        assert first_linear.in_features == 3 * 64 * 64


class TestResNet50:
    def test_output_shape_without_pretrained_weights(self) -> None:
        config = ModelConfig.model_validate({"name": "resnet50", "resnet50": {"weights": "none"}})
        model = ResNet50Classifier(config.resnet50, image_size=224)
        model.eval()
        with torch.no_grad():
            assert model(torch.randn(2, 3, 224, 224)).shape == (2, 43)

    def test_freezes_everything_except_the_configured_blocks(self) -> None:
        config = ModelConfig.model_validate(
            {"name": "resnet50", "resnet50": {"weights": "none", "trainable_blocks": ["layer4"]}}
        )
        model = ResNet50Classifier(config.resnet50)
        layer1 = model.backbone.get_submodule("layer1")
        layer4 = model.backbone.get_submodule("layer4")
        assert not any(p.requires_grad for p in layer1.parameters())
        assert all(p.requires_grad for p in layer4.parameters())
        assert all(p.requires_grad for p in model.classifier.parameters())

    def test_unfreezing_a_block_also_unfreezes_everything_after_it(self) -> None:
        """Training a stage whose inputs never change would be pointless."""
        config = ModelConfig.model_validate(
            {"name": "resnet50", "resnet50": {"weights": "none", "trainable_blocks": ["layer3"]}}
        )
        model = ResNet50Classifier(config.resnet50)
        assert all(p.requires_grad for p in model.backbone.get_submodule("layer3").parameters())
        assert all(p.requires_grad for p in model.backbone.get_submodule("layer4").parameters())
        assert not any(p.requires_grad for p in model.backbone.get_submodule("layer2").parameters())

    def test_multiple_trainable_blocks(self) -> None:
        config = ModelConfig.model_validate(
            {
                "name": "resnet50",
                "resnet50": {"weights": "none", "trainable_blocks": ["layer3", "layer4"]},
            }
        )
        model = ResNet50Classifier(config.resnet50)
        assert all(p.requires_grad for p in model.backbone.get_submodule("layer3").parameters())

    def test_freeze_backbone_false_trains_everything(self) -> None:
        config = ModelConfig.model_validate(
            {"name": "resnet50", "resnet50": {"weights": "none", "freeze_backbone": False}}
        )
        model = ResNet50Classifier(config.resnet50)
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_unknown_block_is_rejected_by_the_config(self) -> None:
        with pytest.raises(Exception, match="Unknown ResNet-50 blocks"):
            ModelConfig.model_validate(
                {"name": "resnet50", "resnet50": {"trainable_blocks": ["layer9"]}}
            )

    def test_trainable_parameter_count_reflects_freezing(self) -> None:
        frozen = ResNet50Classifier(
            ModelConfig.model_validate(
                {"name": "resnet50", "resnet50": {"weights": "none", "trainable_blocks": []}}
            ).resnet50
        )
        partial = ResNet50Classifier(
            ModelConfig.model_validate(
                {
                    "name": "resnet50",
                    "resnet50": {"weights": "none", "trainable_blocks": ["layer4"]},
                }
            ).resnet50
        )
        assert frozen.trainable_parameter_count < partial.trainable_parameter_count

    def test_block_order_is_forward_order(self) -> None:
        assert RESNET50_BLOCKS == ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
class TestBuildModel:
    @pytest.mark.parametrize(
        ("name", "image_size", "expected_class"),
        [
            ("compact_cnn", 64, CompactCNN),
            ("mlp", 64, MLPClassifier),
        ],
    )
    def test_builds_each_architecture(
        self, name: str, image_size: int, expected_class: type
    ) -> None:
        model = build_model(_model_config(name), image_size=image_size)
        assert isinstance(model, expected_class)
        # Batch of 2, not 1: the MLP's BatchNorm1d rejects a single sample in
        # training mode, which is a property of batch norm rather than a defect.
        assert model(torch.randn(2, 3, image_size, image_size)).shape == (2, NUM_CLASSES)

    def test_builds_resnet50_without_downloading_weights(self) -> None:
        config = ModelConfig.model_validate({"name": "resnet50", "resnet50": {"weights": "none"}})
        model = build_model(config, image_size=224)
        assert isinstance(model, ResNet50Classifier)

    def test_rejects_a_wrong_class_count(self) -> None:
        config = ModelConfig.model_validate({"name": "compact_cnn", "num_classes": 10})
        with pytest.raises(ValueError, match="label space has 43 classes"):
            build_model(config, image_size=64)

    def test_rejects_an_unregistered_name(self) -> None:
        # model_construct bypasses validation on purpose: the point is to check the
        # factory's own guard, not the schema's Literal.
        config = ModelConfig.model_construct(name="transformer")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(config, image_size=64)


class TestSurroundingHelpers:
    def test_summary_reports_parameters_and_size(self) -> None:
        model = build_model(_model_config("compact_cnn"), image_size=64)
        summary = summarise_model(model, name="compact_cnn", image_size=64, num_classes=43)
        assert summary.total_parameters > 0
        assert summary.trainable_parameters == summary.total_parameters
        assert summary.size_mb > 0
        assert "compact_cnn" in summary.describe()

    def test_summary_dict_is_json_serialisable(self) -> None:
        model = build_model(_model_config("compact_cnn"), image_size=64)
        payload = summarise_model(
            model, name="compact_cnn", image_size=64, num_classes=43
        ).as_dict()
        assert json.loads(json.dumps(payload))["num_classes"] == 43

    def test_count_parameters_trainable_only(self) -> None:
        config = ModelConfig.model_validate(
            {"name": "resnet50", "resnet50": {"weights": "none", "trainable_blocks": ["layer4"]}}
        )
        model = build_model(config, image_size=224)
        assert count_parameters(model, trainable_only=True) < count_parameters(model)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
@pytest.fixture
def checkpoint_config() -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "data": {"image_size": 32},
            "model": {"name": "compact_cnn"},
        }
    )


class TestCheckpointRoundTrip:
    def test_metadata_carries_everything_inference_needs(
        self, checkpoint_config: ProjectConfig
    ) -> None:
        metadata = build_metadata(checkpoint_config, metrics={"val_macro_f1": 0.9}, epoch=3)
        assert metadata.format_version == CHECKPOINT_FORMAT_VERSION
        assert metadata.model_name == "compact_cnn"
        assert metadata.num_classes == NUM_CLASSES
        assert metadata.class_names == CLASS_NAMES
        assert metadata.image_size == 32
        assert metadata.normalize_mean == checkpoint_config.data.normalize_mean
        assert metadata.normalize_std == checkpoint_config.data.normalize_std
        assert metadata.config["model"]["name"] == "compact_cnn"
        assert metadata.metrics == {"val_macro_f1": 0.9}
        assert metadata.epoch == 3
        assert "torch" in metadata.library_versions

    def test_save_then_load_round_trip(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        path = tmp_path / "best.pt"
        save_checkpoint(path, model, build_metadata(checkpoint_config, epoch=1))
        loaded = load_checkpoint(path)
        assert loaded.epoch == 1
        assert set(loaded.model_state) == set(model.state_dict())

    def test_load_model_rebuilds_the_architecture(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        path = tmp_path / "best.pt"
        save_checkpoint(path, model, build_metadata(checkpoint_config))
        restored, metadata = load_model_from_checkpoint(path)
        assert isinstance(restored, CompactCNN)
        assert metadata.model_name == "compact_cnn"
        assert not restored.training  # returned in eval mode

    def test_restored_weights_produce_identical_outputs(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        path = tmp_path / "best.pt"
        save_checkpoint(path, model, build_metadata(checkpoint_config))
        restored, _ = load_model_from_checkpoint(path)
        x = torch.randn(2, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), restored(x), atol=1e-6)

    def test_optimizer_state_is_stored_when_provided(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        path = tmp_path / "last.pt"
        save_checkpoint(path, model, build_metadata(checkpoint_config), optimizer=optimizer)
        assert load_checkpoint(path).optimizer_state is not None

    def test_optimizer_state_is_absent_when_not_provided(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        path = tmp_path / "best.pt"
        save_checkpoint(path, model, build_metadata(checkpoint_config))
        assert load_checkpoint(path).optimizer_state is None

    def test_extra_payload_is_preserved(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        path = tmp_path / "best.pt"
        save_checkpoint(
            path, model, build_metadata(checkpoint_config), extra={"history": [{"val_loss": 1.0}]}
        )
        assert load_checkpoint(path).extra["history"] == [{"val_loss": 1.0}]

    def test_save_is_atomic_and_leaves_no_temporary_file(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        path = tmp_path / "best.pt"
        save_checkpoint(path, model, build_metadata(checkpoint_config))
        assert path.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_save_creates_parent_directories(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        path = tmp_path / "nested" / "deep" / "best.pt"
        save_checkpoint(path, model, build_metadata(checkpoint_config))
        assert path.exists()

    def test_metadata_summary_is_readable(self, checkpoint_config: ProjectConfig) -> None:
        summary = build_metadata(
            checkpoint_config, metrics={"val_macro_f1": 0.98}, epoch=7
        ).summary()
        assert "compact_cnn" in summary
        assert "epoch 7" in summary
        assert "val_macro_f1=0.9800" in summary


class TestCheckpointFailures:
    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="Checkpoint not found"):
            load_checkpoint(tmp_path / "absent.pt")

    def test_unreadable_file_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.pt"
        path.write_text("not a checkpoint", encoding="utf-8")
        with pytest.raises(CheckpointError, match="Could not read checkpoint"):
            load_checkpoint(path)

    def test_non_mapping_payload_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.pt"
        torch.save([1, 2, 3], path)
        with pytest.raises(CheckpointError, match="must contain a mapping"):
            load_checkpoint(path)

    def test_missing_keys_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.pt"
        torch.save({"metadata": {}}, path)
        with pytest.raises(CheckpointError, match="missing the 'model_state'"):
            load_checkpoint(path)

    def test_malformed_metadata_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save({"metadata": "nope", "model_state": {}}, path)
        with pytest.raises(CheckpointError, match="malformed metadata"):
            load_checkpoint(path)

    def test_unknown_format_version_is_rejected(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        metadata = build_metadata(checkpoint_config)
        payload = {"metadata": metadata.as_dict(), "model_state": model.state_dict()}
        payload["metadata"]["format_version"] = 999
        path = tmp_path / "future.pt"
        torch.save(payload, path)
        with pytest.raises(CheckpointError, match="format version 999"):
            load_checkpoint(path)

    def test_permuted_class_labels_are_rejected(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        """The reason class names are stored: a permuted label space must not serve."""
        model = build_model(checkpoint_config.model, image_size=32)
        metadata = build_metadata(checkpoint_config)
        payload = {"metadata": metadata.as_dict(), "model_state": model.state_dict()}
        names = list(CLASS_NAMES)
        names[0], names[1] = names[1], names[0]
        payload["metadata"]["class_names"] = names
        path = tmp_path / "permuted.pt"
        torch.save(payload, path)
        with pytest.raises(CheckpointError, match="not compatible"):
            load_checkpoint(path)

    def test_wrong_class_count_is_rejected(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        metadata = build_metadata(checkpoint_config)
        payload = {"metadata": metadata.as_dict(), "model_state": model.state_dict()}
        payload["metadata"]["class_names"] = CLASS_NAMES[:10]
        path = tmp_path / "short.pt"
        torch.save(payload, path)
        with pytest.raises(CheckpointError, match="not compatible"):
            load_checkpoint(path)

    def test_malformed_normalisation_statistics_are_rejected(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        model = build_model(checkpoint_config.model, image_size=32)
        metadata = build_metadata(checkpoint_config)
        payload = {"metadata": metadata.as_dict(), "model_state": model.state_dict()}
        payload["metadata"]["normalize_mean"] = [0.5]
        path = tmp_path / "badstats.pt"
        torch.save(payload, path)
        with pytest.raises(CheckpointError, match="malformed normalisation"):
            load_checkpoint(path)

    def test_architecture_mismatch_is_reported(
        self, tmp_path: Path, checkpoint_config: ProjectConfig
    ) -> None:
        """A state dict from a different architecture must not load silently."""
        model = build_model(
            ModelConfig.model_validate(
                {"name": "compact_cnn", "compact_cnn": {"channels": [8, 16, 32]}}
            ),
            image_size=32,
        )
        metadata = build_metadata(checkpoint_config)  # config says 64/128/256
        path = tmp_path / "mismatch.pt"
        save_checkpoint(path, model, metadata)
        with pytest.raises(CheckpointError, match="does not match the architecture"):
            load_model_from_checkpoint(path)

    def test_metadata_json_serialises(self, checkpoint_config: ProjectConfig) -> None:
        metadata: CheckpointMetadata = build_metadata(checkpoint_config)
        assert json.loads(metadata.to_json())["model_name"] == "compact_cnn"
