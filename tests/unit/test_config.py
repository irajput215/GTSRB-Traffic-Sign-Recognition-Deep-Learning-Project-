"""Unit tests for configuration loading, validation and the label space."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gtsrb.config import (
    CLASS_CATEGORIES,
    CLASS_NAMES,
    CLASS_SHORT_NAMES,
    NUM_CLASSES,
    ConfigError,
    ProjectConfig,
    class_category,
    class_name,
    class_short_name,
    deep_merge,
    load_config,
    parse_override,
    validate_label_space,
)
from gtsrb.config.loader import apply_overrides

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
class TestLabelSpace:
    def test_label_tables_are_aligned(self) -> None:
        assert len(CLASS_NAMES) == NUM_CLASSES
        assert len(CLASS_SHORT_NAMES) == NUM_CLASSES
        assert len(CLASS_CATEGORIES) == NUM_CLASSES

    def test_labels_are_unique(self) -> None:
        assert len(set(CLASS_NAMES)) == NUM_CLASSES
        assert len(set(CLASS_SHORT_NAMES)) == NUM_CLASSES

    @pytest.mark.parametrize(
        ("class_id", "expected"),
        [
            (0, "Speed limit (20km/h)"),
            (14, "Stop"),
            (27, "Pedestrians"),
            (42, "End of no passing by vehicles over 3.5 metric tons"),
        ],
    )
    def test_known_class_names(self, class_id: int, expected: str) -> None:
        assert class_name(class_id) == expected

    def test_short_name_and_category_agree_with_index(self) -> None:
        for class_id in range(NUM_CLASSES):
            assert class_short_name(class_id) == CLASS_SHORT_NAMES[class_id]
            assert class_category(class_id) == CLASS_CATEGORIES[class_id]

    @pytest.mark.parametrize("bad_id", [-1, 43, 1000])
    def test_out_of_range_ids_raise(self, bad_id: int) -> None:
        with pytest.raises(ValueError, match="Unknown GTSRB class id"):
            class_name(bad_id)

    def test_validate_label_space_accepts_canonical_order(self) -> None:
        validate_label_space(list(CLASS_NAMES))

    def test_validate_label_space_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="Expected 43 class labels, got 2"):
            validate_label_space(["Stop", "Yield"])

    def test_validate_label_space_rejects_reordered_labels(self) -> None:
        shuffled = list(CLASS_NAMES)
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
        with pytest.raises(ValueError, match="canonical GTSRB ordering"):
            validate_label_space(shuffled)

    def test_validate_label_space_rejects_non_sequence(self) -> None:
        with pytest.raises(ValueError, match="Expected a list or tuple"):
            validate_label_space(42)


# ---------------------------------------------------------------------------
# Merging and overrides
# ---------------------------------------------------------------------------
class TestDeepMerge:
    def test_nested_mappings_are_merged(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"c": 3}}
        assert deep_merge(base, override) == {"a": {"b": 1, "c": 3}}

    def test_lists_are_replaced_not_concatenated(self) -> None:
        assert deep_merge({"a": [1, 2, 3]}, {"a": [9]}) == {"a": [9]}

    def test_original_inputs_are_not_mutated(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestOverrides:
    def test_values_are_parsed_as_yaml(self) -> None:
        assert parse_override("training.epochs=5") == (["training", "epochs"], 5)
        assert parse_override("model.name=mlp") == (["model", "name"], "mlp")
        assert parse_override("training.deterministic=false") == (
            ["training", "deterministic"],
            False,
        )
        assert parse_override("data.normalize_mean=[0.1,0.2,0.3]")[1] == [0.1, 0.2, 0.3]

    def test_missing_equals_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="key=value"):
            parse_override("training.epochs")

    def test_empty_key_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="empty key"):
            parse_override("=5")

    def test_apply_overrides_creates_missing_branches(self) -> None:
        result = apply_overrides({}, ["a.b.c=1"])
        assert result == {"a": {"b": {"c": 1}}}

    def test_apply_overrides_replaces_scalar_with_mapping_when_needed(self) -> None:
        result = apply_overrides({"a": 1}, ["a.b=2"])
        assert result == {"a": {"b": 2}}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
class TestSchemaValidation:
    def test_tracking_defaults_to_a_local_sqlite_backend(self) -> None:
        """MLflow 3.x rejects the legacy file: store, so SQLite is the default."""
        tracking = ProjectConfig().tracking
        assert tracking.enabled is False
        assert tracking.tracking_uri.startswith("sqlite:///")

    def test_defaults_match_original_normalisation_statistics(self) -> None:
        config = ProjectConfig()
        assert config.data.normalize_mean == (0.3403, 0.3121, 0.3214)
        assert config.data.normalize_std == (0.2724, 0.2608, 0.2669)

    def test_unknown_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ProjectConfig.model_validate({"data": {"image_sizes": 64}})

    def test_config_is_immutable(self) -> None:
        config = ProjectConfig()
        with pytest.raises(ValidationError):
            config.training.epochs = 99  # type: ignore[misc]

    def test_zero_std_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="strictly positive"):
            ProjectConfig.model_validate({"data": {"normalize_std": [0.0, 1.0, 1.0]}})

    def test_wrong_class_count_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="num_classes must be 43"):
            ProjectConfig.model_validate({"data": {"num_classes": 10}})

    def test_persistent_workers_requires_workers(self) -> None:
        with pytest.raises(ValidationError, match="persistent_workers=True requires"):
            ProjectConfig.model_validate({"data": {"num_workers": 0, "persistent_workers": True}})

    def test_scale_range_is_ordered(self) -> None:
        with pytest.raises(ValidationError, match="scale_min"):
            ProjectConfig.model_validate(
                {"data": {"augmentation": {"scale_min": 2.0, "scale_max": 1.0}}}
            )

    def test_mlp_hidden_sizes_and_dropouts_must_align(self) -> None:
        with pytest.raises(ValidationError, match="correspond one-to-one"):
            ProjectConfig.model_validate(
                {"model": {"mlp": {"hidden_sizes": [64, 32], "dropouts": [0.1]}}}
            )

    def test_unknown_resnet_block_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown ResNet-50 blocks"):
            ProjectConfig.model_validate({"model": {"resnet50": {"trainable_blocks": ["layer9"]}}})

    def test_monitor_and_mode_must_agree(self) -> None:
        with pytest.raises(ValidationError, match="inconsistent with monitor"):
            ProjectConfig.model_validate(
                {"training": {"checkpoint": {"monitor": "val_loss", "mode": "max"}}}
            )

    def test_register_model_requires_log_model(self) -> None:
        with pytest.raises(ValidationError, match="register_model=True requires"):
            ProjectConfig.model_validate({"tracking": {"register_model": True, "log_model": False}})

    def test_nesterov_only_for_sgd(self) -> None:
        with pytest.raises(ValidationError, match="nesterov"):
            ProjectConfig.model_validate(
                {"training": {"optimizer": {"name": "adam", "nesterov": True}}}
            )

    def test_top_k_within_label_space(self) -> None:
        with pytest.raises(ValidationError, match="top_k must be <= 43"):
            ProjectConfig.model_validate({"inference": {"top_k": 50}})

    def test_architecture_property_selects_active_block(self) -> None:
        config = ProjectConfig.model_validate({"model": {"name": "resnet50"}})
        assert isinstance(config.model.architecture, type(config.model.resnet50))
        cnn = ProjectConfig.model_validate({"model": {"name": "compact_cnn"}})
        assert isinstance(cnn.model.architecture, type(cnn.model.compact_cnn))


class TestFlatParams:
    def test_flat_params_are_scalar_and_dotted(self) -> None:
        flat = ProjectConfig().flat_params()
        assert flat["training.optimizer.lr"] == pytest.approx(1e-3)
        assert flat["model.name"] == "compact_cnn"
        assert flat["data.image_size"] == 64
        assert all(not isinstance(value, dict) for value in flat.values())

    def test_flat_params_cover_every_field(self) -> None:
        flat = ProjectConfig().flat_params()
        assert len(flat) > 50
        assert "tracking.enabled" in flat


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_shipped_configs_load_and_validate(self, config_dir: Path) -> None:
        config = load_config(
            [
                config_dir / "data.yaml",
                config_dir / "model.yaml",
                config_dir / "train.yaml",
                config_dir / "api.yaml",
            ],
            use_env=False,
        )
        assert config.data.image_size == 64
        assert config.model.name == "compact_cnn"
        assert config.training.epochs == 30
        assert config.api.port == 8000

    def test_api_layer_overrides_nothing_in_training(self, config_dir: Path) -> None:
        train_only = load_config(
            [config_dir / "data.yaml", config_dir / "model.yaml", config_dir / "train.yaml"],
            use_env=False,
        )
        with_api = load_config(
            [
                config_dir / "data.yaml",
                config_dir / "model.yaml",
                config_dir / "train.yaml",
                config_dir / "api.yaml",
            ],
            use_env=False,
        )
        assert with_api.training.epochs == train_only.training.epochs

    def test_later_files_take_precedence(self, tmp_path: Path) -> None:
        first = tmp_path / "first.yaml"
        second = tmp_path / "second.yaml"
        first.write_text(yaml.safe_dump({"training": {"epochs": 5}}), encoding="utf-8")
        second.write_text(yaml.safe_dump({"training": {"epochs": 9}}), encoding="utf-8")
        assert load_config([first, second], use_env=False).training.epochs == 9

    def test_cli_overrides_beat_files(self, config_dir: Path) -> None:
        config = load_config(
            [config_dir / "train.yaml"],
            ["training.epochs=1", "model.name=mlp"],
            use_env=False,
        )
        assert config.training.epochs == 1
        assert config.model.name == "mlp"

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config([tmp_path / "nope.yaml"], use_env=False)

    def test_non_mapping_yaml_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_config([bad], use_env=False)

    def test_malformed_yaml_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.yaml"
        bad.write_text("training: {epochs: [1, 2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config([bad], use_env=False)

    def test_empty_file_is_treated_as_empty_mapping(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        assert load_config([empty], use_env=False).training.epochs == 30

    def test_invalid_yaml_value_reports_config_location(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"training": {"epochs": -3}}), encoding="utf-8")
        with pytest.raises(ConfigError, match=r"training\.epochs"):
            load_config([bad], use_env=False)

    def test_invalid_override_reports_source(self, config_dir: Path) -> None:
        with pytest.raises(ConfigError, match="command-line overrides"):
            load_config([config_dir / "train.yaml"], ["training.epochs=-1"], use_env=False)
