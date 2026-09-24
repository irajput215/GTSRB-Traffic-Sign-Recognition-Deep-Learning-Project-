"""Configuration package.

``load_config`` is the single entry point. Nothing else in the codebase should
read a YAML file or an environment variable directly.
"""

from __future__ import annotations

from gtsrb.config.labels import (
    CLASS_CATEGORIES,
    CLASS_CATEGORY_NAMES,
    CLASS_NAMES,
    CLASS_SHORT_NAMES,
    NUM_CLASSES,
    ClassCategory,
    class_category,
    class_name,
    class_short_name,
    is_valid_class_id,
    validate_label_space,
)
from gtsrb.config.loader import (
    ConfigError,
    apply_overrides,
    deep_merge,
    load_config,
    parse_override,
)
from gtsrb.config.schema import (
    ApiConfig,
    AugmentationConfig,
    BalanceConfig,
    CheckpointConfig,
    CompactCNNConfig,
    DataConfig,
    EarlyStoppingConfig,
    InferenceConfig,
    MLPConfig,
    ModelConfig,
    ModelName,
    OptimizerConfig,
    ProjectConfig,
    ResNet50Config,
    SchedulerConfig,
    TrackingConfig,
    TrainingConfig,
)
from gtsrb.config.settings import Settings, apply_env_overrides

__all__ = [
    "CLASS_CATEGORIES",
    "CLASS_CATEGORY_NAMES",
    "CLASS_NAMES",
    "CLASS_SHORT_NAMES",
    "NUM_CLASSES",
    "ApiConfig",
    "AugmentationConfig",
    "BalanceConfig",
    "CheckpointConfig",
    "ClassCategory",
    "CompactCNNConfig",
    "ConfigError",
    "DataConfig",
    "EarlyStoppingConfig",
    "InferenceConfig",
    "MLPConfig",
    "ModelConfig",
    "ModelName",
    "OptimizerConfig",
    "ProjectConfig",
    "ResNet50Config",
    "SchedulerConfig",
    "Settings",
    "TrackingConfig",
    "TrainingConfig",
    "apply_env_overrides",
    "apply_overrides",
    "class_category",
    "class_name",
    "class_short_name",
    "deep_merge",
    "is_valid_class_id",
    "load_config",
    "parse_override",
    "validate_label_space",
]
