"""Validated configuration schema.

Every tunable lives in one of the Pydantic models below. Configuration is
validated once, at the boundary, so that no downstream module has to defend
against a nonsensical value — and so that a typo in a YAML file fails loudly at
startup instead of silently training the wrong thing.

The models are frozen: config is treated as immutable data that is passed down,
never mutated in place. This is also what makes it safe to serialise the exact
config into a checkpoint and reproduce a run later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gtsrb.config.labels import NUM_CLASSES

ModelName = Literal["compact_cnn", "mlp", "resnet50"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]

PositiveInt = Annotated[int, Field(gt=0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class _Base(BaseModel):
    """Shared base: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)


class AugmentationConfig(_Base):
    """Training-time augmentation.

    Defaults reproduce the transforms used in the original project (rotation,
    translation, brightness/contrast jitter) and add two knobs that were absent
    and are cheap: scaling and random erasing. ``random_erasing_p`` defaults to
    ``0`` so the default profile stays equivalent to the recorded run.
    """

    enabled: bool = True
    rotation_degrees: float = Field(default=15.0, ge=0.0, le=180.0)
    translate: UnitFloat = 0.1
    scale_min: float = Field(default=1.0, gt=0.0)
    scale_max: float = Field(default=1.0, gt=0.0)
    shear_degrees: float = Field(default=0.0, ge=0.0, le=90.0)
    brightness: UnitFloat = 0.2
    contrast: UnitFloat = 0.2
    saturation: UnitFloat = 0.0
    hue: float = Field(default=0.0, ge=0.0, le=0.5)
    random_erasing_p: UnitFloat = 0.0

    @model_validator(mode="after")
    def _check_scale_range(self) -> Self:
        if self.scale_min > self.scale_max:
            raise ValueError(
                f"scale_min ({self.scale_min}) must not exceed scale_max ({self.scale_max})"
            )
        return self


class BalanceConfig(_Base):
    """Class-balance strategy for the training split.

    ``mode='weighted'`` draws samples with a ``WeightedRandomSampler`` whose
    per-class weight is ``count ** -power``. ``power=0`` is uniform sampling,
    ``power=1`` is fully class-balanced. This replaces the original project's
    materialised oversampling to a per-class floor: the balance is equivalent,
    memory stays flat, and each epoch draws a different sample of the minority
    classes instead of repeating one fixed list.
    """

    mode: Literal["none", "weighted"] = "weighted"
    power: float = Field(default=1.0, ge=0.0, le=1.0)


class DataConfig(_Base):
    """Dataset location, input geometry and split policy."""

    root: Path = Path("data/raw")
    image_size: PositiveInt = 64
    num_classes: int = Field(default=NUM_CLASSES, ge=1)
    val_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    split_seed: int = 43
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    persistent_workers: bool = False
    normalize_mean: tuple[float, float, float] = (0.3403, 0.3121, 0.3214)
    normalize_std: tuple[float, float, float] = (0.2724, 0.2608, 0.2669)
    split_manifest: Path = Path("artifacts/splits/gtsrb_split.json")
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)

    @field_validator("normalize_std")
    @classmethod
    def _std_must_be_positive(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(s <= 0 for s in value):
            raise ValueError(f"normalize_std must be strictly positive, got {value}")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.num_classes != NUM_CLASSES:
            raise ValueError(
                f"num_classes must be {NUM_CLASSES} for GTSRB, got {self.num_classes}. "
                "Changing this means changing gtsrb.config.labels too."
            )
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0")
        return self


class CompactCNNConfig(_Base):
    """Geometry of the from-scratch CNN used in the original project."""

    channels: tuple[int, int, int] = (64, 128, 256)
    head_hidden: PositiveInt = 512
    conv_dropout: UnitFloat = 0.25
    head_dropout: UnitFloat = 0.5


class MLPConfig(_Base):
    """Fully-connected baseline. Exists to show what spatial structure buys."""

    hidden_sizes: tuple[int, ...] = (2048, 1024, 512, 256)
    dropouts: tuple[float, ...] = (0.3, 0.3, 0.3, 0.2)

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        if len(self.hidden_sizes) != len(self.dropouts):
            raise ValueError(
                f"hidden_sizes has {len(self.hidden_sizes)} entries but dropouts has "
                f"{len(self.dropouts)}; they must correspond one-to-one"
            )
        if any(not 0.0 <= d < 1.0 for d in self.dropouts):
            raise ValueError(f"dropouts must be in [0, 1), got {self.dropouts}")
        return self


class ResNet50Config(_Base):
    """Transfer-learning settings for the ImageNet-pretrained ResNet-50 branch."""

    weights: Literal["imagenet", "none"] = "imagenet"
    freeze_backbone: bool = True
    trainable_blocks: tuple[str, ...] = ("layer4",)
    head_hidden: PositiveInt = 512
    head_dropout: UnitFloat = 0.5
    head_activation: Literal["relu", "gelu"] = "gelu"

    @field_validator("trainable_blocks")
    @classmethod
    def _known_blocks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        allowed = {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"Unknown ResNet-50 blocks {sorted(unknown)}; allowed: {sorted(allowed)}"
            )
        return value


class ModelConfig(_Base):
    """Selects the architecture and carries the per-architecture settings."""

    name: ModelName = "compact_cnn"
    num_classes: int = Field(default=NUM_CLASSES, ge=1)
    compact_cnn: CompactCNNConfig = Field(default_factory=CompactCNNConfig)
    mlp: MLPConfig = Field(default_factory=MLPConfig)
    resnet50: ResNet50Config = Field(default_factory=ResNet50Config)

    @property
    def architecture(self) -> CompactCNNConfig | MLPConfig | ResNet50Config:
        """Return the settings block for the selected architecture."""
        if self.name == "compact_cnn":
            return self.compact_cnn
        if self.name == "mlp":
            return self.mlp
        return self.resnet50


class OptimizerConfig(_Base):
    name: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    nesterov: bool = False

    @model_validator(mode="after")
    def _momentum_only_for_sgd(self) -> Self:
        if self.nesterov and self.name != "sgd":
            raise ValueError("nesterov=True is only meaningful for the 'sgd' optimizer")
        return self


class SchedulerConfig(_Base):
    """Learning-rate schedule.

    ``reduce_on_plateau`` keeps parity with the original run. ``cosine`` and
    ``step`` are provided because they are the schedules the original report
    itself names as future work, and having them configurable makes that
    comparison cheap to run rather than a rewrite.
    """

    name: Literal["none", "reduce_on_plateau", "cosine", "step"] = "reduce_on_plateau"
    monitor: Literal["val_loss", "val_accuracy", "val_macro_f1"] = "val_loss"
    factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    patience: int = Field(default=3, ge=0)
    min_lr: float = Field(default=1e-6, gt=0.0)
    step_size: PositiveInt = 10
    gamma: float = Field(default=0.1, gt=0.0, lt=1.0)


class EarlyStoppingConfig(_Base):
    """Stop training when the monitored metric stops improving.

    Disabled is not the default: the original run trained all 30 epochs and
    selected the best checkpoint, which wasted epochs on the MLP that was still
    only at 65.8% accuracy. Early stopping makes that explicit.
    """

    enabled: bool = True
    monitor: Literal["val_loss", "val_accuracy", "val_macro_f1"] = "val_macro_f1"
    mode: Literal["min", "max"] = "max"
    patience: PositiveInt = 8
    min_delta: float = Field(default=1e-4, ge=0.0)


class CheckpointConfig(_Base):
    directory: Path = Path("artifacts/checkpoints")
    filename: str = "best.pt"
    monitor: Literal["val_loss", "val_accuracy", "val_macro_f1"] = "val_macro_f1"
    mode: Literal["min", "max"] = "max"
    save_last: bool = True


class TrainingConfig(_Base):
    """Training loop settings."""

    epochs: PositiveInt = 30
    batch_size: PositiveInt = 16
    seed: int = 43
    device: DeviceName = "auto"
    deterministic: bool = True
    amp: bool = True
    grad_clip_norm: float | None = Field(default=1.0, gt=0.0)
    loss: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: UnitFloat = 0.0
    log_every_n_steps: PositiveInt = 50
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def _check_monitor_mode_consistency(self) -> Self:
        for label, monitor, mode in (
            ("checkpoint", self.checkpoint.monitor, self.checkpoint.mode),
            ("early_stopping", self.early_stopping.monitor, self.early_stopping.mode),
        ):
            expected = "min" if monitor == "val_loss" else "max"
            if mode != expected:
                raise ValueError(
                    f"{label}.mode={mode!r} is inconsistent with monitor={monitor!r}; "
                    f"expected mode={expected!r}"
                )
        return self


class TrackingConfig(_Base):
    """MLflow experiment tracking.

    Disabled by default and pointed at a local SQLite database when enabled, so the
    project trains and evaluates with no tracking server and no network access.

    ``sqlite:///`` rather than ``file:./mlruns``: MLflow 3.x put the filesystem
    tracking backend into maintenance mode and raises on it unless
    ``MLFLOW_ALLOW_FILE_STORE=true`` is set. A local SQLite file keeps every
    property that mattered about the file store — one file, no server, no
    network — without opting out of a supported backend.
    """

    enabled: bool = False
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "gtsrb-traffic-sign-recognition"
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    log_model: bool = True
    register_model: bool = False
    registered_model_name: str = "gtsrb-traffic-sign-classifier"
    log_artifacts: bool = True

    @model_validator(mode="after")
    def _register_requires_log_model(self) -> Self:
        if self.register_model and not self.log_model:
            raise ValueError("register_model=True requires log_model=True")
        return self


class InferenceConfig(_Base):
    checkpoint_path: Path = Path("artifacts/checkpoints/best.pt")
    device: DeviceName = "auto"
    top_k: PositiveInt = 5
    max_upload_bytes: PositiveInt = 5_000_000

    @model_validator(mode="after")
    def _top_k_within_label_space(self) -> Self:
        if self.top_k > NUM_CLASSES:
            raise ValueError(f"top_k must be <= {NUM_CLASSES}, got {self.top_k}")
        return self


class ApiConfig(_Base):
    # Binds all interfaces: the service runs inside a container.
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    title: str = "GTSRB Traffic Sign Recognition API"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    metrics_enabled: bool = True
    warmup_on_startup: bool = True


class ProjectConfig(_Base):
    """Root configuration object."""

    run_name: str | None = None
    output_dir: Path = Path("artifacts")
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    def flat_params(self) -> dict[str, Any]:
        """Flatten the config into dotted ``key -> scalar`` pairs.

        Used for experiment tracking and for checkpoint metadata, where nested
        dictionaries are awkward and only scalars are meaningful to compare.
        """
        flat: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, BaseModel):
                for field_name in type(value).model_fields:
                    child = f"{prefix}.{field_name}" if prefix else field_name
                    walk(child, getattr(value, field_name))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flat[prefix] = value
            else:
                flat[prefix] = str(value)

        walk("", self)
        return flat
