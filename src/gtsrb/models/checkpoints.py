"""Checkpoint format.

A checkpoint is not just a ``state_dict``. It carries everything needed to rebuild
the model and to explain where the numbers came from:

* the fully resolved configuration, so ``build_model`` can reconstruct the
  architecture without the caller knowing which one it was;
* the class-label ordering, so a checkpoint trained under a different label mapping
  is rejected at load time instead of silently serving wrong names;
* the input geometry and normalisation constants, so inference preprocesses exactly
  the way training did;
* the metrics recorded at the epoch that was saved;
* the git commit and library versions.

The original project saved ``model.state_dict()`` to ``gtsrb_cnn.pth`` with no
metadata. Reconstructing inference from it required the notebook. The
``format_version`` field is what lets this format change later without a silent
misread.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gtsrb.config.labels import CLASS_NAMES, validate_label_space
from gtsrb.config.schema import ProjectConfig
from gtsrb.models.factory import build_model
from gtsrb.runtime import get_logger

logger = get_logger("models.checkpoints")

#: Bumped on an incompatible change to the payload layout.
CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be read or does not match expectations."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Everything stored alongside the weights."""

    format_version: int
    created_at: str
    model_name: str
    num_classes: int
    class_names: tuple[str, ...]
    image_size: int
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    epoch: int = 0
    git_commit: str | None = None
    library_versions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)

    def summary(self) -> str:
        metric = ", ".join(f"{key}={value:.4f}" for key, value in sorted(self.metrics.items()))
        return (
            f"{self.model_name} @ epoch {self.epoch} | {self.image_size}x{self.image_size} | "
            f"{self.num_classes} classes | {metric or 'no metrics'}"
        )


try:  # pragma: no cover - torchvision is a hard dependency, but a version string is optional
    import torchvision

    _TORCHVISION_VERSION: str | None = torchvision.__version__
except ImportError:  # pragma: no cover
    _TORCHVISION_VERSION = None


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    if _TORCHVISION_VERSION is not None:
        versions["torchvision"] = _TORCHVISION_VERSION
    return versions


def _git_commit() -> str | None:
    """Best-effort current commit hash.

    Returns ``None`` rather than raising when git is unavailable or the code is not
    in a repository: a checkpoint is still useful without it.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def build_metadata(
    config: ProjectConfig,
    *,
    metrics: dict[str, float] | None = None,
    epoch: int = 0,
) -> CheckpointMetadata:
    """Assemble checkpoint metadata from a resolved project configuration."""
    return CheckpointMetadata(
        format_version=CHECKPOINT_FORMAT_VERSION,
        created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        class_names=CLASS_NAMES,
        image_size=config.data.image_size,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std,
        config=config.model_dump(mode="json"),
        metrics=dict(metrics or {}),
        epoch=epoch,
        git_commit=_git_commit(),
        library_versions=_library_versions(),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: CheckpointMetadata,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint to ``path``.

    Optimizer and scheduler state are optional and are stored so training can
    resume. ``extra`` holds anything else a caller wants preserved (for example the
    history of per-epoch metrics).

    The write is atomic: the payload goes to a temporary file which is then
    replaced into position, so an interrupted save cannot destroy the previous
    best checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "metadata": metadata.as_dict(),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra is not None:
        payload["extra"] = extra

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

    logger.info(
        "saved checkpoint",
        extra={"path": str(path), "epoch": metadata.epoch, "model": metadata.model_name},
    )
    return path


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A checkpoint read back from disk."""

    metadata: CheckpointMetadata
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    extra: dict[str, Any]

    @property
    def epoch(self) -> int:
        return self.metadata.epoch


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> LoadedCheckpoint:
    """Read a checkpoint from disk.

    Raises:
        CheckpointError: if the file is missing, unreadable, not a checkpoint
            mapping, has an unknown ``format_version``, or its stored class labels do
            not match the canonical GTSRB ordering.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointError(
            f"Checkpoint not found: {path}. Train a model first (make train) or set "
            "GTSRB_CHECKPOINT_PATH to an existing checkpoint."
        )

    try:
        # weights_only=False: the payload includes plain-Python metadata, not only
        # tensors. The file is one this project wrote, and its path comes from
        # configuration rather than from user input.
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        # Any failure to decode is a bad checkpoint; the underlying error text is
        # preserved in the message and the cause chain.
        raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CheckpointError(
            f"Checkpoint {path} must contain a mapping, got {type(payload).__name__}"
        )
    for key in ("metadata", "model_state"):
        if key not in payload:
            raise CheckpointError(f"Checkpoint {path} is missing the {key!r} entry")

    raw_metadata = payload["metadata"]
    if not isinstance(raw_metadata, dict):
        raise CheckpointError(f"Checkpoint {path} has a malformed metadata entry")

    version = raw_metadata.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Checkpoint {path} has format version {version!r}, expected "
            f"{CHECKPOINT_FORMAT_VERSION}. Retrain, or check out the code that wrote it."
        )

    mean_raw = list(raw_metadata["normalize_mean"])
    std_raw = list(raw_metadata["normalize_std"])
    if len(mean_raw) != 3 or len(std_raw) != 3:
        raise CheckpointError(f"Checkpoint {path} has malformed normalisation statistics")

    metadata = CheckpointMetadata(
        format_version=int(version),
        created_at=str(raw_metadata.get("created_at", "unknown")),
        model_name=str(raw_metadata["model_name"]),
        num_classes=int(raw_metadata["num_classes"]),
        class_names=tuple(raw_metadata["class_names"]),
        image_size=int(raw_metadata["image_size"]),
        normalize_mean=(float(mean_raw[0]), float(mean_raw[1]), float(mean_raw[2])),
        normalize_std=(float(std_raw[0]), float(std_raw[1]), float(std_raw[2])),
        config=dict(raw_metadata["config"]),
        metrics={k: float(v) for k, v in raw_metadata.get("metrics", {}).items()},
        epoch=int(raw_metadata.get("epoch", 0)),
        git_commit=raw_metadata.get("git_commit"),
        library_versions=dict(raw_metadata.get("library_versions", {})),
    )

    # The label space check is the reason class names are stored at all: a
    # checkpoint whose labels were permuted would otherwise serve confidently wrong
    # class names.
    try:
        validate_label_space(metadata.class_names)
    except ValueError as exc:
        raise CheckpointError(f"Checkpoint {path} is not compatible: {exc}") from exc

    model_state = payload["model_state"]
    if not isinstance(model_state, dict):
        raise CheckpointError(f"Checkpoint {path} has a malformed model_state entry")

    return LoadedCheckpoint(
        metadata=metadata,
        model_state=model_state,
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        extra=dict(payload.get("extra", {})),
    )


def load_model_from_checkpoint(
    path: Path,
    *,
    device: torch.device | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, CheckpointMetadata]:
    """Rebuild a trained model from a checkpoint.

    The architecture is reconstructed from the configuration stored inside the
    checkpoint, so serving a model requires no knowledge of how it was trained
    beyond the checkpoint path.

    Args:
        path: checkpoint file.
        device: if given, the model is moved there before being returned.
        map_location: where ``torch.load`` should place tensors.
        strict: passed through to ``load_state_dict``. Leave it ``True``: a partial
            match means the checkpoint and the code disagree about the architecture.

    Returns:
        ``(model in eval mode, metadata)``.
    """
    loaded = load_checkpoint(path, map_location=map_location)
    config = ProjectConfig.model_validate(loaded.metadata.config)

    model = build_model(config.model, image_size=loaded.metadata.image_size)
    try:
        model.load_state_dict(loaded.model_state, strict=strict)
    except RuntimeError as exc:
        raise CheckpointError(
            f"Checkpoint {path} does not match the architecture rebuilt from its config "
            f"({loaded.metadata.model_name}): {exc}"
        ) from exc

    model.eval()
    if device is not None:
        model.to(device)

    return model, loaded.metadata


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "CheckpointMetadata",
    "LoadedCheckpoint",
    "build_metadata",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "save_checkpoint",
]
