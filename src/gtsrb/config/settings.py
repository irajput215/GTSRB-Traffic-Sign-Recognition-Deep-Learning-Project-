"""Environment-driven settings.

Anything that legitimately differs between a laptop, CI and a container is an
environment variable here rather than a value baked into a YAML file: the
dataset location, the artefact directory, and the MLflow tracking URI. None of
these are secrets, but the same mechanism is what keeps credentials out of the
repository — see ``.env.example`` for the variables this project reads and where
to put them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gtsrb.config.schema import ProjectConfig


class Settings(BaseSettings):
    """Environment variables prefixed with ``GTSRB_``.

    Nested values use a double underscore, e.g. ``GTSRB_API__PORT=9000``.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTSRB_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data_root: Path | None = None
    output_dir: Path | None = None
    checkpoint_path: Path | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None
    seed: int | None = Field(default=None, ge=0)
    api_host: str | None = None
    api_port: int | None = Field(default=None, ge=1, le=65535)
    api_metrics_enabled: bool | None = None


def _assign(patch: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    """Set ``value`` at the dotted ``path`` inside ``patch``, creating branches."""
    cursor = patch
    for key in path[:-1]:
        node = cursor.get(key)
        if not isinstance(node, dict):
            node = {}
            cursor[key] = node
        cursor = node
    cursor[path[-1]] = value


def apply_env_overrides(config: ProjectConfig) -> ProjectConfig:
    """Return ``config`` with any ``GTSRB_*`` environment variables applied.

    Only variables that are actually set take effect; an unset variable leaves
    the value from the YAML layer untouched.
    """
    settings = Settings()

    # (value, destination path). ``device`` appears twice on purpose: a single
    # environment variable should not be able to leave the trainer on CUDA and
    # the inference service on CPU.
    assignments: tuple[tuple[object, tuple[str, ...]], ...] = (
        (settings.data_root, ("data", "root")),
        (settings.output_dir, ("output_dir",)),
        (settings.checkpoint_path, ("inference", "checkpoint_path")),
        (settings.device, ("training", "device")),
        (settings.device, ("inference", "device")),
        (settings.seed, ("training", "seed")),
        (settings.mlflow_tracking_uri, ("tracking", "tracking_uri")),
        (settings.mlflow_experiment_name, ("tracking", "experiment_name")),
        (settings.api_host, ("api", "host")),
        (settings.api_port, ("api", "port")),
        (settings.api_metrics_enabled, ("api", "metrics_enabled")),
        (settings.log_level, ("api", "log_level")),
    )

    patch: dict[str, Any] = {}
    for value, path in assignments:
        if value is None:
            continue
        _assign(patch, path, str(value) if isinstance(value, Path) else value)

    if not patch:
        return config

    merged = config.model_dump(mode="json")
    for key, value in patch.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            existing.update(value)
        else:
            merged[key] = value
    return ProjectConfig.model_validate(merged)
