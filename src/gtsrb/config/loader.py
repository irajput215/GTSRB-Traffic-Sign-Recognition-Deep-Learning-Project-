"""Configuration loading: YAML files, deep merging and CLI-style overrides.

Precedence, lowest to highest:

1. schema defaults (``gtsrb.config.schema``)
2. YAML files, in the order given on the command line
3. environment variables (``gtsrb.config.settings``)
4. ``key=value`` override strings passed on the command line

Keeping the order explicit matters: it is the difference between "the config file
says 30 epochs" and "the run actually used 30 epochs" when a shell profile
happens to export an override.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from gtsrb.config.schema import ProjectConfig
from gtsrb.config.settings import apply_env_overrides


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded, merged or validated."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(raw).__name__}")
    return raw


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Mappings are merged key-by-key; every other type (including lists) is
    replaced wholesale, because merging two lists element-wise is almost never
    what someone means when they override one.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def parse_override(override: str) -> tuple[list[str], Any]:
    """Parse a ``a.b.c=value`` override string.

    The value is parsed as YAML, so ``training.epochs=5`` yields an ``int``,
    ``model.name=mlp`` a ``str`` and ``['a']`` a list, without needing a type
    annotation on the command line.

    Raises:
        ConfigError: if the override is not of the form ``key=value``.
    """
    key, separator, raw_value = override.partition("=")
    if not separator:
        raise ConfigError(f"Override {override!r} is not of the form key=value")
    key = key.strip()
    if not key:
        raise ConfigError(f"Override {override!r} has an empty key")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse value in override {override!r}: {exc}") from exc
    return key.split("."), value


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key=value`` overrides to a nested config mapping."""
    result = dict(data)
    for override in overrides:
        path, value = parse_override(override)
        cursor: dict[str, Any] = result
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = value
    return result


def load_config(
    paths: list[Path] | None = None,
    overrides: list[str] | None = None,
    *,
    use_env: bool = True,
) -> ProjectConfig:
    """Load, merge and validate the project configuration.

    Args:
        paths: YAML files to merge, in increasing order of precedence.
        overrides: ``key=value`` strings applied last.
        use_env: apply ``GTSRB_*`` environment overrides.

    Raises:
        ConfigError: if a file is missing or malformed, or the merged result
            fails validation.
    """
    merged: dict[str, Any] = {}
    for path in paths or []:
        merged = deep_merge(merged, _read_yaml(path))

    try:
        config = ProjectConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source="YAML configuration")) from exc

    if use_env:
        config = apply_env_overrides(config)

    if overrides:
        patched = apply_overrides(config.model_dump(mode="json"), overrides)
        try:
            config = ProjectConfig.model_validate(patched)
        except ValidationError as exc:
            raise ConfigError(
                _format_validation_error(exc, source="command-line overrides")
            ) from exc

    return config


def _format_validation_error(exc: ValidationError, *, source: str) -> str:
    """Render a Pydantic error as a short, actionable message."""
    lines = [f"Invalid {source}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
