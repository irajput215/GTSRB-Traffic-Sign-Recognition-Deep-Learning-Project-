#!/usr/bin/env python
"""Validate every shipped configuration file.

Run as a pre-flight check and in CI. A YAML file that no longer matches the
schema is a broken build, and this is the cheapest place to catch it: before a
training run starts, not three hours in.

Usage:
    python scripts/validate_configs.py [configs_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from gtsrb.config import ConfigError, load_config

#: Config layers that together form a complete, runnable configuration.
LAYERS: tuple[str, ...] = ("data.yaml", "model.yaml", "train.yaml", "api.yaml")


def validate(configs_dir: Path) -> list[str]:
    """Validate the shipped layers and return a list of failure messages."""
    failures: list[str] = []

    missing = [name for name in LAYERS if not (configs_dir / name).exists()]
    if missing:
        return [f"missing config layer(s): {', '.join(missing)} in {configs_dir}"]

    try:
        config = load_config([configs_dir / name for name in LAYERS], use_env=False)
    except ConfigError as exc:
        return [str(exc)]

    # Cross-layer invariants that no single file can check on its own.
    if config.inference.top_k > config.model.num_classes:
        failures.append("inference.top_k exceeds the configured number of classes")
    if config.tracking.register_model and not config.tracking.log_model:
        failures.append("tracking.register_model requires tracking.log_model")
    if config.data.image_size < 16:
        failures.append(f"data.image_size={config.data.image_size} is too small to train on")

    print(
        f"OK  {configs_dir}: model={config.model.name} "
        f"image_size={config.data.image_size} epochs={config.training.epochs} "
        f"batch_size={config.training.batch_size} tracking={config.tracking.enabled}"
    )
    return failures


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    configs_dir = Path(args[0]) if args else Path(__file__).resolve().parents[1] / "configs"

    failures = validate(configs_dir)
    for message in failures:
        print(f"FAIL {message}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
