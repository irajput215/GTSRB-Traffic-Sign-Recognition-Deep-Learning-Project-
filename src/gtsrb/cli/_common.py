"""Shared command-line helpers.

Every entry point loads configuration the same way and resolves the same default
config layers, so the flags behave identically across ``train``, ``evaluate``,
``predict`` and ``serve``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gtsrb.config import ProjectConfig, load_config
from gtsrb.runtime import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Config layers loaded by default, in increasing order of precedence.
DEFAULT_CONFIG_LAYERS: tuple[str, ...] = (
    "data.yaml",
    "model.yaml",
    "train.yaml",
)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags every entry point shares."""
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        default=[REPO_ROOT / "configs" / name for name in DEFAULT_CONFIG_LAYERS],
        metavar="PATH",
        help="YAML config layers, applied in the order given (default: configs/*.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, repeatable (e.g. --set training.epochs=5)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="emit structured JSON logs instead of human-readable lines",
    )


def load_project_config(args: argparse.Namespace) -> ProjectConfig:
    """Load the configuration described by parsed arguments."""
    return load_config(list(args.config), list(args.overrides))


def setup_logging(args: argparse.Namespace) -> None:
    """Configure logging from parsed arguments."""
    level = getattr(args, "log_level", None) or "INFO"
    configure_logging(level, json_output=bool(getattr(args, "json_logs", False)))


def emit(result: dict[str, Any], *, as_json: bool, logger_name: str) -> None:
    """Print a result, either as JSON on stdout or as a log line.

    Keeping machine-readable output on stdout separate from logs on stderr means
    ``gtsrb-train --json | jq`` works without filtering log noise.
    """
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        get_logger(logger_name).info("completed", extra={"result": result})


__all__ = [
    "DEFAULT_CONFIG_LAYERS",
    "REPO_ROOT",
    "add_common_arguments",
    "emit",
    "load_project_config",
    "setup_logging",
]
