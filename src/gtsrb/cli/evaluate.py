"""Evaluate a trained checkpoint.

Produces metrics, an error analysis and a full set of figures, and optionally
records them in MLflow.

Examples:
    # evaluate the default checkpoint on the test split
    python -m gtsrb.cli.evaluate

    # evaluate a specific checkpoint on the validation split
    python -m gtsrb.cli.evaluate --checkpoint artifacts/checkpoints/best.pt --split val

    # metrics only, no figures (much faster)
    python -m gtsrb.cli.evaluate --no-figures

    # log the evaluation to MLflow
    python -m gtsrb.cli.evaluate --set tracking.enabled=true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gtsrb.cli._common import add_common_arguments, emit, load_project_config, setup_logging
from gtsrb.evaluation import evaluate_checkpoint
from gtsrb.runtime import get_logger
from gtsrb.tracking import build_tracker

logger = get_logger("cli.evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gtsrb-evaluate",
        description="Evaluate a trained GTSRB classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint to evaluate")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="which split to evaluate on (default: test)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="where to write artifacts")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    config = load_project_config(args)
    result = evaluate_checkpoint(
        config,
        args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        render=not args.no_figures,
    )

    payload = result.as_dict()

    tracker = build_tracker(config)
    try:
        if config.tracking.enabled:
            tracker.log_metrics(
                {f"eval_{key}": value for key, value in result.metrics.headline().items()}
            )
            tracker.log_dict(payload["metrics"], f"evaluation/{args.split}_metrics.json")
            tracker.log_dict(
                result.error_analysis.as_dict(),
                f"evaluation/{args.split}_error_analysis.json",
            )
            for artifact in result.artifacts:
                tracker.log_artifact(artifact, artifact_path=f"evaluation/{artifact.parent.name}")
    finally:
        tracker.finish()

    emit(payload, as_json=args.json, logger_name="cli.evaluate")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args)
    try:
        return run(args)
    except Exception as exc:
        logger.error("evaluation failed: %s", exc)
        if args.log_level == "DEBUG":
            logger.exception("traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
