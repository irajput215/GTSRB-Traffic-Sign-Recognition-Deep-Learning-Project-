#!/usr/bin/env python
"""Assert that a training run meets a quality floor.

Used as the gate in the ``model-validation`` workflow. It reads the
``train_result.json`` a training run writes and fails if the model did not reach
the required metrics, so a regression in the data pipeline, the augmentation, the
loss or the schedule is caught by CI rather than discovered later.

Thresholds are passed in rather than hard-coded, because the correct floor depends
on how long the run was allowed to train.

Usage:
    python scripts/check_model_quality.py --monitor val_accuracy --minimum 0.7
    python scripts/check_model_quality.py --result artifacts/checkpoints/train_result.json \\
        --monitor val_macro_f1 --minimum 0.6 --require-artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = REPO_ROOT / "artifacts" / "checkpoints" / "train_result.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT, help="train_result.json")
    parser.add_argument(
        "--monitor",
        default="val_accuracy",
        choices=["val_accuracy", "val_macro_f1", "val_top3_accuracy"],
        help="metric to gate on",
    )
    parser.add_argument("--minimum", type=float, required=True, help="minimum acceptable value")
    parser.add_argument(
        "--min-epochs", type=int, default=1, help="require at least this many epochs to have run"
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="also require that a checkpoint was written",
    )
    return parser.parse_args(argv)


def check(args: argparse.Namespace) -> list[str]:
    """Return a list of failure messages; empty means the run passed."""
    failures: list[str] = []

    if not args.result.exists():
        return [f"training result not found: {args.result}"]

    try:
        payload = json.loads(args.result.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{args.result} is not valid JSON: {exc}"]

    epochs = int(payload.get("epochs_trained", 0))
    if epochs < args.min_epochs:
        failures.append(f"only {epochs} epoch(s) completed; at least {args.min_epochs} required")

    best = payload.get("best_metrics") or {}
    if args.monitor not in best:
        failures.append(f"{args.monitor} is missing from best_metrics")
    else:
        value = float(best[args.monitor])
        print(f"{args.monitor} = {value:.4f} (minimum {args.minimum})")
        if value < args.minimum:
            failures.append(
                f"{args.monitor} {value:.4f} is below the required minimum {args.minimum}"
            )

    if args.require_artifacts:
        checkpoint = payload.get("checkpoint_path")
        if not checkpoint:
            failures.append("no checkpoint_path was recorded")
        elif not Path(checkpoint).exists():
            failures.append(f"recorded checkpoint does not exist: {checkpoint}")

    return failures


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failures = check(args)
    if failures:
        for message in failures:
            print(f"FAIL {message}", file=sys.stderr)
        return 1
    print(f"PASS {args.monitor} gate at {args.minimum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
