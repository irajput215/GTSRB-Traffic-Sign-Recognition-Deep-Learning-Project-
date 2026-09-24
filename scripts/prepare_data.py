#!/usr/bin/env python
"""Prepare and inspect the GTSRB dataset.

Does four things, all optional:

* download the dataset through ``torchvision``;
* validate it and print the class-distribution report;
* re-measure the channel normalisation statistics and compare them against the
  values in ``configs/data.yaml``;
* create (or re-create) the train/validation split manifest.

Usage:
    python scripts/prepare_data.py                      # download, validate, write the split
    python scripts/prepare_data.py --stats              # also re-measure channel statistics
    python scripts/prepare_data.py --force-split        # recompute the split manifest
    python scripts/prepare_data.py --sample-size 500    # decode 500 images instead of all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gtsrb.config import ProjectConfig, load_config
from gtsrb.config.schema import DataConfig
from gtsrb.data import (
    estimate_channel_statistics,
    load_raw_split,
    read_labels,
    validate_dataset,
)
from gtsrb.data.splits import get_or_create_manifest
from gtsrb.data.validation import compare_statistics
from gtsrb.runtime import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYERS = ("data.yaml", "model.yaml", "train.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        default=[REPO_ROOT / "configs" / name for name in DEFAULT_LAYERS],
        help="config layers to load, in order of increasing precedence",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-root", type=Path, default=None, help="override data.root")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    parser.add_argument("--force-split", action="store_true", help="recompute the split manifest")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="images to decode during validation; 0 validates every image",
    )
    parser.add_argument(
        "--stats", action="store_true", help="re-measure channel normalisation statistics"
    )
    parser.add_argument("--stats-sample-size", type=int, default=2000)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ProjectConfig:
    overrides = list(args.overrides)
    if args.data_root is not None:
        overrides.append(f"data.root={args.data_root}")
    return load_config(list(args.config), overrides)


def run(args: argparse.Namespace) -> int:
    config = build_config(args)
    data: DataConfig = config.data

    train = load_raw_split(data.root, "train", download=not args.no_download)
    labels = read_labels(train)

    statistics = validate_dataset(
        train,
        num_classes=data.num_classes,
        sample_size=args.sample_size,
        seed=data.split_seed,
    )
    manifest = get_or_create_manifest(labels, data, force=args.force_split)

    report: dict[str, object] = {
        "data_root": str(data.root),
        "train": {
            "num_samples": statistics.num_samples,
            "num_classes_present": len(statistics.class_counts),
            "min_class_count": statistics.min_class_count,
            "max_class_count": statistics.max_class_count,
            "imbalance_ratio": round(statistics.imbalance_ratio, 3),
            "missing_classes": statistics.missing_classes,
        },
        "split": {
            "seed": manifest.seed,
            "val_fraction": manifest.val_fraction,
            "num_train": manifest.num_train,
            "num_val": manifest.num_val,
            "manifest_path": str(data.split_manifest),
        },
    }

    if args.stats:
        measured = estimate_channel_statistics(
            train,
            sample_size=args.stats_sample_size,
            seed=data.split_seed,
            image_size=data.image_size,
        )
        report["channel_statistics"] = compare_statistics(
            measured[0], measured[1], data.normalize_mean, data.normalize_std
        )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Dataset root        : {data.root}")
        print(f"Training samples    : {statistics.num_samples}")
        print(f"Classes present     : {len(statistics.class_counts)} / {data.num_classes}")
        print(
            f"Per-class counts    : min {statistics.min_class_count}, "
            f"max {statistics.max_class_count}, imbalance {statistics.imbalance_ratio:.2f}x"
        )
        print(f"Missing classes     : {statistics.missing_classes or 'none'}")
        print(
            f"Split               : seed {manifest.seed}, val_fraction {manifest.val_fraction} "
            f"-> {manifest.num_train} train / {manifest.num_val} val"
        )
        print(f"Split manifest      : {data.split_manifest}")
        if "channel_statistics" in report:
            stats = report["channel_statistics"]
            assert isinstance(stats, dict)
            print()
            print("Channel normalisation check (pixel-pooled estimate, model input size)")
            print(f"  measured   mean {stats['measured_mean']}  std {stats['measured_std']}")
            print(f"  configured mean {stats['reference_mean']}  std {stats['reference_std']}")
            print(
                f"  delta      mean {[round(v, 4) for v in stats['mean_delta']]}  "
                f"std {[round(v, 4) for v in stats['std_delta']]}"
            )
            verdict = "within" if stats["within_tolerance"] else "OUTSIDE"
            print(f"  materiality {verdict} the {stats['tolerance']} threshold")
            print(
                "  note: the configured constants are deliberately unchanged so results stay\n"
                "        comparable with the recorded run; see docs/PROJECT_AUDIT.md section 2.3."
            )

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging("INFO")
    logger = get_logger("scripts.prepare_data")
    try:
        return run(args)
    except Exception as exc:
        logger.error("data preparation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
