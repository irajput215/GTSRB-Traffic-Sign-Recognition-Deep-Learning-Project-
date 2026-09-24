"""Train a traffic-sign classifier.

Examples:
    # default run: compact CNN, 30 epochs, configs/*.yaml
    python -m gtsrb.cli.train

    # reproduce the MLP baseline from the original comparison
    python -m gtsrb.cli.train --set model.name=mlp

    # quick smoke run
    python -m gtsrb.cli.train --set training.epochs=1 --set 'data.root=data/raw'

    # resume an interrupted run (last.pt holds optimiser state; best.pt does not)
    python -m gtsrb.cli.train --resume artifacts/checkpoints/last.pt

    # train with MLflow tracking, registering the model
    python -m gtsrb.cli.train --set tracking.enabled=true \
        --set tracking.register_model=true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gtsrb.cli._common import add_common_arguments, emit, load_project_config, setup_logging
from gtsrb.data import build_dataloaders
from gtsrb.models import build_model, load_model_from_checkpoint
from gtsrb.runtime import get_logger, seed_everything
from gtsrb.tracking import build_tracker
from gtsrb.training import Trainer

logger = get_logger("cli.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gtsrb-train",
        description="Train a GTSRB traffic-sign classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_arguments(parser)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--no-download", action="store_true", help="fail instead of downloading GTSRB"
    )
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="recompute the train/validation split manifest",
    )
    parser.add_argument("--run-name", default=None, help="override the run name")
    parser.add_argument("--json", action="store_true", help="print the result summary as JSON")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    config = load_project_config(args)
    if args.run_name:
        config = config.model_copy(update={"run_name": args.run_name})

    # Seed before anything stochastic happens, including loader construction.
    seed_everything(config.training.seed, deterministic=config.training.deterministic)

    tracker = build_tracker(config)
    try:
        tracker.log_params({**config.flat_params(), "run_name": config.run_name})
        # The resolved config is logged as an artifact because neither the YAML
        # files nor the --set overrides are otherwise recoverable after the fact.
        tracker.log_dict(config.model_dump(mode="json"), "resolved_config.json")

        bundle = build_dataloaders(
            config, download=not args.no_download, force_split=args.force_split
        )
        tracker.log_params(
            {
                "data.train_samples": len(bundle.train_dataset),
                "data.val_samples": len(bundle.val_dataset),
                "data.test_samples": len(bundle.test_dataset),
                "data.split_seed": bundle.manifest.seed,
                "data.balance_power": config.data.balance.power,
            }
        )

        model = build_model(config.model, image_size=config.data.image_size)
        trainer = Trainer(config, model, bundle.train, bundle.val, tracker=tracker)
        result = trainer.fit(resume_from=args.resume)

        payload = {
            **result.as_dict(),
            "data": bundle.metadata(),
            "config": config.model_dump(mode="json"),
        }

        # Persist the run summary next to the checkpoints so a run can be inspected
        # without the log.
        summary_path = Path(config.training.checkpoint.directory) / "train_result.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        logger.info(
            "training summary", extra={"summary_path": str(summary_path), **result.best_metrics}
        )

        if result.checkpoint_path is not None:
            # Log the *best* weights, not the last ones. After fit() the in-memory
            # model holds the final epoch, which is often not the best one, so the
            # checkpoint is reloaded rather than reusing the live model.
            best_model, best_metadata = load_model_from_checkpoint(
                Path(result.checkpoint_path), device=trainer.device
            )
            tracker.log_model(best_model, best_metadata)
            tracker.log_artifact(Path(result.checkpoint_path))
        tracker.log_dict(payload, "train_result.json")
    finally:
        # The tracker must be closed even if anything above raises, or the MLflow run
        # is left open and later runs inherit its metadata.
        tracker.finish()

    emit(payload, as_json=args.json, logger_name="cli.train")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args)
    try:
        return run(args)
    except Exception as exc:
        logger.error("training failed: %s", exc)
        if args.log_level == "DEBUG":
            logger.exception("traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
