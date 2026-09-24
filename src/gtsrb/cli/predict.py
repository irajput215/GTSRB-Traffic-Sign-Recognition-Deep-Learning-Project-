"""Predict traffic signs with a trained checkpoint.

Works on a single image, a directory of images, or a manifest, and can print a
human-readable report or machine-readable JSON.

Examples:
    # one image
    python -m gtsrb.cli.predict --image path/to/sign.png

    # every image in a directory, as JSON
    python -m gtsrb.cli.predict --directory samples/ --json

    # a specific checkpoint and top-5 alternatives
    python -m gtsrb.cli.predict --image sign.png \
        --checkpoint artifacts/checkpoints/best.pt --top-k 5

    # machine-readable output for scripting
    python -m gtsrb.cli.predict --image sign.png --json | jq '.predictions[0].class_name'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gtsrb.cli._common import add_common_arguments, emit, load_project_config, setup_logging
from gtsrb.inference import InvalidImageError, Predictor
from gtsrb.runtime import get_logger

logger = get_logger("cli.predict")

#: Extensions treated as images when scanning a directory.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".webp", ".tif", ".tiff"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gtsrb-predict",
        description="Predict traffic signs with a trained GTSRB classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="a single image file")
    source.add_argument("--directory", type=Path, help="a directory of images")
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint to load")
    parser.add_argument("--top-k", type=int, default=None, help="how many alternatives to report")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    return parser.parse_args(argv)


def collect_images(directory: Path) -> list[Path]:
    """Return the image files in ``directory``, sorted for a stable output order.

    Raises:
        FileNotFoundError: if the directory does not exist.
        ValueError: if it contains no images.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(
            f"No images found in {directory}. Looked for: {', '.join(sorted(IMAGE_SUFFIXES))}"
        )
    return images


def run(args: argparse.Namespace) -> int:
    config = load_project_config(args)
    if args.top_k is not None:
        config = config.model_copy(
            update={"inference": config.inference.model_copy(update={"top_k": args.top_k})}
        )

    predictor = Predictor.from_config(config, args.checkpoint)
    predictor.warmup()

    if args.image is not None:
        predictions = [predictor.predict_path(args.image)]
        sources: list[str] = [str(args.image)]
    else:
        paths = collect_images(args.directory)
        predictions = predictor.predict_batch_paths(paths)
        sources = [str(path) for path in paths]

    payload = {
        "model": predictor.info(),
        "predictions": [
            {"source": source, **prediction.as_dict(include_all_scores=True)}
            for source, prediction in zip(sources, predictions, strict=True)
        ],
    }

    if not args.json:
        for source, prediction in zip(sources, predictions, strict=True):
            alternatives = ", ".join(
                f"{score.short_name} {score.confidence:.3f}" for score in prediction.top_k[1:]
            )
            logger.info(
                "%s -> %s (%.4f)%s",
                source,
                prediction.class_name,
                prediction.confidence,
                f" | alternatives: {alternatives}" if alternatives else "",
            )

    emit(payload, as_json=args.json, logger_name="cli.predict")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args)
    try:
        return run(args)
    except (InvalidImageError, FileNotFoundError, ValueError) as exc:
        # These are user-input problems, so report them plainly rather than as a
        # traceback.
        logger.error("prediction failed: %s", exc)
        return 1
    except Exception as exc:
        logger.error("prediction failed: %s", exc)
        if args.log_level == "DEBUG":
            logger.exception("traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
