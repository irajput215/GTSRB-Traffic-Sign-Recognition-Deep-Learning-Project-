"""Serve the inference API.

Examples:
    # default: 0.0.0.0:8000, model from artifacts/checkpoints/best.pt
    python -m gtsrb.cli.serve

    # a specific checkpoint on a specific port
    python -m gtsrb.cli.serve --checkpoint artifacts/checkpoints/best.pt --port 9000

    # produce a reload-friendly dev server
    python -m gtsrb.cli.serve --reload

    # uvicorn can also run the factory directly
    uvicorn --factory gtsrb.api.main:build_app_from_env --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gtsrb.cli._common import add_common_arguments, load_project_config, setup_logging
from gtsrb.runtime import get_logger

logger = get_logger("cli.serve")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gtsrb-serve",
        description="Serve the GTSRB inference API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint to serve")
    parser.add_argument("--host", default=None, help="bind address (default: from config)")
    parser.add_argument("--port", type=int, default=None, help="port (default: from config)")
    parser.add_argument(
        "--reload", action="store_true", help="restart on code changes (development only)"
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    try:
        # Both are optional extras, so they are imported here rather than at module
        # scope: importing this CLI must not require the API dependencies.
        import uvicorn  # noqa: PLC0415

        from gtsrb.api import create_app  # noqa: PLC0415
    except ImportError:
        logger.error(
            "the API extras are not installed. Install them with "
            "'uv sync --extra api' (or 'pip install fastapi uvicorn[standard] prometheus-client')."
        )
        return 1

    config = load_project_config(args)
    if args.checkpoint is not None:
        config = config.model_copy(
            update={
                "inference": config.inference.model_copy(
                    update={"checkpoint_path": args.checkpoint}
                )
            }
        )
    if args.host is not None or args.port is not None:
        config = config.model_copy(
            update={
                "api": config.api.model_copy(
                    update={
                        "host": args.host if args.host is not None else config.api.host,
                        "port": args.port if args.port is not None else config.api.port,
                    }
                )
            }
        )

    app = create_app(config)
    logger.info(
        "starting server",
        extra={
            "host": config.api.host,
            "port": config.api.port,
            "checkpoint": str(config.inference.checkpoint_path),
            "docs_url": f"http://{config.api.host}:{config.api.port}/docs",
        },
    )

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level=config.api.log_level.lower(),
        reload=args.reload,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args)
    try:
        return run(args)
    except Exception as exc:
        logger.error("server failed: %s", exc)
        if args.log_level == "DEBUG":
            logger.exception("traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
