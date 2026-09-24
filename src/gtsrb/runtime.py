"""Device selection, seeding and logging.

These three concerns are shared by training, evaluation, inference and the API,
and each has a subtle failure mode that is worth handling once:

* device selection decides where the model and every batch live;
* seeding has to cover Python, NumPy, PyTorch, cuDNN and DataLoader workers, or
  a "reproducible" run is only reproducible by accident;
* logging has to be configured once, by the entry point, and never by a library
  module that a caller did not ask to reconfigure.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from gtsrb.config.schema import DeviceName

LOGGER_NAME = "gtsrb"

Device = torch.device

_RESOLVED: dict[str, torch.device] = {}


def resolve_device(requested: DeviceName | str = "auto") -> torch.device:
    """Resolve a device request into a concrete ``torch.device``.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. The choice is logged once
    per process so a run's log states where it actually executed rather than
    leaving it to be inferred from timing.

    Args:
        requested: ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Raises:
        ValueError: if an explicitly requested device is unavailable. Silently
            falling back to CPU would turn a GPU run into a very slow CPU run.
    """
    if requested in _RESOLVED:
        return _RESOLVED[requested]

    normalised = str(requested).lower()
    if normalised == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif normalised == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        device = torch.device("cuda")
    elif normalised == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("device='mps' requested but the MPS backend is not available")
        device = torch.device("mps")
    elif normalised == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device {requested!r}; expected one of auto, cpu, cuda, mps")

    _RESOLVED[requested] = device
    return device


def supports_amp(device: torch.device) -> bool:
    """Return whether automatic mixed precision is worth enabling on ``device``.

    Only CUDA has a meaningful AMP path today. On CPU and MPS the autocast
    context is a no-op at best and unsupported at worst, so callers use this to
    skip the GradScaler entirely rather than pay for a no-op.
    """
    return device.type == "cuda"


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed every random number generator this project touches.

    Also sets ``PYTHONHASHSEED`` for this process. Note that this must happen
    before any DataLoader worker is spawned; worker seeding is handled separately
    by ``gtsrb.data.dataset.worker_init_fn``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Best-effort: some operations have no deterministic kernel and will
        # raise instead of silently diverging. That is the desired behaviour for
        # a run that claims reproducibility.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):  # pragma: no cover - backend dependent
            logging.getLogger(LOGGER_NAME).debug(
                "torch.use_deterministic_algorithms is not available on this build"
            )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(_worker_id: int) -> None:
    """Seed a DataLoader worker deterministically from the parent generator.

    Passed as ``worker_init_fn``. Without it, augmentation inside worker
    processes is seeded from the OS and a "seeded" run still varies.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a CPU ``torch.Generator`` for shuffling and weighted sampling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class JsonLogFormatter(logging.Formatter):
    """Render log records as single-line JSON.

    Structured logs are what make request latency and error counts greppable in
    a container without shipping a log pipeline. Extra fields attached via
    ``logger.info(..., extra={"...": ...})`` are merged into the payload, except
    for the standard ``LogRecord`` attributes.
    """

    _RESERVED = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(
    level: str | int = "INFO",
    *,
    json_output: bool = False,
    stream: Any | None = None,
) -> logging.Logger:
    """Configure the ``gtsrb`` logger and return it.

    Idempotent: calling it twice replaces the handler rather than stacking a
    second one, so a test or a reloaded app does not emit every line twice.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    if json_output:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the ``gtsrb`` logger."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
