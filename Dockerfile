# syntax=docker/dockerfile:1.7
#
# Inference image for the GTSRB traffic-sign classifier.
#
# Two stages, and the split earns its keep rather than being ceremony: the builder
# stage carries `uv`, its download cache and the build frontend, none of which
# belong in a runtime image. The runtime stage receives only the finished virtual
# environment.
#
# The dependency set is deliberately minimal — `--extra api` and nothing else. No
# dev tools, no MLflow. MLflow is a training concern; baking it in would add
# hundreds of megabytes to an image whose only job is to answer POST /predict.
#
# CPU-only PyTorch is the single largest image-size decision, and it is configured
# in pyproject.toml via [[tool.uv.index]] + [tool.uv.sources] rather than here. That
# placement matters: the *lock file* decides what `uv sync --frozen` installs, so a
# build-time index override is silently ignored against a lock that points at PyPI.
#
# Measured before that change: 9.63 GB, of which 3.3 GB was the `nvidia` runtime
# libraries and 817 MB was `triton` - a GPU stack for a service that runs on CPU.

# ---------------------------------------------------------------------------
# Stage 1: build the virtual environment
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Pinned so the build is reproducible. Bump deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the
# source, so an edit to a module does not re-resolve and re-download torch.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra api

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api --no-editable

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="GTSRB traffic sign recognition" \
      org.opencontainers.image.description="FastAPI inference service for 43-class German traffic sign classification" \
      org.opencontainers.image.source="https://github.com/irajput215/GTSRB-Traffic-Sign-Recognition-Deep-Learning-Project-" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    GTSRB_API_HOST=0.0.0.0 \
    GTSRB_API_PORT=8000 \
    GTSRB_LOG_LEVEL=INFO

WORKDIR /app

# A dedicated unprivileged user. The service never writes to its own image, so it
# has no reason to be root.
RUN groupadd --system --gid 1001 gtsrb \
    && useradd --system --uid 1001 --gid gtsrb --create-home gtsrb

COPY --from=builder --chown=gtsrb:gtsrb /app/.venv /app/.venv
COPY --chown=gtsrb:gtsrb configs /app/configs

# The checkpoint is not baked into the image: it is a build output, it is large,
# and it changes independently of the code. Mount it, or copy it in for a
# self-contained deployment. See docs/DEPLOYMENT.md.
RUN mkdir -p /app/artifacts/checkpoints && chown -R gtsrb:gtsrb /app/artifacts

USER gtsrb

EXPOSE 8000

# Uses Python rather than curl or wget, so no extra package is needed in the image
# and the check exercises the same interpreter the service runs on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# shell form is avoided so signals reach uvicorn directly and SIGTERM triggers a
# graceful shutdown rather than being swallowed by a shell.
ENTRYPOINT ["python", "-m", "gtsrb.cli.serve", \
            "--config", "configs/data.yaml", "configs/model.yaml", \
            "configs/train.yaml", "configs/api.yaml"]
