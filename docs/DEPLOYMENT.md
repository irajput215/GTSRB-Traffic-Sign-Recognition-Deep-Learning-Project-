# Deployment

The inference service is a stateless container that serves one HTTP API. This
document covers building it, running it, and the decisions behind the image.

## Build

```bash
docker build -t gtsrb-traffic-sign-recognition:local .
```

Or with Compose, which is the recommended local path:

```bash
docker compose up --build
```

## Run

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts/checkpoints:/app/artifacts/checkpoints:ro" \
  -e GTSRB_CHECKPOINT_PATH=/app/artifacts/checkpoints/best.pt \
  gtsrb-traffic-sign-recognition:local
```

Then:

```bash
curl localhost:8000/health
curl localhost:8000/ready
curl -X POST localhost:8000/predict -F "file=@sign.png"
```

Interactive API documentation is at <http://localhost:8000/docs>.

## Where the model comes from

**The checkpoint is not baked into the image.** It is mounted, for three reasons:

1. it is a build output of the training pipeline, not source;
2. the compact CNN checkpoint is ~37 MB and the ResNet-50 checkpoint is ~100 MB,
   so every code change would rebuild a layer containing it;
3. the same image can then serve different model versions, which is what makes a
   canary or a rollback a configuration change rather than an image rebuild.

For a self-contained image, add a stage that copies it in:

```dockerfile
COPY artifacts/checkpoints/best.pt /app/artifacts/checkpoints/best.pt
```

The API will start and report `/ready` as `503` if the checkpoint is missing, with
the reason in the response — it does not crash.

## Measured image size

| Configuration | Size | Note |
| --- | ---: | --- |
| Default PyPI `torch` (the CUDA build) | **9.63 GB** | 3.3 GB `nvidia` runtime + 817 MB `triton` |
| CPU `torch` from the PyTorch CPU index | 1.9 GB | |
| CPU `torch`, unused `pandas` removed | **1.8 GB** | current |

The first row is the measurement that motivated the rest. On Linux, PyPI's `torch`
is the CUDA build, so a naive `pip install` produces a multi-gigabyte GPU stack for
a service that runs on CPU.

## Why the image is built the way it is

### CPU-only PyTorch

The single largest size decision. A default `pip install torch` on Linux resolves
to the CUDA build and pulls its bundled GPU libraries — several gigabytes for a
service that runs on CPU. The builder stage sets:

```dockerfile
UV_INDEX_URL=https://download.pytorch.org/whl/cpu
```

An important detail: this is configured in `pyproject.toml` under
`[[tool.uv.index]]` and `[tool.uv.sources]`, **not** as a `UV_INDEX_URL` in the
Dockerfile. The lock file decides what `uv sync --frozen` installs, so a
build-time index override is silently ignored against a lock that points at PyPI —
the first attempt at this built the 9.63 GB image despite the environment variable
being set. Declaring the source in the project makes the lock itself CPU-only, with
platform markers so macOS still resolves its own wheel:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

If you deploy on GPU nodes, change that index and drop the `GTSRB_DEVICE=cpu`
default.

### Two stages

The builder stage carries `uv`, its download cache and the build frontend; the
runtime stage receives only the finished virtual environment. The split is not
ceremony — it removes the toolchain from the shipped image, and it is also what
lets the dependency layer be cached separately from the source layer, so editing a
module does not re-resolve and re-download torch.

### Minimal dependency set

Only `--extra api` is installed. No dev tooling, and **no MLflow**: MLflow is a
training concern, and baking it in would add hundreds of megabytes to an image
whose only job is to answer `POST /predict`. The checkpoint carries the model; the
tracking store is a training-side artifact.

### Non-root user

The service runs as `gtsrb` (uid 1001) and never writes to its own image. Compose
additionally sets `read_only: true` with a `tmpfs` for `/tmp` and
`no-new-privileges`, so a compromised process cannot persist anything.

### Healthcheck without curl

```dockerfile
HEALTHCHECK CMD python -c "... urllib.request.urlopen('http://127.0.0.1:8000/health') ..."
```

Uses the interpreter already in the image rather than installing `curl` or `wget`
for one check. Note that the healthcheck targets `/health`, not `/ready`: a
container whose model has not loaded is still *alive*, and marking it unhealthy
would cause a restart loop instead of a diagnosable 503.

### Graceful shutdown

`ENTRYPOINT` uses the exec form, so `SIGTERM` reaches uvicorn directly instead of
being swallowed by a shell. uvicorn then drains in-flight requests.

## Configuration

Everything is environment-driven; no rebuild is needed to change it. See
`.env.example` for the full list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GTSRB_CHECKPOINT_PATH` | `artifacts/checkpoints/best.pt` | Checkpoint to serve |
| `GTSRB_DEVICE` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `GTSRB_API_HOST` | `0.0.0.0` | Bind address |
| `GTSRB_API_PORT` | `8000` | Port |
| `GTSRB_API_METRICS_ENABLED` | `true` | Prometheus endpoint |
| `GTSRB_LOG_LEVEL` | `INFO` | Log verbosity |
| `GTSRB_MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Training only |

## Scaling

The service is stateless and CPU-bound, so it scales horizontally by adding
replicas behind a load balancer. Route traffic on **`/ready`**, not `/health`, so
an instance whose model has not loaded yet receives no requests.

For higher throughput per instance, the highest-leverage change is batching:
`Predictor.predict_batch` already runs a single forward pass for a batch of images,
and a small request-queue in front of it would use more of each core than
per-request inference does. That is deliberately not implemented here, because
adding a queue changes the latency/throughput trade-off and should be driven by a
measured need rather than assumed.

## Security review

What was checked, and what the answer is:

| Concern | Status |
| --- | --- |
| Credentials in the repository | **None.** No API key, token, password or private key exists in any tracked file. `detect-private-key` is a pre-commit hook, so a key cannot be committed accidentally. |
| `.env` files | Git-ignored (`.env`, `.env.*`), with `.env.example` committed as documentation. |
| Hardcoded credentials | None. Nothing in this project authenticates to anything; there is no cloud dependency to hold a credential for. |
| Dataset paths | Configurable via `GTSRB_DATA_ROOT` or `data.root` in YAML. No absolute path is baked into the code — the original's `/content/dataset/` is gone. |
| Uploaded images | Validated in memory, never written to disk, never logged, never echoed in an error message. The request log records whether a filename was *supplied*, not the name. A test asserts a path-like filename cannot appear in the logs. |
| Error responses | One shape carrying a stable code, a safe detail and a request id. No stack trace, file path or library message reaches a client — a test triggers an unhandled exception and asserts the internal message is absent from the response. |
| Container privileges | Non-root (uid 1001), read-only root filesystem, `tmpfs` for `/tmp`, `no-new-privileges`. |
| Untrusted input | Upload size is checked **before** decoding; decoding is capped by an explicit pixel limit to bound decompression bombs; truncated files are rejected because Pillow otherwise half-decodes them. |
| Dependency surface | Nine core dependencies, each justified. Specifically removed an unused `pandas` during the Docker work rather than carrying 75 MB of dead weight. |
| ML supply chain | The MLflow artifact is loaded as `pickle`, which is code-executing. Acceptable for an artifact this process produced; **not** acceptable for a model from an untrusted source, which would need `weights_only=True` checkpoint loading and a safetensors export. |

**What is deliberately absent:** authentication, authorization and rate limiting.
Those belong at the ingress, and implementing a half-measure in the application — a
bearer token read from an environment variable, say — would create a false sense of
safety. The limitation is stated rather than papered over.

## Known limitations

- **No authentication.** The API is open. Put it behind an authenticating gateway
  or a service mesh; nothing in the application implements authorization, and
  pretending otherwise with a bearer token in an environment variable would be
  worse than being explicit about it.
- **No request-rate limiting.** Do it at the ingress.
- **Single-process.** uvicorn runs one worker, so one slow request occupies the
  loop for its duration. The model runs on CPU for a fraction of a millisecond per
  image at 64x64, so this has not mattered; a GPU deployment should use
  `--workers` and expect per-worker model copies.
- **No drift detection.** `gtsrb_predictions_total` gives the predicted-class
  distribution, which is the raw signal drift detection would need, but nothing
  alerts on it.
- **Training-only dependencies are still installed in the image.** `scikit-learn`
  (57 MB) and `matplotlib` (37 MB) plus their `scipy` dependency (118 MB) are used
  by evaluation and plotting, not by inference, and account for roughly 210 MB of
  the 1.8 GB. Splitting them into a `[train]` extra would shrink the image further.
  It is not done here because it changes what a bare `pip install` provides, and a
  210 MB reduction on an already-1.8 GB PyTorch image does not justify that. This is
  the next size optimisation if one is ever needed.

## Verifying a deployment

```bash
curl -sf localhost:8000/health && echo alive
curl -sf localhost:8000/ready  && echo ready
curl -s  localhost:8000/model-info | python -m json.tool
curl -s  localhost:8000/metrics-summary | python -m json.tool
```

`/model-info` is the one that answers "which model is this deployment actually
running?" — it reports the architecture, the checkpoint path, the training epoch,
the parameter count and the validation metrics recorded when the checkpoint was
written.
