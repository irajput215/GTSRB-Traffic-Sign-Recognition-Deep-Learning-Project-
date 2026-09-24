# Experiment Tracking

MLflow records every run's configuration, metrics, artifacts and model version.
Tracking is **optional and off by default** — the project trains, evaluates and
serves with no MLflow install and no network access.

## Why MLflow here

The original project's only record of a run was whichever cell outputs happened to
be saved in the notebook: three hyperparameters visible as literals, three metric
tables, and a set of plots. Comparing two configurations meant diffing a 7 MB JSON
blob. MLflow replaces that with a queryable store, and it is the one tracking tool
this project needs — no second framework is layered on top.

## Enable it

```bash
uv sync --extra tracking          # installs mlflow
python -m gtsrb.cli.train --set tracking.enabled=true
```

Equivalent via environment:

```bash
export GTSRB_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python -m gtsrb.cli.train --set tracking.enabled=true
```

## Where runs are stored

The default backend is a **local SQLite database**:

```yaml
tracking:
  enabled: false
  tracking_uri: sqlite:///mlflow.db
```

`sqlite:///` rather than the legacy `file:./mlruns` store because **MLflow 3.x put
the filesystem tracking backend into maintenance mode** and raises on it:

```
MlflowException: The filesystem tracking backend (e.g., './mlruns') is in
maintenance mode and will not receive further updates. Please migrate to a
database backend (e.g., 'sqlite:///mlflow.db') ... If the filesystem backend is
required for your workflow, set `MLFLOW_ALLOW_FILE_STORE=true` to opt out.
```

A local SQLite file keeps everything that mattered about the file store — one
file, no server, no account, no network — on a supported backend. To use the file
store anyway, set `MLFLOW_ALLOW_FILE_STORE=true` and point `tracking_uri` at
`file:./mlruns`.

To use a remote server:

```bash
python -m gtsrb.cli.train \
  --set tracking.enabled=true \
  --set tracking.tracking_uri=http://mlflow.internal:5000
```

Any credential for that server belongs in your shell or a secret manager. Nothing
in this repository reads a secret, and `.env` is git-ignored.

## View the UI

```bash
make mlflow-ui
# MLflow UI available at http://127.0.0.1:5000
```

The tracking URI is passed explicitly so the UI reads the same database the
training runs wrote to.

## What is recorded

| Kind | Contents |
| --- | --- |
| Parameters | Every scalar in the resolved config (dotted keys), run name, and the data pipeline's split sizes, seed and balance power |
| Artifacts | `resolved_config.json` (the fully merged config), `train_result.json` (per-epoch history), the best checkpoint |
| Metrics | Per epoch: `train_loss`, `train_accuracy`, `train_macro_f1`, `val_loss`, `val_accuracy`, `val_macro_precision`, `val_macro_recall`, `val_macro_f1`, `val_top3_accuracy`, `learning_rate` |
| Model | The best checkpoint's weights, with an inferred signature and metadata |

Logging the resolved config as an artifact matters more than it looks: MLflow
stores parameters, but neither the YAML layers nor the `--set` overrides are
recoverable from the run afterwards, and "which flags did we use?" is exactly the
question a later reproduction attempt needs answered.

The **best** checkpoint is logged, not the last. After `fit()` the in-memory model
holds the final epoch, which is frequently not the best one, so the trainer's
checkpoint is reloaded before logging.

## Model lifecycle

```text
configs + --set overrides
        |
        v
   MLflow experiment  ──params, metrics, config artifact──►  run
        |
        v
   best checkpoint (val_macro_f1)
        |
        v
   logged model artifact (signature + metadata)
        |
        v
   registered model   (only if tracking.register_model=true)
        |
        v
   loaded by URI for inspection or deployment
```

Register the model by turning on the flag:

```bash
python -m gtsrb.cli.train \
  --set tracking.enabled=true \
  --set tracking.register_model=true \
  --set tracking.registered_model_name=gtsrb-traffic-sign-classifier
```

Registration is off by default because a registry entry is a claim that a model is
worth promoting; it should be a deliberate act rather than a side effect of
training.

Load a registered version:

```python
import mlflow.pytorch

model = mlflow.pytorch.load_model("models:/gtsrb-traffic-sign-classifier/1")
```

## A note on the artifact serialisation format

Models are logged with `serialization_format="pickle"`, not MLflow 3.x's default
`pt2` traced graph, and that is a measured decision:

- the traced graph is exported against a fixed input example, so it accepts
  **exactly** the batch size it was traced with — a model logged from a batch of 1
  rejects a batch of 2 with `Guard failed: x.size()[0] == 1`;
- calling `.eval()` on a traced module raises `NotImplementedError`, because a
  traced graph has no training state left to switch;
- `pickle` stores the live `nn.Module`, so the registered model accepts any batch
  size and behaves like the model that was trained.

The cost is losing the traced graph's inference speed-up, which this project does
not depend on: the inference service loads the project's own checkpoint format
rather than an MLflow artifact, because a checkpoint also carries the class-label
ordering, input geometry and normalisation statistics that the API needs to
preprocess a request correctly.

## Tests

`tests/integration/test_mlflow_tracking.py` runs against a real MLflow SQLite store
in a temporary directory — no server, no mocks:

```bash
uv run pytest tests/integration/test_mlflow_tracking.py -q
```

It covers run lifecycle and idempotent shutdown, parameter truncation, non-finite
metric filtering (MLflow rejects NaN/infinity and a divergence must not crash
tracking), artifact logging, model logging with signature, model round-trip
loading, and the full register-then-load-from-registry path.
