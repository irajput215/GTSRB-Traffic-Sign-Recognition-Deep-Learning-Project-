# Reproducibility

What is reproducible, how far, and where the limits are. Written to be specific
rather than reassuring: "we set a seed" is not a reproducibility guarantee.

## Quick path

```bash
git clone git@github.com:irajput215/GTSRB-Traffic-Sign-Recognition-Deep-Learning-Project-.git
cd GTSRB-Traffic-Sign-Recognition-Deep-Learning-Project-

make install-all          # or: uv sync --all-extras
make data                 # download GTSRB, validate it, write the split manifest
make train                # ~65 min on an M1 GPU; see "Training cost" below
make evaluate             # metrics, error analysis and figures
make serve                # FastAPI on http://127.0.0.1:8000
```

Everything documented in the README and here was executed in this repository. The
one thing that was **not** produced by these commands is the headline results table
— see [Metrics provenance](#metrics-provenance).

## What is pinned

| Layer | Mechanism | Scope |
| --- | --- | --- |
| Python | `requires-python = ">=3.12"` | 3.12 and 3.13 are both tested in CI |
| Dependencies | `uv.lock`, committed, installed with `--frozen` | Exact versions, plus the CPU/GPU torch split per platform |
| CUDA/CPU wheel choice | `[[tool.uv.index]]` + `[tool.uv.sources]` in `pyproject.toml` | CPU-only by default |
| Dataset | Downloaded by `torchvision`, not committed | Same archive every time |
| Data split | `artifacts/splits/gtsrb_split.json` | Exact indices **and** labels |
| Hyperparameters | `configs/*.yaml`, hashed into the checkpoint | Full resolved config stored |
| Model architecture | `ModelConfig.name` + per-architecture settings | Rebuilt from the checkpoint at load time |
| Container base image | `python:3.12-slim-bookworm`, `uv:0.11.7` | Pinned tags |
| CI runners | `ubuntu-latest`, Python 3.12 + 3.13 | Two interpreters |

## Randomness

`seed_everything(seed, deterministic=True)` covers:

- Python's `random`
- NumPy's global state
- `torch.manual_seed`
- `torch.cuda.manual_seed_all`
- `PYTHONHASHSEED` for the process
- `torch.backends.cudnn.deterministic = True`, `benchmark = False`
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- **DataLoader workers**, via `worker_init_fn=seed_worker` reading
  `torch.initial_seed()`

That last one is the one that is usually missed. The original notebook seeded the
first four and left worker RNG to the OS — and since augmentation runs inside the
workers, a "seeded" run still produced different augmented images each time.

The training loader additionally uses a local `torch.Generator` for both shuffling
and weighted sampling, so the sample order does not depend on the global RNG state.

## Known limits on reproducibility

Stated as limits rather than caveats, because each one has a concrete cause.

1. **Bit-exact reproducibility is not guaranteed across hardware.** cuDNN selects
   different kernels on different GPUs and on MPS, and floating-point reduction
   order differs. Two runs with the same seed on the same machine reproduce; the
   same seed on a different device gives the same *distribution* of results, not
   the same weights. `torch.use_deterministic_algorithms(warn_only=True)` warns
   rather than raising, because some operations have no deterministic kernel.
2. **`--set training.deterministic=false` is faster and non-reproducible.** It
   enables `cudnn.benchmark`, which benchmarks kernels on first use. Useful for
   final runs; not for comparisons.
3. **`num_workers > 0` widens the window.** Worker seeding is deterministic, but
   the *interleaving* of batches is not. Reproduce with `num_workers=0` if a
   comparison must be exact.
4. **The reported metrics have no variance estimate.** Every recorded number is a
   single run: no repeats, no confidence intervals, no seed sweep. Differences
   smaller than run-to-run variation cannot be claimed from this project's data.
5. **The reported run used a different environment.** Colab, Python 3.11.5, an A100
   GPU, `torch` at an unknown version. This repository pins all of that differently.
   The results table is therefore a historical measurement, not a reproducible
   artifact of this code — see below.

## Metrics provenance

This is the part worth being precise about.

The headline results in the README (compact CNN 98.84%, ResNet-50 96.97%, MLP
65.84%) are the **recorded outputs of the original COMP9444 notebook run**, kept in
`legacy/COMP9444_notebook.ipynb` and written up in `docs/original/COMP9444_report.pdf`.
They were not produced by this repository's commands, and they are labelled that way
wherever they appear.

What the refactored pipeline was verified to reproduce exactly:

| Quantity | Original notebook | Refactored pipeline |
| --- | --- | --- |
| Training split size | 26,640 | **26,640** |
| Classes present | 43 | **43 / 43** |
| Per-class min / max | 150 / 1500 | **150 / 1500** |
| Stratified 80/20 split | 21,312 / 5,328 | **21,312 / 5,328** |

The split is identical because the seed (43), the fraction (0.2) and the
stratification are preserved, and the split is now written to a manifest so the
claim is checkable.

### A dataset caveat that has to travel with the numbers

`torchvision`'s GTSRB `train` split contains **26,640** images. The official GTSRB
training partition contains **39,209**.

The original project trained on 26,640 and evaluated on the official 12,630-image
test split. Any comparison with published GTSRB results has to carry that caveat:
this is not a full-GTSRB benchmark. Re-verified during the refactor by downloading
the dataset and reading `len(dataset)` for both splits.

## Training cost, measured

A complete 30-epoch run on the verification machine — Apple M1, 8 GB RAM, MPS
backend, `num_workers=0`:

| Quantity | Value |
| --- | --- |
| Full 30-epoch run | **5,620.7 s (93.7 min)** |
| Per epoch | ~187 s |
| Best epoch | 26 |
| Parameters | 9,557,483 |
| In-memory model size | 36.5 MB |
| `best.pt` on disk | 38,255,275 bytes |
| `last.pt` on disk | 114,733,611 bytes |

An earlier idle-machine measurement gave ~130 s per epoch, projecting to ~65 min. The
real run took 93.7 min because the test suite and several Docker builds were competing
for the same GPU. Both numbers are kept because the difference is itself informative:
on an 8 GB shared machine, concurrent work costs about 45% of training throughput.

The eval on the test split took a further 44 s (12,630 images, including nine figures).

Full results: [RESULTS.md](RESULTS.md).

`last.pt` is three times `best.pt` because it carries Adam's two moment buffers per
parameter. `best.pt` is the served artifact.

## Reproducing on a different machine

**CPU only.** Works unchanged. Set `GTSRB_DEVICE=cpu`; expect roughly an order of
magnitude more time per epoch. `torch.amp` is a no-op on CPU, so training is
full-precision.

**CUDA.** Set `GTSRB_DEVICE=cuda`. AMP activates automatically
(`supports_amp()` returns true only for CUDA). Raise `data.num_workers` to 8 and
set `data.pin_memory=true` — both default to the portable values, not the fast ones.

**Apple MPS.** `GTSRB_DEVICE=auto` selects MPS. AMP stays off because there is no
meaningful autocast path for MPS, so training is full-precision. This is the
configuration the verification run used.

**Different torch version.** The lock pins it. A different version may change kernel
selection and therefore results.

## Re-running an experiment honestly

```bash
# Record the run with MLflow, including the resolved config as an artifact.
make train-tracked

# View it.
make mlflow-ui        # http://127.0.0.1:5000

# Compare two runs.
python -m gtsrb.cli.train --set model.name=mlp \
  --set training.checkpoint.directory=artifacts/checkpoints-mlp
```

Two runs are directly comparable when the manifest path, the seed and the split
fraction are identical — which `verify_manifest` enforces, because a manifest whose
dataset length, seed or validation fraction has changed is rejected rather than
silently reused.

## What to check before believing a number

1. `artifacts/splits/gtsrb_split.json` exists and reports `num_train: 21312`.
2. The run log states the device, the seed and `deterministic`.
3. `artifacts/checkpoints/train_result.json` records `best_epoch` and the full
   metric set.
4. `artifacts/checkpoints/best.pt` metadata matches — `unzip -p` is not needed; use
   `python -c "from gtsrb.models import load_checkpoint; print(load_checkpoint('artifacts/checkpoints/best.pt').metadata.summary())"`.
5. The evaluation's `metrics.json` reports the split and sample count it scored on.
