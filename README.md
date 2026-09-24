# Production-Ready Traffic Sign Recognition (GTSRB)

End-to-end computer vision system for German traffic-sign classification, built
around reproducible PyTorch training, MLflow experiment tracking, structured
evaluation and a containerised FastAPI inference service.

> **Status:** under active refactor. The original analysis was a single Colab
> notebook; it is being rebuilt as an installable package. See
> [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md) for what existed, what was
> wrong with it and what is replacing it. This README is expanded in the
> documentation phase of that work.

## What is here so far

| Path | Purpose |
| --- | --- |
| `src/gtsrb/config/` | Validated Pydantic v2 configuration, YAML loading and the canonical 43-class label space |
| `src/gtsrb/runtime.py` | Device resolution, deterministic seeding and structured logging |
| `configs/` | `data.yaml`, `model.yaml`, `train.yaml`, `api.yaml` |
| `docs/PROJECT_AUDIT.md` | Evidence-based audit and migration plan |
| `docs/images/` | Real evaluation artefacts from the original experiment |

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:irajput215/GTSRB-Traffic-Sign-Recognition-Deep-Learning-Project-.git
cd GTSRB-Traffic-Sign-Recognition-Deep-Learning-Project-
make install          # dev tooling
make install-all      # plus MLflow and FastAPI extras
```

## Inspect the resolved configuration

```bash
make config
```

## Quality gate

```bash
make check            # lint + format check + typecheck + tests
```

## Licence

MIT. The GTSRB dataset is distributed separately and is not included here.
