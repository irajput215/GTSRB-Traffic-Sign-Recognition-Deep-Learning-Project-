# GTSRB traffic sign recognition - developer entry points.
#
# Everything runs through `uv`, which pins the interpreter and installs the
# locked dependency set, so these commands behave the same on a laptop and in CI.
# Run `make` or `make help` for the list.

.DEFAULT_GOAL := help
.PHONY: help install install-all lock lint format format-check typecheck test test-unit \
        test-integration coverage check config data data-stats validate-configs train train-mlp \
        train-resnet smoke evaluate evaluate-val predict clean clean-artifacts pre-commit \
        docker-build docker-run docker-up docker-down mlflow-ui

UV ?= uv
PYTHON ?= $(UV) run python
SRC := src/gtsrb
TESTS := tests

## help: show this message
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /' | sort

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

## install: install the package with dev tooling (no MLflow, no API extras)
install:
	$(UV) sync --extra dev

## install-all: install everything, including MLflow tracking and the API extras
install-all:
	$(UV) sync --all-extras

## lock: refresh uv.lock without installing
lock:
	$(UV) lock

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

## lint: run Ruff over the library, tests and scripts
lint:
	$(UV) run ruff check $(SRC) $(TESTS) scripts

## format: apply Ruff formatting and import sorting
format:
	$(UV) run ruff format $(SRC) $(TESTS) scripts
	$(UV) run ruff check --fix $(SRC) $(TESTS) scripts

## format-check: verify formatting without writing (used by CI)
format-check:
	$(UV) run ruff format --check $(SRC) $(TESTS) scripts

## typecheck: run mypy over the library, tests and scripts
typecheck:
	$(UV) run mypy

## pre-commit: run all pre-commit hooks over every file
pre-commit:
	$(UV) run pre-commit run --all-files

## check: lint, format check, typecheck and test - the full local CI gate
check: lint format-check typecheck test

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

## test: run the full test suite
test:
	$(UV) run pytest

## test-unit: run only the fast unit tests
test-unit:
	$(UV) run pytest -m unit

## test-integration: run only the integration tests
test-integration:
	$(UV) run pytest -m integration

## coverage: run the suite with a coverage report
coverage:
	$(UV) run pytest --cov=gtsrb --cov-report=term-missing --cov-report=xml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

## config: print the fully resolved configuration (train + api layers)
config:
	$(PYTHON) -c "from pathlib import Path; \
from gtsrb.config import load_config; \
print(load_config([Path('configs/data.yaml'), Path('configs/model.yaml'), Path('configs/train.yaml'), Path('configs/api.yaml')]).model_dump_json(indent=2))"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

## data: download GTSRB, validate it and write the train/validation split manifest
data:
	$(PYTHON) scripts/prepare_data.py

## data-stats: re-measure the channel normalisation statistics and compare them
data-stats:
	$(PYTHON) scripts/prepare_data.py --stats --stats-sample-size 4000

## validate-configs: check every shipped YAML layer against the schema
validate-configs:
	$(PYTHON) scripts/validate_configs.py

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

## train: train the default model (compact CNN, 30 epochs)
train:
	$(PYTHON) -m gtsrb.cli.train

## train-mlp: train the MLP baseline from the original three-way comparison
train-mlp:
	$(PYTHON) -m gtsrb.cli.train --set model.name=mlp

## train-resnet: train the ResNet-50 transfer-learning model (256x256, slower)
train-resnet:
	$(PYTHON) -m gtsrb.cli.train --set model.name=resnet50 --set data.image_size=256 \
		--set training.batch_size=32 --set training.optimizer.lr=5e-3

## smoke: one epoch on the default model, to check the pipeline end to end
smoke:
	$(PYTHON) -m gtsrb.cli.train --set training.epochs=1 --set training.batch_size=64 \
		--set training.early_stopping.enabled=false

## train-tracked: train with MLflow tracking and register the resulting model
train-tracked:
	$(PYTHON) -m gtsrb.cli.train --set tracking.enabled=true \
		--set tracking.register_model=true

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

## evaluate: evaluate the default checkpoint on the test split
evaluate:
	$(PYTHON) -m gtsrb.cli.evaluate --split test

## evaluate-val: evaluate the default checkpoint on the validation split
evaluate-val:
	$(PYTHON) -m gtsrb.cli.evaluate --split val

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

## predict: classify one image with the default checkpoint
predict:
	$(PYTHON) -m gtsrb.cli.predict --image $(IMAGE) --json

# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------

## mlflow-ui: serve the MLflow UI for the local tracking database
mlflow-ui:
	$(UV) run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

## clean: remove tool caches and compiled Python
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

## clean-artifacts: remove generated checkpoints, reports and MLflow runs
clean-artifacts:
	rm -rf artifacts mlruns
