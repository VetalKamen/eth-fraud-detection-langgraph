# eth-fraud-detection-langgraph

Ethereum fraud detection pipeline with typed configuration, deterministic data preparation, persisted preprocessing, baseline PyTorch training, Captum attributions, and LangGraph orchestration.

## Prerequisites

- Python 3.11 or newer available as `python3`
- Enough local disk space for the repo-local virtualenv under `local_test/.venv`

## Bootstrap

The canonical local bootstrap path is repo-local and does not depend on globally installed `ruff`, `mypy`, `pytest`, `langgraph`, or `captum`.

```bash
make bootstrap
```

This creates or updates `local_test/.venv` and installs the project plus all declared dev dependencies.

## Commands

- `make format`: format the repo
- `make format-check`: verify formatting without changing files
- `make lint`: run Ruff on `src` and `tests`
- `make type`: run strict mypy on `src` and `tests`
- `make test-unit`: run unit tests only
- `make test-integration`: run integration tests only
- `make test`: run the full test suite
- `make check`: run format check, lint, typecheck, and the full test suite

## CI

CI uses the same command surface as local development:

```bash
make bootstrap
make format-check
make lint
make type
make test
```

## Current Pipeline Surface

- `src/app/dataset.py`: explicit-schema parquet ingestion with deterministic entity-level splits
- `src/app/features.py`: train-only preprocessing fit plus persisted feature-order contract
- `src/app/training.py`: deterministic baseline linear classifier with self-describing checkpoints
- `src/app/explainability.py`: Captum Integrated Gradients with checkpoint/preprocessor fingerprint validation
- `src/graph/workflow.py`: linear LangGraph pipeline with validated artifact reuse and stale-input rejection

## Reproducibility Notes

- Artifact reuse is fail-closed and fingerprint-aware.
- Prepared datasets and explanation outputs embed their provenance metadata directly in parquet instead of relying on sidecar manifest files.
- Reusing the same `run_id` with changed raw input is rejected by the orchestration layer.
- The repo-local venv in `local_test/` is the supported verification environment for now.
