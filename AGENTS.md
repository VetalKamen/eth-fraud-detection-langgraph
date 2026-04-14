# AGENTS.md

## Project Summary
- Intended app purpose, inferred from `pyproject.toml`: Ethereum fraud detection using PyTorch for modeling, Captum for explainability, and LangGraph for orchestration.
- Current state: the planned repository loop is fully initialized. The repo now has a working data-to-explanations pipeline, LangGraph orchestration, repo-local bootstrap/check commands shared by CI and local development, and atomic publication across the current single-file artifact model.
- Runtime boundaries currently present in code:
  - `src/app/`: application-facing bootstrap and settings boundary.
  - `src/graph/`: orchestration state boundary for future LangGraph nodes.
  - `.github/workflows/ci.yml`: CI boundary for format, lint, typecheck, and test.
## Correctness Notes
- Prepared datasets and explanation outputs now embed provenance in parquet metadata instead of relying on sidecar manifest files.
- JSON/text and checkpoint outputs use the same atomic temp-file-and-replace helper.
- Rerun reuse remains provenance-aware and fail-closed on fingerprint or metadata drift.

## Verified Commands
- Install: `make install`
  - Repo-defined underlying command: `python -m pip install -U pip && pip install -e ".[dev]"`
  - Verified from `Makefile` and `.github/workflows/ci.yml`
- Format: `make format`
- Format check: `ruff format --check .`
- Lint: `make lint`
- Typecheck: `make type`
- Test: `make test`
- Build: no build command is defined in this repo
- Docker: no Dockerfile or compose configuration is present in this repo

## Local Environment Notes
- `make test` passes in the current shell.
- `make lint` and `make type` currently fail locally because `ruff` and `mypy` are not installed yet.
- The current shell exposes `python3`, not `python`; keep the repo-declared install command unchanged unless the project standard changes.
- An isolated toolchain is available under `local_test/.venv/` for verification without modifying the system Python.

## Repo Layout
- `src/app/config.py`: typed settings for dataset, artifacts, and runtime controls
- `src/app/bootstrap.py`: single bootstrap API returning settings plus initial pipeline state
- `src/app/dataset.py`: parquet ingestion, schema validation, deterministic entity-level splits, and prepared dataset persistence
- `src/app/features.py`: train-split-only preprocessing fit, transform-only inference helpers, stable feature ordering, and persisted preprocessor metadata
- `src/app/training.py`: deterministic baseline linear classifier training, metrics, and self-describing checkpoint persistence
- `src/app/explainability.py`: Captum Integrated Gradients on logits with checkpoint/preprocessor fingerprint validation and provenance embedded in explanation parquet metadata
- `src/app/artifacts.py`: atomic temp-file publication helpers plus parquet metadata utilities
- `src/graph/state.py`: validated state contract for future LangGraph nodes
- `src/graph/workflow.py`: linear LangGraph pipeline with validated artifact reuse and stale-input rejection on rerun
- `Makefile`: repo-local bootstrap and quality gates through `local_test/.venv`
- `tests/unit/test_config.py`: coverage for defaults, environment overrides, and stage validation
- `tests/unit/test_dataset.py`: coverage for prepared output, missing columns, split determinism, explicit feature allowlists, invalid labels, and fail-closed artifact overwrites
- `tests/unit/test_features.py`: coverage for persisted preprocessors, train-only fitting, stable feature matrix ordering, and fail-closed preprocessor overwrites
- `tests/unit/test_training.py`: coverage for checkpoint metadata, deterministic model weights and metrics, row-order invariance, stale preprocessor rejection, and torch deterministic-mode restoration
- `tests/unit/test_explainability.py`: coverage for aligned attribution outputs, checkpoint/preprocessor drift rejection, IG completeness on the linear baseline, determinism, and single-file overwrite guards
- `tests/unit/test_artifacts.py`: coverage for atomic text, parquet, binary/checkpoint, cleanup-on-error, and overwrite-disabled writes
- `tests/integration/test_graph_pipeline.py`: end-to-end graph execution, idempotent rerun reuse, and raw-input drift invalidation
- `.github/workflows/ci.yml`: Python 3.11 CI workflow using the same command set as the Makefile

## Next Action
- No blocking implementation task remains for the current repository goal set.
