from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app import AppSettings, create_runtime
from graph import PipelineState


def test_default_settings_provide_stable_paths_and_runtime_bootstrap() -> None:
    settings = AppSettings()

    assert settings.dataset.input_path == Path("data/ethereum_transactions.parquet")
    assert settings.dataset.label_column == "is_fraud"
    assert settings.artifacts.run_dir() == Path("artifacts/local")
    assert settings.artifacts.prepared_dataset_path() == Path(
        "artifacts/local/prepared/dataset.parquet"
    )
    assert settings.artifacts.preprocessor_path() == Path(
        "artifacts/local/models/preprocessor.json"
    )
    assert settings.artifacts.checkpoint_path() == Path("artifacts/local/models/fraud-detector.pt")
    assert settings.artifacts.explanation_path() == Path(
        "artifacts/local/explanations/attributions.parquet"
    )

    runtime = create_runtime(settings)

    assert runtime.initial_state.stage == "configured"
    assert runtime.initial_state.run_id == "local"
    assert runtime.initial_state.raw_dataset_path == settings.dataset.input_path
    assert runtime.initial_state.artifact_dir == settings.artifacts.run_dir()
    assert runtime.initial_state.split_column == settings.dataset.split_column


def test_environment_overrides_apply_to_nested_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ETH_FRAUD_DATASET__INPUT_PATH", "fixtures/train.parquet")
    monkeypatch.setenv("ETH_FRAUD_DATASET__LABEL_COLUMN", "fraud_label")
    monkeypatch.setenv("ETH_FRAUD_ARTIFACTS__RUN_ID", "integration")
    monkeypatch.setenv("ETH_FRAUD_RUNTIME__BATCH_SIZE", "64")

    settings = AppSettings()

    assert settings.dataset.input_path == Path("fixtures/train.parquet")
    assert settings.dataset.label_column == "fraud_label"
    assert settings.artifacts.run_dir() == Path("artifacts/integration")
    assert settings.runtime.batch_size == 64


def test_pipeline_state_rejects_incomplete_stage_transitions() -> None:
    with pytest.raises(ValidationError, match="prepared_dataset_path"):
        PipelineState(
            run_id="run-1",
            stage="dataset_loaded",
            raw_dataset_path=Path("data/raw.parquet"),
            artifact_dir=Path("artifacts/run-1"),
            label_column="is_fraud",
        )

    with pytest.raises(ValidationError, match="checkpoint_path"):
        PipelineState(
            run_id="run-1",
            stage="trained",
            raw_dataset_path=Path("data/raw.parquet"),
            raw_dataset_fingerprint="raw-sha",
            artifact_dir=Path("artifacts/run-1"),
            label_column="is_fraud",
            split_column="split",
            feature_columns=("amount",),
            prepared_dataset_path=Path("artifacts/run-1/prepared/dataset.parquet"),
            prepared_dataset_fingerprint="prepared-sha",
            preprocessor_path=Path("artifacts/run-1/models/preprocessor.json"),
            preprocessor_fingerprint="preprocessor-sha",
            metrics={"auc": 0.92},
        )


def test_pipeline_state_accepts_fully_populated_explained_stage() -> None:
    state = PipelineState(
        run_id="run-1",
        stage="explained",
        raw_dataset_path=Path("data/raw.parquet"),
        raw_dataset_fingerprint="raw-sha",
        artifact_dir=Path("artifacts/run-1"),
        label_column="is_fraud",
        split_column="split",
        feature_columns=("amount", "gas_used"),
        prepared_dataset_path=Path("artifacts/run-1/prepared/dataset.parquet"),
        prepared_dataset_fingerprint="prepared-sha",
        preprocessor_path=Path("artifacts/run-1/models/preprocessor.json"),
        preprocessor_fingerprint="preprocessor-sha",
        checkpoint_path=Path("artifacts/run-1/models/fraud-detector.pt"),
        checkpoint_fingerprint="checkpoint-sha",
        explanation_path=Path("artifacts/run-1/explanations/attributions.parquet"),
        explanation_fingerprint="explanation-sha",
        metrics={"auc": 0.92},
    )

    assert state.stage == "explained"
    assert state.feature_columns == ("amount", "gas_used")
