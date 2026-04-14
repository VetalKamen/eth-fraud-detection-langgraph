from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app import (
    AppSettings,
    ArtifactSettings,
    DatasetSettings,
    RuntimeSettings,
    TrainingSettings,
    compute_binary_classification_metrics,
    ingest_dataset,
    load_model_checkpoint,
    persist_preprocessor,
    prepare_features,
    train_baseline_model,
)


def _make_raw_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "address": [
                "0xa1",
                "0xa1",
                "0xb2",
                "0xb2",
                "0xc3",
                "0xc3",
                "0xd4",
                "0xd4",
                "0xe5",
                "0xe5",
                "0xf6",
                "0xf6",
            ],
            "is_fraud": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "amount": [1.0, 1.2, 5.0, 5.4, 1.5, 1.7, 8.0, 8.5, 2.1, 2.3, 9.5, 9.9],
            "gas_used": [
                21_000,
                22_000,
                41_000,
                42_000,
                23_000,
                24_000,
                49_000,
                48_000,
                25_000,
                26_000,
                58_000,
                57_000,
            ],
        }
    )


def _build_settings(input_path: Path, artifact_root: Path, *, run_id: str) -> AppSettings:
    return AppSettings(
        dataset=DatasetSettings(
            input_path=input_path,
            label_column="is_fraud",
            entity_id_column="address",
            feature_columns=("amount", "gas_used"),
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=7,
        ),
        artifacts=ArtifactSettings(root_dir=artifact_root, run_id=run_id),
        runtime=RuntimeSettings(),
        training=TrainingSettings(seed=11, epochs=40, learning_rate=0.1, weight_decay=0.0),
    )


def test_train_baseline_model_writes_checkpoint_and_updates_state(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts", run_id="run-01")
    ingested = ingest_dataset(settings)
    prepared = prepare_features(ingested.state, settings)

    training_run = train_baseline_model(prepared.state, settings)

    assert training_run.state.stage == "trained"
    assert training_run.state.checkpoint_path == settings.artifacts.checkpoint_path()
    assert training_run.state.checkpoint_path.exists()
    assert training_run.state.preprocessor_fingerprint is not None
    assert training_run.state.checkpoint_fingerprint is not None
    assert set(training_run.metrics) >= {
        "train_loss",
        "validation_loss",
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
    }
    checkpoint = load_model_checkpoint(training_run.state.checkpoint_path)
    assert checkpoint["model_name"] == "linear-bce-baseline"
    assert checkpoint["input_dim"] == 2
    assert checkpoint["feature_columns"] == ["amount", "gas_used"]
    assert checkpoint["label_column"] == "is_fraud"
    assert checkpoint["preprocessor_path"] == str(settings.artifacts.preprocessor_path())
    assert checkpoint["preprocessor_fingerprint"] == training_run.state.preprocessor_fingerprint
    assert checkpoint["model_state_dict"]["linear.weight"].shape == (1, 2)


def test_train_baseline_model_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)

    settings_a = _build_settings(raw_path, tmp_path / "artifacts-a", run_id="run-a")
    settings_b = _build_settings(raw_path, tmp_path / "artifacts-b", run_id="run-b")

    run_a = train_baseline_model(
        prepare_features(ingest_dataset(settings_a).state, settings_a).state,
        settings_a,
    )
    run_b = train_baseline_model(
        prepare_features(ingest_dataset(settings_b).state, settings_b).state,
        settings_b,
    )

    checkpoint_a = load_model_checkpoint(run_a.checkpoint_path)
    checkpoint_b = load_model_checkpoint(run_b.checkpoint_path)
    assert run_a.metrics == run_b.metrics
    for tensor_name in checkpoint_a["model_state_dict"]:
        assert torch.equal(
            checkpoint_a["model_state_dict"][tensor_name],
            checkpoint_b["model_state_dict"][tensor_name],
        )


def test_compute_binary_classification_metrics_on_small_fixture() -> None:
    probabilities = np.array([0.9, 0.7, 0.4, 0.2], dtype=np.float32)
    labels = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    metrics = compute_binary_classification_metrics(probabilities, labels, prefix="validation")

    assert metrics == {
        "validation_accuracy": 1.0,
        "validation_precision": 1.0,
        "validation_recall": 1.0,
    }


def test_train_baseline_model_is_invariant_to_raw_row_order(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    reordered_path = tmp_path / "raw-reordered.parquet"
    raw_frame = _make_raw_dataset()
    raw_frame.to_parquet(raw_path, index=False)
    raw_frame.sample(frac=1.0, random_state=23).reset_index(drop=True).to_parquet(
        reordered_path,
        index=False,
    )

    settings_a = _build_settings(raw_path, tmp_path / "artifacts-a", run_id="run-a")
    settings_b = _build_settings(reordered_path, tmp_path / "artifacts-b", run_id="run-b")

    run_a = train_baseline_model(
        prepare_features(ingest_dataset(settings_a).state, settings_a).state,
        settings_a,
    )
    run_b = train_baseline_model(
        prepare_features(ingest_dataset(settings_b).state, settings_b).state,
        settings_b,
    )

    assert run_a.metrics == run_b.metrics


def test_train_baseline_model_rejects_mismatched_preprocessor(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts", run_id="run-01")
    ingested = ingest_dataset(settings)
    prepared = prepare_features(ingested.state, settings)
    assert prepared.state.preprocessor_path is not None

    mismatched = prepared.preprocessor.model_copy(update={"split_column": "wrong_split"})
    persist_preprocessor(mismatched, prepared.state.preprocessor_path, overwrite=True)

    try:
        train_baseline_model(prepared.state, settings, overwrite=True)
    except ValueError as exc:
        assert "split_column" in str(exc)
    else:
        raise AssertionError("expected mismatched preprocessor to be rejected")


def test_train_baseline_model_restores_deterministic_torch_flag(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts", run_id="run-01")
    ingested = ingest_dataset(settings)
    prepared = prepare_features(ingested.state, settings)
    previous = torch.are_deterministic_algorithms_enabled()

    try:
        torch.use_deterministic_algorithms(False)
        train_baseline_model(prepared.state, settings)
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(previous)
