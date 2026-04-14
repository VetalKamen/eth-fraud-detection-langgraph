from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import (
    AppSettings,
    ArtifactSettings,
    DatasetSettings,
    RuntimeSettings,
    build_feature_matrix,
    fit_feature_preprocessor,
    ingest_dataset,
    load_preprocessor,
    prepare_features,
    transform_features,
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
            "amount": [1.0, 1.2, 5.0, 5.3, 1.5, 1.7, 8.0, 8.4, 2.1, 2.4, 9.5, 9.8],
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


def _build_settings(
    input_path: Path, artifact_root: Path, *, run_id: str = "run-01"
) -> AppSettings:
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
    )


def test_prepare_features_persists_preprocessor_and_updates_state(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")
    ingested = ingest_dataset(settings)

    prepared = prepare_features(ingested.state, settings)

    assert prepared.state.stage == "features_ready"
    assert prepared.state.preprocessor_path == settings.artifacts.preprocessor_path()
    assert prepared.state.preprocessor_path.exists()
    assert prepared.state.preprocessor_fingerprint is not None
    assert prepared.preprocessor.feature_columns == ("amount", "gas_used")
    assert list(prepared.transformed_frame.columns) == list(ingested.prepared_frame.columns)
    assert prepared.transformed_frame["amount"].notna().all()
    assert prepared.transformed_frame["gas_used"].notna().all()


def test_feature_preprocessor_uses_only_training_statistics() -> None:
    prepared_frame = pd.DataFrame(
        {
            "address": ["0xa1", "0xb2", "0xc3", "0xd4"],
            "is_fraud": [0, 1, 0, 1],
            "amount": [0.0, 2.0, 100.0, 200.0],
            "gas_used": [10.0, 14.0, 500.0, 700.0],
            "split": ["train", "train", "validation", "test"],
        }
    )

    preprocessor = fit_feature_preprocessor(
        prepared_frame,
        feature_columns=("amount", "gas_used"),
        split_column="split",
        label_column="is_fraud",
    )

    assert preprocessor.means == {"amount": 1.0, "gas_used": 12.0}
    assert preprocessor.scales == {"amount": 1.0, "gas_used": 2.0}


def test_transform_features_is_deterministic_after_reload(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")
    ingested = ingest_dataset(settings)
    prepared = prepare_features(ingested.state, settings)
    assert prepared.state.preprocessor_path is not None

    loaded_preprocessor = load_preprocessor(prepared.state.preprocessor_path)

    expected = transform_features(ingested.prepared_frame, prepared.preprocessor)
    actual = transform_features(ingested.prepared_frame, loaded_preprocessor)
    pd.testing.assert_frame_equal(expected, actual)


def test_build_feature_matrix_uses_persisted_feature_order_not_dataframe_order(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")
    ingested = ingest_dataset(settings)
    prepared = prepare_features(ingested.state, settings)

    reordered = ingested.prepared_frame[["gas_used", "is_fraud", "address", "split", "amount"]]
    matrix = build_feature_matrix(reordered, prepared.preprocessor)

    expected = transform_features(ingested.prepared_frame, prepared.preprocessor).loc[
        :, ["amount", "gas_used"]
    ]
    np.testing.assert_allclose(matrix, expected.to_numpy(dtype=np.float32))


def test_prepare_features_fails_closed_when_preprocessor_exists(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")
    ingested = ingest_dataset(settings)

    prepare_features(ingested.state, settings)

    with pytest.raises(FileExistsError, match="Preprocessor artifact already exists"):
        prepare_features(ingested.state, settings)


def test_prepare_features_rejects_state_without_feature_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")
    ingested = ingest_dataset(settings)
    invalid_state = ingested.state.model_copy(update={"feature_columns": ()})

    with pytest.raises(ValueError, match="feature_columns before feature preparation"):
        prepare_features(invalid_state, settings)
