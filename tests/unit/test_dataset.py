from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app import (
    AppSettings,
    ArtifactSettings,
    DatasetSettings,
    RuntimeSettings,
    assign_splits,
    ingest_dataset,
)


def _make_dataset() -> pd.DataFrame:
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
            "amount": [1.5, 1.7, 9.2, 9.8, 2.0, 2.2, 12.0, 12.4, 3.1, 3.4, 15.3, 15.5],
            "gas_used": [
                21_000,
                22_000,
                45_000,
                44_500,
                23_000,
                23_500,
                51_000,
                50_500,
                24_000,
                25_000,
                60_000,
                61_000,
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


def test_ingest_dataset_writes_prepared_output_and_updates_state(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")

    result = ingest_dataset(settings)

    assert result.schema.feature_columns == ("amount", "gas_used")
    assert result.state.stage == "dataset_loaded"
    assert result.state.prepared_dataset_path == settings.artifacts.prepared_dataset_path()
    assert result.state.feature_columns == ("amount", "gas_used")
    assert result.state.split_column == "split"
    assert result.state.prepared_dataset_path.exists()
    assert set(result.prepared_frame["split"]) == {"train", "validation", "test"}
    assert result.prepared_frame.groupby("address")["split"].nunique().max() == 1

    persisted_frame = pd.read_parquet(result.state.prepared_dataset_path)
    pd.testing.assert_frame_equal(persisted_frame, result.prepared_frame, check_like=False)


def test_ingest_dataset_rejects_missing_expected_feature_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_dataset().drop(columns=["gas_used"]).to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")

    with pytest.raises(ValueError, match="missing required columns: gas_used"):
        ingest_dataset(settings)


def test_assign_splits_is_reproducible_when_input_row_order_changes() -> None:
    original = _make_dataset()
    reordered = original.sample(frac=1.0, random_state=19).reset_index(drop=True)

    assigned_original = assign_splits(
        original,
        entity_id_column="address",
        split_column="split",
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=7,
    )
    assigned_reordered = assign_splits(
        reordered,
        entity_id_column="address",
        split_column="split",
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=7,
    )

    expected = assigned_original[["address", "split"]].drop_duplicates().sort_values("address")
    actual = assigned_reordered[["address", "split"]].drop_duplicates().sort_values("address")
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))


def test_ingest_dataset_rejects_non_binary_labels(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    invalid = _make_dataset()
    invalid.loc[0, "is_fraud"] = 2
    invalid.to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")

    with pytest.raises(ValueError, match="binary labels"):
        ingest_dataset(settings)


def test_ingest_dataset_rejects_datasets_that_cannot_produce_validation_split(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.parquet"
    tiny = _make_dataset().iloc[:4].reset_index(drop=True)
    tiny.to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")

    with pytest.raises(ValueError, match="missing required splits: validation"):
        ingest_dataset(settings)


def test_ingest_dataset_drops_unconfigured_extra_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    frame = _make_dataset()
    frame["wallet_owner_email"] = [f"user{i}@example.com" for i in range(len(frame))]
    frame.to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")

    result = ingest_dataset(settings)

    assert "wallet_owner_email" not in result.prepared_frame.columns
    assert list(result.prepared_frame.columns) == [
        "address",
        "is_fraud",
        "amount",
        "gas_used",
        "split",
    ]


def test_ingest_dataset_requires_explicit_feature_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_dataset().to_parquet(raw_path, index=False)
    settings = AppSettings(
        dataset=DatasetSettings(
            input_path=raw_path,
            label_column="is_fraud",
            entity_id_column="address",
            random_seed=7,
        ),
        artifacts=ArtifactSettings(root_dir=tmp_path / "artifacts", run_id="run-01"),
        runtime=RuntimeSettings(),
    )

    with pytest.raises(ValueError, match="feature_columns must be configured explicitly"):
        ingest_dataset(settings)


def test_ingest_dataset_fails_closed_when_prepared_artifact_exists(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts")

    ingest_dataset(settings)

    with pytest.raises(FileExistsError, match="Prepared dataset already exists"):
        ingest_dataset(settings)
