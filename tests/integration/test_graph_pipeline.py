from __future__ import annotations

from pathlib import Path

import pandas as pd

from app import (
    AppSettings,
    ArtifactSettings,
    DatasetSettings,
    RuntimeSettings,
    TrainingSettings,
    load_explanation_manifest,
    run_pipeline,
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
        runtime=RuntimeSettings(explain_sample_size=8),
        training=TrainingSettings(seed=11, epochs=40, learning_rate=0.1, weight_decay=0.0),
    )


def test_run_pipeline_executes_end_to_end_and_is_idempotent(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts", run_id="graph-run")

    first_state = run_pipeline(settings)
    tracked_paths = [
        settings.artifacts.prepared_dataset_path(),
        settings.artifacts.preprocessor_path(),
        settings.artifacts.checkpoint_path(),
        settings.artifacts.explanation_path(),
    ]
    tracked_mtimes = {path: path.stat().st_mtime_ns for path in tracked_paths}

    second_state = run_pipeline(settings)

    assert first_state.stage == "explained"
    assert second_state.stage == "explained"
    assert second_state.preprocessor_fingerprint == first_state.preprocessor_fingerprint
    assert second_state.checkpoint_fingerprint == first_state.checkpoint_fingerprint
    assert second_state.explanation_path == settings.artifacts.explanation_path()
    assert load_explanation_manifest(settings.artifacts.explanation_path()).feature_columns == (
        "amount",
        "gas_used",
    )
    assert all(path.exists() for path in tracked_paths)
    assert {path: path.stat().st_mtime_ns for path in tracked_paths} == tracked_mtimes


def test_run_pipeline_rejects_reuse_after_raw_input_changes(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    raw_frame = _make_raw_dataset()
    raw_frame.to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts", run_id="graph-run")

    run_pipeline(settings)

    mutated = raw_frame.copy()
    mutated.loc[0, "amount"] = 999.0
    mutated.to_parquet(raw_path, index=False)

    try:
        run_pipeline(settings)
    except ValueError as exc:
        assert "raw_dataset_fingerprint" in str(exc)
    else:
        raise AssertionError("expected raw input drift to invalidate graph reuse")


def test_run_pipeline_rejects_tampered_prepared_dataset(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts", run_id="graph-run")

    run_pipeline(settings)

    tampered = pd.read_parquet(settings.artifacts.prepared_dataset_path())
    tampered.loc[0, "amount"] = -999.0
    tampered.to_parquet(settings.artifacts.prepared_dataset_path(), index=False)

    try:
        run_pipeline(settings)
    except ValueError as exc:
        assert "prepared dataset" in str(exc).lower()
    else:
        raise AssertionError("expected tampered prepared dataset to invalidate graph reuse")


def test_run_pipeline_rejects_tampered_explanation_output(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings = _build_settings(raw_path, tmp_path / "artifacts", run_id="graph-run")

    run_pipeline(settings)

    tampered = pd.read_parquet(settings.artifacts.explanation_path())
    tampered.loc[0, "attribution__amount"] = 1234.0
    tampered.to_parquet(settings.artifacts.explanation_path(), index=False)

    try:
        run_pipeline(settings)
    except ValueError as exc:
        assert "explanation" in str(exc).lower()
    else:
        raise AssertionError("expected tampered explanation output to invalidate graph reuse")
