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
    TrainingRun,
    TrainingSettings,
    generate_feature_attributions,
    ingest_dataset,
    load_explanation_manifest,
    load_model_checkpoint,
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
        runtime=RuntimeSettings(explain_sample_size=8),
        training=TrainingSettings(seed=11, epochs=40, learning_rate=0.1, weight_decay=0.0),
    )


def _train_pipeline(
    raw_path: Path, artifact_root: Path, *, run_id: str
) -> tuple[AppSettings, TrainingRun]:
    settings = _build_settings(raw_path, artifact_root, run_id=run_id)
    ingested = ingest_dataset(settings)
    prepared = prepare_features(ingested.state, settings)
    trained = train_baseline_model(prepared.state, settings)
    return settings, trained


def test_generate_feature_attributions_persists_aligned_outputs(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings, trained = _train_pipeline(raw_path, tmp_path / "artifacts", run_id="run-01")

    explanation_run = generate_feature_attributions(trained.state, settings)

    assert explanation_run.state.stage == "explained"
    assert explanation_run.explanation_path == settings.artifacts.explanation_path()
    assert explanation_run.explanation_path.exists()
    assert explanation_run.manifest.feature_columns == ("amount", "gas_used")
    assert (
        explanation_run.manifest.preprocessor_fingerprint == trained.state.preprocessor_fingerprint
    )
    assert explanation_run.manifest.checkpoint_fingerprint == trained.state.checkpoint_fingerprint
    assert list(explanation_run.attributions_frame.columns[-2:]) == [
        "attribution__amount",
        "attribution__gas_used",
    ]
    loaded_manifest = load_explanation_manifest(explanation_run.explanation_path)
    assert loaded_manifest.feature_columns == ("amount", "gas_used")
    persisted_frame = pd.read_parquet(explanation_run.explanation_path)
    pd.testing.assert_frame_equal(
        persisted_frame, explanation_run.attributions_frame, check_like=False
    )


def test_generate_feature_attributions_rejects_mismatched_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings, trained = _train_pipeline(raw_path, tmp_path / "artifacts", run_id="run-01")
    assert trained.state.checkpoint_path is not None
    checkpoint = load_model_checkpoint(trained.state.checkpoint_path)
    checkpoint["feature_columns"] = ["gas_used", "amount"]
    torch.save(checkpoint, trained.state.checkpoint_path)

    try:
        generate_feature_attributions(trained.state, settings, overwrite=True)
    except ValueError as exc:
        assert "feature_columns" in str(exc)
    else:
        raise AssertionError("expected mismatched checkpoint metadata to be rejected")


def test_generate_feature_attributions_is_deterministic_and_row_order_invariant(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.parquet"
    reordered_path = tmp_path / "raw-reordered.parquet"
    raw_frame = _make_raw_dataset()
    raw_frame.to_parquet(raw_path, index=False)
    raw_frame.sample(frac=1.0, random_state=23).reset_index(drop=True).to_parquet(
        reordered_path,
        index=False,
    )

    settings_a, trained_a = _train_pipeline(raw_path, tmp_path / "artifacts-a", run_id="run-a")
    settings_b, trained_b = _train_pipeline(
        reordered_path, tmp_path / "artifacts-b", run_id="run-b"
    )

    run_a = generate_feature_attributions(trained_a.state, settings_a)
    run_b = generate_feature_attributions(trained_b.state, settings_b)

    cols = ["split", "label", "attribution__amount", "attribution__gas_used"]
    pd.testing.assert_frame_equal(
        run_a.attributions_frame.loc[:, cols].reset_index(drop=True),
        run_b.attributions_frame.loc[:, cols].reset_index(drop=True),
    )


def test_integrated_gradients_tracks_logit_difference_for_linear_model(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings, trained = _train_pipeline(raw_path, tmp_path / "artifacts", run_id="run-01")

    explanation_run = generate_feature_attributions(trained.state, settings)

    attribution_sum = explanation_run.attributions_frame[
        ["attribution__amount", "attribution__gas_used"]
    ].sum(axis=1)
    logit_difference = (
        explanation_run.attributions_frame["predicted_logit"]
        - explanation_run.attributions_frame["baseline_logit"]
    )
    np.testing.assert_allclose(attribution_sum.to_numpy(), logit_difference.to_numpy(), atol=1e-4)


def test_generate_feature_attributions_fails_closed_when_artifact_exists(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _make_raw_dataset().to_parquet(raw_path, index=False)
    settings, trained = _train_pipeline(raw_path, tmp_path / "artifacts", run_id="run-01")

    generate_feature_attributions(trained.state, settings)

    try:
        generate_feature_attributions(trained.state, settings)
    except FileExistsError as exc:
        assert "Explanation artifact already exists" in str(exc)
    else:
        raise AssertionError("expected explanation artifact overwrite to fail closed")
