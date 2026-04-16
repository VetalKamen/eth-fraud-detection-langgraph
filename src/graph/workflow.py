from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from langgraph.graph import END, START, StateGraph

from app.artifacts import compute_file_sha256
from app.bootstrap import create_runtime
from app.config import AppSettings
from app.dataset import (
    ingest_dataset,
    load_prepared_dataset_manifest,
    validate_prepared_dataset_against_settings,
)
from app.explainability import (
    ExplanationArtifactManifest,
    compute_explanations,
    generate_feature_attributions,
    load_explanation_manifest,
)
from app.features import (
    load_prepared_dataset,
    load_preprocessor,
    prepare_features,
    validate_preprocessor_against_state,
)
from app.training import (
    _preserve_training_rng_state,
    build_model_checkpoint,
    fit_baseline_model,
    load_model_checkpoint,
    train_baseline_model,
)
from graph.state import PipelineState


def build_pipeline_graph(settings: AppSettings, *, overwrite: bool = False) -> Any:
    graph = StateGraph(PipelineState)
    graph.add_node("ingest_dataset", _wrap_stage(_run_dataset_stage, settings, overwrite))
    graph.add_node("prepare_features", _wrap_stage(_run_feature_stage, settings, overwrite))
    graph.add_node("train_model", _wrap_stage(_run_training_stage, settings, overwrite))
    graph.add_node(
        "generate_explanations", _wrap_stage(_run_explainability_stage, settings, overwrite)
    )
    graph.add_edge(START, "ingest_dataset")
    graph.add_edge("ingest_dataset", "prepare_features")
    graph.add_edge("prepare_features", "train_model")
    graph.add_edge("train_model", "generate_explanations")
    graph.add_edge("generate_explanations", END)
    return graph.compile()


def run_pipeline(
    settings: AppSettings | None = None,
    *,
    initial_state: PipelineState | None = None,
    overwrite: bool = False,
) -> PipelineState:
    runtime = create_runtime(settings)
    graph = build_pipeline_graph(runtime.settings, overwrite=overwrite)
    input_state = initial_state or runtime.initial_state
    output = graph.invoke(input_state)
    return PipelineState.model_validate(output)


def _wrap_stage(
    stage_fn: Any,
    settings: AppSettings,
    overwrite: bool,
) -> Any:
    def node(state_input: PipelineState | dict[str, Any]) -> dict[str, Any]:
        state = _coerce_state(state_input)
        updated_state = stage_fn(state, settings, overwrite)
        return cast(dict[str, Any], updated_state.model_dump(mode="python"))

    return node


def _run_dataset_stage(
    state: PipelineState, settings: AppSettings, overwrite: bool
) -> PipelineState:
    try:
        return ingest_dataset(settings, overwrite=overwrite).state
    except FileExistsError:
        return _recover_dataset_state(state, settings)


def _run_feature_stage(
    state: PipelineState, settings: AppSettings, overwrite: bool
) -> PipelineState:
    try:
        return prepare_features(state, settings, overwrite=overwrite).state
    except FileExistsError:
        return _recover_feature_state(state, settings)


def _run_training_stage(
    state: PipelineState, settings: AppSettings, overwrite: bool
) -> PipelineState:
    try:
        return train_baseline_model(state, settings, overwrite=overwrite).state
    except FileExistsError:
        return _recover_training_state(state, settings)


def _run_explainability_stage(
    state: PipelineState, settings: AppSettings, overwrite: bool
) -> PipelineState:
    try:
        return generate_feature_attributions(state, settings, overwrite=overwrite).state
    except FileExistsError:
        return _recover_explanation_state(state, settings)


def _recover_dataset_state(state: PipelineState, settings: AppSettings) -> PipelineState:
    prepared_path = settings.artifacts.prepared_dataset_path()
    prepared_frame = load_prepared_dataset(prepared_path)
    try:
        manifest = load_prepared_dataset_manifest(prepared_path)
    except FileNotFoundError as exc:
        raise ValueError("Existing prepared dataset metadata is missing or invalid") from exc
    prepared_dataset_fingerprint = compute_file_sha256(prepared_path)
    schema, current_raw_fingerprint = validate_prepared_dataset_against_settings(
        prepared_frame,
        manifest,
        settings,
    )

    return state.model_copy(
        update={
            "stage": "dataset_loaded",
            "raw_dataset_path": settings.dataset.input_path,
            "raw_dataset_fingerprint": current_raw_fingerprint,
            "artifact_dir": settings.artifacts.run_dir(),
            "label_column": schema.label_column,
            "split_column": schema.split_column,
            "feature_columns": schema.feature_columns,
            "prepared_dataset_path": prepared_path,
            "prepared_dataset_fingerprint": prepared_dataset_fingerprint,
        }
    )


def _recover_feature_state(state: PipelineState, settings: AppSettings) -> PipelineState:
    preprocessor_path = settings.artifacts.preprocessor_path()
    preprocessor = load_preprocessor(preprocessor_path)
    preprocessor_fingerprint = compute_file_sha256(preprocessor_path)
    if tuple(preprocessor.feature_columns) != state.feature_columns:
        raise ValueError("Existing preprocessor feature_columns do not match PipelineState")
    if preprocessor.label_column != state.label_column:
        raise ValueError("Existing preprocessor label_column does not match PipelineState")
    if preprocessor.split_column != state.split_column:
        raise ValueError("Existing preprocessor split_column does not match PipelineState")
    prepared_frame = load_prepared_dataset(cast(Path, state.prepared_dataset_path))
    validate_preprocessor_against_state(
        frame=prepared_frame,
        preprocessor=preprocessor,
        state=state,
    )

    return state.model_copy(
        update={
            "stage": "features_ready",
            "preprocessor_path": preprocessor_path,
            "preprocessor_fingerprint": preprocessor_fingerprint,
            "feature_columns": preprocessor.feature_columns,
        }
    )


def _recover_training_state(state: PipelineState, settings: AppSettings) -> PipelineState:
    checkpoint_path = settings.artifacts.checkpoint_path()
    checkpoint = load_model_checkpoint(checkpoint_path)
    checkpoint_fingerprint = compute_file_sha256(checkpoint_path)

    expected_feature_columns = list(state.feature_columns)
    if checkpoint.get("model_name") != "linear-bce-baseline":
        raise ValueError(f"Unsupported checkpoint model_name: {checkpoint.get('model_name')!r}")
    if checkpoint.get("input_dim") != len(state.feature_columns):
        raise ValueError("Existing checkpoint input_dim does not match PipelineState")
    if checkpoint.get("feature_columns") != expected_feature_columns:
        raise ValueError("Existing checkpoint feature_columns do not match PipelineState")
    if checkpoint.get("label_column") != state.label_column:
        raise ValueError("Existing checkpoint label_column does not match PipelineState")
    if checkpoint.get("split_column") != state.split_column:
        raise ValueError("Existing checkpoint split_column does not match PipelineState")
    if checkpoint.get("preprocessor_path") != str(state.preprocessor_path):
        raise ValueError("Existing checkpoint preprocessor_path does not match PipelineState")
    if checkpoint.get("preprocessor_fingerprint") != state.preprocessor_fingerprint:
        raise ValueError(
            "Existing checkpoint preprocessor_fingerprint does not match PipelineState"
        )
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Existing checkpoint metrics are missing or invalid")
    prepared_frame = load_prepared_dataset(cast(Path, state.prepared_dataset_path))
    preprocessor = load_preprocessor(cast(Path, state.preprocessor_path))
    with _preserve_training_rng_state():
        torch.manual_seed(settings.training.seed)
        torch.use_deterministic_algorithms(True)
        expected_model, expected_metrics = fit_baseline_model(
            frame=prepared_frame,
            preprocessor=preprocessor,
            state=state,
            settings=settings,
        )
    expected_checkpoint = build_model_checkpoint(
        model=expected_model,
        metrics=expected_metrics,
        preprocessor=preprocessor,
        state=state,
        settings=settings,
    )
    if checkpoint.get("metrics") != expected_checkpoint["metrics"]:
        raise ValueError(
            "Existing checkpoint metrics do not match the deterministic training output"
        )
    expected_state_dict = expected_checkpoint["model_state_dict"]
    checkpoint_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(checkpoint_state_dict, dict):
        raise ValueError("Existing checkpoint model_state_dict is missing or invalid")
    for name, tensor in expected_state_dict.items():
        if name not in checkpoint_state_dict or not torch.equal(
            cast(torch.Tensor, checkpoint_state_dict[name]),
            tensor,
        ):
            raise ValueError(
                "Existing checkpoint weights do not match the deterministic training output"
            )

    return state.model_copy(
        update={
            "stage": "trained",
            "checkpoint_path": checkpoint_path,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "metrics": cast(dict[str, float], metrics),
        }
    )


def _recover_explanation_state(state: PipelineState, settings: AppSettings) -> PipelineState:
    explanation_path = settings.artifacts.explanation_path()
    if not explanation_path.exists():
        raise FileExistsError("Expected explanation output to exist for graph reuse")

    try:
        manifest = load_explanation_manifest(explanation_path)
    except FileNotFoundError as exc:
        raise ValueError("Existing explanation metadata is missing or invalid") from exc
    explanation_fingerprint = compute_file_sha256(explanation_path)
    expected_frame, expected_manifest = compute_explanations(state, settings)
    try:
        actual_explanation = pd.read_parquet(explanation_path).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_explanation, expected_frame.reset_index(drop=True))
    except AssertionError as exc:
        raise ValueError(
            "Existing explanation artifact does not match the deterministic attribution output"
        ) from exc
    _validate_manifest_for_state(
        manifest,
        state,
        explanation_fingerprint,
        explanation_path,
    )
    if manifest.model_dump(mode="json") != expected_manifest.model_dump(mode="json"):
        raise ValueError(
            "Existing explanation metadata does not match the deterministic attribution output"
        )
    return state.model_copy(
        update={
            "stage": "explained",
            "explanation_path": explanation_path,
            "explanation_fingerprint": explanation_fingerprint,
        }
    )


def _validate_manifest_for_state(
    manifest: ExplanationArtifactManifest,
    state: PipelineState,
    explanation_fingerprint: str,
    explanation_path: Path,
) -> None:
    if manifest.feature_columns != state.feature_columns:
        raise ValueError("Existing explanation manifest feature_columns do not match PipelineState")
    if manifest.label_column != state.label_column:
        raise ValueError("Existing explanation manifest label_column does not match PipelineState")
    if manifest.split_column != state.split_column:
        raise ValueError("Existing explanation manifest split_column does not match PipelineState")
    if manifest.checkpoint_path != str(state.checkpoint_path):
        raise ValueError(
            "Existing explanation manifest checkpoint_path does not match PipelineState"
        )
    if manifest.checkpoint_fingerprint != state.checkpoint_fingerprint:
        raise ValueError(
            "Existing explanation manifest checkpoint_fingerprint does not match PipelineState"
        )
    if manifest.preprocessor_path != str(state.preprocessor_path):
        raise ValueError(
            "Existing explanation manifest preprocessor_path does not match PipelineState"
        )
    if manifest.preprocessor_fingerprint != state.preprocessor_fingerprint:
        raise ValueError(
            "Existing explanation manifest preprocessor_fingerprint does not match PipelineState"
        )


def _coerce_state(state_input: PipelineState | dict[str, Any]) -> PipelineState:
    if isinstance(state_input, PipelineState):
        return state_input
    return PipelineState.model_validate(state_input)
