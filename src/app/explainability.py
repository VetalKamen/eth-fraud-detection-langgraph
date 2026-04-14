from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from app.artifacts import (
    atomic_write_parquet,
    compute_file_sha256,
    encode_json_metadata,
    read_parquet_json_metadata,
)
from app.config import AppSettings
from app.features import build_feature_matrix, load_prepared_dataset, load_preprocessor
from app.training import load_model_checkpoint, load_model_from_checkpoint
from graph.state import PipelineState

ExplanationSplit = Literal["train", "validation", "test"]
FloatArray = NDArray[np.float32] | NDArray[np.float64]
_EXPLANATION_MANIFEST_KEY = "eth_fraud.explanation_manifest"


class ExplanationArtifactManifest(BaseModel):
    """Metadata persisted alongside attributions to prevent artifact drift."""

    model_config = ConfigDict(frozen=True)

    method: str
    baseline_kind: str
    feature_columns: tuple[str, ...]
    label_column: str
    split_column: str
    input_dim: int
    model_name: str
    checkpoint_path: str
    checkpoint_fingerprint: str
    preprocessor_path: str
    preprocessor_fingerprint: str


@dataclass(frozen=True, slots=True)
class ExplanationRun:
    """Generated attributions plus the updated pipeline state."""

    attributions_frame: pd.DataFrame
    manifest: ExplanationArtifactManifest
    explanation_path: Path
    state: PipelineState


def compute_explanations(
    state: PipelineState,
    settings: AppSettings,
    *,
    split: ExplanationSplit = "validation",
) -> tuple[pd.DataFrame, ExplanationArtifactManifest]:
    if state.prepared_dataset_path is None:
        raise ValueError("PipelineState must include prepared_dataset_path before explainability")
    if state.preprocessor_path is None:
        raise ValueError("PipelineState must include preprocessor_path before explainability")
    if state.preprocessor_fingerprint is None:
        raise ValueError(
            "PipelineState must include preprocessor_fingerprint before explainability"
        )
    if state.checkpoint_path is None:
        raise ValueError("PipelineState must include checkpoint_path before explainability")
    if state.checkpoint_fingerprint is None:
        raise ValueError("PipelineState must include checkpoint_fingerprint before explainability")
    if not state.feature_columns:
        raise ValueError("PipelineState must include feature_columns before explainability")

    prepared_frame = load_prepared_dataset(state.prepared_dataset_path)
    preprocessor = load_preprocessor(state.preprocessor_path)
    checkpoint = load_model_checkpoint(state.checkpoint_path)
    preprocessor_fingerprint = compute_file_sha256(state.preprocessor_path)
    checkpoint_fingerprint = compute_file_sha256(state.checkpoint_path)
    _validate_explainability_contract(
        preprocessor=preprocessor,
        checkpoint=checkpoint,
        state=state,
        preprocessor_fingerprint=preprocessor_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )

    selection = prepared_frame.loc[prepared_frame[state.split_column] == split].copy()
    if selection.empty:
        raise ValueError(f"Prepared dataset does not contain any rows for split {split!r}")
    selection = selection.sort_index().head(settings.runtime.explain_sample_size).copy()

    feature_matrix = build_feature_matrix(selection, preprocessor)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Feature matrix contains non-finite values; cannot generate explanations")

    model = load_model_from_checkpoint(state.checkpoint_path)
    model.eval()

    input_tensor = torch.from_numpy(feature_matrix)
    baseline_tensor = torch.zeros_like(input_tensor)
    integrated_gradients = IntegratedGradients(model)
    attributions_tensor, deltas_tensor = cast(
        tuple[torch.Tensor, torch.Tensor],
        integrated_gradients.attribute(
            input_tensor,
            baselines=baseline_tensor,
            return_convergence_delta=True,
        ),
    )

    with torch.no_grad():
        logits_tensor = model(input_tensor).squeeze(1)
        baseline_logits_tensor = model(baseline_tensor).squeeze(1)
        probabilities_tensor = torch.sigmoid(logits_tensor)

    manifest = ExplanationArtifactManifest(
        method="integrated_gradients",
        baseline_kind="zero",
        feature_columns=tuple(preprocessor.feature_columns),
        label_column=preprocessor.label_column,
        split_column=preprocessor.split_column,
        input_dim=int(checkpoint["input_dim"]),
        model_name=str(checkpoint["model_name"]),
        checkpoint_path=str(state.checkpoint_path),
        checkpoint_fingerprint=checkpoint_fingerprint,
        preprocessor_path=str(state.preprocessor_path),
        preprocessor_fingerprint=preprocessor_fingerprint,
    )
    frame = _build_attribution_frame(
        selection=selection,
        state=state,
        feature_columns=tuple(preprocessor.feature_columns),
        attributions=attributions_tensor.detach().cpu().numpy(),
        logits=logits_tensor.detach().cpu().numpy(),
        baseline_logits=baseline_logits_tensor.detach().cpu().numpy(),
        probabilities=probabilities_tensor.detach().cpu().numpy(),
        deltas=deltas_tensor.detach().cpu().numpy(),
    )
    return frame, manifest


def generate_feature_attributions(
    state: PipelineState,
    settings: AppSettings,
    *,
    split: ExplanationSplit = "validation",
    overwrite: bool = False,
) -> ExplanationRun:
    attribution_frame, manifest = compute_explanations(state, settings, split=split)
    explanation_path = settings.artifacts.explanation_path()
    _ensure_explanation_target_available(
        explanation_path=explanation_path,
        overwrite=overwrite,
    )
    explanation_path = _persist_explanations(
        attribution_frame,
        explanation_path,
        manifest=manifest,
        overwrite=overwrite,
    )
    explanation_fingerprint = compute_file_sha256(explanation_path)
    updated_state = state.model_copy(
        update={
            "stage": "explained",
            "explanation_path": explanation_path,
            "explanation_fingerprint": explanation_fingerprint,
        }
    )
    return ExplanationRun(
        attributions_frame=attribution_frame,
        manifest=load_explanation_manifest(explanation_path),
        explanation_path=explanation_path,
        state=updated_state,
    )


def load_explanation_manifest(path: Path) -> ExplanationArtifactManifest:
    manifest_payload = read_parquet_json_metadata(path, _EXPLANATION_MANIFEST_KEY)
    if manifest_payload is None:
        raise FileNotFoundError(f"Explanation manifest metadata not found in parquet: {path}")
    return ExplanationArtifactManifest.model_validate(manifest_payload)


def _validate_explainability_contract(
    *,
    preprocessor: Any,
    checkpoint: dict[str, Any],
    state: PipelineState,
    preprocessor_fingerprint: str,
    checkpoint_fingerprint: str,
) -> None:
    checkpoint_feature_columns = tuple(checkpoint.get("feature_columns", []))
    if state.feature_columns != tuple(preprocessor.feature_columns):
        raise ValueError("PipelineState feature_columns do not match the loaded preprocessor")
    if state.feature_columns != checkpoint_feature_columns:
        raise ValueError("PipelineState feature_columns do not match the loaded checkpoint")
    if preprocessor.label_column != state.label_column:
        raise ValueError("Loaded preprocessor label_column does not match PipelineState")
    if checkpoint.get("label_column") != state.label_column:
        raise ValueError("Loaded checkpoint label_column does not match PipelineState")
    if preprocessor.split_column != state.split_column:
        raise ValueError("Loaded preprocessor split_column does not match PipelineState")
    if checkpoint.get("split_column") != state.split_column:
        raise ValueError("Loaded checkpoint split_column does not match PipelineState")
    if checkpoint.get("preprocessor_path") != str(state.preprocessor_path):
        raise ValueError("Loaded checkpoint preprocessor_path does not match PipelineState")
    if checkpoint.get("preprocessor_fingerprint") != state.preprocessor_fingerprint:
        raise ValueError("Loaded checkpoint preprocessor_fingerprint does not match PipelineState")
    if checkpoint.get("model_name") != "linear-bce-baseline":
        raise ValueError(f"Unsupported checkpoint model_name: {checkpoint.get('model_name')!r}")
    if checkpoint.get("input_dim") != len(state.feature_columns):
        raise ValueError("Loaded checkpoint input_dim does not match feature_columns length")
    if preprocessor_fingerprint != state.preprocessor_fingerprint:
        raise ValueError("Loaded preprocessor fingerprint does not match PipelineState")
    if checkpoint_fingerprint != state.checkpoint_fingerprint:
        raise ValueError("Loaded checkpoint fingerprint does not match PipelineState")


def _build_attribution_frame(
    *,
    selection: pd.DataFrame,
    state: PipelineState,
    feature_columns: tuple[str, ...],
    attributions: FloatArray,
    logits: FloatArray,
    baseline_logits: FloatArray,
    probabilities: FloatArray,
    deltas: FloatArray,
) -> pd.DataFrame:
    attribution_columns = {
        f"attribution__{feature_name}": attributions[:, index]
        for index, feature_name in enumerate(feature_columns)
    }
    frame = pd.DataFrame(
        {
            "sample_index": selection.index.to_numpy(copy=True),
            "run_id": state.run_id,
            "split": selection[state.split_column].to_numpy(copy=True),
            "label": selection[state.label_column].to_numpy(copy=True),
            "predicted_logit": logits,
            "baseline_logit": baseline_logits,
            "predicted_probability": probabilities,
            "convergence_delta": deltas,
            "checkpoint_path": str(state.checkpoint_path),
            "preprocessor_path": str(state.preprocessor_path),
            "method": "integrated_gradients",
            **attribution_columns,
        }
    )
    return frame


def _persist_explanations(
    frame: pd.DataFrame,
    path: Path,
    *,
    manifest: ExplanationArtifactManifest,
    overwrite: bool = False,
) -> Path:
    return atomic_write_parquet(
        path,
        frame,
        overwrite=overwrite,
        exists_message=f"Explanation artifact already exists: {path}",
        metadata=encode_json_metadata(_EXPLANATION_MANIFEST_KEY, manifest.model_dump(mode="json")),
    )


def _ensure_explanation_target_available(
    *,
    explanation_path: Path,
    overwrite: bool,
) -> None:
    if explanation_path.exists() and not overwrite:
        raise FileExistsError(f"Explanation artifact already exists: {explanation_path}")
