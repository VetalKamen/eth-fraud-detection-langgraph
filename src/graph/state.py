from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PipelineStage = Literal["configured", "dataset_loaded", "features_ready", "trained", "explained"]

_STAGE_ORDER: dict[PipelineStage, int] = {
    "configured": 0,
    "dataset_loaded": 1,
    "features_ready": 2,
    "trained": 3,
    "explained": 4,
}


def _stage_at_least(current: PipelineStage, required: PipelineStage) -> bool:
    return _STAGE_ORDER[current] >= _STAGE_ORDER[required]


class PipelineState(BaseModel):
    """Validated state contract shared between future LangGraph nodes."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    stage: PipelineStage = "configured"
    raw_dataset_path: Path
    raw_dataset_fingerprint: str | None = None
    artifact_dir: Path
    label_column: str
    split_column: str = "split"
    feature_columns: tuple[str, ...] = ()
    prepared_dataset_path: Path | None = None
    prepared_dataset_fingerprint: str | None = None
    preprocessor_path: Path | None = None
    preprocessor_fingerprint: str | None = None
    checkpoint_path: Path | None = None
    checkpoint_fingerprint: str | None = None
    explanation_path: Path | None = None
    explanation_fingerprint: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_stage_contract(self) -> PipelineState:
        if not self.run_id.strip():
            raise ValueError("run_id must not be blank")
        if not self.label_column.strip():
            raise ValueError("label_column must not be blank")
        if not self.split_column.strip():
            raise ValueError("split_column must not be blank")

        if _stage_at_least(self.stage, "dataset_loaded") and self.prepared_dataset_path is None:
            raise ValueError("prepared_dataset_path is required from stage 'dataset_loaded' onward")
        if _stage_at_least(self.stage, "dataset_loaded") and self.raw_dataset_fingerprint is None:
            raise ValueError(
                "raw_dataset_fingerprint is required from stage 'dataset_loaded' onward"
            )
        if (
            _stage_at_least(self.stage, "dataset_loaded")
            and self.prepared_dataset_fingerprint is None
        ):
            raise ValueError(
                "prepared_dataset_fingerprint is required from stage 'dataset_loaded' onward"
            )
        if _stage_at_least(self.stage, "features_ready"):
            if self.preprocessor_path is None:
                raise ValueError("preprocessor_path is required from stage 'features_ready' onward")
            if self.preprocessor_fingerprint is None:
                raise ValueError(
                    "preprocessor_fingerprint is required from stage 'features_ready' onward"
                )
            if not self.feature_columns:
                raise ValueError("feature_columns are required from stage 'features_ready' onward")
        if _stage_at_least(self.stage, "trained"):
            if self.checkpoint_path is None:
                raise ValueError("checkpoint_path is required from stage 'trained' onward")
            if self.checkpoint_fingerprint is None:
                raise ValueError("checkpoint_fingerprint is required from stage 'trained' onward")
            if not self.metrics:
                raise ValueError("metrics are required from stage 'trained' onward")
        if _stage_at_least(self.stage, "explained"):
            if self.explanation_path is None:
                raise ValueError("explanation_path is required from stage 'explained' onward")
            if self.explanation_fingerprint is None:
                raise ValueError(
                    "explanation_fingerprint is required from stage 'explained' onward"
                )

        return self
