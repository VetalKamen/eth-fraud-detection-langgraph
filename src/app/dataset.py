from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.artifacts import (
    atomic_write_parquet,
    compute_file_sha256,
    encode_json_metadata,
    read_parquet_json_metadata,
)
from app.config import AppSettings, DatasetSettings
from graph.state import PipelineState

DatasetSplit = Literal["train", "validation", "test"]
_SPLIT_ORDER: tuple[DatasetSplit, ...] = ("train", "validation", "test")
_ALLOWED_LABEL_VALUES = (0, 1)
_PREPARED_DATASET_MANIFEST_KEY = "eth_fraud.prepared_dataset_manifest"


class DatasetSchema(BaseModel):
    """Resolved dataset schema after validation and feature selection."""

    model_config = ConfigDict(frozen=True)

    label_column: str
    entity_id_column: str
    feature_columns: tuple[str, ...]
    split_column: str = "split"

    @field_validator("label_column", "entity_id_column", "split_column")
    @classmethod
    def validate_column_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("schema column names must not be blank")
        return normalized

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("feature_columns must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("feature_columns must be unique")
        return value

    @model_validator(mode="after")
    def validate_reserved_columns(self) -> DatasetSchema:
        reserved_columns = {self.label_column, self.entity_id_column, self.split_column}
        overlap = reserved_columns.intersection(self.feature_columns)
        if overlap:
            overlap_list = ", ".join(sorted(overlap))
            raise ValueError(f"feature_columns must not include reserved columns: {overlap_list}")
        return self

    def required_columns(self) -> tuple[str, ...]:
        return (self.entity_id_column, self.label_column, *self.feature_columns)


@dataclass(frozen=True, slots=True)
class IngestedDataset:
    """Prepared dataset plus the pipeline state needed by downstream stages."""

    schema: DatasetSchema
    prepared_frame: pd.DataFrame
    state: PipelineState
    split_counts: dict[DatasetSplit, int]


class PreparedDatasetManifest(BaseModel):
    """Dataset provenance persisted beside the prepared parquet."""

    model_config = ConfigDict(frozen=True)

    raw_dataset_path: str
    raw_dataset_fingerprint: str
    label_column: str
    split_column: str
    entity_id_column: str
    feature_columns: tuple[str, ...]


def load_raw_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported dataset format for {path}; expected a .parquet file")
    return pd.read_parquet(path)


def resolve_dataset_schema(frame: pd.DataFrame, settings: DatasetSettings) -> DatasetSchema:
    if not settings.feature_columns:
        raise ValueError(
            "Dataset feature_columns must be configured explicitly before ingestion to avoid schema drift"
        )
    feature_columns = settings.feature_columns
    return DatasetSchema(
        label_column=settings.label_column,
        entity_id_column=settings.entity_id_column,
        feature_columns=feature_columns,
        split_column=settings.split_column,
    )


def validate_raw_dataset(frame: pd.DataFrame, schema: DatasetSchema) -> None:
    missing_columns = [
        column for column in schema.required_columns() if column not in frame.columns
    ]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    if frame.empty:
        raise ValueError("Dataset is empty")

    if frame[schema.label_column].isna().any():
        raise ValueError(f"Column '{schema.label_column}' contains null values")
    if frame[schema.entity_id_column].isna().any():
        raise ValueError(f"Column '{schema.entity_id_column}' contains null values")

    observed_labels = set(frame[schema.label_column].tolist())
    invalid_labels = sorted(
        repr(label) for label in observed_labels if label not in _ALLOWED_LABEL_VALUES
    )
    if invalid_labels:
        invalid_list = ", ".join(invalid_labels)
        raise ValueError(
            f"Column '{schema.label_column}' must contain only binary labels 0/1 or booleans; "
            f"found: {invalid_list}"
        )
    if len(observed_labels) < 2:
        raise ValueError(f"Column '{schema.label_column}' must contain both fraud classes")


def assign_splits(
    frame: pd.DataFrame,
    *,
    entity_id_column: str,
    split_column: str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> pd.DataFrame:
    unique_entities = sorted({entity for entity in frame[entity_id_column].tolist()}, key=repr)
    entity_count = len(unique_entities)
    if entity_count == 0:
        raise ValueError("Dataset must contain at least one entity")

    ratio_targets: dict[DatasetSplit, float] = {
        "train": train_ratio * entity_count,
        "validation": validation_ratio * entity_count,
        "test": test_ratio * entity_count,
    }
    split_entity_counts = _allocate_entity_counts(ratio_targets, entity_count)

    rng = np.random.default_rng(random_seed)
    shuffled_entities = list(rng.permutation(unique_entities))

    split_for_entity: dict[object, DatasetSplit] = {}
    start_index = 0
    for split_name in _SPLIT_ORDER:
        end_index = start_index + split_entity_counts[split_name]
        for entity in shuffled_entities[start_index:end_index]:
            split_for_entity[entity] = split_name
        start_index = end_index

    prepared_frame = frame.copy()
    prepared_frame[split_column] = prepared_frame[entity_id_column].map(split_for_entity)

    if prepared_frame[split_column].isna().any():
        raise ValueError("Failed to assign a dataset split to every record")

    return prepared_frame


def persist_prepared_dataset(
    frame: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool = False,
    manifest: PreparedDatasetManifest | None = None,
) -> Path:
    metadata = None
    if manifest is not None:
        metadata = encode_json_metadata(
            _PREPARED_DATASET_MANIFEST_KEY,
            manifest.model_dump(mode="json"),
        )
    return atomic_write_parquet(
        path,
        frame,
        overwrite=overwrite,
        exists_message=f"Prepared dataset already exists: {path}",
        metadata=metadata,
    )


def ingest_dataset(settings: AppSettings, *, overwrite: bool = False) -> IngestedDataset:
    prepared_frame, schema, raw_dataset_fingerprint = build_prepared_dataset(settings)
    persisted_columns = list(dict.fromkeys((*schema.required_columns(), schema.split_column)))
    prepared_frame = prepared_frame.loc[:, persisted_columns].copy()
    prepared_path = persist_prepared_dataset(
        prepared_frame,
        settings.artifacts.prepared_dataset_path(),
        overwrite=overwrite,
        manifest=PreparedDatasetManifest(
            raw_dataset_path=str(settings.dataset.input_path),
            raw_dataset_fingerprint=raw_dataset_fingerprint,
            label_column=schema.label_column,
            split_column=schema.split_column,
            entity_id_column=schema.entity_id_column,
            feature_columns=schema.feature_columns,
        ),
    )
    prepared_dataset_fingerprint = compute_file_sha256(prepared_path)

    split_counts = {
        split_name: int((prepared_frame[schema.split_column] == split_name).sum())
        for split_name in _SPLIT_ORDER
    }
    state = PipelineState(
        run_id=settings.artifacts.run_id,
        stage="dataset_loaded",
        raw_dataset_path=settings.dataset.input_path,
        raw_dataset_fingerprint=raw_dataset_fingerprint,
        artifact_dir=settings.artifacts.run_dir(),
        label_column=schema.label_column,
        split_column=schema.split_column,
        feature_columns=schema.feature_columns,
        prepared_dataset_path=prepared_path,
        prepared_dataset_fingerprint=prepared_dataset_fingerprint,
    )
    return IngestedDataset(
        schema=schema,
        prepared_frame=prepared_frame,
        state=state,
        split_counts=split_counts,
    )


def build_prepared_dataset(settings: AppSettings) -> tuple[pd.DataFrame, DatasetSchema, str]:
    raw_dataset_fingerprint = compute_file_sha256(settings.dataset.input_path)
    raw_frame = load_raw_dataset(settings.dataset.input_path)
    schema = resolve_dataset_schema(raw_frame, settings.dataset)
    validate_raw_dataset(raw_frame, schema)
    prepared_frame = assign_splits(
        raw_frame,
        entity_id_column=schema.entity_id_column,
        split_column=schema.split_column,
        train_ratio=settings.dataset.train_ratio,
        validation_ratio=settings.dataset.validation_ratio,
        test_ratio=settings.dataset.test_ratio,
        random_seed=settings.dataset.random_seed,
    )
    persisted_columns = list(dict.fromkeys((*schema.required_columns(), schema.split_column)))
    prepared_frame = prepared_frame.loc[:, persisted_columns].copy()
    return prepared_frame, schema, raw_dataset_fingerprint


def load_prepared_dataset_manifest(path: Path) -> PreparedDatasetManifest:
    manifest_payload = read_parquet_json_metadata(path, _PREPARED_DATASET_MANIFEST_KEY)
    if manifest_payload is None:
        raise FileNotFoundError(f"Prepared dataset manifest metadata not found in parquet: {path}")
    return PreparedDatasetManifest.model_validate(manifest_payload)


def validate_prepared_dataset_against_settings(
    frame: pd.DataFrame,
    manifest: PreparedDatasetManifest,
    settings: AppSettings,
) -> tuple[DatasetSchema, str]:
    expected_frame, schema, raw_dataset_fingerprint = build_prepared_dataset(settings)
    if manifest.raw_dataset_path != str(settings.dataset.input_path):
        raise ValueError("Prepared dataset metadata raw_dataset_path does not match settings")
    if manifest.raw_dataset_fingerprint != raw_dataset_fingerprint:
        raise ValueError(
            "Prepared dataset metadata raw_dataset_fingerprint does not match input data"
        )
    if manifest.label_column != schema.label_column:
        raise ValueError("Prepared dataset metadata label_column does not match settings")
    if manifest.split_column != schema.split_column:
        raise ValueError("Prepared dataset metadata split_column does not match settings")
    if manifest.entity_id_column != settings.dataset.entity_id_column:
        raise ValueError("Prepared dataset metadata entity_id_column does not match settings")
    if manifest.feature_columns != schema.feature_columns:
        raise ValueError("Prepared dataset metadata feature_columns do not match settings")
    try:
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True), expected_frame.reset_index(drop=True)
        )
    except AssertionError as exc:
        raise ValueError(
            "Prepared dataset contents do not match the expected deterministic output"
        ) from exc
    return schema, raw_dataset_fingerprint


def _allocate_entity_counts(
    ratio_targets: dict[DatasetSplit, float], entity_count: int
) -> dict[DatasetSplit, int]:
    split_counts = {split_name: int(target) for split_name, target in ratio_targets.items()}
    remaining = entity_count - sum(split_counts.values())
    remainders = sorted(
        _SPLIT_ORDER,
        key=lambda split_name: (
            ratio_targets[split_name] - split_counts[split_name],
            -_SPLIT_ORDER.index(split_name),
        ),
        reverse=True,
    )

    for split_name in remainders[:remaining]:
        split_counts[split_name] += 1

    return split_counts
