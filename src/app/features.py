from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from app.artifacts import atomic_write_text, compute_file_sha256
from app.config import AppSettings
from graph.state import PipelineState

_EXPECTED_SPLITS = {"train", "validation", "test"}


class FeaturePreprocessor(BaseModel):
    """Persisted preprocessing contract used for training and inference."""

    model_config = ConfigDict(frozen=True)

    feature_columns: tuple[str, ...]
    split_column: str
    label_column: str
    imputation_values: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]

    @model_validator(mode="after")
    def validate_feature_statistics(self) -> FeaturePreprocessor:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        expected = set(self.feature_columns)
        for field_name, mapping in (
            ("imputation_values", self.imputation_values),
            ("means", self.means),
            ("scales", self.scales),
        ):
            observed = set(mapping)
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                details = []
                if missing:
                    details.append(f"missing keys: {', '.join(missing)}")
                if extra:
                    details.append(f"extra keys: {', '.join(extra)}")
                raise ValueError(
                    f"{field_name} must match feature_columns exactly ({'; '.join(details)})"
                )
        return self


@dataclass(frozen=True, slots=True)
class PreparedFeatures:
    """In-memory transformed dataset plus persisted preprocessing state."""

    transformed_frame: pd.DataFrame
    preprocessor: FeaturePreprocessor
    state: PipelineState


def load_prepared_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Unsupported prepared dataset format for {path}; expected a .parquet file"
        )
    return pd.read_parquet(path)


def fit_feature_preprocessor(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    split_column: str,
    label_column: str,
) -> FeaturePreprocessor:
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    _validate_feature_frame(
        frame,
        feature_columns=feature_columns,
        split_column=split_column,
        require_split_column=True,
    )
    training_frame = frame.loc[frame[split_column] == "train"]
    if training_frame.empty:
        raise ValueError("Prepared dataset must contain at least one training record")

    imputation_values: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}

    for column in feature_columns:
        numeric_train = pd.to_numeric(training_frame[column], errors="raise")
        imputation_value = float(numeric_train.mean(skipna=True))
        if np.isnan(imputation_value):
            raise ValueError(
                f"Feature column '{column}' cannot be entirely null in the training split"
            )

        filled_train = numeric_train.fillna(imputation_value)
        mean_value = float(filled_train.mean())
        scale_value = float(filled_train.std(ddof=0))
        if scale_value == 0.0 or np.isnan(scale_value):
            scale_value = 1.0

        imputation_values[column] = imputation_value
        means[column] = mean_value
        scales[column] = scale_value

    return FeaturePreprocessor(
        feature_columns=feature_columns,
        split_column=split_column,
        label_column=label_column,
        imputation_values=imputation_values,
        means=means,
        scales=scales,
    )


def transform_features(frame: pd.DataFrame, preprocessor: FeaturePreprocessor) -> pd.DataFrame:
    _validate_feature_frame(
        frame,
        feature_columns=preprocessor.feature_columns,
        split_column=preprocessor.split_column,
        require_split_column=False,
    )

    transformed = frame.copy()
    for column in preprocessor.feature_columns:
        numeric_column = pd.to_numeric(transformed[column], errors="raise")
        filled_column = numeric_column.fillna(preprocessor.imputation_values[column])
        transformed[column] = (filled_column - preprocessor.means[column]) / preprocessor.scales[
            column
        ]

    return transformed


def build_feature_matrix(
    frame: pd.DataFrame, preprocessor: FeaturePreprocessor
) -> NDArray[np.float32]:
    transformed = transform_features(frame, preprocessor)
    matrix = transformed.loc[:, list(preprocessor.feature_columns)].to_numpy(
        dtype=np.float32, copy=True
    )
    return matrix


def persist_preprocessor(
    preprocessor: FeaturePreprocessor, path: Path, *, overwrite: bool = False
) -> Path:
    return atomic_write_text(
        path,
        preprocessor.model_dump_json(indent=2) + "\n",
        overwrite=overwrite,
        exists_message=f"Preprocessor artifact already exists: {path}",
    )


def load_preprocessor(path: Path) -> FeaturePreprocessor:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {path}")
    return FeaturePreprocessor.model_validate_json(path.read_text(encoding="utf-8"))


def prepare_features(
    state: PipelineState,
    settings: AppSettings,
    *,
    overwrite: bool = False,
) -> PreparedFeatures:
    if state.prepared_dataset_path is None:
        raise ValueError(
            "PipelineState must include prepared_dataset_path before feature preparation"
        )
    if not state.feature_columns:
        raise ValueError("PipelineState must include feature_columns before feature preparation")

    prepared_frame = load_prepared_dataset(state.prepared_dataset_path)
    preprocessor = fit_feature_preprocessor(
        prepared_frame,
        feature_columns=state.feature_columns,
        split_column=state.split_column,
        label_column=state.label_column,
    )
    transformed_frame = transform_features(prepared_frame, preprocessor)
    preprocessor_path = persist_preprocessor(
        preprocessor,
        settings.artifacts.preprocessor_path(),
        overwrite=overwrite,
    )
    preprocessor_fingerprint = compute_file_sha256(preprocessor_path)

    updated_state = state.model_copy(
        update={
            "stage": "features_ready",
            "preprocessor_path": preprocessor_path,
            "preprocessor_fingerprint": preprocessor_fingerprint,
            "feature_columns": preprocessor.feature_columns,
        }
    )
    return PreparedFeatures(
        transformed_frame=transformed_frame,
        preprocessor=preprocessor,
        state=updated_state,
    )


def validate_preprocessor_against_state(
    *,
    frame: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    state: PipelineState,
) -> None:
    expected = fit_feature_preprocessor(
        frame,
        feature_columns=state.feature_columns,
        split_column=state.split_column,
        label_column=state.label_column,
    )
    if preprocessor.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError(
            "Existing preprocessor contents do not match the expected deterministic output"
        )


def _validate_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    split_column: str,
    require_split_column: bool,
) -> None:
    expected_columns = list(feature_columns)
    if require_split_column:
        expected_columns.append(split_column)
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Prepared dataset is missing required columns: {missing_list}")

    if require_split_column:
        observed_splits = set(frame[split_column].dropna().tolist())
        unexpected_splits = sorted(
            repr(split_name) for split_name in observed_splits if split_name not in _EXPECTED_SPLITS
        )
        if unexpected_splits:
            unexpected_list = ", ".join(unexpected_splits)
            raise ValueError(
                f"Prepared dataset contains unexpected split values: {unexpected_list}"
            )
