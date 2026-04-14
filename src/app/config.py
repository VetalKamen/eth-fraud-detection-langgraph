from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetSettings(BaseModel):
    """Input dataset contract for training and evaluation."""

    model_config = ConfigDict(frozen=True)

    input_path: Path = Path("data/ethereum_transactions.parquet")
    label_column: str = "is_fraud"
    entity_id_column: str = "address"
    feature_columns: tuple[str, ...] = ()
    split_column: str = "split"
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    @field_validator("label_column", "entity_id_column", "split_column")
    @classmethod
    def validate_column_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("column names must not be blank")
        return normalized

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(column.strip() for column in value)
        if any(not column for column in normalized):
            raise ValueError("feature_columns must not contain blank names")
        if len(set(normalized)) != len(normalized):
            raise ValueError("feature_columns must be unique")
        return normalized

    @field_validator("train_ratio", "validation_ratio", "test_ratio")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("split ratios must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_split_ratios(self) -> DatasetSettings:
        reserved_columns = {self.label_column, self.entity_id_column, self.split_column}
        feature_column_overlap = reserved_columns.intersection(self.feature_columns)
        if feature_column_overlap:
            overlap_list = ", ".join(sorted(feature_column_overlap))
            raise ValueError(f"feature_columns must not include reserved columns: {overlap_list}")
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train_ratio, validation_ratio, and test_ratio must sum to 1.0")
        return self


class ArtifactSettings(BaseModel):
    """Artifact layout contract to keep runs isolated and idempotent."""

    model_config = ConfigDict(frozen=True)

    root_dir: Path = Path("artifacts")
    run_id: str = "local"
    prepared_dataset_filename: str = "dataset.parquet"
    preprocessor_filename: str = "preprocessor.json"
    checkpoint_filename: str = "fraud-detector.pt"
    explanation_filename: str = "attributions.parquet"

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("run_id must not be blank")
        if "/" in normalized or "\\" in normalized:
            raise ValueError("run_id must be a single path segment")
        return normalized

    def run_dir(self) -> Path:
        return self.root_dir / self.run_id

    def prepared_dataset_path(self) -> Path:
        return self.run_dir() / "prepared" / self.prepared_dataset_filename

    def preprocessor_path(self) -> Path:
        return self.run_dir() / "models" / self.preprocessor_filename

    def checkpoint_path(self) -> Path:
        return self.run_dir() / "models" / self.checkpoint_filename

    def explanation_path(self) -> Path:
        return self.run_dir() / "explanations" / self.explanation_filename


class RuntimeSettings(BaseModel):
    """Runtime knobs that should remain independent from data and artifacts."""

    model_config = ConfigDict(frozen=True)

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: int = Field(default=512, gt=0)
    num_workers: int = Field(default=0, ge=0)
    explain_sample_size: int = Field(default=128, gt=0)


class TrainingSettings(BaseModel):
    """Training-specific controls kept separate from data and artifact config."""

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    epochs: int = Field(default=50, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class AppSettings(BaseSettings):
    """Top-level settings loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_prefix="ETH_FRAUD_",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
