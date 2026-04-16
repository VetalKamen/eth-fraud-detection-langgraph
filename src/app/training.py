from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from app.artifacts import atomic_torch_save, compute_file_sha256
from app.config import AppSettings
from app.features import build_feature_matrix, load_prepared_dataset, load_preprocessor
from graph.state import PipelineState


class BaselineFraudClassifier(nn.Module):
    """Minimal deterministic binary classifier for the first training loop."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        logits = cast(Tensor, self.linear(inputs))
        return logits


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Trained model artifacts and updated pipeline state."""

    model: BaselineFraudClassifier
    checkpoint_path: Path
    metrics: dict[str, float]
    state: PipelineState


def train_baseline_model(
    state: PipelineState,
    settings: AppSettings,
    *,
    overwrite: bool = False,
) -> TrainingRun:
    if state.prepared_dataset_path is None:
        raise ValueError("PipelineState must include prepared_dataset_path before training")
    if state.preprocessor_path is None:
        raise ValueError("PipelineState must include preprocessor_path before training")
    if state.preprocessor_fingerprint is None:
        raise ValueError("PipelineState must include preprocessor_fingerprint before training")
    if not state.feature_columns:
        raise ValueError("PipelineState must include feature_columns before training")

    with _preserve_training_rng_state():
        _set_training_seed(settings.training.seed)

        frame = load_prepared_dataset(state.prepared_dataset_path)
        preprocessor = load_preprocessor(state.preprocessor_path)
        _validate_preprocessor_matches_state(
            preprocessor,
            state,
            compute_file_sha256(state.preprocessor_path),
        )

        model, metrics = fit_baseline_model(
            frame=frame,
            preprocessor=preprocessor,
            state=state,
            settings=settings,
        )

        checkpoint_path = persist_model_checkpoint(
            build_model_checkpoint(
                model=model,
                metrics=metrics,
                preprocessor=preprocessor,
                state=state,
                settings=settings,
            ),
            settings.artifacts.checkpoint_path(),
            overwrite=overwrite,
        )
        checkpoint_fingerprint = compute_file_sha256(checkpoint_path)

        updated_state = state.model_copy(
            update={
                "stage": "trained",
                "checkpoint_path": checkpoint_path,
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "metrics": metrics,
            }
        )
        return TrainingRun(
            model=model,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            state=updated_state,
        )


def compute_binary_classification_metrics(
    probabilities: NDArray[np.float32] | NDArray[np.float64],
    labels: NDArray[np.float32],
    *,
    prefix: str,
) -> dict[str, float]:
    predicted_labels = (probabilities >= 0.5).astype(np.float32)
    true_positive = float(np.logical_and(predicted_labels == 1.0, labels == 1.0).sum())
    false_positive = float(np.logical_and(predicted_labels == 1.0, labels == 0.0).sum())
    false_negative = float(np.logical_and(predicted_labels == 0.0, labels == 1.0).sum())
    accuracy = float((predicted_labels == labels).mean())
    precision = (
        0.0
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = (
        0.0
        if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative)
    )
    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
    }


def persist_model_checkpoint(
    checkpoint: dict[str, Any], path: Path, *, overwrite: bool = False
) -> Path:
    return atomic_torch_save(
        path,
        checkpoint,
        overwrite=overwrite,
        exists_message=f"Model checkpoint already exists: {path}",
    )


def load_model_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return cast(dict[str, Any], checkpoint)


def load_model_from_checkpoint(path: Path) -> BaselineFraudClassifier:
    checkpoint = load_model_checkpoint(path)
    model_name = checkpoint.get("model_name")
    if model_name != "linear-bce-baseline":
        raise ValueError(f"Unsupported checkpoint model_name: {model_name!r}")
    input_dim = checkpoint.get("input_dim")
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError(f"Checkpoint input_dim must be a positive integer, got: {input_dim!r}")

    model = BaselineFraudClassifier(input_dim=input_dim)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model_state_dict is missing or invalid")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@contextmanager
def _preserve_training_rng_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    previous_deterministic_mode = torch.are_deterministic_algorithms_enabled()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(previous_deterministic_mode)


def _validate_preprocessor_matches_state(
    preprocessor: Any,
    state: PipelineState,
    preprocessor_fingerprint: str,
) -> None:
    if tuple(preprocessor.feature_columns) != state.feature_columns:
        raise ValueError("Loaded preprocessor feature_columns do not match PipelineState")
    if preprocessor.label_column != state.label_column:
        raise ValueError("Loaded preprocessor label_column does not match PipelineState")
    if preprocessor.split_column != state.split_column:
        raise ValueError("Loaded preprocessor split_column does not match PipelineState")
    if preprocessor_fingerprint != state.preprocessor_fingerprint:
        raise ValueError("Loaded preprocessor fingerprint does not match PipelineState")


def fit_baseline_model(
    *,
    frame: Any,
    preprocessor: Any,
    state: PipelineState,
    settings: AppSettings,
) -> tuple[BaselineFraudClassifier, dict[str, float]]:
    train_frame = frame.loc[frame[state.split_column] == "train"]
    validation_frame = frame.loc[frame[state.split_column] == "validation"]
    if train_frame.empty:
        raise ValueError("Prepared dataset must contain a training split before training")
    if validation_frame.empty:
        raise ValueError("Prepared dataset must contain a validation split before training")

    train_features = build_feature_matrix(train_frame, preprocessor)
    validation_features = build_feature_matrix(validation_frame, preprocessor)
    train_labels = _extract_labels(train_frame, state.label_column)
    validation_labels = _extract_labels(validation_frame, state.label_column)

    model = BaselineFraudClassifier(input_dim=train_features.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()

    train_tensor = torch.from_numpy(train_features)
    train_label_tensor = torch.from_numpy(train_labels).unsqueeze(1)
    validation_tensor = torch.from_numpy(validation_features)
    validation_label_tensor = torch.from_numpy(validation_labels).unsqueeze(1)

    for _ in range(settings.training.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tensor)
        loss = loss_function(logits, train_label_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = model(train_tensor)
        validation_logits = model(validation_tensor)

    metrics = {
        "train_loss": float(loss_function(train_logits, train_label_tensor).item()),
        "validation_loss": float(loss_function(validation_logits, validation_label_tensor).item()),
        **compute_binary_classification_metrics(
            torch.sigmoid(validation_logits).squeeze(1).cpu().numpy(),
            validation_labels,
            prefix="validation",
        ),
    }
    return model, metrics


def build_model_checkpoint(
    *,
    model: BaselineFraudClassifier,
    metrics: dict[str, float],
    preprocessor: Any,
    state: PipelineState,
    settings: AppSettings,
) -> dict[str, Any]:
    return {
        "model_name": "linear-bce-baseline",
        "input_dim": len(preprocessor.feature_columns),
        "feature_columns": list(preprocessor.feature_columns),
        "label_column": state.label_column,
        "split_column": state.split_column,
        "preprocessor_path": str(state.preprocessor_path),
        "preprocessor_fingerprint": state.preprocessor_fingerprint,
        "training": settings.training.model_dump(),
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }


def _extract_labels(frame: Any, label_column: str) -> NDArray[np.float32]:
    labels = frame[label_column].to_numpy(dtype=np.float32, copy=True)
    return cast(NDArray[np.float32], labels)
