from __future__ import annotations

from dataclasses import dataclass

from app.config import AppSettings, get_settings
from graph.state import PipelineState


@dataclass(frozen=True, slots=True)
class ApplicationRuntime:
    """Shared bootstrap contract for future CLI and graph entrypoints."""

    settings: AppSettings
    initial_state: PipelineState


def create_runtime(settings: AppSettings | None = None) -> ApplicationRuntime:
    resolved_settings = settings or get_settings()
    return ApplicationRuntime(
        settings=resolved_settings,
        initial_state=PipelineState(
            run_id=resolved_settings.artifacts.run_id,
            stage="configured",
            raw_dataset_path=resolved_settings.dataset.input_path,
            artifact_dir=resolved_settings.artifacts.run_dir(),
            label_column=resolved_settings.dataset.label_column,
            split_column=resolved_settings.dataset.split_column,
            feature_columns=resolved_settings.dataset.feature_columns,
        ),
    )
