from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from app.artifacts import (
    atomic_temp_path,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_parquet,
    atomic_write_text,
)


def test_atomic_write_text_replaces_target_without_leaving_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"

    atomic_write_text(target, '{"ok": true}\n')

    assert target.read_text(encoding="utf-8") == '{"ok": true}\n'
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_parquet_round_trips_without_partial_target(tmp_path: Path) -> None:
    target = tmp_path / "artifact.parquet"
    frame = pd.DataFrame({"value": [1, 2, 3]})

    atomic_write_parquet(target, frame)

    persisted = pd.read_parquet(target)
    pd.testing.assert_frame_equal(persisted, frame)
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_torch_save_round_trips_checkpoint(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pt"

    atomic_torch_save(target, {"weight": torch.tensor([1.0, 2.0])})

    checkpoint = torch.load(target, map_location="cpu")
    assert torch.equal(checkpoint["weight"], torch.tensor([1.0, 2.0]))


def test_atomic_temp_path_cleans_up_temp_file_after_writer_error(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"

    with pytest.raises(RuntimeError, match="boom"), atomic_temp_path(target) as temp_path:
        temp_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_rejects_existing_target_when_overwrite_disabled(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"original")

    with pytest.raises(FileExistsError, match="Artifact already exists"):
        atomic_write_bytes(target, b"replacement", overwrite=False)

    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob("*.tmp")) == []
