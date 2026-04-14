from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from json import dumps, loads
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_temp_path(
    target: Path,
    *,
    overwrite: bool = False,
    exists_message: str | None = None,
) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    error_message = exists_message or f"Artifact already exists: {target}"
    if target.exists() and not overwrite:
        raise FileExistsError(error_message)

    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    committed = False

    try:
        yield temp_path
        if overwrite:
            os.replace(temp_path, target)
            committed = True
        else:
            # Link-then-unlink avoids the final existence-check/replace race.
            os.link(temp_path, target)
            temp_path.unlink()
            committed = True
    finally:
        if temp_path.exists() and not committed:
            temp_path.unlink()


def atomic_write_path(
    target: Path,
    write_fn: Callable[[Path], None],
    *,
    overwrite: bool = False,
    exists_message: str | None = None,
) -> Path:
    with atomic_temp_path(
        target,
        overwrite=overwrite,
        exists_message=exists_message,
    ) as temp_path:
        write_fn(temp_path)
    return target


def atomic_write_text(
    target: Path,
    content: str,
    *,
    overwrite: bool = False,
    encoding: str = "utf-8",
    exists_message: str | None = None,
) -> Path:
    def _write(temp_path: Path) -> None:
        temp_path.write_text(content, encoding=encoding)

    return atomic_write_path(
        target,
        _write,
        overwrite=overwrite,
        exists_message=exists_message,
    )


def atomic_write_bytes(
    target: Path,
    content: bytes,
    *,
    overwrite: bool = False,
    exists_message: str | None = None,
) -> Path:
    def _write(temp_path: Path) -> None:
        temp_path.write_bytes(content)

    return atomic_write_path(
        target,
        _write,
        overwrite=overwrite,
        exists_message=exists_message,
    )


def atomic_write_parquet(
    target: Path,
    frame: pd.DataFrame,
    *,
    overwrite: bool = False,
    exists_message: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Path:
    def _write(temp_path: Path) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        schema_metadata = dict(table.schema.metadata or {})
        if metadata:
            schema_metadata.update(
                {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}
            )
            table = table.replace_schema_metadata(schema_metadata)
        pq.write_table(table, temp_path)

    return atomic_write_path(
        target,
        _write,
        overwrite=overwrite,
        exists_message=exists_message,
    )


def atomic_torch_save(
    target: Path,
    obj: object,
    *,
    overwrite: bool = False,
    exists_message: str | None = None,
) -> Path:
    def _write(temp_path: Path) -> None:
        import torch

        torch.save(obj, temp_path)

    return atomic_write_path(
        target,
        _write,
        overwrite=overwrite,
        exists_message=exists_message,
    )


def read_parquet_metadata(path: Path) -> dict[str, str]:
    metadata = pq.read_schema(path).metadata or {}
    return {key.decode("utf-8"): value.decode("utf-8") for key, value in metadata.items()}


def read_parquet_json_metadata(path: Path, key: str) -> dict[str, Any] | None:
    metadata = read_parquet_metadata(path)
    raw_value = metadata.get(key)
    if raw_value is None:
        return None
    return cast(dict[str, Any], loads(raw_value))


def encode_json_metadata(key: str, payload: dict[str, Any]) -> dict[str, str]:
    return {key: dumps(payload, sort_keys=True)}
