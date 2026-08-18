from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .dataset import Dataset
from .exceptions import UnsafePathError
from .models import Frame


@dataclass(slots=True)
class ExtractionResult:
    selected: int = 0
    written: int = 0
    skipped: int = 0
    bytes_written: int = 0


def extract_frames(
    dataset: Dataset, output: str | Path, *, camera: str | None = None,
    frame_start: int | None = None, frame_end: int | None = None,
    source_file_id: int | None = None, source_uuid: str | None = None,
    shard: str | None = None, timestamp_start: int | None = None,
    timestamp_end: int | None = None, overwrite: str = "error",
    source_frame_names: bool = False, dry_run: bool = False,
    custom_extension: str = ".bin", progress: Callable[[int], None] | None = None,
) -> ExtractionResult:
    if overwrite not in {"error", "skip", "replace"}:
        raise ValueError("overwrite must be error, skip, or replace")
    root = Path(output).resolve()
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
    result = ExtractionResult()
    for frame in dataset.iter_frames(camera=camera, frame_start=frame_start, frame_end=frame_end,
                                     source_file_id=source_file_id, source_uuid=source_uuid, shard=shard,
                                     timestamp_start=timestamp_start, timestamp_end=timestamp_end):
        payload = frame.encoded_bytes
        if payload is None:
            result.skipped += 1
            continue
        result.selected += 1
        record = frame.record
        stem = f"source_{record.source_file_id:06d}_{record.source_frame_number:012d}" if source_frame_names else f"{record.frame_id:012d}"
        extension = record.storage_format.extension if record.storage_format else custom_extension
        if extension == ".bin" and custom_extension:
            extension = custom_extension if custom_extension.startswith(".") else "." + custom_extension
        destination = (root / f"{stem}{extension}").resolve()
        if destination.parent != root:
            raise UnsafePathError("generated extraction path escaped output root")
        if destination.exists():
            if overwrite == "skip":
                result.skipped += 1; continue
            if overwrite == "error":
                raise FileExistsError(destination)
        if not dry_run:
            frame.save(destination, overwrite=overwrite == "replace")
        result.written += 1
        result.bytes_written += len(payload)
        if progress and result.written % 1000 == 0:
            progress(result.written)
    return result

