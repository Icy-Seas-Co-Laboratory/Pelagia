from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .exceptions import DatasetStateError, FormatError, FrameNotFoundError
from .history import History
from .manifest import Manifest
from .metadata import Metadata
from .models import Frame
from .shard import ShardReader
from .util import confined_path
from .util import hash_file


class Dataset:
    def __init__(self, root: Path, manifest: Manifest, metadata: Metadata) -> None:
        self.root = root
        self.manifest = manifest
        self.metadata = metadata
        self.history = History(root / "history.jsonl")

    @classmethod
    def open(cls, path: str | Path, *, allow_incomplete: bool = False) -> "Dataset":
        root = Path(path)
        if not root.is_dir():
            raise FormatError(f"dataset directory does not exist: {root}")
        manifest = Manifest.read(root / "manifest.json")
        if manifest.state not in {"complete", "verified", "modified"} and not allow_incomplete:
            raise DatasetStateError(f"dataset state is {manifest.state!r}; pass allow_incomplete=True to inspect it")
        return cls(root, manifest, Metadata.read(root / "metadata.toml"))

    @property
    def frame_count(self) -> int:
        return sum(int(shard.get("frame_count", 0)) for shard in self.manifest.shards)

    @property
    def encoded_bytes(self) -> int:
        return sum(int(shard.get("encoded_bytes", 0)) for shard in self.manifest.shards)

    def iter_shards(self, *, camera: str | None = None, shard: str | None = None) -> Iterator[tuple[dict, ShardReader]]:
        for record in self.manifest.shards:
            if camera is not None and camera not in {record.get("stream_name"), record.get("stream_uuid")}:
                continue
            if shard is not None and shard not in {record.get("relative_path"), record.get("shard_uuid"), Path(record.get("relative_path", "")).name}:
                continue
            path = confined_path(self.root, str(record["relative_path"]))
            yield record, ShardReader(path)

    def iter_frames(
        self, *, camera: str | None = None, frame_start: int | None = None,
        frame_end: int | None = None, source_file_id: int | None = None,
        source_uuid: str | None = None, shard: str | None = None,
        timestamp_start: int | None = None, timestamp_end: int | None = None,
    ) -> Iterator[Frame]:
        if source_uuid is not None:
            source = next((item for item in self.manifest.source_files if item.get("source_uuid") == source_uuid), None)
            if source is None:
                return
            source_file_id = int(source["source_file_id"])
        for record, reader in self.iter_shards(camera=camera, shard=shard):
            first, last = record.get("first_frame"), record.get("last_frame")
            if frame_start is not None and last is not None and int(last) < frame_start:
                continue
            if frame_end is not None and first is not None and int(first) > frame_end:
                continue
            yield from reader.iter_frames(frame_start=frame_start, frame_end=frame_end,
                                          source_file_id=source_file_id, timestamp_start=timestamp_start,
                                          timestamp_end=timestamp_end)

    def get_frame(self, *, camera: str, frame_number: int) -> Frame:
        matches = self.iter_frames(camera=camera, frame_start=frame_number, frame_end=frame_number)
        frame = next(matches, None)
        if frame is None:
            raise FrameNotFoundError(f"{camera}:{frame_number}")
        return frame

    def summary(self) -> dict:
        statuses: dict[str, int] = {}
        codecs: dict[str, int] = {}
        for _, reader in self.iter_shards():
            shard_statuses, shard_codecs = reader.summary_counts()
            for status, count in shard_statuses.items():
                statuses[status] = statuses.get(status, 0) + count
            for codec, count in shard_codecs.items():
                codecs[codec] = codecs.get(codec, 0) + count
        timestamps = [(s.get("first_timestamp"), s.get("last_timestamp")) for s in self.manifest.shards]
        return {
            "format": self.manifest.format,
            "format_version": self.manifest.format_version,
            "schema_version": self.manifest.schema_version,
            "dataset_uuid": self.manifest.dataset_uuid,
            "state": self.manifest.state,
            "title": self.metadata.title,
            "collection": self.metadata.data.get("collection", {}),
            "instruments": self.metadata.data.get("instruments", []),
            "streams": self.metadata.data.get("streams", []),
            "source_files": len(self.manifest.source_files),
            "shards": len(self.manifest.shards),
            "total_frames": self.frame_count,
            "encoded_image_bytes": self.encoded_bytes,
            "package_bytes": sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()),
            "storage_distribution": codecs,
            "status_distribution": statuses,
            "timestamp_range_ns": {
                "first": min((a for a, _ in timestamps if a is not None), default=None),
                "last": max((b for _, b in timestamps if b is not None), default=None),
            },
            "validation": self.manifest.validation,
        }

    def regenerate_checksums(self) -> Path:
        """Regenerate package checksums after an intentional metadata/history edit."""
        paths = sorted(path for path in self.root.rglob("*") if path.is_file()
                       and path.name != "checksums.sha256"
                       and not path.name.endswith((".partial", ".tmp")))
        temporary = self.root / "checksums.sha256.tmp"
        temporary.write_text("".join(f"{hash_file(path)}  {path.relative_to(self.root).as_posix()}\n" for path in paths), encoding="utf-8")
        temporary.replace(self.root / "checksums.sha256")
        return self.root / "checksums.sha256"

    def save_metadata(self, metadata: Metadata | None = None, *, operator: str | None = None, message: str | None = None) -> None:
        """Persist metadata, record provenance, and restore package checksums."""
        if metadata is not None:
            self.metadata = metadata
        self.metadata.write(self.root / "metadata.toml")
        self.history.append(operation="metadata_modified", operator=operator, message=message)
        self.manifest.state = "modified"
        self.manifest.write(self.root / "manifest.json")
        self.regenerate_checksums()
