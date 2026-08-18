from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid4, uuid5

from .constants import FORMAT_NAME, FORMAT_VERSION
from .dataset import Dataset
from .exceptions import DatasetStateError
from .history import History
from .manifest import Manifest
from .metadata import Metadata, default_metadata
from .models import FrameRecord, HashRecord, SourceFile, StorageFormat, new_uuid
from .shard import ShardWriter
from .tool_scripts import TOOL_FILES
from .util import hash_file, parse_size, safe_relative_path, utc_now


class DatasetBuilder:
    def __init__(
        self, output: str | Path, *, title: str | None = None, description: str = "",
        shard_target_bytes: int | str = 10_000_000_000, shard_target_size: int | str | None = None,
        maximum_frame_count: int | None = None, source_file_boundary: bool = False,
        dataset_uuid: UUID | str | None = None, metadata: Metadata | Mapping[str, Any] | None = None,
    ) -> None:
        self.output = Path(output)
        self.shard_target_bytes = parse_size(shard_target_size or shard_target_bytes)
        self.maximum_frame_count = maximum_frame_count
        self.source_file_boundary = source_file_boundary
        self.dataset_uuid = new_uuid(dataset_uuid)
        if metadata is None:
            self.metadata = default_metadata(title or self.output.name, description)
        elif isinstance(metadata, Metadata):
            self.metadata = metadata
        else:
            self.metadata = Metadata(dict(metadata))
        self.manifest = Manifest(str(self.dataset_uuid))
        self.history = History(self.output / "history.jsonl")
        self._sources: dict[int, SourceFile] = {}
        self._sources_by_uuid: dict[UUID, SourceFile] = {}
        self._writers: dict[str, ShardWriter] = {}
        self._stream_uuids: dict[str, UUID] = {}
        self._shard_numbers: dict[str, int] = {}
        self._last_source_per_stream: dict[str, int] = {}
        self._closed = False
        self._initialize()

    def _initialize(self) -> None:
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        for directory in ("data", "calibration", "preview", "tools"):
            (self.output / directory).mkdir()
        self.manifest.write(self.output / "manifest.json")
        self.metadata.write(self.output / "metadata.toml")
        self.history.append(operation="dataset_created", outputs=[{"dataset_uuid": str(self.dataset_uuid)}])

    def __enter__(self) -> "DatasetBuilder":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: object) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort(message=str(exc))

    def register_source_file(
        self, path: str | Path | None = None, *, original_filename: str | None = None,
        source_uuid: UUID | str | None = None, original_relative_path: str | None = None,
        original_absolute_path: str | None = None, byte_size: int | None = None,
        sha256: str | None = None, file_hash: HashRecord | None = None,
        container: str | None = None, codec: str | None = None, pixel_format: str | None = None,
        width: int | None = None, height: int | None = None, frame_rate: tuple[int, int] | None = None,
        frame_count: int | None = None, start_timestamp: str | None = None, end_timestamp: str | None = None,
    ) -> SourceFile:
        source_path = Path(path) if path is not None else None
        if original_filename is None:
            if source_path is None:
                raise ValueError("original_filename is required when path is omitted")
            original_filename = source_path.name
        if byte_size is None and source_path is not None and source_path.is_file():
            byte_size = source_path.stat().st_size
        if file_hash is None and sha256:
            file_hash = HashRecord("sha256", "source_file", sha256)
        if file_hash is not None and file_hash.target != "source_file":
            raise ValueError("source file hash target must be 'source_file'")
        if original_absolute_path is None and source_path is not None and source_path.is_absolute():
            original_absolute_path = str(source_path)
        identifier = len(self._sources) + 1
        source = SourceFile(identifier, new_uuid(source_uuid), original_filename, original_relative_path,
                            original_absolute_path, byte_size, file_hash, container, codec, pixel_format,
                            width, height, frame_rate[0] if frame_rate else None, frame_rate[1] if frame_rate else None,
                            frame_count, start_timestamp, end_timestamp)
        self._sources[identifier] = source
        self._sources_by_uuid[source.source_uuid] = source
        record = self._source_manifest_record(source)
        self.manifest.source_files.append(record)
        self.manifest.write(self.output / "manifest.json")
        self.history.append(operation="source_ingested", inputs=[record])
        return source

    add_source_file = register_source_file

    @staticmethod
    def _source_manifest_record(source: SourceFile) -> dict[str, Any]:
        return {
            "source_file_id": source.source_file_id, "source_uuid": str(source.source_uuid),
            "original_filename": source.original_filename, "original_relative_path": source.original_relative_path,
            "original_absolute_path": source.original_absolute_path, "byte_size": source.byte_size,
            "file_hash": source.file_hash.to_dict() if source.file_hash else None, "container": source.container,
            "codec": source.codec, "pixel_format": source.pixel_format, "width": source.width, "height": source.height,
            "frame_rate": [source.frame_rate_num, source.frame_rate_den] if source.frame_rate_num is not None else None,
            "frame_count": source.frame_count, "start_timestamp": source.start_timestamp,
            "end_timestamp": source.end_timestamp,
        }

    def _stream_uuid(self, stream: str) -> UUID:
        return self._stream_uuids.setdefault(stream, uuid5(self.dataset_uuid, f"stream:{stream}"))

    def register_stream(self, name: str, *, stream_uuid: UUID | str | None = None) -> UUID:
        """Register a persistent stream identity before adding its frames."""
        if not name.strip():
            raise ValueError("stream name must not be empty")
        identifier = new_uuid(stream_uuid) if stream_uuid is not None else uuid5(self.dataset_uuid, f"stream:{name}")
        existing = self._stream_uuids.get(name)
        if existing is not None and existing != identifier:
            raise ValueError(f"stream {name!r} is already registered with a different UUID")
        if any(other_name != name and other_uuid == identifier for other_name, other_uuid in self._stream_uuids.items()):
            raise ValueError("stream UUID is already assigned to another stream name")
        self._stream_uuids[name] = identifier
        return identifier

    @staticmethod
    def _slug(stream: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "_" for char in stream).strip("_")
        return slug or "stream"

    def _new_writer(self, stream: str) -> ShardWriter:
        if stream not in self._stream_uuids:
            self.register_stream(stream)
        number = self._shard_numbers.get(stream, 0) + 1
        self._shard_numbers[stream] = number
        filename = f"{self._slug(stream)}_{number:06d}.sqlite"
        writer = ShardWriter(self.output / "data" / filename, stream_uuid=self._stream_uuid(stream), stream_name=stream)
        self._writers[stream] = writer
        return writer

    def _finalize_writer(self, stream: str) -> None:
        writer = self._writers.pop(stream, None)
        if writer is None:
            return
        if writer.frame_count == 0:
            writer.abandon()
            return
        shard = writer.finalize()
        relative = shard.path.relative_to(self.output).as_posix()
        record = {
            "shard_uuid": str(shard.shard_uuid), "relative_path": relative,
            "byte_size": shard.path.stat().st_size,
            "file_hash": {"algorithm": "sha256", "target": "shard_file", "value": hash_file(shard.path)},
            "stream_uuid": str(shard.stream_uuid), "stream_name": shard.stream_name,
            "first_frame": shard.first_frame, "last_frame": shard.last_frame,
            "frame_count": shard.frame_count, "first_timestamp": writer.first_timestamp,
            "last_timestamp": writer.last_timestamp, "encoded_bytes": writer.encoded_bytes,
        }
        self.manifest.shards.append(record)
        self.manifest.write(self.output / "manifest.json")
        self.history.append(operation="shard_finalized", outputs=[record])

    def boundary(self, stream: str | None = None) -> None:
        for name in [stream] if stream is not None else list(self._writers):
            self._finalize_writer(name)

    def add_frame(
        self, *, stream: str, source_file: SourceFile, frame_id: int,
        source_frame_number: int, encoded_bytes: bytes | None,
        storage_format: StorageFormat | None = None, **kwargs: Any,
    ) -> None:
        if self._closed:
            raise DatasetStateError("builder is closed")
        if source_file.source_file_id not in self._sources:
            raise ValueError("source_file was not registered with this builder")
        if self.source_file_boundary and self._last_source_per_stream.get(stream) not in {None, source_file.source_file_id}:
            self._finalize_writer(stream)
        writer = self._writers.get(stream)
        incoming = len(encoded_bytes or b"")
        if writer is not None and writer.frame_count and (
            writer.encoded_bytes + incoming > self.shard_target_bytes or
            (self.maximum_frame_count is not None and writer.frame_count >= self.maximum_frame_count)
        ):
            self._finalize_writer(stream)
            writer = None
        writer = writer or self._new_writer(stream)
        writer.add(FrameRecord(frame_id, source_file.source_file_id, source_frame_number, encoded_bytes,
                               storage_format=storage_format, **kwargs), source_file)
        self._last_source_per_stream[stream] = source_file.source_file_id

    def add_frames(self, records: Iterable[tuple[str, SourceFile, FrameRecord]]) -> None:
        for stream, source, record in records:
            self.add_frame(stream=stream, source_file=source, frame_id=record.frame_id,
                           source_frame_number=record.source_frame_number, encoded_bytes=record.encoded_bytes,
                           storage_format=record.storage_format,
                           timestamp_ns=record.timestamp_ns, source_timestamp_ns=record.source_timestamp_ns,
                           timestamp_source=record.timestamp_source, clock_source=record.clock_source,
                           timezone=record.timezone, utc_conversion=record.utc_conversion,
                           timestamp_precision_ns=record.timestamp_precision_ns,
                           synchronization_method=record.synchronization_method,
                           known_offset_ns=record.known_offset_ns, known_drift_ppb=record.known_drift_ppb,
                           interpolated=record.interpolated, width=record.width, height=record.height,
                           status=record.status, blob_hash=record.blob_hash,
                           decoded_pixel_hash=record.decoded_pixel_hash)

    def add_resource(self, path: str | Path, *, kind: str, relative_name: str | None = None, description: str | None = None) -> dict[str, Any]:
        if kind not in {"calibration", "preview"}:
            raise ValueError("resource kind must be calibration or preview")
        source = Path(path)
        relative = safe_relative_path(relative_name or source.name)
        destination = self.output / kind / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        record = {"relative_path": destination.relative_to(self.output).as_posix(), "byte_size": destination.stat().st_size,
                  "file_hash": {"algorithm": "sha256", "target": "package_file", "value": hash_file(destination)},
                  "description": description, "authoritative": False if kind == "preview" else None}
        getattr(self.manifest, "previews" if kind == "preview" else "calibration").append(record)
        return record

    def add_resource_bytes(self, data: bytes, *, kind: str, relative_name: str,
                           description: str | None = None, attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Add an in-memory package resource without an intermediate source file."""
        if kind not in {"calibration", "preview"}:
            raise ValueError("resource kind must be calibration or preview")
        relative = safe_relative_path(relative_name)
        destination = self.output / kind / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(data)
        record: dict[str, Any] = {
            "relative_path": destination.relative_to(self.output).as_posix(), "byte_size": len(data),
            "file_hash": {"algorithm": "sha256", "target": "package_file", "value": hash_file(destination)},
            "description": description, "authoritative": False if kind == "preview" else None,
        }
        record.update(dict(attributes or {}))
        getattr(self.manifest, "previews" if kind == "preview" else "calibration").append(record)
        return record

    def _write_tools(self) -> None:
        for name, content in TOOL_FILES.items():
            path = self.output / "tools" / name
            path.write_text(content, encoding="utf-8")

    def _write_readme(self) -> None:
        title = self.metadata.title or self.output.name
        content = f"""# {title}

This directory is a self-contained {FORMAT_NAME} {FORMAT_VERSION} dataset. Its SQLite files under `data/` are the authoritative retained encoded images and frame-level provenance. Treat finalized shards as immutable archival artifacts.

## Layout

- `manifest.json`: physical inventory and shard index
- `metadata.toml`: human-editable scientific and collection metadata
- `history.jsonl`: append-only processing provenance (one JSON event per line)
- `checksums.sha256`: SHA-256 checksums for every finalized package file except itself
- `data/*.sqlite`: standalone SQLite image shards
- `calibration/`: ordinary calibration resources
- `preview/`: non-authoritative derivatives
- `tools/`: standard-library inspection, extraction, and verification scripts

Each shard contains `frames`, `source_files`, `storage_formats`, and `shard_metadata`. The exact encoded image bytes are stored in `frames.blob`; ordinary extraction copies them without decoding or re-encoding. A null BLOB is an explicit missing/failed/removed record, never an implicit sequence collapse.

Inspect manually with `sqlite3 data/SHARD.sqlite '.tables'` or run:

```bash
python tools/inspect.py .
python tools/inspect.py . --json
python tools/extract.py . --frame 12345 --output extracted/one
python tools/extract.py . --camera CAMERA --frames 1000:2000 --output extracted
python tools/verify.py . --level full
python tools/verify.py . --level archival --json
```

Frame and timestamp ranges are inclusive. JPEG and PNG payloads are written with conventional extensions; unknown representations use `.bin`. Null-payload status records are skipped rather than converted into invented images. The extractor also accepts one shard directly: `python tools/extract.py data/SHARD.sqlite --all --output extracted`.

Verification levels become progressively more expensive: `quick` checks the package inventory and file hashes; `structural` also checks every SQLite database and its declared structure; `full` streams all frame records and hashes; `archival` adds strict source-count and completion-provenance requirements. An archival success is a mechanical integrity result, not authorization to delete original acquisitions. Retention policy, backups, scientific acceptance, and responsible-person approval remain separate decisions. These tools never delete source data.

An interrupted writer leaves `data/*.sqlite.partial` files unlisted by the manifest. With the package installed, list them using `pii shards . --partials` or move them intact to a recovery directory using `pii shards . --quarantine-partials RECOVERY_DIR`. They are never deleted automatically.

Hashes always have an algorithm and semantic target. Frame hashes cover exact stored BLOB bytes; shard hashes cover complete finalized SQLite files. The checksum file covers every regular finalized package file except itself, so there is no self-reference. Editing the manifest, metadata, or history therefore requires checksum regeneration. Manifest shard hashes remain authoritative for shard-file integrity.

Timestamp values do not imply accuracy: provenance fields record source, clock, timezone/UTC conversion, precision, synchronization, offset, drift, and interpolation where known. Scientific interpretation belongs in `metadata.toml`; processing actions belong in `history.jsonl`.

The normative reference is **Scientific Image Interchange Format 1.0**, maintained with the `pelagia_interchange` source at `docs/interchange-specification.md`. These files use ordinary JSON, TOML, JSON Lines, SHA-256 text, and SQLite so they remain accessible without Pelagia.
"""
        (self.output / "README.md").write_text(content, encoding="utf-8")

    def _write_checksums(self) -> None:
        paths = sorted(path for path in self.output.rglob("*") if path.is_file()
                       and path.name != "checksums.sha256"
                       and not path.name.endswith(".partial") and not path.name.endswith(".tmp"))
        content = "".join(f"{hash_file(path)}  {path.relative_to(self.output).as_posix()}\n" for path in paths)
        temporary = self.output / "checksums.sha256.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(self.output / "checksums.sha256")

    def finalize(self) -> Dataset:
        if self._closed:
            return Dataset.open(self.output)
        self.manifest.state = "finalizing"
        self.manifest.write(self.output / "manifest.json")
        for stream in list(self._writers):
            self._finalize_writer(stream)
        self._write_tools()
        self._write_readme()
        self.history.append(operation="dataset_finalized", outputs=[{"dataset_uuid": str(self.dataset_uuid)}])
        self.metadata.write(self.output / "metadata.toml")
        self.manifest.state = "complete"
        self.manifest.write(self.output / "manifest.json")
        self._write_checksums()
        self._closed = True
        return Dataset.open(self.output)

    def abort(self, *, message: str | None = None) -> None:
        if self._closed:
            return
        for writer in self._writers.values():
            writer.abandon()
        self._writers.clear()
        self.history.append(operation="dataset_creation_interrupted", status="failed", message=message)
        self.manifest.state = "building"
        self.manifest.write(self.output / "manifest.json")
        self._closed = True

    @staticmethod
    def partials(path: str | Path) -> list[Path]:
        return sorted(Path(path).glob("data/*.sqlite.partial"))

    @staticmethod
    def quarantine_partials(path: str | Path, destination: str | Path) -> list[Path]:
        """Move abandoned partial shards to a caller-selected recovery directory."""
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        moved: list[Path] = []
        for partial in DatasetBuilder.partials(path):
            target = destination_path / partial.name
            if target.exists():
                raise FileExistsError(target)
            partial.replace(target)
            moved.append(target)
        return moved
