from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from .constants import FORMAT_ID, FORMAT_VERSION, LIBRARY_VERSION, SCHEMA_VERSION
from .exceptions import CompatibilityError, FormatError
from .util import utc_now


@dataclass(slots=True)
class Manifest:
    dataset_uuid: str
    created_at: str = field(default_factory=utc_now)
    state: str = "building"
    format: str = FORMAT_ID
    format_version: str = FORMAT_VERSION
    schema_version: str = SCHEMA_VERSION
    shards: list[dict[str, Any]] = field(default_factory=list)
    acquisition_segments: list[dict[str, Any]] = field(default_factory=list)
    calibration: list[dict[str, Any]] = field(default_factory=list)
    previews: list[dict[str, Any]] = field(default_factory=list)
    software: dict[str, Any] = field(default_factory=lambda: {"name": "pelagia_interchange", "version": LIBRARY_VERSION})
    validation: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        try:
            UUID(self.dataset_uuid)
        except ValueError as exc:
            raise FormatError("manifest dataset_uuid is invalid") from exc
        if self.format != FORMAT_ID:
            raise CompatibilityError(f"unsupported format {self.format!r}")
        if self.format_version != FORMAT_VERSION:
            raise CompatibilityError(
                f"format {self.format_version} is not supported by this pre-1.0 reader; expected {FORMAT_VERSION}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise CompatibilityError(f"unsupported schema version {self.schema_version}")
        if self.state not in {"building", "finalizing", "complete", "verified", "modified"}:
            raise FormatError(f"invalid dataset state {self.state!r}")
        if not isinstance(self.shards, list) or not isinstance(self.acquisition_segments, list):
            raise FormatError("manifest shards and acquisition_segments must be arrays")
        if not isinstance(self.calibration, list) or not isinstance(self.previews, list):
            raise FormatError("manifest calibration and previews must be arrays")
        shard_ids: set[str] = set()
        shard_paths: set[str] = set()
        stream_ids_by_name: dict[str, str] = {}
        stream_names_by_id: dict[str, str] = {}
        for number, shard in enumerate(self.shards):
            if not isinstance(shard, dict):
                raise FormatError(f"shards[{number}] must be an object")
            required = {"shard_uuid", "relative_path", "byte_size", "file_hash", "stream_uuid", "stream_name",
                        "first_frame", "last_frame", "frame_count", "first_timestamp", "last_timestamp"}
            missing = required - set(shard)
            if missing:
                raise FormatError(f"shards[{number}] missing fields: {sorted(missing)}")
            try:
                UUID(str(shard["shard_uuid"])); UUID(str(shard["stream_uuid"]))
            except ValueError as exc:
                raise FormatError(f"shards[{number}] contains an invalid UUID") from exc
            if not isinstance(shard["relative_path"], str) or not isinstance(shard["stream_name"], str):
                raise FormatError(f"shards[{number}] has an invalid path or stream name")
            shard_uuid = str(shard["shard_uuid"])
            if shard_uuid in shard_ids or shard["relative_path"] in shard_paths:
                raise FormatError(f"shards[{number}] duplicates a shard UUID or path")
            shard_ids.add(shard_uuid); shard_paths.add(shard["relative_path"])
            stream_uuid, stream_name = str(shard["stream_uuid"]), shard["stream_name"]
            if stream_ids_by_name.get(stream_name, stream_uuid) != stream_uuid or stream_names_by_id.get(stream_uuid, stream_name) != stream_name:
                raise FormatError(f"shards[{number}] has an inconsistent stream name/UUID mapping")
            stream_ids_by_name[stream_name] = stream_uuid; stream_names_by_id[stream_uuid] = stream_name
            if not isinstance(shard["byte_size"], int) or shard["byte_size"] < 0 or not isinstance(shard["frame_count"], int) or shard["frame_count"] < 0:
                raise FormatError(f"shards[{number}] has an invalid size or frame count")
            hash_record = shard["file_hash"]
            if not isinstance(hash_record, dict) or not all(isinstance(hash_record.get(key), str) and hash_record[key] for key in ("algorithm", "target", "value")):
                raise FormatError(f"shards[{number}] has an invalid file_hash")
        segment_ids: set[int] = set()
        segment_uuids: set[str] = set()
        for number, segment in enumerate(self.acquisition_segments):
            if not isinstance(segment, dict):
                raise FormatError(f"acquisition_segments[{number}] must be an object")
            if not all(key in segment for key in ("acquisition_segment_id", "acquisition_segment_uuid", "segment_name", "acquisition_mode")):
                raise FormatError(f"acquisition_segments[{number}] is missing required identity fields")
            try:
                UUID(str(segment["acquisition_segment_uuid"]))
            except ValueError as exc:
                raise FormatError(f"acquisition_segments[{number}] has an invalid UUID") from exc
            identifier = segment["acquisition_segment_id"]
            if not isinstance(identifier, int) or identifier < 1:
                raise FormatError(f"acquisition_segments[{number}] has an invalid acquisition_segment_id")
            segment_uuid = str(segment["acquisition_segment_uuid"])
            if identifier in segment_ids or segment_uuid in segment_uuids:
                raise FormatError(f"acquisition_segments[{number}] duplicates an acquisition identity")
            if segment["acquisition_mode"] not in {"direct_frame_capture", "imported_video"}:
                raise FormatError(f"acquisition_segments[{number}] has an invalid acquisition_mode")
            if not isinstance(segment["segment_name"], str) or not segment["segment_name"].strip():
                raise FormatError(f"acquisition_segments[{number}] has an invalid segment_name")
            if segment["acquisition_mode"] == "imported_video" and not isinstance(segment.get("import_provenance"), dict):
                raise FormatError(f"acquisition_segments[{number}] imported_video requires import_provenance")
            segment_ids.add(identifier); segment_uuids.add(segment_uuid)
        for collection_name, records in (("calibration", self.calibration), ("previews", self.previews)):
            for number, record in enumerate(records):
                if not isinstance(record, dict) or not all(key in record for key in ("relative_path", "byte_size", "file_hash")):
                    raise FormatError(f"{collection_name}[{number}] is not a valid resource record")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        extra = value.pop("extra_fields")
        value.update(extra)
        return value

    def write(self, path: Path) -> None:
        self.validate()
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FormatError(f"cannot read manifest {path}: {exc}") from exc
        known = {field.name for field in cls.__dataclass_fields__.values()} - {"extra_fields"}
        extensions = dict(raw.get("extensions", {}))
        extra_fields = {key: raw[key] for key in set(raw) - known}
        values = {key: value for key, value in raw.items() if key in known}
        values["extensions"] = extensions
        values["extra_fields"] = extra_fields
        try:
            manifest = cls(**values)
        except TypeError as exc:
            raise FormatError(f"invalid manifest fields: {exc}") from exc
        manifest.validate()
        return manifest

    @property
    def source_files(self) -> list[dict[str, Any]]:
        """Deprecated in-memory alias for pre-0.2 integrations."""
        legacy: list[dict[str, Any]] = []
        for segment in self.acquisition_segments:
            provenance = dict(segment.get("import_provenance") or {})
            legacy.append({
                "source_file_id": segment["acquisition_segment_id"],
                "source_uuid": segment["acquisition_segment_uuid"],
                "original_filename": provenance.get("original_filename", segment["segment_name"]),
                "original_relative_path": provenance.get("original_relative_path"),
                "original_absolute_path": provenance.get("original_absolute_path"),
                "byte_size": provenance.get("byte_size"), "file_hash": provenance.get("file_hash"),
                "container": provenance.get("container"), "codec": provenance.get("codec"),
                "pixel_format": provenance.get("pixel_format"), "width": provenance.get("width"),
                "height": provenance.get("height"), "frame_rate": provenance.get("frame_rate"),
                "frame_count": segment.get("expected_frame_count"),
                "start_timestamp": segment.get("started_at"), "end_timestamp": segment.get("ended_at"),
            })
        return legacy
