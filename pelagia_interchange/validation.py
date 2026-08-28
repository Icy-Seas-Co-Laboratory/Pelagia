from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .constants import REQUIRED_PACKAGE_FILES, SCHEMA_VERSION
from .exceptions import FormatError, UnsafePathError
from .history import History
from .manifest import Manifest
from .metadata import Metadata
from .shard import REQUIRED_SHARD_TABLES, ShardReader
from .util import available_hash, confined_path, hash_bytes, hash_file, safe_relative_path, utc_now

VerificationLevel = Literal["quick", "structural", "full", "archival"]


@dataclass(slots=True)
class VerificationIssue:
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(slots=True)
class VerificationResult:
    level: str
    errors: list[VerificationIssue] = field(default_factory=list)
    warnings: list[VerificationIssue] = field(default_factory=list)
    checked_shards: int = 0
    checked_frames: int = 0
    checked_blob_bytes: int = 0
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def archival_ready(self) -> bool:
        return self.level == "archival" and self.valid

    def add_error(self, code: str, message: str, path: Path | str | None = None) -> None:
        self.errors.append(VerificationIssue(code, message, str(path) if path else None))

    def add_warning(self, code: str, message: str, path: Path | str | None = None) -> None:
        self.warnings.append(VerificationIssue(code, message, str(path) if path else None))

    def to_dict(self) -> dict:
        return {
            "valid": self.valid, "archival_ready": self.archival_ready, "level": self.level,
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
            "checked_shards": self.checked_shards, "checked_frames": self.checked_frames,
            "checked_blob_bytes": self.checked_blob_bytes, "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class Validator:
    def __init__(self, path: str | Path) -> None:
        self.root = Path(path)

    def verify(self, level: VerificationLevel = "quick", *, image_signatures: bool = False) -> VerificationResult:
        if level not in {"quick", "structural", "full", "archival"}:
            raise ValueError(f"unknown verification level {level!r}")
        result = VerificationResult(level)
        manifest: Manifest | None = None
        for name in REQUIRED_PACKAGE_FILES:
            if not (self.root / name).is_file():
                result.add_error("missing_package_file", f"required file is missing: {name}", name)
        try:
            manifest = Manifest.read(self.root / "manifest.json")
        except (FormatError, OSError) as exc:
            result.add_error("invalid_manifest", str(exc), "manifest.json")
        if manifest is not None:
            self._quick(manifest, result)
        if level in {"structural", "full", "archival"} and manifest is not None:
            self._structural(manifest, result)
        if level in {"full", "archival"} and manifest is not None:
            self._full(manifest, result, image_signatures=image_signatures, archival=level == "archival")
        if level == "archival" and manifest is not None:
            self._archival(manifest, result)
        result.completed_at = utc_now()
        return result

    def _quick(self, manifest: Manifest, result: VerificationResult) -> None:
        for shard in manifest.shards:
            try:
                path = confined_path(self.root, shard["relative_path"])
            except (KeyError, UnsafePathError) as exc:
                result.add_error("unsafe_shard_path", str(exc))
                continue
            if not path.is_file():
                result.add_error("missing_shard", "manifest shard does not exist", path)
                continue
            actual_size = path.stat().st_size
            if actual_size != shard.get("byte_size"):
                result.add_error("shard_size_mismatch", f"expected {shard.get('byte_size')}, got {actual_size}", path)
            record = shard.get("file_hash") or {}
            algorithm, expected = record.get("algorithm"), record.get("value")
            if record.get("target") != "shard_file":
                result.add_error("ambiguous_shard_hash", "shard hash target must be shard_file", path)
            elif not algorithm or not expected:
                result.add_error("missing_shard_hash", "shard hash algorithm/value is missing", path)
            elif not available_hash(algorithm):
                (result.add_error if result.level == "archival" else result.add_warning)("unavailable_hash", f"cannot verify optional algorithm {algorithm}", path)
            elif hash_file(path, algorithm) != expected:
                result.add_error("shard_hash_mismatch", f"{algorithm} does not match", path)
        for collection_name, records in (("calibration", manifest.calibration), ("preview", manifest.previews)):
            for record in records:
                try:
                    path = confined_path(self.root, record["relative_path"])
                    if not path.is_file():
                        result.add_error("missing_resource", f"manifest {collection_name} resource does not exist", path); continue
                    if path.stat().st_size != record.get("byte_size"):
                        result.add_error("resource_size_mismatch", f"size mismatch for {collection_name} resource", path)
                    hash_record = record.get("file_hash") or {}
                    algorithm = hash_record.get("algorithm")
                    if hash_record.get("target") != "package_file" or not algorithm or not hash_record.get("value"):
                        result.add_error("invalid_resource_hash", f"invalid {collection_name} resource hash", path)
                    elif not available_hash(algorithm):
                        (result.add_error if result.level == "archival" else result.add_warning)("unavailable_hash", f"cannot verify optional algorithm {algorithm}", path)
                    elif hash_file(path, algorithm) != hash_record["value"]:
                        result.add_error("resource_hash_mismatch", f"hash mismatch for {collection_name} resource", path)
                except (KeyError, OSError, UnsafePathError) as exc:
                    result.add_error("invalid_resource", str(exc))
        checksum_path = self.root / "checksums.sha256"
        if checksum_path.is_file():
            try:
                seen: set[str] = set()
                for number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    digest, separator, relative = line.partition("  ")
                    if not separator or len(digest) != 64:
                        result.add_error("invalid_checksum_line", f"invalid checksum line {number}", checksum_path)
                        continue
                    try:
                        path = confined_path(self.root, relative)
                    except UnsafePathError as exc:
                        result.add_error("unsafe_checksum_path", str(exc), checksum_path); continue
                    if not path.is_file():
                        result.add_error("missing_checksummed_file", f"missing {relative}", path)
                    elif hash_file(path) != digest:
                        result.add_error("package_checksum_mismatch", f"SHA-256 mismatch for {relative}", path)
                    if relative in seen:
                        result.add_error("duplicate_checksum_path", f"duplicate checksum path {relative}", checksum_path)
                    seen.add(relative)
                expected = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()
                            and path.name != "checksums.sha256" and not path.name.endswith((".partial", ".tmp"))}
                for relative in sorted(expected - seen):
                    result.add_error("unchecksummed_package_file", f"package file is absent from checksums: {relative}", relative)
                for relative in sorted(seen - expected):
                    result.add_error("stale_checksum_entry", f"checksum entry is not a finalized package file: {relative}", relative)
            except OSError as exc:
                result.add_error("checksum_read_error", str(exc), checksum_path)

    def _structural(self, manifest: Manifest, result: VerificationResult) -> None:
        try:
            metadata = Metadata.read(self.root / "metadata.toml")
            for error in metadata.validate(require_title=True):
                result.add_error("invalid_metadata", error, "metadata.toml")
        except FormatError as exc:
            result.add_error("invalid_metadata", str(exc), "metadata.toml")
        segments_by_id = {int(segment["acquisition_segment_id"]): segment
                          for segment in manifest.acquisition_segments if "acquisition_segment_id" in segment}
        segment_ids = set(segments_by_id)
        for shard in manifest.shards:
            try:
                path = confined_path(self.root, shard["relative_path"])
                reader = ShardReader(path)
                if reader.integrity_check() != "ok":
                    result.add_error("sqlite_integrity", "SQLite integrity_check failed", path); continue
                missing = REQUIRED_SHARD_TABLES - reader.tables()
                if missing:
                    result.add_error("missing_tables", f"missing tables: {sorted(missing)}", path); continue
                metadata = reader.metadata()
                counts = reader.counts()
                aggregates = reader.aggregate_metadata()
                if str(metadata.get("schema_version")) != SCHEMA_VERSION:
                    result.add_error("schema_mismatch", "shard schema version is unsupported", path)
                for key in ("shard_uuid", "frame_count", "first_frame", "last_frame", "first_timestamp", "last_timestamp"):
                    if metadata.get(key) != shard.get(key):
                        result.add_error("manifest_shard_mismatch", f"{key} differs between manifest and shard", path)
                for key in ("frame_count", "first_frame", "last_frame", "first_timestamp", "last_timestamp", "encoded_bytes"):
                    if aggregates.get(key) != metadata.get(key):
                        result.add_error("shard_aggregate_mismatch", f"{key} differs between shard metadata and frame rows", path)
                if counts["frames"] != int(shard.get("frame_count", -1)):
                    result.add_error("frame_count_mismatch", f"manifest declares {shard.get('frame_count')}, database has {counts['frames']}", path)
                with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
                    unknown = (connection.execute("SELECT count(*) FROM frames").fetchone()[0] if not segment_ids else
                               connection.execute("SELECT count(*) FROM frames WHERE acquisition_segment_id NOT IN (%s)" % ",".join("?" for _ in segment_ids), tuple(segment_ids)).fetchone()[0])
                    if unknown:
                        result.add_error("unknown_acquisition_mapping", f"{unknown} frames reference segments absent from manifest", path)
                    for segment_id, segment_uuid, name, mode, expected_count in connection.execute(
                        "SELECT acquisition_segment_id,acquisition_segment_uuid,segment_name,acquisition_mode,expected_frame_count FROM acquisition_segments"):
                        declared = segments_by_id.get(int(segment_id))
                        if (declared is None or segment_uuid != declared.get("acquisition_segment_uuid") or
                            name != declared.get("segment_name") or mode != declared.get("acquisition_mode") or
                            expected_count != declared.get("expected_frame_count")):
                            result.add_error("acquisition_segment_mismatch", f"acquisition segment {segment_id} differs between manifest and shard", path)
                result.checked_shards += 1
            except (FormatError, sqlite3.Error, OSError, KeyError, UnsafePathError) as exc:
                result.add_error("shard_structure", str(exc), shard.get("relative_path"))

    def _full(self, manifest: Manifest, result: VerificationResult, *, image_signatures: bool, archival: bool) -> None:
        previous_by_stream: dict[str, int] = {}
        previous_source_frame: dict[tuple[str, int], int] = {}
        ordered = sorted(manifest.shards, key=lambda item: (str(item.get("stream_uuid")), item.get("first_frame") if item.get("first_frame") is not None else -1))
        for shard in ordered:
            stream = str(shard.get("stream_uuid"))
            try:
                reader = ShardReader(confined_path(self.root, shard["relative_path"]))
                for frame in reader.iter_frames():
                    record = frame.record
                    result.checked_frames += 1
                    payload = record.encoded_bytes
                    result.checked_blob_bytes += len(payload or b"")
                    if record.declared_byte_size != len(payload or b""):
                        result.add_error("blob_size_mismatch", f"stored BLOB size mismatch for frame {record.frame_id}")
                    previous = previous_by_stream.get(stream)
                    if previous is not None:
                        if record.frame_id == previous:
                            result.add_error("duplicate_frame_id", f"duplicate frame ID {record.frame_id} in stream {stream}")
                        elif record.frame_id > previous + 1:
                            result.add_error("unrepresented_frame_gap", f"stream {stream} skips frame IDs {previous + 1}:{record.frame_id - 1}")
                    previous_by_stream[stream] = record.frame_id
                    source_key = (stream, record.source_file_id)
                    previous_source = previous_source_frame.get(source_key)
                    if previous_source is not None and record.source_frame_number <= previous_source:
                        result.add_error("source_sequence_order", f"source frame {record.source_frame_number} is not increasing")
                    elif previous_source is not None and record.source_frame_number > previous_source + 1:
                        result.add_error("unrepresented_source_gap", f"source {record.source_file_id} skips source frames {previous_source + 1}:{record.source_frame_number - 1}")
                    previous_source_frame[source_key] = record.source_frame_number
                    if payload is None:
                        if record.status.value == "valid":
                            result.add_error("valid_without_blob", f"valid frame {record.frame_id} has no BLOB")
                        continue
                    if record.blob_hash is None:
                        (result.add_error if archival else result.add_warning)("missing_blob_hash", f"frame {record.frame_id} has no stored_blob hash")
                    elif record.blob_hash.target != "stored_blob":
                        result.add_error("ambiguous_blob_hash", f"frame {record.frame_id} has incorrect hash target")
                    elif not available_hash(record.blob_hash.algorithm):
                        (result.add_error if archival else result.add_warning)("unavailable_hash", f"cannot verify {record.blob_hash.algorithm} for frame {record.frame_id}")
                    elif hash_bytes(payload, record.blob_hash.algorithm) != record.blob_hash.value:
                        result.add_error("blob_hash_mismatch", f"stored BLOB hash mismatch for frame {record.frame_id}")
                    if image_signatures and record.storage_format:
                        codec = record.storage_format.codec.lower()
                        valid = codec not in {"jpeg", "jpg", "png"} or (codec in {"jpeg", "jpg"} and payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")) or (codec == "png" and payload.startswith(b"\x89PNG\r\n\x1a\n"))
                        if not valid:
                            result.add_error("image_signature", f"frame {record.frame_id} is not recognizable {codec}")
            except (FormatError, OSError, UnsafePathError) as exc:
                result.add_error("frame_scan_error", str(exc), shard.get("relative_path"))

    def _archival(self, manifest: Manifest, result: VerificationResult) -> None:
        partials = sorted(self.root.glob("data/*.partial"))
        if partials:
            result.add_error("partial_shards", f"{len(partials)} partial shard(s) remain")
        try:
            events = list(History(self.root / "history.jsonl"))
            successful = {event.get("operation") for event in events if event.get("status") == "success"}
            for required in ("dataset_created", "dataset_finalized"):
                if required not in successful:
                    result.add_error("missing_history_event", f"successful {required} history event is required")
            if manifest.shards and "shard_finalized" not in successful:
                result.add_error("missing_history_event", "successful shard_finalized history event is required")
        except FormatError as exc:
            result.add_error("invalid_history", str(exc), "history.jsonl")
        expected = {int(segment["acquisition_segment_id"]): segment.get("expected_frame_count")
                    for segment in manifest.acquisition_segments}
        observed = {identifier: 0 for identifier in expected}
        for shard in manifest.shards:
            try:
                path = confined_path(self.root, shard["relative_path"])
                with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
                    for source_id, count in connection.execute("SELECT acquisition_segment_id,count(*) FROM frames GROUP BY acquisition_segment_id"):
                        observed[source_id] = observed.get(source_id, 0) + count
            except (sqlite3.Error, OSError, UnsafePathError) as exc:
                result.add_error("source_reconciliation", str(exc), shard.get("relative_path"))
        for source_id, frame_count in expected.items():
            if frame_count is None:
                result.add_error("unknown_expected_frames", f"acquisition segment {source_id} has no expected_frame_count")
            elif observed.get(source_id, 0) != int(frame_count):
                result.add_error("acquisition_frame_count", f"acquisition segment {source_id}: expected {frame_count}, represented {observed.get(source_id, 0)}")
