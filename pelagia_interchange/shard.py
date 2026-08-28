from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import UUID, uuid4

from .constants import FORMAT_VERSION, SCHEMA_VERSION
from .exceptions import FormatError, FrameNotFoundError, IntegrityError
from .models import AcquisitionSegment, Frame, FrameRecord, FrameStatus, HashRecord, SourceFile, StorageFormat
from .schema import REQUIRED_TABLES, initialize
from .util import canonical_json, hash_bytes, utc_now


_SQLITE_BUSY_TIMEOUT_MS = 30_000


def _connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    try:
        if read_only:
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True,
                                         timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000)
            connection.execute("PRAGMA query_only=ON")
        else:
            connection = sqlite3.connect(path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000)
        connection.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error as exc:
        raise FormatError(f"cannot open SQLite shard {path}: {exc}") from exc


@dataclass(slots=True)
class Shard:
    path: Path
    shard_uuid: UUID
    stream_uuid: UUID
    stream_name: str
    first_frame: int | None
    last_frame: int | None
    frame_count: int


class ShardWriter:
    """Streaming builder for one not-yet-authoritative SQLite shard."""

    def __init__(self, final_path: Path, *, stream_uuid: UUID, stream_name: str, shard_uuid: UUID | None = None, batch_size: int = 1000) -> None:
        self.final_path = final_path
        self.partial_path = final_path.with_name(final_path.name + ".partial")
        self.stream_uuid = stream_uuid
        self.stream_name = stream_name
        self.shard_uuid = shard_uuid or uuid4()
        self.batch_size = batch_size
        self._pending = 0
        self._closed = False
        self.created_at = utc_now()
        self._storage_ids: dict[tuple[Any, ...], int] = {}
        self._source_ids: set[int] = set()
        self.first_frame: int | None = None
        self.last_frame: int | None = None
        self.first_timestamp: int | None = None
        self.last_timestamp: int | None = None
        self.frame_count = 0
        self.encoded_bytes = 0
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.partial_path.exists() or self.final_path.exists():
            raise FileExistsError(self.partial_path if self.partial_path.exists() else self.final_path)
        self.connection = _connect(self.partial_path)
        try:
            self.connection.execute("PRAGMA page_size=65536")
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA cache_size=-131072")
            self.connection.execute("PRAGMA foreign_keys=ON")
            initialize(self.connection)
            self.connection.executemany(
                "INSERT OR REPLACE INTO shard_metadata(key,value) VALUES (?,?)",
                (("shard_uuid", json.dumps(str(self.shard_uuid))),
                 ("format_version", json.dumps(FORMAT_VERSION)),
                 ("schema_version", json.dumps(SCHEMA_VERSION)),
                 ("created_at", json.dumps(self.created_at)),
                 ("created_by", json.dumps("pelagia_interchange")),
                 ("camera_or_stream_id", json.dumps(str(self.stream_uuid))),
                 ("camera_or_stream_name", json.dumps(self.stream_name))),
            )
            self.connection.commit()
            self.connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            self.connection.close()
            self._closed = True
            raise FormatError(f"cannot initialize SQLite shard {self.partial_path}: {exc}") from exc

    @classmethod
    def resume(
        cls, final_path: Path, *, stream_uuid: UUID, stream_name: str,
        shard_uuid: UUID | None = None, batch_size: int = 1000,
    ) -> "ShardWriter":
        """Reopen a durable partial shard and continue its current transaction.

        A partial may have been written by an older package version.  The
        stream identity is therefore taken from its metadata when present and
        otherwise supplied by the caller; the latter is the compatibility path
        for the original partial-shard implementation.
        """
        partial_path = final_path.with_name(final_path.name + ".partial")
        if not partial_path.is_file():
            raise FileNotFoundError(partial_path)
        if final_path.exists():
            raise FileExistsError(final_path)
        self = cls.__new__(cls)
        self.final_path = final_path
        self.partial_path = partial_path
        self.stream_uuid = stream_uuid
        self.stream_name = stream_name
        self.batch_size = batch_size
        self._pending = 0
        self._closed = False
        self._storage_ids = {}
        self._source_ids = set()
        self.first_frame = None
        self.last_frame = None
        self.first_timestamp = None
        self.last_timestamp = None
        self.frame_count = 0
        self.encoded_bytes = 0
        self.created_at = utc_now()
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = _connect(partial_path)
        try:
            # Opening a rollback-journal database lets SQLite recover a prior
            # interrupted transaction.  Explicitly close any transaction this
            # connection could have inherited before validating and acquiring
            # the sole writer lock below.  A live external writer is allowed to
            # finish for ``busy_timeout``; it is never forcibly broken.
            self.connection.rollback()
            self.connection.execute("PRAGMA journal_mode=DELETE")
            tables = {str(row[0]) for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            missing = REQUIRED_TABLES - tables
            if missing:
                raise FormatError(f"partial shard is missing tables: {sorted(missing)}")
            metadata = {row["key"]: json.loads(row["value"]) for row in self.connection.execute(
                "SELECT key,value FROM shard_metadata"
            )}
            stored_stream = metadata.get("camera_or_stream_id")
            stored_name = metadata.get("camera_or_stream_name")
            if stored_stream is not None and str(stored_stream) != str(stream_uuid):
                raise FormatError(f"partial shard stream UUID {stored_stream!r} does not match {stream_uuid}")
            if stored_name is not None and str(stored_name) != stream_name:
                raise FormatError(f"partial shard stream name {stored_name!r} does not match {stream_name!r}")
            self.shard_uuid = UUID(str(metadata.get("shard_uuid"))) if metadata.get("shard_uuid") else (shard_uuid or uuid4())
            if self.connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise IntegrityError(f"partial shard failed integrity_check: {partial_path}")
            for row in self.connection.execute("SELECT acquisition_segment_id FROM acquisition_segments"):
                self._source_ids.add(int(row[0]))
            for row in self.connection.execute(
                "SELECT storage_id,codec,codec_version,quality,pixel_format,bit_depth,encoder,encoder_version,parameters_json FROM storage_formats"
            ):
                key = (row["codec"].lower(), row["codec_version"], row["quality"], row["pixel_format"],
                       row["bit_depth"], row["encoder"], row["encoder_version"], row["parameters_json"])
                self._storage_ids[key] = int(row["storage_id"])
            aggregate = self.connection.execute(
                "SELECT count(*),min(frame_id),max(frame_id),min(timestamp_ns),max(timestamp_ns),coalesce(sum(byte_size),0) FROM frames"
            ).fetchone()
            self.frame_count = int(aggregate[0])
            self.first_frame, self.last_frame = aggregate[1], aggregate[2]
            self.first_timestamp, self.last_timestamp = aggregate[3], aggregate[4]
            self.encoded_bytes = int(aggregate[5])
            self.created_at = str(metadata.get("created_at") or self.created_at)
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("BEGIN IMMEDIATE")
        except (sqlite3.Error, json.JSONDecodeError, ValueError, TypeError) as exc:
            self.connection.close()
            self._closed = True
            if isinstance(exc, FormatError):
                raise
            raise FormatError(f"cannot reopen SQLite shard {partial_path}: {exc}") from exc
        except (FormatError, IntegrityError):
            self.connection.close()
            self._closed = True
            raise
        return self

    def register_source(self, source: AcquisitionSegment) -> None:
        if source.source_file_id in self._source_ids:
            return
        file_hash = source.file_hash
        self.connection.execute(
            """INSERT INTO acquisition_segments VALUES (?,?,?,?,?,?,?,?,?)""",
            (source.acquisition_segment_id, str(source.acquisition_segment_uuid), source.segment_name,
             source.acquisition_mode, source.expected_frame_count,
             canonical_json(dict(source.capture_configuration)), source.started_at, source.ended_at,
             canonical_json(dict(source.import_provenance)) if source.import_provenance is not None else None),
        )
        self._source_ids.add(source.source_file_id)

    def _storage_id(self, storage: StorageFormat) -> int:
        parameters = canonical_json(dict(storage.parameters))
        key = (storage.codec.lower(), storage.codec_version, storage.quality, storage.pixel_format,
               storage.bit_depth, storage.encoder, storage.encoder_version, parameters)
        if key not in self._storage_ids:
            cursor = self.connection.execute(
                """INSERT INTO storage_formats(codec,codec_version,quality,pixel_format,bit_depth,encoder,
                   encoder_version,parameters_json,description) VALUES (?,?,?,?,?,?,?,?,?)""", key + (storage.description,)
            )
            self._storage_ids[key] = int(cursor.lastrowid)
        return self._storage_ids[key]

    def add(self, record: FrameRecord, source: AcquisitionSegment) -> None:
        if self._closed:
            raise RuntimeError("shard writer is closed")
        blob = record.encoded_bytes
        blob_hash = record.blob_hash
        pixel_hash = record.decoded_pixel_hash
        try:
            self.register_source(source)
            storage_id = self._storage_id(record.storage_format) if record.storage_format else None
            self.connection.execute(
                """INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record.frame_id, record.source_file_id, record.source_frame_number,
                 record.timestamp_ns, record.source_timestamp_ns, record.timestamp_source,
                 record.clock_source, record.timezone, record.utc_conversion,
                 record.timestamp_precision_ns, record.synchronization_method,
                 record.known_offset_ns, record.known_drift_ppb, int(record.interpolated),
                 storage_id, sqlite3.Binary(blob) if blob is not None else None, len(blob or b""),
                 record.width, record.height, blob_hash.value if blob_hash else None,
                 blob_hash.algorithm if blob_hash else None, pixel_hash.value if pixel_hash else None,
                 pixel_hash.algorithm if pixel_hash else None, record.status.value),
            )
        except sqlite3.Error as exc:
            raise FormatError(f"cannot add frame {record.frame_id}: {exc}") from exc
        self.frame_count += 1
        self.encoded_bytes += len(blob or b"")
        self.first_frame = record.frame_id if self.first_frame is None else min(self.first_frame, record.frame_id)
        self.last_frame = record.frame_id if self.last_frame is None else max(self.last_frame, record.frame_id)
        if record.timestamp_ns is not None:
            self.first_timestamp = record.timestamp_ns if self.first_timestamp is None else min(self.first_timestamp, record.timestamp_ns)
            self.last_timestamp = record.timestamp_ns if self.last_timestamp is None else max(self.last_timestamp, record.timestamp_ns)
        self._pending += 1
        if self._pending >= self.batch_size:
            try:
                self.connection.commit()
                self.connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise FormatError(f"cannot commit frame batch in {self.partial_path}: {exc}") from exc
            self._pending = 0

    def finalize(self) -> Shard:
        if self._closed:
            raise RuntimeError("shard writer is closed")
        metadata = {
            "shard_uuid": str(self.shard_uuid), "format_version": FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION, "created_at": self.created_at,
            "created_by": "pelagia_interchange", "camera_or_stream_id": str(self.stream_uuid),
            "camera_or_stream_name": self.stream_name, "first_frame": self.first_frame,
            "last_frame": self.last_frame, "frame_count": self.frame_count,
            "first_timestamp": self.first_timestamp, "last_timestamp": self.last_timestamp,
            "encoded_bytes": self.encoded_bytes,
        }
        try:
            self.connection.executemany(
                "INSERT OR REPLACE INTO shard_metadata(key,value) VALUES (?,?)",
                ((key, json.dumps(value, separators=(",", ":"))) for key, value in metadata.items()),
            )
            self.connection.commit()
            self.connection.execute("ANALYZE")
            self.connection.commit()
            check = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
            if check != "ok":
                raise IntegrityError(f"new shard failed integrity_check: {check}")
        except sqlite3.Error as exc:
            self.connection.rollback()
            self.connection.close()
            self._closed = True
            raise FormatError(f"cannot finalize SQLite shard {self.partial_path}: {exc}") from exc
        except IntegrityError:
            self.connection.close()
            self._closed = True
            raise
        self.connection.close(); self._closed = True
        self.partial_path.replace(self.final_path)
        return Shard(self.final_path, self.shard_uuid, self.stream_uuid, self.stream_name,
                     self.first_frame, self.last_frame, self.frame_count)

    def abandon(self) -> Path:
        """Close while deliberately preserving the partial file for recovery."""
        if not self._closed:
            self.connection.rollback()
            self.connection.close()
            self._closed = True
        return self.partial_path


class ShardReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connection(self) -> sqlite3.Connection:
        return _connect(self.path, read_only=True)

    def metadata(self) -> dict[str, Any]:
        with self._connection() as connection:
            try:
                return {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT key,value FROM shard_metadata")}
            except sqlite3.Error as exc:
                raise FormatError(f"cannot read shard metadata from {self.path}: {exc}") from exc

    def integrity_check(self) -> str:
        with self._connection() as connection:
            try:
                return str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            except sqlite3.Error as exc:
                raise FormatError(f"cannot check shard {self.path}: {exc}") from exc

    def tables(self) -> set[str]:
        with self._connection() as connection:
            try:
                return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            except sqlite3.Error as exc:
                raise FormatError(f"cannot inspect tables in {self.path}: {exc}") from exc

    def iter_frames(
        self, *, frame_start: int | None = None, frame_end: int | None = None,
        acquisition_segment_id: int | None = None, timestamp_start: int | None = None,
        timestamp_end: int | None = None,
    ) -> Iterator[Frame]:
        clauses: list[str] = []
        values: list[int] = []
        for sql, value in (("f.frame_id>=?", frame_start), ("f.frame_id<=?", frame_end),
                           ("f.acquisition_segment_id=?", acquisition_segment_id), ("f.timestamp_ns>=?", timestamp_start),
                           ("f.timestamp_ns<=?", timestamp_end)):
            if value is not None:
                clauses.append(sql); values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = """SELECT f.*, s.codec storage_codec,s.codec_version storage_codec_version,s.quality storage_quality,
                 s.pixel_format storage_pixel_format,s.bit_depth storage_bit_depth,s.encoder storage_encoder,
                 s.encoder_version storage_encoder_version,s.parameters_json storage_parameters_json,
                 s.description storage_description FROM frames f LEFT JOIN storage_formats s USING(storage_id)""" + where + " ORDER BY f.frame_id"
        connection = self._connection()
        try:
            cursor = connection.execute(sql, values)
            while row := cursor.fetchone():
                storage = None
                if row["storage_id"] is not None:
                    storage = StorageFormat(row["storage_codec"], row["storage_codec_version"], row["storage_quality"],
                                            row["storage_pixel_format"], row["storage_bit_depth"], row["storage_encoder"],
                                            row["storage_encoder_version"], json.loads(row["storage_parameters_json"]),
                                            row["storage_description"])
                blob_hash = HashRecord(row["hash_algorithm"], "stored_blob", row["hash"]) if row["hash"] else None
                pixel_hash = HashRecord(row["decoded_pixel_hash_algorithm"], "decoded_pixels", row["decoded_pixel_hash"]) if row["decoded_pixel_hash"] else None
                yield Frame(FrameRecord(
                    frame_id=row["frame_id"], source_file_id=row["acquisition_segment_id"],
                    source_frame_number=row["acquisition_frame_number"], encoded_bytes=row["blob"],
                    storage_format=storage, timestamp_ns=row["timestamp_ns"], source_timestamp_ns=row["source_timestamp_ns"],
                    timestamp_source=row["timestamp_source"], clock_source=row["clock_source"], timezone=row["timezone"],
                    utc_conversion=row["utc_conversion"], timestamp_precision_ns=row["timestamp_precision_ns"],
                    synchronization_method=row["synchronization_method"], known_offset_ns=row["known_offset_ns"],
                    known_drift_ppb=row["known_drift_ppb"], interpolated=bool(row["interpolated"]), width=row["width"],
                    height=row["height"], status=row["status"], blob_hash=blob_hash, decoded_pixel_hash=pixel_hash,
                    declared_byte_size=row["byte_size"],
                ))
        except sqlite3.Error as exc:
            raise FormatError(f"cannot read frames from {self.path}: {exc}") from exc
        finally:
            connection.close()

    def get_frame(self, frame_id: int) -> Frame:
        frame = next(self.iter_frames(frame_start=frame_id, frame_end=frame_id), None)
        if frame is None:
            raise FrameNotFoundError(frame_id)
        return frame

    def counts(self) -> dict[str, int]:
        with self._connection() as connection:
            try:
                row = connection.execute("SELECT count(*) frames,coalesce(sum(byte_size),0) bytes FROM frames").fetchone()
                return {"frames": int(row[0]), "encoded_bytes": int(row[1])}
            except sqlite3.Error as exc:
                raise FormatError(f"cannot count frames in {self.path}: {exc}") from exc

    def aggregate_metadata(self) -> dict[str, int | None]:
        with self._connection() as connection:
            try:
                row = connection.execute(
                    "SELECT count(*),min(frame_id),max(frame_id),min(timestamp_ns),max(timestamp_ns),coalesce(sum(byte_size),0) FROM frames"
                ).fetchone()
                return {"frame_count": int(row[0]), "first_frame": row[1], "last_frame": row[2],
                        "first_timestamp": row[3], "last_timestamp": row[4], "encoded_bytes": int(row[5])}
            except sqlite3.Error as exc:
                raise FormatError(f"cannot aggregate shard {self.path}: {exc}") from exc

    def source_progress(self, source_file_id: int) -> dict[str, int | None]:
        """Return durable prefix information for one acquisition segment."""
        with self._connection() as connection:
            try:
                row = connection.execute(
                    "SELECT count(*),min(acquisition_frame_number),max(acquisition_frame_number),max(frame_id) "
                    "FROM frames WHERE acquisition_segment_id=?", (source_file_id,)
                ).fetchone()
                return {"frame_count": int(row[0]), "first_source_frame": row[1],
                        "last_source_frame": row[2], "last_frame": row[3]}
            except sqlite3.Error as exc:
                raise FormatError(f"cannot inspect source progress in {self.path}: {exc}") from exc

    def summary_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        """Return status/codec distributions without reading frame BLOBs."""
        with self._connection() as connection:
            try:
                statuses = {str(name): int(count) for name, count in connection.execute(
                    "SELECT status,count(*) FROM frames GROUP BY status")}
                codecs = {str(name): int(count) for name, count in connection.execute(
                    "SELECT coalesce(s.codec,'none'),count(*) FROM frames f LEFT JOIN storage_formats s USING(storage_id) GROUP BY s.codec")}
                return statuses, codecs
            except sqlite3.Error as exc:
                raise FormatError(f"cannot summarize shard {self.path}: {exc}") from exc


REQUIRED_SHARD_TABLES = REQUIRED_TABLES
