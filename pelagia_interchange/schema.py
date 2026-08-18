from __future__ import annotations

import sqlite3

from .constants import FORMAT_VERSION, SCHEMA_VERSION

DDL = """
CREATE TABLE storage_formats (
 storage_id INTEGER PRIMARY KEY,
 codec TEXT NOT NULL,
 codec_version TEXT,
 quality INTEGER,
 pixel_format TEXT,
 bit_depth INTEGER,
 encoder TEXT,
 encoder_version TEXT,
 parameters_json TEXT NOT NULL DEFAULT '{}',
 description TEXT,
 UNIQUE(codec, codec_version, quality, pixel_format, bit_depth, encoder, encoder_version, parameters_json)
);
CREATE TABLE source_files (
 source_file_id INTEGER PRIMARY KEY,
 source_uuid TEXT NOT NULL UNIQUE,
 original_filename TEXT NOT NULL,
 original_relative_path TEXT,
 original_absolute_path TEXT,
 byte_size INTEGER,
 hash TEXT,
 hash_algorithm TEXT,
 container TEXT,
 codec TEXT,
 pixel_format TEXT,
 width INTEGER,
 height INTEGER,
 frame_rate_num INTEGER,
 frame_rate_den INTEGER,
 frame_count INTEGER,
 start_timestamp TEXT,
 end_timestamp TEXT
);
CREATE TABLE frames (
 frame_id INTEGER PRIMARY KEY,
 source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
 source_frame_number INTEGER NOT NULL,
 timestamp_ns INTEGER,
 source_timestamp_ns INTEGER,
 timestamp_source TEXT,
 clock_source TEXT,
 timezone TEXT,
 utc_conversion TEXT,
 timestamp_precision_ns INTEGER,
 synchronization_method TEXT,
 known_offset_ns INTEGER,
 known_drift_ppb REAL,
 interpolated INTEGER NOT NULL DEFAULT 0 CHECK(interpolated IN (0,1)),
 storage_id INTEGER REFERENCES storage_formats(storage_id),
 blob BLOB,
 byte_size INTEGER NOT NULL,
 width INTEGER,
 height INTEGER,
 hash TEXT,
 hash_algorithm TEXT,
 decoded_pixel_hash TEXT,
 decoded_pixel_hash_algorithm TEXT,
 status TEXT NOT NULL CHECK(status IN ('valid','missing','decode_failed','duplicate','intentionally_removed','timestamp_invalid','corrupt')),
 UNIQUE(source_file_id, source_frame_number),
 CHECK((blob IS NOT NULL AND storage_id IS NOT NULL) OR (blob IS NULL AND status != 'valid'))
);
CREATE INDEX frames_source_number ON frames(source_file_id, source_frame_number);
CREATE INDEX frames_timestamp ON frames(timestamp_ns);
CREATE TABLE shard_metadata (
 key TEXT PRIMARY KEY,
 value TEXT NOT NULL
);
"""

REQUIRED_TABLES = {"frames", "source_files", "storage_formats", "shard_metadata"}


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(DDL)
    connection.executemany(
        "INSERT INTO shard_metadata(key, value) VALUES (?, ?)",
        (("format_version", FORMAT_VERSION), ("schema_version", SCHEMA_VERSION)),
    )

