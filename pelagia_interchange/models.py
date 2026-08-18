from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID, uuid4

from .util import hash_bytes


class FrameStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    DECODE_FAILED = "decode_failed"
    DUPLICATE = "duplicate"
    INTENTIONALLY_REMOVED = "intentionally_removed"
    TIMESTAMP_INVALID = "timestamp_invalid"
    CORRUPT = "corrupt"


@dataclass(frozen=True, slots=True)
class HashRecord:
    algorithm: str
    target: str
    value: str

    def __post_init__(self) -> None:
        if not self.algorithm or not self.target or not self.value:
            raise ValueError("hash records require algorithm, semantic target, and value")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StorageFormat:
    codec: str
    codec_version: str | None = None
    quality: int | None = None
    pixel_format: str | None = None
    bit_depth: int | None = None
    encoder: str | None = None
    encoder_version: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.codec.strip():
            raise ValueError("storage codec must not be empty")
        if self.quality is not None and self.quality < 0:
            raise ValueError("storage quality must be non-negative")
        if self.bit_depth is not None and self.bit_depth <= 0:
            raise ValueError("bit depth must be positive")
        if self.description is None:
            parts = [self.codec.upper()]
            if self.quality is not None:
                parts.append(f"quality {self.quality}")
            if self.pixel_format:
                parts.append(self.pixel_format)
            object.__setattr__(self, "description", ", ".join(parts))

    @property
    def extension(self) -> str:
        return {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png"}.get(self.codec.lower(), ".bin")


@dataclass(frozen=True, slots=True)
class SourceFile:
    source_file_id: int
    source_uuid: UUID
    original_filename: str
    original_relative_path: str | None = None
    original_absolute_path: str | None = None
    byte_size: int | None = None
    file_hash: HashRecord | None = None
    container: str | None = None
    codec: str | None = None
    pixel_format: str | None = None
    width: int | None = None
    height: int | None = None
    frame_rate_num: int | None = None
    frame_rate_den: int | None = None
    frame_count: int | None = None
    start_timestamp: str | None = None
    end_timestamp: str | None = None


@dataclass(slots=True)
class FrameRecord:
    frame_id: int
    source_file_id: int
    source_frame_number: int
    encoded_bytes: bytes | None
    storage_format: StorageFormat | None = None
    timestamp_ns: int | None = None
    source_timestamp_ns: int | None = None
    timestamp_source: str | None = None
    clock_source: str | None = None
    timezone: str | None = None
    utc_conversion: str | None = None
    timestamp_precision_ns: int | None = None
    synchronization_method: str | None = None
    known_offset_ns: int | None = None
    known_drift_ppb: float | None = None
    interpolated: bool = False
    width: int | None = None
    height: int | None = None
    status: FrameStatus | str = FrameStatus.VALID
    blob_hash: HashRecord | None = None
    decoded_pixel_hash: HashRecord | None = None
    declared_byte_size: int | None = None

    def __post_init__(self) -> None:
        self.status = FrameStatus(self.status)
        if self.frame_id < 0 or self.source_frame_number < 0:
            raise ValueError("frame identifiers must be non-negative")
        if self.status is FrameStatus.VALID and self.encoded_bytes is None:
            raise ValueError("valid frames require encoded bytes")
        if self.encoded_bytes is not None and self.storage_format is None:
            raise ValueError("frames with encoded bytes require a storage format")
        if self.blob_hash is None and self.encoded_bytes is not None:
            self.blob_hash = HashRecord("sha256", "stored_blob", hash_bytes(self.encoded_bytes))
        if self.blob_hash is not None and self.blob_hash.target != "stored_blob":
            raise ValueError("blob_hash target must be 'stored_blob'")
        if self.decoded_pixel_hash is not None and self.decoded_pixel_hash.target != "decoded_pixels":
            raise ValueError("decoded_pixel_hash target must be 'decoded_pixels'")
        if self.declared_byte_size is None:
            self.declared_byte_size = len(self.encoded_bytes or b"")


@dataclass(frozen=True, slots=True)
class Frame:
    record: FrameRecord

    @property
    def encoded_bytes(self) -> bytes | None:
        return self.record.encoded_bytes

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        if self.encoded_bytes is None:
            raise ValueError(f"frame {self.record.frame_id} has no retained payload")
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.encoded_bytes)
        return destination


def new_uuid(value: UUID | str | None = None) -> UUID:
    return UUID(str(value)) if value is not None else uuid4()
