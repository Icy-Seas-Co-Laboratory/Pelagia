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
class AcquisitionSegment:
    """One contiguous acquisition run whose frame payloads are canonical.

    ``import_provenance`` is deliberately optional: it records an AVI or other
    legacy carrier when one was used to create a package, but is never the
    authoritative representation of a frame.
    """
    acquisition_segment_id: int
    acquisition_segment_uuid: UUID
    segment_name: str
    acquisition_mode: str = "direct_frame_capture"
    expected_frame_count: int | None = None
    capture_configuration: Mapping[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    ended_at: str | None = None
    import_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.acquisition_segment_id < 1:
            raise ValueError("acquisition_segment_id must be positive")
        if not self.segment_name.strip():
            raise ValueError("segment_name must not be empty")
        if self.acquisition_mode not in {"direct_frame_capture", "imported_video"}:
            raise ValueError("acquisition_mode must be direct_frame_capture or imported_video")
        if self.expected_frame_count is not None and self.expected_frame_count < 0:
            raise ValueError("expected_frame_count must be non-negative")
        if self.acquisition_mode == "imported_video" and self.import_provenance is None:
            raise ValueError("imported_video segments require import_provenance")

    # Transitional read aliases.  They keep the Python API usable for legacy
    # ingestion callers while the on-disk 0.2 vocabulary remains unambiguous.
    @property
    def source_file_id(self) -> int: return self.acquisition_segment_id
    @property
    def source_uuid(self) -> UUID: return self.acquisition_segment_uuid
    @property
    def original_filename(self) -> str: return self.segment_name
    @property
    def original_relative_path(self) -> str | None: return (self.import_provenance or {}).get("original_relative_path")
    @property
    def original_absolute_path(self) -> str | None: return (self.import_provenance or {}).get("original_absolute_path")
    @property
    def byte_size(self) -> int | None: return (self.import_provenance or {}).get("byte_size")
    @property
    def file_hash(self) -> HashRecord | None:
        value = (self.import_provenance or {}).get("file_hash")
        return HashRecord(value["algorithm"], "source_file", value["value"]) if value else None
    @property
    def container(self) -> str | None: return (self.import_provenance or {}).get("container")
    @property
    def codec(self) -> str | None: return (self.import_provenance or {}).get("codec")
    @property
    def pixel_format(self) -> str | None: return (self.import_provenance or {}).get("pixel_format")
    @property
    def width(self) -> int | None: return (self.import_provenance or {}).get("width")
    @property
    def height(self) -> int | None: return (self.import_provenance or {}).get("height")
    @property
    def frame_rate_num(self) -> int | None:
        rate = (self.import_provenance or {}).get("frame_rate") or (None, None); return rate[0]
    @property
    def frame_rate_den(self) -> int | None:
        rate = (self.import_provenance or {}).get("frame_rate") or (None, None); return rate[1]
    @property
    def frame_count(self) -> int | None: return self.expected_frame_count
    @property
    def start_timestamp(self) -> str | None: return self.started_at
    @property
    def end_timestamp(self) -> str | None: return self.ended_at


# Kept as an import alias only; newly authored packages must use
# AcquisitionSegment and register_acquisition_segment.
SourceFile = AcquisitionSegment


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

    @property
    def acquisition_segment_id(self) -> int:
        return self.source_file_id

    @property
    def acquisition_frame_number(self) -> int:
        return self.source_frame_number


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
