from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


DEFAULT_COLLECTION = "none"


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row or {})


def normalize_collections(value: Any = None) -> list[str]:
    """Normalize user-supplied collection names into a stable non-empty list."""
    if value is None:
        return [DEFAULT_COLLECTION]
    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        raw_values = list(value)

    collections: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        collection = str(item).strip()
        if not collection:
            continue
        if collection in seen:
            continue
        collections.append(collection)
        seen.add(collection)
    return collections or [DEFAULT_COLLECTION]


class AssetKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"


class PipelineStage(str, Enum):
    INGEST_RUN = "ingest_run"
    EXTRACT_FRAMES = "extract_frames"
    BACKGROUND_FRAMES = "background_frames"
    PREPROCESS_FRAMES = "preprocess_frames"
    SEGMENT = "segment"
    ROI_REFINEMENT = "roi_refinement"
    CLASSIFY = "classify"
    PUBLISH = "publish"
    TRAIN_MODEL = "train_model"
    IO_IMPORT = "io_import"
    IO_EXPORT = "io_export"
    IO_UPLOAD = "io_upload"
    IO_DOWNLOAD = "io_download"


class JobStatus(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    WORKING = "working"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTERED = "dead_lettered"


@dataclass(slots=True)
class RawAssetManifest:
    asset_id: str
    filename: str
    path: str
    kind: AssetKind
    size_bytes: int
    checksum: str
    collections: list[str] = field(default_factory=lambda: [DEFAULT_COLLECTION])
    media_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.collections = normalize_collections(self.collections)


@dataclass(slots=True)
class RunManifest:
    run_id: str
    run_key: str
    instrument: str
    source_path: str
    source_type: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    assets: list[RawAssetManifest] = field(default_factory=list)


@dataclass(slots=True)
class WorkItem:
    job_id: str
    run_id: str
    stage: PipelineStage
    asset_id: str | None
    status: JobStatus = JobStatus.QUEUED
    priority: int = 100
    max_attempts: int = 3
    depends_on: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlannedRun:
    manifest: RunManifest
    jobs: list[WorkItem] = field(default_factory=list)


@dataclass(slots=True)
class FrameRecord:
    asset_id: str
    frame_index: int
    width: int
    height: int
    preview_thumbhash: bytes
    kvstore_hash: str
    captured_at: datetime | None = None
    source_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    run_id: str | None = None
    bbox_x: int = 0
    bbox_y: int = 0
    parent_frame_id: str | None = None
    payload_ref: str | None = None
    payload_encoding: str | None = None
    payload_format: str | None = None
    payload_dtype: str | None = None
    payload_shape: list[int] = field(default_factory=list)
    preprocessed_kvstore_hash: str | None = None
    preprocessed_preview_thumbhash: bytes | None = None
    preprocessed_payload_ref: str | None = None
    preprocessed_payload_encoding: str | None = None
    preprocessed_payload_format: str | None = None
    preprocessed_payload_dtype: str | None = None
    preprocessed_payload_shape: list[int] = field(default_factory=list)
    preprocessed_metadata: dict[str, Any] = field(default_factory=dict)
    background_kvstore_hash: str | None = None
    background_payload_ref: str | None = None
    background_payload_encoding: str | None = None
    background_payload_format: str | None = None
    background_payload_dtype: str | None = None
    background_payload_shape: list[int] = field(default_factory=list)
    background_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Any) -> "FrameRecord":
        data = _row_dict(row)
        metadata = dict(data.get("metadata") or {})
        return cls(
            id=data.get("id"),
            run_id=None if data.get("run_id") is None else str(data["run_id"]),
            asset_id=str(data["asset_id"]),
            frame_index=int(data["frame_index"]),
            captured_at=data.get("captured_at"),
            width=int(data["width"]),
            height=int(data["height"]),
            bbox_x=int(data.get("bbox_x", metadata.get("bbox_x", 0)) or 0),
            bbox_y=int(data.get("bbox_y", metadata.get("bbox_y", 0)) or 0),
            parent_frame_id=(
                None
                if data.get("parent_frame_id", metadata.get("parent_frame_id")) is None
                else str(data.get("parent_frame_id", metadata.get("parent_frame_id")))
            ),
            source_ref=data.get("source_ref"),
            kvstore_hash=data.get("kvstore_hash") or data.get("frame_hash"),
            preview_thumbhash=data.get("preview_thumbhash", data.get("frame_png", b"")),
            payload_ref=(
                data.get("payload_ref")
                or metadata.get("kvstore_key")
                or data.get("kvstore_hash")
                or data.get("frame_hash")
            ),
            payload_encoding=data.get("payload_encoding") or metadata.get("kvstore_encoding"),
            payload_format=data.get("payload_format") or metadata.get("kvstore_format"),
            payload_dtype=data.get("payload_dtype") or metadata.get("dtype"),
            payload_shape=list(data.get("payload_shape") or metadata.get("shape") or []),
            preprocessed_kvstore_hash=data.get("preprocessed_kvstore_hash"),
            preprocessed_preview_thumbhash=data.get("preprocessed_preview_thumbhash"),
            preprocessed_payload_ref=data.get("preprocessed_payload_ref"),
            preprocessed_payload_encoding=data.get("preprocessed_payload_encoding"),
            preprocessed_payload_format=data.get("preprocessed_payload_format"),
            preprocessed_payload_dtype=data.get("preprocessed_payload_dtype"),
            preprocessed_payload_shape=list(data.get("preprocessed_payload_shape") or []),
            preprocessed_metadata=dict(data.get("preprocessed_metadata") or {}),
            background_kvstore_hash=data.get("background_kvstore_hash"),
            background_payload_ref=data.get("background_payload_ref"),
            background_payload_encoding=data.get("background_payload_encoding"),
            background_payload_format=data.get("background_payload_format"),
            background_payload_dtype=data.get("background_payload_dtype"),
            background_payload_shape=list(data.get("background_payload_shape") or []),
            background_metadata=dict(data.get("background_metadata") or {}),
            metadata=metadata,
            created_at=data.get("created_at"),
        )


@dataclass(slots=True)
class DetectionRecord:
    run_id: str
    frame_id: str
    roi_index: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    area: float
    perimeter: float
    major_axis_length: float
    minor_axis_length: float
    min_gray_value: int
    mean_gray_value: float
    roi_payload: bytes | None
    mask_payload: bytes | None = None
    crop_bbox_x: int | None = None
    crop_bbox_y: int | None = None
    crop_bbox_w: int | None = None
    crop_bbox_h: int | None = None
    roi_encoding: str | None = None
    roi_format: str | None = None
    roi_dtype: str | None = None
    roi_shape: list[int] = field(default_factory=list)
    mask_encoding: str | None = None
    mask_format: str | None = None
    mask_dtype: str | None = None
    mask_shape: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Any) -> "DetectionRecord":
        data = _row_dict(row)
        return cls(
            id=None if data.get("id") is None else str(data["id"]),
            run_id=str(data["run_id"]),
            frame_id=str(data["frame_id"]),
            roi_index=int(data["roi_index"]),
            bbox_x=int(data["bbox_x"]),
            bbox_y=int(data["bbox_y"]),
            bbox_w=int(data["bbox_w"]),
            bbox_h=int(data["bbox_h"]),
            crop_bbox_x=data.get("crop_bbox_x"),
            crop_bbox_y=data.get("crop_bbox_y"),
            crop_bbox_w=data.get("crop_bbox_w"),
            crop_bbox_h=data.get("crop_bbox_h"),
            area=float(data.get("area") or 0),
            perimeter=float(data.get("perimeter") or 0),
            major_axis_length=float(data.get("major_axis_length") or 0),
            minor_axis_length=float(data.get("minor_axis_length") or 0),
            min_gray_value=int(data.get("min_gray_value") or 0),
            mean_gray_value=float(data.get("mean_gray_value") or 0),
            roi_payload=data.get("roi_payload"),
            mask_payload=data.get("mask_payload"),
            roi_encoding=data.get("roi_encoding"),
            roi_format=data.get("roi_format"),
            roi_dtype=data.get("roi_dtype"),
            roi_shape=list(data.get("roi_shape") or []),
            mask_encoding=data.get("mask_encoding"),
            mask_format=data.get("mask_format"),
            mask_dtype=data.get("mask_dtype"),
            mask_shape=list(data.get("mask_shape") or []),
            metadata=dict(data.get("metadata") or {}),
            created_at=data.get("created_at"),
        )


@dataclass(slots=True)
class ModelRecord:
    model_key: str
    model_name: str
    version: str
    task: str = "classification"
    artifact_uri: str | None = None
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClassificationResultRecord:
    detection_id: str
    model_id: str
    label: str | None
    score: float | None
    scores: dict[str, float] = field(default_factory=dict)
    embedding: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
