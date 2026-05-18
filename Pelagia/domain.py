from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AssetKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"


class PipelineStage(str, Enum):
    INGEST_RUN = "ingest_run"
    EXTRACT_FRAMES = "extract_frames"
    SEGMENT = "segment"
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
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTERED = "dead_lettered"


@dataclass(slots=True)
class RawAssetManifest:
    asset_id: str
    asset_key: str
    path: str
    kind: AssetKind
    size_bytes: int
    checksum: str
    media_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


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
    frame_png: bytes
    frame_hash: str
    captured_at: datetime | None = None
    source_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionRecord:
    run_id: str
    frame_id: int
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
