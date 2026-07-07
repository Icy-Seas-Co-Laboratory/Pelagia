from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    """Base response model that documents common fields without dropping extras."""

    model_config = ConfigDict(extra="allow")


class PageMetadata(FlexibleModel):
    limit: int | None = None
    offset: int = 0
    count: int = 0
    next_offset: int | None = None


class BBox(FlexibleModel):
    x: int | float
    y: int | float
    w: int | float
    h: int | float


class RawAssetSummary(FlexibleModel):
    id: str | None = None
    run_id: str | None = None
    filename: str | None = None
    path: str | None = None
    kind: str | None = None
    collections: list[str] = Field(default_factory=list)
    frame_count: int | None = None


class FrameSummary(FlexibleModel):
    id: str | None = None
    run_id: str | None = None
    asset_id: str | None = None
    frame_index: int | None = None
    width: int | None = None
    height: int | None = None
    has_preprocessed_payload: bool | None = None


class DetectionSummary(FlexibleModel):
    id: str | None = None
    run_id: str | None = None
    asset_id: str | None = None
    frame_id: str | None = None
    frame_index: int | None = None
    roi_index: int | None = None
    bbox_x: int | None = None
    bbox_y: int | None = None
    bbox_w: int | None = None
    bbox_h: int | None = None
    crop_bbox_x: int | None = None
    crop_bbox_y: int | None = None
    crop_bbox_w: int | None = None
    crop_bbox_h: int | None = None
    bbox: BBox | None = None
    crop_bbox: BBox | None = None
    area: float | None = None
    perimeter: float | None = None
    roi_encoding: str | None = None
    roi_format: str | None = None
    roi_payload_bytes: int | None = None
    mask_encoding: str | None = None
    mask_format: str | None = None
    mask_payload_bytes: int | None = None
    candidate_detection_id: str | None = None
    refined_detection_id: str | None = None
    primary_candidate_detection_id: str | None = None
    candidate_detection_ids: list[str] | None = None
    refinement_relationship: str | None = None
    refined_roi_url: str | None = None
    refined_mask_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatrixImageResponse(FlexibleModel):
    dtype: str
    shape: list[int]
    data: list[Any]
    scale: float | None = None
    requested_width: int | None = None
    requested_height: int | None = None


class DetectionImageMatrixResponse(MatrixImageResponse):
    detection_id: str
    frame_id: str | None = None
    asset_id: str | None = None
    payload_kind: str
    pad_square: bool | None = None
    inverted: bool | None = None
    scale_bar: bool | None = None


class DetectionsListResponse(FlexibleModel):
    detections: list[DetectionSummary]
    page: PageMetadata


class DetectionDetailResponse(FlexibleModel):
    detection: DetectionSummary


class AssetsListResponse(FlexibleModel):
    assets: list[RawAssetSummary]


class AssetDetailResponse(FlexibleModel):
    asset: RawAssetSummary


class FramesListResponse(FlexibleModel):
    frames: list[FrameSummary]


class JobSummary(FlexibleModel):
    id: str | None = None
    run_id: str | None = None
    asset_id: str | None = None
    stage: str | None = None
    status: str | None = None
    priority: int | None = None
    attempt_count: int | None = None
    max_attempts: int | None = None
    worker_id: str | None = None
    summary: str | None = None
    progress: dict[str, Any] | None = None
    control_reason: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    lease_expires_at: datetime | None = None
    payload_bytes: int | None = None
    result_bytes: int | None = None
    progress_bytes: int | None = None
    logs_tail_count: int | None = None
    payload: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class JobsListResponse(FlexibleModel):
    jobs: list[JobSummary]


class JobSummaryProgress(FlexibleModel):
    known_total_units: float | None = None
    completed_units: float | None = None
    failed_units: float | None = None
    skipped_units: float | None = None
    percent: float | None = None


class JobAggregateSummary(FlexibleModel):
    stage: str | None = None
    status: str | None = None
    job_count: int = 0
    queued: int | None = None
    leased: int | None = None
    working: int | None = None
    paused: int | None = None
    succeeded: int | None = None
    failed: int | None = None
    cancelled: int | None = None
    dead_lettered: int | None = None
    progress: JobSummaryProgress | None = None


class JobsSummaryResponse(FlexibleModel):
    filters: dict[str, Any]
    total: JobAggregateSummary
    by_stage: list[JobAggregateSummary] = Field(default_factory=list)
    by_status: list[JobAggregateSummary] = Field(default_factory=list)
    recent_jobs: list[JobSummary] = Field(default_factory=list)


class JobDetailResponse(FlexibleModel):
    job: JobSummary


class JobsClearResponse(FlexibleModel):
    matched_count: int = 0
    cancellable_count: int = 0
    cancelled_count: int = 0
    deleted_count: int = 0
    dry_run: bool = False
    jobs: list[JobSummary] = Field(default_factory=list)


class FrameProcessingStatus(FlexibleModel):
    project_id: str | None = None
    frame_id: str
    asset_id: str | None = None
    run_id: str | None = None
    frame_index: int | None = None
    collections: list[str] = Field(default_factory=list)
    preprocessing_status: str = "unknown"
    preprocessing_job_id: str | None = None
    preprocessing_completed_at: datetime | None = None
    candidate_detection_status: str = "unknown"
    candidate_detection_job_id: str | None = None
    candidate_detection_completed_at: datetime | None = None
    candidate_detection_count: int = 0
    roi_refinement_status: str = "unknown"
    roi_refinement_job_id: str | None = None
    roi_refinement_completed_at: datetime | None = None
    refined_detection_count: int = 0
    unrefined_candidate_count: int = 0
    updated_at: datetime | None = None
    asset_filename: str | None = None
    asset_kind: str | None = None


class ProcessingStatusSummary(FlexibleModel):
    total_frame_count: int = 0
    preprocessing_succeeded_count: int = 0
    candidate_detection_succeeded_count: int = 0
    roi_refinement_succeeded_count: int = 0
    frames_with_candidates_count: int = 0
    frames_with_refined_rois_count: int = 0
    candidate_detection_count: int = 0
    refined_detection_count: int = 0
    unrefined_candidate_count: int = 0
    updated_at: datetime | None = None
    by_status: dict[str, dict[str, int]] = Field(default_factory=dict)


class ProcessingStatusSnapshot(FlexibleModel):
    id: str | None = None
    project_id: str
    session_id: str | None = None
    status_version: int = 1
    generated_at: datetime | None = None
    updated_at: datetime | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class ProcessingStatusSummaryResponse(FlexibleModel):
    summary: ProcessingStatusSummary
    snapshot: ProcessingStatusSnapshot


class ProcessingStatusFramesResponse(FlexibleModel):
    frames: list[FrameProcessingStatus] = Field(default_factory=list)
    next_cursor: str | None = None
    page: PageMetadata


class ProcessingStatusFrameIdsResponse(FlexibleModel):
    frame_ids: list[str] = Field(default_factory=list)
    next_cursor: str | None = None
    page: PageMetadata


class ProcessingStatusRebuildResponse(FlexibleModel):
    status: str
    rebuilt_frame_count: int = 0
    summary: ProcessingStatusSummary
    snapshot: ProcessingStatusSnapshot


class RunSummary(FlexibleModel):
    id: str | None = None
    run_id: str | None = None
    run_key: str | None = None
    status: str | None = None
    source_path: str | None = None
    source_type: str | None = None


class RunsListResponse(FlexibleModel):
    runs: list[RunSummary]


class RunDetailResponse(FlexibleModel):
    run: RunSummary


class FrameContextResponse(FlexibleModel):
    frame: FrameSummary
    asset: RawAssetSummary
    image_urls: dict[str, str | None]
    detections: list[DetectionSummary]
    detection_count: int
    page: PageMetadata


class OptionsResponse(FlexibleModel):
    pipeline_stage_order: list[str] = Field(default_factory=list)
    supported: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    fields: dict[str, Any] = Field(default_factory=dict)


class SystemCapabilitiesResponse(FlexibleModel):
    name: str
    version: str
    supported: dict[str, Any] = Field(default_factory=dict)
    processing: dict[str, Any] = Field(default_factory=dict)
