from __future__ import annotations

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
    payload: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class JobsListResponse(FlexibleModel):
    jobs: list[JobSummary]


class JobDetailResponse(FlexibleModel):
    job: JobSummary


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
