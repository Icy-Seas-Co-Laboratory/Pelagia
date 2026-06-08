from __future__ import annotations

from typing import Literal

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ...processing.frame_store import retrieve_frame
    from ...processing.detection_candidate import segment_frame
    from ._common import as_response, detection_summary, get_context, get_repository

    class SegmentFrameRequest(BaseModel):
        threshold: int | float | None = None
        frame_payload_kind: Literal["original", "raw", "preprocessed", "processed", "corrected"] = "original"
        apply_preprocessing: bool | None = None
        flatfield_correction: bool | None = None
        flatfield_q: float | None = None
        flatfield_axis: int | None = None
        apply_mask: bool | None = None
        crop_enabled: bool | None = None
        crop_x: int | None = None
        crop_y: int | None = None
        crop_w: int | None = None
        crop_h: int | None = None
        background_correction: bool | None = None
        background_percentile: int | float | None = None
        invert_intensity: bool | None = None
        min_perimeter: int | float | None = None
        max_perimeter: int | float | None = None
        padding: int | None = None
        roi_encoding: Literal["png", "raw", "zstd", "auto"] | None = None
        zstd_min_bytes: int | None = None

    class QueueSegmentationRequest(SegmentFrameRequest):
        run_id: str | None = None
        asset_id: str | None = None
        frame_ids: list[str] | None = None
        start_frame: int | None = None
        end_frame: int | None = None
        limit: int | None = None
        priority: int | None = None
        depends_on: list[str] | None = None

    router = APIRouter(prefix="/segmentation", tags=["segmentation"])

    @router.post("/frames/{frame_id}")
    def segment_stored_frame(request: Request, frame_id: str, body: SegmentFrameRequest) -> dict:
        context = get_context(request)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")
        if not frame_record.run_id:
            raise HTTPException(
                status_code=409,
                detail=f"Frame {frame_id!r} does not include a run_id.",
            )

        try:
            defaults = context.config.processing.segmentation
            flatfield_defaults = context.config.processing.flatfield
            preprocessing_defaults = context.config.processing.preprocessing
            frame = retrieve_frame(frame_id, context=context, payload_kind=body.frame_payload_kind)
            detections = segment_frame(
                frame,
                frame_record=frame_record,
                threshold=body.threshold,
                apply_preprocessing=(
                    body.frame_payload_kind in {"original", "raw"}
                    if body.apply_preprocessing is None
                    else body.apply_preprocessing
                ),
                flatfield_correction=(
                    flatfield_defaults.flatfield_correction
                    if body.flatfield_correction is None
                    else body.flatfield_correction
                ),
                flatfield_q=flatfield_defaults.flatfield_q if body.flatfield_q is None else body.flatfield_q,
                flatfield_axis=(
                    flatfield_defaults.flatfield_axis if body.flatfield_axis is None else body.flatfield_axis
                ),
                apply_mask=preprocessing_defaults.apply_mask if body.apply_mask is None else body.apply_mask,
                crop_enabled=(
                    preprocessing_defaults.crop_enabled
                    if body.crop_enabled is None
                    else body.crop_enabled
                ),
                crop_x=preprocessing_defaults.crop_x if body.crop_x is None else body.crop_x,
                crop_y=preprocessing_defaults.crop_y if body.crop_y is None else body.crop_y,
                crop_w=preprocessing_defaults.crop_w if body.crop_w is None else body.crop_w,
                crop_h=preprocessing_defaults.crop_h if body.crop_h is None else body.crop_h,
                background_correction=(
                    preprocessing_defaults.background_correction
                    if body.background_correction is None
                    else body.background_correction
                ),
                background_percentile=(
                    preprocessing_defaults.background_percentile
                    if body.background_percentile is None
                    else body.background_percentile
                ),
                invert_intensity=(
                    preprocessing_defaults.invert_intensity
                    if body.invert_intensity is None
                    else body.invert_intensity
                ),
                min_perimeter=defaults.min_perimeter if body.min_perimeter is None else body.min_perimeter,
                max_perimeter=defaults.max_perimeter if body.max_perimeter is None else body.max_perimeter,
                padding=defaults.padding if body.padding is None else body.padding,
                roi_encoding=defaults.roi_encoding if body.roi_encoding is None else body.roi_encoding,
                zstd_min_bytes=defaults.zstd_min_bytes if body.zstd_min_bytes is None else body.zstd_min_bytes,
            )
            inserted = repository.replace_frame_detections(
                frame_record.run_id,
                [frame_id],
                detections,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return as_response(
            {
                "frame_id": frame_id,
                "run_id": frame_record.run_id,
                "asset_id": frame_record.asset_id,
                "detection_count": len(inserted),
                "detections": [detection_summary(row) for row in inserted],
            }
        )

    @router.post("/jobs")
    def queue_segmentation_job(request: Request, body: QueueSegmentationRequest) -> dict:
        repository = get_repository(request)
        processing_defaults = get_context(request).config.processing
        defaults = processing_defaults.segmentation
        flatfield_defaults = processing_defaults.flatfield
        preprocessing_defaults = processing_defaults.preprocessing
        run_id = body.run_id
        asset_id = body.asset_id

        if asset_id is None and body.frame_ids:
            first_frame = repository.get_frame_record(body.frame_ids[0])
            if first_frame is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Frame {body.frame_ids[0]!r} was not found.",
                )
            run_id = run_id or first_frame.run_id
            asset_id = first_frame.asset_id

        if asset_id is None:
            raise HTTPException(
                status_code=422,
                detail="Segmentation jobs require asset_id or at least one frame_id.",
            )

        if run_id is None:
            asset = repository.get_asset(asset_id)
            if asset is None:
                raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} was not found.")
            run_id = asset.get("run_id")

        payload = {
            "frame_ids": body.frame_ids or [],
            "start_frame": body.start_frame,
            "end_frame": body.end_frame,
            "limit": body.limit,
            "threshold": body.threshold,
            "frame_payload_kind": body.frame_payload_kind,
            "apply_preprocessing": (
                body.frame_payload_kind in {"original", "raw"}
                if body.apply_preprocessing is None
                else body.apply_preprocessing
            ),
            "flatfield_correction": (
                flatfield_defaults.flatfield_correction
                if body.flatfield_correction is None
                else body.flatfield_correction
            ),
            "flatfield_q": flatfield_defaults.flatfield_q if body.flatfield_q is None else body.flatfield_q,
            "flatfield_axis": flatfield_defaults.flatfield_axis if body.flatfield_axis is None else body.flatfield_axis,
            "apply_mask": preprocessing_defaults.apply_mask if body.apply_mask is None else body.apply_mask,
            "crop_enabled": (
                preprocessing_defaults.crop_enabled
                if body.crop_enabled is None
                else body.crop_enabled
            ),
            "crop_x": preprocessing_defaults.crop_x if body.crop_x is None else body.crop_x,
            "crop_y": preprocessing_defaults.crop_y if body.crop_y is None else body.crop_y,
            "crop_w": preprocessing_defaults.crop_w if body.crop_w is None else body.crop_w,
            "crop_h": preprocessing_defaults.crop_h if body.crop_h is None else body.crop_h,
            "background_correction": (
                preprocessing_defaults.background_correction
                if body.background_correction is None
                else body.background_correction
            ),
            "background_percentile": (
                preprocessing_defaults.background_percentile
                if body.background_percentile is None
                else body.background_percentile
            ),
            "invert_intensity": (
                preprocessing_defaults.invert_intensity
                if body.invert_intensity is None
                else body.invert_intensity
            ),
            "min_perimeter": defaults.min_perimeter if body.min_perimeter is None else body.min_perimeter,
            "max_perimeter": defaults.max_perimeter if body.max_perimeter is None else body.max_perimeter,
            "padding": defaults.padding if body.padding is None else body.padding,
            "roi_encoding": defaults.roi_encoding if body.roi_encoding is None else body.roi_encoding,
            "zstd_min_bytes": defaults.zstd_min_bytes if body.zstd_min_bytes is None else body.zstd_min_bytes,
        }
        job = repository.create_job(
            "segment",
            run_id=run_id,
            asset_id=asset_id,
            priority=body.priority,
            payload=payload,
            depends_on=body.depends_on or [],
            summary=f"segment queued for asset {asset_id}",
        )
        return {"job": as_response(job)}
else:
    router = None
