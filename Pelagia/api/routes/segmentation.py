from __future__ import annotations

from typing import Literal

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ...processing.frame_store import retrieve_frame
    from ...processing.segmentation import segment_frame
    from ._common import as_response, detection_summary, get_context, get_repository

    class SegmentFrameRequest(BaseModel):
        threshold: int | float | None = None
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
            frame = retrieve_frame(frame_id, context=context)
            detections = segment_frame(
                frame,
                threshold=body.threshold,
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
        defaults = get_context(request).config.processing.segmentation
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
