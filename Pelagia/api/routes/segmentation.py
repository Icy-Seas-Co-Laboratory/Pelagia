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
        min_perimeter: int | float = 0
        max_perimeter: int | float | None = None
        padding: int = 0
        roi_encoding: Literal["png", "raw", "zstd", "auto"] = "zstd"
        zstd_min_bytes: int = 1024

    router = APIRouter(prefix="/segmentation", tags=["segmentation"])

    @router.post("/frames/{frame_id}")
    def segment_stored_frame(request: Request, frame_id: int, body: SegmentFrameRequest) -> dict:
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
            frame = retrieve_frame(frame_id, context=context)
            detections = segment_frame(
                frame,
                threshold=body.threshold,
                min_perimeter=body.min_perimeter,
                max_perimeter=body.max_perimeter,
                padding=body.padding,
                roi_encoding=body.roi_encoding,
                zstd_min_bytes=body.zstd_min_bytes,
            )
            inserted = repository.replace_detections(
                frame_record.run_id,
                frame_record.asset_id,
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
else:
    router = None
