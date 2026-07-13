from __future__ import annotations

from typing import Literal
from urllib.parse import urlencode

try:
    from fastapi import APIRouter, HTTPException, Request, Response
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    import numpy as np

    from ..schemas import FrameContextResponse
    from ..auth import require_project_write, scoped_project_id
    from ...domain import PipelineStage
    from ...processing.frame_correction import generate_background_for_frames
    from ...processing.frame_preprocess import preprocess_frame_for_segmentation
    from ...processing.frame_store import retrieve_frame, store_preprocessed_frame
    from ...processing.codec_registry import image_extension
    from ...services.job_commands import FrameBackgroundCommand, PreprocessFramesCommand
    from ._common import (
        as_response,
        detection_summary,
        frame_summary,
        get_context,
        get_repository,
        mark_frame_stage_status,
        page_metadata,
        touch_processing_status_snapshot,
    )
    from ._images import encode_image, preview_image, resize_image_to_dimension, scale_image

    router = APIRouter(prefix="/frame", tags=["frame"])
    frames_router = APIRouter(prefix="/frames", tags=["frame"])
    routers = [frames_router]

    class FramePreprocessRequest(BaseModel):
        frame_id: str | None = None
        frame_ids: list[str] | None = None
        asset_id: str | None = None
        frame_num: int | None = None
        start_frame: int | None = None
        end_frame: int | None = None
        limit: int | None = None
        flatfield_correction: bool | None = None
        flatfield_q: float | None = None
        flatfield_axis: int | None = None
        flatfield_min_field_value: int | float | None = None
        flatfield_max_field_value: int | float | None = None
        apply_mask: bool | None = None
        crop_enabled: bool | None = None
        crop_x: int | None = None
        crop_y: int | None = None
        crop_w: int | None = None
        crop_h: int | None = None
        background_correction: bool | None = None
        background_min_field_value: int | float | None = None
        background_max_field_value: int | float | None = None
        invert_intensity: bool | None = None
        store: bool = True
        encoding: Literal["png", "jpg", "jxl", "jxs", "raw", "zstd"] | None = None
        quality: int | None = None
        response_format: Literal["metadata", "matrix"] = "metadata"

    class QueueFramePreprocessRequest(BaseModel):
        run_id: str | None = None
        asset_id: str | None = None
        frame_id: str | None = None
        frame_ids: list[str] | None = None
        start_frame: int | None = None
        end_frame: int | None = None
        limit: int | None = None
        flatfield_correction: bool | None = None
        flatfield_q: float | None = None
        flatfield_axis: int | None = None
        flatfield_min_field_value: int | float | None = None
        flatfield_max_field_value: int | float | None = None
        apply_mask: bool | None = None
        crop_enabled: bool | None = None
        crop_x: int | None = None
        crop_y: int | None = None
        crop_w: int | None = None
        crop_h: int | None = None
        background_correction: bool | None = None
        background_min_field_value: int | float | None = None
        background_max_field_value: int | float | None = None
        invert_intensity: bool | None = None
        encoding: Literal["png", "jpg", "jxl", "jxs", "raw", "zstd"] | None = None
        quality: int | None = None
        priority: int | None = None
        depends_on: list[str] | None = None

    class FrameBackgroundRequest(BaseModel):
        run_id: str | None = None
        asset_id: str | None = None
        frame_id: str | None = None
        frame_ids: list[str] | None = None
        start_frame: int | None = None
        end_frame: int | None = None
        limit: int | None = None
        payload_kind: Literal["original", "raw", "preprocessed", "processed", "corrected"] = "original"
        encoding: Literal["png", "jpg", "jxl", "jxs", "raw", "zstd"] = "zstd"
        quality: int | None = None

    class QueueFrameBackgroundRequest(FrameBackgroundRequest):
        priority: int | None = None
        depends_on: list[str] | None = None
        dry_run: bool = False

    def _resolve_frame_row(request: Request, frame_id: str | None, asset_id: str | None, frame_num: int | None) -> dict:
        repository = get_repository(request)
        project_id = scoped_project_id(request)
        if frame_id is not None:
            row = repository.get_frame(frame_id, project_id=project_id)
            label = f"Frame {frame_id!r}"
        elif asset_id is not None and frame_num is not None:
            row = repository.get_frame_by_asset_index(asset_id, frame_num, project_id=project_id)
            label = f"Frame {frame_num!r} for asset {asset_id!r}"
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide frame_id, or provide both asset_id and frame_num.",
            )
        if row is None:
            raise HTTPException(status_code=404, detail=f"{label} was not found.")
        return row

    def _frame_data_response(
        *,
        request: Request,
        row: dict,
        payload_kind: str,
        format: str,
        preview_max_dim: int,
        scale: float,
        width: int | None = None,
        height: int | None = None,
    ):
        context = get_context(request).for_project(scoped_project_id(request))
        try:
            frame = retrieve_frame(str(row["id"]), context=context, payload_kind=payload_kind)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        array = frame.read()
        if array is None:
            raise HTTPException(status_code=404, detail=f"Frame {row['id']!r} has no image data.")

        requested = format.lower()
        source_height, source_width = np.asarray(array).shape[:2]
        resized = (
            resize_image_to_dimension(array, width=width, height=height)
            if width is not None or height is not None
            else scale_image(array, scale)
        )
        if width is not None or height is not None:
            size_headers = {
                "X-Pelagia-Width": str(np.asarray(resized).shape[1]),
                "X-Pelagia-Height": str(np.asarray(resized).shape[0]),
            }
            if width is not None:
                size_headers["X-Pelagia-Resize-Width"] = str(width)
            if height is not None:
                size_headers["X-Pelagia-Resize-Height"] = str(height)
        else:
            size_headers = {"X-Pelagia-Scale": str(scale)}

        if requested == "matrix":
            matrix = np.asarray(resized)
            return as_response(
                {
                    "frame_id": row["id"],
                    "asset_id": row["asset_id"],
                    "frame_num": row["frame_index"],
                    "payload_kind": payload_kind,
                    "dtype": str(matrix.dtype),
                    "shape": list(matrix.shape),
                    "scale": None if width is not None or height is not None else scale,
                    "requested_width": width,
                    "requested_height": height,
                    "data": matrix.tolist(),
                }
            )
        if requested == "preview":
            delivered = preview_image(resized, preview_max_dim)
            payload, media_type = encode_image(delivered, "png")
            extension = "png"
            headers = {
                "X-Pelagia-Preview": "true",
                "X-Pelagia-Preview-Max-Dim": str(preview_max_dim),
                **size_headers,
            }
        else:
            delivered = resized
            payload, media_type = encode_image(delivered, requested)
            extension = image_extension(requested)
            headers = size_headers
        delivered_height, delivered_width = np.asarray(delivered).shape[:2]

        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f'inline; filename="{row["id"]}_{payload_kind}.{extension}"'
                ),
                "X-Pelagia-Frame-Id": str(row["id"]),
                "X-Pelagia-Payload-Kind": payload_kind,
                "X-Pelagia-Source-Width": str(source_width),
                "X-Pelagia-Source-Height": str(source_height),
                "X-Pelagia-Image-Width": str(delivered_width),
                "X-Pelagia-Image-Height": str(delivered_height),
                "X-Pelagia-Scale-X": str(delivered_width / source_width),
                "X-Pelagia-Scale-Y": str(delivered_height / source_height),
                **headers,
            },
        )

    def _frame_image_url(
        *,
        path: str,
        frame_id: str,
        width: int | None,
        height: int | None,
        scale: float,
    ) -> str:
        params: dict[str, str | int | float] = {
            "frame_id": frame_id,
            "format": "jpg",
        }
        if width is not None:
            params["width"] = width
        elif height is not None:
            params["height"] = height
        else:
            params["scale"] = scale
        return f"{path}?{urlencode(params)}"

    def _has_preprocessed_payload(row: dict) -> bool:
        return bool(row.get("preprocessed_payload_ref") or row.get("preprocessed_kvstore_hash"))

    def _resolve_background_target(request: Request, body: FrameBackgroundRequest) -> tuple[str | None, str | None, list[str]]:
        repository = get_repository(request)
        frame_ids = list(body.frame_ids or [])
        if body.frame_id is not None:
            frame_ids.append(body.frame_id)
        frame_ids = list(dict.fromkeys(str(frame_id) for frame_id in frame_ids))

        run_id = body.run_id
        asset_id = body.asset_id
        if not frame_ids:
            if asset_id is None:
                raise HTTPException(
                    status_code=422,
                    detail="Background generation requires asset_id, frame_id, or frame_ids.",
                )
            frames = repository.list_frames(
                asset_id,
                project_id=scoped_project_id(request),
                start_frame=body.start_frame,
                end_frame=body.end_frame,
                limit=body.limit,
            )
            frame_ids = [str(frame["id"]) for frame in frames]

        if not frame_ids:
            return run_id, asset_id, []

        for frame_id in frame_ids:
            frame_record = repository.get_frame_record(frame_id, project_id=scoped_project_id(request))
            if frame_record is None:
                raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")
            if asset_id is None:
                asset_id = frame_record.asset_id
            elif frame_record.asset_id != asset_id:
                raise HTTPException(
                    status_code=422,
                    detail="Background generation may only process frames from one asset.",
                )
            if run_id is None:
                run_id = frame_record.run_id
        return run_id, asset_id, frame_ids

    @frames_router.get("/processing-state")
    def list_frame_processing_state(
        request: Request,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        preprocessing_state: str | None = None,
        detection_state: str | None = None,
        refinement_state: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        sort_by: Literal["asset_frame", "frame", "captured_at", "filename", "roi_count", "refined_count"] = "asset_frame",
        sort_dir: Literal["asc", "desc"] = "asc",
        limit: int = 1000,
        offset: int = 0,
    ) -> dict:
        stats = get_repository(request).list_frame_processing_state(
            project_id=scoped_project_id(request),
            run_id=run_id,
            asset_id=asset_id,
            collection=collection,
            kind=kind,
            filename=filename,
            preprocessing_state=preprocessing_state,
            detection_state=detection_state,
            refinement_state=refinement_state,
            start_frame=start_frame,
            end_frame=end_frame,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
        )
        return as_response(
            {
                **stats,
                "page": page_metadata(limit=limit, offset=offset, count=len(stats.get("frames", []))),
            }
        )

    @frames_router.get("/{frame_id}/context", response_model=FrameContextResponse)
    def get_frame_context(
        request: Request,
        frame_id: str,
        width: int | None = None,
        height: int | None = None,
        scale: float = 1.0,
        include_detections: bool = True,
        detection_limit: int = 500,
        detection_offset: int = 0,
        frame_payload_kind: Literal["original", "preprocessed"] = "preprocessed",
    ) -> dict:
        repository = get_repository(request)
        project_id = scoped_project_id(request)
        row = repository.get_frame(frame_id, project_id=project_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")

        asset = repository.get_asset(row["asset_id"], project_id=project_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"Asset {row['asset_id']!r} was not found.")

        detections = []
        if include_detections:
            detections = repository.list_detections(
                row["asset_id"],
                project_id=project_id,
                frame_id=frame_id,
                limit=detection_limit,
                offset=detection_offset,
            )
        detection_summaries = [detection_summary(detection) for detection in detections]
        image_urls = {
            "original": _frame_image_url(
                path="/frame/original",
                frame_id=frame_id,
                width=width,
                height=height,
                scale=scale,
            ),
            "preprocessed": (
                _frame_image_url(
                    path="/frame/preprocessed",
                    frame_id=frame_id,
                    width=width,
                    height=height,
                    scale=scale,
                )
                if _has_preprocessed_payload(row)
                else None
            ),
        }
        return as_response(
            {
                "frame": frame_summary(row),
                "asset": asset,
                "image_urls": image_urls,
                "frame_payload_kind": (
                    frame_payload_kind if frame_payload_kind == "original" or image_urls["preprocessed"] else "original"
                ),
                "detections": detection_summaries,
                "detection_count": len(detection_summaries),
                "page": page_metadata(
                    limit=detection_limit,
                    offset=detection_offset,
                    count=len(detection_summaries),
                ),
            }
        )

    @router.head("/original")
    @router.get("/original")
    @frames_router.head("/original")
    @frames_router.get("/original")
    def get_original_frame(
        request: Request,
        frame_id: str | None = None,
        asset_id: str | None = None,
        frame_num: int | None = None,
        format: str = "png",
        preview_max_dim: int = 128,
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
    ):
        row = _resolve_frame_row(request, frame_id, asset_id, frame_num)
        return _frame_data_response(
            request=request,
            row=row,
            payload_kind="original",
            format=format,
            preview_max_dim=preview_max_dim,
            scale=scale,
            width=width,
            height=height,
        )

    @router.head("/preprocessed")
    @router.get("/preprocessed")
    @frames_router.head("/preprocessed")
    @frames_router.get("/preprocessed")
    @frames_router.head("/preprocess")
    @frames_router.get("/preprocess")
    def get_preprocessed_frame(
        request: Request,
        frame_id: str | None = None,
        asset_id: str | None = None,
        frame_num: int | None = None,
        format: str = "png",
        preview_max_dim: int = 128,
        scale: float = 1.0,
        width: int | None = None,
        height: int | None = None,
    ):
        row = _resolve_frame_row(request, frame_id, asset_id, frame_num)
        return _frame_data_response(
            request=request,
            row=row,
            payload_kind="preprocessed",
            format=format,
            preview_max_dim=preview_max_dim,
            scale=scale,
            width=width,
            height=height,
        )

    def _preprocess_resolved_frame(request: Request, row: dict, body: FramePreprocessRequest) -> dict:
        project_id = scoped_project_id(request)
        context = get_context(request).for_project(project_id)
        repository = get_repository(request)
        source_frame = retrieve_frame(str(row["id"]), context=context, payload_kind="original")
        processed = preprocess_frame_for_segmentation(
            source_frame,
            flatfield_correction=body.flatfield_correction,
            flatfield_q=body.flatfield_q,
            flatfield_axis=body.flatfield_axis,
            flatfield_min_field_value=body.flatfield_min_field_value,
            flatfield_max_field_value=body.flatfield_max_field_value,
            apply_mask=body.apply_mask,
            crop_enabled=body.crop_enabled,
            crop_x=body.crop_x,
            crop_y=body.crop_y,
            crop_w=body.crop_w,
            crop_h=body.crop_h,
            background_correction=body.background_correction,
            background_min_field_value=body.background_min_field_value,
            background_max_field_value=body.background_max_field_value,
            invert_intensity=body.invert_intensity,
            context=context,
        )
        array = processed.read()
        if array is None:
            raise HTTPException(status_code=500, detail="Preprocessing produced no frame data.")

        stored_row = None
        if body.store:
            stored_row = store_preprocessed_frame(
                str(row["id"]),
                processed,
                context=context,
                encoding=body.encoding,
                quality=body.quality,
            )

        response = {
            "frame_id": row["id"],
            "asset_id": row["asset_id"],
            "frame_num": row["frame_index"],
            "stored": body.store,
            "dtype": str(np.asarray(array).dtype),
            "shape": list(np.asarray(array).shape),
            "preprocessing": processed.metadata,
        }
        if stored_row is not None:
            mark_frame_stage_status(
                repository,
                project_id=project_id,
                frame_ids=[str(row["id"])],
                stage=PipelineStage.PREPROCESS_FRAMES.value,
                status="succeeded",
            )
            touch_processing_status_snapshot(repository, project_id=project_id)
            response["frame"] = frame_summary(stored_row)
        if body.response_format == "matrix":
            response["data"] = np.asarray(array).tolist()
        return response

    @router.post("/preprocess")
    def preprocess_frame(request: Request, body: FramePreprocessRequest) -> dict:
        repository = get_repository(request)
        requested_frame_ids = list(body.frame_ids or [])
        if body.frame_id is not None:
            requested_frame_ids.append(body.frame_id)
        requested_frame_ids = list(dict.fromkeys(requested_frame_ids))

        if requested_frame_ids:
            if body.asset_id is not None or body.frame_num is not None:
                raise HTTPException(
                    status_code=422,
                    detail="Provide frame_ids/frame_id, or provide asset_id and frame_num, not both.",
                )
            if body.response_format == "matrix" and len(requested_frame_ids) > 1:
                raise HTTPException(
                    status_code=422,
                    detail="response_format='matrix' is only supported for single-frame preprocessing.",
                )
            frames = [
                _preprocess_resolved_frame(
                    request,
                    _resolve_frame_row(request, frame_id, None, None),
                    body,
                )
                for frame_id in requested_frame_ids
            ]
            return as_response(
                {
                    "frame_count": len(frames),
                    "frame_ids": [str(frame["frame_id"]) for frame in frames],
                    "stored": body.store,
                    "frames": frames,
                }
            )

        if body.asset_id is not None and body.frame_num is None:
            rows = repository.list_frames(
                body.asset_id,
                project_id=scoped_project_id(request),
                start_frame=body.start_frame,
                end_frame=body.end_frame,
                limit=body.limit,
            )
            if body.response_format == "matrix" and len(rows) > 1:
                raise HTTPException(
                    status_code=422,
                    detail="response_format='matrix' is only supported for single-frame preprocessing.",
                )
            frames = [_preprocess_resolved_frame(request, row, body) for row in rows]
            return as_response(
                {
                    "frame_count": len(frames),
                    "frame_ids": [str(frame["frame_id"]) for frame in frames],
                    "asset_id": body.asset_id,
                    "stored": body.store,
                    "frames": frames,
                }
            )

        row = _resolve_frame_row(request, body.frame_id, body.asset_id, body.frame_num)
        return as_response(_preprocess_resolved_frame(request, row, body))

    @router.post("/preprocess/jobs")
    def queue_frame_preprocess_job(request: Request, body: QueueFramePreprocessRequest) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        processing_defaults = get_context(request).config.processing
        flatfield_defaults = processing_defaults.flatfield
        preprocessing_defaults = processing_defaults.preprocessing
        frame_ids = list(body.frame_ids or [])
        if body.frame_id is not None:
            frame_ids.append(body.frame_id)

        run_id = body.run_id
        asset_id = body.asset_id
        if asset_id is None and frame_ids:
            first_frame = repository.get_frame_record(frame_ids[0], project_id=auth.project_id)
            if first_frame is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Frame {frame_ids[0]!r} was not found.",
                )
            run_id = run_id or first_frame.run_id
            asset_id = first_frame.asset_id

        if asset_id is None:
            raise HTTPException(
                status_code=422,
                detail="Preprocess jobs require asset_id or at least one frame_id.",
            )

        if run_id is None:
            asset = repository.get_asset(asset_id, project_id=auth.project_id)
            if asset is None:
                raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} was not found.")
            run_id = asset.get("run_id")

        payload = {
            "frame_ids": frame_ids,
            "start_frame": body.start_frame,
            "end_frame": body.end_frame,
            "limit": body.limit,
            "flatfield_correction": (
                flatfield_defaults.flatfield_correction
                if body.flatfield_correction is None
                else body.flatfield_correction
            ),
            "flatfield_q": flatfield_defaults.flatfield_q if body.flatfield_q is None else body.flatfield_q,
            "flatfield_axis": flatfield_defaults.flatfield_axis if body.flatfield_axis is None else body.flatfield_axis,
            "flatfield_min_field_value": (
                flatfield_defaults.flatfield_min_field_value
                if body.flatfield_min_field_value is None
                else body.flatfield_min_field_value
            ),
            "flatfield_max_field_value": (
                flatfield_defaults.flatfield_max_field_value
                if body.flatfield_max_field_value is None
                else body.flatfield_max_field_value
            ),
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
            "background_min_field_value": (
                preprocessing_defaults.background_min_field_value
                if body.background_min_field_value is None
                else body.background_min_field_value
            ),
            "background_max_field_value": (
                preprocessing_defaults.background_max_field_value
                if body.background_max_field_value is None
                else body.background_max_field_value
            ),
            "invert_intensity": (
                preprocessing_defaults.invert_intensity
                if body.invert_intensity is None
                else body.invert_intensity
            ),
            "encoding": body.encoding,
            "quality": body.quality,
        }
        payload = PreprocessFramesCommand.from_payload(payload).to_payload()
        try:
            job = repository.create_job(
                PipelineStage.PREPROCESS_FRAMES,
                project_id=auth.project_id,
                run_id=run_id,
                asset_id=asset_id,
                priority=body.priority,
                payload=payload,
                depends_on=body.depends_on or [],
                summary=f"preprocess queued for asset {asset_id}",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"job": as_response(job)}

    @router.post("/background")
    def generate_frame_background(request: Request, body: FrameBackgroundRequest) -> dict:
        run_id, asset_id, frame_ids = _resolve_background_target(request, body)
        if not frame_ids:
            return as_response(
                {
                    "stage": PipelineStage.BACKGROUND_FRAMES.value,
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "frame_count": 0,
                    "frame_ids": [],
                }
            )
        try:
            result = generate_background_for_frames(
                frame_ids,
                context=get_context(request).for_project(scoped_project_id(request)),
                payload_kind=body.payload_kind,
                encoding=body.encoding,
                quality=body.quality,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        result.update(
            {
                "stage": PipelineStage.BACKGROUND_FRAMES.value,
                "run_id": run_id,
                "asset_id": asset_id,
            }
        )
        return as_response(result)

    @router.post("/background/jobs")
    def queue_frame_background_job(request: Request, body: QueueFrameBackgroundRequest) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        run_id, asset_id, frame_ids = _resolve_background_target(request, body)
        payload = {
            "frame_ids": frame_ids,
            "start_frame": body.start_frame,
            "end_frame": body.end_frame,
            "limit": body.limit,
            "payload_kind": body.payload_kind,
            "encoding": body.encoding,
            "quality": body.quality,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        payload = FrameBackgroundCommand.from_payload(payload).to_payload()
        if body.dry_run:
            return as_response(
                {
                    "dry_run": True,
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "priority": body.priority,
                    "depends_on": body.depends_on or [],
                    "payload": payload,
                }
            )
        if asset_id is None:
            raise HTTPException(status_code=422, detail="Background jobs require an asset_id or resolvable frame_ids.")
        try:
            job = repository.create_job(
                PipelineStage.BACKGROUND_FRAMES,
                project_id=auth.project_id,
                run_id=run_id,
                asset_id=asset_id,
                priority=body.priority,
                payload=payload,
                depends_on=body.depends_on or [],
                summary=f"background queued for asset {asset_id}",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"job": as_response(job)}
else:
    router = None
    frames_router = None
    routers = []
