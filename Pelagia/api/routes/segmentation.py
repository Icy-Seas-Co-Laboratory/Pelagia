from __future__ import annotations

from typing import Any, Literal

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_write, scoped_project_id
    from ..schemas import OptionsResponse
    from ...domain import PipelineStage
    from ...processing.frame_store import retrieve_frame
    from ...processing.detection_candidate import segment_frame
    from ...processing.segmentation_options import (
        MASK_AUGMENTATION_STEPS,
        ROI_ASSEMBLY_METHODS,
        ROI_ENCODINGS,
        THRESHOLD_METHODS,
        flatten_segmentation_options,
        resolve_segmentation_options,
        segmentation_capabilities,
        segment_frame_kwargs,
    )
    from ._common import (
        as_response,
        detection_summary,
        get_context,
        get_repository,
        mark_frame_stage_status,
        refresh_frame_status_counts,
        touch_processing_status_snapshot,
    )

    ThresholdMethod = Literal[
        "manual",
        "otsu",
        "bounded_otsu",
        "bounded_otsu_canny",
        "canny",
        "adaptive_mean",
        "adaptive_gaussian",
        "percentile_background",
        "hysteresis",
        "sobel_edges",
        "auto",
    ]
    MaskAugmentationStep = Literal[
        "none",
        "dilate",
        "erode",
        "open",
        "close",
        "fill_holes",
        "remove_small_components",
        "clear_border",
    ]
    RoiAssemblyMethod = Literal["connected_components", "contours"]
    RoiEncoding = Literal["png", "jpg", "jxl", "raw", "zstd", "auto"]

    class SegmentFrameRequest(BaseModel):
        threshold: int | float | None = None
        threshold_method: ThresholdMethod | None = None
        manual_threshold: int | float | None = None
        thresholding_maximum_value: int | float | None = None
        bounded_otsu_min_contrast: int | float | None = None
        bounded_otsu_max_foreground_fraction: float | None = None
        canny_enabled: bool | None = None
        canny_low_threshold: int | float | None = None
        canny_high_threshold: int | float | None = None
        canny_blur_kernel: int | None = None
        dilate_kernel_w: int | None = None
        dilate_kernel_h: int | None = None
        dilate_iterations: int | None = None
        erode_kernel_w: int | None = None
        erode_kernel_h: int | None = None
        erode_iterations: int | None = None
        open_kernel_w: int | None = None
        open_kernel_h: int | None = None
        open_iterations: int | None = None
        close_kernel_w: int | None = None
        close_kernel_h: int | None = None
        close_iterations: int | None = None
        fill_holes: bool | None = None
        remove_small_components: bool | None = None
        min_component_area: int | float | None = None
        clear_border: bool | None = None
        adaptive_block_size: int | None = None
        adaptive_c: int | float | None = None
        percentile_background_percentile: int | float | None = None
        percentile_min_contrast: int | float | None = None
        hysteresis_low_threshold: int | float | None = None
        hysteresis_high_threshold: int | float | None = None
        hysteresis_connectivity: int | None = None
        sobel_percentile: int | float | None = None
        sobel_threshold: int | float | None = None
        sobel_kernel_size: int | None = None
        frame_payload_kind: Literal["original", "raw", "preprocessed", "processed", "corrected"] = "original"
        apply_preprocessing: bool | None = None
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
        mask_augmentation_enabled: bool | None = None
        mask_augmentation_steps: list[MaskAugmentationStep] | None = None
        roi_assembly_method: RoiAssemblyMethod | None = None
        roi_assembly_connectivity: int | None = None
        min_area: int | float | None = None
        max_area: int | float | None = None
        min_perimeter: int | float | None = None
        max_perimeter: int | float | None = None
        min_width: int | float | None = None
        max_width: int | float | None = None
        min_height: int | float | None = None
        max_height: int | float | None = None
        min_width_plus_height: int | float | None = None
        max_width_plus_height: int | float | None = None
        padding: int | None = None
        roi_encoding: RoiEncoding | None = None
        zstd_min_bytes: int | None = None
        store_roi_payload_min_area: int | float | None = None
        store_roi_payload_min_width: int | float | None = None
        store_roi_payload_min_height: int | float | None = None
        store_roi_payload_min_width_plus_height: int | float | None = None
        always_store_mask: bool | None = None

    class QueueSegmentationRequest(SegmentFrameRequest):
        run_id: str | None = None
        asset_id: str | None = None
        frame_ids: list[str] | None = None
        start_frame: int | None = None
        end_frame: int | None = None
        limit: int | None = None
        priority: int | None = None
        depends_on: list[str] | None = None
        dry_run: bool = False

    router = APIRouter(prefix="/segmentation", tags=["segmentation"])

    def _model_dict(model: BaseModel) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def _segmentation_overrides(body: SegmentFrameRequest | QueueSegmentationRequest) -> dict[str, Any]:
        ignored = {
            "run_id",
            "asset_id",
            "frame_ids",
            "start_frame",
            "end_frame",
            "limit",
            "priority",
            "depends_on",
            "dry_run",
        }
        return {key: value for key, value in _model_dict(body).items() if key not in ignored}

    def _resolve_options(request: Request, body: SegmentFrameRequest | QueueSegmentationRequest) -> dict[str, dict[str, Any]]:
        try:
            return resolve_segmentation_options(
                _segmentation_overrides(body),
                get_context(request).config.processing,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _has_preprocessed_payload(row: dict[str, Any]) -> bool:
        return bool(row.get("preprocessed_payload_ref") or row.get("preprocessed_kvstore_hash"))

    def _requested_segmentation_frames(
        repository,
        body: QueueSegmentationRequest,
        *,
        asset_id: str | None,
        project_id: str,
    ) -> list[dict[str, Any]]:
        if body.frame_ids:
            frames = []
            for frame_id in dict.fromkeys(str(frame_id) for frame_id in body.frame_ids):
                frame = repository.get_frame(frame_id, project_id=project_id)
                if frame is None:
                    raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")
                frames.append(frame)
            return frames
        if asset_id is None:
            return []
        return repository.list_frames(
            asset_id,
            project_id=project_id,
            start_frame=body.start_frame,
            end_frame=body.end_frame,
            limit=body.limit,
        )

    def _validate_preprocessed_source(
        repository,
        body: QueueSegmentationRequest,
        *,
        asset_id: str | None,
        project_id: str,
        frame_payload_kind: str,
    ) -> None:
        if frame_payload_kind not in {"preprocessed", "processed", "corrected"}:
            return
        frames = _requested_segmentation_frames(
            repository,
            body,
            asset_id=asset_id,
            project_id=project_id,
        )
        missing = [str(frame["id"]) for frame in frames if not _has_preprocessed_payload(frame)]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Segmentation requested {frame_payload_kind!r} frame payloads, "
                    f"but {len(missing)} selected frame(s) lack preprocessed payloads: {preview}{suffix}."
                ),
            )

    def _stage_metadata(detections: list[Any], *, fallback_detection_count: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if detections:
            first = detections[0]
            if isinstance(first, dict):
                metadata = first.get("metadata") or {}
            else:
                metadata = getattr(first, "metadata", {}) or {}
        counts = dict(metadata.get("stage_counts") or {})
        counts.setdefault("recorded_detection_count", fallback_detection_count)
        return {
            "bbox_coordinate_space": metadata.get("bbox_coordinate_space"),
            "processed_frame_shape": metadata.get("processed_frame_shape"),
            "stage_counts": counts,
            "candidate_limit": metadata.get("candidate_limit"),
            "candidate_limit_applied": metadata.get("candidate_limit_applied"),
            "stage_durations_ms": dict(metadata.get("stage_durations_ms") or {}),
        }

    @router.get("/options", response_model=OptionsResponse)
    def get_segmentation_options(request: Request) -> dict:
        return as_response(segmentation_capabilities(get_context(request).config.processing))

    @router.post("/validate")
    def validate_segmentation_request(request: Request, body: QueueSegmentationRequest) -> dict:
        resolved_options = _resolve_options(request, body)
        payload = {
            "frame_ids": body.frame_ids or [],
            "start_frame": body.start_frame,
            "end_frame": body.end_frame,
            "limit": body.limit,
            **flatten_segmentation_options(resolved_options),
        }
        return as_response(
            {
                "valid": True,
                "dry_run": True,
                "run_id": body.run_id,
                "asset_id": body.asset_id,
                "depends_on": body.depends_on or [],
                "priority": body.priority,
                "resolved_options": resolved_options,
                "payload": payload,
                "supported": {
                    "threshold_methods": THRESHOLD_METHODS,
                    "mask_augmentation_steps": MASK_AUGMENTATION_STEPS,
                    "roi_assembly_methods": ROI_ASSEMBLY_METHODS,
                    "roi_encoding_options": ROI_ENCODINGS,
                },
            }
        )

    def _segment_resolved_frame(request: Request, frame_id: str, body: SegmentFrameRequest) -> dict:
        repository = get_repository(request)
        project_id = scoped_project_id(request)
        context = get_context(request).for_project(project_id)
        frame_record = repository.get_frame_record(frame_id, project_id=project_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")
        if not frame_record.run_id:
            raise HTTPException(
                status_code=409,
                detail=f"Frame {frame_id!r} does not include a run_id.",
            )

        try:
            resolved_options = _resolve_options(request, body)
            flat_options = flatten_segmentation_options(resolved_options)
            frame = retrieve_frame(
                frame_id,
                context=context,
                payload_kind=flat_options["frame_payload_kind"],
            )
            detections = segment_frame(
                frame,
                frame_record=frame_record,
                **segment_frame_kwargs(resolved_options),
                context=context,
            )
            inserted = repository.replace_frame_detections(
                frame_record.run_id,
                [frame_id],
                detections,
                project_id=project_id,
            )
            mark_frame_stage_status(
                repository,
                project_id=project_id,
                frame_ids=[frame_id],
                stage=PipelineStage.SEGMENT.value,
                status="succeeded",
            )
            refresh_frame_status_counts(
                repository,
                project_id=project_id,
                frame_ids=[frame_id],
                asset_id=frame_record.asset_id,
            )
            touch_processing_status_snapshot(repository, project_id=project_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "frame_id": frame_id,
            "run_id": frame_record.run_id,
            "asset_id": frame_record.asset_id,
            "frame_payload_kind": flat_options["frame_payload_kind"],
            "apply_preprocessing": flat_options["apply_preprocessing"],
            "resolved_options": resolved_options,
            **_stage_metadata(detections, fallback_detection_count=len(inserted)),
            "detection_count": len(inserted),
            "detections": [detection_summary(row) for row in inserted],
        }

    @router.post("/frames/{frame_id}")
    def segment_stored_frame(request: Request, frame_id: str, body: SegmentFrameRequest) -> dict:
        return as_response(_segment_resolved_frame(request, frame_id, body))

    @router.post("/frames")
    def segment_stored_frames(request: Request, body: QueueSegmentationRequest) -> dict:
        repository = get_repository(request)
        frame_ids = list(dict.fromkeys(str(frame_id) for frame_id in (body.frame_ids or [])))

        if not frame_ids:
            if body.asset_id is None:
                raise HTTPException(
                    status_code=422,
                    detail="Provide frame_ids or asset_id for batch segmentation.",
                )
            frames = repository.list_frames(
                body.asset_id,
                project_id=scoped_project_id(request),
                start_frame=body.start_frame,
                end_frame=body.end_frame,
                limit=body.limit,
            )
            frame_ids = [str(frame["id"]) for frame in frames]

        frames = [_segment_resolved_frame(request, frame_id, body) for frame_id in frame_ids]
        return as_response(
            {
                "frame_count": len(frames),
                "detection_count": sum(int(frame["detection_count"]) for frame in frames),
                "frame_ids": frame_ids,
                "frames": frames,
            }
        )

    @router.post("/jobs")
    def queue_segmentation_job(request: Request, body: QueueSegmentationRequest) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        resolved_options = _resolve_options(request, body)
        flat_options = flatten_segmentation_options(resolved_options)
        run_id = body.run_id
        asset_id = body.asset_id

        if asset_id is None and body.frame_ids:
            first_frame = repository.get_frame_record(body.frame_ids[0], project_id=auth.project_id)
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
            asset = repository.get_asset(asset_id, project_id=auth.project_id)
            if asset is None:
                raise HTTPException(status_code=404, detail=f"Asset {asset_id!r} was not found.")
            run_id = asset.get("run_id")

        _validate_preprocessed_source(
            repository,
            body,
            asset_id=asset_id,
            project_id=auth.project_id,
            frame_payload_kind=flat_options["frame_payload_kind"],
        )

        payload = {
            "frame_ids": body.frame_ids or [],
            "start_frame": body.start_frame,
            "end_frame": body.end_frame,
            "limit": body.limit,
            **flat_options,
        }
        if body.dry_run:
            return as_response(
                {
                    "dry_run": True,
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "priority": body.priority,
                    "depends_on": body.depends_on or [],
                    "resolved_options": resolved_options,
                    "payload": payload,
                }
            )
        try:
            job = repository.create_job(
                "segment",
                project_id=auth.project_id,
                run_id=run_id,
                asset_id=asset_id,
                priority=body.priority,
                payload=payload,
                depends_on=body.depends_on or [],
                summary=f"segment queued for asset {asset_id}",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"job": as_response(job)}
else:
    router = None
