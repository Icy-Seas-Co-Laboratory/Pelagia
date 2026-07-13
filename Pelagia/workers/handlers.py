from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..domain import JobStatus, PipelineStage
from ..domain import DetectionRecord
from ..domain import normalize_collections
from ..processing import ingest as ingest_module
from ..processing.detection_candidate import segment_frame
from ..processing.detection_refinement import (
    RoiRefinementOptions,
    refine_detections,
    refined_storage_candidate_detection_id,
)
from ..processing.frame_correction import generate_background_for_frames
from ..processing.frame_preprocess import preprocess_frame_for_segmentation
from ..processing.frame_store import retrieve_frame, store_preprocessed_frame
from ..processing.oracle_unet_refiner import resolve_refinement_model
from ..processing.segmentation_options import resolve_segmentation_options, segment_frame_kwargs
from ..services.context import AppContext
from ..services.project_settings import resolve_project_storage_settings
from ..services.job_commands import (
    ExtractFramesCommand,
    FrameBackgroundCommand,
    FrameSelection,
    PreprocessFramesCommand,
    RoiRefinementCommand,
    SegmentFramesCommand,
)
from .progress import JobProgressReporter


JobHandler = Callable[[dict[str, Any], AppContext], dict[str, Any]]


class HandlerRegistry:
    """Maps pipeline stages to processing functions."""

    def __init__(self) -> None:
        self._handlers: dict[PipelineStage, JobHandler] = {}

    def register(self, stage: PipelineStage, handler: JobHandler) -> None:
        """Register a callable for a pipeline stage."""
        self._handlers[stage] = handler

    def handle(self, job: dict[str, Any], context: AppContext) -> dict[str, Any]:
        """Dispatch a leased job to its registered handler."""
        stage = PipelineStage(job["stage"])
        if stage not in self._handlers:
            raise KeyError(f"No worker handler registered for stage {stage.value!r}.")
        return self._handlers[stage](job, context)


def _job_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("Job payload must be a JSON object.")
    return payload


def _job_identifier(job: dict[str, Any], key: str, payload: dict[str, Any]) -> str:
    value = job.get(key) or payload.get(key)
    if not value:
        raise ValueError(f"{job.get('stage', 'Worker')} job requires {key}.")
    return str(value)


def _payload_frame_ids(payload: dict[str, Any]) -> list[str]:
    frame_ids = payload.get("frame_ids")
    if frame_ids:
        return [str(frame_id) for frame_id in frame_ids]
    frame_id = payload.get("frame_id")
    if frame_id:
        return [str(frame_id)]
    return []


def _job_project_id(job: dict[str, Any], context: AppContext) -> str | None:
    return None if job.get("project_id") is None else str(job.get("project_id"))


def _status_stage(stage: PipelineStage | str) -> str | None:
    value = stage.value if isinstance(stage, PipelineStage) else str(stage)
    if value in {
        PipelineStage.PREPROCESS_FRAMES.value,
        PipelineStage.SEGMENT.value,
        PipelineStage.ROI_REFINEMENT.value,
    }:
        return value
    return None


def _mark_frame_stage_status(
    context: AppContext,
    *,
    project_id: str | None,
    frame_ids: list[str],
    stage: PipelineStage | str,
    status: JobStatus | str,
    job_id: str | None = None,
) -> None:
    if not project_id or not frame_ids or context.repository is None:
        return
    stage_value = _status_stage(stage)
    if stage_value is None:
        return
    updater = getattr(context.repository, "upsert_frame_stage_status", None)
    if not callable(updater):
        return
    status_value = status.value if isinstance(status, JobStatus) else str(status)
    updater(
        project_id=project_id,
        frame_ids=frame_ids,
        stage=stage_value,
        status=status_value,
        job_id=job_id,
    )


def _ensure_frame_status_rows(
    context: AppContext,
    *,
    project_id: str | None,
    frame_ids: list[str],
    asset_id: str | None = None,
) -> None:
    if not project_id or context.repository is None:
        return
    ensure_rows = getattr(context.repository, "ensure_frame_status_rows", None)
    if callable(ensure_rows):
        ensure_rows(project_id=project_id, frame_ids=frame_ids, asset_id=asset_id)


def _refresh_frame_status_counts(
    context: AppContext,
    *,
    project_id: str | None,
    frame_ids: list[str],
    asset_id: str | None = None,
) -> None:
    if not project_id or context.repository is None:
        return
    refresh_counts = getattr(context.repository, "refresh_frame_status_counts", None)
    if callable(refresh_counts):
        refresh_counts(project_id=project_id, frame_ids=frame_ids, asset_id=asset_id)


def _touch_processing_status_snapshot(context: AppContext, *, project_id: str | None) -> None:
    if not project_id or context.repository is None:
        return
    touch_snapshot = getattr(context.repository, "touch_processing_status_snapshot", None)
    if callable(touch_snapshot):
        touch_snapshot(project_id=project_id)


def mark_job_frame_stage_failed(job: dict[str, Any], context: AppContext) -> None:
    try:
        payload = _job_payload(job)
        project_id = _job_project_id(job, context)
        _mark_frame_stage_status(
            context,
            project_id=project_id,
            frame_ids=_payload_frame_ids(payload),
            stage=str(job.get("stage")),
            status=JobStatus.FAILED,
            job_id=None if job.get("id") is None else str(job.get("id")),
        )
        _touch_processing_status_snapshot(context, project_id=project_id)
    except Exception:
        return


def extract_frames_handler(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    """Worker handler for extracting and storing frames from a registered asset."""
    if context.repository is None:
        raise RuntimeError("Extract frames handler requires a PostgresRepository.")

    command = ExtractFramesCommand.from_payload(_job_payload(job))
    payload = command.to_payload()
    project_id = _job_project_id(job, context)
    run_id = _job_identifier(job, "run_id", payload)
    asset_id = _job_identifier(job, "asset_id", payload)

    asset = context.repository.get_asset(asset_id, project_id=project_id)
    if asset is None:
        raise KeyError(f"Raw asset {asset_id!r} was not found.")

    source_path = payload.get("source_path") or asset.get("path")
    if not source_path:
        raise ValueError(f"Raw asset {asset_id!r} does not include a source path.")

    metadata = {**dict(asset.get("metadata") or {}), **dict(payload.get("metadata") or {})}
    collections = normalize_collections(payload.get("collections") or asset.get("collections"))
    metadata.setdefault("collections", collections)
    metadata.setdefault("worker_job_id", str(job.get("id")))
    metadata.setdefault("worker_stage", PipelineStage.EXTRACT_FRAMES.value)
    ingest_defaults = context.config.processing.video_ingest
    preprocessing_defaults = context.config.processing.preprocessing
    progress = JobProgressReporter(
        job,
        context,
        stage=PipelineStage.EXTRACT_FRAMES.value,
        unit="frames",
        total=0,
    )
    progress.start(f"Extracting frames from {Path(str(source_path)).name}")
    ingest_progress_state = {
        "source_frame_count": 0,
        "source_frames_read": 0,
        "stored_frame_count": 0,
        "stored_tile_count": 0,
    }

    def _progress_dict(values: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}

    def report_ingest_progress(update: dict[str, Any]) -> None:
        source_frame_count = int(update.get("source_frame_count") or 0)
        if source_frame_count > 0:
            ingest_progress_state["source_frame_count"] = max(
                int(ingest_progress_state["source_frame_count"]),
                source_frame_count,
            )
            progress.total = int(ingest_progress_state["source_frame_count"])

        source_frames_read = int(
            update.get("source_frames_read")
            or update.get("stored_frame_count")
            or update.get("stored_tile_count")
            or 0
        )
        ingest_progress_state["source_frames_read"] = max(
            int(ingest_progress_state["source_frames_read"]),
            source_frames_read,
        )
        ingest_progress_state["stored_frame_count"] = max(
            int(ingest_progress_state["stored_frame_count"]),
            int(update.get("stored_frame_count") or 0),
        )
        ingest_progress_state["stored_tile_count"] = max(
            int(ingest_progress_state["stored_tile_count"]),
            int(update.get("stored_tile_count") or 0),
        )

        event = str(update.get("event") or "ingesting")
        if event == "started":
            message = f"Starting extraction from {Path(str(source_path)).name}"
        elif event == "video_opened":
            message = f"Opened video {Path(str(source_path)).name}"
        elif event == "completed":
            message = f"Completed extraction from {Path(str(source_path)).name}"
        elif event == "failed":
            message = f"Extraction failed for {Path(str(source_path)).name}"
        else:
            message = f"Extracting frames from {Path(str(source_path)).name}"

        progress.update(
            source_frames_read,
            current=_progress_dict(
                {
                    "event": event,
                    "filename": update.get("filename"),
                    "frame_index": update.get("frame_index"),
                    "tile_number": update.get("tile_number"),
                    "source_frame_start": update.get("source_frame_start"),
                    "source_frame_end": update.get("source_frame_end"),
                    "partial_tile": update.get("partial_tile"),
                }
            ),
            secondary=_progress_dict(
                {
                    "stored_frame_count": update.get("stored_frame_count"),
                    "stored_tile_count": update.get("stored_tile_count"),
                    "n_tile": update.get("n_tile"),
                    "estimated_tile_count": update.get("estimated_tile_count"),
                    "fps": update.get("fps"),
                }
            ),
            message=message,
            force=event in {"started", "video_opened", "completed", "failed"},
        )

    source_is_folder = Path(str(source_path)).expanduser().is_dir()
    asset_kind = str(asset.get("kind") or payload.get("kind") or "").lower()
    if source_is_folder or asset_kind == "image_sequence":
        frame_rows = ingest_module.ingest_image_folder(
            source_path,
            recursive=bool(payload.get("recursive", False)),
            context=context,
            run_id=run_id,
            asset_id=asset_id,
            metadata=metadata,
            progress_callback=report_ingest_progress,
        )
    else:
        frame_rows = ingest_module.ingest_video_file(
            source_path,
            n_tile=int(payload.get("n_tile", ingest_defaults.n_tile)),
            context=context,
            run_id=run_id,
            asset_id=asset_id,
            metadata=metadata,
            adaptive_background_subtraction=bool(
                payload.get(
                    "adaptive_background_subtraction",
                    preprocessing_defaults.adaptive_background_subtraction,
                )
            ),
            adaptive_background_period=int(
                payload.get(
                    "adaptive_background_period",
                    preprocessing_defaults.adaptive_background_period,
                )
            ),
            apply_mask=bool(
                payload.get("apply_mask", preprocessing_defaults.apply_mask)
            ),
            mask_path=payload.get("mask_path", preprocessing_defaults.mask_path),
            progress_callback=report_ingest_progress,
        )

    result: dict[str, Any] = {
        "stage": PipelineStage.EXTRACT_FRAMES.value,
        "project_id": project_id,
        "run_id": run_id,
        "asset_id": asset_id,
        "source_path": str(source_path),
        "frame_count": len(frame_rows),
        "frame_ids": [row.get("id") for row in frame_rows],
    }
    status_frame_ids = [str(row.get("id")) for row in frame_rows if row.get("id")]
    _ensure_frame_status_rows(
        context,
        project_id=project_id,
        frame_ids=status_frame_ids,
        asset_id=asset_id,
    )
    _touch_processing_status_snapshot(context, project_id=project_id)
    final_completed = max(
        len(frame_rows),
        int(ingest_progress_state["source_frames_read"]),
        int(ingest_progress_state["source_frame_count"]),
    )
    progress.total = final_completed
    progress.finish(
        completed=final_completed,
        secondary=_progress_dict(
            {
                "stored_frame_count": len(frame_rows),
                "stored_tile_count": (
                    int(ingest_progress_state["stored_tile_count"])
                    if ingest_progress_state["stored_tile_count"]
                    else None
                ),
                "source_frame_count": int(ingest_progress_state["source_frame_count"]) or final_completed,
            }
        ),
        message=f"Extracted {len(frame_rows)} frame{'s' if len(frame_rows) != 1 else ''}",
    )

    if payload.get("enqueue_segment"):
        roi_recording_defaults = context.config.processing.roi_recording
        segment_job = context.repository.create_job(
            PipelineStage.SEGMENT,
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            payload=SegmentFramesCommand(
                selection=FrameSelection(frame_ids=tuple(result["frame_ids"])),
                options={
                    "padding": payload.get("padding", roi_recording_defaults.padding),
                    "roi_encoding": payload.get("roi_encoding", roi_recording_defaults.roi_encoding),
                    "collections": collections,
                },
            ).to_payload(),
            depends_on=[str(job["id"])],
            summary=f"segment queued for {len(frame_rows)} extracted frames",
        )
        result["segment_job_id"] = segment_job["id"]

    return result


def preprocess_frames_handler(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    """Worker handler for creating and storing preprocessed frame payloads."""
    if context.repository is None:
        raise RuntimeError("Preprocess frames handler requires a PostgresRepository.")

    command = PreprocessFramesCommand.from_payload(_job_payload(job))
    payload = command.to_payload()
    project_id = _job_project_id(job, context)
    asset_id = job.get("asset_id") or payload.get("asset_id")
    frame_ids = _payload_frame_ids(payload)

    if not frame_ids:
        if not asset_id:
            raise ValueError("Preprocess job requires asset_id, frame_id, or frame_ids.")
        frames = context.repository.list_frames(
            str(asset_id),
            project_id=project_id,
            start_frame=payload.get("start_frame"),
            end_frame=payload.get("end_frame"),
            limit=payload.get("limit"),
        )
        frame_ids = [str(frame["id"]) for frame in frames]

    if not frame_ids:
        return {
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "project_id": project_id,
            "run_id": job.get("run_id") or payload.get("run_id"),
            "asset_id": asset_id,
            "frame_count": 0,
            "frame_ids": [],
        }

    resolved_asset_id = None if asset_id is None else str(asset_id)
    resolved_run_id = None if job.get("run_id") is None else str(job["run_id"])
    flatfield_defaults = context.config.processing.flatfield
    preprocessing_defaults = context.config.processing.preprocessing
    flatfield_correction = payload.get("flatfield_correction", flatfield_defaults.flatfield_correction)
    flatfield_q = payload.get("flatfield_q", flatfield_defaults.flatfield_q)
    flatfield_axis = payload.get("flatfield_axis", flatfield_defaults.flatfield_axis)
    flatfield_min_field_value = payload.get(
        "flatfield_min_field_value",
        flatfield_defaults.flatfield_min_field_value,
    )
    flatfield_max_field_value = payload.get(
        "flatfield_max_field_value",
        flatfield_defaults.flatfield_max_field_value,
    )
    apply_mask = payload.get("apply_mask", preprocessing_defaults.apply_mask)
    crop_enabled = payload.get("crop_enabled", preprocessing_defaults.crop_enabled)
    crop_x = payload.get("crop_x", preprocessing_defaults.crop_x)
    crop_y = payload.get("crop_y", preprocessing_defaults.crop_y)
    crop_w = payload.get("crop_w", preprocessing_defaults.crop_w)
    crop_h = payload.get("crop_h", preprocessing_defaults.crop_h)
    background_correction = payload.get(
        "background_correction",
        preprocessing_defaults.background_correction,
    )
    background_min_field_value = payload.get(
        "background_min_field_value",
        preprocessing_defaults.background_min_field_value,
    )
    background_max_field_value = payload.get(
        "background_max_field_value",
        preprocessing_defaults.background_max_field_value,
    )
    invert_intensity = payload.get("invert_intensity", preprocessing_defaults.invert_intensity)
    encoding = payload.get("encoding")
    quality = payload.get("quality")
    stored_rows = []
    progress = JobProgressReporter(
        job,
        context,
        stage=PipelineStage.PREPROCESS_FRAMES.value,
        unit="frames",
        total=len(frame_ids),
    )
    progress.start(f"Preprocessing {len(frame_ids)} frame{'s' if len(frame_ids) != 1 else ''}")
    _mark_frame_stage_status(
        context,
        project_id=project_id,
        frame_ids=frame_ids,
        stage=PipelineStage.PREPROCESS_FRAMES,
        status=JobStatus.WORKING,
        job_id=None if job.get("id") is None else str(job.get("id")),
    )

    missing_background_frame_ids = []
    for frame_id in frame_ids:
        frame_record = context.repository.get_frame_record(frame_id, project_id=project_id)
        if frame_record is None:
            raise KeyError(f"Frame {frame_id!r} was not found.")
        if resolved_asset_id is None:
            resolved_asset_id = frame_record.asset_id
        elif frame_record.asset_id != resolved_asset_id:
            raise ValueError("Preprocess jobs may only process frames from one asset.")
        if resolved_run_id is None:
            resolved_run_id = frame_record.run_id
        if not (frame_record.background_payload_ref or frame_record.background_kvstore_hash):
            missing_background_frame_ids.append(frame_id)

    background_generation_result = None

    def generate_preprocess_background(reason: str) -> dict[str, Any]:
        progress.update(
            0,
            secondary={
                "background_generation": "started",
                "missing_background_frames": len(missing_background_frame_ids),
                "reason": reason,
            },
            message=(
                "Generating background for "
                f"{len(frame_ids)} frame{'s' if len(frame_ids) != 1 else ''}"
            ),
            force=True,
        )
        background_kwargs = {
            "context": context,
            "payload_kind": str(payload.get("background_payload_kind", "original")),
            "encoding": str(payload.get("background_encoding", "zstd")),
        }
        if payload.get("background_quality") is not None:
            background_kwargs["quality"] = payload.get("background_quality")
        result = generate_background_for_frames(frame_ids, **background_kwargs)
        progress.update(
            0,
            secondary={
                "background_generation": "completed",
                "background_payload_ref": result.get("background_payload_ref"),
                "background_source_frame_count": result.get("frame_count"),
                "updated_background_frame_count": result.get("updated_frame_count"),
                "reason": reason,
            },
            message=(
                "Generated background for "
                f"{len(frame_ids)} frame{'s' if len(frame_ids) != 1 else ''}"
            ),
            force=True,
        )
        return result

    if background_correction and missing_background_frame_ids:
        background_generation_result = generate_preprocess_background("missing_background_payload")

    for index, frame_id in enumerate(frame_ids, start=1):
        try:
            frame = retrieve_frame(frame_id, context=context, payload_kind="original")
        except KeyError:
            if not background_correction or background_generation_result is not None:
                raise
            background_generation_result = generate_preprocess_background("background_retrieve_failed")
            frame = retrieve_frame(frame_id, context=context, payload_kind="original")
        processed = preprocess_frame_for_segmentation(
            frame,
            flatfield_correction=flatfield_correction,
            flatfield_q=flatfield_q,
            flatfield_axis=flatfield_axis,
            flatfield_min_field_value=flatfield_min_field_value,
            flatfield_max_field_value=flatfield_max_field_value,
            apply_mask=apply_mask,
            crop_enabled=crop_enabled,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            background_correction=background_correction,
            background_min_field_value=background_min_field_value,
            background_max_field_value=background_max_field_value,
            invert_intensity=invert_intensity,
            context=context,
        )
        stored_rows.append(
            store_preprocessed_frame(
                frame_id,
                processed,
                context=context,
                encoding=encoding,
                quality=None if quality is None else int(quality),
            )
        )
        progress.update(
            index,
            current={"frame_id": frame_id, "index": index},
            message=f"Preprocessed {index}/{len(frame_ids)} frames",
        )

    if resolved_run_id is None or resolved_asset_id is None:
        raise ValueError("Preprocess job could not resolve run_id and asset_id.")

    progress.finish(
        completed=len(stored_rows),
        secondary={"preprocessed_frames": len(stored_rows)},
        message=f"Preprocessed {len(stored_rows)} frame{'s' if len(stored_rows) != 1 else ''}",
    )
    result = {
        "stage": PipelineStage.PREPROCESS_FRAMES.value,
        "project_id": project_id,
        "run_id": resolved_run_id,
        "asset_id": resolved_asset_id,
        "frame_count": len(stored_rows),
        "frame_ids": frame_ids,
        "preprocessed_frame_ids": [str(row.get("id")) for row in stored_rows],
    }
    if background_generation_result is not None:
        result["background_generation"] = background_generation_result
    _mark_frame_stage_status(
        context,
        project_id=project_id,
        frame_ids=frame_ids,
        stage=PipelineStage.PREPROCESS_FRAMES,
        status=JobStatus.SUCCEEDED,
        job_id=None if job.get("id") is None else str(job.get("id")),
    )
    _touch_processing_status_snapshot(context, project_id=project_id)
    return result


def background_frames_handler(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    """Worker handler for generating and assigning mean background fields."""
    if context.repository is None:
        raise RuntimeError("Background frames handler requires a PostgresRepository.")

    command = FrameBackgroundCommand.from_payload(_job_payload(job))
    payload = command.to_payload()
    project_id = _job_project_id(job, context)
    asset_id = job.get("asset_id") or payload.get("asset_id")
    frame_ids = _payload_frame_ids(payload)

    if not frame_ids:
        if not asset_id:
            raise ValueError("Background job requires asset_id, frame_id, or frame_ids.")
        frames = context.repository.list_frames(
            str(asset_id),
            project_id=project_id,
            start_frame=payload.get("start_frame"),
            end_frame=payload.get("end_frame"),
            limit=payload.get("limit"),
        )
        frame_ids = [str(frame["id"]) for frame in frames]

    if not frame_ids:
        return {
            "stage": PipelineStage.BACKGROUND_FRAMES.value,
            "project_id": project_id,
            "run_id": job.get("run_id") or payload.get("run_id"),
            "asset_id": asset_id,
            "frame_count": 0,
            "frame_ids": [],
        }

    resolved_asset_id = None if asset_id is None else str(asset_id)
    resolved_run_id = None if job.get("run_id") is None else str(job["run_id"])
    progress = JobProgressReporter(
        job,
        context,
        stage=PipelineStage.BACKGROUND_FRAMES.value,
        unit="frames",
        total=len(frame_ids),
    )
    progress.start(f"Generating background from {len(frame_ids)} frame{'s' if len(frame_ids) != 1 else ''}")
    for frame_id in frame_ids:
        frame_record = context.repository.get_frame_record(frame_id, project_id=project_id)
        if frame_record is None:
            raise KeyError(f"Frame {frame_id!r} was not found.")
        if resolved_asset_id is None:
            resolved_asset_id = frame_record.asset_id
        elif frame_record.asset_id != resolved_asset_id:
            raise ValueError("Background jobs may only process frames from one asset.")
        if resolved_run_id is None:
            resolved_run_id = frame_record.run_id

    background_kwargs = {
        "context": context,
        "payload_kind": str(payload.get("payload_kind", "original")),
        "encoding": str(payload.get("encoding", "zstd")),
    }
    if payload.get("quality") is not None:
        background_kwargs["quality"] = payload.get("quality")
    result = generate_background_for_frames(frame_ids, **background_kwargs)
    result.update(
        {
            "stage": PipelineStage.BACKGROUND_FRAMES.value,
            "project_id": project_id,
            "run_id": resolved_run_id,
            "asset_id": resolved_asset_id,
        }
    )
    progress.finish(
        completed=len(frame_ids),
        message=f"Generated background from {len(frame_ids)} frame{'s' if len(frame_ids) != 1 else ''}",
    )
    return result


def roi_detection_handler(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    """Worker handler for segmenting stored frames into ROI detections."""
    if context.repository is None:
        raise RuntimeError("ROI detection handler requires a PostgresRepository.")

    command = SegmentFramesCommand.from_payload(_job_payload(job))
    payload = command.to_payload()
    project_id = _job_project_id(job, context)
    asset_id = job.get("asset_id") or payload.get("asset_id")
    frame_ids = _payload_frame_ids(payload)

    if not frame_ids:
        if not asset_id:
            raise ValueError("Segment job requires asset_id, frame_id, or frame_ids.")
        frames = context.repository.list_frames(
            str(asset_id),
            project_id=project_id,
            start_frame=payload.get("start_frame"),
            end_frame=payload.get("end_frame"),
            limit=payload.get("limit"),
        )
        frame_ids = [str(frame["id"]) for frame in frames]

    if not frame_ids:
        return {
            "stage": PipelineStage.SEGMENT.value,
            "project_id": project_id,
            "run_id": job.get("run_id") or payload.get("run_id"),
            "asset_id": asset_id,
            "frame_count": 0,
            "detection_count": 0,
            "frame_ids": [],
            "detection_ids": [],
        }

    detections = []
    resolved_asset_id = None if asset_id is None else str(asset_id)
    resolved_run_id = None if job.get("run_id") is None else str(job["run_id"])
    segmentation_payload = dict(payload)
    if segmentation_payload.get("roi_encoding") is None:
        segmentation_payload["roi_encoding"] = resolve_project_storage_settings(
            context,
            project_id,
        ).roi_encoding
    resolved_options = resolve_segmentation_options(segmentation_payload, context.config.processing)
    frame_payload_kind = resolved_options["source"]["frame_payload_kind"]
    frame_kwargs = segment_frame_kwargs(resolved_options)
    progress = JobProgressReporter(
        job,
        context,
        stage=PipelineStage.SEGMENT.value,
        unit="frames",
        total=len(frame_ids),
    )
    progress.start(f"Segmenting {len(frame_ids)} frame{'s' if len(frame_ids) != 1 else ''}")
    _mark_frame_stage_status(
        context,
        project_id=project_id,
        frame_ids=frame_ids,
        stage=PipelineStage.SEGMENT,
        status=JobStatus.WORKING,
        job_id=None if job.get("id") is None else str(job.get("id")),
    )

    for index, frame_id in enumerate(frame_ids, start=1):
        frame_record = context.repository.get_frame_record(frame_id, project_id=project_id)
        if frame_record is None:
            raise KeyError(f"Frame {frame_id!r} was not found.")
        if resolved_asset_id is None:
            resolved_asset_id = frame_record.asset_id
        elif frame_record.asset_id != resolved_asset_id:
            raise ValueError("Segment jobs may only process frames from one asset.")
        if resolved_run_id is None:
            resolved_run_id = frame_record.run_id

        try:
            frame = retrieve_frame(frame_id, context=context, payload_kind=frame_payload_kind)
        except TypeError as exc:
            if "payload_kind" not in str(exc):
                raise
            frame = retrieve_frame(frame_id, context=context)
        detections.extend(
            segment_frame(
                frame,
                frame_record=frame_record,
                **frame_kwargs,
                context=context,
            )
        )
        progress.update(
            index,
            current={"frame_id": frame_id, "index": index},
            secondary={"detections_created": len(detections)},
            message=f"Segmented {index}/{len(frame_ids)} frames",
        )

    if resolved_run_id is None or resolved_asset_id is None:
        raise ValueError("Segment job could not resolve run_id and asset_id.")

    inserted = context.repository.replace_frame_detections(
        resolved_run_id,
        frame_ids,
        detections,
        project_id=project_id,
    )
    _mark_frame_stage_status(
        context,
        project_id=project_id,
        frame_ids=frame_ids,
        stage=PipelineStage.SEGMENT,
        status=JobStatus.SUCCEEDED,
        job_id=None if job.get("id") is None else str(job.get("id")),
    )
    _refresh_frame_status_counts(
        context,
        project_id=project_id,
        frame_ids=frame_ids,
        asset_id=resolved_asset_id,
    )
    _touch_processing_status_snapshot(context, project_id=project_id)
    progress.finish(
        completed=len(frame_ids),
        secondary={"detections_created": len(inserted)},
        message=f"Segmented {len(frame_ids)} frame{'s' if len(frame_ids) != 1 else ''}; {len(inserted)} detections",
    )
    return {
        "stage": PipelineStage.SEGMENT.value,
        "project_id": project_id,
        "run_id": resolved_run_id,
        "asset_id": resolved_asset_id,
        "frame_count": len(frame_ids),
        "detection_count": len(inserted),
        "frame_ids": frame_ids,
        "detection_ids": [row.get("id") for row in inserted],
        "resolved_options": resolved_options,
    }


def _roi_refinement_options_from_payload(payload: dict[str, Any], context: AppContext) -> RoiRefinementOptions:
    defaults = context.config.processing.roi_refinement
    return RoiRefinementOptions(
        tile_size=int(payload.get("tile_size", defaults.tile_size)),
        overlap_fraction=float(payload.get("overlap_fraction", defaults.overlap_fraction)),
        max_iterations=int(payload.get("max_iterations", defaults.max_iterations)),
        expansion_pixels=payload.get("expansion_pixels", defaults.expansion_pixels),
        edge_touch_margin=int(payload.get("edge_touch_margin", defaults.edge_touch_margin)),
        output_threshold=float(payload.get("output_threshold", defaults.output_threshold)),
        batch_size=payload.get("batch_size", defaults.batch_size),
        encoding=_resolved_roi_refinement_encoding(payload.get("encoding", defaults.encoding)),
        overlap_reconciliation_enabled=bool(
            payload.get(
                "overlap_reconciliation_enabled",
                defaults.overlap_reconciliation_enabled,
            )
        ),
        overlap_iou_threshold=float(
            payload.get("overlap_iou_threshold", defaults.overlap_iou_threshold)
        ),
        overlap_containment_threshold=float(
            payload.get(
                "overlap_containment_threshold",
                defaults.overlap_containment_threshold,
            )
        ),
        residual_discovery_enabled=bool(
            payload.get(
                "residual_discovery_enabled",
                defaults.residual_discovery_enabled,
            )
        ),
        residual_max_iterations=int(
            payload.get("residual_max_iterations", defaults.residual_max_iterations)
        ),
        residual_roi_assembly_method=payload.get(
            "residual_roi_assembly_method",
            defaults.residual_roi_assembly_method,
        ),
        residual_roi_assembly_connectivity=int(
            payload.get(
                "residual_roi_assembly_connectivity",
                defaults.residual_roi_assembly_connectivity,
            )
        ),
        residual_min_area=payload.get("residual_min_area", defaults.residual_min_area),
        residual_min_width=payload.get("residual_min_width", defaults.residual_min_width),
        residual_min_height=payload.get("residual_min_height", defaults.residual_min_height),
        residual_min_width_plus_height=payload.get(
            "residual_min_width_plus_height",
            defaults.residual_min_width_plus_height,
        ),
        residual_padding=payload.get("residual_padding", defaults.residual_padding),
    )


def _resolved_roi_refinement_encoding(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return None if normalized in {"", "auto", "default", "none", "null"} else normalized


def roi_refinement_handler(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    """Worker handler for refining candidate ROI masks into detections_refined rows."""
    if context.repository is None:
        raise RuntimeError("ROI refinement handler requires a PostgresRepository.")

    command = RoiRefinementCommand.from_payload(_job_payload(job))
    payload = command.to_payload()
    project_id = _job_project_id(job, context)
    detection_ids = [str(detection_id) for detection_id in payload.get("detection_ids", []) if detection_id]
    if not detection_ids:
        raise ValueError("ROI refinement job requires detection_ids.")

    progress = JobProgressReporter(
        job,
        context,
        stage=PipelineStage.ROI_REFINEMENT.value,
        unit="rois",
        total=len(detection_ids),
    )
    progress.start(f"Refining {len(detection_ids)} ROI{'s' if len(detection_ids) != 1 else ''}")

    missing_ids = []
    candidate_rows = []
    for detection_id in detection_ids:
        row = context.repository.get_detection(detection_id, project_id=project_id)
        if row is None:
            missing_ids.append(detection_id)
        else:
            candidate_rows.append(row)
    if missing_ids:
        raise KeyError(f"Detection(s) not found: {', '.join(missing_ids)}")
    candidate_frame_ids = sorted({str(row["frame_id"]) for row in candidate_rows if row.get("frame_id")})
    _mark_frame_stage_status(
        context,
        project_id=project_id,
        frame_ids=candidate_frame_ids,
        stage=PipelineStage.ROI_REFINEMENT,
        status=JobStatus.WORKING,
        job_id=None if job.get("id") is None else str(job.get("id")),
    )
    progress.update(
        0,
        current={"loaded_candidates": len(candidate_rows)},
        secondary={"loaded_candidates": len(candidate_rows)},
        message=f"Loaded {len(candidate_rows)} ROI candidate{'s' if len(candidate_rows) != 1 else ''}",
        force=True,
    )

    defaults = context.config.processing.roi_refinement
    model = resolve_refinement_model(
        context.config,
        model_kind=payload.get("model_kind"),
        model_ref=payload.get("model_ref"),
        model_run_dir=payload.get("model_run_dir"),
        model_artifact=payload.get("model_artifact", defaults.model_artifact),
    )
    method = "identity" if model.__class__.__name__ == "IdentityRoiRefinementModel" else (
        getattr(model, "method_name", None) or model.__class__.__name__
    )
    options = _roi_refinement_options_from_payload(payload, context)
    if payload.get("encoding") is None:
        options.encoding = resolve_project_storage_settings(context, project_id).roi_encoding
    allow_frame_expansion = bool(payload.get("allow_frame_expansion", True))
    expansion_payload_kind = str(payload.get("expansion_frame_payload_kind", "preprocessed"))
    frame_loader = None
    if allow_frame_expansion:
        frame_cache: dict[str, Any] = {}

        def frame_loader(frame_id: str):
            resolved_frame_id = str(frame_id)
            if resolved_frame_id not in frame_cache:
                frame_cache[resolved_frame_id] = retrieve_frame(
                    resolved_frame_id,
                    context=context,
                    payload_kind=expansion_payload_kind,
                ).read()
            return frame_cache[resolved_frame_id]

    detection_records = [DetectionRecord.from_row(row) for row in candidate_rows]
    results = refine_detections(
        detection_records,
        model=model,
        frame_loader=frame_loader,
        options=options,
        method=method,
    )
    refined_records = [result.as_detection_record(encoding=options.encoding) for result in results]
    stored = context.repository.upsert_refined_detections(
        [
            (refined_storage_candidate_detection_id(result), refined)
            for result, refined in zip(results, refined_records)
        ],
        job_id=job.get("id"),
        project_id=project_id,
    )
    run_ids = sorted({record.run_id for record in detection_records})
    frame_ids = sorted({record.frame_id for record in detection_records})
    _mark_frame_stage_status(
        context,
        project_id=project_id,
        frame_ids=frame_ids,
        stage=PipelineStage.ROI_REFINEMENT,
        status=JobStatus.SUCCEEDED,
        job_id=None if job.get("id") is None else str(job.get("id")),
    )
    _refresh_frame_status_counts(
        context,
        project_id=project_id,
        frame_ids=frame_ids,
    )
    _touch_processing_status_snapshot(context, project_id=project_id)
    progress.finish(
        completed=len(detection_records),
        secondary={
            "refined_created": len(refined_records),
            "stored_count": len(stored),
            "synthetic_refined_count": sum(
                1 for result in results if result.candidate_detection.metadata.get("synthetic_candidate")
            ),
        },
        message=f"Refined {len(refined_records)} ROI{'s' if len(refined_records) != 1 else ''}",
    )
    return {
        "stage": PipelineStage.ROI_REFINEMENT.value,
        "project_id": project_id,
        "run_id": job.get("run_id") or payload.get("run_id") or (run_ids[0] if len(run_ids) == 1 else None),
        "asset_id": job.get("asset_id") or payload.get("asset_id"),
        "detection_count": len(detection_records),
        "refined_count": len(refined_records),
        "stored_count": len(stored),
        "synthetic_refined_count": sum(
            1 for result in results if result.candidate_detection.metadata.get("synthetic_candidate")
        ),
        "detection_ids": detection_ids,
        "refined_detection_ids": [row.get("id") for row in stored],
        "frame_ids": frame_ids,
        "model_kind": payload.get("model_kind") or defaults.model_kind,
        "model_ref": payload.get("model_ref"),
        "refinement_method": method,
        "resolved_options": {
            "tile_size": options.tile_size,
            "overlap_fraction": options.overlap_fraction,
            "max_iterations": options.max_iterations,
            "expansion_pixels": options.expansion_pixels,
            "edge_touch_margin": options.edge_touch_margin,
            "output_threshold": options.output_threshold,
            "batch_size": options.batch_size,
            "encoding": options.encoding,
            "overlap_reconciliation_enabled": options.overlap_reconciliation_enabled,
            "overlap_iou_threshold": options.overlap_iou_threshold,
            "overlap_containment_threshold": options.overlap_containment_threshold,
            "residual_discovery_enabled": options.residual_discovery_enabled,
            "residual_max_iterations": options.residual_max_iterations,
            "residual_roi_assembly_method": options.residual_roi_assembly_method,
            "residual_roi_assembly_connectivity": options.residual_roi_assembly_connectivity,
            "residual_min_area": options.residual_min_area,
            "residual_min_width": options.residual_min_width,
            "residual_min_height": options.residual_min_height,
            "residual_min_width_plus_height": options.residual_min_width_plus_height,
            "residual_padding": options.residual_padding,
            "allow_frame_expansion": allow_frame_expansion,
            "expansion_frame_payload_kind": expansion_payload_kind,
        },
    }


def default_handler_registry() -> HandlerRegistry:
    """Build the default worker stage registry."""
    registry = HandlerRegistry()
    registry.register(PipelineStage.EXTRACT_FRAMES, extract_frames_handler)
    registry.register(PipelineStage.BACKGROUND_FRAMES, background_frames_handler)
    registry.register(PipelineStage.PREPROCESS_FRAMES, preprocess_frames_handler)
    registry.register(PipelineStage.SEGMENT, roi_detection_handler)
    registry.register(PipelineStage.ROI_REFINEMENT, roi_refinement_handler)
    return registry
