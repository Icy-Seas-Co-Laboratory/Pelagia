from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..domain import PipelineStage
from ..domain import normalize_collections
from ..processing import video_ingest as video_ingest_module
from ..processing.frame_store import retrieve_frame
from ..processing.segmentation import segment_frame
from ..services.context import AppContext


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


def extract_frames_handler(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    """Worker handler for extracting and storing frames from a registered asset."""
    if context.repository is None:
        raise RuntimeError("Extract frames handler requires a PostgresRepository.")

    payload = _job_payload(job)
    run_id = _job_identifier(job, "run_id", payload)
    asset_id = _job_identifier(job, "asset_id", payload)

    asset = context.repository.get_asset(asset_id)
    if asset is None:
        raise KeyError(f"Raw asset {asset_id!r} was not found.")

    source_path = payload.get("source_path") or asset.get("path")
    if not source_path:
        raise ValueError(f"Raw asset {asset_id!r} does not include a source path.")

    metadata = dict(payload.get("metadata") or {})
    collections = normalize_collections(payload.get("collections") or asset.get("collections"))
    metadata.setdefault("collections", collections)
    metadata.setdefault("worker_job_id", str(job.get("id")))
    metadata.setdefault("worker_stage", PipelineStage.EXTRACT_FRAMES.value)

    frame_rows = video_ingest_module.ingest_video_file(
        source_path,
        n_tile=int(payload.get("n_tile", 1)),
        context=context,
        run_id=run_id,
        asset_id=asset_id,
        metadata=metadata,
        flatfield_correction=bool(payload.get("flatfield_correction", True)),
        flatfield_q=float(payload.get("flatfield_q", 0.9)),
        flatfield_axis=int(payload.get("flatfield_axis", 0)),
    )

    result: dict[str, Any] = {
        "stage": PipelineStage.EXTRACT_FRAMES.value,
        "run_id": run_id,
        "asset_id": asset_id,
        "source_path": str(source_path),
        "frame_count": len(frame_rows),
        "frame_ids": [row.get("id") for row in frame_rows],
    }

    if payload.get("enqueue_segment"):
        segment_job = context.repository.create_job(
            PipelineStage.SEGMENT,
            run_id=run_id,
            asset_id=asset_id,
            payload={
                "frame_ids": result["frame_ids"],
                "padding": payload.get("segmentation_padding", payload.get("padding", 0)),
                "roi_encoding": payload.get("roi_encoding", "zstd"),
                "collections": collections,
            },
            depends_on=[str(job["id"])],
            summary=f"segment queued for {len(frame_rows)} extracted frames",
        )
        result["segment_job_id"] = segment_job["id"]

    return result


def roi_detection_handler(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    """Worker handler for segmenting stored frames into ROI detections."""
    if context.repository is None:
        raise RuntimeError("ROI detection handler requires a PostgresRepository.")

    payload = _job_payload(job)
    asset_id = job.get("asset_id") or payload.get("asset_id")
    frame_ids = _payload_frame_ids(payload)

    if not frame_ids:
        if not asset_id:
            raise ValueError("Segment job requires asset_id, frame_id, or frame_ids.")
        frames = context.repository.list_frames(
            str(asset_id),
            start_frame=payload.get("start_frame"),
            end_frame=payload.get("end_frame"),
            limit=payload.get("limit"),
        )
        frame_ids = [str(frame["id"]) for frame in frames]

    if not frame_ids:
        return {
            "stage": PipelineStage.SEGMENT.value,
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
    padding = payload.get("padding", payload.get("segmentation_padding", 0))
    zstd_min_bytes = payload.get("zstd_min_bytes", 1024)
    min_perimeter = payload.get("min_perimeter", 0)

    for frame_id in frame_ids:
        frame_record = context.repository.get_frame_record(frame_id)
        if frame_record is None:
            raise KeyError(f"Frame {frame_id!r} was not found.")
        if resolved_asset_id is None:
            resolved_asset_id = frame_record.asset_id
        elif frame_record.asset_id != resolved_asset_id:
            raise ValueError("Segment jobs may only process frames from one asset.")
        if resolved_run_id is None:
            resolved_run_id = frame_record.run_id

        frame = retrieve_frame(frame_id, context=context)
        detections.extend(
            segment_frame(
                frame,
                threshold=payload.get("threshold"),
                min_perimeter=0 if min_perimeter is None else min_perimeter,
                max_perimeter=payload.get("max_perimeter"),
                padding=0 if padding is None else int(padding),
                roi_encoding=payload.get("roi_encoding", "zstd"),
                zstd_min_bytes=1024 if zstd_min_bytes is None else int(zstd_min_bytes),
            )
        )

    if resolved_run_id is None or resolved_asset_id is None:
        raise ValueError("Segment job could not resolve run_id and asset_id.")

    inserted = context.repository.replace_frame_detections(
        resolved_run_id,
        frame_ids,
        detections,
    )
    return {
        "stage": PipelineStage.SEGMENT.value,
        "run_id": resolved_run_id,
        "asset_id": resolved_asset_id,
        "frame_count": len(frame_ids),
        "detection_count": len(inserted),
        "frame_ids": frame_ids,
        "detection_ids": [row.get("id") for row in inserted],
    }


def default_handler_registry() -> HandlerRegistry:
    """Build the default worker stage registry."""
    registry = HandlerRegistry()
    registry.register(PipelineStage.EXTRACT_FRAMES, extract_frames_handler)
    registry.register(PipelineStage.SEGMENT, roi_detection_handler)
    return registry
