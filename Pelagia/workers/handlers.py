from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..domain import PipelineStage
from ..domain import normalize_collections
from ..processing import video_ingest as video_ingest_module
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
        raise ValueError(f"Extract frames job requires {key}.")
    return str(value)


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


def default_handler_registry() -> HandlerRegistry:
    """Build the default worker stage registry."""
    registry = HandlerRegistry()
    registry.register(PipelineStage.EXTRACT_FRAMES, extract_frames_handler)
    return registry
