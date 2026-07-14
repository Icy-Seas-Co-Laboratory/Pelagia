"""Backend-owned planning and batching for processing jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..services.context import AppContext
from .job_commands import PreprocessFramesCommand, RoiRefinementCommand, SegmentFramesCommand, FrameSelection


Ordering = Literal["optimized", "input"]


@dataclass(frozen=True, slots=True)
class PreprocessQueueRequest:
    filters: dict[str, Any]
    options: dict[str, Any]
    max_units: int = 250
    ordering: Ordering = "optimized"
    priority: int | None = None
    dry_run: bool = False


class ProcessingQueueService:
    """Resolve work units and create efficiently ordered processing jobs."""

    def __init__(self, context: AppContext):
        if context.repository is None:
            raise RuntimeError("Processing queue operations require a PostgresRepository.")
        self.context = context
        self.repository = context.repository

    def queue_preprocess(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        if request.max_units < 1 or request.max_units > 10_000:
            raise ValueError("batch.max_units must be between 1 and 10000.")
        if request.ordering not in {"optimized", "input"}:
            raise ValueError("batch.ordering must be 'optimized' or 'input'.")
        filters = {**request.filters}
        filters["preprocessing_status"] = filters.get("preprocessing_status") or ["unknown", "failed"]
        frames = self._plan_by_assets("plan_preprocess_frames", project_id=project_id, filters=filters)
        if request.ordering == "input":
            frames.sort(key=lambda row: (str(row["asset_id"]), int(row["frame_index"]), str(row["frame_id"])))
            ordering = "asset_frame"
        else:
            frames.sort(key=lambda row: (str(row.get("payload_ref") or ""), str(row["frame_id"])))
            ordering = "kvstore_hash"
        batches = [frames[index:index + request.max_units] for index in range(0, len(frames), request.max_units)]
        planned_jobs = [self._job_for_batch(batch, request.options) for batch in batches]
        result = {
            "stage": "preprocess_frames",
            "unit": "frames",
            "matched_count": len(frames),
            "job_count": len(planned_jobs),
            "ordering": ordering,
            "batch_sizes": [len(batch) for batch in batches],
            "sample_frame_ids": [str(row["frame_id"]) for row in frames[:20]],
            "dry_run": request.dry_run,
        }
        if request.dry_run:
            return result
        created = self.repository.create_preprocess_jobs(
            project_id=project_id,
            jobs=planned_jobs,
            eligible_statuses=filters["preprocessing_status"],
            priority=request.priority,
        )
        return {**result, "job_ids": [str(row["id"]) for row in created]}

    def queue_segment(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        payload_kind = str(request.options.get("frame_payload_kind", "original"))
        filters = {**request.filters}
        filters["candidate_detection_status"] = filters.get("candidate_detection_status") or ["unknown", "failed"]
        frames = self._plan_by_assets(
            "plan_segment_frames", project_id=project_id, filters=filters, payload_kind=payload_kind
        )
        return self._queue_frame_stage("segment", frames, request, project_id=project_id, command=SegmentFramesCommand, ordering="kvstore_hash")

    def queue_roi_refinement(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        detections = self._plan_by_assets(
            "plan_roi_refinement_detections", project_id=project_id, filters=request.filters
        )
        return self._queue_detection_stage(detections, request, project_id=project_id)

    def _queue_frame_stage(self, stage: str, frames: list[dict[str, Any]], request: PreprocessQueueRequest, *, project_id: str, command, ordering: str) -> dict[str, Any]:
        if request.max_units < 1 or request.max_units > 10_000:
            raise ValueError("batch.max_units must be between 1 and 10000.")
        frames.sort(key=lambda row: (str(row.get("payload_ref") or ""), str(row["frame_id"])))
        batches = [frames[index:index + request.max_units] for index in range(0, len(frames), request.max_units)]
        result = {"stage": stage, "unit": "frames", "matched_count": len(frames), "job_count": len(batches), "ordering": ordering, "batch_sizes": [len(batch) for batch in batches], "dry_run": request.dry_run}
        if request.dry_run:
            return result
        jobs = []
        for batch in batches:
            frame_ids = [str(row["frame_id"]) for row in batch]
            assets, runs = {str(row["asset_id"]) for row in batch}, {str(row["run_id"]) for row in batch if row.get("run_id")}
            payload = command(selection=FrameSelection(frame_ids=tuple(frame_ids)), options=dict(request.options)).to_payload()
            jobs.append(self.repository.create_job(stage, project_id=project_id, run_id=next(iter(runs)) if len(runs) == 1 else None, asset_id=next(iter(assets)) if len(assets) == 1 else None, priority=request.priority, payload=payload, summary=f"{stage} queued for {len(frame_ids)} frames"))
        return {**result, "job_ids": [str(job["id"]) for job in jobs]}

    def _queue_detection_stage(self, detections: list[dict[str, Any]], request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        if request.max_units < 1 or request.max_units > 10_000:
            raise ValueError("batch.max_units must be between 1 and 10000.")
        detections.sort(key=lambda row: (str(row["frame_id"]), int(row.get("roi_index") or 0), str(row["detection_id"])))
        batches = [detections[index:index + request.max_units] for index in range(0, len(detections), request.max_units)]
        result = {"stage": "roi_refinement", "unit": "detections", "matched_count": len(detections), "job_count": len(batches), "ordering": "frame_id", "batch_sizes": [len(batch) for batch in batches], "dry_run": request.dry_run}
        if request.dry_run:
            return result
        jobs = []
        for batch in batches:
            detection_ids = [str(row["detection_id"]) for row in batch]
            assets, runs = {str(row["asset_id"]) for row in batch}, {str(row["run_id"]) for row in batch if row.get("run_id")}
            payload = RoiRefinementCommand(detection_ids=tuple(detection_ids), options=dict(request.options)).to_payload()
            jobs.append(self.repository.create_job("roi_refinement", project_id=project_id, run_id=next(iter(runs)) if len(runs) == 1 else None, asset_id=next(iter(assets)) if len(assets) == 1 else None, priority=request.priority, payload=payload, summary=f"roi refinement queued for {len(detection_ids)} detections"))
        return {**result, "job_ids": [str(job["id"]) for job in jobs]}

    def _plan_by_assets(self, method_name: str, *, project_id: str, filters: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Apply a multi-asset filter through existing project-scoped planner queries."""
        asset_ids = [str(asset_id) for asset_id in filters.get("asset_ids") or [] if asset_id]
        asset_ids = list(dict.fromkeys(asset_ids))
        base_filters = {key: value for key, value in filters.items() if key != "asset_ids"}
        planner = getattr(self.repository, method_name)
        if not asset_ids:
            return planner(project_id=project_id, filters=base_filters, **kwargs)
        rows: list[dict[str, Any]] = []
        for asset_id in asset_ids:
            rows.extend(planner(project_id=project_id, filters={**base_filters, "asset_id": asset_id}, **kwargs))
        return rows

    @staticmethod
    def _job_for_batch(batch: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        frame_ids = [str(row["frame_id"]) for row in batch]
        asset_ids = {str(row["asset_id"]) for row in batch}
        run_ids = {str(row["run_id"]) for row in batch if row.get("run_id") is not None}
        payload = PreprocessFramesCommand(
            selection=FrameSelection(frame_ids=tuple(frame_ids)),
            options=dict(options),
        ).to_payload()
        asset_id = next(iter(asset_ids)) if len(asset_ids) == 1 else None
        run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
        return {
            "frame_ids": frame_ids,
            "asset_id": asset_id,
            "run_id": run_id,
            "payload": payload,
            "summary": f"preprocess queued for {len(frame_ids)} frames",
        }
