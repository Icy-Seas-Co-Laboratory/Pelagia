"""Backend-owned planning and batching for processing jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..services.context import AppContext
from .job_commands import PreprocessFramesCommand, RoiRefinementCommand, SegmentFramesCommand, FrameSelection


PREPROCESS_FRAMES_PER_JOB = 1_000
SEGMENT_FRAMES_PER_JOB = 1_000
ROI_REFINEMENT_DETECTIONS_PER_JOB = 10_000


@dataclass(frozen=True, slots=True)
class PreprocessQueueRequest:
    filters: dict[str, Any]
    options: dict[str, Any]
    priority: int | None = None
    dry_run: bool = False
    submitted_by_user_id: str | None = None
    submitted_by_username: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingSeriesRequest:
    steps: tuple[dict[str, Any], ...]
    selection: dict[str, Any] | None = None
    preset_snapshot: dict[str, Any] | None = None
    failure_policy: str = "fail_fast"
    priority: int | None = None
    dry_run: bool = False
    submitted_by_user_id: str | None = None
    submitted_by_username: str | None = None


class ProcessingQueueService:
    """Resolve work units and create efficiently ordered processing jobs."""

    def __init__(self, context: AppContext):
        if context.repository is None:
            raise RuntimeError("Processing queue operations require a PostgresRepository.")
        self.context = context
        self.repository = context.repository

    def queue_preprocess(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        filters = {**request.filters}
        filters["preprocessing_status"] = filters.get("preprocessing_status") or ["unknown", "failed"]
        frames = self._plan_by_assets("plan_preprocess_frames", project_id=project_id, filters=filters)
        frames.sort(key=lambda row: (str(row.get("payload_ref") or ""), str(row["frame_id"])))
        batches = self._batches(frames, PREPROCESS_FRAMES_PER_JOB)
        planned_jobs = [self._job_for_batch(batch, request.options) for batch in batches]
        result = {
            "stage": "preprocess_frames",
            "unit": "frames",
            "matched_count": len(frames),
            "job_count": len(planned_jobs),
            "ordering": "kvstore_hash",
            "max_units_per_job": PREPROCESS_FRAMES_PER_JOB,
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
            submitted_by_user_id=request.submitted_by_user_id,
            submitted_by_username=request.submitted_by_username,
        )
        return {**result, "job_ids": [str(row["id"]) for row in created]}

    def queue_segment(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        payload_kind = str(request.options.get("frame_payload_kind", "original"))
        filters = {**request.filters}
        filters["candidate_detection_status"] = filters.get("candidate_detection_status") or ["unknown", "failed"]
        frames = self._plan_by_assets(
            "plan_segment_frames", project_id=project_id, filters=filters, payload_kind=payload_kind
        )
        return self._queue_frame_stage(
            "segment",
            frames,
            request,
            project_id=project_id,
            command=SegmentFramesCommand,
            max_units=SEGMENT_FRAMES_PER_JOB,
            ordering="kvstore_hash",
        )

    def queue_roi_refinement(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        detections = self._plan_by_assets(
            "plan_roi_refinement_detections", project_id=project_id, filters=request.filters
        )
        return self._queue_detection_stage(
            detections,
            request,
            project_id=project_id,
            max_units=ROI_REFINEMENT_DETECTIONS_PER_JOB,
        )

    def create_series(self, request: ProcessingSeriesRequest, *, project_id: str) -> dict[str, Any]:
        """Persist a staged plan, then let the director materialize only its first step."""
        self._validate_series_steps(request.steps)
        if request.failure_policy not in {"fail_fast", "continue"}:
            raise ValueError("failure_policy must be one of: fail_fast, continue.")
        if request.dry_run:
            planned_steps = [self._plan_step(step, project_id=project_id, dry_run=True) for step in request.steps]
            return {
                "dry_run": True,
                "failure_policy": request.failure_policy,
                "steps": planned_steps,
                "eligibility": self._eligibility(planned_steps),
            }
        series = self.repository.create_processing_series(
            project_id=project_id, steps=request.steps, selection=request.selection or {},
            preset_snapshot=request.preset_snapshot or {}, failure_policy=request.failure_policy,
            priority=request.priority, submitted_by_user_id=request.submitted_by_user_id,
            submitted_by_username=request.submitted_by_username,
        )
        return self.advance_series(str(series["id"]), project_id=project_id) or series

    @staticmethod
    def _eligibility(planned_steps: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        total = 0
        for step in planned_steps:
            matched = int(step.get("matched_count") or 0)
            total += matched
            rows.append({
                "stage": step.get("stage"),
                "eligible_count": matched,
                "ineligible_count": 0,
                "job_count": int(step.get("job_count") or 0),
                "reasons": {},
            })
        return {"eligible_count": total, "ineligible_count": 0, "by_step": rows}

    def advance_series_for_job(self, job_id: str) -> dict[str, Any] | None:
        """Completion hook used by workers; it is safe to call more than once."""
        advanced = self.repository.advance_processing_series_for_job(job_id)
        if advanced is None or not advanced.get("ready"):
            return advanced
        series = self.repository.get_processing_series(str(advanced["series_id"]))
        if series is None or series.get("status") in {"failed", "cancelled", "paused"}:
            return advanced
        self.advance_series(str(advanced["series_id"]), project_id=str(series["project_id"]))
        return advanced

    def advance_series(self, series_id: str, *, project_id: str) -> dict[str, Any] | None:
        """Plan queued steps one at a time, recursively passing empty selections."""
        while True:
            step = self.repository.claim_processing_series_step(series_id, project_id=project_id)
            if step is None:
                return self.repository.get_processing_series(series_id, project_id=project_id)
            result = self._plan_step(step, project_id=project_id, dry_run=False)
            self.repository.attach_processing_work_units(
                series_id=series_id, step_id=str(step["id"]), job_ids=result.get("job_ids", []),
                matched_count=int(result["matched_count"]),
            )
            # An empty selection is explicitly recorded as skipped and the next
            # stage can be evaluated immediately.  Nonempty work awaits workers.
            if result.get("job_ids"):
                return self.repository.get_processing_series(series_id, project_id=project_id)

    def _plan_step(self, step: dict[str, Any], *, project_id: str, dry_run: bool) -> dict[str, Any]:
        request = PreprocessQueueRequest(
            filters=dict(step.get("filters") or {}), options=dict(step.get("options") or {}),
            priority=step.get("priority", step.get("series_priority")), dry_run=dry_run,
            submitted_by_user_id=step.get("submitted_by_user_id"),
            submitted_by_username=step.get("submitted_by_username"),
        )
        stage = str(step["stage"])
        if stage == "preprocess_frames":
            return self.queue_preprocess(request, project_id=project_id)
        if stage == "segment":
            return self.queue_segment(request, project_id=project_id)
        if stage == "roi_refinement":
            return self.queue_roi_refinement(request, project_id=project_id)
        raise ValueError(f"Unsupported processing series stage: {stage}.")

    @staticmethod
    def _validate_series_steps(steps: tuple[dict[str, Any], ...]) -> None:
        if not steps:
            raise ValueError("A processing series requires at least one step.")
        canonical_order = ("preprocess_frames", "segment", "roi_refinement")
        allowed = set(canonical_order)
        invalid = [str(step.get("stage")) for step in steps if str(step.get("stage")) not in allowed]
        if invalid:
            raise ValueError(f"Unsupported processing series stage(s): {', '.join(invalid)}.")
        stages = tuple(str(step["stage"]) for step in steps)
        if len(set(stages)) != len(stages) or tuple(sorted(stages, key=canonical_order.index)) != stages:
            raise ValueError("Processing series stages must use the canonical Pelagia order.")

    def _queue_frame_stage(
        self,
        stage: str,
        frames: list[dict[str, Any]],
        request: PreprocessQueueRequest,
        *,
        project_id: str,
        command,
        max_units: int,
        ordering: str,
    ) -> dict[str, Any]:
        frames.sort(key=lambda row: (str(row.get("payload_ref") or ""), str(row["frame_id"])))
        batches = self._batches(frames, max_units)
        result = {
            "stage": stage,
            "unit": "frames",
            "matched_count": len(frames),
            "job_count": len(batches),
            "ordering": ordering,
            "max_units_per_job": max_units,
            "batch_sizes": [len(batch) for batch in batches],
            "dry_run": request.dry_run,
        }
        if request.dry_run:
            return result
        jobs = []
        for batch in batches:
            frame_ids = [str(row["frame_id"]) for row in batch]
            assets, runs = {str(row["asset_id"]) for row in batch}, {str(row["run_id"]) for row in batch if row.get("run_id")}
            payload = command(selection=FrameSelection(frame_ids=tuple(frame_ids)), options=dict(request.options)).to_payload()
            jobs.append(
                self.repository.create_job(
                    stage,
                    project_id=project_id,
                    run_id=next(iter(runs)) if len(runs) == 1 else None,
                    asset_id=next(iter(assets)) if len(assets) == 1 else None,
                    priority=request.priority,
                    payload=payload,
                    summary=f"{stage} queued for {len(frame_ids)} frames",
                    submitted_by_user_id=request.submitted_by_user_id,
                    submitted_by_username=request.submitted_by_username,
                )
            )
        return {**result, "job_ids": [str(job["id"]) for job in jobs]}

    def _queue_detection_stage(
        self,
        detections: list[dict[str, Any]],
        request: PreprocessQueueRequest,
        *,
        project_id: str,
        max_units: int,
    ) -> dict[str, Any]:
        detections.sort(key=lambda row: (str(row["frame_id"]), int(row.get("roi_index") or 0), str(row["detection_id"])))
        batches = self._batches(detections, max_units)
        result = {
            "stage": "roi_refinement",
            "unit": "detections",
            "matched_count": len(detections),
            "job_count": len(batches),
            "ordering": "frame_id",
            "max_units_per_job": max_units,
            "batch_sizes": [len(batch) for batch in batches],
            "dry_run": request.dry_run,
        }
        if request.dry_run:
            return result
        jobs = []
        for batch in batches:
            detection_ids = [str(row["detection_id"]) for row in batch]
            assets, runs = {str(row["asset_id"]) for row in batch}, {str(row["run_id"]) for row in batch if row.get("run_id")}
            payload = RoiRefinementCommand(detection_ids=tuple(detection_ids), options=dict(request.options)).to_payload()
            jobs.append(
                self.repository.create_job(
                    "roi_refinement",
                    project_id=project_id,
                    run_id=next(iter(runs)) if len(runs) == 1 else None,
                    asset_id=next(iter(assets)) if len(assets) == 1 else None,
                    priority=request.priority,
                    payload=payload,
                    summary=f"roi refinement queued for {len(detection_ids)} detections",
                    submitted_by_user_id=request.submitted_by_user_id,
                    submitted_by_username=request.submitted_by_username,
                )
            )
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
    def _batches(units: list[dict[str, Any]], max_units: int) -> list[list[dict[str, Any]]]:
        return [units[index:index + max_units] for index in range(0, len(units), max_units)]

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
