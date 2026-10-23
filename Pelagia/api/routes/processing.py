"""Backend-owned processing queue planning endpoints."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_write, scoped_project_id
    from ...services.context import AppContext
    from ...services.processing_queue import (
        PreprocessQueueRequest, ProcessingQueueService, ProcessingSeriesRequest as ProcessingSeriesServiceRequest,
        ProcessingSubmissionConflict, ProcessingSubmissionService,
    )
    from ._common import as_response, get_context

    class ProcessingQueueFilters(BaseModel):
        model_config = ConfigDict(extra="forbid")

        run_id: str | None = None
        asset_ids: list[str] = Field(default_factory=list)
        frame_ids: list[str] = Field(default_factory=list)
        collection: list[str] = Field(default_factory=list)
        preprocessing_status: list[str] = Field(default_factory=list)
        candidate_detection_status: list[str] = Field(default_factory=list)
        roi_refinement_status: list[str] = Field(default_factory=list)
        refinement_state: list[Literal["unrefined", "refined"]] = Field(default_factory=lambda: ["unrefined"])
        start_frame: int | None = None
        end_frame: int | None = None

    class ProcessingQueueRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        stage: Literal["preprocess_frames", "segment", "roi_refinement"]
        filters: ProcessingQueueFilters = Field(default_factory=ProcessingQueueFilters)
        options: dict[str, Any] = Field(default_factory=dict)
        priority: int | None = None
        dry_run: bool = False

    class ProcessingSeriesStep(BaseModel):
        model_config = ConfigDict(extra="forbid")
        stage: Literal["preprocess_frames", "segment", "roi_refinement"]
        filters: ProcessingQueueFilters = Field(default_factory=ProcessingQueueFilters)
        options: dict[str, Any] = Field(default_factory=dict)
        enabled: bool = True
        failure_policy: Literal["fail_fast", "continue"] | None = None

    class ProcessingSeriesRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        steps: list[ProcessingSeriesStep]
        targets: dict[str, Any] = Field(default_factory=dict)
        selection: dict[str, Any] = Field(default_factory=dict)
        preset_snapshot: dict[str, Any] = Field(default_factory=dict)
        failure_policy: Literal["fail_fast", "continue"] = "fail_fast"
        priority: int | None = None
        dry_run: bool = False

    class SeriesReasonRequest(BaseModel):
        reason: str | None = None

    class ProcessingSubmissionRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        request_id: UUID
        kind: Literal["queue", "series"]
        request: dict[str, Any]

    router = APIRouter(prefix="/processing", tags=["processing"])

    @router.post("/queue")
    def queue_processing(request: Request, body: ProcessingQueueRequest) -> dict:
        auth = require_project_write(request)
        try:
            filters = body.filters.model_dump() if hasattr(body.filters, "model_dump") else body.filters.dict()
            service = ProcessingQueueService(get_context(request))
            queue_request = PreprocessQueueRequest(
                filters=filters,
                options=body.options,
                priority=body.priority,
                dry_run=body.dry_run,
                submitted_by_user_id=auth.user_id,
                submitted_by_username=auth.username,
            )
            if body.stage == "preprocess_frames":
                result = service.queue_preprocess(queue_request, project_id=auth.project_id)
            elif body.stage == "segment":
                result = service.queue_segment(queue_request, project_id=auth.project_id)
            else:
                result = service.queue_roi_refinement(queue_request, project_id=auth.project_id)
        except (KeyError, ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(result)

    def _series_request_payload(body: ProcessingSeriesRequest) -> dict[str, Any]:
        """Return the persisted, resolved series input used by both endpoints."""
        selection = dict(body.selection or body.targets or {})
        common_filters = {
            "asset_ids": selection.get("asset_ids") or [],
            "frame_ids": selection.get("frame_ids") or [],
            "collection": selection.get("collections") or selection.get("collection") or [],
            "start_frame": selection.get("start_frame"),
            "end_frame": selection.get("end_frame"),
        }
        steps = []
        for step in body.steps:
            if not step.enabled:
                continue
            filters = step.filters.model_dump() if hasattr(step.filters, "model_dump") else step.filters.dict()
            filters = {**common_filters, **{key: value for key, value in filters.items() if value not in (None, [], {})}}
            steps.append({"stage": step.stage, "filters": filters, "options": step.options, "failure_policy": step.failure_policy})
        return {
            "steps": steps, "selection": selection, "preset_snapshot": body.preset_snapshot,
            "failure_policy": body.failure_policy, "priority": body.priority, "dry_run": body.dry_run,
        }

    @router.post("/submissions", status_code=202)
    def submit_processing(request: Request, body: ProcessingSubmissionRequest) -> dict:
        """Accept a criteria-based request without synchronously enumerating units."""
        auth = require_project_write(request)
        try:
            if body.kind == "queue":
                queue_body = ProcessingQueueRequest(**body.request)
                if queue_body.dry_run:
                    raise ValueError("dry_run is not supported for durable processing submissions.")
                persisted_request = queue_body.model_dump() if hasattr(queue_body, "model_dump") else queue_body.dict()
            else:
                series_body = ProcessingSeriesRequest(**body.request)
                if series_body.dry_run:
                    raise ValueError("dry_run is not supported for durable processing submissions.")
                persisted_request = _series_request_payload(series_body)
            submission = ProcessingSubmissionService(get_context(request)).submit(
                kind=body.kind, request=persisted_request, request_id=str(body.request_id), project_id=auth.project_id,
                submitted_by_user_id=auth.user_id, submitted_by_username=auth.username,
            )
        except ProcessingSubmissionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        reused_active = bool(submission.pop("_reused_active_submission", False))
        return {"submission": as_response(submission), "reused_active_submission": reused_active}

    @router.get("/submissions")
    def list_processing_submissions(request: Request, limit: int = 50) -> dict:
        repository = get_context(request).repository
        return {"submissions": as_response(repository.list_processing_submissions(
            project_id=scoped_project_id(request), limit=limit,
        ))}

    @router.get("/submissions/{submission_id}")
    def get_processing_submission(request: Request, submission_id: str) -> dict:
        submission = get_context(request).repository.get_processing_submission(
            submission_id, project_id=scoped_project_id(request),
        )
        if submission is None:
            raise HTTPException(status_code=404, detail=f"Processing submission {submission_id!r} was not found.")
        return {"submission": as_response(submission)}

    @router.post("/submissions/{submission_id}/cancel")
    def cancel_processing_submission(
        request: Request, submission_id: str, body: SeriesReasonRequest | None = None,
    ) -> dict:
        """Stop future planning pages and cooperatively cancel committed jobs."""
        auth = require_project_write(request)
        submission = get_context(request).repository.cancel_processing_submission(
            submission_id, project_id=auth.project_id,
            reason=None if body is None else body.reason,
        )
        if submission is None:
            raise HTTPException(status_code=404, detail=f"Processing submission {submission_id!r} was not found.")
        return {"submission": as_response(submission)}

    @router.post("/series")
    def create_processing_series(request: Request, body: ProcessingSeriesRequest) -> dict:
        auth = require_project_write(request)
        payload = _series_request_payload(body)
        try:
            result = ProcessingQueueService(get_context(request)).create_series(
                ProcessingSeriesServiceRequest(steps=tuple(payload["steps"]), selection=payload["selection"], preset_snapshot=body.preset_snapshot,
                                        failure_policy=body.failure_policy, priority=body.priority,
                                        dry_run=body.dry_run, submitted_by_user_id=auth.user_id, submitted_by_username=auth.username),
                project_id=auth.project_id,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.dry_run:
            return as_response(result)
        return as_response({"series": result, "dry_run": False})

    @router.get("/series")
    def list_processing_series(request: Request, limit: int = 100, offset: int = 0) -> dict:
        repository = get_context(request).repository
        return {"series": as_response(repository.list_processing_series(
            project_id=scoped_project_id(request), limit=limit, offset=offset,
        ))}

    @router.get("/series/{series_id}")
    def get_processing_series(request: Request, series_id: str) -> dict:
        series = get_context(request).repository.get_processing_series(series_id, project_id=scoped_project_id(request))
        if series is None:
            raise HTTPException(status_code=404, detail=f"Processing series {series_id!r} was not found.")
        return as_response(series)

    @router.get("/series/{series_id}/summary")
    def get_processing_series_summary(request: Request, series_id: str) -> dict:
        return get_processing_series(request, series_id)

    @router.get("/series/{series_id}/units")
    def get_processing_series_units(request: Request, series_id: str) -> dict:
        repository = get_context(request).repository
        if repository.get_processing_series(series_id, project_id=scoped_project_id(request)) is None:
            raise HTTPException(status_code=404, detail=f"Processing series {series_id!r} was not found.")
        return {"units": as_response(repository.list_processing_work_units(series_id, project_id=scoped_project_id(request)))}

    @router.post("/series/{series_id}/{action}")
    def control_processing_series(request: Request, series_id: str, action: Literal["pause", "resume", "cancel", "retry"], body: SeriesReasonRequest | None = None) -> dict:
        auth = require_project_write(request)
        repository = get_context(request).repository
        reason = None if body is None else body.reason
        # Resuming a series must also hand control back to the director: the
        # final job of a step may have completed while the series was paused.
        # The service performs that idempotent advancement after persistence.
        if action == "resume":
            series = ProcessingQueueService(get_context(request)).resume_series(
                series_id,
                project_id=auth.project_id,
                reason=reason,
            )
            if series is None:
                raise HTTPException(status_code=404, detail=f"Processing series {series_id!r} was not found.")
            return as_response(series)
        method = getattr(repository, f"{action}_processing_series")
        series = method(series_id, project_id=auth.project_id, reason=reason)
        if series is None:
            raise HTTPException(status_code=404, detail=f"Processing series {series_id!r} was not found.")
        return as_response(series)
else:
    router = None
