"""Backend-owned processing queue planning endpoints."""

from __future__ import annotations

from typing import Any, Literal

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_write, scoped_project_id
    from ...services.context import AppContext
    from ...services.processing_queue import PreprocessQueueRequest, ProcessingQueueService, ProcessingSeriesRequest as ProcessingSeriesServiceRequest
    from ._common import as_response, get_context

    class ProcessingQueueFilters(BaseModel):
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
        failure_policy: Literal["fail_fast", "continue", "stop_series", "retry_failed"] = "fail_fast"
        priority: int | None = None
        dry_run: bool = False

    class SeriesReasonRequest(BaseModel):
        reason: str | None = None

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
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(result)

    @router.post("/series")
    def create_processing_series(request: Request, body: ProcessingSeriesRequest) -> dict:
        auth = require_project_write(request)
        if not body.preset_snapshot:
            raise HTTPException(status_code=422, detail="A preset snapshot is required to create a processing series.")
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
        failure_policy = "fail_fast" if body.failure_policy in {"fail_fast", "stop_series"} else "continue"
        try:
            result = ProcessingQueueService(get_context(request)).create_series(
                ProcessingSeriesServiceRequest(steps=tuple(steps), selection=selection, preset_snapshot=body.preset_snapshot,
                                        failure_policy=failure_policy, priority=body.priority,
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
        method = getattr(repository, f"{action}_processing_series")
        series = method(series_id, project_id=auth.project_id, reason=None if body is None else body.reason)
        if series is None:
            raise HTTPException(status_code=404, detail=f"Processing series {series_id!r} was not found.")
        if action in {"resume", "retry"}:
            ProcessingQueueService(get_context(request)).advance_series(series_id, project_id=auth.project_id)
        return as_response(series)
else:
    router = None
