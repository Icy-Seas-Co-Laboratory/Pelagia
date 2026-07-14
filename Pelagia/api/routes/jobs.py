from __future__ import annotations

from typing import Any
from uuid import UUID

from ...domain import JobStatus, PipelineStage

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..schemas import JobDetailResponse, JobsClearResponse, JobsListResponse, JobsSummaryResponse
    from ..auth import require_project_write, scoped_project_id
    from ...services.pipeline import PipelineService
    from ._common import as_response, get_context, get_repository

    def _bounded_limit(limit: int | None) -> int:
        return min(max(1, 100 if limit is None else limit), 1000)

    def _bounded_offset(offset: int | None) -> int:
        return max(0, 0 if offset is None else offset)

    def _query_values(values: list[str] | None) -> list[str]:
        if not values:
            return []
        resolved: list[str] = []
        for value in values:
            for item in str(value).split(","):
                stripped = item.strip()
                if stripped:
                    resolved.append(stripped)
        return resolved

    def _enum_values(values: list[str] | None, enum_type: type[JobStatus] | type[PipelineStage], label: str) -> list[str]:
        allowed = {item.value for item in enum_type}
        resolved = _query_values(values)
        invalid = [value for value in resolved if value not in allowed]
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid {label}: {', '.join(invalid)}. Expected one of: {', '.join(sorted(allowed))}.",
            )
        return resolved

    def _uuid_values(values: list[str] | None) -> list[str]:
        resolved = _query_values(values)
        invalid: list[str] = []
        for value in resolved:
            try:
                UUID(value)
            except ValueError:
                invalid.append(value)
        if invalid:
            raise HTTPException(status_code=422, detail=f"Invalid job id(s): {', '.join(invalid)}.")
        return resolved

    class CreateJobRequest(BaseModel):
        stage: PipelineStage
        run_id: str | None = None
        asset_id: str | None = None
        status: JobStatus = JobStatus.QUEUED
        priority: int | None = None
        max_attempts: int | None = None
        payload: dict[str, Any] = Field(default_factory=dict)
        depends_on: list[str] = Field(default_factory=list)
        summary: str | None = None

    class ReasonRequest(BaseModel):
        reason: str | None = None

    class PriorityRequest(BaseModel):
        priority: int
        reason: str | None = None

    class ClearJobsRequest(BaseModel):
        run_id: str | None = None
        asset_id: str | None = None
        status: list[str] = Field(default_factory=list)
        stage: list[str] = Field(default_factory=list)
        ids: list[str] = Field(default_factory=list)
        worker_id: str | None = None
        reason: str | None = None
        mode: str = "cancel"
        dry_run: bool = False

    router = APIRouter(prefix="/jobs", tags=["jobs"])

    @router.get("", response_model=JobsListResponse, response_model_exclude_none=True)
    def list_jobs(
        request: Request,
        run_id: str | None = None,
        asset_id: str | None = None,
        status: list[str] | None = Query(None),
        stage: list[str] | None = Query(None),
        ids: list[str] | None = Query(None),
        worker_id: str | None = None,
        limit: int | None = 100,
        offset: int = 0,
        cursor: str | None = None,
        include_details: bool = False,
        include_progress: bool = True,
        include_payload: bool = False,
        include_result: bool = False,
        sort: str = "created_at",
        direction: str = "desc",
    ) -> dict[str, list]:
        repository = get_repository(request)
        resolved_limit = _bounded_limit(limit)
        statuses = _enum_values(status, JobStatus, "status")
        stages = _enum_values(stage, PipelineStage, "stage")
        job_ids = _uuid_values(ids)
        jobs = repository.list_jobs(
            run_id=run_id,
            asset_id=asset_id,
            project_id=scoped_project_id(request),
            statuses=statuses,
            stages=stages,
            job_ids=job_ids,
            worker_id=worker_id,
            limit=resolved_limit,
            offset=_bounded_offset(offset),
            cursor=cursor,
            include_details=include_details,
            include_progress=include_progress,
            include_payload=include_payload,
            include_result=include_result,
            sort=sort,
            direction=direction,
        )
        return {"jobs": as_response(jobs)}

    @router.get("/summary", response_model=JobsSummaryResponse, response_model_exclude_none=True)
    def summarize_jobs(
        request: Request,
        run_id: str | None = None,
        asset_id: str | None = None,
        status: list[str] | None = Query(None),
        stage: list[str] | None = Query(None),
        ids: list[str] | None = Query(None),
        worker_id: str | None = None,
        include_recent: bool = False,
        recent_limit: int = 5,
    ) -> dict[str, Any]:
        repository = get_repository(request)
        statuses = _enum_values(status, JobStatus, "status")
        stages = _enum_values(stage, PipelineStage, "stage")
        job_ids = _uuid_values(ids)
        return as_response(
            repository.summarize_jobs(
                project_id=scoped_project_id(request),
                run_id=run_id,
                asset_id=asset_id,
                statuses=statuses,
                stages=stages,
                job_ids=job_ids,
                worker_id=worker_id,
                include_recent=include_recent,
                recent_limit=_bounded_limit(recent_limit),
            )
        )

    @router.post("", response_model=JobDetailResponse, response_model_exclude_none=True)
    def create_job(request: Request, body: CreateJobRequest) -> dict:
        auth = require_project_write(request)
        try:
            job = PipelineService(get_context(request)).queue(
                body.stage,
                project_id=auth.project_id,
                run_id=body.run_id,
                asset_id=body.asset_id,
                status=body.status,
                priority=body.priority,
                max_attempts=body.max_attempts,
                payload=body.payload,
                depends_on=body.depends_on,
                summary=body.summary,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"job": as_response(job)}

    @router.get("/events")
    def list_job_events(
        request: Request,
        after_id: int | None = None,
        run_id: str | None = None,
        job_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, list]:
        repository = get_repository(request)
        events = repository.list_job_events(
            project_id=scoped_project_id(request),
            after_id=after_id,
            run_id=run_id,
            job_id=job_id,
            limit=limit,
            offset=_bounded_offset(offset),
        )
        return {"events": as_response(events)}

    @router.post("/clear", response_model=JobsClearResponse, response_model_exclude_none=True)
    def clear_jobs(request: Request, body: ClearJobsRequest | None = None) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        body = body or ClearJobsRequest()
        statuses = _enum_values(body.status, JobStatus, "status")
        stages = _enum_values(body.stage, PipelineStage, "stage")
        job_ids = _uuid_values(body.ids)
        if body.mode not in {"cancel", "delete"}:
            raise HTTPException(status_code=422, detail="mode must be one of: cancel, delete.")
        if body.mode == "delete":
            active_statuses = {JobStatus.QUEUED.value, JobStatus.LEASED.value, JobStatus.WORKING.value, JobStatus.PAUSED.value}
            if not statuses:
                raise HTTPException(status_code=422, detail="Delete mode requires explicit terminal status filters.")
            if active_statuses.intersection(statuses):
                raise HTTPException(status_code=422, detail="Delete mode can only clear terminal job statuses.")
            result = repository.delete_jobs(
                project_id=auth.project_id,
                run_id=body.run_id,
                asset_id=body.asset_id,
                statuses=statuses,
                stages=stages,
                job_ids=job_ids,
                worker_id=body.worker_id,
                reason=body.reason,
                dry_run=body.dry_run,
            )
            return as_response(result)
        result = repository.cancel_jobs(
            project_id=auth.project_id,
            run_id=body.run_id,
            asset_id=body.asset_id,
            statuses=statuses,
            stages=stages,
            job_ids=job_ids,
            worker_id=body.worker_id,
            reason=body.reason,
            dry_run=body.dry_run,
        )
        return as_response(result)

    @router.get("/{job_id}", response_model=JobDetailResponse, response_model_exclude_none=True)
    def get_job(request: Request, job_id: str) -> dict:
        repository = get_repository(request)
        job = repository.get_job(job_id, project_id=scoped_project_id(request))
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} was not found.")
        return {"job": as_response(job)}

    @router.post("/{job_id}/pause", response_model=JobDetailResponse, response_model_exclude_none=True)
    def pause_job(request: Request, job_id: str, body: ReasonRequest | None = None) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        job = repository.pause_job(job_id, reason=None if body is None else body.reason, project_id=auth.project_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} was not found.")
        return {"job": as_response(job)}

    @router.post("/{job_id}/resume", response_model=JobDetailResponse, response_model_exclude_none=True)
    def resume_job(request: Request, job_id: str, body: ReasonRequest | None = None) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        job = repository.resume_job(job_id, reason=None if body is None else body.reason, project_id=auth.project_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Paused job {job_id!r} was not found.")
        return {"job": as_response(job)}

    @router.post("/{job_id}/retry", response_model=JobDetailResponse, response_model_exclude_none=True)
    def retry_job(request: Request, job_id: str) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        job = repository.retry_job(job_id, project_id=auth.project_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Failed, dead-lettered, or cancelled job {job_id!r} was not found.",
            )
        return {"job": as_response(job)}

    @router.post("/{job_id}/priority", response_model=JobDetailResponse, response_model_exclude_none=True)
    def set_job_priority(request: Request, job_id: str, body: PriorityRequest) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        job = repository.set_job_priority(job_id, body.priority, reason=body.reason, project_id=auth.project_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} was not found.")
        return {"job": as_response(job)}
else:
    router = None
