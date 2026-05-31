from __future__ import annotations

from typing import Any

from ...domain import JobStatus, PipelineStage

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ._common import as_response, get_repository

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

    router = APIRouter(prefix="/jobs", tags=["jobs"])

    @router.get("")
    def list_jobs(
        request: Request,
        run_id: str | None = None,
        asset_id: str | None = None,
        status: JobStatus | None = None,
        stage: PipelineStage | None = None,
        worker_id: str | None = None,
        limit: int | None = 100,
        cursor: str | None = None,
    ) -> dict[str, list]:
        repository = get_repository(request)
        jobs = repository.list_jobs(
            run_id=run_id,
            asset_id=asset_id,
            status=None if status is None else status.value,
            stage=None if stage is None else stage.value,
            worker_id=worker_id,
            limit=limit,
            cursor=cursor,
        )
        return {"jobs": as_response(jobs)}

    @router.post("")
    def create_job(request: Request, body: CreateJobRequest) -> dict:
        repository = get_repository(request)
        job = repository.create_job(
            body.stage,
            run_id=body.run_id,
            asset_id=body.asset_id,
            status=body.status,
            priority=body.priority,
            max_attempts=body.max_attempts,
            payload=body.payload,
            depends_on=body.depends_on,
            summary=body.summary,
        )
        return {"job": as_response(job)}

    @router.get("/events")
    def list_job_events(
        request: Request,
        after_id: int | None = None,
        run_id: str | None = None,
        job_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, list]:
        repository = get_repository(request)
        events = repository.list_job_events(
            after_id=after_id,
            run_id=run_id,
            job_id=job_id,
            limit=limit,
        )
        return {"events": as_response(events)}

    @router.get("/{job_id}")
    def get_job(request: Request, job_id: str) -> dict:
        repository = get_repository(request)
        job = repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} was not found.")
        return {"job": as_response(job)}

    @router.post("/{job_id}/pause")
    def pause_job(request: Request, job_id: str, body: ReasonRequest | None = None) -> dict:
        repository = get_repository(request)
        job = repository.pause_job(job_id, reason=None if body is None else body.reason)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} was not found.")
        return {"job": as_response(job)}

    @router.post("/{job_id}/resume")
    def resume_job(request: Request, job_id: str, body: ReasonRequest | None = None) -> dict:
        repository = get_repository(request)
        job = repository.resume_job(job_id, reason=None if body is None else body.reason)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Paused job {job_id!r} was not found.")
        return {"job": as_response(job)}

    @router.post("/{job_id}/retry")
    def retry_job(request: Request, job_id: str) -> dict:
        repository = get_repository(request)
        job = repository.retry_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Failed, dead-lettered, or cancelled job {job_id!r} was not found.",
            )
        return {"job": as_response(job)}

    @router.post("/{job_id}/priority")
    def set_job_priority(request: Request, job_id: str, body: PriorityRequest) -> dict:
        repository = get_repository(request)
        job = repository.set_job_priority(job_id, body.priority, reason=body.reason)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} was not found.")
        return {"job": as_response(job)}
else:
    router = None
