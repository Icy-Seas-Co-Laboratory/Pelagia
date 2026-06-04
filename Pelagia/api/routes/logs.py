from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ._common import as_response, get_repository

    def _bounded_limit(limit: int | None) -> int:
        return min(max(1, 100 if limit is None else limit), 1000)

    class CreateLogRequest(BaseModel):
        event_type: str
        message: str | None = None
        level: str = "info"
        logger: str = "pelagia.api"
        run_id: str | None = None
        asset_id: str | None = None
        job_id: str | None = None
        worker_id: str | None = None
        request_id: str | None = None
        duration_ms: float | None = None
        payload: dict[str, Any] = Field(default_factory=dict)

    router = APIRouter(prefix="/logs", tags=["logs"])

    @router.get("")
    def list_logs(
        request: Request,
        after_id: int | None = None,
        before_id: int | None = None,
        level: str | None = None,
        event_type: str | None = None,
        logger: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        request_id: str | None = None,
        limit: int | None = 100,
    ) -> dict[str, list]:
        repository = get_repository(request)
        logs = repository.list_logs(
            after_id=after_id,
            before_id=before_id,
            level=level,
            event_type=event_type,
            logger=logger,
            run_id=run_id,
            asset_id=asset_id,
            job_id=job_id,
            worker_id=worker_id,
            request_id=request_id,
            limit=_bounded_limit(limit),
        )
        return {"logs": as_response(logs)}

    @router.post("")
    def create_log(request: Request, body: CreateLogRequest) -> dict:
        repository = get_repository(request)
        row = repository.append_log(
            event_type=body.event_type,
            message=body.message,
            level=body.level,
            logger=body.logger,
            run_id=body.run_id,
            asset_id=body.asset_id,
            job_id=body.job_id,
            worker_id=body.worker_id,
            request_id=body.request_id,
            duration_ms=body.duration_ms,
            payload=body.payload,
        )
        return {"log": as_response(row)}
else:
    router = None
