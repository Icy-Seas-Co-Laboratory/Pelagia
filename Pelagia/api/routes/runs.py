from __future__ import annotations

from datetime import datetime, timezone
import uuid

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..schemas import AssetsListResponse, JobsListResponse, RunDetailResponse, RunsListResponse
    from ..auth import require_project_write, scoped_project_id
    from ...domain import PlannedRun, RunManifest
    from ._common import as_response, get_repository

    def _bounded_limit(limit: int | None) -> int:
        return min(max(1, 100 if limit is None else limit), 1000)

    def _bounded_offset(offset: int | None) -> int:
        return max(0, 0 if offset is None else offset)

    router = APIRouter(prefix="/runs", tags=["runs"])

    class CreateRunRequest(BaseModel):
        run_key: str
        instrument: str = "telemetry"
        source_path: str = ""
        source_type: str = "telemetry"
        metadata: dict = Field(default_factory=dict)

    @router.post("", response_model=RunDetailResponse, status_code=201)
    def create_run(request: Request, body: CreateRunRequest) -> dict:
        auth = require_project_write(request)
        run_key = body.run_key.strip()
        if not run_key:
            raise HTTPException(status_code=422, detail="run_key must not be blank.")
        run_id = str(uuid.uuid4())
        registration = get_repository(request).register_planned_run(
            PlannedRun(
                manifest=RunManifest(
                    run_id=run_id,
                    run_key=run_key,
                    instrument=body.instrument.strip() or "telemetry",
                    source_path=body.source_path.strip(),
                    source_type=body.source_type.strip() or "telemetry",
                    created_at=datetime.now(timezone.utc),
                    metadata={
                        **dict(body.metadata or {}),
                        "project_id": str(auth.project_id),
                        "api_endpoint": "POST /runs",
                    },
                )
            ),
            project_id=str(auth.project_id),
        )
        return {"run": as_response(registration.get("run") or {"id": run_id, "run_key": run_key})}

    @router.get("", response_model=RunsListResponse)
    def list_runs(
        request: Request,
        limit: int = 100,
        offset: int = 0,
        collection: str | None = None,
        run_key: str | None = None,
        instrument: str | None = None,
        source_type: str | None = None,
        status: str | None = None,
        source_path: str | None = None,
    ) -> dict[str, list]:
        return {
            "runs": as_response(
                get_repository(request).list_runs(
                    limit=limit,
                    offset=offset,
                    project_id=scoped_project_id(request),
                    collection=collection,
                    run_key=run_key,
                    instrument=instrument,
                    source_type=source_type,
                    status=status,
                    source_path=source_path,
                )
            )
        }

    @router.get("/{run_id}", response_model=RunDetailResponse)
    def get_run(request: Request, run_id: str) -> dict:
        run = get_repository(request).get_run(run_id, project_id=scoped_project_id(request))
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return {"run": as_response(run)}

    @router.get("/{run_id}/assets", response_model=AssetsListResponse)
    def list_run_assets(
        request: Request,
        run_id: str,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        path: str | None = None,
        checksum: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        media_count: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, list]:
        return {
            "assets": as_response(
                get_repository(request).list_assets(
                    run_id=run_id,
                    project_id=scoped_project_id(request),
                    collection=collection,
                    kind=kind,
                    filename=filename,
                    path=path,
                    checksum=checksum,
                    min_size_bytes=min_size_bytes,
                    max_size_bytes=max_size_bytes,
                    media_count=media_count,
                    limit=limit,
                    offset=offset,
                )
            )
        }

    @router.get("/{run_id}/jobs", response_model=JobsListResponse)
    def list_run_jobs(
        request: Request,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
        include_details: bool = False,
    ) -> dict[str, list]:
        return {
            "jobs": as_response(
                get_repository(request).list_jobs(
                    run_id=run_id,
                    project_id=scoped_project_id(request),
                    limit=_bounded_limit(limit),
                    offset=_bounded_offset(offset),
                    include_details=include_details,
                )
            )
        }

    @router.post("/{run_id}/cancel", response_model=RunDetailResponse)
    def cancel_run(request: Request, run_id: str) -> dict:
        auth = require_project_write(request)
        run = get_repository(request).cancel_run(run_id, project_id=auth.project_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return {"run": as_response(run)}

    @router.post("/{run_id}/reconcile", response_model=RunDetailResponse)
    def reconcile_run(request: Request, run_id: str) -> dict:
        auth = require_project_write(request)
        run = get_repository(request).reconcile_run(run_id, project_id=auth.project_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return {"run": as_response(run)}
else:
    router = None
