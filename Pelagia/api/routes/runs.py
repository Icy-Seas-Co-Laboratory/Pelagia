from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..schemas import AssetsListResponse, JobsListResponse, RunDetailResponse, RunsListResponse
    from ._common import as_response, get_repository

    def _bounded_limit(limit: int | None) -> int:
        return min(max(1, 100 if limit is None else limit), 1000)

    def _bounded_offset(offset: int | None) -> int:
        return max(0, 0 if offset is None else offset)

    router = APIRouter(prefix="/runs", tags=["runs"])

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
        run = get_repository(request).get_run(run_id)
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
                    limit=_bounded_limit(limit),
                    offset=_bounded_offset(offset),
                    include_details=include_details,
                )
            )
        }

    @router.post("/{run_id}/cancel", response_model=RunDetailResponse)
    def cancel_run(request: Request, run_id: str) -> dict:
        run = get_repository(request).cancel_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return {"run": as_response(run)}

    @router.post("/{run_id}/reconcile", response_model=RunDetailResponse)
    def reconcile_run(request: Request, run_id: str) -> dict:
        run = get_repository(request).reconcile_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return {"run": as_response(run)}
else:
    router = None
