from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ._common import as_response, get_repository

    router = APIRouter(prefix="/runs", tags=["runs"])

    @router.get("")
    def list_runs(
        request: Request,
        limit: int = 100,
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
                    collection=collection,
                    run_key=run_key,
                    instrument=instrument,
                    source_type=source_type,
                    status=status,
                    source_path=source_path,
                )
            )
        }

    @router.get("/{run_id}")
    def get_run(request: Request, run_id: str) -> dict:
        run = get_repository(request).get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return {"run": as_response(run)}

    @router.get("/{run_id}/assets")
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
                )
            )
        }

    @router.get("/{run_id}/jobs")
    def list_run_jobs(request: Request, run_id: str, limit: int = 100) -> dict[str, list]:
        return {"jobs": as_response(get_repository(request).list_jobs(run_id=run_id, limit=limit))}

    @router.post("/{run_id}/cancel")
    def cancel_run(request: Request, run_id: str) -> dict:
        run = get_repository(request).cancel_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return {"run": as_response(run)}

    @router.post("/{run_id}/reconcile")
    def reconcile_run(request: Request, run_id: str) -> dict:
        run = get_repository(request).reconcile_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found.")
        return {"run": as_response(run)}
else:
    router = None
