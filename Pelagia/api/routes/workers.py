from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ._common import as_response, get_repository

    class ShutdownRequest(BaseModel):
        reason: str | None = None

    router = APIRouter(prefix="/workers", tags=["workers"])

    @router.get("")
    def list_workers(
        request: Request,
        status: str | None = None,
        capability: str | None = None,
        shutdown_requested: bool | None = None,
        limit: int = 100,
    ) -> dict[str, list]:
        return {
            "workers": as_response(
                get_repository(request).list_worker_sessions(
                    status=status,
                    capability=capability,
                    shutdown_requested=shutdown_requested,
                    limit=limit,
                )
            )
        }

    @router.get("/{worker_id}")
    def get_worker(request: Request, worker_id: str) -> dict:
        worker = get_repository(request).get_worker_session(worker_id)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"Worker {worker_id!r} was not found.")
        return {"worker": as_response(worker)}

    @router.post("/{worker_id}/shutdown")
    def request_worker_shutdown(
        request: Request,
        worker_id: str,
        body: ShutdownRequest | None = None,
    ) -> dict:
        worker = get_repository(request).request_worker_shutdown(
            worker_id,
            reason=None if body is None else body.reason,
        )
        if worker is None:
            raise HTTPException(status_code=404, detail=f"Worker {worker_id!r} was not found.")
        return {"worker": as_response(worker)}
else:
    router = None
