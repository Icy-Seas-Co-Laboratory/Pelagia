"""Persistent live-sandbox frame endpoints."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import scoped_project_id
    from ._common import as_response, frame_summary, get_context, get_repository

    router = APIRouter(prefix="/live", tags=["live"])

    def _delete_unreferenced_payload(context: Any, key: str) -> dict[str, Any]:
        repository = context.repository
        if repository is not None and hasattr(repository, "count_frame_payload_references"):
            if repository.count_frame_payload_references(key) > 0:
                return {"key": key, "deleted": False, "referenced": True, "missing": False}
        try:
            context.kvstore.key_delete(key)
            return {"key": key, "deleted": True, "referenced": False, "missing": False}
        except KeyError:
            return {"key": key, "deleted": False, "referenced": False, "missing": True}

    @router.get("/sandbox")
    def list_live_sandbox_frames(request: Request, source_frame_id: str | None = None, operation: str | None = None, limit: int = 100, offset: int = 0) -> dict:
        if limit < 1:
            raise HTTPException(status_code=422, detail="limit must be >= 1.")
        rows = get_repository(request).list_live_frame_copies(source_frame_id=source_frame_id, operation=operation, project_id=scoped_project_id(request), limit=limit, offset=offset)
        return as_response({"sandbox_frames": [frame_summary(row) for row in rows], "limit": limit, "offset": max(0, int(offset)), "count": len(rows)})

    @router.delete("/sandbox/{sandbox_frame_id}")
    def delete_live_sandbox_frame(request: Request, sandbox_frame_id: str) -> dict:
        project_id = scoped_project_id(request)
        context = get_context(request).for_project(project_id)
        result = get_repository(request).delete_live_frame_copy(sandbox_frame_id, project_id=project_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Live sandbox frame {sandbox_frame_id!r} was not found.")
        deleted = [_delete_unreferenced_payload(context, str(key)) for key in result.get("unreferenced_kvstore_keys", [])]
        return as_response({"status": "deleted", "sandbox_frame_id": sandbox_frame_id, "frame": frame_summary(result["frame"]), "generated_kvstore_keys": result.get("generated_kvstore_keys", []), "deleted_kvstore_keys": deleted})
else:
    router = None
