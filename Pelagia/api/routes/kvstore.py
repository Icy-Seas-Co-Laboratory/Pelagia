from __future__ import annotations

try:
    from fastapi import APIRouter, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_read
    from ._common import as_response, get_context, kvstore_status

    router = APIRouter(prefix="/kvstore", tags=["kvstore"])

    def _store_status(kvstore, *, deep: bool = False) -> dict | None:
        if kvstore is None or not hasattr(kvstore, "status"):
            return None
        try:
            return kvstore.status(deep=deep)
        except TypeError:
            return kvstore.status()
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _store_health(kvstore, status: dict | None, *, deep: bool = False) -> dict:
        if kvstore is None:
            return {"healthy": False, "errors": ["Cold SkerryKV storage is not configured."]}
        if not hasattr(kvstore, "check_health"):
            return {
                "healthy": bool(status is not None and status.get("initialized", False)),
                "errors": [] if status is not None and status.get("initialized", False) else ["SkerryKV status is unavailable."],
            }
        if status is not None and not status.get("initialized", False):
            return {"healthy": False, "errors": ["KVStore is not initialized."]}
        if not deep:
            return {
                "healthy": True,
                "errors": [],
                "warnings": ["Deep KVStore health check was not run."],
                "deep": False,
            }
        try:
            return kvstore.check_health()
        except Exception as exc:
            return {"healthy": False, "errors": [str(exc)]}

    @router.get("")
    def get_store(
        request: Request,
        deep_status: bool = False,
        include_health: bool = False,
    ) -> dict:
        auth = require_project_read(request)
        context = get_context(request).for_project(auth.project_id)
        status = _store_status(context.kvstore, deep=deep_status)
        return as_response(
            {
                "backend": "skerrykv",
                "endpoint": context.config.cold_storage.endpoint,
                "timeout_seconds": context.config.cold_storage.timeout_seconds,
                "status": status,
                "health": _store_health(context.kvstore, status, deep=include_health),
            }
        )

    @router.get("/status")
    def get_store_status(request: Request, deep: bool = False) -> dict:
        auth = require_project_read(request)
        kvstore = get_context(request).kvstore_for_project(auth.project_id, initialize=False)
        if kvstore is None:
            return {"configured": False, "initialized": False}
        return kvstore_status(kvstore, deep=deep)

    @router.get("/health")
    def get_store_health(request: Request) -> dict:
        auth = require_project_read(request)
        kvstore = get_context(request).kvstore_for_project(auth.project_id, initialize=False)
        if kvstore is None:
            return {"healthy": False, "errors": ["Project cold SkerryKV storage is not configured."]}
        if not kvstore.initialized:
            return {
                "healthy": False,
                "errors": ["SkerryKV storage is not initialized."],
                "status": kvstore_status(kvstore),
            }
        status = kvstore_status(kvstore, deep=True)
        return as_response({"healthy": bool(status.get("initialized", False)), "status": status})
else:
    router = None
