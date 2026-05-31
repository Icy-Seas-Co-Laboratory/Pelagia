from __future__ import annotations

try:
    from fastapi import APIRouter, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ._common import as_response, get_context, get_kvstore, kvstore_status

    router = APIRouter(prefix="/kvstore", tags=["kvstore"])

    def _store_status(kvstore) -> dict | None:
        if kvstore is None or not hasattr(kvstore, "status"):
            return None
        try:
            return kvstore.status()
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _store_health(kvstore, status: dict | None) -> dict:
        if kvstore is None:
            return {"healthy": False, "errors": ["KVStore is not configured."]}
        if not hasattr(kvstore, "check_health"):
            return {"healthy": False, "errors": ["KVStore health check is not available."]}
        if status is not None and not status.get("initialized", False):
            return {"healthy": False, "errors": ["KVStore is not initialized."]}
        try:
            return kvstore.check_health()
        except Exception as exc:
            return {"healthy": False, "errors": [str(exc)]}

    @router.get("")
    def get_store(request: Request) -> dict:
        context = get_context(request)
        status = _store_status(context.kvstore)
        return as_response(
            {
                "root_path": context.config.kvstore.root_path,
                "configured_hash_algorithm": context.config.kvstore.hash_algorithm,
                "configured_prefix_length": context.config.kvstore.prefix_length,
                "status": status,
                "health": _store_health(context.kvstore, status),
            }
        )

    @router.get("/status")
    def get_store_status(request: Request) -> dict:
        return kvstore_status(get_kvstore(request))

    @router.get("/health")
    def get_store_health(request: Request) -> dict:
        kvstore = get_kvstore(request)
        if not kvstore.initialized:
            return {
                "healthy": False,
                "errors": ["KVStore is not initialized."],
                "status": kvstore_status(kvstore),
            }
        return as_response(kvstore.check_health())
else:
    router = None
