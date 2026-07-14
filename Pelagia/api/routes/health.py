from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover - route import is optional without FastAPI
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ._common import get_context, get_kvstore, get_repository, kvstore_status, postgres_ping

    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("")
    def get_health(request: Request) -> dict:
        context = get_context(request)
        return {
            "status": "ok",
            "postgres_configured": context.repository is not None,
            "kvstore_configured": bool(context._project_kvstores),
        }

    @router.get("/postgres")
    def get_postgres_health(request: Request) -> dict:
        try:
            return postgres_ping(get_repository(request))
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"healthy": False, "error": str(exc)},
            ) from exc

    @router.get("/kvstore")
    def get_kvstore_health(request: Request) -> dict:
        kvstore = get_kvstore(request)
        status = kvstore_status(kvstore)
        if not status.get("initialized"):
            return {"healthy": False, "status": status, "errors": ["KVStore is not initialized."]}
        return kvstore.check_health()
else:
    router = None
