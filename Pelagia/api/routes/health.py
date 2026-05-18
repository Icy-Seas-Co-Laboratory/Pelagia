from __future__ import annotations

try:
    from fastapi import APIRouter
except ImportError:  # pragma: no cover - route import is optional without FastAPI
    APIRouter = None  # type: ignore


if APIRouter is not None:
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("")
    def get_health() -> dict[str, str]:
        return {"status": "ok"}
else:
    router = None
