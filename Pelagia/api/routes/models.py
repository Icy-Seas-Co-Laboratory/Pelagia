from __future__ import annotations

try:
    from fastapi import APIRouter
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    router = APIRouter(prefix="/models", tags=["models"])

    @router.get("")
    def list_models() -> dict[str, list]:
        return {"models": []}
else:
    router = None
