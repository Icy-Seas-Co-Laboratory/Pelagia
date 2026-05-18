from __future__ import annotations

try:
    from fastapi import APIRouter
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    router = APIRouter(prefix="/assets", tags=["assets"])

    @router.get("")
    def list_assets() -> dict[str, list]:
        return {"assets": []}
else:
    router = None
