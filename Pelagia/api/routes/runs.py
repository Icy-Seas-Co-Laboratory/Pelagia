from __future__ import annotations

try:
    from fastapi import APIRouter
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    router = APIRouter(prefix="/runs", tags=["runs"])

    @router.get("")
    def list_runs() -> dict[str, list]:
        return {"runs": []}
else:
    router = None
