from __future__ import annotations

try:
    from fastapi import APIRouter
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    router = APIRouter(prefix="/jobs", tags=["jobs"])

    @router.get("")
    def list_jobs() -> dict[str, list]:
        return {"jobs": []}
else:
    router = None
