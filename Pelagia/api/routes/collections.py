from __future__ import annotations

try:
    from fastapi import APIRouter, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ._common import as_response, get_repository

    router = APIRouter(prefix="/collections", tags=["collections"])

    @router.get("")
    def list_collections(
        request: Request,
        collection: str | None = None,
        limit: int = 100,
    ) -> dict[str, list]:
        return {
            "collections": as_response(
                get_repository(request).list_collections(collection=collection, limit=limit)
            )
        }
else:
    router = None
