from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import scoped_project_id
    from ._common import as_response, get_repository

    router = APIRouter(prefix="/models", tags=["models"])

    @router.get("")
    def list_models(
        request: Request,
        model_key: str | None = None,
        model_name: str | None = None,
        version: str | None = None,
        task: str | None = None,
        artifact_uri: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, list]:
        return {
            "models": as_response(
                get_repository(request).list_models(
                    project_id=scoped_project_id(request),
                    model_key=model_key,
                    model_name=model_name,
                    version=version,
                    task=task,
                    artifact_uri=artifact_uri,
                    limit=limit,
                    offset=max(0, offset),
                )
            )
        }

    @router.get("/{model_id}")
    def get_model(request: Request, model_id: str) -> dict:
        model = get_repository(request).get_model(model_id, project_id=scoped_project_id(request))
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} was not found.")
        return {"model": as_response(model)}
else:
    router = None
