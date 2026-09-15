"""Project-scoped, asynchronous scientific export bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import UUID

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_read
    from ...domain import PipelineStage
    from ...services.exports.contracts import ExportProduct, ExportRequest, TabularFormat
    from ...services.exports.snapshot import prepare_export_snapshot
    from ...services.pipeline import PipelineService
    from ._common import as_response, get_context, get_repository

    class ExportBundleRequest(BaseModel):
        """A durable request for one or more export products."""

        model_config = ConfigDict(extra="forbid")
        products: list[Literal[
            "raw_roi_statistics", "binned_roi_statistics", "roi_evidence", "telemetry",
        ]] = Field(min_length=1)
        formats: dict[str, Literal["xlsx", "json", "sqlite"]] = Field(default_factory=dict)
        asset_ids: list[str] = Field(default_factory=list)
        run_ids: list[str] = Field(default_factory=list)
        telemetry_source_ids: list[str] = Field(default_factory=list)
        roi_stage: Literal["refined"] = "refined"
        filters: dict[str, Any] = Field(default_factory=dict)

        def as_export_request(self) -> ExportRequest:
            return ExportRequest(
                products=tuple(ExportProduct(value) for value in self.products),
                formats=dict(self.formats),
                asset_ids=tuple(self.asset_ids),
                run_ids=tuple(self.run_ids),
                telemetry_source_ids=tuple(self.telemetry_source_ids),
                roi_stage=self.roi_stage,
                filters=dict(self.filters),
            )

    router = APIRouter(prefix="/exports", tags=["exports"])

    def _artifact_response(artifact: dict[str, Any], *, repository: Any | None = None, project_id: str | None = None) -> dict[str, Any]:
        value = dict(artifact)
        value.pop("artifact_path", None)
        if value.get("status") == "succeeded":
            value["download_url"] = f"/exports/{value['id']}/download"
        list_jobs = getattr(repository, "list_jobs", None)
        if callable(list_jobs) and value.get("job_id"):
            jobs = list_jobs(
                project_id=project_id,
                job_ids=[str(value["job_id"])],
                limit=1,
                include_details=False,
                include_progress=True,
            )
            job = next((item for item in jobs if str(item.get("id")) == str(value["job_id"])), None)
            if job is not None:
                value["job"] = as_response(job)
        return as_response(value)

    def _uuid_or_none(value: str | None) -> str | None:
        try:
            return str(UUID(str(value)))
        except (TypeError, ValueError):
            return None

    def _prepare_snapshot(repository: Any, *, project_id: str, export_request: ExportRequest, config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        prepared = getattr(repository, "prepare_export_snapshot", None)
        if callable(prepared):
            return prepared(project_id=project_id, request=export_request.as_dict(), config=config)
        return prepare_export_snapshot(repository, project_id=project_id, request=export_request.as_dict(), config=config)

    @router.get("/options")
    def export_options(request: Request) -> dict[str, Any]:
        require_project_read(request)
        return {
            "products": [product.value for product in ExportProduct],
            "formats": [file_format.value for file_format in TabularFormat],
            "roi_stages": ["refined"],
            "bundle_format_version": "1.0",
        }

    @router.post("", status_code=202)
    def create_export(request: Request, body: ExportBundleRequest) -> dict[str, Any]:
        auth = require_project_read(request)
        assert auth.project_id is not None
        try:
            export_request = body.as_export_request()
            repository = get_repository(request)
            artifact = repository.create_export_artifact(
                project_id=str(auth.project_id),
                request=export_request.as_dict(),
                requested_by_user_id=_uuid_or_none(auth.user_id),
                requested_by_username=auth.username,
                estimate={"status": "pending", "message": "Calculated by the export worker before writing."},
            )
            job = PipelineService(get_context(request)).queue(
                PipelineStage.EXPORT_BUNDLE,
                project_id=str(auth.project_id),
                payload={"export_id": str(artifact["id"])},
                summary=f"Export bundle {artifact['id']}",
                submitted_by_user_id=auth.user_id,
                submitted_by_username=auth.username,
            )
            artifact = repository.attach_export_artifact_job(
                str(artifact["id"]), str(job["id"]), project_id=str(auth.project_id),
            ) or artifact
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"export": _artifact_response(artifact, repository=repository, project_id=str(auth.project_id)), "job": as_response(job), "estimate": artifact.get("estimate")}

    @router.post("/estimate")
    def estimate_export(request: Request, body: ExportBundleRequest) -> dict[str, Any]:
        auth = require_project_read(request)
        assert auth.project_id is not None
        try:
            export_request = body.as_export_request()
            _snapshot, estimate = _prepare_snapshot(get_repository(request), project_id=str(auth.project_id), export_request=export_request, config=None)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"estimate": estimate}

    @router.get("")
    def list_exports(
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        auth = require_project_read(request)
        assert auth.project_id is not None
        repository = get_repository(request)
        artifacts = repository.list_export_artifacts(
            project_id=str(auth.project_id), limit=limit, offset=offset,
        )
        job_ids = [str(item["job_id"]) for item in artifacts if item.get("job_id")]
        jobs_by_id: dict[str, dict[str, Any]] = {}
        if job_ids:
            jobs_by_id = {
                str(job["id"]): job
                for job in repository.list_jobs(
                    project_id=str(auth.project_id),
                    job_ids=job_ids,
                    limit=len(job_ids),
                    include_details=False,
                    include_progress=True,
                )
            }
        responses = []
        for artifact in artifacts:
            response = _artifact_response(artifact)
            job = jobs_by_id.get(str(artifact.get("job_id")))
            if job is not None:
                response["job"] = as_response(job)
            responses.append(response)
        return {"exports": responses, "limit": limit, "offset": offset}

    @router.get("/{export_id}")
    def get_export(request: Request, export_id: str) -> dict[str, Any]:
        auth = require_project_read(request)
        assert auth.project_id is not None
        artifact = get_repository(request).get_export_artifact(export_id, project_id=str(auth.project_id))
        if artifact is None:
            raise HTTPException(status_code=404, detail="Export bundle was not found.")
        return {"export": _artifact_response(artifact, repository=get_repository(request), project_id=str(auth.project_id))}

    @router.get("/{export_id}/download")
    def download_export(request: Request, export_id: str):
        auth = require_project_read(request)
        assert auth.project_id is not None
        artifact = get_repository(request).get_export_artifact(export_id, project_id=str(auth.project_id))
        if artifact is None:
            raise HTTPException(status_code=404, detail="Export bundle was not found.")
        if artifact.get("status") != "succeeded" or not artifact.get("artifact_path"):
            raise HTTPException(status_code=409, detail="Export bundle is not ready for download.")
        path = Path(str(artifact["artifact_path"]))
        if not path.is_file():
            raise HTTPException(status_code=410, detail="Export bundle artifact is no longer available.")
        return FileResponse(path, media_type="application/zip", filename="export.zip")
else:
    router = None
