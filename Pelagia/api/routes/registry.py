from __future__ import annotations

import sqlite3
import shutil
import base64
import binascii
import hashlib
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import APIRouter, HTTPException, Query, Request, Response
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from oracle_data_contracts.datasets import DatasetSchemaError

    from ..auth import require_project_read, require_project_write
    from ...services.registry_transfer import RegistryTransferError
    from ...services.registry_workspace import RegistryWorkspaceService
    from ...services.pipeline import PipelineService
    from ...services.taxonomy import default_registry_vocabulary, default_registry_vocabulary_summary
    from ._common import as_response, get_context, get_repository
    from ...domain import PipelineStage

    router = APIRouter(prefix="/registry", tags=["registry"])

    class OpenDatasetRequest(BaseModel):
        path: str
        migrate: bool = True
        backup: bool = True
        thaw_frozen: bool = True

    class ExportDatasetRequest(BaseModel):
        path: str
        replace_source: bool = False

    class ImportDatasetRequest(BaseModel):
        path: str
        workspace_id: str | None = None
        expected_source_sha256: str | None = None
        include_labels: bool = True
        include_evidence: bool = True
        duplicate_policy: Literal["skip", "merge"] = "skip"
        duplicate_actions: dict[str, Literal["skip", "merge"]] = Field(default_factory=dict)

    class WorkspaceCreateRequest(BaseModel):
        name: str = Field(min_length=1, max_length=160)
        dataset_type: Literal["classification", "mask_refinement"] = "classification"
        title: str | None = Field(default=None, max_length=300)
        description: str | None = Field(default=None, max_length=4000)
        activate: bool = True

    class TransferFilters(BaseModel):
        model_config = {"extra": "forbid"}
        annotation_states: list[Literal["labeled", "unlabeled"]] | None = None
        review_states: list[Literal["verified", "unreviewed", "needs_review", "rejected", "no_annotation"]] | None = None
        label_ids: list[str] | None = None
        search: str | None = None
        evidence_source: str | None = None
        ml_state: Literal["all", "with", "without"] = "all"
        confidence_min: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
        confidence_max: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
        knn_similarity_min: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
        knn_agreement_min: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
        prototype_similarity_min: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
        ml_disagreement: bool = False
        target_tag_ids: list[str] | None = None
        image_tag_ids: list[str] | None = None
        target_tags_untagged: bool = False
        image_tags_untagged: bool = False
        bbox_area_min: float | None = Field(default=None, ge=0, allow_inf_nan=False)
        bbox_area_max: float | None = Field(default=None, ge=0, allow_inf_nan=False)
        bbox_longest_side_min: float | None = Field(default=None, ge=0, allow_inf_nan=False)
        bbox_longest_side_max: float | None = Field(default=None, ge=0, allow_inf_nan=False)
        bbox_aspect_ratio_min: float | None = Field(default=None, ge=0, allow_inf_nan=False)
        bbox_aspect_ratio_max: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    class WorkspaceTransferPreviewRequest(BaseModel):
        source_workspace_id: str
        destination_workspace_id: str
        mode: Literal["copy", "move"]
        item_ids: list[str] | None = None
        filters: TransferFilters | None = None

    DuplicateAction = Literal["skip", "overwrite", "keep_destination_remove_source"]

    class WorkspaceTransferExecuteRequest(BaseModel):
        source_workspace_id: str
        destination_workspace_id: str
        mode: Literal["copy", "move"]
        item_ids: list[str] = Field(min_length=1)
        expected_preview_token: str = Field(min_length=1)
        duplicate_policy: DuplicateAction = "skip"
        duplicate_actions: dict[str, DuplicateAction] = Field(default_factory=dict)

    class LabelCreate(BaseModel):
        name: str = Field(min_length=1, max_length=160)
        display_name: str | None = None
        parent_label_id: str | None = None
        rank: str | None = None
        description: str | None = None

    class LabelUpdate(BaseModel):
        name: str | None = Field(default=None, min_length=1, max_length=160)
        display_name: str | None = None
        description: str | None = None
        deprecated: bool | None = None

    class LabelReassignRequest(BaseModel):
        source_label_id: str
        target_label_id: str
        deprecate_source: bool = True

    class AnnotationRequest(BaseModel):
        item_ids: list[str] = Field(min_length=1)
        label_id: str

    class RemoveAnnotationRequest(BaseModel):
        item_ids: list[str] = Field(min_length=1)

    class ReviewRequest(BaseModel):
        item_ids: list[str] = Field(min_length=1)
        decision: Literal["verified", "rejected", "needs_review"]

    class TagCreate(BaseModel):
        name: str = Field(min_length=1, max_length=160)
        scope: Literal["target_tags", "image_tags"]
        parent_tag_id: str | None = None
        exclusive_within_parent: bool = False

    class TagAnnotationRequest(BaseModel):
        item_ids: list[str] = Field(min_length=1)
        tag_id: str
        assigned: bool = True

    class SpatialBoxRequest(BaseModel):
        x: int
        y: int
        w: int = Field(gt=0)
        h: int = Field(gt=0)

    class GeometryUpdateRequest(BaseModel):
        bbox: SpatialBoxRequest
        expected_updated_at: str = Field(min_length=1)

    class MaskRevisionRequest(BaseModel):
        mask_png_base64: str = Field(min_length=1)
        # Required even when null: null means the caller observed no current revision.
        expected_current_annotation_id: str | None
        method: str = Field(default="manual-erase", min_length=1, max_length=100)
        parameters: dict[str, Any] = Field(default_factory=dict)
        notes: str | None = Field(default=None, max_length=4000)

    def service(request: Request, *, write: bool = False) -> RegistryWorkspaceService:
        auth = require_project_write(request) if write else require_project_read(request)
        return RegistryWorkspaceService(get_repository(request), str(auth.project_id), auth.username)

    def allowed_roots(request: Request) -> list[Path]:
        browser = request.app.state.context.config.file_browser
        roots = [browser.root_path_import_dir, *browser.allowed_root_paths]
        return list(dict.fromkeys(Path(value).expanduser().resolve() for value in roots))

    def upload_paths(request: Request) -> tuple[Path, Path]:
        auth = require_project_read(request)
        root = (allowed_roots(request)[0] / "registry-uploads").resolve()
        scope = hashlib.sha256(f"{auth.project_id}\x00{auth.username}".encode()).hexdigest()
        return root, root / scope

    def resolve_path(request: Request, value: str, *, must_exist: bool = False) -> Path:
        path = Path(value).expanduser().resolve()
        roots = allowed_roots(request)
        if roots and not any(path == root or path.is_relative_to(root) for root in roots):
            raise HTTPException(403, "Path is outside the configured Registry roots")
        upload_root, own_uploads = upload_paths(request)
        if path.is_relative_to(upload_root) and path != upload_root and not path.is_relative_to(own_uploads):
            raise HTTPException(403, "Uploaded datasets belong to another Registry user or project")
        if must_exist and not path.is_file():
            raise HTTPException(404, "SQLite dataset was not found")
        return path

    def translated_error(exc: Exception) -> HTTPException:
        text = str(exc)
        return HTTPException(409 if "already loaded" in text.lower() or "changed" in text.lower() else 400, text)

    @router.get("/health")
    def health(request: Request):
        return {"status": "ok", "dataset_open": service(request).active_workspace() is not None}

    @router.get("/files")
    def files(request: Request, path: str | None = None):
        require_project_read(request)
        roots = allowed_roots(request)
        current = resolve_path(request, path, must_exist=False) if path else (roots[0] if roots else Path.cwd())
        if not current.is_dir(): raise HTTPException(400, "Path is not a directory")
        entries = []
        upload_root, own_uploads = upload_paths(request)
        for entry in sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold())):
            resolved = entry.resolve()
            if resolved.is_relative_to(upload_root) and resolved != upload_root and not resolved.is_relative_to(own_uploads):
                continue
            if not entry.is_dir() and entry.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}: continue
            stat = entry.stat()
            entries.append({"name": entry.name, "path": str(entry), "is_directory": entry.is_dir(),
                            "size": None if entry.is_dir() else stat.st_size,
                            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
        parent = current.parent if current.parent != current and (not roots or any(current.parent == root or current.parent.is_relative_to(root) for root in roots)) else None
        return {"current_path": str(current), "parent_path": str(parent) if parent else None,
                "root_path": str(roots[0]) if roots else str(current), "roots": [str(root) for root in roots], "entries": entries}

    @router.post("/dataset/upload", status_code=201)
    async def upload_dataset(request: Request, filename: str = Query("dataset.sqlite")):
        require_project_write(request)
        if Path(filename).suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
            raise HTTPException(422, "Upload a .sqlite, .sqlite3, or .db dataset")
        _, directory = upload_paths(request)
        directory.mkdir(parents=True, exist_ok=True)
        destination = resolve_path(request, str(directory / f"{uuid.uuid4().hex}.sqlite"))
        size = 0
        try:
            with destination.open("xb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 2 * 1024**3:
                        raise HTTPException(413, "SQLite uploads are limited to 2 GiB; use server browsing for larger files")
                    output.write(chunk)
            with destination.open("rb") as source:
                if source.read(16) != b"SQLite format 3\x00":
                    raise HTTPException(400, "The uploaded file is not a SQLite database")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return {"path": str(destination), "size_bytes": size}

    @router.get("/dataset/download")
    def download_dataset(request: Request):
        from ...services.registry_download import snapshot_sqlite_workspace

        auth = require_project_read(request)
        temporary = tempfile.NamedTemporaryFile(prefix="pelagia-download-", suffix=".sqlite", delete=False)
        destination = Path(temporary.name)
        temporary.close()
        try:
            snapshot_sqlite_workspace(get_repository(request), destination,
                                      project_id=str(auth.project_id), owner_username=auth.username)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            if isinstance(exc, RegistryTransferError):
                raise translated_error(exc) from exc
            raise
        return FileResponse(destination, media_type="application/vnd.sqlite3", filename="pelagia-registry.sqlite",
                            headers={"Cache-Control": "no-store"},
                            background=BackgroundTask(destination.unlink, missing_ok=True))

    @router.post("/dataset/open", status_code=202)
    def open_dataset(request: Request, body: OpenDatasetRequest):
        auth = require_project_write(request); source = resolve_path(request, body.path, must_exist=True)
        job = PipelineService(get_context(request)).queue(
            PipelineStage.REGISTRY_LOAD, project_id=str(auth.project_id),
            payload={"source_path": str(source), "source_size_bytes": source.stat().st_size,
                     "owner_username": auth.username},
            summary=f"Load Registry dataset {source.name}", submitted_by_user_id=auth.user_id,
            submitted_by_username=auth.username,
        )
        return {"job": as_response(job)}

    @router.get("/dataset")
    def dataset(request: Request): return as_response(service(request).summary())

    @router.get("/dataset/details")
    def dataset_details(request: Request): return as_response(service(request).details())

    @router.get("/workspaces")
    def workspaces(request: Request):
        return {"workspaces": as_response(service(request).list_workspaces())}

    @router.post("/workspaces/{workspace_id}/activate")
    def activate_workspace(request: Request, workspace_id: str):
        try:
            return as_response(service(request, write=True).activate_workspace(workspace_id))
        except KeyError as exc:
            raise HTTPException(404, "Registry workspace was not found") from exc

    @router.post("/workspaces", status_code=201)
    def create_workspace(request: Request, body: WorkspaceCreateRequest):
        from ...services.registry_workspace_transfer import create_blank_workspace

        auth = require_project_write(request)
        try:
            return as_response(create_blank_workspace(get_repository(request), project_id=str(auth.project_id),
                owner_username=auth.username, **body.model_dump()))
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc

    @router.post("/transfers/preview")
    def preview_transfer(request: Request, body: WorkspaceTransferPreviewRequest):
        from ...services.registry_workspace_transfer import preview_workspace_transfer

        auth = require_project_write(request)
        if body.item_ids is not None and body.filters is not None:
            raise HTTPException(422, "Choose explicit ROI IDs or filters for the transfer selection")
        try:
            item_ids = body.item_ids
            if item_ids is None:
                item_ids = service(request).matching_item_ids(body.source_workspace_id,
                    **(body.filters.model_dump(exclude_none=True) if body.filters else {}))
            result = preview_workspace_transfer(get_repository(request), project_id=str(auth.project_id),
                owner_username=auth.username, source_workspace_id=body.source_workspace_id,
                destination_workspace_id=body.destination_workspace_id, item_ids=item_ids, mode=body.mode)
            return as_response({**result, "item_ids": item_ids,
                "source_workspace_id": body.source_workspace_id,
                "destination_workspace_id": body.destination_workspace_id})
        except KeyError as exc:
            raise HTTPException(404, "Registry workspace or ROI was not found") from exc
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc

    @router.post("/transfers/execute", status_code=202)
    def execute_transfer(request: Request, body: WorkspaceTransferExecuteRequest):
        from ...services.registry_workspace_transfer import preview_workspace_transfer, validate_transfer_actions

        auth = require_project_write(request)
        if body.mode == "copy" and "keep_destination_remove_source" in [body.duplicate_policy, *body.duplicate_actions.values()]:
            raise HTTPException(422, "Removing a source duplicate is only available for moves")
        try:
            preview = preview_workspace_transfer(get_repository(request), project_id=str(auth.project_id),
                owner_username=auth.username, source_workspace_id=body.source_workspace_id,
                destination_workspace_id=body.destination_workspace_id, item_ids=body.item_ids, mode=body.mode)
            if preview["preview_token"] != body.expected_preview_token:
                raise HTTPException(409, "Registry workspace contents changed; preview the transfer again")
            validate_transfer_actions(preview, body.mode, body.duplicate_policy, body.duplicate_actions)
        except KeyError as exc:
            raise HTTPException(404, "Registry workspace or ROI was not found") from exc
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc
        job = PipelineService(get_context(request)).queue(
            PipelineStage.REGISTRY_WORKSPACE_TRANSFER, project_id=str(auth.project_id),
            payload={**body.model_dump(), "owner_username": auth.username},
            summary=f"{body.mode.title()} {len(body.item_ids)} Registry ROIs",
            submitted_by_user_id=auth.user_id, submitted_by_username=auth.username,
        )
        return {"job": as_response(job)}

    @router.post("/dataset/export", status_code=202)
    def export_dataset(request: Request, body: ExportDatasetRequest):
        auth = require_project_write(request); workspace = service(request, write=True).active_workspace()
        if not workspace: raise HTTPException(409, "No Registry dataset is loaded")
        destination = resolve_path(request, body.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        required_bytes = max(int(workspace["source_size_bytes"] * 1.2), workspace["source_size_bytes"] + 16 * 1024 * 1024)
        if shutil.disk_usage(destination.parent).free < required_bytes:
            raise HTTPException(507, f"Export requires at least {required_bytes} bytes of free space")
        job = PipelineService(get_context(request)).queue(
            PipelineStage.REGISTRY_EXPORT, project_id=str(auth.project_id),
            payload={"workspace_id": str(workspace["id"]), "destination_path": str(destination),
                     "replace_source": body.replace_source, "owner_username": auth.username,
                     "estimated_bytes": workspace["source_size_bytes"]},
            summary=f"Export Registry dataset to {destination.name}", submitted_by_user_id=auth.user_id,
            submitted_by_username=auth.username,
        )
        return {"job": as_response(job)}

    @router.delete("/dataset")
    def purge_dataset(request: Request): return service(request, write=True).purge()

    @router.get("/items")
    def items(request: Request, limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0),
              annotation_state: str = "all", review: str = "all", label_id: list[str] | None = Query(None),
              search: str | None = None, sort: str = "original", random_seed: str = "registry",
              evidence_source: str | None = None, ml_state: str = "all", confidence_min: float | None = Query(None, ge=0, le=1, allow_inf_nan=False),
              confidence_max: float | None = Query(None, ge=0, le=1, allow_inf_nan=False), knn_similarity_min: float | None = None,
              knn_agreement_min: float | None = None, prototype_similarity_min: float | None = None,
              ml_disagreement: bool = False,
              annotation_states: list[str] | None = Query(None), review_states: list[str] | None = Query(None),
              target_tag_id: list[str] | None = Query(None), image_tag_id: list[str] | None = Query(None),
              target_tags_untagged: bool = False, image_tags_untagged: bool = False,
              bbox_area_min: float | None = Query(None, ge=0, allow_inf_nan=False), bbox_area_max: float | None = Query(None, ge=0, allow_inf_nan=False),
              bbox_longest_side_min: float | None = Query(None, ge=0, allow_inf_nan=False), bbox_longest_side_max: float | None = Query(None, ge=0, allow_inf_nan=False),
              bbox_aspect_ratio_min: float | None = Query(None, ge=0, allow_inf_nan=False), bbox_aspect_ratio_max: float | None = Query(None, ge=0, allow_inf_nan=False)):
        try:
            return as_response(service(request).list_items(limit=limit, offset=offset, annotation_state=annotation_state,
            review=review, label_ids=label_id, search=search, sort=sort, evidence_source=evidence_source,
            ml_state=ml_state, confidence_min=confidence_min, confidence_max=confidence_max,
            knn_similarity_min=knn_similarity_min, knn_agreement_min=knn_agreement_min,
            prototype_similarity_min=prototype_similarity_min, ml_disagreement=ml_disagreement,
            annotation_states=[] if annotation_states == ["__none__"] else annotation_states,
            review_states=[] if review_states == ["__none__"] else review_states,
            target_tag_ids=target_tag_id, image_tag_ids=image_tag_id,
            target_tags_untagged=target_tags_untagged, image_tags_untagged=image_tags_untagged,
            bbox_area_min=bbox_area_min, bbox_area_max=bbox_area_max,
            bbox_longest_side_min=bbox_longest_side_min, bbox_longest_side_max=bbox_longest_side_max,
            bbox_aspect_ratio_min=bbox_aspect_ratio_min, bbox_aspect_ratio_max=bbox_aspect_ratio_max))
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc

    @router.get("/items/facets")
    def item_filter_facets(request: Request, evidence_source: str | None = None):
        try:
            return as_response(service(request).filter_facets(evidence_source=evidence_source))
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc

    @router.get("/items/{item_id}")
    def item(request: Request, item_id: str):
        try: return as_response(service(request).item_detail(item_id))
        except KeyError as exc: raise HTTPException(404, "Item not found") from exc

    @router.get("/items/{item_id}/image")
    def image(request: Request, item_id: str):
        try: payload, media = service(request).image(item_id)
        except KeyError as exc: raise HTTPException(404, "Image not found") from exc
        return Response(payload, media_type=media, headers={"Cache-Control": "private, max-age=3600"})

    @router.get("/items/{item_id}/thumbnail")
    def thumbnail(request: Request, item_id: str, size: int = Query(256, ge=48, le=1024)):
        try: payload, media = service(request).image(item_id, size)
        except KeyError as exc: raise HTTPException(404, "Image not found") from exc
        return Response(payload, media_type=media, headers={"Cache-Control": "private, max-age=3600"})

    @router.patch("/items/{item_id}/geometry")
    def update_item_geometry(request: Request, item_id: str, body: GeometryUpdateRequest):
        try:
            return as_response(service(request, write=True).update_geometry(
                item_id, body.bbox.model_dump(), expected_updated_at=body.expected_updated_at,
            ))
        except KeyError as exc:
            raise HTTPException(404, "Item not found") from exc
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc

    @router.get("/items/{item_id}/masks")
    def masks(request: Request, item_id: str):
        try:
            return as_response(service(request).mask_layers(item_id))
        except KeyError as exc:
            raise HTTPException(404, "Item not found") from exc
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc

    @router.get("/items/{item_id}/masks/{layer}")
    def mask_image(request: Request, item_id: str, layer: Literal["candidate", "current"]):
        try:
            payload, media = service(request).mask_image(item_id, layer)
        except KeyError as exc:
            raise HTTPException(404, "Mask layer not found") from exc
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc
        return Response(payload, media_type=media, headers={"Cache-Control": "private, max-age=3600"})

    @router.post("/items/{item_id}/masks", status_code=201)
    def save_mask(request: Request, item_id: str, body: MaskRevisionRequest):
        try:
            try:
                payload = base64.b64decode(body.mask_png_base64, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise HTTPException(422, "mask_png_base64 must be valid base64") from exc
            return as_response(service(request, write=True).save_mask_revision(
                item_id, payload, expected_current_annotation_id=body.expected_current_annotation_id,
                method=body.method, parameters=body.parameters, notes=body.notes,
            ))
        except KeyError as exc:
            raise HTTPException(404, "Item not found") from exc
        except RegistryTransferError as exc:
            raise translated_error(exc) from exc

    @router.get("/labels")
    def labels(request: Request, include_deprecated: bool = True): return as_response(service(request).labels(include_deprecated))

    @router.post("/labels", status_code=201)
    def add_label(request: Request, body: LabelCreate): return as_response(service(request, write=True).add_label(body.model_dump()))

    @router.patch("/labels/{label_id}")
    def update_label(request: Request, label_id: str, body: LabelUpdate):
        try: return as_response(service(request, write=True).update_label(label_id, body.model_dump(exclude_unset=True)))
        except KeyError as exc: raise HTTPException(404, "Label not found") from exc

    @router.post("/labels/reassign")
    def reassign_label(request: Request, body: LabelReassignRequest):
        value = service(request, write=True)
        try:
            return as_response(value.reassign_label(body.source_label_id, body.target_label_id, value.owner_username,
                                                     deprecate_source=body.deprecate_source))
        except RegistryTransferError as exc: raise translated_error(exc) from exc

    @router.post("/annotations/bulk")
    def annotate(request: Request, body: AnnotationRequest):
        value = service(request, write=True); return {"annotations": as_response(value.assign(body.item_ids, body.label_id, value.owner_username))}

    @router.post("/annotations/remove/bulk")
    def remove_annotations(request: Request, body: RemoveAnnotationRequest):
        value = service(request, write=True); return {"annotations": as_response(value.assign(body.item_ids, "", value.owner_username, remove=True))}

    @router.post("/reviews/bulk")
    def reviews(request: Request, body: ReviewRequest):
        value = service(request, write=True); return {"reviews": as_response(value.review(body.item_ids, body.decision, value.owner_username))}

    @router.get("/tags")
    def tags(request: Request): return as_response(service(request).tags())

    @router.post("/tags", status_code=201)
    def add_tag(request: Request, body: TagCreate): return as_response(service(request, write=True).add_tag(body.model_dump()))

    @router.post("/item-tags/bulk")
    def annotate_tags(request: Request, body: TagAnnotationRequest):
        value = service(request, write=True); return {"annotations": as_response(value.set_tag(body.item_ids, body.tag_id, body.assigned, value.owner_username))}

    @router.get("/evidence/sources")
    def evidence_sources(request: Request): return as_response(service(request).evidence_sources())

    @router.get("/evidence/items/{item_id}")
    def item_evidence(request: Request, item_id: str, source_key: str | None = None):
        return as_response(service(request).item_evidence(item_id, source_key))

    @router.get("/evidence/neighbors/{item_id}")
    def evidence_neighbors(request: Request, item_id: str, source_key: str):
        return as_response(service(request).neighbor_items(item_id, source_key))

    @router.get("/similarity/{item_id}")
    def similarity(request: Request, item_id: str, source_key: str, limit: int = Query(500, ge=1, le=1000),
                   offset: int = Query(0, ge=0), minimum: float = Query(-1.0, ge=-1.0, le=1.0)):
        try: return as_response(service(request).similar_items(item_id, source_key, limit=limit, offset=offset, minimum=minimum))
        except RegistryTransferError as exc: raise translated_error(exc) from exc

    @router.get("/vocabulary")
    def vocabulary(request: Request): require_project_read(request); return default_registry_vocabulary()

    @router.get("/vocabularies")
    def vocabularies(request: Request): require_project_read(request); return [default_registry_vocabulary_summary()]

    @router.post("/vocabularies/reload")
    def reload_vocabularies(request: Request): require_project_read(request); return [default_registry_vocabulary_summary()]

    @router.get("/vocabularies/{vocabulary_key}")
    def installed_vocabulary(request: Request, vocabulary_key: str):
        require_project_read(request); value = default_registry_vocabulary()
        if vocabulary_key != value["catalog_key"]: raise HTTPException(404, "Vocabulary not found")
        return value

    @router.post("/vocabularies/{vocabulary_key}/taxonomy/{concept_id}")
    def materialize_label(request: Request, vocabulary_key: str, concept_id: str):
        vocabulary = default_registry_vocabulary()
        if vocabulary_key != vocabulary["catalog_key"]: raise HTTPException(404, "Vocabulary not found")
        concept = next((node for node in vocabulary["taxonomy"]["nodes"] if node["id"] == concept_id), None)
        if not concept: raise HTTPException(404, "Vocabulary concept not found")
        if not concept.get("selectable", True): raise HTTPException(400, "Organizational taxonomy nodes cannot be assigned")
        data = {"name": concept["name"], "display_name": concept.get("display_name") or concept["name"],
                "rank": concept.get("rank"), "description": f"Preferred concept from {vocabulary['vocabulary']['name']}.",
                "metadata": {"registry": {"standard_concept_id": concept_id, "standard_concept": concept,
                  "vocabulary": vocabulary["vocabulary"], "sources": vocabulary["sources"]}}}
        return as_response(service(request, write=True).add_label(data))

    @router.post("/undo")
    def undo(request: Request):
        value = service(request, write=True)
        try: return as_response(value.undo(value.owner_username))
        except RegistryTransferError as exc: raise translated_error(exc) from exc

    @router.post("/import/preview")
    def import_preview(request: Request, body: ImportDatasetRequest):
        from ...services.registry_import import preview_sqlite_import

        auth = require_project_read(request)
        source = resolve_path(request, body.path, must_exist=True)
        try:
            return as_response(preview_sqlite_import(get_repository(request), source,
                project_id=str(auth.project_id), owner_username=auth.username, workspace_id=body.workspace_id))
        except (RegistryTransferError, DatasetSchemaError, sqlite3.Error) as exc:
            raise translated_error(exc) from exc

    @router.post("/import/execute")
    def import_execute(request: Request, body: ImportDatasetRequest):
        from ...services.registry_import import import_sqlite_into_workspace

        auth = require_project_write(request)
        source = resolve_path(request, body.path, must_exist=True)
        try:
            return as_response(import_sqlite_into_workspace(get_repository(request), source,
                project_id=str(auth.project_id), owner_username=auth.username, workspace_id=body.workspace_id,
                expected_source_sha256=body.expected_source_sha256, include_labels=body.include_labels,
                include_evidence=body.include_evidence, duplicate_policy=body.duplicate_policy,
                duplicate_actions=body.duplicate_actions, actor=auth.username))
        except (RegistryTransferError, DatasetSchemaError, sqlite3.Error) as exc:
            raise translated_error(exc) from exc
