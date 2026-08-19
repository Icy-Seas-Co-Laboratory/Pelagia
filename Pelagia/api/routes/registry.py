from __future__ import annotations

import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import APIRouter, HTTPException, Query, Request, Response
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
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

    def service(request: Request, *, write: bool = False) -> RegistryWorkspaceService:
        auth = require_project_write(request) if write else require_project_read(request)
        return RegistryWorkspaceService(get_repository(request), str(auth.project_id), auth.username)

    def allowed_roots(request: Request) -> list[Path]:
        browser = request.app.state.context.config.file_browser
        roots = [browser.root_path_import_dir, *browser.allowed_root_paths]
        return list(dict.fromkeys(Path(value).expanduser().resolve() for value in roots))

    def resolve_path(request: Request, value: str, *, must_exist: bool = False) -> Path:
        path = Path(value).expanduser().resolve()
        roots = allowed_roots(request)
        if roots and not any(path == root or path.is_relative_to(root) for root in roots):
            raise HTTPException(403, "Path is outside the configured Registry roots")
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
        roots = allowed_roots(request)
        current = resolve_path(request, path, must_exist=False) if path else (roots[0] if roots else Path.cwd())
        if not current.is_dir(): raise HTTPException(400, "Path is not a directory")
        entries = []
        for entry in sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold())):
            if not entry.is_dir() and entry.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}: continue
            stat = entry.stat()
            entries.append({"name": entry.name, "path": str(entry), "is_directory": entry.is_dir(),
                            "size": None if entry.is_dir() else stat.st_size,
                            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
        parent = current.parent if current.parent != current and (not roots or any(current.parent == root or current.parent.is_relative_to(root) for root in roots)) else None
        return {"current_path": str(current), "parent_path": str(parent) if parent else None,
                "root_path": str(roots[0]) if roots else str(current), "roots": [str(root) for root in roots], "entries": entries}

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
              evidence_source: str | None = None, ml_state: str = "all", confidence_min: float | None = None,
              confidence_max: float | None = None, knn_similarity_min: float | None = None,
              knn_agreement_min: float | None = None, prototype_similarity_min: float | None = None,
              ml_disagreement: bool = False):
        return as_response(service(request).list_items(limit=limit, offset=offset, annotation_state=annotation_state,
            review=review, label_ids=label_id, search=search, sort=sort, evidence_source=evidence_source,
            ml_state=ml_state, confidence_min=confidence_min, confidence_max=confidence_max,
            knn_similarity_min=knn_similarity_min, knn_agreement_min=knn_agreement_min,
            prototype_similarity_min=prototype_similarity_min, ml_disagreement=ml_disagreement))

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
    def import_preview(request: Request, body: dict[str, Any]):
        source = resolve_path(request, str(body.get("path", "")), must_exist=True)
        with sqlite3.connect(source) as connection:
            connection.row_factory = sqlite3.Row
            dataset = connection.execute("SELECT * FROM dataset WHERE singleton=1").fetchone()
            count = connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0]
        return {"source": {"path": str(source), "dataset_id": dataset["dataset_id"], "name": dataset["name"],
          "dataset_type": dataset["dataset_type"], "lifecycle": dataset["lifecycle"], "schema_version": "unknown", "size_bytes": source.stat().st_size},
          "total_items": count, "new_items": count, "duplicate_count": 0, "duplicates": [],
          "labels": {"available": True, "annotation_count": 0}, "descriptors": {"available": True, "annotation_count": 0},
          "evidence": {"available": True, "registry_count": 0, "legacy_count": 0}}

    @router.post("/import/execute")
    def import_execute(request: Request):
        require_project_write(request)
        raise HTTPException(501, "Merging a second SQLite revision into a loaded PostgreSQL workspace is not yet available")
