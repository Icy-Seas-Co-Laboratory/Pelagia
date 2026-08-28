from __future__ import annotations

import shutil
import uuid
from hashlib import sha256
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_read, require_project_write
    from ...domain import PipelineStage
    from ...processing.oracle_client import OracleInferenceClient, OracleInferenceError
    from ...services.pipeline import PipelineService
    from ...services.feature_space import (
        FeatureSpaceError,
        FeatureSpaceService,
        parse_feature_space_source,
    )
    from ...services.registry_generation import preview_registry_dataset
    from ...services.telemetry import parse_telemetry_filters
    from ...services.taxonomy import default_taxonomy_dictionary
    from ._common import as_response, get_context, get_repository

    class LabelCreate(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str = Field(min_length=1, max_length=160)
        display_name: str | None = None
        stable_concept_id: str | None = None
        parent_label_id: str | None = None
        metadata: dict = Field(default_factory=dict)

    class AnnotationRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        roi_ids: list[str] = Field(min_length=1)
        label_id: str
        suggested_by_evidence_id: str | None = None
        notes: str | None = None

    class ReviewRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        roi_ids: list[str] = Field(min_length=1)
        decision: str
        notes: str | None = None

    class AnnotationRemovalRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        roi_ids: list[str] = Field(min_length=1)
        notes: str | None = None

    class ClassificationTargetSelectionRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        asset_ids: list[str] = Field(default_factory=list)
        collections: list[str] = Field(default_factory=list)
        annotation_state: Literal["all", "labeled", "unlabeled"] = "all"
        review_state: Literal[
            "all", "unreviewed", "verified", "rejected", "needs_review"
        ] = "all"
        evidence_state: Literal[
            "all",
            "missing_model",
            "available_model",
            "missing_any",
            "available_any",
            "disagreement",
        ] = "missing_model"
        label_id: str | None = None
        label_source: Literal["any", "human", "prediction"] = "any"
        min_area: float | None = Field(default=None, ge=0)
        max_area: float | None = Field(default=None, ge=0)
        search: str | None = Field(default=None, max_length=500)

    class ClassificationJobRequest(BaseModel):
        model_config = ConfigDict(extra="forbid", protected_namespaces=())
        roi_ids: list[str] = Field(default_factory=list)
        model_ref: str | None = None
        evidence_kind: Literal["classification", "clustering"] = "classification"
        selection: ClassificationTargetSelectionRequest | None = None
        priority: int | None = None

    class ClassificationPreviewRequest(BaseModel):
        model_config = ConfigDict(extra="forbid", protected_namespaces=())
        roi_ids: list[str] = Field(default_factory=list)
        model_ref: str | None = None
        evidence_kind: Literal["classification", "clustering"] = "classification"
        selection: ClassificationTargetSelectionRequest | None = None

    class FeatureSpaceAnalysisRequest(BaseModel):
        """Parameters for one ephemeral, reproducible UMAP/HDBSCAN analysis."""

        model_config = ConfigDict(extra="forbid")
        source_key: str = Field(min_length=3, max_length=200)
        min_cluster_size: int = Field(default=5, ge=2, le=1000)
        min_samples: int | None = Field(default=None, ge=1, le=1000)
        cluster_selection_epsilon: float = Field(default=0.0, ge=0.0, le=10.0)
        force: bool = False

    class RegistryDatasetSelectionRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        asset_ids: list[str] = Field(default_factory=list)
        annotation_state: Literal["all", "labeled", "unlabeled"] = "all"
        review_state: Literal[
            "all", "unreviewed", "verified", "rejected", "needs_review"
        ] = "all"
        evidence_state: Literal["all", "available", "missing", "disagreement"] = "all"
        min_area: float | None = Field(default=None, ge=0)
        max_area: float | None = Field(default=None, ge=0)

    class RegistryDatasetPreviewRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        selection: RegistryDatasetSelectionRequest = Field(default_factory=RegistryDatasetSelectionRequest)
        subsample_ratio: int = Field(default=1, ge=1, le=1000)

    class RegistryDatasetExportRequest(RegistryDatasetPreviewRequest):
        name: str = Field(min_length=1, max_length=160)
        path: str = Field(min_length=1)

    router = APIRouter(prefix="/curation", tags=["curation"])

    def _actor_user_id(value: str) -> str | None:
        try:
            return str(uuid.UUID(value))
        except (TypeError, ValueError):
            return None

    def _with_urls(item: dict) -> dict:
        value = dict(item)
        roi_id = value.get("id")
        if roi_id:
            value["roi_url"] = f"/refined-detections/{roi_id}/roi?format=jpg&width=320"
            value["thumbnail_url"] = f"/refined-detections/{roi_id}/roi?format=jpg&width=180"
        value.pop("total_count", None)
        return value

    def _feature_space_service(request: Request) -> FeatureSpaceService:
        auth = require_project_read(request)
        return FeatureSpaceService(get_context(request), project_id=auth.project_id)

    def _feature_space_analysis_payload(body: FeatureSpaceAnalysisRequest) -> dict:
        """Build the stable cache identity consumed by the ephemeral worker."""

        source = parse_feature_space_source(body.source_key)
        parameters = {
            "analysis_version": 1,
            "source_key": source.key,
            "min_cluster_size": body.min_cluster_size,
            "min_samples": body.min_samples,
            "cluster_selection_epsilon": body.cluster_selection_epsilon,
        }
        cache_key = sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {**parameters, "cache_key": cache_key}

    def _registry_root(request: Request) -> Path:
        return Path(get_context(request).config.file_browser.root_path_import_dir).expanduser().resolve()

    def _registry_destination(request: Request, value: str) -> Path:
        destination = Path(value).expanduser().resolve()
        roots = [_registry_root(request), *(
            Path(path).expanduser().resolve()
            for path in get_context(request).config.file_browser.allowed_root_paths
        )]
        if roots and not any(destination == root or destination.is_relative_to(root) for root in roots):
            raise HTTPException(status_code=403, detail="Path is outside the configured Registry roots")
        if destination.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
            raise HTTPException(status_code=422, detail="Registry datasets must use a .sqlite, .sqlite3, or .db extension")
        return destination

    def _validated_registry_selection(body: RegistryDatasetPreviewRequest) -> dict:
        selection = body.selection.model_dump(exclude_none=True)
        if selection.get("min_area") is not None and selection.get("max_area") is not None:
            if selection["min_area"] > selection["max_area"]:
                raise HTTPException(status_code=422, detail="Minimum ROI area cannot exceed maximum ROI area.")
        return selection

    def _classification_request(
        request: Request,
        *,
        model_ref: str | None,
        evidence_kind: str = "classification",
        roi_ids: list[str],
        selection: ClassificationTargetSelectionRequest | None,
    ) -> tuple[str, dict, int]:
        resolved_model_ref = (
            model_ref
            or (
                get_context(request).config.oracle.default_classification_model
                if evidence_kind == "classification"
                else ""
            )
        ).strip()
        if not resolved_model_ref:
            raise HTTPException(status_code=422, detail=f"A {evidence_kind} model is required.")
        if selection is None:
            resolved_selection = {"evidence_state": "all" if roi_ids else "missing_model"}
        else:
            resolved_selection = selection.model_dump(exclude_none=True)
        min_area = resolved_selection.get("min_area")
        max_area = resolved_selection.get("max_area")
        if min_area is not None and max_area is not None and min_area > max_area:
            raise HTTPException(status_code=422, detail="Minimum ROI area cannot exceed maximum ROI area.")
        auth = require_project_read(request)
        target_count = get_repository(request).count_classification_targets(
            project_id=auth.project_id,
            model_ref=resolved_model_ref,
            evidence_kind=evidence_kind,
            roi_ids=roi_ids,
            selection=resolved_selection,
        )
        return resolved_model_ref, resolved_selection, target_count

    @router.get("/options")
    def options(request: Request) -> dict:
        auth = require_project_read(request)
        repository = get_repository(request)
        context = get_context(request)
        client = context.oracle
        owns_client = not callable(getattr(client, "list_models", None))
        if owns_client:
            client = OracleInferenceClient(context.config.oracle)
        clustering_models = []
        try:
            models = client.list_models(task="classification")
            clustering_models = client.list_models(task="clustering")
            all_models = [*models, *clustering_models]
            available_count = sum(model.get("available") is not False for model in models)
            clustering_available_count = sum(
                model.get("available") is not False for model in clustering_models
            )
            oracle = {
                "enabled": True,
                "status": "ready" if all_models and (available_count or clustering_available_count) else "unavailable",
                "registered_model_count": len(models),
                "available_model_count": available_count,
                "registered_clustering_model_count": len(clustering_models),
                "available_clustering_model_count": clustering_available_count,
            }
            if all_models and not (available_count or clustering_available_count):
                oracle["error"] = "Oracle Builder has no usable evidence models."
        except OracleInferenceError as exc:
            models = []
            oracle = {
                "enabled": get_context(request).config.oracle.enabled,
                "status": "unavailable",
                "error": str(exc),
            }
        finally:
            if owns_client:
                client.close()
        assets = repository.list_assets(project_id=auth.project_id, limit=1000)
        export_root = _registry_root(request)
        default_filename = datetime.now(timezone.utc).strftime("pelagia-registry-%Y%m%d-%H%M%S.sqlite")
        return as_response(
            {
                "oracle": oracle,
                "models": models,
                "clustering_models": clustering_models,
                "default_model_ref": get_context(request).config.oracle.default_classification_model,
                "labels": repository.list_curation_labels(project_id=auth.project_id),
                "assets": [
                    {"id": str(asset["id"]), "filename": asset.get("filename") or str(asset["id"]),
                     "kind": asset.get("kind")}
                    for asset in assets if asset.get("id") is not None
                ],
                "registry_export": {
                    "root_path": str(export_root),
                    "default_path": str(export_root / default_filename),
                },
                "default_label_dictionary": default_taxonomy_dictionary(),
                "ownership": {
                    "human_ground_truth": "pelagia",
                    "model_execution": "oracle_builder",
                    "review_interface": "pelagiaview",
                },
            }
        )

    @router.post("/registry-datasets/preview")
    def preview_registry_export(request: Request, body: RegistryDatasetPreviewRequest) -> dict:
        auth = require_project_read(request)
        selection = _validated_registry_selection(body)
        return as_response(preview_registry_dataset(
            get_repository(request), project_id=str(auth.project_id), selection=selection,
            subsample_ratio=body.subsample_ratio,
        ))

    @router.post("/registry-datasets", status_code=202)
    def create_registry_dataset(request: Request, body: RegistryDatasetExportRequest) -> dict:
        auth = require_project_write(request)
        selection = _validated_registry_selection(body)
        preview = preview_registry_dataset(
            get_repository(request), project_id=str(auth.project_id), selection=selection,
            subsample_ratio=body.subsample_ratio,
        )
        if preview["selected_count"] < 1:
            raise HTTPException(status_code=422, detail="No refined ROIs match this dataset selection.")
        destination = _registry_destination(request, body.path)
        if destination.exists():
            raise HTTPException(status_code=409, detail="Destination already exists; choose a new SQLite filename.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        required_bytes = max(preview["estimated_sqlite_bytes"] * 2, 16 * 1024 * 1024)
        if shutil.disk_usage(destination.parent).free < required_bytes:
            raise HTTPException(status_code=507, detail=f"Dataset generation requires at least {required_bytes} bytes of free space")
        dataset_id = str(uuid.uuid4())
        revision_id = str(uuid.uuid4())
        job = PipelineService(get_context(request)).queue(
            PipelineStage.REGISTRY_GENERATE, project_id=str(auth.project_id),
            payload={
                "destination_path": str(destination), "name": body.name.strip(),
                "selection": selection, "subsample_ratio": body.subsample_ratio,
                "selected_count": preview["selected_count"], "dataset_id": dataset_id,
                "revision_id": revision_id, "owner_username": auth.username,
                "requested_at": datetime.now(timezone.utc).isoformat(),
            },
            summary=f"Generate Registry dataset with {preview['selected_count']} ROIs",
            submitted_by_user_id=auth.user_id, submitted_by_username=auth.username,
        )
        return {"job": as_response(job), "preview": preview,
                "dataset_id": dataset_id, "revision_id": revision_id,
                "destination_path": str(destination)}

    @router.get("/labels")
    def labels(request: Request, include_deprecated: bool = False) -> dict:
        auth = require_project_read(request)
        return {
            "labels": as_response(
                get_repository(request).list_curation_labels(
                    project_id=auth.project_id,
                    include_deprecated=include_deprecated,
                )
            )
        }

    @router.post("/labels", status_code=201)
    def create_label(request: Request, body: LabelCreate) -> dict:
        auth = require_project_write(request)
        try:
            row = get_repository(request).create_curation_label(
                project_id=auth.project_id,
                **body.model_dump(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Parent label was not found: {exc.args[0]}") from exc
        except Exception as exc:
            if "unique" in str(exc).lower():
                raise HTTPException(status_code=409, detail="A label with that name already exists") from exc
            raise
        return {"label": as_response(row)}

    @router.post("/labels/import-defaults")
    def import_default_labels(request: Request) -> dict:
        auth = require_project_write(request)
        result = get_repository(request).import_curation_label_dictionary(
            project_id=auth.project_id,
            dictionary=default_taxonomy_dictionary(),
        )
        return as_response(result)

    @router.get("/rois")
    def rois(
        request: Request,
        annotation_state: Literal["all", "labeled", "unlabeled"] = "all",
        review_state: Literal["all", "unreviewed", "verified", "rejected", "needs_review"] = "all",
        label_id: str | None = None,
        label_source: Literal["any", "human", "prediction"] = "any",
        evidence_state: Literal["all", "available", "missing", "disagreement"] = "all",
        search: str | None = None,
        sort_by: str = "oldest",
        limit: int = Query(120, ge=1, le=500),
        offset: int = Query(0, ge=0),
        telemetry_filter: list[str] = Query(default=[]),
    ) -> dict:
        auth = require_project_read(request)
        try:
            telemetry_filters = parse_telemetry_filters(telemetry_filter)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = get_repository(request).list_curation_rois(
            project_id=auth.project_id,
            annotation_state=annotation_state,
            review_state=review_state,
            label_id=label_id,
            label_source=label_source,
            evidence_state=evidence_state,
            search=search,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
            telemetry_filters=telemetry_filters,
        )
        result["items"] = [_with_urls(item) for item in result["items"]]
        return as_response(result)

    @router.get("/rois/{roi_id}")
    def roi_detail(request: Request, roi_id: str) -> dict:
        auth = require_project_read(request)
        row = get_repository(request).get_curation_roi(roi_id, project_id=auth.project_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Curatable ROI was not found")
        return {"roi": as_response(_with_urls(row))}

    @router.get("/feature-space/sources")
    def feature_space_sources(request: Request) -> dict:
        return {"sources": as_response(_feature_space_service(request).sources())}

    @router.get("/feature-space/rois")
    def feature_space_rois(
        request: Request,
        source_key: str,
        limit: int = Query(120, ge=1, le=250),
    ) -> dict:
        try:
            result = _feature_space_service(request).browse_rois(source_key=source_key, limit=limit)
        except FeatureSpaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response({**result, "items": [_with_urls(item) for item in result["items"]]})

    @router.get("/feature-space/similar/{roi_id}")
    def feature_space_similar(
        request: Request,
        roi_id: str,
        source_key: str,
        limit: int = Query(80, ge=1, le=250),
        minimum: float = Query(-1.0, ge=-1.0, le=1.0),
    ) -> dict:
        try:
            result = _feature_space_service(request).similar_rois(
                roi_id=roi_id, source_key=source_key, limit=limit, minimum=minimum
            )
        except FeatureSpaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response({**result, "items": [_with_urls(item) for item in result["items"]]})

    @router.post("/feature-space/umap/analysis", status_code=202)
    def queue_feature_space_umap_analysis(
        request: Request, body: FeatureSpaceAnalysisRequest
    ) -> dict:
        """Queue or reuse a short-lived UMAP/HDBSCAN exploration result."""

        auth = require_project_write(request)
        try:
            payload = _feature_space_analysis_payload(body)
            job, disposition = get_repository(request).enqueue_feature_space_analysis(
                project_id=str(auth.project_id),
                payload=payload,
                submitted_by_user_id=auth.user_id,
                submitted_by_username=auth.username,
                force=body.force,
            )
        except FeatureSpaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "job": as_response(job),
            "disposition": disposition,
            "cache_key": payload["cache_key"],
            "ephemeral": True,
        }

    @router.get("/feature-space/umap", deprecated=True)
    def feature_space_umap(request: Request) -> dict:
        """Retired synchronous endpoint; analysis now belongs to dedicated workers."""

        require_project_read(request)
        raise HTTPException(
            status_code=410,
            detail=(
                "Synchronous UMAP/HDBSCAN analysis has been retired. "
                "POST /curation/feature-space/umap/analysis and poll /jobs/{job_id}."
            ),
        )

    @router.get("/feature-space/clusters")
    def feature_space_clusters(request: Request, source_key: str) -> dict:
        try:
            result = _feature_space_service(request).clusters(source_key=source_key)
        except FeatureSpaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(result)

    @router.get("/feature-space/clusters/{cluster_id}/rois")
    def feature_space_cluster_members(
        request: Request,
        cluster_id: str,
        source_key: str,
        limit: int = Query(120, ge=1, le=250),
        offset: int = Query(0, ge=0),
    ) -> dict:
        try:
            result = _feature_space_service(request).cluster_members(
                source_key=source_key, cluster_id=cluster_id, limit=limit, offset=offset
            )
        except FeatureSpaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response({**result, "items": [_with_urls(item) for item in result["items"]]})

    @router.post("/annotations")
    def annotate(request: Request, body: AnnotationRequest) -> dict:
        auth = require_project_write(request)
        try:
            rows = get_repository(request).assign_curation_labels(
                project_id=auth.project_id,
                roi_ids=body.roi_ids,
                label_id=body.label_id,
                actor_user_id=_actor_user_id(auth.user_id),
                actor_username=auth.username,
                suggested_by_evidence_id=body.suggested_by_evidence_id,
                notes=body.notes,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Label or ROI was not found: {exc.args[0]}") from exc
        return {"annotations": as_response(rows)}

    @router.post("/reviews")
    def review(request: Request, body: ReviewRequest) -> dict:
        if body.decision not in {"verified", "rejected", "needs_review"}:
            raise HTTPException(status_code=422, detail="Unsupported review decision")
        auth = require_project_write(request)
        try:
            rows = get_repository(request).review_curation_annotations(
                project_id=auth.project_id,
                roi_ids=body.roi_ids,
                decision=body.decision,
                reviewer_user_id=_actor_user_id(auth.user_id),
                reviewer_username=auth.username,
                notes=body.notes,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Current annotation was not found: {exc.args[0]}") from exc
        return {"reviews": as_response(rows)}

    @router.post("/annotations/remove")
    def remove_annotation(request: Request, body: AnnotationRemovalRequest) -> dict:
        auth = require_project_write(request)
        try:
            rows = get_repository(request).remove_curation_labels(
                project_id=auth.project_id,
                roi_ids=body.roi_ids,
                actor_username=auth.username,
                notes=body.notes,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Current annotation was not found: {exc.args[0]}") from exc
        return {"annotations": as_response(rows)}

    @router.post("/classification-targets/preview")
    @router.post("/clustering-targets/preview")
    def preview_classification_targets(
        request: Request, body: ClassificationPreviewRequest
    ) -> dict:
        evidence_kind = "clustering" if request.url.path.endswith("/clustering-targets/preview") else body.evidence_kind
        model_ref, selection, target_count = _classification_request(
            request,
            model_ref=body.model_ref,
            evidence_kind=evidence_kind,
            roi_ids=body.roi_ids,
            selection=body.selection,
        )
        return {
            "model_ref": model_ref,
            "evidence_kind": evidence_kind,
            "selection": selection,
            "target_count": target_count,
            "explicit_roi_count": len(set(body.roi_ids)),
        }

    @router.post("/classification-jobs", status_code=202)
    @router.post("/clustering-jobs", status_code=202)
    def queue_classification(request: Request, body: ClassificationJobRequest) -> dict:
        auth = require_project_write(request)
        evidence_kind = "clustering" if request.url.path.endswith("/clustering-jobs") else body.evidence_kind
        model_ref, selection, target_count = _classification_request(
            request,
            model_ref=body.model_ref,
            evidence_kind=evidence_kind,
            roi_ids=body.roi_ids,
            selection=body.selection,
        )
        if target_count < 1:
            raise HTTPException(status_code=422, detail="No refined ROIs match this evidence query.")
        job = PipelineService(get_context(request)).queue(
            PipelineStage.CLASSIFY,
            project_id=auth.project_id,
            payload={
                "roi_ids": body.roi_ids,
                "model_ref": model_ref,
                "evidence_kind": evidence_kind,
                "selection": selection,
            },
            priority=body.priority,
            summary=f"Generate ML evidence for {target_count} refined ROIs with {model_ref}",
            submitted_by_user_id=auth.user_id,
            submitted_by_username=auth.username,
        )
        return {
            "job": as_response(job),
            "model_ref": model_ref,
            "evidence_kind": evidence_kind,
            "selection": selection,
            "target_count": target_count,
            "explicit_roi_count": len(set(body.roi_ids)),
        }
else:
    router = None
