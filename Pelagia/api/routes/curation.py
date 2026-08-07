from __future__ import annotations

import uuid
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

    class ClassificationJobRequest(BaseModel):
        model_config = ConfigDict(extra="forbid", protected_namespaces=())
        roi_ids: list[str] = Field(default_factory=list)
        model_ref: str | None = None
        priority: int | None = None

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

    @router.get("/options")
    def options(request: Request) -> dict:
        auth = require_project_read(request)
        repository = get_repository(request)
        context = get_context(request)
        client = context.oracle
        owns_client = not callable(getattr(client, "list_models", None))
        if owns_client:
            client = OracleInferenceClient(context.config.oracle)
        try:
            models = client.list_models(task="classification")
            oracle = {"enabled": True, "status": "ready"}
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
        return as_response(
            {
                "oracle": oracle,
                "models": models,
                "default_model_ref": get_context(request).config.oracle.default_classification_model,
                "labels": repository.list_curation_labels(project_id=auth.project_id),
                "ownership": {
                    "human_ground_truth": "pelagia",
                    "model_execution": "oracle_builder",
                    "review_interface": "pelagiaview",
                },
            }
        )

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

    @router.get("/rois")
    def rois(
        request: Request,
        annotation_state: Literal["all", "labeled", "unlabeled"] = "all",
        review_state: Literal["all", "unreviewed", "verified", "rejected", "needs_review"] = "all",
        label_id: str | None = None,
        evidence_state: Literal["all", "available", "missing", "disagreement"] = "all",
        search: str | None = None,
        sort_by: str = "oldest",
        limit: int = Query(120, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict:
        auth = require_project_read(request)
        result = get_repository(request).list_curation_rois(
            project_id=auth.project_id,
            annotation_state=annotation_state,
            review_state=review_state,
            label_id=label_id,
            evidence_state=evidence_state,
            search=search,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
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

    @router.post("/classification-jobs", status_code=202)
    def queue_classification(request: Request, body: ClassificationJobRequest) -> dict:
        auth = require_project_write(request)
        model_ref = body.model_ref or get_context(request).config.oracle.default_classification_model
        job = PipelineService(get_context(request)).queue(
            PipelineStage.CLASSIFY,
            project_id=auth.project_id,
            payload={"roi_ids": body.roi_ids, "model_ref": model_ref},
            priority=body.priority,
            summary=f"Classify {'selected' if body.roi_ids else 'all'} refined ROIs with {model_ref}",
        )
        return {"job": as_response(job), "model_ref": model_ref}
else:
    router = None
