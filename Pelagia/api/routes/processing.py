"""Backend-owned processing queue planning endpoints."""

from __future__ import annotations

from typing import Any, Literal

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_write
    from ...services.context import AppContext
    from ...services.processing_queue import PreprocessQueueRequest, ProcessingQueueService
    from ._common import as_response, get_context

    class ProcessingQueueFilters(BaseModel):
        run_id: str | None = None
        asset_ids: list[str] = Field(default_factory=list)
        collection: list[str] = Field(default_factory=list)
        preprocessing_status: list[str] = Field(default_factory=list)
        candidate_detection_status: list[str] = Field(default_factory=list)
        roi_refinement_status: list[str] = Field(default_factory=list)
        refinement_state: list[Literal["unrefined", "refined"]] = Field(default_factory=lambda: ["unrefined"])
        start_frame: int | None = None
        end_frame: int | None = None

    class ProcessingQueueBatch(BaseModel):
        max_units: int = 250
        ordering: Literal["optimized", "input"] = "optimized"

    class ProcessingQueueRequest(BaseModel):
        stage: Literal["preprocess_frames", "segment", "roi_refinement"]
        filters: ProcessingQueueFilters = Field(default_factory=ProcessingQueueFilters)
        options: dict[str, Any] = Field(default_factory=dict)
        batch: ProcessingQueueBatch = Field(default_factory=ProcessingQueueBatch)
        priority: int | None = None
        dry_run: bool = False

    router = APIRouter(prefix="/processing", tags=["processing"])

    @router.post("/queue")
    def queue_processing(request: Request, body: ProcessingQueueRequest) -> dict:
        auth = require_project_write(request)
        try:
            filters = body.filters.model_dump() if hasattr(body.filters, "model_dump") else body.filters.dict()
            service = ProcessingQueueService(get_context(request))
            queue_request = PreprocessQueueRequest(filters=filters, options=body.options, max_units=body.batch.max_units, ordering=body.batch.ordering, priority=body.priority, dry_run=body.dry_run)
            if body.stage == "preprocess_frames":
                result = service.queue_preprocess(queue_request, project_id=auth.project_id)
            elif body.stage == "segment":
                result = service.queue_segment(queue_request, project_id=auth.project_id)
            else:
                result = service.queue_roi_refinement(queue_request, project_id=auth.project_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(result)
else:
    router = None
