from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Query, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_read, require_project_write
    from ..schemas import (
        ProcessingStatusFrameIdsResponse,
        ProcessingStatusFacetsResponse,
        ProcessingStatusFramesResponse,
        ProcessingStatusRebuildResponse,
        ProcessingStatusSummaryResponse,
    )
    from ...services.processing_status import ProcessingStatusService
    from ._common import as_response, get_repository, page_metadata

    router = APIRouter(prefix="/processing/status", tags=["processing-status"])

    def _bounded_limit(limit: int | None, *, default: int, maximum: int) -> int:
        return min(max(1, default if limit is None else int(limit)), maximum)

    def _query_values(values: list[str] | None) -> list[str]:
        if not values:
            return []
        resolved: list[str] = []
        for value in values:
            for item in str(value).split(","):
                stripped = item.strip()
                if stripped:
                    resolved.append(stripped)
        return resolved

    def _status_filters(
        preprocessing_status: list[str] | None,
        candidate_detection_status: list[str] | None,
        roi_refinement_status: list[str] | None,
    ) -> dict[str, list[str]]:
        return {
            "preprocessing_status": _query_values(preprocessing_status),
            "candidate_detection_status": _query_values(candidate_detection_status),
            "roi_refinement_status": _query_values(roi_refinement_status),
        }

    def _filter_kwargs(
        *,
        run_id: str | None,
        asset_id: str | None,
        collection: str | None,
        preprocessing_status: list[str] | None,
        candidate_detection_status: list[str] | None,
        roi_refinement_status: list[str] | None,
        has_candidates: bool | None,
        has_refined_rois: bool | None,
        start_frame: int | None,
        end_frame: int | None,
    ) -> dict:
        return {
            "run_id": run_id,
            "asset_id": asset_id,
            "collection": collection,
            **_status_filters(preprocessing_status, candidate_detection_status, roi_refinement_status),
            "has_candidates": has_candidates,
            "has_refined_rois": has_refined_rois,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    @router.get("/summary", response_model=ProcessingStatusSummaryResponse)
    def get_frame_status_summary(
        request: Request,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
        preprocessing_status: list[str] | None = Query(None),
        candidate_detection_status: list[str] | None = Query(None),
        roi_refinement_status: list[str] | None = Query(None),
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> dict:
        auth = require_project_read(request)
        repository = get_repository(request)
        filters = _filter_kwargs(
            run_id=run_id,
            asset_id=asset_id,
            collection=collection,
            preprocessing_status=preprocessing_status,
            candidate_detection_status=candidate_detection_status,
            roi_refinement_status=roi_refinement_status,
            has_candidates=has_candidates,
            has_refined_rois=has_refined_rois,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        try:
            result = ProcessingStatusService(repository).summary(
                project_id=auth.project_id,
                filters=filters,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(result)

    @router.get("/facets", response_model=ProcessingStatusFacetsResponse)
    def get_frame_status_facets(
        request: Request,
        run_id: str | None = None,
        asset_id: list[str] | None = Query(None),
        collection: list[str] | None = Query(None),
        preprocessing_status: list[str] | None = Query(None),
        candidate_detection_status: list[str] | None = Query(None),
        roi_refinement_status: list[str] | None = Query(None),
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> dict:
        auth = require_project_read(request)
        repository = get_repository(request)
        filters = {
            "run_id": run_id,
            "asset_ids": _query_values(asset_id),
            "collections": _query_values(collection),
            **_status_filters(preprocessing_status, candidate_detection_status, roi_refinement_status),
            "has_candidates": has_candidates,
            "has_refined_rois": has_refined_rois,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        try:
            result = ProcessingStatusService(repository).facets(
                project_id=auth.project_id,
                filters=filters,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(result)

    @router.get("/frames/ids", response_model=ProcessingStatusFrameIdsResponse)
    def list_frame_status_ids(
        request: Request,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
        preprocessing_status: list[str] | None = Query(None),
        candidate_detection_status: list[str] | None = Query(None),
        roi_refinement_status: list[str] | None = Query(None),
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = 5000,
        cursor: str | None = None,
        offset: int = 0,
    ) -> dict:
        auth = require_project_read(request)
        repository = get_repository(request)
        resolved_limit = _bounded_limit(limit, default=5000, maximum=50000)
        try:
            result = repository.list_frame_status_ids(
                project_id=auth.project_id,
                **_filter_kwargs(
                    run_id=run_id,
                    asset_id=asset_id,
                    collection=collection,
                    preprocessing_status=preprocessing_status,
                    candidate_detection_status=candidate_detection_status,
                    roi_refinement_status=roi_refinement_status,
                    has_candidates=has_candidates,
                    has_refined_rois=has_refined_rois,
                    start_frame=start_frame,
                    end_frame=end_frame,
                ),
                limit=resolved_limit,
                cursor=cursor,
                offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(
            {
                **result,
                "page": page_metadata(limit=resolved_limit, offset=offset, count=len(result.get("frame_ids", []))),
            }
        )

    @router.get("/frames", response_model=ProcessingStatusFramesResponse)
    def list_frame_status(
        request: Request,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
        preprocessing_status: list[str] | None = Query(None),
        candidate_detection_status: list[str] | None = Query(None),
        roi_refinement_status: list[str] | None = Query(None),
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = 1000,
        cursor: str | None = None,
        offset: int = 0,
    ) -> dict:
        auth = require_project_read(request)
        repository = get_repository(request)
        resolved_limit = _bounded_limit(limit, default=1000, maximum=10000)
        try:
            result = repository.list_frame_status(
                project_id=auth.project_id,
                **_filter_kwargs(
                    run_id=run_id,
                    asset_id=asset_id,
                    collection=collection,
                    preprocessing_status=preprocessing_status,
                    candidate_detection_status=candidate_detection_status,
                    roi_refinement_status=roi_refinement_status,
                    has_candidates=has_candidates,
                    has_refined_rois=has_refined_rois,
                    start_frame=start_frame,
                    end_frame=end_frame,
                ),
                limit=resolved_limit,
                cursor=cursor,
                offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(
            {
                **result,
                "page": page_metadata(limit=resolved_limit, offset=offset, count=len(result.get("frames", []))),
            }
        )

    @router.post("/rebuild", response_model=ProcessingStatusRebuildResponse)
    def rebuild_frame_status(request: Request, asset_id: str | None = None) -> dict:
        auth = require_project_write(request)
        repository = get_repository(request)
        try:
            result = repository.rebuild_frame_status(project_id=auth.project_id, asset_id=asset_id)
            snapshot_summary = repository.get_frame_status_summary(project_id=auth.project_id)
            snapshot = repository.get_or_create_processing_status_snapshot(
                project_id=auth.project_id,
                session_id=auth.session_id,
                summary=snapshot_summary,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response({"status": "rebuilt", **result, "snapshot": snapshot})
else:  # pragma: no cover
    router = None
