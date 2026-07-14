from __future__ import annotations

import json
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Query, Request, Response
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_auth
    from ...services.io_exports import (
        AUTH_TABLES,
        DATASET_EXPORTS,
        EXPORT_FORMATS,
        TABLE_EXPORTS,
        ExportService,
    )
    from ._common import as_response, get_repository

    router = APIRouter(prefix="/io", tags=["io"])

    FILTER_KEYS = {
        "run_id",
        "asset_id",
        "frame_id",
        "collection",
        "kind",
        "filename",
        "path",
        "checksum",
        "min_size_bytes",
        "max_size_bytes",
        "media_count",
        "start_frame",
        "end_frame",
        "status",
        "stage",
        "worker_id",
        "job_id",
        "request_id",
        "event_type",
        "level",
        "logger",
        "roi_index",
        "min_bbox_x",
        "max_bbox_x",
        "min_bbox_y",
        "max_bbox_y",
        "min_bbox_w",
        "max_bbox_w",
        "min_bbox_h",
        "max_bbox_h",
        "min_area",
        "max_area",
        "min_perimeter",
        "max_perimeter",
        "roi_encoding",
        "roi_format",
        "mask_encoding",
        "mask_format",
        "model_key",
        "task",
        "username",
        "project_key",
        "active_only",
        "limit",
        "offset",
    }

    def _filters(**values: Any) -> dict[str, Any]:
        return {
            key: value
            for key, value in values.items()
            if key in FILTER_KEYS and value is not None
        }

    def _download(payload) -> Response:
        return Response(
            content=payload.content,
            media_type=payload.media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{payload.filename}"',
                "X-Pelagia-Export-Rows": json.dumps(as_response(payload.row_counts), sort_keys=True),
            },
        )

    def _require_table_export_permission(auth, tables: list[str], include_all_projects: bool) -> None:
        auth_tables = sorted(set(tables) & AUTH_TABLES)
        if auth_tables and not auth.is_admin:
            raise HTTPException(
                status_code=403,
                detail=f"Admin permission is required to export auth table(s): {', '.join(auth_tables)}.",
            )
        if include_all_projects and not auth.is_admin:
            raise HTTPException(status_code=403, detail="Admin permission is required to export all projects.")

    def _require_export_scope(auth, include_all_projects: bool) -> None:
        if auth.project_id is None and not include_all_projects:
            raise HTTPException(status_code=403, detail="Select or create a project before exporting project data.")

    @router.get("")
    def io_index(request: Request) -> dict[str, Any]:
        require_auth(request)
        return {
            "exports": {
                "formats": sorted(EXPORT_FORMATS),
                "tables": list(TABLE_EXPORTS),
                "datasets": sorted(DATASET_EXPORTS),
                "endpoints": {
                    "table": "/io/export/table/{table_name}",
                    "tables": "/io/export/tables",
                    "frame_metadata": "/io/export/datasets/frame-metadata",
                    "roi_metadata": "/io/export/datasets/roi-metadata",
                },
            }
        }

    @router.get("/export/options")
    def export_options(request: Request) -> dict[str, Any]:
        return io_index(request)["exports"]

    @router.get("/export/table/{table_name}")
    def export_table(
        request: Request,
        table_name: str,
        format: str = "xlsx",
        include_all_projects: bool = False,
        run_id: str | None = None,
        asset_id: str | None = None,
        frame_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        path: str | None = None,
        checksum: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        media_count: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        status: str | None = None,
        stage: str | None = None,
        worker_id: str | None = None,
        job_id: str | None = None,
        request_id: str | None = None,
        event_type: str | None = None,
        level: str | None = None,
        logger: str | None = None,
        roi_index: int | None = None,
        min_bbox_x: int | None = None,
        max_bbox_x: int | None = None,
        min_bbox_y: int | None = None,
        max_bbox_y: int | None = None,
        min_bbox_w: int | None = None,
        max_bbox_w: int | None = None,
        min_bbox_h: int | None = None,
        max_bbox_h: int | None = None,
        min_area: float | None = None,
        max_area: float | None = None,
        min_perimeter: float | None = None,
        max_perimeter: float | None = None,
        roi_encoding: str | None = None,
        roi_format: str | None = None,
        mask_encoding: str | None = None,
        mask_format: str | None = None,
        model_key: str | None = None,
        task: str | None = None,
        username: str | None = None,
        project_key: str | None = None,
        active_only: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> Response:
        auth = require_auth(request)
        _require_export_scope(auth, include_all_projects)
        _require_table_export_permission(auth, [table_name], include_all_projects)
        try:
            payload = ExportService(get_repository(request)).table_export(
                [table_name],
                file_format=format,
                project_id=auth.project_id,
                include_all_projects=include_all_projects,
                filters=_filters(**locals()),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _download(payload)

    @router.get("/export/tables")
    def export_tables(
        request: Request,
        tables: list[str] = Query(default_factory=list),
        format: str = "xlsx",
        include_all_projects: bool = False,
        run_id: str | None = None,
        asset_id: str | None = None,
        frame_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        path: str | None = None,
        checksum: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        media_count: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        status: str | None = None,
        stage: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Response:
        auth = require_auth(request)
        _require_export_scope(auth, include_all_projects)
        resolved_tables = tables or ["runs", "raw_assets", "frames", "detection_candidate"]
        _require_table_export_permission(auth, resolved_tables, include_all_projects)
        try:
            payload = ExportService(get_repository(request)).table_export(
                resolved_tables,
                file_format=format,
                project_id=auth.project_id,
                include_all_projects=include_all_projects,
                filters=_filters(**locals()),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _download(payload)

    @router.get("/export/datasets/frame-metadata")
    def export_frame_metadata(
        request: Request,
        format: str = "xlsx",
        include_all_projects: bool = False,
        run_id: str | None = None,
        asset_id: str | None = None,
        frame_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        path: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        status: str | None = None,
        stage: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Response:
        auth = require_auth(request)
        _require_export_scope(auth, include_all_projects)
        if include_all_projects and not auth.is_admin:
            raise HTTPException(status_code=403, detail="Admin permission is required to export all projects.")
        try:
            payload = ExportService(get_repository(request)).dataset_export(
                "frame_metadata",
                file_format=format,
                project_id=auth.project_id,
                include_all_projects=include_all_projects,
                filters=_filters(**locals()),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _download(payload)

    @router.get("/export/datasets/roi-metadata")
    def export_roi_metadata(
        request: Request,
        format: str = "xlsx",
        include_all_projects: bool = False,
        run_id: str | None = None,
        asset_id: str | None = None,
        frame_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        path: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        roi_index: int | None = None,
        min_bbox_w: int | None = None,
        max_bbox_w: int | None = None,
        min_bbox_h: int | None = None,
        max_bbox_h: int | None = None,
        min_area: float | None = None,
        max_area: float | None = None,
        min_perimeter: float | None = None,
        max_perimeter: float | None = None,
        roi_encoding: str | None = None,
        mask_encoding: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Response:
        auth = require_auth(request)
        _require_export_scope(auth, include_all_projects)
        if include_all_projects and not auth.is_admin:
            raise HTTPException(status_code=403, detail="Admin permission is required to export all projects.")
        try:
            payload = ExportService(get_repository(request)).dataset_export(
                "roi_metadata",
                file_format=format,
                project_id=auth.project_id,
                include_all_projects=include_all_projects,
                filters=_filters(**locals()),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _download(payload)
else:
    router = None
