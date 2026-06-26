from __future__ import annotations

import base64
import io
import json
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Sequence
from uuid import UUID
from xml.sax.saxutils import escape as xml_escape

from ..storage.postgres import REQUIRED_SCHEMA_TABLES
from ..utils.serialization import json_ready
from ..utils.validation import validate_schema_name


EXPORT_FORMATS = {"sqlite", "xlsx"}
DATASET_EXPORTS = {"frame_metadata", "roi_metadata"}
AUTH_TABLES = {"users", "project_memberships", "user_sessions"}
GLOBAL_TABLES = AUTH_TABLES | {"worker_sessions"}
TABLE_EXPORTS = tuple(REQUIRED_SCHEMA_TABLES)
DEFAULT_EXPORT_LIMIT = 10_000
MAX_EXPORT_LIMIT = 100_000


@dataclass(slots=True)
class ExportPayload:
    filename: str
    media_type: str
    content: bytes
    row_counts: dict[str, int]


class ExportService:
    """Build table and compiled dataset exports from the current repository."""

    def __init__(self, repository: Any):
        self.repository = repository
        self.schema = validate_schema_name(repository.schema)

    def table_export(
        self,
        table_names: Sequence[str],
        *,
        file_format: str,
        project_id: str | None,
        include_all_projects: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> ExportPayload:
        resolved_format = _export_format(file_format)
        tables = [_table_name(table) for table in table_names]
        if not tables:
            raise ValueError("At least one table is required.")
        filters = dict(filters or {})
        sheets: dict[str, list[dict[str, Any]]] = {}
        for table in tables:
            sheets[table] = self._fetch_table_rows(
                table,
                project_id=None if include_all_projects else project_id,
                filters=filters,
            )
        stem = "pelagia-tables" if len(tables) > 1 else f"pelagia-{tables[0]}"
        return _payload_from_sheets(stem, resolved_format, sheets)

    def dataset_export(
        self,
        dataset_name: str,
        *,
        file_format: str,
        project_id: str | None,
        include_all_projects: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> ExportPayload:
        resolved_format = _export_format(file_format)
        dataset = _dataset_name(dataset_name)
        filters = dict(filters or {})
        if dataset == "frame_metadata":
            rows = self._fetch_frame_metadata(
                project_id=None if include_all_projects else project_id,
                filters=filters,
            )
        elif dataset == "roi_metadata":
            rows = self._fetch_roi_metadata(
                project_id=None if include_all_projects else project_id,
                filters=filters,
            )
        else:  # pragma: no cover - protected by _dataset_name
            raise ValueError(f"Unsupported dataset export {dataset_name!r}.")
        return _payload_from_sheets(f"pelagia-{dataset}", resolved_format, {dataset: rows})

    def _fetch_table_rows(
        self,
        table: str,
        *,
        project_id: str | None,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        alias, joins = _table_source(table, self.schema)
        clauses: list[str] = []
        params: list[Any] = []
        self._add_project_scope(table, alias, clauses, params, project_id)
        self._add_common_filters(table, alias, clauses, params, filters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_by = _table_order_by(table, alias)
        limit, offset = _limit_offset(filters)
        params.extend([limit, offset])
        query = f"""
            SELECT {alias}.*
            FROM {self.schema}.{table} {alias}
            {joins}
            {where}
            {order_by}
            LIMIT %s OFFSET %s
        """
        return self._fetch(query, params)

    def _fetch_frame_metadata(
        self,
        *,
        project_id: str | None,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        self._add_asset_frame_filters(clauses, params, filters)
        if status := filters.get("status"):
            clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM {schema}.processing_jobs jobs
                    WHERE jobs.asset_id = assets.id AND jobs.status = %s
                )
                """.format(schema=self.schema)
            )
            params.append(status)
        if stage := filters.get("stage"):
            clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM {schema}.processing_jobs jobs
                    WHERE jobs.asset_id = assets.id AND jobs.stage = %s
                )
                """.format(schema=self.schema)
            )
            params.append(stage)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit, offset = _limit_offset(filters)
        params.extend([limit, offset])
        query = f"""
            SELECT
                frames.id AS frame_id,
                frames.run_id,
                frames.asset_id,
                assets.filename AS asset_filename,
                assets.path AS asset_path,
                assets.kind AS asset_kind,
                assets.collections AS asset_collections,
                assets.size_bytes AS asset_size_bytes,
                assets.media_count AS asset_media_count,
                runs.run_key,
                runs.instrument,
                runs.source_type,
                runs.status AS run_status,
                frames.frame_index,
                frames.width,
                frames.height,
                frames.bbox_x,
                frames.bbox_y,
                frames.parent_frame_id,
                frames.source_ref,
                frames.payload_ref,
                frames.payload_encoding,
                frames.payload_format,
                frames.payload_dtype,
                frames.payload_shape,
                frames.preprocessed_payload_ref,
                frames.preprocessed_payload_encoding,
                frames.preprocessed_payload_format,
                frames.preprocessed_payload_dtype,
                frames.preprocessed_payload_shape,
                frames.background_payload_ref,
                frames.background_payload_encoding,
                frames.background_payload_format,
                frames.background_payload_dtype,
                frames.background_payload_shape,
                frames.metadata AS frame_metadata,
                frames.preprocessed_metadata,
                frames.background_metadata,
                frames.created_at AS frame_created_at,
                job_summary.job_count,
                job_summary.latest_job_status,
                job_summary.latest_job_stage,
                job_summary.latest_job_updated_at
            FROM {self.schema}.frames frames
            JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
            JOIN {self.schema}.runs runs ON runs.id = frames.run_id
            LEFT JOIN LATERAL (
                SELECT
                    COUNT(*)::int AS job_count,
                    (ARRAY_AGG(jobs.status ORDER BY jobs.updated_at DESC, jobs.id DESC))[1] AS latest_job_status,
                    (ARRAY_AGG(jobs.stage ORDER BY jobs.updated_at DESC, jobs.id DESC))[1] AS latest_job_stage,
                    MAX(jobs.updated_at) AS latest_job_updated_at
                FROM {self.schema}.processing_jobs jobs
                WHERE jobs.asset_id = assets.id OR jobs.run_id = runs.id
            ) job_summary ON TRUE
            {where}
            ORDER BY assets.filename ASC NULLS LAST, frames.frame_index ASC, frames.id ASC
            LIMIT %s OFFSET %s
        """
        return self._fetch(query, params)

    def _fetch_roi_metadata(
        self,
        *,
        project_id: str | None,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        self._add_asset_frame_filters(clauses, params, filters)
        self._add_detection_filters("detections", clauses, params, filters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit, offset = _limit_offset(filters)
        params.extend([limit, offset])
        query = f"""
            SELECT
                detections.id AS detection_id,
                detections.run_id,
                detections.frame_id,
                frames.asset_id,
                assets.filename AS asset_filename,
                assets.path AS asset_path,
                assets.kind AS asset_kind,
                assets.collections AS asset_collections,
                runs.run_key,
                frames.frame_index,
                detections.roi_index,
                detections.bbox_x,
                detections.bbox_y,
                detections.bbox_w,
                detections.bbox_h,
                detections.crop_bbox_x,
                detections.crop_bbox_y,
                detections.crop_bbox_w,
                detections.crop_bbox_h,
                detections.area,
                detections.perimeter,
                detections.major_axis_length,
                detections.minor_axis_length,
                detections.min_gray_value,
                detections.mean_gray_value,
                detections.roi_encoding,
                detections.roi_format,
                detections.roi_dtype,
                detections.roi_shape,
                detections.mask_encoding,
                detections.mask_format,
                detections.mask_dtype,
                detections.mask_shape,
                octet_length(detections.roi_payload) AS roi_payload_bytes,
                octet_length(detections.mask_payload) AS mask_payload_bytes,
                detections.metadata AS detection_metadata,
                detections.created_at AS detection_created_at,
                refined.id AS refined_detection_id,
                refined.refinement_method AS refined_method,
                refined.created_at AS refined_created_at,
                job_summary.job_count,
                job_summary.latest_job_status,
                job_summary.latest_job_stage,
                job_summary.latest_job_updated_at
            FROM {self.schema}.detection_candidate detections
            JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
            JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
            JOIN {self.schema}.runs runs ON runs.id = detections.run_id
            LEFT JOIN LATERAL (
                SELECT refined.*
                FROM {self.schema}.detections_refined refined
                WHERE refined.candidate_detection_id = detections.id
                ORDER BY refined.created_at DESC, refined.id DESC
                LIMIT 1
            ) refined ON TRUE
            LEFT JOIN LATERAL (
                SELECT
                    COUNT(*)::int AS job_count,
                    (ARRAY_AGG(jobs.status ORDER BY jobs.updated_at DESC, jobs.id DESC))[1] AS latest_job_status,
                    (ARRAY_AGG(jobs.stage ORDER BY jobs.updated_at DESC, jobs.id DESC))[1] AS latest_job_stage,
                    MAX(jobs.updated_at) AS latest_job_updated_at
                FROM {self.schema}.processing_jobs jobs
                WHERE jobs.asset_id = assets.id OR jobs.run_id = runs.id
            ) job_summary ON TRUE
            {where}
            ORDER BY assets.filename ASC NULLS LAST, frames.frame_index ASC, detections.roi_index ASC
            LIMIT %s OFFSET %s
        """
        return self._fetch(query, params)

    def _fetch(self, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        with self.repository.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                return [dict(row) for row in cursor.fetchall()]

    def _add_project_scope(
        self,
        table: str,
        alias: str,
        clauses: list[str],
        params: list[Any],
        project_id: str | None,
    ) -> None:
        if project_id is None or table in GLOBAL_TABLES:
            return
        if table == "projects":
            clauses.append(f"{alias}.id = %s")
            params.append(project_id)
            return
        if table in {"runs", "raw_assets", "models", "processing_jobs", "logs"}:
            clauses.append(f"{alias}.project_id = %s")
            params.append(project_id)
        elif table == "frames":
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        elif table in {"detection_candidate", "detections_refined", "classification_results"}:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        elif table in {"processing_job_dependencies", "job_events"}:
            clauses.append("jobs.project_id = %s")
            params.append(project_id)

    def _add_common_filters(
        self,
        table: str,
        alias: str,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        if table == "runs":
            self._add_run_filters(alias, clauses, params, filters)
        if table == "raw_assets":
            self._add_asset_filters(clauses, params, filters)
            self._add_raw_asset_frame_exists_filters(clauses, params, filters)
        if table in {"frames", "detection_candidate", "detections_refined", "classification_results"}:
            self._add_asset_frame_filters(clauses, params, filters)
        if table in {"detection_candidate", "detections_refined", "classification_results"}:
            detection_alias = "detections" if table == "classification_results" else alias
            self._add_detection_filters(detection_alias, clauses, params, filters)
        if table in {"processing_jobs", "job_events", "processing_job_dependencies"}:
            self._add_job_filters("jobs" if table != "processing_jobs" else alias, clauses, params, filters)
        if table == "logs":
            self._add_log_filters(alias, clauses, params, filters)
        if table == "models":
            if model_key := filters.get("model_key"):
                clauses.append(f"{alias}.model_key ILIKE %s")
                params.append(f"%{model_key}%")
            if task := filters.get("task"):
                clauses.append(f"{alias}.task = %s")
                params.append(task)
        if table == "users":
            if username := filters.get("username"):
                clauses.append(f"{alias}.username ILIKE %s")
                params.append(f"%{username}%")
            if filters.get("active_only"):
                clauses.append(f"{alias}.is_active IS TRUE")
        if table == "projects":
            if project_key := filters.get("project_key"):
                clauses.append(f"{alias}.project_key ILIKE %s")
                params.append(f"%{project_key}%")
            if filters.get("active_only"):
                clauses.append(f"{alias}.is_active IS TRUE")

    def _add_run_filters(
        self,
        alias: str,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        if run_id := filters.get("run_id"):
            clauses.append(f"{alias}.id = %s")
            params.append(run_id)
        if collection := filters.get("collection"):
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM {self.schema}.raw_assets assets
                    WHERE assets.run_id = {alias}.id AND %s = ANY(assets.collections)
                )
                """
            )
            params.append(collection)
        asset_filters = {
            "asset_id": "assets.id = %s",
            "kind": "assets.kind = %s",
            "checksum": "assets.checksum = %s",
            "media_count": "assets.media_count = %s",
        }
        for key, clause in asset_filters.items():
            if filters.get(key) is not None:
                clauses.append(
                    f"""
                    EXISTS (
                        SELECT 1
                        FROM {self.schema}.raw_assets assets
                        WHERE assets.run_id = {alias}.id AND {clause}
                    )
                    """
                )
                params.append(filters[key])
        for key, column in {"filename": "filename", "path": "path"}.items():
            if value := filters.get(key):
                clauses.append(
                    f"""
                    EXISTS (
                        SELECT 1
                        FROM {self.schema}.raw_assets assets
                        WHERE assets.run_id = {alias}.id AND assets.{column} ILIKE %s
                    )
                    """
                )
                params.append(f"%{value}%")
        if filters.get("min_size_bytes") is not None:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM {self.schema}.raw_assets assets
                    WHERE assets.run_id = {alias}.id AND assets.size_bytes >= %s
                )
                """
            )
            params.append(filters["min_size_bytes"])
        if filters.get("max_size_bytes") is not None:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM {self.schema}.raw_assets assets
                    WHERE assets.run_id = {alias}.id AND assets.size_bytes <= %s
                )
                """
            )
            params.append(filters["max_size_bytes"])
        for key, column in {
            "run_key": "run_key",
            "instrument": "instrument",
            "source_type": "source_type",
            "status": "status",
        }.items():
            if value := filters.get(key):
                clauses.append(f"{alias}.{column} = %s")
                params.append(value)

    def _add_asset_frame_filters(
        self,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        self._add_asset_filters(clauses, params, filters)
        self._add_frame_filters("frames", clauses, params, filters)

    def _add_asset_filters(
        self,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        if run_id := filters.get("run_id"):
            clauses.append("runs.id = %s")
            params.append(run_id)
        if asset_id := filters.get("asset_id"):
            clauses.append("assets.id = %s")
            params.append(asset_id)
        if collection := filters.get("collection"):
            clauses.append("%s = ANY(assets.collections)")
            params.append(collection)
        if kind := filters.get("kind"):
            clauses.append("assets.kind = %s")
            params.append(kind)
        if filename := filters.get("filename"):
            clauses.append("assets.filename ILIKE %s")
            params.append(f"%{filename}%")
        if path := filters.get("path"):
            clauses.append("assets.path ILIKE %s")
            params.append(f"%{path}%")
        if checksum := filters.get("checksum"):
            clauses.append("assets.checksum = %s")
            params.append(checksum)
        if filters.get("min_size_bytes") is not None:
            clauses.append("assets.size_bytes >= %s")
            params.append(filters["min_size_bytes"])
        if filters.get("max_size_bytes") is not None:
            clauses.append("assets.size_bytes <= %s")
            params.append(filters["max_size_bytes"])
        if filters.get("media_count") is not None:
            clauses.append("assets.media_count = %s")
            params.append(filters["media_count"])

    def _add_frame_filters(
        self,
        alias: str,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        if frame_id := filters.get("frame_id"):
            clauses.append(f"{alias}.id = %s")
            params.append(frame_id)
        if filters.get("start_frame") is not None:
            clauses.append(f"{alias}.frame_index >= %s")
            params.append(filters["start_frame"])
        if filters.get("end_frame") is not None:
            clauses.append(f"{alias}.frame_index <= %s")
            params.append(filters["end_frame"])

    def _add_raw_asset_frame_exists_filters(
        self,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        frame_clauses: list[str] = []
        frame_params: list[Any] = []
        self._add_frame_filters("frames", frame_clauses, frame_params, filters)
        if not frame_clauses:
            return
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM {self.schema}.frames frames
                WHERE frames.asset_id = assets.id AND {' AND '.join(frame_clauses)}
            )
            """
        )
        params.extend(frame_params)

    def _add_detection_filters(
        self,
        alias: str,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        ranges = [
            ("bbox_x", ">=", "min_bbox_x"),
            ("bbox_x", "<=", "max_bbox_x"),
            ("bbox_y", ">=", "min_bbox_y"),
            ("bbox_y", "<=", "max_bbox_y"),
            ("bbox_w", ">=", "min_bbox_w"),
            ("bbox_w", "<=", "max_bbox_w"),
            ("bbox_h", ">=", "min_bbox_h"),
            ("bbox_h", "<=", "max_bbox_h"),
            ("area", ">=", "min_area"),
            ("area", "<=", "max_area"),
            ("perimeter", ">=", "min_perimeter"),
            ("perimeter", "<=", "max_perimeter"),
        ]
        for column, operator, key in ranges:
            if filters.get(key) is not None:
                clauses.append(f"{alias}.{column} {operator} %s")
                params.append(filters[key])
        if filters.get("roi_index") is not None:
            clauses.append(f"{alias}.roi_index = %s")
            params.append(filters["roi_index"])
        for column in ("roi_encoding", "roi_format", "mask_encoding", "mask_format"):
            if value := filters.get(column):
                clauses.append(f"{alias}.{column} = %s")
                params.append(value)

    def _add_job_filters(
        self,
        alias: str,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        if run_id := filters.get("run_id"):
            clauses.append(f"{alias}.run_id = %s")
            params.append(run_id)
        if asset_id := filters.get("asset_id"):
            clauses.append(f"{alias}.asset_id = %s")
            params.append(asset_id)
        if job_id := filters.get("job_id"):
            clauses.append(f"{alias}.id = %s")
            params.append(job_id)
        if status := filters.get("status"):
            clauses.append(f"{alias}.status = %s")
            params.append(status)
        if stage := filters.get("stage"):
            clauses.append(f"{alias}.stage = %s")
            params.append(stage)
        if worker_id := filters.get("worker_id"):
            clauses.append(f"{alias}.worker_id = %s")
            params.append(worker_id)

    def _add_log_filters(
        self,
        alias: str,
        clauses: list[str],
        params: list[Any],
        filters: dict[str, Any],
    ) -> None:
        for column in ("run_id", "asset_id", "job_id", "worker_id", "request_id", "event_type", "level", "logger"):
            if value := filters.get(column):
                clauses.append(f"{alias}.{column} = %s")
                params.append(value)


def _table_source(table: str, schema: str) -> tuple[str, str]:
    if table == "runs":
        return "runs", ""
    if table == "raw_assets":
        return "assets", f"LEFT JOIN {schema}.runs runs ON runs.id = assets.run_id"
    if table == "frames":
        return "frames", f"JOIN {schema}.raw_assets assets ON assets.id = frames.asset_id JOIN {schema}.runs runs ON runs.id = frames.run_id"
    if table == "detection_candidate":
        return "detections", f"JOIN {schema}.frames frames ON frames.id = detections.frame_id JOIN {schema}.raw_assets assets ON assets.id = frames.asset_id JOIN {schema}.runs runs ON runs.id = detections.run_id"
    if table == "detections_refined":
        return "refined", f"JOIN {schema}.frames frames ON frames.id = refined.frame_id JOIN {schema}.raw_assets assets ON assets.id = frames.asset_id JOIN {schema}.runs runs ON runs.id = refined.run_id"
    if table == "classification_results":
        return "results", f"JOIN {schema}.detection_candidate detections ON detections.id = results.detection_id JOIN {schema}.frames frames ON frames.id = detections.frame_id JOIN {schema}.raw_assets assets ON assets.id = frames.asset_id JOIN {schema}.runs runs ON runs.id = detections.run_id"
    if table == "processing_jobs":
        return "jobs", ""
    if table == "processing_job_dependencies":
        return "deps", f"JOIN {schema}.processing_jobs jobs ON jobs.id = deps.job_id"
    if table == "job_events":
        return "events", f"LEFT JOIN {schema}.processing_jobs jobs ON jobs.id = events.job_id"
    if table == "models":
        return "models", ""
    if table == "logs":
        return "logs", ""
    if table == "project_memberships":
        return "memberships", ""
    if table == "user_sessions":
        return "sessions", ""
    if table == "worker_sessions":
        return "workers", ""
    if table == "users":
        return "users", ""
    if table == "projects":
        return "projects", ""
    raise ValueError(f"Unsupported table export {table!r}.")


def _table_order_by(table: str, alias: str) -> str:
    if table in {"job_events", "logs"}:
        return f"ORDER BY {alias}.created_at DESC, {alias}.id DESC"
    if table in {"frames"}:
        return f"ORDER BY {alias}.frame_index ASC, {alias}.id ASC"
    if table in {"detection_candidate", "detections_refined"}:
        return f"ORDER BY {alias}.created_at DESC, {alias}.roi_index ASC"
    if table == "processing_job_dependencies":
        return f"ORDER BY {alias}.job_id ASC, {alias}.depends_on_job_id ASC"
    if table == "worker_sessions":
        return f"ORDER BY {alias}.updated_at DESC"
    if table == "project_memberships":
        return f"ORDER BY {alias}.created_at DESC"
    if table in {"users", "projects"}:
        return f"ORDER BY {alias}.created_at DESC"
    return f"ORDER BY {alias}.created_at DESC"


def _payload_from_sheets(
    stem: str,
    file_format: str,
    sheets: dict[str, list[dict[str, Any]]],
) -> ExportPayload:
    row_counts = {name: len(rows) for name, rows in sheets.items()}
    if file_format == "sqlite":
        return ExportPayload(
            filename=f"{stem}.sqlite",
            media_type="application/vnd.sqlite3",
            content=_sqlite_bytes(sheets),
            row_counts=row_counts,
        )
    return ExportPayload(
        filename=f"{stem}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        content=_xlsx_bytes(sheets),
        row_counts=row_counts,
    )


def _sqlite_bytes(sheets: dict[str, list[dict[str, Any]]]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as handle:
        connection = sqlite3.connect(handle.name)
        try:
            for name, rows in sheets.items():
                table = _sqlite_identifier(name)
                columns = _columns_for_rows(rows)
                if not columns:
                    columns = ["empty_export"]
                column_sql = ", ".join(
                    f'"{_sqlite_identifier(column)}" {_sqlite_type(rows, column)}'
                    for column in columns
                )
                connection.execute(f'CREATE TABLE "{table}" ({column_sql})')
                if rows:
                    placeholders = ", ".join("?" for _ in columns)
                    column_names = ", ".join(f'"{_sqlite_identifier(column)}"' for column in columns)
                    connection.executemany(
                        f'INSERT INTO "{table}" ({column_names}) VALUES ({placeholders})',
                        [[_sqlite_value(row.get(column)) for column in columns] for row in rows],
                    )
            connection.commit()
        finally:
            connection.close()
        handle.seek(0)
        return handle.read()


def _xlsx_bytes(sheets: dict[str, list[dict[str, Any]]]) -> bytes:
    sheet_names = _sheet_names(list(sheets))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(len(sheets)))
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml(sheet_names))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(sheets)))
        archive.writestr("xl/styles.xml", _styles_xml())
        for index, (logical_name, rows) in enumerate(sheets.items(), start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _worksheet_xml(rows),
            )
    return output.getvalue()


def _worksheet_xml(rows: list[dict[str, Any]]) -> str:
    columns = _columns_for_rows(rows)
    row_xml = []
    if columns:
        row_xml.append(_xlsx_row(1, columns))
        for row_index, row in enumerate(rows, start=2):
            row_xml.append(_xlsx_row(row_index, [_xlsx_value(row.get(column)) for column in columns]))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(row_xml)
        + '</sheetData></worksheet>'
    )


def _xlsx_row(index: int, values: Sequence[Any]) -> str:
    cells = []
    for column_index, value in enumerate(values, start=1):
        ref = f"{_excel_column(column_index)}{index}"
        text = xml_escape("" if value is None else str(value))
        cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
    return f'<row r="{index}">{"".join(cells)}</row>'


def _columns_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(str(key))
                seen.add(str(key))
    return columns


def _sqlite_type(rows: list[dict[str, Any]], column: str) -> str:
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, (bytes, bytearray, memoryview)):
            return "BLOB"
        if isinstance(value, bool):
            return "INTEGER"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, (float, Decimal)):
            return "REAL"
        return "TEXT"
    return "TEXT"


def _sqlite_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, memoryview):
        return bytes(value)
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, UUID)):
        return str(value)
    return json.dumps(json_ready(value), sort_keys=True)


def _xlsx_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, memoryview):
        value = bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(json_ready(value), sort_keys=True)
    if isinstance(value, (datetime, date, UUID, Decimal)):
        return str(value)
    return str(value)


def _export_format(value: str) -> str:
    normalized = str(value or "xlsx").lower()
    if normalized not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format {value!r}; expected one of: {', '.join(sorted(EXPORT_FORMATS))}.")
    return normalized


def _table_name(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in TABLE_EXPORTS:
        raise ValueError(f"Unsupported table export {value!r}.")
    return normalized


def _dataset_name(value: str) -> str:
    normalized = str(value).strip().replace("-", "_")
    if normalized not in DATASET_EXPORTS:
        raise ValueError(f"Unsupported dataset export {value!r}.")
    return normalized


def _limit_offset(filters: dict[str, Any]) -> tuple[int, int]:
    limit = filters.get("limit")
    offset = filters.get("offset")
    resolved_limit = DEFAULT_EXPORT_LIMIT if limit is None else int(limit)
    resolved_limit = min(max(1, resolved_limit), MAX_EXPORT_LIMIT)
    resolved_offset = max(0, int(offset or 0))
    return resolved_limit, resolved_offset


def _sqlite_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "export"


def _sheet_names(names: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(names, start=1):
        base = re.sub(r"[\[\]:*?/\\]+", "_", name)[:31] or f"sheet{index}"
        candidate = base
        suffix = 1
        while candidate.lower() in seen:
            suffix += 1
            trimmed = base[: 31 - len(str(suffix)) - 1]
            candidate = f"{trimmed}_{suffix}"
        seen.add(candidate.lower())
        resolved.append(candidate)
    return resolved


def _excel_column(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _content_types_xml(sheet_count: int) -> str:
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f"{sheets}</Types>"
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{xml_escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def _workbook_rels_xml(sheet_count: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, sheet_count + 1)
    )
    rels += (
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )
