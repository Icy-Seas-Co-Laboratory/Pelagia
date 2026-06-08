from __future__ import annotations

import json
from importlib.resources import files
from urllib.parse import urlparse
from typing import Any, Sequence

from ..config import CoreConfig
from ..domain import ClassificationResultRecord, DetectionRecord, FrameRecord, JobStatus, ModelRecord, PipelineStage, PlannedRun, normalize_collections
from ..utils.serialization import json_ready
from ..utils.validation import validate_schema_name

try:
    import psycopg
    from psycopg import conninfo, sql
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - exercised only when postgres extras are absent
    psycopg = None
    conninfo = None
    sql = None
    dict_row = None


REQUIRED_SCHEMA_TABLES = (
    "runs",
    "raw_assets",
    "frames",
    "detection_candidate",
    "detections_refined",
    "models",
    "classification_results",
    "processing_jobs",
    "processing_job_dependencies",
    "worker_sessions",
    "job_events",
    "logs",
)


def render_schema(schema: str = "seasight") -> str:
    schema = validate_schema_name(schema)
    template = files(__package__).joinpath("sql", "schema.sql").read_text(encoding="utf-8")
    return template.replace("{schema}", schema).strip()


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError("psycopg is required for PostgreSQL operations. Install seasight_core[postgres].")


def _event_level(event_type: str) -> str:
    lowered = event_type.lower()
    if any(token in lowered for token in ("failed", "error", "dead_lettered")):
        return "error"
    if any(token in lowered for token in ("retry", "requeued", "paused", "shutdown")):
        return "warning"
    if any(token in lowered for token in ("heartbeat", "touched", "progress")):
        return "debug"
    return "info"


def _event_message(event_type: str, payload: dict[str, Any]) -> str:
    if event_type.startswith("job."):
        stage = payload.get("stage")
        suffix = f" for {stage}" if stage else ""
        return f"Job event {event_type}{suffix}"
    if event_type.startswith("worker."):
        worker_id = payload.get("worker_id")
        suffix = f" for {worker_id}" if worker_id else ""
        return f"Worker event {event_type}{suffix}"
    return event_type.replace(".", " ")


class PostgresRepository:
    def __init__(self, config: CoreConfig):
        _require_psycopg()
        self.config = config
        self.schema = validate_schema_name(config.database.schema_name)

    def connect(self):
        return psycopg.connect(
            self.config.database.dsn,
            connect_timeout=self.config.database.connect_timeout_s,
            row_factory=dict_row,
            autocommit=False,
        )

    def ensure_database_exists(self) -> None:
        dsn_fields = self._dsn_fields()
        database_name = dsn_fields.get("dbname")
        if not database_name:
            raise RuntimeError("Database DSN must include a database name for initialization.")

        admin_dsn = self._admin_dsn(dsn_fields)
        with psycopg.connect(
            admin_dsn,
            connect_timeout=self.config.database.connect_timeout_s,
            row_factory=dict_row,
            autocommit=True,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
                if cursor.fetchone():
                    return
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name))
                )

    def initialize_schema(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if self.config.database.statement_timeout_ms > 0:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        (str(self.config.database.statement_timeout_ms),),
                    )
                cursor.execute(render_schema(self.schema))
            connection.commit()

    def schema_status(self) -> dict[str, Any]:
        required = list(REQUIRED_SCHEMA_TABLES)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = ANY(%s)
                    """,
                    (self.schema, required),
                )
                existing = sorted(row["table_name"] for row in cursor.fetchall())
        missing = sorted(set(required) - set(existing))
        return {
            "schema": self.schema,
            "ready": not missing,
            "required_tables": required,
            "existing_tables": existing,
            "missing_tables": missing,
        }

    def purge_all(self) -> dict[str, Any]:
        """Delete all Pelagia rows while preserving the schema, indexes, and functions."""
        tables = list(REQUIRED_SCHEMA_TABLES)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                before: dict[str, int] = {}
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) AS count FROM {self.schema}.{table}")
                    before[table] = cursor.fetchone()["count"]
                table_list = ", ".join(f"{self.schema}.{table}" for table in tables)
                cursor.execute(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE")
            connection.commit()
        return {
            "schema": self.schema,
            "tables": before,
            "total_rows_deleted": sum(before.values()),
        }

    def _dsn_fields(self) -> dict[str, Any]:
        fields = conninfo.conninfo_to_dict(self.config.database.dsn)
        dbname = fields.get("dbname")
        if not dbname:
            parsed = urlparse(self.config.database.dsn)
            if parsed.path and parsed.path != "/":
                fields["dbname"] = parsed.path.lstrip("/")
        return fields

    @staticmethod
    def _admin_dsn(fields: dict[str, Any]) -> str:
        admin_fields = dict(fields)
        admin_fields["dbname"] = admin_fields.get("maintenance_db") or "postgres"
        return conninfo.make_conninfo(**admin_fields)

    def list_runs(
        self,
        limit: int = 100,
        offset: int = 0,
        collection: str | None = None,
        run_key: str | None = None,
        instrument: str | None = None,
        source_type: str | None = None,
        status: str | None = None,
        source_path: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if collection:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM {self.schema}.raw_assets assets
                    WHERE assets.run_id = runs.id AND %s = ANY(assets.collections)
                )
                """
            )
            params.append(collection)
        if run_key:
            clauses.append("run_key ILIKE %s")
            params.append(f"%{run_key}%")
        if instrument:
            clauses.append("instrument = %s")
            params.append(instrument)
        if source_type:
            clauses.append("source_type = %s")
            params.append(source_type)
        if status:
            clauses.append("status = %s")
            params.append(status)
        if source_path:
            clauses.append("source_path ILIKE %s")
            params.append(f"%{source_path}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.runs {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    tuple(params),
                )
                return cursor.fetchall()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.runs WHERE id = %s", (run_id,))
                run_row = cursor.fetchone()
                if run_row is None:
                    return None
                cursor.execute(
                    f"SELECT status, COUNT(*) AS count FROM {self.schema}.processing_jobs WHERE run_id = %s GROUP BY status ORDER BY status",
                    (run_id,),
                )
                run_row["job_summary"] = cursor.fetchall()
                return run_row

    def list_jobs(
        self,
        run_id: str | None = None,
        asset_id: str | None = None,
        status: str | None = None,
        stage: str | None = None,
        worker_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        include_details: bool = True,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        if asset_id:
            clauses.append("asset_id = %s")
            params.append(asset_id)
        if status:
            clauses.append("status = %s")
            params.append(status)
        if stage:
            clauses.append("stage = %s")
            params.append(stage)
        if worker_id:
            clauses.append("worker_id = %s")
            params.append(worker_id)
        if cursor:
            try:
                cursor_created_at, cursor_id = cursor.split("|", 1)
            except ValueError:
                cursor_created_at = ""
                cursor_id = ""
            if cursor_created_at and cursor_id:
                clauses.append("(created_at, id) < (%s::timestamptz, %s::uuid)")
                params.extend([cursor_created_at, cursor_id])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = "LIMIT %s" if limit else ""
        offset_sql = "OFFSET %s" if offset else ""
        if limit:
            params.append(limit)
        if offset:
            params.append(max(0, int(offset)))
        select_sql = "*"
        if not include_details:
            select_sql = """
                id,
                run_id,
                asset_id,
                stage,
                status,
                priority,
                attempt_count,
                max_attempts,
                lease_expires_at,
                worker_id,
                summary,
                control_reason,
                error_message,
                created_at,
                updated_at,
                started_at,
                finished_at,
                jsonb_typeof(payload) AS payload_type,
                pg_column_size(payload) AS payload_bytes,
                jsonb_typeof(result) AS result_type,
                pg_column_size(result) AS result_bytes,
                jsonb_typeof(progress) AS progress_type,
                pg_column_size(progress) AS progress_bytes,
                jsonb_array_length(logs_tail) AS logs_tail_count
            """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {select_sql} FROM {self.schema}.processing_jobs {where} ORDER BY created_at DESC, id DESC {limit_sql} {offset_sql}",
                    tuple(params),
                )
                return cursor.fetchall()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.processing_jobs WHERE id = %s", (job_id,))
                return cursor.fetchone()

    def list_worker_sessions(
        self,
        status: str | None = None,
        capability: str | None = None,
        shutdown_requested: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        if capability:
            clauses.append("capabilities ? %s")
            params.append(capability)
        if shutdown_requested is not None:
            clauses.append("shutdown_requested = %s")
            params.append(shutdown_requested)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM {self.schema}.worker_sessions
                    {where}
                    ORDER BY last_heartbeat DESC, updated_at DESC, worker_id ASC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def register_planned_run(self, planned_run: PlannedRun) -> dict[str, Any]:
        manifest = planned_run.manifest
        schema = self.schema

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.runs (id, run_key, instrument, source_path, source_type, metadata, status)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'registered')
                    RETURNING id, run_key, source_path, source_type, status, created_at
                    """,
                    (
                        manifest.run_id,
                        manifest.run_key,
                        manifest.instrument,
                        manifest.source_path,
                        manifest.source_type,
                        json.dumps(json_ready(manifest.metadata)),
                    ),
                )
                run_row = cursor.fetchone()

                cursor.executemany(
                    f"""
                    INSERT INTO {schema}.raw_assets
                    (id, run_id, filename, path, kind, checksum, size_bytes, collections, media_count, metadata)
                    VALUES (%s, %s, %s, %s, %s::{schema}.asset_kind, %s, %s, %s, %s, %s::jsonb)
                    """,
                    [
                        (
                            asset.asset_id,
                            manifest.run_id,
                            asset.filename,
                            asset.path,
                            asset.kind.value,
                            asset.checksum,
                            asset.size_bytes,
                            normalize_collections(asset.collections),
                            asset.media_count,
                            json.dumps(json_ready(asset.metadata)),
                        )
                        for asset in manifest.assets
                    ],
                )

                cursor.executemany(
                    f"""
                    INSERT INTO {schema}.processing_jobs
                    (id, run_id, asset_id, stage, status, priority, attempt_count, max_attempts, payload)
                    VALUES (%s, %s, %s, %s::{schema}.stage_name, %s::{schema}.job_status, %s, 0, %s, %s::jsonb)
                    """,
                    [
                        (
                            job.job_id,
                            job.run_id,
                            job.asset_id,
                            job.stage.value,
                            job.status.value,
                            job.priority,
                            job.max_attempts,
                            json.dumps(json_ready(job.payload)),
                        )
                        for job in planned_run.jobs
                    ],
                )

                dependency_rows = [
                    (job.job_id, dependency)
                    for job in planned_run.jobs
                    for dependency in job.depends_on
                ]
                if dependency_rows:
                    cursor.executemany(
                        f"""
                        INSERT INTO {schema}.processing_job_dependencies (job_id, depends_on_job_id)
                        VALUES (%s, %s)
                        """,
                        dependency_rows,
                    )

                for job in planned_run.jobs:
                    self._append_job_event(
                        cursor,
                        job.job_id,
                        "job.created",
                        {
                            "stage": job.stage.value,
                            "status": job.status.value,
                            "run_id": job.run_id,
                            "asset_id": job.asset_id,
                            "priority": job.priority,
                            "depends_on": list(job.depends_on),
                        },
                    )

            connection.commit()

        return {"run": run_row, "asset_count": len(manifest.assets), "job_count": len(planned_run.jobs)}

    def list_assets(
        self,
        run_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        path: str | None = None,
        checksum: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        media_count: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        if collection:
            clauses.append("%s = ANY(collections)")
            params.append(collection)
        if kind:
            clauses.append("kind = %s")
            params.append(kind)
        if filename:
            clauses.append("filename ILIKE %s")
            params.append(f"%{filename}%")
        if path:
            clauses.append("path ILIKE %s")
            params.append(f"%{path}%")
        if checksum:
            clauses.append("checksum = %s")
            params.append(checksum)
        if min_size_bytes is not None:
            clauses.append("size_bytes >= %s")
            params.append(min_size_bytes)
        if max_size_bytes is not None:
            clauses.append("size_bytes <= %s")
            params.append(max_size_bytes)
        if media_count is not None:
            clauses.append("media_count = %s")
            params.append(media_count)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.raw_assets {where} ORDER BY created_at DESC, filename ASC LIMIT %s OFFSET %s",
                    tuple(params),
                )
                return cursor.fetchall()

    def list_collections(
        self,
        collection: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        having = ""
        params: list[Any] = []
        if collection:
            having = "WHERE collection ILIKE %s"
            params.append(f"%{collection}%")
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM (
                        SELECT collection, COUNT(*) AS asset_count
                        FROM {self.schema}.raw_assets assets
                        CROSS JOIN LATERAL unnest(assets.collections) AS collection
                        GROUP BY collection
                    ) collections
                    {having}
                    ORDER BY collection ASC
                    LIMIT %s OFFSET %s
                    """
                    ,
                    tuple(params),
                )
                return cursor.fetchall()

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.raw_assets WHERE id = %s", (asset_id,))
                return cursor.fetchone()

    def count_frames(self, asset_id: str) -> int:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) AS frame_count FROM {self.schema}.frames WHERE asset_id = %s", (asset_id,))
                row = cursor.fetchone()
        return int(row["frame_count"] if row is not None else 0)

    def replace_frames(self, run_id: str, frames: Sequence[FrameRecord]) -> list[dict[str, Any]]:
        if not frames:
            return []
        asset_id = frames[0].asset_id
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {self.schema}.frames WHERE asset_id = %s", (asset_id,))
                inserted: list[dict[str, Any]] = []
                for frame in frames:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.frames
                        (run_id, asset_id, frame_index, captured_at, width, height,
                         bbox_x, bbox_y, parent_frame_id, source_ref, kvstore_hash, preview_thumbhash,
                         payload_ref, payload_encoding, payload_format, payload_dtype, payload_shape, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        RETURNING *;
                        """,
                        (
                            frame.run_id or run_id,
                            frame.asset_id,
                            frame.frame_index,
                            frame.captured_at,
                            frame.width,
                            frame.height,
                            frame.bbox_x,
                            frame.bbox_y,
                            frame.parent_frame_id,
                            frame.source_ref,
                            frame.kvstore_hash,
                            frame.preview_thumbhash,
                            frame.payload_ref or frame.metadata.get("kvstore_key") or frame.kvstore_hash,
                            frame.payload_encoding or frame.metadata.get("kvstore_encoding"),
                            frame.payload_format or frame.metadata.get("kvstore_format"),
                            frame.payload_dtype or frame.metadata.get("dtype"),
                            json.dumps(json_ready(frame.payload_shape or frame.metadata.get("shape") or [])),
                            json.dumps(json_ready(frame.metadata)),
                        ),
                    )
                    inserted.append(cursor.fetchone())
            connection.commit()
        return inserted

    def list_frames(
        self,
        asset_id: str,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["asset_id = %s"]
        params: list[Any] = [asset_id]
        if start_frame is not None:
            clauses.append("frame_index >= %s")
            params.append(start_frame)
        if end_frame is not None:
            clauses.append("frame_index <= %s")
            params.append(end_frame)
        limit_sql = "LIMIT %s" if limit is not None else ""
        offset_sql = "OFFSET %s" if offset else ""
        if limit is not None:
            params.append(limit)
        if offset:
            params.append(max(0, int(offset)))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM {self.schema}.frames
                    WHERE {' AND '.join(clauses)}
                    ORDER BY frame_index DESC
                    {limit_sql}
                    {offset_sql}
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def get_frame(self, frame_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.frames WHERE id = %s", (frame_id,))
                return cursor.fetchone()

    def get_frame_by_asset_index(self, asset_id: str, frame_index: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM {self.schema}.frames
                    WHERE asset_id = %s AND frame_index = %s
                    """,
                    (asset_id, frame_index),
                )
                return cursor.fetchone()

    def list_frame_records(
        self,
        asset_id: str,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[FrameRecord]:
        return [
            FrameRecord.from_row(row)
            for row in self.list_frames(
                asset_id,
                start_frame=start_frame,
                end_frame=end_frame,
                limit=limit,
                offset=offset,
            )
        ]

    def get_frame_record(self, frame_id: str) -> FrameRecord | None:
        row = self.get_frame(frame_id)
        if row is None:
            return None
        return FrameRecord.from_row(row)

    def update_frame_preprocessed_payload(
        self,
        frame_id: str,
        *,
        kvstore_hash: str,
        preview_thumbhash: bytes,
        payload_ref: str,
        payload_encoding: str,
        payload_format: str,
        payload_dtype: str,
        payload_shape: Sequence[int],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.frames
                    SET
                        preprocessed_kvstore_hash = %s,
                        preprocessed_preview_thumbhash = %s,
                        preprocessed_payload_ref = %s,
                        preprocessed_payload_encoding = %s,
                        preprocessed_payload_format = %s,
                        preprocessed_payload_dtype = %s,
                        preprocessed_payload_shape = %s::jsonb,
                        preprocessed_metadata = %s::jsonb
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (
                        kvstore_hash,
                        preview_thumbhash,
                        payload_ref,
                        payload_encoding,
                        payload_format,
                        payload_dtype,
                        json.dumps(json_ready(list(payload_shape))),
                        json.dumps(json_ready(metadata or {})),
                        frame_id,
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        if row is None:
            raise KeyError(frame_id)
        return row

    def _insert_detection_rows(
        self,
        cursor,
        run_id: str,
        detections: Sequence[DetectionRecord],
    ) -> list[dict[str, Any]]:
        inserted: list[dict[str, Any]] = []
        for detection in detections:
            cursor.execute(
                f"""
                INSERT INTO {self.schema}.detection_candidate
                (run_id, frame_id, roi_index, bbox_x, bbox_y, bbox_w, bbox_h,
                 crop_bbox_x, crop_bbox_y, crop_bbox_w, crop_bbox_h,
                 area, perimeter, major_axis_length, minor_axis_length,
                 min_gray_value, mean_gray_value, roi_payload, mask_payload,
                 roi_encoding, roi_format, roi_dtype, roi_shape,
                 mask_encoding, mask_format, mask_dtype, mask_shape, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s,
                        %s::jsonb, %s::jsonb)
                RETURNING *;
                """,
                (
                    run_id,
                    detection.frame_id,
                    detection.roi_index,
                    detection.bbox_x,
                    detection.bbox_y,
                    detection.bbox_w,
                    detection.bbox_h,
                    detection.crop_bbox_x,
                    detection.crop_bbox_y,
                    detection.crop_bbox_w,
                    detection.crop_bbox_h,
                    detection.area,
                    detection.perimeter,
                    detection.major_axis_length,
                    detection.minor_axis_length,
                    detection.min_gray_value,
                    detection.mean_gray_value,
                    detection.roi_payload,
                    detection.mask_payload,
                    detection.roi_encoding,
                    detection.roi_format,
                    detection.roi_dtype,
                    json.dumps(json_ready(detection.roi_shape)),
                    detection.mask_encoding,
                    detection.mask_format,
                    detection.mask_dtype,
                    json.dumps(json_ready(detection.mask_shape)),
                    json.dumps(json_ready(detection.metadata)),
                ),
            )
            inserted.append(cursor.fetchone())
        return inserted

    def upsert_refined_detections(
        self,
        refined_detections: Sequence[tuple[str, DetectionRecord]],
    ) -> list[dict[str, Any]]:
        if not refined_detections:
            return []
        inserted: list[dict[str, Any]] = []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for candidate_detection_id, detection in refined_detections:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.detections_refined
                        (candidate_detection_id, run_id, frame_id, roi_index, bbox_x, bbox_y, bbox_w, bbox_h,
                         crop_bbox_x, crop_bbox_y, crop_bbox_w, crop_bbox_h,
                         area, perimeter, major_axis_length, minor_axis_length,
                         min_gray_value, mean_gray_value, roi_payload, mask_payload,
                         roi_encoding, roi_format, roi_dtype, roi_shape,
                         mask_encoding, mask_format, mask_dtype, mask_shape, refinement_method, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s,
                                %s, %s::jsonb, %s, %s::jsonb)
                        ON CONFLICT (candidate_detection_id) DO UPDATE SET
                            run_id = EXCLUDED.run_id,
                            frame_id = EXCLUDED.frame_id,
                            roi_index = EXCLUDED.roi_index,
                            bbox_x = EXCLUDED.bbox_x,
                            bbox_y = EXCLUDED.bbox_y,
                            bbox_w = EXCLUDED.bbox_w,
                            bbox_h = EXCLUDED.bbox_h,
                            crop_bbox_x = EXCLUDED.crop_bbox_x,
                            crop_bbox_y = EXCLUDED.crop_bbox_y,
                            crop_bbox_w = EXCLUDED.crop_bbox_w,
                            crop_bbox_h = EXCLUDED.crop_bbox_h,
                            area = EXCLUDED.area,
                            perimeter = EXCLUDED.perimeter,
                            major_axis_length = EXCLUDED.major_axis_length,
                            minor_axis_length = EXCLUDED.minor_axis_length,
                            min_gray_value = EXCLUDED.min_gray_value,
                            mean_gray_value = EXCLUDED.mean_gray_value,
                            roi_payload = EXCLUDED.roi_payload,
                            mask_payload = EXCLUDED.mask_payload,
                            roi_encoding = EXCLUDED.roi_encoding,
                            roi_format = EXCLUDED.roi_format,
                            roi_dtype = EXCLUDED.roi_dtype,
                            roi_shape = EXCLUDED.roi_shape,
                            mask_encoding = EXCLUDED.mask_encoding,
                            mask_format = EXCLUDED.mask_format,
                            mask_dtype = EXCLUDED.mask_dtype,
                            mask_shape = EXCLUDED.mask_shape,
                            refinement_method = EXCLUDED.refinement_method,
                            metadata = EXCLUDED.metadata
                        RETURNING *;
                        """,
                        (
                            candidate_detection_id,
                            detection.run_id,
                            detection.frame_id,
                            detection.roi_index,
                            detection.bbox_x,
                            detection.bbox_y,
                            detection.bbox_w,
                            detection.bbox_h,
                            detection.crop_bbox_x,
                            detection.crop_bbox_y,
                            detection.crop_bbox_w,
                            detection.crop_bbox_h,
                            detection.area,
                            detection.perimeter,
                            detection.major_axis_length,
                            detection.minor_axis_length,
                            detection.min_gray_value,
                            detection.mean_gray_value,
                            detection.roi_payload,
                            detection.mask_payload,
                            detection.roi_encoding,
                            detection.roi_format,
                            detection.roi_dtype,
                            json.dumps(json_ready(detection.roi_shape)),
                            detection.mask_encoding,
                            detection.mask_format,
                            detection.mask_dtype,
                            json.dumps(json_ready(detection.mask_shape)),
                            detection.metadata.get("refinement_method", "identity"),
                            json.dumps(json_ready(detection.metadata)),
                        ),
                    )
                    inserted.append(cursor.fetchone())
            connection.commit()
        return inserted

    def replace_detections(self, run_id: str, asset_id: str, detections: Sequence[DetectionRecord]) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.detection_candidate
                    WHERE frame_id IN (SELECT id FROM {self.schema}.frames WHERE asset_id = %s)
                    """,
                    (asset_id,),
                )
                inserted = self._insert_detection_rows(cursor, run_id, detections)
            connection.commit()
        return inserted

    def replace_frame_detections(
        self,
        run_id: str,
        frame_ids: Sequence[str],
        detections: Sequence[DetectionRecord],
    ) -> list[dict[str, Any]]:
        resolved_frame_ids = [str(frame_id) for frame_id in frame_ids]
        if not resolved_frame_ids:
            return []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.detection_candidate
                    WHERE frame_id = ANY(%s)
                    """,
                    (resolved_frame_ids,),
                )
                inserted = self._insert_detection_rows(cursor, run_id, detections)
            connection.commit()
        return inserted

    def list_detections(
        self,
        asset_id: str | None = None,
        *,
        run_id: str | None = None,
        collection: str | None = None,
        frame_id: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
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
        sort_by: str = "asset_frame",
        sort_dir: str = "desc",
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if asset_id:
            clauses.append("frames.asset_id = %s")
            params.append(asset_id)
        if run_id:
            clauses.append("detections.run_id = %s")
            params.append(run_id)
        if collection:
            clauses.append("%s = ANY(assets.collections)")
            params.append(collection)
        if frame_id:
            clauses.append("detections.frame_id = %s")
            params.append(frame_id)
        if start_frame is not None:
            clauses.append("frames.frame_index >= %s")
            params.append(start_frame)
        if end_frame is not None:
            clauses.append("frames.frame_index <= %s")
            params.append(end_frame)
        if roi_index is not None:
            clauses.append("detections.roi_index = %s")
            params.append(roi_index)

        range_filters = [
            ("detections.bbox_x", ">=", min_bbox_x),
            ("detections.bbox_x", "<=", max_bbox_x),
            ("detections.bbox_y", ">=", min_bbox_y),
            ("detections.bbox_y", "<=", max_bbox_y),
            ("detections.bbox_w", ">=", min_bbox_w),
            ("detections.bbox_w", "<=", max_bbox_w),
            ("detections.bbox_h", ">=", min_bbox_h),
            ("detections.bbox_h", "<=", max_bbox_h),
            ("detections.area", ">=", min_area),
            ("detections.area", "<=", max_area),
            ("detections.perimeter", ">=", min_perimeter),
            ("detections.perimeter", "<=", max_perimeter),
        ]
        for column, operator, value in range_filters:
            if value is not None:
                clauses.append(f"{column} {operator} %s")
                params.append(value)

        exact_filters = [
            ("detections.roi_encoding", roi_encoding),
            ("detections.roi_format", roi_format),
            ("detections.mask_encoding", mask_encoding),
            ("detections.mask_format", mask_format),
        ]
        for column, value in exact_filters:
            if value:
                clauses.append(f"{column} = %s")
                params.append(value)

        direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"
        sort_key = str(sort_by or "asset_frame").lower()
        order_by_options = {
            "area": f"detections.area {direction} NULLS LAST, frames.frame_index DESC, detections.roi_index ASC",
            "byte_size": f"octet_length(detections.roi_payload) {direction} NULLS LAST, frames.frame_index DESC, detections.roi_index ASC",
            "id": f"detections.id {direction}",
            "asset_frame": f"assets.filename {direction} NULLS LAST, frames.frame_index {direction}, detections.roi_index {direction}",
        }
        order_by = order_by_options.get(sort_key, order_by_options["asset_frame"])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = "LIMIT %s" if limit is not None else ""
        offset_sql = "OFFSET %s" if offset else ""
        if limit is not None:
            params.append(limit)
        if offset:
            params.append(max(0, int(offset)))

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        detections.*,
                        frames.asset_id,
                        frames.frame_index,
                        assets.filename AS asset_filename
                    FROM {self.schema}.detection_candidate detections
                    JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    {where}
                    ORDER BY {order_by}
                    {limit_sql}
                    {offset_sql}
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def get_detection(self, detection_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        detections.*,
                        frames.asset_id,
                        frames.frame_index,
                        assets.filename AS asset_filename
                    FROM {self.schema}.detection_candidate detections
                    JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE detections.id = %s
                    """,
                    (detection_id,),
                )
                return cursor.fetchone()

    def list_detection_records(self, asset_id: str) -> list[DetectionRecord]:
        return [DetectionRecord.from_row(row) for row in self.list_detections(asset_id)]

    def list_asset_detection_stats(
        self,
        *,
        run_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        min_detection_count: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses = []
        params: list[Any] = []
        if run_id:
            clauses.append("assets.run_id = %s")
            params.append(run_id)
        if collection:
            clauses.append("%s = ANY(assets.collections)")
            params.append(collection)
        if kind:
            clauses.append("assets.kind = %s")
            params.append(kind)
        if filename:
            clauses.append("assets.filename ILIKE %s")
            params.append(f"%{filename}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        having = "HAVING COUNT(detections.id) >= %s" if min_detection_count is not None else ""
        aggregate_params = tuple(params + ([] if min_detection_count is None else [min_detection_count]))

        query = f"""
            WITH asset_detection_counts AS (
                SELECT
                    assets.id AS asset_id,
                    assets.run_id,
                    assets.filename,
                    assets.kind,
                    assets.collections,
                    COUNT(DISTINCT frames.id) AS frame_count,
                    COUNT(detections.id) AS detection_count
                FROM {self.schema}.raw_assets assets
                LEFT JOIN {self.schema}.frames frames ON frames.asset_id = assets.id
                LEFT JOIN {self.schema}.detection_candidate detections ON detections.frame_id = frames.id
                {where}
                GROUP BY assets.id, assets.run_id, assets.filename, assets.kind, assets.collections
                {having}
            )
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query
                    + """
                    SELECT
                        COUNT(*) AS total_asset_count,
                        COUNT(*) FILTER (WHERE detection_count > 0) AS identified_asset_count,
                        COALESCE(SUM(detection_count), 0) AS total_detection_count
                    FROM asset_detection_counts
                    """,
                    aggregate_params,
                )
                summary = cursor.fetchone()
                cursor.execute(
                    query
                    + """
                    SELECT *
                    FROM asset_detection_counts
                    ORDER BY detection_count DESC, filename ASC
                    LIMIT %s OFFSET %s
                    """,
                    aggregate_params + (limit, max(0, int(offset))),
                )
                assets = cursor.fetchall()

        return {
            "summary": {
                "total_asset_count": summary["total_asset_count"],
                "identified_asset_count": summary["identified_asset_count"],
                "total_detection_count": summary["total_detection_count"],
            },
            "assets": assets,
        }

    def register_model(self, model: ModelRecord) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.models
                    (model_key, model_name, version, task, artifact_uri, labels, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (model_key) DO UPDATE SET
                        model_name = EXCLUDED.model_name,
                        version = EXCLUDED.version,
                        task = EXCLUDED.task,
                        artifact_uri = EXCLUDED.artifact_uri,
                        labels = EXCLUDED.labels,
                        metadata = EXCLUDED.metadata
                    RETURNING *;
                    """,
                    (
                        model.model_key,
                        model.model_name,
                        model.version,
                        model.task,
                        model.artifact_uri,
                        json.dumps(model.labels),
                        json.dumps(json_ready(model.metadata)),
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def list_models(
        self,
        model_key: str | None = None,
        model_name: str | None = None,
        version: str | None = None,
        task: str | None = None,
        artifact_uri: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if model_key:
            clauses.append("model_key ILIKE %s")
            params.append(f"%{model_key}%")
        if model_name:
            clauses.append("model_name ILIKE %s")
            params.append(f"%{model_name}%")
        if version:
            clauses.append("version = %s")
            params.append(version)
        if task:
            clauses.append("task = %s")
            params.append(task)
        if artifact_uri:
            clauses.append("artifact_uri ILIKE %s")
            params.append(f"%{artifact_uri}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.models {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    tuple(params),
                )
                return cursor.fetchall()

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.models WHERE id = %s", (model_id,))
                return cursor.fetchone()

    def get_model_by_key(self, model_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.models WHERE model_key = %s", (model_key,))
                return cursor.fetchone()

    def replace_classification_results(
        self,
        model_id: str,
        detection_ids: Sequence[str],
        results: Sequence[ClassificationResultRecord],
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if detection_ids:
                    cursor.execute(
                        f"""
                        DELETE FROM {self.schema}.classification_results
                        WHERE model_id = %s AND detection_id = ANY(%s)
                        """,
                        (model_id, list(detection_ids)),
                    )
                inserted: list[dict[str, Any]] = []
                for result in results:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.classification_results
                        (detection_id, model_id, label, score, scores, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                        RETURNING *;
                        """,
                        (
                            result.detection_id,
                            result.model_id,
                            result.label,
                            result.score,
                            json.dumps(json_ready(result.scores)),
                            json.dumps(json_ready(result.embedding)),
                            json.dumps(json_ready(result.metadata)),
                        ),
                    )
                    inserted.append(cursor.fetchone())
            connection.commit()
        return inserted

    def create_job(
        self,
        stage: PipelineStage | str,
        *,
        run_id: str | None = None,
        asset_id: str | None = None,
        status: JobStatus | str = JobStatus.QUEUED,
        priority: int | None = None,
        max_attempts: int | None = None,
        payload: dict[str, Any] | None = None,
        depends_on: Sequence[str] | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        stage_value = stage.value if isinstance(stage, PipelineStage) else stage
        status_value = status.value if isinstance(status, JobStatus) else status
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.processing_jobs
                    (run_id, asset_id, stage, status, priority, attempt_count, max_attempts, payload, summary)
                    VALUES (%s, %s, %s::{self.schema}.stage_name, %s::{self.schema}.job_status, %s, 0, %s, %s::jsonb, %s)
                    RETURNING *;
                    """,
                    (
                        run_id,
                        asset_id,
                        stage_value,
                        status_value,
                        priority if priority is not None else self.config.queue.default_priority,
                        max_attempts if max_attempts is not None else self.config.queue.max_attempts,
                        json.dumps(json_ready(payload or {})),
                        summary,
                    ),
                )
                row = cursor.fetchone()
                for dependency in depends_on or []:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.processing_job_dependencies (job_id, depends_on_job_id)
                        VALUES (%s, %s)
                        """,
                        (row["id"], dependency),
                    )
                self._append_job_event(
                    cursor,
                    row["id"],
                    "job.created",
                    {
                        "stage": row["stage"],
                        "status": row["status"],
                        "run_id": row.get("run_id"),
                        "asset_id": row.get("asset_id"),
                        "priority": row.get("priority"),
                        "depends_on": [str(dependency) for dependency in depends_on or []],
                    },
                )
            connection.commit()
        return row

    def update_job_payload(self, job_id: str, payload: dict[str, Any], summary: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET payload = %s::jsonb,
                        summary = COALESCE(%s, summary),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (json.dumps(json_ready(payload)), summary, job_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(
                        cursor,
                        job_id,
                        "job.payload_updated",
                        {"summary": summary, "payload": payload},
                    )
            connection.commit()
        return row

    def update_job_progress(
        self,
        job_id: str,
        progress: dict[str, Any],
        *,
        summary: str | None = None,
        log_message: str | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_job(job_id)
        if current is None:
            return None
        logs_tail = list(current.get("logs_tail") or [])
        if log_message:
            logs_tail.append(log_message)
            logs_tail = logs_tail[-20:]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET progress = %s::jsonb,
                        summary = COALESCE(%s, summary),
                        logs_tail = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (
                        json.dumps(json_ready(progress)),
                        summary,
                        json.dumps(json_ready(logs_tail)),
                        job_id,
                    ),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(
                        cursor,
                        job_id,
                        "job.progress_updated",
                        {
                            "progress": progress,
                            "summary": summary,
                            "log_message": log_message,
                        },
                    )
            connection.commit()
        return row

    def append_job_event(self, job_id: str | None, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                row = self._append_job_event(cursor, job_id, event_type, payload)
            connection.commit()
        return row

    def append_log(
        self,
        *,
        event_type: str,
        message: str | None = None,
        level: str = "info",
        logger: str = "pelagia",
        run_id: str | None = None,
        asset_id: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        request_id: str | None = None,
        duration_ms: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                row = self._append_log(
                    cursor,
                    event_type=event_type,
                    message=message,
                    level=level,
                    logger=logger,
                    run_id=run_id,
                    asset_id=asset_id,
                    job_id=job_id,
                    worker_id=worker_id,
                    request_id=request_id,
                    duration_ms=duration_ms,
                    payload=payload,
                )
            connection.commit()
        return row

    def _append_log(
        self,
        cursor,
        *,
        event_type: str,
        message: str | None = None,
        level: str = "info",
        logger: str = "pelagia",
        run_id: str | None = None,
        asset_id: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        request_id: str | None = None,
        duration_ms: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cursor.execute(
            f"""
            INSERT INTO {self.schema}.logs
            (level, logger, event_type, message, run_id, asset_id, job_id, worker_id, request_id, duration_ms, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING *;
            """,
            (
                str(level).lower(),
                logger,
                event_type,
                message,
                run_id,
                asset_id,
                job_id,
                worker_id,
                request_id,
                duration_ms,
                json.dumps(json_ready(payload or {})),
            ),
        )
        return cursor.fetchone()

    def _append_job_event(
        self,
        cursor,
        job_id: str | None,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cursor.execute(
            f"""
            INSERT INTO {self.schema}.job_events (job_id, event_type, payload)
            VALUES (%s, %s, %s::jsonb)
            RETURNING *;
            """,
            (job_id, event_type, json.dumps(json_ready(payload or {}))),
        )
        row = cursor.fetchone()
        log_payload = dict(payload or {})
        run_id = log_payload.get("run_id")
        asset_id = log_payload.get("asset_id")
        worker_id = log_payload.get("worker_id")
        if job_id is not None and (run_id is None or asset_id is None):
            cursor.execute(
                f"""
                SELECT run_id, asset_id
                FROM {self.schema}.processing_jobs
                WHERE id = %s
                """,
                (job_id,),
            )
            job_row = cursor.fetchone()
            if job_row is not None:
                run_id = run_id or job_row.get("run_id")
                asset_id = asset_id or job_row.get("asset_id")
        self._append_log(
            cursor,
            event_type=event_type,
            message=_event_message(event_type, log_payload),
            level=_event_level(event_type),
            logger="pelagia.jobs",
            run_id=run_id,
            asset_id=asset_id,
            job_id=job_id,
            worker_id=worker_id,
            payload=log_payload,
        )
        return row

    def _append_worker_event(
        self,
        cursor,
        event_type: str,
        worker_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_payload = {"worker_id": worker_id}
        resolved_payload.update(payload or {})
        return self._append_job_event(cursor, None, event_type, resolved_payload)

    def list_job_events(
        self,
        *,
        after_id: int | None = None,
        run_id: str | None = None,
        job_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        joins = ""
        if after_id is not None:
            clauses.append("events.id > %s")
            params.append(after_id)
        if job_id:
            clauses.append("events.job_id = %s")
            params.append(job_id)
        if run_id:
            joins = f"LEFT JOIN {self.schema}.processing_jobs jobs ON jobs.id = events.job_id"
            clauses.append("jobs.run_id = %s")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT events.*
                    FROM {self.schema}.job_events events
                    {joins}
                    {where}
                    ORDER BY events.id DESC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def list_logs(
        self,
        *,
        after_id: int | None = None,
        before_id: int | None = None,
        level: str | None = None,
        event_type: str | None = None,
        logger: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        request_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if after_id is not None:
            clauses.append("id > %s")
            params.append(after_id)
        if before_id is not None:
            clauses.append("id < %s")
            params.append(before_id)
        if level:
            clauses.append("level = %s")
            params.append(str(level).lower())
        if event_type:
            clauses.append("event_type = %s")
            params.append(event_type)
        if logger:
            clauses.append("logger = %s")
            params.append(logger)
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        if asset_id:
            clauses.append("asset_id = %s")
            params.append(asset_id)
        if job_id:
            clauses.append("job_id = %s")
            params.append(job_id)
        if worker_id:
            clauses.append("worker_id = %s")
            params.append(worker_id)
        if request_id:
            clauses.append("request_id = %s")
            params.append(request_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM {self.schema}.logs
                    {where}
                    ORDER BY id DESC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def set_job_priority(self, job_id: str, priority: int, reason: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET priority = %s,
                        control_reason = COALESCE(%s, control_reason),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (priority, reason, job_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(
                        cursor,
                        job_id,
                        "job.priority_updated",
                        {"priority": priority, "reason": reason},
                    )
            connection.commit()
        return row

    def pause_job(self, job_id: str, reason: str | None = None) -> dict[str, Any] | None:
        current = self.get_job(job_id)
        if current is None:
            return None
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if current["status"] == JobStatus.QUEUED.value:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.processing_jobs
                        SET status = 'paused',
                            control_reason = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING *;
                        """,
                        (reason, job_id),
                    )
                elif current["status"] == JobStatus.LEASED.value:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.processing_jobs
                        SET control_reason = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING *;
                        """,
                        (f"pause_requested:{reason or 'user_requested'}", job_id),
                    )
                else:
                    cursor.execute(f"SELECT * FROM {self.schema}.processing_jobs WHERE id = %s", (job_id,))
                row = cursor.fetchone()
                if row is not None:
                    if current["status"] == JobStatus.QUEUED.value:
                        self._append_job_event(
                            cursor,
                            job_id,
                            "job.paused",
                            {"reason": reason, "previous_status": current["status"]},
                        )
                    elif current["status"] == JobStatus.LEASED.value:
                        self._append_job_event(
                            cursor,
                            job_id,
                            "job.pause_requested",
                            {"reason": reason, "previous_status": current["status"]},
                        )
            connection.commit()
        return row

    def finalize_paused_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET status = 'paused',
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (job_id,),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(cursor, job_id, "job.paused", {"finalized": True})
            connection.commit()
        return row

    def resume_job(self, job_id: str, reason: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET status = 'queued',
                        control_reason = %s,
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        finished_at = NULL,
                        updated_at = NOW()
                    WHERE id = %s AND status = 'paused'
                    RETURNING *;
                    """,
                    (reason, job_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(cursor, job_id, "job.resumed", {"reason": reason})
            connection.commit()
        return row

    def get_status_summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT status, COUNT(*) AS count
                    FROM {self.schema}.processing_jobs
                    GROUP BY status
                    """
                )
                job_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
                cursor.execute(f"SELECT COUNT(*) AS count FROM {self.schema}.worker_sessions")
                total_workers = cursor.fetchone()["count"]
                cursor.execute(
                    f"""
                    SELECT COUNT(*) AS count
                    FROM {self.schema}.worker_sessions
                    WHERE last_heartbeat >= NOW() - (%s * INTERVAL '1 second')
                    """,
                    (self.config.queue.heartbeat_interval_seconds * 2,),
                )
                online_workers = cursor.fetchone()["count"]
                cursor.execute(
                    f"""
                    SELECT COUNT(*) AS count
                    FROM {self.schema}.worker_sessions
                    WHERE status = 'working'
                    """
                )
                busy_workers = cursor.fetchone()["count"]
        return {
            "queue": job_counts,
            "workers": {
                "total": total_workers,
                "online": online_workers,
                "busy": busy_workers,
            },
        }

    def touch_worker(
        self,
        worker_id: str,
        status: str,
        leased_job_id: str | None = None,
        capabilities: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        pid: int | None = None,
        shutdown_requested: bool | None = None,
    ) -> dict[str, Any]:
        shutdown_sql = (
            "COALESCE(EXCLUDED.shutdown_requested, worker_sessions.shutdown_requested)"
            if shutdown_requested is None
            else "EXCLUDED.shutdown_requested"
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.worker_sessions
                    (worker_id, pid, status, leased_job_id, capabilities, metadata, shutdown_requested, last_heartbeat)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, NOW())
                    ON CONFLICT (worker_id) DO UPDATE SET
                        pid = COALESCE(EXCLUDED.pid, worker_sessions.pid),
                        status = EXCLUDED.status,
                        leased_job_id = EXCLUDED.leased_job_id,
                        capabilities = EXCLUDED.capabilities,
                        metadata = EXCLUDED.metadata,
                        shutdown_requested = {shutdown_sql},
                        last_heartbeat = NOW()
                    RETURNING *;
                    """,
                    (
                        worker_id,
                        pid,
                        status,
                        leased_job_id,
                        json.dumps(list(capabilities or [])),
                        json.dumps(json_ready(metadata or {})),
                        False if shutdown_requested is None else shutdown_requested,
                    ),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_worker_event(
                        cursor,
                        "worker.touched",
                        worker_id,
                        {
                            "pid": row.get("pid"),
                            "status": row.get("status"),
                            "leased_job_id": row.get("leased_job_id"),
                            "capabilities": row.get("capabilities"),
                            "shutdown_requested": row.get("shutdown_requested"),
                        },
                    )
            connection.commit()
        return row

    def get_worker_session(self, worker_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.worker_sessions WHERE worker_id = %s",
                    (worker_id,),
                )
                return cursor.fetchone()

    def request_worker_shutdown(self, worker_id: str, reason: str | None = None) -> dict[str, Any] | None:
        metadata = {"shutdown_reason": reason} if reason else {}
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.worker_sessions
                    SET shutdown_requested = true,
                        metadata = metadata || %s::jsonb,
                        updated_at = NOW()
                    WHERE worker_id = %s
                    RETURNING *;
                    """,
                    (json.dumps(json_ready(metadata)), worker_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_worker_event(
                        cursor,
                        "worker.shutdown_requested",
                        worker_id,
                        {"reason": reason, "pid": row.get("pid"), "status": row.get("status")},
                    )
            connection.commit()
        return row

    def heartbeat(self, worker_id: str, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                        updated_at = NOW()
                    WHERE id = %s AND worker_id = %s AND status = 'leased'
                    RETURNING *;
                    """,
                    (self.config.queue.lease_seconds, job_id, worker_id),
                )
                job_row = cursor.fetchone()
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.worker_sessions
                    SET status = 'working',
                        leased_job_id = %s,
                        last_heartbeat = NOW()
                    WHERE worker_id = %s
                    RETURNING *;
                    """,
                    (job_id, worker_id),
                )
                if job_row is not None:
                    self._append_job_event(
                        cursor,
                        job_id,
                        "job.heartbeat",
                        {"worker_id": worker_id},
                    )
                    self._append_worker_event(
                        cursor,
                        "worker.heartbeat",
                        worker_id,
                        {"job_id": job_id},
                    )
            connection.commit()
        return job_row

    def requeue_expired_jobs(self) -> dict[str, int]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH expired AS (
                        SELECT id, attempt_count, max_attempts
                        FROM {self.schema}.processing_jobs
                        WHERE status = 'leased' AND lease_expires_at < NOW()
                    )
                    UPDATE {self.schema}.processing_jobs jobs
                    SET
                        status = CASE WHEN expired.attempt_count >= expired.max_attempts THEN 'dead_lettered'::{self.schema}.job_status
                                      ELSE 'queued'::{self.schema}.job_status END,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        control_reason = NULL,
                        error_message = CASE WHEN expired.attempt_count >= expired.max_attempts
                                             THEN COALESCE(jobs.error_message, 'Lease expired and job reached max attempts')
                                             ELSE jobs.error_message END,
                        finished_at = CASE WHEN expired.attempt_count >= expired.max_attempts THEN NOW() ELSE NULL END,
                        updated_at = NOW()
                    FROM expired
                    WHERE jobs.id = expired.id
                    RETURNING jobs.id, jobs.status, jobs.attempt_count, jobs.max_attempts;
                    """
                )
                rows = cursor.fetchall()
                for row in rows:
                    event_type = (
                        "job.dead_lettered"
                        if row["status"] == JobStatus.DEAD_LETTERED.value
                        else "job.requeued"
                    )
                    self._append_job_event(
                        cursor,
                        row["id"],
                        event_type,
                        {
                            "reason": "lease_expired",
                            "attempt_count": row.get("attempt_count"),
                            "max_attempts": row.get("max_attempts"),
                        },
                    )
            connection.commit()
        queued = sum(1 for row in rows if row["status"] == "queued")
        dead_lettered = sum(1 for row in rows if row["status"] == "dead_lettered")
        return {"queued": queued, "dead_lettered": dead_lettered}

    def claim_jobs(
        self,
        worker_id: str,
        stages: Sequence[PipelineStage] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        limit = limit or self.config.queue.max_claim_count
        params: list[Any] = []
        stage_clause = ""

        if stages:
            placeholders = ", ".join(["%s"] * len(stages))
            stage_clause = f"AND jobs.stage IN ({placeholders})"
            params.extend(stage.value for stage in stages)

        params.extend([limit, worker_id, self.config.queue.lease_seconds])

        query = f"""
            WITH candidate AS (
                SELECT jobs.id
                FROM {self.schema}.processing_jobs jobs
                WHERE jobs.status = 'queued'
                  {stage_clause}
                  AND NOT EXISTS (
                      SELECT 1
                      FROM {self.schema}.processing_job_dependencies deps
                      JOIN {self.schema}.processing_jobs upstream ON upstream.id = deps.depends_on_job_id
                      WHERE deps.job_id = jobs.id
                        AND upstream.status <> 'succeeded'
                  )
                ORDER BY jobs.priority ASC, jobs.created_at ASC
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            UPDATE {self.schema}.processing_jobs AS jobs
            SET
                status = 'leased',
                worker_id = %s,
                lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                control_reason = NULL,
                attempt_count = attempt_count + 1,
                started_at = COALESCE(started_at, NOW()),
                updated_at = NOW()
            FROM candidate
            WHERE jobs.id = candidate.id
            RETURNING jobs.*;
        """

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                for row in rows:
                    self._append_job_event(
                        cursor,
                        row["id"],
                        "job.leased",
                        {
                            "worker_id": worker_id,
                            "stage": row.get("stage"),
                            "attempt_count": row.get("attempt_count"),
                            "lease_expires_at": row.get("lease_expires_at"),
                        },
                    )
            connection.commit()

        return rows

    def complete_job(self, job_id: str, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET
                        status = 'succeeded',
                        result = %s::jsonb,
                        error_message = NULL,
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        control_reason = NULL,
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (json.dumps(json_ready(result or {})), job_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(
                        cursor,
                        job_id,
                        "job.completed",
                        {"result": result or {}},
                    )
            connection.commit()
        return row

    def record_failure(
        self,
        job_id: str,
        error_message: str,
        result: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> dict[str, Any] | None:
        current = self.get_job(job_id)
        if current is None:
            return None

        if retryable and current["attempt_count"] < current["max_attempts"]:
            next_status = JobStatus.QUEUED.value
            finished_at_sql = "NULL"
        else:
            next_status = JobStatus.DEAD_LETTERED.value if retryable else JobStatus.FAILED.value
            finished_at_sql = "NOW()"

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET
                        status = %s::{self.schema}.job_status,
                        result = %s::jsonb,
                        error_message = %s,
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        control_reason = NULL,
                        finished_at = {finished_at_sql},
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (
                        next_status,
                        json.dumps(json_ready(result or {})),
                        error_message,
                        job_id,
                    ),
                )
                row = cursor.fetchone()
                if row is not None:
                    if next_status == JobStatus.QUEUED.value:
                        event_type = "job.failed_retryable"
                    elif next_status == JobStatus.DEAD_LETTERED.value:
                        event_type = "job.dead_lettered"
                    else:
                        event_type = "job.failed"
                    self._append_job_event(
                        cursor,
                        job_id,
                        event_type,
                        {
                            "error_message": error_message,
                            "retryable": retryable,
                            "next_status": next_status,
                            "result": result or {},
                        },
                    )
            connection.commit()
        return row

    def fail_job(self, job_id: str, error_message: str, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return self.record_failure(job_id=job_id, error_message=error_message, result=result, retryable=False)

    def retry_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET
                        status = 'queued',
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        control_reason = NULL,
                        error_message = NULL,
                        finished_at = NULL,
                        updated_at = NOW()
                    WHERE id = %s AND status IN ('failed', 'dead_lettered', 'cancelled')
                    RETURNING *;
                    """,
                    (job_id,),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(cursor, job_id, "job.retried", {})
            connection.commit()
        return row

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET status = 'cancelled',
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        control_reason = NULL,
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE run_id = %s AND status IN ('queued', 'leased', 'paused')
                    RETURNING id, status
                    """,
                    (run_id,),
                )
                job_rows = cursor.fetchall()
                for job_row in job_rows:
                    self._append_job_event(
                        cursor,
                        job_row["id"],
                        "job.cancelled",
                        {"run_id": run_id},
                    )
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.runs
                    SET status = 'cancelled', updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (run_id,),
                )
                run_row = cursor.fetchone()
            connection.commit()
        return run_row

    def reconcile_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT status, COUNT(*) AS count
                    FROM {self.schema}.processing_jobs
                    WHERE run_id = %s
                    GROUP BY status
                    """,
                    (run_id,),
                )
                counts = {row["status"]: row["count"] for row in cursor.fetchall()}

                if counts.get("dead_lettered") or counts.get("failed"):
                    run_status = "failed"
                elif counts.get("cancelled"):
                    run_status = "cancelled"
                elif counts.get("leased"):
                    run_status = "running"
                elif counts.get("paused"):
                    run_status = "paused"
                elif counts.get("queued"):
                    run_status = "queued"
                elif counts and all(status == "succeeded" for status in counts):
                    run_status = "completed"
                else:
                    run_status = "registered"

                cursor.execute(
                    f"""
                    UPDATE {self.schema}.runs
                    SET status = %s, updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (run_status, run_id),
                )
                run_row = cursor.fetchone()
            connection.commit()
        return run_row
