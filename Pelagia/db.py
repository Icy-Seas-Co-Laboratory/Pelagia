from __future__ import annotations

import json
from urllib.parse import urlparse
from typing import Any, Sequence

from .config import CoreConfig
from .domain import ClassificationResultRecord, DetectionRecord, FrameRecord, JobStatus, ModelRecord, PipelineStage, PlannedRun
from .util import json_ready, validate_schema_name

try:
    import psycopg
    from psycopg import conninfo, sql
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - exercised only when postgres extras are absent
    psycopg = None
    conninfo = None
    sql = None
    dict_row = None


def render_schema(schema: str = "seasight") -> str:
    schema = validate_schema_name(schema)
    return f"""
CREATE SCHEMA IF NOT EXISTS {schema};
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'asset_kind' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.asset_kind AS ENUM ('video', 'image', 'image_sequence');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'stage_name' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.stage_name AS ENUM ('ingest_run', 'extract_frames', 'segment', 'classify', 'publish', 'train_model', 'io_import', 'io_export', 'io_upload', 'io_download');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'job_status' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.job_status AS ENUM ('queued', 'leased', 'paused', 'succeeded', 'failed', 'cancelled', 'dead_lettered');
    END IF;
END $$;

ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'train_model';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_import';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_export';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_upload';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_download';

ALTER TYPE {schema}.job_status ADD VALUE IF NOT EXISTS 'paused';

CREATE OR REPLACE FUNCTION {schema}.set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS {schema}.runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_key text NOT NULL UNIQUE,
    instrument text NOT NULL,
    source_path text NOT NULL,
    source_type text NOT NULL,
    status text NOT NULL DEFAULT 'registered',
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.raw_assets (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    asset_key text NOT NULL,
    path text NOT NULL,
    kind {schema}.asset_kind NOT NULL,
    checksum text NOT NULL,
    size_bytes bigint NOT NULL,
    media_count integer NOT NULL DEFAULT 1,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, asset_key)
);

CREATE TABLE IF NOT EXISTS {schema}.frames (
    id bigserial PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    asset_id uuid NOT NULL REFERENCES {schema}.raw_assets(id) ON DELETE CASCADE,
    frame_index integer NOT NULL,
    captured_at timestamptz,
    width integer NOT NULL,
    height integer NOT NULL,
    source_ref text,
    frame_hash text NOT NULL,
    frame_png bytea NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (asset_id, frame_index)
);

CREATE TABLE IF NOT EXISTS {schema}.detections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    frame_id bigint NOT NULL REFERENCES {schema}.frames(id) ON DELETE CASCADE,
    roi_index integer NOT NULL,
    bbox_x integer NOT NULL,
    bbox_y integer NOT NULL,
    bbox_w integer NOT NULL,
    bbox_h integer NOT NULL,
    area double precision,
    perimeter double precision,
    major_axis_length double precision,
    minor_axis_length double precision,
    min_gray_value integer,
    mean_gray_value double precision,
    crop_png bytea,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (frame_id, roi_index)
);

CREATE TABLE IF NOT EXISTS {schema}.models (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_key text NOT NULL UNIQUE,
    model_name text NOT NULL,
    version text NOT NULL,
    task text NOT NULL DEFAULT 'classification',
    artifact_uri text,
    labels jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.classification_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id uuid NOT NULL REFERENCES {schema}.detections(id) ON DELETE CASCADE,
    model_id uuid NOT NULL REFERENCES {schema}.models(id) ON DELETE CASCADE,
    label text,
    score double precision,
    scores jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    embedding jsonb,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (detection_id, model_id)
);

CREATE TABLE IF NOT EXISTS {schema}.processing_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    asset_id uuid REFERENCES {schema}.raw_assets(id) ON DELETE CASCADE,
    stage {schema}.stage_name NOT NULL,
    status {schema}.job_status NOT NULL DEFAULT 'queued',
    priority integer NOT NULL DEFAULT 100,
    attempt_count integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL DEFAULT 3,
    lease_expires_at timestamptz,
    worker_id text,
    payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    result jsonb,
    progress jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    logs_tail jsonb NOT NULL DEFAULT '[]'::jsonb,
    summary text,
    control_reason text,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    started_at timestamptz,
    finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS {schema}.processing_job_dependencies (
    job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    depends_on_job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, depends_on_job_id)
);

CREATE TABLE IF NOT EXISTS {schema}.worker_sessions (
    worker_id text PRIMARY KEY,
    status text NOT NULL DEFAULT 'idle',
    leased_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    capabilities jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    last_heartbeat timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.job_events (
    id bigserial PRIMARY KEY,
    job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    event_type text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_{schema}_raw_assets_run_id ON {schema}.raw_assets (run_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frames_asset_id ON {schema}.frames (asset_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_{schema}_detections_frame_id ON {schema}.detections (frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_classification_results_detection_id ON {schema}.classification_results (detection_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_status ON {schema}.processing_jobs (status, stage, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_run_id ON {schema}.processing_jobs (run_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_job_dependencies_depends_on ON {schema}.processing_job_dependencies (depends_on_job_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_job_events_job_id ON {schema}.job_events (job_id, id);

DROP TRIGGER IF EXISTS trg_runs_updated_at ON {schema}.runs;
CREATE TRIGGER trg_runs_updated_at
BEFORE UPDATE ON {schema}.runs
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();

DROP TRIGGER IF EXISTS trg_processing_jobs_updated_at ON {schema}.processing_jobs;
CREATE TRIGGER trg_processing_jobs_updated_at
BEFORE UPDATE ON {schema}.processing_jobs
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();

DROP TRIGGER IF EXISTS trg_worker_sessions_updated_at ON {schema}.worker_sessions;
CREATE TRIGGER trg_worker_sessions_updated_at
BEFORE UPDATE ON {schema}.worker_sessions
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();
""".strip()


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError("psycopg is required for PostgreSQL operations. Install seasight_core[postgres].")


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
                    cursor.execute("SET statement_timeout = %s", (self.config.database.statement_timeout_ms,))
                cursor.execute(render_schema(self.schema))
            connection.commit()

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

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.runs ORDER BY created_at DESC LIMIT %s",
                    (limit,),
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
        status: str | None = None,
        stage: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        if status:
            clauses.append("status = %s")
            params.append(status)
        if stage:
            clauses.append("stage = %s")
            params.append(stage)
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
        if limit:
            params.append(limit)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.processing_jobs {where} ORDER BY created_at DESC, id DESC {limit_sql}",
                    tuple(params),
                )
                return cursor.fetchall()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.processing_jobs WHERE id = %s", (job_id,))
                return cursor.fetchone()

    def list_worker_sessions(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.worker_sessions ORDER BY worker_id ASC")
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
                    (id, run_id, asset_key, path, kind, checksum, size_bytes, media_count, metadata)
                    VALUES (%s, %s, %s, %s, %s::{schema}.asset_kind, %s, %s, %s, %s::jsonb)
                    """,
                    [
                        (
                            asset.asset_id,
                            manifest.run_id,
                            asset.asset_key,
                            asset.path,
                            asset.kind.value,
                            asset.checksum,
                            asset.size_bytes,
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

            connection.commit()

        return {"run": run_row, "asset_count": len(manifest.assets), "job_count": len(planned_run.jobs)}

    def list_assets(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.raw_assets WHERE run_id = %s ORDER BY asset_key ASC",
                    (run_id,),
                )
                return cursor.fetchall()

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.raw_assets WHERE id = %s", (asset_id,))
                return cursor.fetchone()

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
                        (run_id, asset_id, frame_index, captured_at, width, height, source_ref, frame_hash, frame_png, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        RETURNING *;
                        """,
                        (
                            run_id,
                            frame.asset_id,
                            frame.frame_index,
                            frame.captured_at,
                            frame.width,
                            frame.height,
                            frame.source_ref,
                            frame.frame_hash,
                            frame.frame_png,
                            json.dumps(json_ready(frame.metadata)),
                        ),
                    )
                    inserted.append(cursor.fetchone())
            connection.commit()
        return inserted

    def list_frames(self, asset_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.frames WHERE asset_id = %s ORDER BY frame_index ASC",
                    (asset_id,),
                )
                return cursor.fetchall()

    def replace_detections(self, run_id: str, asset_id: str, detections: Sequence[DetectionRecord]) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.detections
                    WHERE frame_id IN (SELECT id FROM {self.schema}.frames WHERE asset_id = %s)
                    """,
                    (asset_id,),
                )
                inserted: list[dict[str, Any]] = []
                for detection in detections:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.detections
                        (run_id, frame_id, roi_index, bbox_x, bbox_y, bbox_w, bbox_h, area, perimeter,
                         major_axis_length, minor_axis_length, min_gray_value, mean_gray_value, crop_png, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
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
                            detection.area,
                            detection.perimeter,
                            detection.major_axis_length,
                            detection.minor_axis_length,
                            detection.min_gray_value,
                            detection.mean_gray_value,
                            detection.crop_png,
                            json.dumps(json_ready(detection.metadata)),
                        ),
                    )
                    inserted.append(cursor.fetchone())
            connection.commit()
        return inserted

    def list_detections(self, asset_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT detections.*
                    FROM {self.schema}.detections detections
                    JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                    WHERE frames.asset_id = %s
                    ORDER BY frames.frame_index ASC, detections.roi_index ASC
                    """,
                    (asset_id,),
                )
                return cursor.fetchall()

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

    def list_models(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.models ORDER BY created_at DESC")
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
            connection.commit()
        return row

    def append_job_event(self, job_id: str | None, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.job_events (job_id, event_type, payload)
                    VALUES (%s, %s, %s::jsonb)
                    RETURNING *;
                    """,
                    (job_id, event_type, json.dumps(json_ready(payload or {}))),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def list_job_events(
        self,
        *,
        after_id: int | None = None,
        run_id: str | None = None,
        job_id: str | None = None,
        limit: int = 100,
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
        params.append(limit)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT events.*
                    FROM {self.schema}.job_events events
                    {joins}
                    {where}
                    ORDER BY events.id ASC
                    LIMIT %s
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
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.worker_sessions
                    (worker_id, status, leased_job_id, capabilities, metadata, last_heartbeat)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, NOW())
                    ON CONFLICT (worker_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        leased_job_id = EXCLUDED.leased_job_id,
                        capabilities = EXCLUDED.capabilities,
                        metadata = EXCLUDED.metadata,
                        last_heartbeat = NOW()
                    RETURNING *;
                    """,
                    (
                        worker_id,
                        status,
                        leased_job_id,
                        json.dumps(list(capabilities or [])),
                        json.dumps(json_ready(metadata or {})),
                    ),
                )
                row = cursor.fetchone()
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
                    RETURNING jobs.status;
                    """
                )
                rows = cursor.fetchall()
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
                    """,
                    (run_id,),
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
