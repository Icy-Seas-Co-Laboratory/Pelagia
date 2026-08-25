from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from urllib.parse import urlparse
from typing import Any, Iterable, Mapping, Sequence

from ..config import CoreConfig
from ..domain import AssetKind, ClassificationResultRecord, DetectionRecord, FrameRecord, JobStatus, ModelRecord, PipelineStage, PlannedRun, normalize_collections
from ..utils.serialization import json_ready
from ..utils.validation import validate_schema_name

try:
    import psycopg
    from psycopg import conninfo, sql
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - exercised only when postgres extras are absent
    psycopg = None
    conninfo = None
    sql = None
    dict_row = None
    ConnectionPool = None


REQUIRED_SCHEMA_TABLES = (
    "schema_migrations",
    "users",
    "projects",
    "project_memberships",
    "user_sessions",
    "runs",
    "raw_assets",
    "frames",
    "telemetry_sources",
    "telemetry_sensors",
    "telemetry_parameters",
    "telemetry_streams",
    "telemetry_observations",
    "timeline_event_types",
    "timeline_events",
    "detection_candidate",
    "detections_refined",
    "models",
    "classification_results",
    "classification_labels",
    "classification_inference_runs",
    "classification_evidence",
    "clustering_evidence",
    "roi_label_annotations",
    "roi_annotation_reviews",
    "registry_workspaces",
    "registry_items",
    "processing_jobs",
    "processing_job_dependencies",
    "processing_series",
    "processing_series_steps",
    "processing_work_units",
    "project_processing_status_snapshots",
    "frame_processing_status",
    "worker_sessions",
    "job_events",
    "logs",
)

DEFAULT_PROJECT_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_PROJECT_KEY = "default"
PROJECT_ROLES = {"viewer", "editor", "manager", "admin"}
FRAME_PROCESSING_STATUS_VALUES = (
    "unknown",
    "queued",
    "leased",
    "working",
    "succeeded",
    "failed",
    "cancelled",
    "dead_lettered",
)
FRAME_PROCESSING_STATUSES = set(FRAME_PROCESSING_STATUS_VALUES)
PASSWORD_HASH_ITERATIONS = 260_000
DEFAULT_SESSION_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class ClassificationEvidenceContext:
    """Run-scoped database identifiers required to persist classification evidence.

    A model's artifact and class mapping are invariant for a classification run.
    Resolving them once avoids repeating catalog writes for every ROI.
    """

    project_id: str
    inference_run_id: str
    model_artifact_id: str
    class_label_ids: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class ClusteringEvidenceContext:
    """Run-scoped identifiers required to persist cluster evidence."""

    project_id: str
    inference_run_id: str
    model_artifact_id: str


def _initial_job_progress(
    stage: str,
    status: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Seed queued jobs with known work-unit totals for aggregate progress."""
    total = 0
    unit = "units"
    if stage in {
        PipelineStage.PREPROCESS_FRAMES.value,
        PipelineStage.SEGMENT.value,
        PipelineStage.BACKGROUND_FRAMES.value,
    }:
        frame_ids = [value for value in payload.get("frame_ids") or [] if value]
        if payload.get("frame_id"):
            frame_ids.append(payload["frame_id"])
        total = len(dict.fromkeys(str(frame_id) for frame_id in frame_ids))
        unit = "frames"
    elif stage == PipelineStage.ROI_REFINEMENT.value:
        total = len(dict.fromkeys(str(value) for value in payload.get("detection_ids") or [] if value))
        unit = "rois"
    elif stage == PipelineStage.CLASSIFY.value:
        total = len(dict.fromkeys(str(value) for value in payload.get("roi_ids") or [] if value))
        unit = "rois"

    if total <= 0:
        return {}
    return {
        "schema_version": 1,
        "stage": stage,
        "unit": unit,
        "total": total,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "percent": 0.0,
        "current": {},
        "secondary": {},
        "rates": {"units_per_second": None},
        "message": f"{status} for {total} {unit}",
    }


def render_schema(schema: str = "seasight") -> str:
    schema = validate_schema_name(schema)
    template = files(__package__).joinpath("sql", "schema.sql").read_text(encoding="utf-8")
    return template.replace("{schema}", schema).strip()


def available_migrations() -> list[dict[str, str]]:
    migrations_dir = files(__package__).joinpath("sql", "migrations")
    if not migrations_dir.is_dir():
        return []
    migrations = []
    for item in sorted(migrations_dir.iterdir(), key=lambda path: path.name):
        if not item.name.endswith(".sql"):
            continue
        template = item.read_text(encoding="utf-8")
        migrations.append(
            {
                "migration_id": item.name.removesuffix(".sql"),
                "filename": item.name,
                "description": template.splitlines()[0].removeprefix("--").strip() if template.splitlines() else "",
                "checksum": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "template": template,
            }
        )
    return migrations


def render_migration(migration: dict[str, str], schema: str) -> str:
    return migration["template"].replace("{schema}", validate_schema_name(schema)).strip()


def _require_psycopg() -> None:
    if psycopg is None or ConnectionPool is None:
        raise RuntimeError("psycopg and psycopg-pool are required for PostgreSQL operations. Install Pelagia[postgres].")


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


def hash_password(password: str, *, iterations: int = PASSWORD_HASH_ITERATIONS) -> str:
    """Hash a password using only stdlib primitives for the initial auth foundation."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        int(iterations),
    )
    return f"pbkdf2_sha256${int(iterations)}${salt}${digest.hex()}"


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        algorithm, iterations, salt, expected = password_hash.split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        iteration_count = int(iterations)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        iteration_count,
    ).hex()
    return hmac.compare_digest(digest, expected)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PostgresRepository:
    def __init__(self, config: CoreConfig):
        _require_psycopg()
        self.config = config
        self.schema = validate_schema_name(config.database.schema_name)
        if config.database.pool_min_size < 0:
            raise ValueError("database.pool_min_size must be non-negative.")
        if config.database.pool_max_size < 1:
            raise ValueError("database.pool_max_size must be at least 1.")
        if config.database.pool_min_size > config.database.pool_max_size:
            raise ValueError("database.pool_min_size cannot exceed database.pool_max_size.")
        self._pool = ConnectionPool(
            conninfo=config.database.dsn,
            min_size=config.database.pool_min_size,
            max_size=config.database.pool_max_size,
            timeout=config.database.pool_timeout_s,
            kwargs={
                "connect_timeout": config.database.connect_timeout_s,
                "row_factory": dict_row,
                "autocommit": False,
            },
            open=True,
        )
        # Scoped views are the preferred application-facing dependencies.  Keep
        # this facade intact while the underlying SQL moves out incrementally.
        from .scoped import CatalogRepository, FrameRepository, IdentityRepository, JobRepository, TelemetryRepository

        self.identity = IdentityRepository(self)
        self.catalog = CatalogRepository(self)
        self.frames = FrameRepository(self)
        self.jobs = JobRepository(self)
        self.telemetry = TelemetryRepository(self)

    def connect(self):
        """Borrow a bounded, reusable connection for one repository operation."""

        return self._pool.connection()

    def close(self) -> None:
        """Close this process's connection pool."""

        self._pool.close()

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

    def initialize_schema(self, *, statement_timeout_ms: int | None = None) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                timeout_ms = (
                    self.config.database.statement_timeout_ms
                    if statement_timeout_ms is None
                    else max(0, int(statement_timeout_ms))
                )
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(timeout_ms),),
                )
                cursor.execute(render_schema(self.schema))
                self._apply_migrations(cursor)
            connection.commit()

    def _apply_migrations(self, cursor) -> list[dict[str, Any]]:
        applied_now = []
        for migration in available_migrations():
            cursor.execute(
                f"""
                SELECT migration_id, checksum
                FROM {self.schema}.schema_migrations
                WHERE migration_id = %s
                """,
                (migration["migration_id"],),
            )
            existing = cursor.fetchone()
            if existing is not None:
                if existing["checksum"] != migration["checksum"]:
                    raise RuntimeError(
                        f"Migration {migration['migration_id']} checksum mismatch. "
                        "The database has a different migration body recorded."
                    )
                continue
            cursor.execute(render_migration(migration, self.schema))
            cursor.execute(
                f"""
                INSERT INTO {self.schema}.schema_migrations
                    (migration_id, checksum, description, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING migration_id, checksum, description, applied_at
                """,
                (
                    migration["migration_id"],
                    migration["checksum"],
                    migration["description"],
                    json.dumps({"filename": migration["filename"]}),
                ),
            )
            applied_now.append(cursor.fetchone())
        return applied_now

    def list_schema_migrations(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s
                      AND table_name = 'schema_migrations'
                    """,
                    (self.schema,),
                )
                if cursor.fetchone() is None:
                    return []
                cursor.execute(
                    f"""
                    SELECT migration_id, checksum, description, metadata, applied_at
                    FROM {self.schema}.schema_migrations
                    ORDER BY migration_id
                    """
                )
                return cursor.fetchall()

    def migration_status(self) -> dict[str, Any]:
        available = available_migrations()
        applied = self.list_schema_migrations()
        applied_by_id = {row["migration_id"]: row for row in applied}
        pending = [
            {
                "migration_id": migration["migration_id"],
                "checksum": migration["checksum"],
                "description": migration["description"],
            }
            for migration in available
            if migration["migration_id"] not in applied_by_id
        ]
        checksum_mismatches = [
            {
                "migration_id": migration["migration_id"],
                "expected_checksum": migration["checksum"],
                "applied_checksum": applied_by_id[migration["migration_id"]]["checksum"],
            }
            for migration in available
            if migration["migration_id"] in applied_by_id
            and applied_by_id[migration["migration_id"]]["checksum"] != migration["checksum"]
        ]
        return {
            "available_count": len(available),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied": applied,
            "pending": pending,
            "checksum_mismatches": checksum_mismatches,
            "ready": not pending and not checksum_mismatches,
        }

    def ensure_default_project(self) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                row = self._ensure_default_project(cursor)
            connection.commit()
        return row

    def _ensure_default_project(self, cursor) -> dict[str, Any]:
        cursor.execute(
            f"""
            INSERT INTO {self.schema}.projects (id, project_key, project_name, description, metadata)
            VALUES (
                %s,
                %s,
                'Default',
                'Default project for existing Pelagia data.',
                %s::jsonb
            )
            ON CONFLICT (project_key) DO UPDATE SET
                project_name = COALESCE({self.schema}.projects.project_name, EXCLUDED.project_name),
                description = COALESCE({self.schema}.projects.description, EXCLUDED.description),
                metadata = {self.schema}.projects.metadata || EXCLUDED.metadata
            RETURNING *;
            """,
            (
                DEFAULT_PROJECT_ID,
                DEFAULT_PROJECT_KEY,
                json.dumps({"system_default": True}),
            ),
        )
        return cursor.fetchone()

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
        migrations = self.migration_status() if "schema_migrations" in existing else {
            "available_count": len(available_migrations()),
            "applied_count": 0,
            "pending_count": len(available_migrations()),
            "applied": [],
            "pending": [
                {
                    "migration_id": migration["migration_id"],
                    "checksum": migration["checksum"],
                    "description": migration["description"],
                }
                for migration in available_migrations()
            ],
            "checksum_mismatches": [],
            "ready": False,
        }
        return {
            "schema": self.schema,
            "ready": not missing and bool(migrations.get("ready")),
            "required_tables": required,
            "existing_tables": existing,
            "missing_tables": missing,
            "migrations": migrations,
        }

    def purge_all(
        self,
        *,
        exact_counts: bool = True,
        preserve_migrations: bool = True,
    ) -> dict[str, Any]:
        """Delete all Pelagia rows while preserving the schema, indexes, and functions.

        Destructive maintenance must not inherit the statement timeout used to
        protect interactive queries. Callers resetting a large installation can
        also omit the pre-reset ``COUNT(*)`` scans, which are informational and
        can be substantially slower than the ``TRUNCATE`` itself.
        """
        requested_tables = [
            table
            for table in REQUIRED_SCHEMA_TABLES
            if not preserve_migrations or table != "schema_migrations"
        ]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = 0")
                cursor.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = ANY(%s)
                    """,
                    (self.schema, requested_tables),
                )
                existing = {row["table_name"] for row in cursor.fetchall()}
                tables = [table for table in requested_tables if table in existing]
                before: dict[str, int] | None = None
                if exact_counts:
                    before = {}
                    for table in tables:
                        cursor.execute(f"SELECT COUNT(*) AS count FROM {self.schema}.{table}")
                        before[table] = cursor.fetchone()["count"]
                if tables:
                    table_list = ", ".join(f"{self.schema}.{table}" for table in tables)
                    cursor.execute(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE")
            connection.commit()
        return {
            "schema": self.schema,
            "tables": before,
            "total_rows_deleted": None if before is None else sum(before.values()),
            "exact_counts_collected": exact_counts,
            "purged_tables": tables,
            "missing_tables": [table for table in requested_tables if table not in existing],
            "preserved_tables": ["schema_migrations"] if preserve_migrations else [],
        }

    def create_user(
        self,
        username: str,
        *,
        password: str | None = None,
        display_name: str | None = None,
        is_admin: bool = False,
        is_active: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_username = self._normalize_username(username)
        password_hash = hash_password(password) if password is not None else None
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.users
                    (username, display_name, password_hash, is_active, is_admin, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING *;
                    """,
                    (
                        normalized_username,
                        display_name,
                        password_hash,
                        is_active,
                        is_admin,
                        json.dumps(json_ready(metadata or {})),
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.users WHERE id = %s", (user_id,))
                return cursor.fetchone()

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.users WHERE username = %s",
                    (self._normalize_username(username),),
                )
                return cursor.fetchone()

    def list_users(
        self,
        *,
        project_id: str | None = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        join = ""
        select_membership = ""
        if project_id:
            join = f"JOIN {self.schema}.project_memberships memberships ON memberships.user_id = users.id"
            clauses.append("memberships.project_id = %s")
            params.append(project_id)
            select_membership = ", memberships.project_id, memberships.role AS project_role"
        if active_only:
            clauses.append("users.is_active")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        users.id,
                        users.username,
                        users.display_name,
                        users.is_active,
                        users.is_admin,
                        users.metadata,
                        users.created_at,
                        users.updated_at
                        {select_membership}
                    FROM {self.schema}.users users
                    {join}
                    {where}
                    ORDER BY users.username ASC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def deactivate_user(
        self,
        user_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.users
                    SET
                        is_active = false,
                        metadata = metadata || %s::jsonb
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (json.dumps(json_ready(metadata or {})), user_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.user_sessions
                        SET revoked_at = COALESCE(revoked_at, NOW())
                        WHERE user_id = %s;
                        """,
                        (user_id,),
                    )
            connection.commit()
        return row

    def reset_user_password(
        self,
        user_id: str,
        password: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        password_hash = hash_password(password)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.users
                    SET
                        password_hash = %s,
                        metadata = metadata || %s::jsonb
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (password_hash, json.dumps(json_ready(metadata or {})), user_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.user_sessions
                        SET revoked_at = COALESCE(revoked_at, NOW())
                        WHERE user_id = %s;
                        """,
                        (user_id,),
                    )
            connection.commit()
        return row

    def delete_user(self, user_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.schema}.users WHERE id = %s RETURNING *;",
                    (user_id,),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def verify_user_password(self, username: str, password: str) -> dict[str, Any] | None:
        user = self.get_user_by_username(username)
        if user is None or not user.get("is_active"):
            return None
        return user if verify_password(password, user.get("password_hash")) else None

    def create_project(
        self,
        project_key: str,
        *,
        project_name: str | None = None,
        description: str | None = None,
        kvstore_root_path: str | None = None,
        is_active: bool = True,
        settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_key = self._normalize_project_key(project_key)
        resolved_name = project_name or normalized_key
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.projects
                    (project_key, project_name, description, kvstore_root_path, is_active, settings, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    RETURNING *;
                    """,
                    (
                        normalized_key,
                        resolved_name,
                        description,
                        kvstore_root_path,
                        is_active,
                        json.dumps(json_ready(settings or {})),
                        json.dumps(json_ready(metadata or {})),
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def list_projects(
        self,
        *,
        user_id: str | None = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        join = ""
        if user_id:
            join = f"JOIN {self.schema}.project_memberships memberships ON memberships.project_id = projects.id"
            clauses.append("memberships.user_id = %s")
            params.append(user_id)
        if active_only:
            clauses.append("projects.is_active")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT projects.*
                           {', memberships.role AS membership_role' if user_id else ''}
                    FROM {self.schema}.projects projects
                    {join}
                    {where}
                    ORDER BY projects.project_key ASC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def list_user_projects(self, user_id: str, *, active_only: bool = True) -> list[dict[str, Any]]:
        return self.list_projects(user_id=user_id, active_only=active_only)

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.projects WHERE id = %s", (project_id,))
                return cursor.fetchone()

    def get_project_by_key(self, project_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.projects WHERE project_key = %s",
                    (self._normalize_project_key(project_key),),
                )
                return cursor.fetchone()

    def update_project(
        self,
        project_id: str,
        *,
        project_name: str | None = None,
        description: str | None = None,
        kvstore_root_path: str | None = None,
        is_active: bool | None = None,
        settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        updates: list[str] = []
        params: list[Any] = []
        if project_name is not None:
            updates.append("project_name = %s")
            params.append(project_name)
        if description is not None:
            updates.append("description = %s")
            params.append(description)
        if kvstore_root_path is not None:
            updates.append("kvstore_root_path = %s")
            params.append(kvstore_root_path)
        if is_active is not None:
            updates.append("is_active = %s")
            params.append(is_active)
        if settings is not None:
            updates.append("settings = %s::jsonb")
            params.append(json.dumps(json_ready(settings)))
        if metadata is not None:
            updates.append("metadata = metadata || %s::jsonb")
            params.append(json.dumps(json_ready(metadata)))
        if not updates:
            return self.get_project(project_id)
        params.append(project_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.projects
                    SET {', '.join(updates)}
                    WHERE id = %s
                    RETURNING *;
                    """,
                    tuple(params),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def deactivate_project(
        self,
        project_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.projects
                    SET
                        is_active = false,
                        metadata = metadata || %s::jsonb
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (json.dumps(json_ready(metadata or {})), project_id),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def add_project_member(
        self,
        user_id: str,
        project_id: str,
        *,
        role: str = "viewer",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_role = self._normalize_project_role(role)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.project_memberships
                    (user_id, project_id, role, metadata)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (user_id, project_id) DO UPDATE SET
                        role = EXCLUDED.role,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    RETURNING *;
                    """,
                    (
                        user_id,
                        project_id,
                        resolved_role,
                        json.dumps(json_ready(metadata or {})),
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def get_project_membership(self, user_id: str, project_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT memberships.*, users.username, projects.project_key, projects.project_name
                    FROM {self.schema}.project_memberships memberships
                    JOIN {self.schema}.users users ON users.id = memberships.user_id
                    JOIN {self.schema}.projects projects ON projects.id = memberships.project_id
                    WHERE memberships.user_id = %s AND memberships.project_id = %s
                    """,
                    (user_id, project_id),
                )
                return cursor.fetchone()

    def create_session(
        self,
        user_id: str,
        project_id: str | None,
        *,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        user_agent: str | None = None,
        remote_addr: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user = self.get_user(user_id)
        if user is None or not user.get("is_active"):
            raise ValueError("Cannot create a session for an inactive or missing user.")
        if project_id is None:
            if not user.get("is_admin"):
                raise PermissionError("Only user admins may create a session without a project.")
        else:
            project = self.get_project(project_id)
            if project is None or not project.get("is_active"):
                raise ValueError("Cannot create a session for an inactive or missing project.")
            if not user.get("is_admin") and self.get_project_membership(user_id, project_id) is None:
                raise PermissionError("User is not a member of the requested project.")

        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, int(ttl_seconds)))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.user_sessions
                    (user_id, project_id, token_hash, user_agent, remote_addr, metadata, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING *;
                    """,
                    (
                        user_id,
                        project_id,
                        hash_session_token(token),
                        user_agent,
                        remote_addr,
                        json.dumps(json_ready(metadata or {})),
                        expires_at,
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        return {"token": token, "session": row}

    def get_session(self, session_token: str, *, touch: bool = True) -> dict[str, Any] | None:
        token_hash = hash_session_token(session_token)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if touch:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.user_sessions
                        SET last_seen_at = NOW()
                        WHERE token_hash = %s
                          AND revoked_at IS NULL
                          AND expires_at > NOW()
                        RETURNING *;
                        """,
                        (token_hash,),
                    )
                    session = cursor.fetchone()
                else:
                    cursor.execute(
                        f"""
                        SELECT *
                        FROM {self.schema}.user_sessions
                        WHERE token_hash = %s
                          AND revoked_at IS NULL
                          AND expires_at > NOW()
                        """,
                        (token_hash,),
                    )
                    session = cursor.fetchone()
                if session is None:
                    connection.commit()
                    return None
                cursor.execute(
                    f"""
                    SELECT
                        sessions.*,
                        users.username,
                        users.display_name,
                        users.is_admin,
                        projects.project_key,
                        projects.project_name,
                        COALESCE(
                            memberships.role,
                            CASE WHEN sessions.project_id IS NOT NULL AND users.is_admin THEN 'admin' END
                        ) AS project_role
                    FROM {self.schema}.user_sessions sessions
                    JOIN {self.schema}.users users ON users.id = sessions.user_id
                    LEFT JOIN {self.schema}.projects projects ON projects.id = sessions.project_id
                    LEFT JOIN {self.schema}.project_memberships memberships
                      ON memberships.user_id = sessions.user_id
                     AND memberships.project_id = sessions.project_id
                    WHERE sessions.id = %s
                      AND users.is_active
                      AND (sessions.project_id IS NULL OR projects.is_active)
                    """,
                    (session["id"],),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def revoke_session(self, session_token: str) -> dict[str, Any] | None:
        token_hash = hash_session_token(session_token)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.user_sessions
                    SET revoked_at = COALESCE(revoked_at, NOW())
                    WHERE token_hash = %s
                    RETURNING *;
                    """,
                    (token_hash,),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def revoke_user_sessions(self, user_id: str) -> int:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.user_sessions
                    SET revoked_at = COALESCE(revoked_at, NOW())
                    WHERE user_id = %s AND revoked_at IS NULL
                    """,
                    (user_id,),
                )
                count = cursor.rowcount
            connection.commit()
        return int(count)

    @staticmethod
    def _normalize_username(username: str) -> str:
        normalized = str(username).strip().lower()
        if not normalized:
            raise ValueError("username must be non-empty.")
        return normalized

    @staticmethod
    def _normalize_project_key(project_key: str) -> str:
        normalized = str(project_key).strip().lower()
        if not normalized:
            raise ValueError("project_key must be non-empty.")
        return normalized

    @staticmethod
    def _normalize_project_role(role: str) -> str:
        normalized = str(role).strip().lower()
        if normalized not in PROJECT_ROLES:
            raise ValueError(
                f"project role must be one of: {', '.join(sorted(PROJECT_ROLES))}."
            )
        return normalized

    @staticmethod
    def _required_project_id(project_id: str | None, context: str) -> str:
        if project_id:
            return str(project_id)
        raise ValueError(f"project_id is required for {context}.")

    def _resolve_project_id(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        if project_id:
            return str(project_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if run_id:
                    cursor.execute(
                        f"SELECT project_id FROM {self.schema}.runs WHERE id = %s",
                        (run_id,),
                    )
                    row = cursor.fetchone()
                    if row is not None and row.get("project_id"):
                        return str(row["project_id"])
                if asset_id:
                    cursor.execute(
                        f"SELECT project_id FROM {self.schema}.raw_assets WHERE id = %s",
                        (asset_id,),
                    )
                    row = cursor.fetchone()
                    if row is not None and row.get("project_id"):
                        return str(row["project_id"])
                if job_id:
                    cursor.execute(
                        f"SELECT project_id FROM {self.schema}.processing_jobs WHERE id = %s",
                        (job_id,),
                    )
                    row = cursor.fetchone()
                    if row is not None and row.get("project_id"):
                        return str(row["project_id"])
        raise ValueError("project_id is required when it cannot be derived from an existing resource.")

    def _ensure_project_scope(
        self,
        cursor,
        project_id: str | None,
        *,
        run_id: str | None = None,
        asset_id: str | None = None,
        job_ids: Sequence[str] | None = None,
        frame_ids: Sequence[str] | None = None,
        detection_ids: Sequence[str] | None = None,
    ) -> None:
        if not project_id:
            return
        resolved_project_id = str(project_id)
        if run_id:
            cursor.execute(
                f"SELECT 1 FROM {self.schema}.runs WHERE id = %s AND project_id = %s",
                (run_id, resolved_project_id),
            )
            if cursor.fetchone() is None:
                raise KeyError(f"Run {run_id!r} was not found in project {resolved_project_id!r}.")
        if asset_id:
            cursor.execute(
                f"SELECT 1 FROM {self.schema}.raw_assets WHERE id = %s AND project_id = %s",
                (asset_id, resolved_project_id),
            )
            if cursor.fetchone() is None:
                raise KeyError(f"Asset {asset_id!r} was not found in project {resolved_project_id!r}.")

        def _missing_ids(values: Sequence[str] | None, query: str) -> list[str]:
            resolved = [str(value) for value in values or [] if value]
            if not resolved:
                return []
            cursor.execute(query, (resolved, resolved_project_id))
            found = {str(row["id"]) for row in cursor.fetchall()}
            return [value for value in resolved if value not in found]

        missing_jobs = _missing_ids(
            job_ids,
            f"""
            SELECT id
            FROM {self.schema}.processing_jobs
            WHERE id = ANY(%s::uuid[]) AND project_id = %s
            """,
        )
        if missing_jobs:
            raise KeyError(f"Job(s) not found in project {resolved_project_id!r}: {', '.join(missing_jobs)}")

        missing_frames = _missing_ids(
            frame_ids,
            f"""
            SELECT frames.id
            FROM {self.schema}.frames frames
            JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
            WHERE frames.id = ANY(%s::uuid[]) AND assets.project_id = %s
            """,
        )
        if missing_frames:
            raise KeyError(f"Frame(s) not found in project {resolved_project_id!r}: {', '.join(missing_frames)}")

        missing_detections = _missing_ids(
            detection_ids,
            f"""
            SELECT detections.id
            FROM {self.schema}.detection_candidate detections
            JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
            JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
            WHERE detections.id = ANY(%s::uuid[]) AND assets.project_id = %s
            """,
        )
        if missing_detections:
            raise KeyError(
                f"Detection(s) not found in project {resolved_project_id!r}: {', '.join(missing_detections)}"
            )

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
        project_id: str | None = None,
        collection: str | None = None,
        run_key: str | None = None,
        instrument: str | None = None,
        source_type: str | None = None,
        status: str | None = None,
        source_path: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if project_id:
            clauses.append("runs.project_id = %s")
            params.append(project_id)
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

    def get_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["id = %s"]
                params: list[Any] = [run_id]
                if project_id:
                    clauses.append("project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"SELECT * FROM {self.schema}.runs WHERE {' AND '.join(clauses)}",
                    tuple(params),
                )
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
        project_id: str | None = None,
        status: str | None = None,
        stage: str | None = None,
        statuses: Sequence[str] | None = None,
        stages: Sequence[str] | None = None,
        job_ids: Sequence[str] | None = None,
        worker_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        include_details: bool = True,
        include_progress: bool = True,
        include_payload: bool = False,
        include_result: bool = False,
        sort: str = "created_at",
        direction: str = "desc",
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        resolved_statuses = list(statuses or ([] if status is None else [status]))
        resolved_stages = list(stages or ([] if stage is None else [stage]))
        clauses, params = self._job_filter_clauses(
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            statuses=resolved_statuses,
            stages=resolved_stages,
            job_ids=job_ids,
            worker_id=worker_id,
        )
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
            selected_columns = [
                "id",
                "run_id",
                "asset_id",
                "stage",
                "status",
                "priority",
                "attempt_count",
                "max_attempts",
                "lease_expires_at",
                "worker_id",
                "summary",
                "control_reason",
                "error_message",
                "created_at",
                "updated_at",
                "started_at",
                "finished_at",
            ]
            if include_progress:
                selected_columns.append("progress")
            if include_payload:
                selected_columns.append("payload")
            if include_result:
                selected_columns.append("result")
            selected_columns.extend(
                [
                    "jsonb_typeof(payload) AS payload_type",
                    "pg_column_size(payload) AS payload_bytes",
                    "jsonb_typeof(result) AS result_type",
                    "pg_column_size(result) AS result_bytes",
                    "jsonb_typeof(progress) AS progress_type",
                    "pg_column_size(progress) AS progress_bytes",
                    "jsonb_array_length(logs_tail) AS logs_tail_count",
                ]
            )
            select_sql = ",\n                ".join(selected_columns)
        order_column = self._job_sort_column(sort)
        order_direction = "ASC" if str(direction).lower() == "asc" else "DESC"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {select_sql} FROM {self.schema}.processing_jobs {where} ORDER BY {order_column} {order_direction}, id {order_direction} {limit_sql} {offset_sql}",
                    tuple(params),
                )
                return cursor.fetchall()

    def _job_filter_clauses(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        statuses: Sequence[str] | None = None,
        stages: Sequence[str] | None = None,
        job_ids: Sequence[str] | None = None,
        worker_id: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def enum_in_clause(column: str, values: Sequence[str], enum_name: str) -> str:
            placeholders = ", ".join([f"%s::{self.schema}.{enum_name}" for _ in values])
            return f"{column} IN ({placeholders})"

        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        if asset_id:
            clauses.append("asset_id = %s")
            params.append(asset_id)
        resolved_statuses = [str(value) for value in statuses or [] if value]
        if resolved_statuses:
            clauses.append(enum_in_clause("status", resolved_statuses, "job_status"))
            params.extend(resolved_statuses)
        resolved_stages = [str(value) for value in stages or [] if value]
        if resolved_stages:
            clauses.append(enum_in_clause("stage", resolved_stages, "stage_name"))
            params.extend(resolved_stages)
        resolved_job_ids = [str(value) for value in job_ids or [] if value]
        if resolved_job_ids:
            placeholders = ", ".join(["%s::uuid" for _ in resolved_job_ids])
            clauses.append(f"id IN ({placeholders})")
            params.extend(resolved_job_ids)
        if worker_id:
            clauses.append("worker_id = %s")
            params.append(worker_id)
        return clauses, params

    def _job_sort_column(self, sort: str | None) -> str:
        allowed = {
            "created_at": "created_at",
            "updated_at": "updated_at",
            "priority": "priority",
            "stage": "stage",
            "status": "status",
        }
        return allowed.get(str(sort or "").lower(), "created_at")

    def summarize_jobs(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        statuses: Sequence[str] | None = None,
        stages: Sequence[str] | None = None,
        job_ids: Sequence[str] | None = None,
        worker_id: str | None = None,
        include_recent: bool = False,
        recent_limit: int = 5,
    ) -> dict[str, Any]:
        clauses, params = self._job_filter_clauses(
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            statuses=statuses,
            stages=stages,
            job_ids=job_ids,
            worker_id=worker_id,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        progress_select = self._progress_aggregate_sql()
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        COUNT(*)::bigint AS job_count,
                        COUNT(*) FILTER (WHERE status = 'queued')::bigint AS queued,
                        COUNT(*) FILTER (WHERE status = 'leased')::bigint AS leased,
                        COUNT(*) FILTER (WHERE status = 'working')::bigint AS working,
                        COUNT(*) FILTER (WHERE status = 'paused')::bigint AS paused,
                        COUNT(*) FILTER (WHERE status = 'succeeded')::bigint AS succeeded,
                        COUNT(*) FILTER (WHERE status = 'failed')::bigint AS failed,
                        COUNT(*) FILTER (WHERE status = 'cancelled')::bigint AS cancelled,
                        COUNT(*) FILTER (WHERE status = 'dead_lettered')::bigint AS dead_lettered,
                        {progress_select}
                    FROM {self.schema}.processing_jobs
                    {where}
                    """,
                    tuple(params),
                )
                total = cursor.fetchone() or {}
                cursor.execute(
                    f"""
                    SELECT
                        stage,
                        COUNT(*)::bigint AS job_count,
                        COUNT(*) FILTER (WHERE status = 'queued')::bigint AS queued,
                        COUNT(*) FILTER (WHERE status = 'leased')::bigint AS leased,
                        COUNT(*) FILTER (WHERE status = 'working')::bigint AS working,
                        COUNT(*) FILTER (WHERE status = 'paused')::bigint AS paused,
                        COUNT(*) FILTER (WHERE status = 'succeeded')::bigint AS succeeded,
                        COUNT(*) FILTER (WHERE status = 'failed')::bigint AS failed,
                        COUNT(*) FILTER (WHERE status = 'cancelled')::bigint AS cancelled,
                        COUNT(*) FILTER (WHERE status = 'dead_lettered')::bigint AS dead_lettered,
                        {progress_select}
                    FROM {self.schema}.processing_jobs
                    {where}
                    GROUP BY stage
                    ORDER BY stage
                    """,
                    tuple(params),
                )
                by_stage = cursor.fetchall()
                cursor.execute(
                    f"""
                    SELECT status, COUNT(*)::bigint AS job_count
                    FROM {self.schema}.processing_jobs
                    {where}
                    GROUP BY status
                    ORDER BY status
                    """,
                    tuple(params),
                )
                by_status = cursor.fetchall()
                recent_jobs: list[dict[str, Any]] = []
                if include_recent:
                    recent_jobs = self.list_jobs(
                        project_id=project_id,
                        run_id=run_id,
                        asset_id=asset_id,
                        statuses=statuses,
                        stages=stages,
                        job_ids=job_ids,
                        worker_id=worker_id,
                        limit=recent_limit,
                        include_details=False,
                        include_progress=True,
                        sort="updated_at",
                        direction="desc",
                    )
        return {
            "filters": {
                "run_id": run_id,
                "project_id": project_id,
                "asset_id": asset_id,
                "status": list(statuses or []),
                "stage": list(stages or []),
                "ids": list(job_ids or []),
                "worker_id": worker_id,
            },
            "total": self._job_summary_row(total),
            "by_stage": [self._job_summary_row(row) for row in by_stage],
            "by_status": by_status,
            "recent_jobs": recent_jobs,
        }

    def _progress_aggregate_sql(self) -> str:
        def numeric_jsonb(key: str) -> str:
            return f"""
                CASE
                    WHEN progress ? '{key}' AND progress->>'{key}' ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    THEN (progress->>'{key}')::numeric
                    ELSE 0
                END
            """

        return f"""
            SUM({numeric_jsonb("total")}) AS known_total_units,
            SUM({numeric_jsonb("completed")}) AS completed_units,
            SUM({numeric_jsonb("failed")}) AS failed_units,
            SUM({numeric_jsonb("skipped")}) AS skipped_units
        """

    def _job_summary_row(self, row: dict[str, Any]) -> dict[str, Any]:
        known_total = float(row.get("known_total_units") or 0)
        completed = float(row.get("completed_units") or 0)
        progress = {
            "known_total_units": known_total,
            "completed_units": completed,
            "failed_units": float(row.get("failed_units") or 0),
            "skipped_units": float(row.get("skipped_units") or 0),
            "percent": (completed / known_total * 100.0) if known_total > 0 else None,
        }
        return {
            **row,
            "progress": progress,
        }

    def get_job(self, job_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["id = %s"]
                params: list[Any] = [job_id]
                if project_id:
                    clauses.append("project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"SELECT * FROM {self.schema}.processing_jobs WHERE {' AND '.join(clauses)}",
                    tuple(params),
                )
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

    def register_planned_run(
        self,
        planned_run: PlannedRun,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        manifest = planned_run.manifest
        schema = self.schema
        resolved_project_id = self._required_project_id(
            project_id or manifest.metadata.get("project_id"),
            "register_planned_run",
        )

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.runs (id, project_id, run_key, instrument, source_path, source_type, metadata, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, 'registered')
                    RETURNING id, project_id, run_key, source_path, source_type, status, created_at
                    """,
                    (
                        manifest.run_id,
                        resolved_project_id,
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
                    (id, project_id, run_id, filename, path, kind, checksum, size_bytes, collections, media_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::{schema}.asset_kind, %s, %s, %s, %s, %s::jsonb)
                    """,
                    [
                        (
                            asset.asset_id,
                            resolved_project_id,
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
                    (id, project_id, run_id, asset_id, stage, status, priority, attempt_count, max_attempts, payload)
                    VALUES (%s, %s, %s, %s, %s::{schema}.stage_name, %s::{schema}.job_status, %s, 0, %s, %s::jsonb)
                    """,
                    [
                        (
                            job.job_id,
                            resolved_project_id,
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
        project_id: str | None = None,
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
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
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
        project_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        inner_clauses = []
        outer_clauses = []
        params: list[Any] = []
        if project_id:
            inner_clauses.append("assets.project_id = %s")
            params.append(project_id)
        if collection:
            outer_clauses.append("collection ILIKE %s")
            params.append(f"%{collection}%")
        inner_where = f"WHERE {' AND '.join(inner_clauses)}" if inner_clauses else ""
        outer_where = f"WHERE {' AND '.join(outer_clauses)}" if outer_clauses else ""
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
                        {inner_where}
                        GROUP BY collection
                    ) collections
                    {outer_where}
                    ORDER BY collection ASC
                    LIMIT %s OFFSET %s
                    """
                    ,
                    tuple(params),
                )
                return cursor.fetchall()

    def get_asset(self, asset_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["id = %s"]
                params: list[Any] = [asset_id]
                if project_id:
                    clauses.append("project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"SELECT * FROM {self.schema}.raw_assets WHERE {' AND '.join(clauses)}",
                    tuple(params),
                )
                return cursor.fetchone()

    def update_asset_collections(
        self,
        asset_id: str,
        collections: Any,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        resolved_collections = normalize_collections(collections)
        clauses = ["id = %s"]
        params: list[Any] = [asset_id]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.raw_assets
                    SET
                        collections = %s,
                        metadata = jsonb_set(metadata, '{{collections}}', %s::jsonb, true)
                    WHERE {' AND '.join(clauses)}
                    RETURNING *;
                    """,
                    (resolved_collections, json.dumps(json_ready(resolved_collections)), *params),
                )
                asset = cursor.fetchone()
                if asset is None:
                    connection.rollback()
                    return None
                if str(asset.get("kind")) == AssetKind.TELEMETRY.value:
                    connection.rollback()
                    raise ValueError(
                        "Telemetry source assets are immutable; archive or remove the telemetry source explicitly."
                    )

                status_params: list[Any] = [resolved_collections, str(asset["id"])]
                status_clauses = ["asset_id = %s"]
                if project_id:
                    status_clauses.append("project_id = %s")
                    status_params.append(project_id)
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.frame_processing_status
                    SET collections = %s, updated_at = NOW()
                    WHERE {' AND '.join(status_clauses)};
                    """,
                    tuple(status_params),
                )
            connection.commit()
        return asset

    def delete_asset(self, asset_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        clauses = ["assets.id = %s"]
        params: list[Any] = [asset_id]
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        where = " AND ".join(clauses)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT assets.*,
                           COUNT(DISTINCT frames.id) AS frame_count
                    FROM {self.schema}.raw_assets assets
                    LEFT JOIN {self.schema}.frames frames ON frames.asset_id = assets.id
                    WHERE {where}
                    GROUP BY assets.id;
                    """,
                    tuple(params),
                )
                asset = cursor.fetchone()
                if asset is None:
                    connection.rollback()
                    return None

                cursor.execute(
                    f"""
                    SELECT
                        (
                            SELECT COUNT(*)
                            FROM {self.schema}.detection_candidate detections
                            JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                            WHERE frames.asset_id = %s
                        ) AS candidate_detection_count,
                        (
                            SELECT COUNT(*)
                            FROM {self.schema}.detections_refined refined
                            WHERE refined.frame_id IN (
                                SELECT id FROM {self.schema}.frames WHERE asset_id = %s
                            )
                            OR refined.candidate_detection_id IN (
                                SELECT detections.id
                                FROM {self.schema}.detection_candidate detections
                                JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                                WHERE frames.asset_id = %s
                            )
                        ) AS refined_detection_count;
                    """,
                    (asset_id, asset_id, asset_id),
                )
                detection_counts = cursor.fetchone() or {}

                cursor.execute(
                    f"""
                    SELECT DISTINCT payload_ref
                    FROM (
                        SELECT frames.kvstore_hash AS payload_ref
                        FROM {self.schema}.frames frames
                        WHERE frames.asset_id = %s
                        UNION ALL
                        SELECT frames.payload_ref AS payload_ref
                        FROM {self.schema}.frames frames
                        WHERE frames.asset_id = %s
                        UNION ALL
                        SELECT frames.preprocessed_kvstore_hash AS payload_ref
                        FROM {self.schema}.frames frames
                        WHERE frames.asset_id = %s
                        UNION ALL
                        SELECT frames.preprocessed_payload_ref AS payload_ref
                        FROM {self.schema}.frames frames
                        WHERE frames.asset_id = %s
                        UNION ALL
                        SELECT frames.background_kvstore_hash AS payload_ref
                        FROM {self.schema}.frames frames
                        WHERE frames.asset_id = %s
                        UNION ALL
                        SELECT frames.background_payload_ref AS payload_ref
                        FROM {self.schema}.frames frames
                        WHERE frames.asset_id = %s
                    ) refs
                    WHERE payload_ref IS NOT NULL AND payload_ref <> '';
                    """,
                    (asset_id, asset_id, asset_id, asset_id, asset_id, asset_id),
                )
                payload_refs = sorted({str(row["payload_ref"]) for row in cursor.fetchall() if row.get("payload_ref")})

                unreferenced_refs: list[str] = []
                for payload_ref in payload_refs:
                    cursor.execute(
                        f"""
                        SELECT COUNT(*) AS count
                        FROM {self.schema}.frames frames
                        WHERE frames.asset_id <> %s
                          AND (
                            frames.kvstore_hash = %s
                            OR frames.payload_ref = %s
                            OR frames.preprocessed_kvstore_hash = %s
                            OR frames.preprocessed_payload_ref = %s
                            OR frames.background_kvstore_hash = %s
                            OR frames.background_payload_ref = %s
                          );
                        """,
                        (asset_id, payload_ref, payload_ref, payload_ref, payload_ref, payload_ref, payload_ref),
                    )
                    ref_row = cursor.fetchone()
                    if int(ref_row["count"] if ref_row is not None else 0) == 0:
                        unreferenced_refs.append(payload_ref)

                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.detections_refined refined
                    WHERE refined.frame_id IN (
                        SELECT id FROM {self.schema}.frames WHERE asset_id = %s
                    )
                    OR refined.candidate_detection_id IN (
                        SELECT detections.id
                        FROM {self.schema}.detection_candidate detections
                        JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                        WHERE frames.asset_id = %s
                    );
                    """,
                    (asset_id, asset_id),
                )
                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.detection_candidate detections
                    USING {self.schema}.frames frames
                    WHERE detections.frame_id = frames.id
                      AND frames.asset_id = %s;
                    """,
                    (asset_id,),
                )

                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.raw_assets
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (asset_id,),
                )
                deleted = cursor.fetchone()
            connection.commit()
        if deleted is None:
            return None
        return {
            "asset": deleted,
            "frame_count": int(asset.get("frame_count") or 0),
            "candidate_detection_count": int(detection_counts.get("candidate_detection_count") or 0),
            "refined_detection_count": int(detection_counts.get("refined_detection_count") or 0),
            "generated_kvstore_keys": payload_refs,
            "unreferenced_kvstore_keys": unreferenced_refs,
        }

    def count_frames(self, asset_id: str, *, project_id: str | None = None) -> int:
        clauses = ["frames.asset_id = %s"]
        params: list[Any] = [asset_id]
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT COUNT(*) AS frame_count
                    FROM {self.schema}.frames frames
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    """,
                    tuple(params),
                )
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
        project_id: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["frames.asset_id = %s"]
        params: list[Any] = [asset_id]
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
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
                    SELECT frames.*
                    FROM {self.schema}.frames frames
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY frames.frame_index DESC
                    {limit_sql}
                    {offset_sql}
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def get_frame(self, frame_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["frames.id = %s"]
                params: list[Any] = [frame_id]
                if project_id:
                    clauses.append("assets.project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"""
                    SELECT frames.*
                    FROM {self.schema}.frames frames
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    """,
                    tuple(params),
                )
                return cursor.fetchone()

    def get_frame_by_asset_index(
        self,
        asset_id: str,
        frame_index: int,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["frames.asset_id = %s", "frames.frame_index = %s"]
                params: list[Any] = [asset_id, frame_index]
                if project_id:
                    clauses.append("assets.project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"""
                    SELECT frames.*
                    FROM {self.schema}.frames frames
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    """,
                    tuple(params),
                )
                return cursor.fetchone()

    def list_frame_records(
        self,
        asset_id: str,
        project_id: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[FrameRecord]:
        return [
            FrameRecord.from_row(row)
            for row in self.list_frames(
                asset_id,
                project_id=project_id,
                start_frame=start_frame,
                end_frame=end_frame,
                limit=limit,
                offset=offset,
            )
        ]

    def get_frame_record(self, frame_id: str, *, project_id: str | None = None) -> FrameRecord | None:
        row = self.get_frame(frame_id, project_id=project_id)
        if row is None:
            return None
        return FrameRecord.from_row(row)

    def get_frame_records(
        self,
        frame_ids: Sequence[str],
        *,
        project_id: str | None = None,
    ) -> list[FrameRecord]:
        """Load selected frame records in caller-provided order."""
        resolved_frame_ids = [str(frame_id) for frame_id in frame_ids]
        if not resolved_frame_ids:
            return []
        clauses = ["frames.id = ANY(%s)"]
        params: list[Any] = [resolved_frame_ids]
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT frames.*
                    FROM {self.schema}.frames frames
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    """,
                    tuple(params),
                )
                rows = cursor.fetchall()
        records_by_id = {
            str(row["id"]): FrameRecord.from_row(row)
            for row in rows
        }
        return [records_by_id[frame_id] for frame_id in resolved_frame_ids if frame_id in records_by_id]

    def create_live_frame_copy(
        self,
        frame_id: str,
        *,
        operation: str,
        project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a sandbox frame row that shares the source payload but owns live outputs."""
        live_metadata = {
            "live_preview": {
                "is_sandbox": True,
                "source_frame_id": str(frame_id),
                "operation": str(operation),
                **dict(metadata or {}),
            }
        }
        with self.connect() as connection:
            with connection.cursor() as cursor:
                project_clause = ""
                project_params: list[Any] = []
                if project_id is not None:
                    project_clause = "AND assets.project_id = %s"
                    project_params.append(project_id)
                cursor.execute(
                    f"""
                    SELECT assets.id
                    FROM {self.schema}.raw_assets assets
                    JOIN {self.schema}.frames source ON source.asset_id = assets.id
                    WHERE source.id = %s
                      {project_clause}
                    FOR UPDATE
                    """,
                    (frame_id, *project_params),
                )
                if cursor.fetchone() is None:
                    raise KeyError(frame_id)
                cursor.execute(
                    f"""
                    WITH source AS (
                        SELECT *
                        FROM {self.schema}.frames
                        WHERE id = %s
                    ),
                    next_index AS (
                        SELECT
                            CASE
                                WHEN MIN(frame_index) FILTER (WHERE frame_index < 0) IS NULL THEN -1
                                ELSE MIN(frame_index) FILTER (WHERE frame_index < 0) - 1
                            END AS frame_index
                        FROM {self.schema}.frames
                        WHERE asset_id = (SELECT asset_id FROM source)
                    )
                    INSERT INTO {self.schema}.frames
                    (run_id, asset_id, frame_index, captured_at, width, height,
                     bbox_x, bbox_y, parent_frame_id, source_ref, kvstore_hash, preview_thumbhash,
                     payload_ref, payload_encoding, payload_format, payload_dtype, payload_shape,
                     background_kvstore_hash, background_payload_ref, background_payload_encoding,
                     background_payload_format, background_payload_dtype, background_payload_shape,
                     background_metadata, flatfield_profile, flatfield_metadata, metadata)
                    SELECT
                        source.run_id,
                        source.asset_id,
                        next_index.frame_index,
                        source.captured_at,
                        source.width,
                        source.height,
                        source.bbox_x,
                        source.bbox_y,
                        source.id,
                        source.source_ref,
                        source.kvstore_hash,
                        source.preview_thumbhash,
                        source.payload_ref,
                        source.payload_encoding,
                        source.payload_format,
                        source.payload_dtype,
                        source.payload_shape,
                        source.background_kvstore_hash,
                        source.background_payload_ref,
                        source.background_payload_encoding,
                        source.background_payload_format,
                        source.background_payload_dtype,
                        source.background_payload_shape,
                        source.background_metadata,
                        source.flatfield_profile,
                        source.flatfield_metadata,
                        COALESCE(source.metadata, '{{}}'::jsonb) || %s::jsonb
                    FROM source, next_index
                    RETURNING *;
                    """,
                    (frame_id, json.dumps(json_ready(live_metadata))),
                )
                row = cursor.fetchone()
            connection.commit()
        if row is None:
            raise KeyError(frame_id)
        return row

    def list_live_frame_copies(
        self,
        *,
        source_frame_id: str | None = None,
        operation: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["frames.metadata->'live_preview'->>'is_sandbox' = 'true'"]
        params: list[Any] = []
        if source_frame_id:
            clauses.append("frames.metadata->'live_preview'->>'source_frame_id' = %s")
            params.append(str(source_frame_id))
        if operation:
            clauses.append("frames.metadata->'live_preview'->>'operation' = %s")
            params.append(str(operation))
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        params.extend([limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT frames.*
                    FROM {self.schema}.frames
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY frames.created_at DESC, frames.id DESC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def count_frame_payload_references(
        self,
        payload_ref: str,
        *,
        exclude_frame_id: str | None = None,
    ) -> int:
        clauses = [
            """
            (
                kvstore_hash = %s
                OR payload_ref = %s
                OR preprocessed_kvstore_hash = %s
                OR preprocessed_payload_ref = %s
                OR background_kvstore_hash = %s
                OR background_payload_ref = %s
            )
            """
        ]
        params: list[Any] = [payload_ref] * 6
        if exclude_frame_id is not None:
            clauses.append("id <> %s")
            params.append(exclude_frame_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT COUNT(*) AS count
                    FROM {self.schema}.frames
                    WHERE {' AND '.join(clauses)}
                    """,
                    tuple(params),
                )
                row = cursor.fetchone()
        return int(row["count"] if row is not None else 0)

    def delete_live_frame_copy(self, frame_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        """Delete one live-preview sandbox frame and return generated payload refs."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                project_clause = ""
                project_params: list[Any] = []
                if project_id is not None:
                    project_clause = "AND assets.project_id = %s"
                    project_params.append(project_id)
                cursor.execute(
                    f"""
                    SELECT frames.*
                    FROM {self.schema}.frames
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE frames.id = %s
                      AND frames.metadata->'live_preview'->>'is_sandbox' = 'true'
                      {project_clause}
                    """,
                    (frame_id, *project_params),
                )
                row = cursor.fetchone()
                if row is None:
                    connection.rollback()
                    return None
                generated_keys = sorted(
                    {
                        str(key)
                        for key in (
                            row.get("preprocessed_payload_ref"),
                            row.get("preprocessed_kvstore_hash"),
                            row.get("background_payload_ref"),
                            row.get("background_kvstore_hash"),
                        )
                        if key
                    }
                )
                unreferenced_keys = []
                for key in generated_keys:
                    cursor.execute(
                        f"""
                        SELECT COUNT(*) AS count
                        FROM {self.schema}.frames
                        WHERE id <> %s
                          AND (
                            kvstore_hash = %s
                            OR payload_ref = %s
                            OR preprocessed_kvstore_hash = %s
                            OR preprocessed_payload_ref = %s
                            OR background_kvstore_hash = %s
                            OR background_payload_ref = %s
                          )
                        """,
                        (frame_id, key, key, key, key, key, key),
                    )
                    ref_row = cursor.fetchone()
                    if int(ref_row["count"] if ref_row is not None else 0) == 0:
                        unreferenced_keys.append(key)
                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.frames
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (frame_id,),
                )
                deleted = cursor.fetchone()
            connection.commit()
        if deleted is None:
            return None
        return {
            "frame": deleted,
            "generated_kvstore_keys": generated_keys,
            "unreferenced_kvstore_keys": unreferenced_keys,
        }

    def update_frame_preprocessed_payload(
        self,
        frame_id: str,
        *,
        project_id: str | None = None,
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
                self._ensure_project_scope(cursor, project_id, frame_ids=[frame_id])
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

    def update_frame_preprocessed_payloads(
        self,
        payloads: Sequence[dict[str, Any]],
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Persist preprocessed payload metadata in one database transaction."""
        if not payloads:
            return []
        frame_ids = [str(payload["frame_id"]) for payload in payloads]
        statement = f"""
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
        """
        parameters = [
            (
                payload["kvstore_hash"],
                payload["preview_thumbhash"],
                payload["payload_ref"],
                payload["payload_encoding"],
                payload["payload_format"],
                payload["payload_dtype"],
                json.dumps(json_ready(list(payload["payload_shape"]))),
                json.dumps(json_ready(payload.get("metadata") or {})),
                payload["frame_id"],
            )
            for payload in payloads
        ]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_project_scope(cursor, project_id, frame_ids=frame_ids)
                cursor.executemany(statement, parameters)
                cursor.execute(
                    f"SELECT * FROM {self.schema}.frames WHERE id = ANY(%s)",
                    (frame_ids,),
                )
                rows = cursor.fetchall()
            connection.commit()
        rows_by_id = {str(row["id"]): row for row in rows}
        missing = [frame_id for frame_id in frame_ids if frame_id not in rows_by_id]
        if missing:
            raise KeyError(missing[0])
        return [rows_by_id[frame_id] for frame_id in frame_ids]

    def update_frame_background_payloads(
        self,
        frame_ids: Sequence[str],
        *,
        project_id: str | None = None,
        kvstore_hash: str,
        payload_ref: str,
        payload_encoding: str,
        payload_format: str,
        payload_dtype: str,
        payload_shape: Sequence[int],
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        resolved_frame_ids = [str(frame_id) for frame_id in frame_ids]
        if not resolved_frame_ids:
            return []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_project_scope(cursor, project_id, frame_ids=resolved_frame_ids)
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.frames
                    SET
                        background_kvstore_hash = %s,
                        background_payload_ref = %s,
                        background_payload_encoding = %s,
                        background_payload_format = %s,
                        background_payload_dtype = %s,
                        background_payload_shape = %s::jsonb,
                        background_metadata = %s::jsonb
                    WHERE id = ANY(%s::uuid[])
                    RETURNING *;
                    """,
                    (
                        kvstore_hash,
                        payload_ref,
                        payload_encoding,
                        payload_format,
                        payload_dtype,
                        json.dumps(json_ready(list(payload_shape))),
                        json.dumps(json_ready(metadata or {})),
                        resolved_frame_ids,
                    ),
                )
                rows = cursor.fetchall()
            connection.commit()
        if len(rows) != len(resolved_frame_ids):
            found_ids = {str(row["id"]) for row in rows}
            missing = [frame_id for frame_id in resolved_frame_ids if frame_id not in found_ids]
            raise KeyError(f"Frame(s) not found: {', '.join(missing)}")
        return rows

    def update_frame_background_payload_assignments(
        self,
        assignments: Sequence[dict[str, Any]],
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Assign independently generated background payloads in one transaction."""
        if not assignments:
            return []
        frame_ids = [str(assignment["frame_id"]) for assignment in assignments]
        payload = [
            {
                **assignment,
                "frame_id": str(assignment["frame_id"]),
                "payload_shape": (
                    json_ready(list(assignment.get("payload_shape") or []))
                    if "payload_shape" in assignment
                    else None
                ),
                "metadata": (
                    json_ready(dict(assignment.get("metadata") or {}))
                    if "metadata" in assignment
                    else None
                ),
                "flatfield_profile": json_ready(assignment.get("flatfield_profile")),
                "flatfield_metadata": (
                    json_ready(dict(assignment.get("flatfield_metadata") or {}))
                    if "flatfield_metadata" in assignment
                    else None
                ),
            }
            for assignment in assignments
        ]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_project_scope(cursor, project_id, frame_ids=frame_ids)
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.frames AS frames
                    SET
                        background_kvstore_hash = COALESCE(assignment.kvstore_hash, frames.background_kvstore_hash),
                        background_payload_ref = COALESCE(assignment.payload_ref, frames.background_payload_ref),
                        background_payload_encoding = COALESCE(assignment.payload_encoding, frames.background_payload_encoding),
                        background_payload_format = COALESCE(assignment.payload_format, frames.background_payload_format),
                        background_payload_dtype = COALESCE(assignment.payload_dtype, frames.background_payload_dtype),
                        background_payload_shape = COALESCE(assignment.payload_shape, frames.background_payload_shape),
                        background_metadata = COALESCE(assignment.metadata, frames.background_metadata),
                        flatfield_profile = COALESCE(assignment.flatfield_profile, frames.flatfield_profile),
                        flatfield_metadata = COALESCE(assignment.flatfield_metadata, frames.flatfield_metadata)
                    FROM jsonb_to_recordset(%s::jsonb) AS assignment(
                        frame_id uuid,
                        kvstore_hash text,
                        payload_ref text,
                        payload_encoding text,
                        payload_format text,
                        payload_dtype text,
                        payload_shape jsonb,
                        metadata jsonb,
                        flatfield_profile real[],
                        flatfield_metadata jsonb
                    )
                    WHERE frames.id = assignment.frame_id
                    RETURNING frames.*;
                    """,
                    (json.dumps(payload),),
                )
                rows = cursor.fetchall()
            connection.commit()
        if len(rows) != len(frame_ids):
            found_ids = {str(row["id"]) for row in rows}
            missing = [frame_id for frame_id in frame_ids if frame_id not in found_ids]
            raise KeyError(f"Frame(s) not found: {', '.join(missing)}")
        return rows

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
        *,
        job_id: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not refined_detections:
            return []
        inserted: list[dict[str, Any]] = []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_project_scope(
                    cursor,
                    project_id,
                    job_ids=[job_id] if job_id else None,
                    detection_ids=[candidate_id for candidate_id, _ in refined_detections],
                    frame_ids=[detection.frame_id for _, detection in refined_detections],
                )
                for candidate_detection_id, detection in refined_detections:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.detections_refined
                        (candidate_detection_id, job_id, run_id, frame_id, roi_index, bbox_x, bbox_y, bbox_w, bbox_h,
                         crop_bbox_x, crop_bbox_y, crop_bbox_w, crop_bbox_h,
                         area, perimeter, major_axis_length, minor_axis_length,
                         min_gray_value, mean_gray_value, roi_payload, mask_payload,
                         roi_encoding, roi_format, roi_dtype, roi_shape,
                         mask_encoding, mask_format, mask_dtype, mask_shape, refinement_method, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                                %s, %s, %s::jsonb, %s, %s::jsonb)
                        RETURNING *;
                        """,
                        (
                            candidate_detection_id,
                            job_id,
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
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        resolved_frame_ids = [str(frame_id) for frame_id in frame_ids]
        if not resolved_frame_ids:
            return []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_project_scope(
                    cursor,
                    project_id,
                    run_id=run_id,
                    frame_ids=resolved_frame_ids,
                )
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
        project_id: str | None = None,
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
        has_roi_payload: bool | None = None,
        refinement_state: str | None = None,
        telemetry_filters: Sequence[Mapping[str, Any]] | None = None,
        sort_by: str = "asset_frame",
        sort_dir: str = "desc",
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
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

        telemetry_clauses, telemetry_params = self._telemetry_filter_clauses(
            telemetry_filters,
            timestamp_sql="frames.captured_at",
            run_sql="detections.run_id",
            project_sql="assets.project_id",
        )
        clauses.extend(telemetry_clauses)
        params.extend(telemetry_params)

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

        if has_roi_payload is True:
            clauses.append("detections.roi_payload IS NOT NULL AND octet_length(detections.roi_payload) > 0")
        elif has_roi_payload is False:
            clauses.append("(detections.roi_payload IS NULL OR octet_length(detections.roi_payload) = 0)")

        normalized_refinement_state = str(refinement_state or "").replace("_", "-").lower()
        if normalized_refinement_state in {"refined", "has-refinement", "has-refined"}:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM {self.schema}.detections_refined refined
                    WHERE refined.candidate_detection_id = detections.id
                )
                """
            )
        elif normalized_refinement_state in {"unrefined", "needs-refinement", "needs-refined", "none"}:
            clauses.append(
                f"""
                NOT EXISTS (
                    SELECT 1
                    FROM {self.schema}.detections_refined refined
                    WHERE refined.candidate_detection_id = detections.id
                )
                """
            )
        refinement_join = f"""
                    LEFT JOIN LATERAL (
                        SELECT refined.id, refined.refinement_method
                        FROM {self.schema}.detections_refined refined
                        WHERE refined.candidate_detection_id = detections.id
                        ORDER BY refined.created_at DESC, refined.id DESC
                        LIMIT 1
                    ) refined ON TRUE
        """

        direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"
        sort_key = str(sort_by or "asset_frame").lower()
        order_by_options = {
            "area": f"detections.area {direction} NULLS LAST, frames.frame_index DESC, detections.roi_index ASC",
            "byte_size": f"octet_length(detections.roi_payload) {direction} NULLS LAST, frames.frame_index DESC, detections.roi_index ASC",
            "id": f"detections.id {direction}",
            "asset_frame": f"assets.filename {direction} NULLS LAST, frames.frame_index {direction}, detections.roi_index {direction}",
            "random": "random()",
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
                        frames.captured_at AS captured_at,
                        assets.filename AS asset_filename,
                        refined.id AS refined_detection_id,
                        refined.refinement_method AS refined_detection_method
                    FROM {self.schema}.detection_candidate detections
                    JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    {refinement_join}
                    {where}
                    ORDER BY {order_by}
                    {limit_sql}
                    {offset_sql}
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def _telemetry_filter_clauses(
        self,
        telemetry_filters: Sequence[Any] | None,
        *,
        timestamp_sql: str,
        run_sql: str,
        project_sql: str,
    ) -> tuple[list[str], list[Any]]:
        """Build correlated range predicates using configured telemetry lookup rules."""
        clauses: list[str] = []
        params: list[Any] = []
        for index, telemetry_filter in enumerate(telemetry_filters or ()):
            if isinstance(telemetry_filter, Mapping):
                parameter_key = str(
                    telemetry_filter.get("parameter_key")
                    or telemetry_filter.get("parameter")
                    or ""
                )
                min_value = telemetry_filter.get("min_value", telemetry_filter.get("min"))
                max_value = telemetry_filter.get("max_value", telemetry_filter.get("max"))
            else:
                parameter_key = str(getattr(telemetry_filter, "parameter_key", ""))
                min_value = getattr(telemetry_filter, "min_value", None)
                max_value = getattr(telemetry_filter, "max_value", None)
            stream_alias = f"telemetry_stream_{index}"
            previous_alias = f"telemetry_previous_{index}"
            next_alias = f"telemetry_next_{index}"
            value_alias = f"telemetry_value_{index}"
            resolved_value = f"{value_alias}.value"
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM (
                        SELECT streams.id, streams.interpolation, streams.max_gap, streams.metadata
                        FROM {self.schema}.telemetry_streams streams
                        JOIN {self.schema}.telemetry_parameters parameters
                          ON parameters.id = streams.parameter_id
                        WHERE streams.project_id = {project_sql}
                          AND streams.run_id = {run_sql}
                          AND parameters.parameter_key = %s
                        ORDER BY streams.is_default DESC, streams.priority ASC, streams.id ASC
                        LIMIT 1
                    ) {stream_alias}
                    LEFT JOIN LATERAL (
                        SELECT observations.observed_at, observations.value
                        FROM {self.schema}.telemetry_observations observations
                        WHERE observations.stream_id = {stream_alias}.id
                          AND observations.observed_at <= {timestamp_sql}
                          AND NOT COALESCE(
                              {stream_alias}.metadata->'excluded_qc_flags'
                              @> to_jsonb(observations.qc_flag), false
                          )
                        ORDER BY observations.observed_at DESC
                        LIMIT 1
                    ) {previous_alias} ON TRUE
                    LEFT JOIN LATERAL (
                        SELECT observations.observed_at, observations.value
                        FROM {self.schema}.telemetry_observations observations
                        WHERE observations.stream_id = {stream_alias}.id
                          AND observations.observed_at > {timestamp_sql}
                          AND NOT COALESCE(
                              {stream_alias}.metadata->'excluded_qc_flags'
                              @> to_jsonb(observations.qc_flag), false
                          )
                        ORDER BY observations.observed_at ASC
                        LIMIT 1
                    ) {next_alias} ON TRUE
                    CROSS JOIN LATERAL (
                        SELECT CASE
                            WHEN {timestamp_sql} IS NULL THEN NULL
                            WHEN {previous_alias}.observed_at = {timestamp_sql}
                                THEN {previous_alias}.value
                            WHEN {stream_alias}.interpolation = 'none'
                                THEN NULL
                            WHEN {stream_alias}.interpolation = 'previous'
                                AND {previous_alias}.observed_at IS NOT NULL
                                AND ({stream_alias}.max_gap IS NULL OR {timestamp_sql} - {previous_alias}.observed_at <= {stream_alias}.max_gap)
                                THEN {previous_alias}.value
                            WHEN {stream_alias}.interpolation = 'nearest'
                                AND (
                                    {stream_alias}.max_gap IS NULL
                                    OR LEAST(
                                        COALESCE({timestamp_sql} - {previous_alias}.observed_at, interval '100000 years'),
                                        COALESCE({next_alias}.observed_at - {timestamp_sql}, interval '100000 years')
                                    ) <= {stream_alias}.max_gap
                                )
                                THEN CASE
                                    WHEN {previous_alias}.observed_at IS NULL THEN {next_alias}.value
                                    WHEN {next_alias}.observed_at IS NULL THEN {previous_alias}.value
                                    WHEN {timestamp_sql} - {previous_alias}.observed_at <= {next_alias}.observed_at - {timestamp_sql}
                                        THEN {previous_alias}.value
                                    ELSE {next_alias}.value
                                END
                            WHEN {stream_alias}.interpolation = 'linear'
                                AND {previous_alias}.observed_at IS NOT NULL
                                AND {next_alias}.observed_at IS NOT NULL
                                AND ({stream_alias}.max_gap IS NULL OR {next_alias}.observed_at - {previous_alias}.observed_at <= {stream_alias}.max_gap)
                                THEN {previous_alias}.value
                                    + EXTRACT(EPOCH FROM ({timestamp_sql} - {previous_alias}.observed_at))
                                    / NULLIF(EXTRACT(EPOCH FROM ({next_alias}.observed_at - {previous_alias}.observed_at)), 0)
                                    * ({next_alias}.value - {previous_alias}.value)
                            ELSE NULL
                        END AS value
                    ) {value_alias}
                    WHERE {resolved_value} IS NOT NULL
                      {f"AND {resolved_value} >= %s" if min_value is not None else ""}
                      {f"AND {resolved_value} <= %s" if max_value is not None else ""}
                )
                """
            )
            params.append(parameter_key)
            if min_value is not None:
                params.append(min_value)
            if max_value is not None:
                params.append(max_value)
        return clauses, params

    def get_detections(
        self,
        detection_ids: Sequence[str],
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load candidate detections in request order using one query and connection."""

        resolved_ids = [str(detection_id) for detection_id in detection_ids if detection_id]
        if not resolved_ids:
            return []
        unique_ids = list(dict.fromkeys(resolved_ids))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["detections.id = ANY(%s)"]
                params: list[Any] = [unique_ids]
                if project_id:
                    clauses.append("assets.project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"""
                    SELECT
                        detections.*,
                        frames.asset_id,
                        frames.frame_index,
                        frames.captured_at AS captured_at,
                        assets.filename AS asset_filename,
                        refined.id AS refined_detection_id,
                        refined.refinement_method AS refined_detection_method
                    FROM {self.schema}.detection_candidate detections
                    JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    LEFT JOIN LATERAL (
                        SELECT refined.id, refined.refinement_method
                        FROM {self.schema}.detections_refined refined
                        WHERE refined.candidate_detection_id = detections.id
                        ORDER BY refined.created_at DESC, refined.id DESC
                        LIMIT 1
                    ) refined ON TRUE
                    WHERE {' AND '.join(clauses)}
                    """,
                    tuple(params),
                )
                rows_by_id = {str(row["id"]): row for row in cursor.fetchall()}
        return [rows_by_id[detection_id] for detection_id in resolved_ids if detection_id in rows_by_id]

    def get_detection(self, detection_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        rows = self.get_detections([detection_id], project_id=project_id)
        return rows[0] if rows else None

    def get_refined_detection_for_candidate(
        self,
        detection_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["refined.candidate_detection_id = %s"]
                params: list[Any] = [detection_id]
                if project_id:
                    clauses.append("assets.project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"""
                    SELECT
                        refined.*,
                        refined.candidate_detection_id,
                        frames.asset_id,
                        frames.frame_index,
                        frames.captured_at AS captured_at,
                        assets.filename AS asset_filename
                    FROM {self.schema}.detections_refined refined
                    JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY refined.created_at DESC, refined.id DESC
                    LIMIT 1
                    """,
                    tuple(params),
                )
                return cursor.fetchone()

    def get_refined_detection(
        self,
        refined_detection_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["refined.id = %s"]
                params: list[Any] = [refined_detection_id]
                if project_id:
                    clauses.append("assets.project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"""
                    SELECT
                        refined.*,
                        refined.candidate_detection_id,
                        frames.asset_id,
                        frames.frame_index,
                        frames.captured_at AS captured_at,
                        assets.filename AS asset_filename
                    FROM {self.schema}.detections_refined refined
                    JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    """,
                    tuple(params),
                )
                return cursor.fetchone()

    def list_detection_records(self, asset_id: str) -> list[DetectionRecord]:
        return [DetectionRecord.from_row(row) for row in self.list_detections(asset_id)]

    def list_asset_detection_stats(
        self,
        *,
        project_id: str | None = None,
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
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
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

    def list_asset_processing_state(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        preprocessing_state: str | None = None,
        detection_state: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses = []
        params: list[Any] = []
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
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

        state_clauses = []
        if preprocessing_state in {"has-preprocessed", "has_preprocessed"}:
            state_clauses.append("preprocessed_frame_count > 0")
        elif preprocessing_state in {"needs-preprocessed", "needs_preprocessed", "none"}:
            state_clauses.append("preprocessed_frame_count = 0")
        elif preprocessing_state in {"fully-preprocessed", "fully_preprocessed", "complete"}:
            state_clauses.append("frame_count > 0 AND preprocessed_frame_count = frame_count")
        elif preprocessing_state in {"partially-preprocessed", "partially_preprocessed", "partial"}:
            state_clauses.append("preprocessed_frame_count > 0 AND preprocessed_frame_count < frame_count")

        if detection_state in {"has-detections", "has_detections"}:
            state_clauses.append("detection_count > 0")
        elif detection_state in {"needs-detections", "needs_detections", "none"}:
            state_clauses.append("detection_count = 0")
        elif detection_state in {"fully-detected", "fully_detected", "complete"}:
            state_clauses.append("frame_count > 0 AND detected_frame_count = frame_count")
        elif detection_state in {"partially-detected", "partially_detected", "partial"}:
            state_clauses.append("detected_frame_count > 0 AND detected_frame_count < frame_count")
        state_where = f"WHERE {' AND '.join(state_clauses)}" if state_clauses else ""

        query = f"""
            WITH asset_processing_counts AS (
                SELECT
                    assets.id AS asset_id,
                    assets.run_id,
                    assets.filename,
                    assets.kind,
                    assets.collections,
                    COUNT(DISTINCT frames.id) AS frame_count,
                    COUNT(DISTINCT frames.id) FILTER (
                        WHERE frames.preprocessed_payload_ref IS NOT NULL
                           OR frames.preprocessed_kvstore_hash IS NOT NULL
                    ) AS preprocessed_frame_count,
                    COUNT(DISTINCT frames.id) FILTER (
                        WHERE detections.id IS NOT NULL
                    ) AS detected_frame_count,
                    COUNT(detections.id) AS detection_count
                FROM {self.schema}.raw_assets assets
                LEFT JOIN {self.schema}.frames frames ON frames.asset_id = assets.id
                LEFT JOIN {self.schema}.detection_candidate detections ON detections.frame_id = frames.id
                {where}
                GROUP BY assets.id, assets.run_id, assets.filename, assets.kind, assets.collections
            ),
            asset_processing_state AS (
                SELECT
                    *,
                    CASE
                        WHEN frame_count > 0 AND preprocessed_frame_count = frame_count THEN 'fully-preprocessed'
                        WHEN preprocessed_frame_count > 0 THEN 'partially-preprocessed'
                        ELSE 'needs-preprocessed'
                    END AS preprocessing_state,
                    CASE
                        WHEN frame_count > 0 AND detected_frame_count = frame_count THEN 'fully-detected'
                        WHEN detection_count > 0 THEN 'partially-detected'
                        ELSE 'needs-detections'
                    END AS detection_state
                FROM asset_processing_counts
            )
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query
                    + f"""
                    SELECT
                        COUNT(*) AS total_asset_count,
                        COALESCE(SUM(frame_count), 0) AS total_frame_count,
                        COALESCE(SUM(preprocessed_frame_count), 0) AS total_preprocessed_frame_count,
                        COALESCE(SUM(detected_frame_count), 0) AS total_detected_frame_count,
                        COALESCE(SUM(detection_count), 0) AS total_detection_count
                    FROM asset_processing_state
                    {state_where}
                    """,
                    tuple(params),
                )
                summary = cursor.fetchone()
                cursor.execute(
                    query
                    + f"""
                    SELECT *
                    FROM asset_processing_state
                    {state_where}
                    ORDER BY filename ASC NULLS LAST, asset_id ASC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params + [limit, max(0, int(offset))]),
                )
                assets = cursor.fetchall()

        return {
            "summary": {
                "total_asset_count": summary["total_asset_count"],
                "total_frame_count": summary["total_frame_count"],
                "total_preprocessed_frame_count": summary["total_preprocessed_frame_count"],
                "total_detected_frame_count": summary["total_detected_frame_count"],
                "total_detection_count": summary["total_detection_count"],
            },
            "assets": assets,
        }

    def list_frame_processing_state(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        filename: str | None = None,
        preprocessing_state: str | None = None,
        detection_state: str | None = None,
        refinement_state: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        sort_by: str = "asset_frame",
        sort_dir: str = "asc",
        limit: int = 1000,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses = []
        params: list[Any] = []
        if project_id:
            clauses.append("assets.project_id = %s")
            params.append(project_id)
        if run_id:
            clauses.append("assets.run_id = %s")
            params.append(run_id)
        if asset_id:
            clauses.append("assets.id = %s")
            params.append(asset_id)
        if collection:
            clauses.append("%s = ANY(assets.collections)")
            params.append(collection)
        if kind:
            clauses.append("assets.kind = %s")
            params.append(kind)
        if filename:
            clauses.append("assets.filename ILIKE %s")
            params.append(f"%{filename}%")
        if start_frame is not None:
            clauses.append("frames.frame_index >= %s")
            params.append(start_frame)
        if end_frame is not None:
            clauses.append("frames.frame_index <= %s")
            params.append(end_frame)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        state_clauses = []
        if preprocessing_state in {"has-preprocessed", "has_preprocessed"}:
            state_clauses.append("has_preprocessed_payload")
        elif preprocessing_state in {"needs-preprocessed", "needs_preprocessed", "none"}:
            state_clauses.append("NOT has_preprocessed_payload")
        elif preprocessing_state in {"fully-preprocessed", "fully_preprocessed", "complete"}:
            state_clauses.append("has_preprocessed_payload")
        elif preprocessing_state in {"partially-preprocessed", "partially_preprocessed", "partial"}:
            state_clauses.append("FALSE")

        if detection_state in {"has-detections", "has_detections"}:
            state_clauses.append("detection_count > 0")
        elif detection_state in {"needs-detections", "needs_detections", "none"}:
            state_clauses.append("detection_count = 0")
        elif detection_state in {"fully-detected", "fully_detected", "complete"}:
            state_clauses.append("detection_count > 0")
        elif detection_state in {"partially-detected", "partially_detected", "partial"}:
            state_clauses.append("FALSE")
        if refinement_state in {"has-refinement", "has_refinement", "refined"}:
            state_clauses.append("refined_candidate_detection_count > 0")
        elif refinement_state in {"needs-refinement", "needs_refinement", "unrefined", "none"}:
            state_clauses.append("unrefined_detection_count > 0")
        elif refinement_state in {"fully-refined", "fully_refined", "complete"}:
            state_clauses.append("detection_count > 0 AND unrefined_detection_count = 0")
        elif refinement_state in {"partially-refined", "partially_refined", "partial"}:
            state_clauses.append("refined_candidate_detection_count > 0 AND unrefined_detection_count > 0")
        elif refinement_state in {"no-detections", "no_detections"}:
            state_clauses.append("detection_count = 0")
        state_where = f"WHERE {' AND '.join(state_clauses)}" if state_clauses else ""
        sort_key = str(sort_by or "asset_frame").lower()
        direction = "DESC" if str(sort_dir or "asc").lower() == "desc" else "ASC"
        order_by_options = {
            "asset_frame": f"asset_filename {direction} NULLS LAST, asset_id {direction}, frame_index {direction}",
            "frame": f"frame_index {direction}, asset_filename ASC NULLS LAST, asset_id ASC",
            "captured_at": f"captured_at {direction} NULLS LAST, asset_filename ASC NULLS LAST, frame_index ASC",
            "filename": f"asset_filename {direction} NULLS LAST, frame_index ASC",
            "roi_count": f"detection_count {direction}, asset_filename ASC NULLS LAST, frame_index ASC",
            "refined_count": f"refined_detection_count {direction}, asset_filename ASC NULLS LAST, frame_index ASC",
        }
        order_by = order_by_options.get(sort_key, order_by_options["asset_frame"])

        query = f"""
            WITH frame_processing_counts AS (
                SELECT
                    frames.id AS frame_id,
                    frames.run_id,
                    frames.asset_id,
                    frames.frame_index,
                    frames.captured_at,
                    assets.filename AS asset_filename,
                    assets.kind,
                    assets.collections,
                    (
                        frames.preprocessed_payload_ref IS NOT NULL
                        OR frames.preprocessed_kvstore_hash IS NOT NULL
                    ) AS has_preprocessed_payload,
                    COUNT(DISTINCT detections.id) AS detection_count,
                    COUNT(DISTINCT detections.id) FILTER (
                        WHERE refined.candidate_detection_id IS NOT NULL
                    ) AS refined_candidate_detection_count,
                    COUNT(refined.id) AS refined_detection_count
                FROM {self.schema}.frames frames
                JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                LEFT JOIN {self.schema}.detection_candidate detections ON detections.frame_id = frames.id
                LEFT JOIN {self.schema}.detections_refined refined ON refined.candidate_detection_id = detections.id
                {where}
                GROUP BY
                    frames.id,
                    frames.run_id,
                    frames.asset_id,
                    frames.frame_index,
                    frames.captured_at,
                    assets.filename,
                    assets.kind,
                    assets.collections,
                    frames.preprocessed_payload_ref,
                    frames.preprocessed_kvstore_hash
            ),
            frame_processing_state AS (
                SELECT
                    *,
                    CASE
                        WHEN has_preprocessed_payload THEN 'fully-preprocessed'
                        ELSE 'needs-preprocessed'
                    END AS preprocessing_state,
                    CASE
                        WHEN detection_count > 0 THEN 'fully-detected'
                        ELSE 'needs-detections'
                    END AS detection_state,
                    GREATEST(detection_count - refined_candidate_detection_count, 0) AS unrefined_detection_count,
                    CASE
                        WHEN detection_count = 0 THEN 'no-detections'
                        WHEN refined_candidate_detection_count = 0 THEN 'needs-refinement'
                        WHEN refined_candidate_detection_count >= detection_count THEN 'fully-refined'
                        ELSE 'partially-refined'
                    END AS refinement_state
                FROM frame_processing_counts
            )
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query
                    + f"""
                    SELECT
                        COUNT(*) AS total_frame_count,
                        COUNT(*) FILTER (WHERE has_preprocessed_payload) AS total_preprocessed_frame_count,
                        COUNT(*) FILTER (WHERE detection_count > 0) AS total_detected_frame_count,
                        COALESCE(SUM(detection_count), 0) AS total_detection_count,
                        COALESCE(SUM(refined_candidate_detection_count), 0) AS total_refined_candidate_detection_count,
                        COALESCE(SUM(unrefined_detection_count), 0) AS total_unrefined_detection_count,
                        COALESCE(SUM(refined_detection_count), 0) AS total_refined_detection_count
                    FROM frame_processing_state
                    {state_where}
                    """,
                    tuple(params),
                )
                summary = cursor.fetchone()
                cursor.execute(
                    query
                    + f"""
                    SELECT *
                    FROM frame_processing_state
                    {state_where}
                    ORDER BY {order_by}
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params + [limit, max(0, int(offset))]),
                )
                frames = cursor.fetchall()

        return {
            "summary": {
                "total_frame_count": summary["total_frame_count"],
                "total_preprocessed_frame_count": summary["total_preprocessed_frame_count"],
                "total_detected_frame_count": summary["total_detected_frame_count"],
                "total_detection_count": summary["total_detection_count"],
                "total_refined_candidate_detection_count": summary["total_refined_candidate_detection_count"],
                "total_unrefined_detection_count": summary["total_unrefined_detection_count"],
                "total_refined_detection_count": summary["total_refined_detection_count"],
            },
            "frames": frames,
        }

    @staticmethod
    def _normalize_frame_processing_status(status: str | JobStatus) -> str:
        value = status.value if isinstance(status, JobStatus) else status
        normalized = str(value).strip().lower()
        if normalized not in FRAME_PROCESSING_STATUSES:
            raise ValueError(
                f"frame processing status must be one of: {', '.join(sorted(FRAME_PROCESSING_STATUSES))}."
            )
        return normalized

    @staticmethod
    def _frame_status_next_cursor(rows: Sequence[dict[str, Any]], limit: int) -> str | None:
        if len(rows) < limit or not rows:
            return None
        last = rows[-1]
        return f"{last['asset_id']}|{last['frame_index']}|{last['frame_id']}"

    @staticmethod
    def _parse_frame_status_cursor(cursor: str | None) -> tuple[str, int, str] | None:
        if not cursor:
            return None
        parts = str(cursor).split("|")
        if len(parts) != 3:
            raise ValueError("cursor must have the form asset_id|frame_index|frame_id.")
        try:
            frame_index = int(parts[1])
        except ValueError as exc:
            raise ValueError("cursor frame_index must be an integer.") from exc
        return parts[0], frame_index, parts[2]

    def _frame_status_filters(
        self,
        *,
        project_id: str,
        run_id: str | None = None,
        asset_id: str | None = None,
        asset_ids: Sequence[str] | None = None,
        frame_ids: Sequence[str] | None = None,
        collection: str | None = None,
        collections: Sequence[str] | None = None,
        preprocessing_status: Sequence[str] | None = None,
        candidate_detection_status: Sequence[str] | None = None,
        roi_refinement_status: Sequence[str] | None = None,
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        cursor: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        clauses = ["status.project_id = %s"]
        params: list[Any] = [project_id]
        if run_id:
            clauses.append("status.run_id = %s")
            params.append(run_id)
        if asset_id:
            clauses.append("status.asset_id = %s")
            params.append(asset_id)
        if asset_ids:
            clauses.append("status.asset_id = ANY(%s::uuid[])")
            params.append(list(dict.fromkeys(str(value) for value in asset_ids if value)))
        if frame_ids:
            clauses.append("status.frame_id = ANY(%s::uuid[])")
            params.append(list(dict.fromkeys(str(value) for value in frame_ids if value)))
        if collection:
            clauses.append("status.collections @> ARRAY[%s]::text[]")
            params.append(collection)
        if collections:
            clauses.append("status.collections && %s::text[]")
            params.append(list(dict.fromkeys(str(value) for value in collections if value)))
        for column, values in (
            ("preprocessing_status", preprocessing_status),
            ("candidate_detection_status", candidate_detection_status),
            ("roi_refinement_status", roi_refinement_status),
        ):
            normalized = [self._normalize_frame_processing_status(value) for value in (values or []) if value]
            if normalized:
                placeholders = ", ".join(["%s" for _ in normalized])
                clauses.append(f"status.{column} IN ({placeholders})")
                params.extend(normalized)
        if has_candidates is not None:
            clauses.append("status.candidate_detection_count > 0" if has_candidates else "status.candidate_detection_count = 0")
        if has_refined_rois is not None:
            clauses.append("status.refined_detection_count > 0" if has_refined_rois else "status.refined_detection_count = 0")
        if start_frame is not None:
            clauses.append("status.frame_index >= %s")
            params.append(start_frame)
        if end_frame is not None:
            clauses.append("status.frame_index <= %s")
            params.append(end_frame)
        parsed_cursor = self._parse_frame_status_cursor(cursor)
        if parsed_cursor is not None:
            cursor_asset_id, cursor_frame_index, cursor_frame_id = parsed_cursor
            clauses.append("(status.asset_id, status.frame_index, status.frame_id) > (%s::uuid, %s, %s::uuid)")
            params.extend([cursor_asset_id, cursor_frame_index, cursor_frame_id])
        return clauses, params

    @staticmethod
    def _frame_status_summary_columns() -> str:
        columns = [
            "COUNT(*)::bigint AS total_frame_count",
            "COUNT(*) FILTER (WHERE preprocessing_status = 'succeeded')::bigint AS preprocessing_succeeded_count",
            "COUNT(*) FILTER (WHERE candidate_detection_status = 'succeeded')::bigint AS candidate_detection_succeeded_count",
            "COUNT(*) FILTER (WHERE roi_refinement_status = 'succeeded')::bigint AS roi_refinement_succeeded_count",
            "COUNT(*) FILTER (WHERE candidate_detection_count > 0)::bigint AS frames_with_candidates_count",
            "COUNT(*) FILTER (WHERE refined_detection_count > 0)::bigint AS frames_with_refined_rois_count",
            "COALESCE(SUM(candidate_detection_count), 0)::bigint AS candidate_detection_count",
            "COALESCE(SUM(refined_detection_count), 0)::bigint AS refined_detection_count",
            "COALESCE(SUM(unrefined_candidate_count), 0)::bigint AS unrefined_candidate_count",
            "MAX(updated_at) AS updated_at",
        ]
        for stage in ("preprocessing", "candidate_detection", "roi_refinement"):
            for status in FRAME_PROCESSING_STATUS_VALUES:
                columns.append(
                    f"COUNT(*) FILTER (WHERE {stage}_status = '{status}')::bigint "
                    f"AS {stage}_{status}_count"
                )
        return ",\n".join(columns)

    @staticmethod
    def _frame_status_summary_from_row(row: dict[str, Any] | None) -> dict[str, Any]:
        row = row or {}
        by_status: dict[str, dict[str, int]] = {}
        for stage in ("preprocessing", "candidate_detection", "roi_refinement"):
            counts = {
                status: int(row.get(f"{stage}_{status}_count") or 0)
                for status in FRAME_PROCESSING_STATUS_VALUES
            }
            by_status[stage] = {status: count for status, count in counts.items() if count > 0}
        return {
            "total_frame_count": row.get("total_frame_count", 0),
            "preprocessing_succeeded_count": row.get("preprocessing_succeeded_count", 0),
            "candidate_detection_succeeded_count": row.get("candidate_detection_succeeded_count", 0),
            "roi_refinement_succeeded_count": row.get("roi_refinement_succeeded_count", 0),
            "frames_with_candidates_count": row.get("frames_with_candidates_count", 0),
            "frames_with_refined_rois_count": row.get("frames_with_refined_rois_count", 0),
            "candidate_detection_count": row.get("candidate_detection_count", 0),
            "refined_detection_count": row.get("refined_detection_count", 0),
            "unrefined_candidate_count": row.get("unrefined_candidate_count", 0),
            "updated_at": row.get("updated_at"),
            "by_status": by_status,
        }

    def ensure_frame_status_rows(
        self,
        *,
        project_id: str,
        frame_ids: Sequence[str] | None = None,
        asset_id: str | None = None,
    ) -> int:
        resolved_project_id = self._required_project_id(project_id, "ensure_frame_status_rows")
        clauses = ["assets.project_id = %s"]
        params: list[Any] = [resolved_project_id]
        if frame_ids:
            clauses.append("frames.id = ANY(%s::uuid[])")
            params.append([str(frame_id) for frame_id in frame_ids])
        if asset_id:
            clauses.append("frames.asset_id = %s")
            params.append(asset_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.frame_processing_status
                        (project_id, frame_id, asset_id, run_id, frame_index, collections, updated_at)
                    SELECT
                        assets.project_id,
                        frames.id,
                        frames.asset_id,
                        frames.run_id,
                        frames.frame_index,
                        assets.collections,
                        NOW()
                    FROM {self.schema}.frames frames
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE {' AND '.join(clauses)}
                    ON CONFLICT (project_id, frame_id) DO UPDATE SET
                        asset_id = EXCLUDED.asset_id,
                        run_id = EXCLUDED.run_id,
                        frame_index = EXCLUDED.frame_index,
                        collections = EXCLUDED.collections,
                        updated_at = NOW()
                    """,
                    tuple(params),
                )
                count = cursor.rowcount
            connection.commit()
        return int(count or 0)

    def upsert_frame_stage_status(
        self,
        *,
        project_id: str,
        frame_ids: Sequence[str],
        stage: str,
        status: str,
        job_id: str | None = None,
        candidate_detection_count: int | None = None,
        refined_detection_count: int | None = None,
        unrefined_candidate_count: int | None = None,
        completed_at: datetime | None = None,
    ) -> int:
        if not frame_ids:
            return 0
        resolved_project_id = self._required_project_id(project_id, "upsert_frame_stage_status")
        normalized_status = self._normalize_frame_processing_status(status)
        stage_value = stage.value if isinstance(stage, PipelineStage) else str(stage)
        stage_map = {
            "preprocess_frames": ("preprocessing_status", "preprocessing_job_id", "preprocessing_completed_at"),
            "segment": ("candidate_detection_status", "candidate_detection_job_id", "candidate_detection_completed_at"),
            "roi_refinement": ("roi_refinement_status", "roi_refinement_job_id", "roi_refinement_completed_at"),
        }
        if stage_value not in stage_map:
            raise ValueError("stage must be one of: preprocess_frames, segment, roi_refinement.")
        status_column, job_column, completed_column = stage_map[stage_value]
        self.ensure_frame_status_rows(project_id=resolved_project_id, frame_ids=frame_ids)
        completed_value = completed_at
        if completed_value is None and normalized_status == JobStatus.SUCCEEDED.value:
            completed_value = datetime.now(timezone.utc)
        extra_assignments: list[str] = []
        params: list[Any] = [normalized_status, job_id, completed_value]
        if stage_value == "segment" and candidate_detection_count is not None:
            extra_assignments.append("candidate_detection_count = %s")
            params.append(max(0, int(candidate_detection_count)))
        if stage_value == "roi_refinement":
            if refined_detection_count is not None:
                extra_assignments.append("refined_detection_count = %s")
                params.append(max(0, int(refined_detection_count)))
            if unrefined_candidate_count is not None:
                extra_assignments.append("unrefined_candidate_count = %s")
                params.append(max(0, int(unrefined_candidate_count)))
        params.extend([resolved_project_id, [str(frame_id) for frame_id in frame_ids]])
        extra_sql = ", " + ", ".join(extra_assignments) if extra_assignments else ""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.frame_processing_status
                    SET
                        {status_column} = %s,
                        {job_column} = %s,
                        {completed_column} = %s,
                        updated_at = NOW()
                        {extra_sql}
                    WHERE project_id = %s
                      AND frame_id = ANY(%s::uuid[])
                    """,
                    tuple(params),
                )
                count = cursor.rowcount
            connection.commit()
        return int(count or 0)

    def refresh_frame_status_counts(
        self,
        *,
        project_id: str,
        frame_ids: Sequence[str] | None = None,
        asset_id: str | None = None,
    ) -> int:
        resolved_project_id = self._required_project_id(project_id, "refresh_frame_status_counts")
        clauses = ["assets.project_id = %s"]
        params: list[Any] = [resolved_project_id]
        if frame_ids:
            clauses.append("frames.id = ANY(%s::uuid[])")
            params.append([str(frame_id) for frame_id in frame_ids])
        if asset_id:
            clauses.append("frames.asset_id = %s")
            params.append(asset_id)
        self.ensure_frame_status_rows(project_id=resolved_project_id, frame_ids=frame_ids, asset_id=asset_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH candidate_counts AS (
                        SELECT detections.frame_id, COUNT(*)::integer AS candidate_detection_count
                        FROM {self.schema}.detection_candidate detections
                        JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                        JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                        WHERE {' AND '.join(clauses)}
                        GROUP BY detections.frame_id
                    ),
                    refined_counts AS (
                        SELECT
                            refined.frame_id,
                            COUNT(*)::integer AS refined_detection_count,
                            COUNT(DISTINCT refined.candidate_detection_id)::integer AS refined_candidate_detection_count
                        FROM {self.schema}.detections_refined refined
                        JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                        JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                        WHERE {' AND '.join(clauses)}
                        GROUP BY refined.frame_id
                    ),
                    selected_frames AS (
                        SELECT frames.id AS frame_id
                        FROM {self.schema}.frames frames
                        JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                        WHERE {' AND '.join(clauses)}
                    )
                    UPDATE {self.schema}.frame_processing_status status
                    SET
                        candidate_detection_count = COALESCE(candidate_counts.candidate_detection_count, 0),
                        refined_detection_count = COALESCE(refined_counts.refined_detection_count, 0),
                        unrefined_candidate_count = GREATEST(
                            COALESCE(candidate_counts.candidate_detection_count, 0)
                            - COALESCE(refined_counts.refined_candidate_detection_count, 0),
                            0
                        ),
                        updated_at = NOW()
                    FROM selected_frames
                    LEFT JOIN candidate_counts ON candidate_counts.frame_id = selected_frames.frame_id
                    LEFT JOIN refined_counts ON refined_counts.frame_id = selected_frames.frame_id
                    WHERE status.project_id = %s
                      AND status.frame_id = selected_frames.frame_id
                    """,
                    tuple(params + params + params + [resolved_project_id]),
                )
                count = cursor.rowcount
            connection.commit()
        return int(count or 0)

    def rebuild_frame_status(
        self,
        *,
        project_id: str,
        asset_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_project_id = self._required_project_id(project_id, "rebuild_frame_status")
        clauses = ["assets.project_id = %s"]
        params: list[Any] = [resolved_project_id]
        if asset_id:
            clauses.append("frames.asset_id = %s")
            params.append(asset_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH candidate_counts AS (
                        SELECT detections.frame_id, COUNT(*)::integer AS candidate_detection_count
                        FROM {self.schema}.detection_candidate detections
                        JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
                        JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                        WHERE {' AND '.join(clauses)}
                        GROUP BY detections.frame_id
                    ),
                    refined_counts AS (
                        SELECT
                            refined.frame_id,
                            COUNT(*)::integer AS refined_detection_count,
                            COUNT(DISTINCT refined.candidate_detection_id)::integer AS refined_candidate_detection_count
                        FROM {self.schema}.detections_refined refined
                        JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                        JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                        WHERE {' AND '.join(clauses)}
                        GROUP BY refined.frame_id
                    ),
                    source_rows AS (
                        SELECT
                            assets.project_id,
                            frames.id AS frame_id,
                            frames.asset_id,
                            frames.run_id,
                            frames.frame_index,
                            assets.collections,
                            CASE
                                WHEN frames.preprocessed_payload_ref IS NOT NULL
                                  OR frames.preprocessed_kvstore_hash IS NOT NULL
                                THEN 'succeeded'
                                ELSE COALESCE(existing.preprocessing_status, 'unknown')
                            END AS preprocessing_status,
                            CASE
                                WHEN COALESCE(candidate_counts.candidate_detection_count, 0) > 0
                                THEN 'succeeded'
                                ELSE COALESCE(existing.candidate_detection_status, 'unknown')
                            END AS candidate_detection_status,
                            COALESCE(candidate_counts.candidate_detection_count, 0) AS candidate_detection_count,
                            CASE
                                WHEN COALESCE(refined_counts.refined_detection_count, 0) > 0
                                THEN 'succeeded'
                                ELSE COALESCE(existing.roi_refinement_status, 'unknown')
                            END AS roi_refinement_status,
                            COALESCE(refined_counts.refined_detection_count, 0) AS refined_detection_count,
                            GREATEST(
                                COALESCE(candidate_counts.candidate_detection_count, 0)
                                - COALESCE(refined_counts.refined_candidate_detection_count, 0),
                                0
                            ) AS unrefined_candidate_count
                        FROM {self.schema}.frames frames
                        JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                        LEFT JOIN candidate_counts ON candidate_counts.frame_id = frames.id
                        LEFT JOIN refined_counts ON refined_counts.frame_id = frames.id
                        LEFT JOIN {self.schema}.frame_processing_status existing
                          ON existing.project_id = assets.project_id
                         AND existing.frame_id = frames.id
                        WHERE {' AND '.join(clauses)}
                    ),
                    upserted AS (
                        INSERT INTO {self.schema}.frame_processing_status
                            (
                                project_id, frame_id, asset_id, run_id, frame_index, collections,
                                preprocessing_status, candidate_detection_status, candidate_detection_count,
                                roi_refinement_status, refined_detection_count, unrefined_candidate_count,
                                updated_at
                            )
                        SELECT
                            project_id, frame_id, asset_id, run_id, frame_index, collections,
                            preprocessing_status, candidate_detection_status, candidate_detection_count,
                            roi_refinement_status, refined_detection_count, unrefined_candidate_count,
                            NOW()
                        FROM source_rows
                        ON CONFLICT (project_id, frame_id) DO UPDATE SET
                            asset_id = EXCLUDED.asset_id,
                            run_id = EXCLUDED.run_id,
                            frame_index = EXCLUDED.frame_index,
                            collections = EXCLUDED.collections,
                            preprocessing_status = EXCLUDED.preprocessing_status,
                            candidate_detection_status = EXCLUDED.candidate_detection_status,
                            candidate_detection_count = EXCLUDED.candidate_detection_count,
                            roi_refinement_status = EXCLUDED.roi_refinement_status,
                            refined_detection_count = EXCLUDED.refined_detection_count,
                            unrefined_candidate_count = EXCLUDED.unrefined_candidate_count,
                            updated_at = NOW()
                        RETURNING frame_id
                    )
                    SELECT COUNT(*)::bigint AS rebuilt_frame_count FROM upserted
                    """,
                    tuple(params + params + params),
                )
                row = cursor.fetchone()
            connection.commit()
        summary = self.get_frame_status_summary(project_id=resolved_project_id, asset_id=asset_id)
        return {
            "rebuilt_frame_count": 0 if row is None else row["rebuilt_frame_count"],
            "summary": summary,
        }

    def touch_processing_status_snapshot(
        self,
        *,
        project_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_project_id = self._required_project_id(project_id, "touch_processing_status_snapshot")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if session_id is None:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.project_processing_status_snapshots
                        SET status_version = status_version + 1,
                            updated_at = NOW()
                        WHERE project_id = %s
                          AND session_id IS NULL
                        RETURNING *
                        """,
                        (resolved_project_id,),
                    )
                else:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.project_processing_status_snapshots
                        SET status_version = status_version + 1,
                            updated_at = NOW()
                        WHERE project_id = %s
                          AND session_id = %s
                        RETURNING *
                        """,
                        (resolved_project_id, session_id),
                    )
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.project_processing_status_snapshots
                            (project_id, session_id, status_version, updated_at, summary)
                        VALUES (%s, %s, 1, NOW(), '{{}}'::jsonb)
                        RETURNING *
                        """,
                        (resolved_project_id, session_id),
                    )
                    row = cursor.fetchone()
            connection.commit()
        return row

    def list_frame_status(
        self,
        *,
        project_id: str,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
        preprocessing_status: Sequence[str] | None = None,
        candidate_detection_status: Sequence[str] | None = None,
        roi_refinement_status: Sequence[str] | None = None,
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int = 1000,
        cursor: str | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        resolved_project_id = self._required_project_id(project_id, "list_frame_status")
        bounded_limit = min(max(1, int(limit)), 10000)
        clauses, params = self._frame_status_filters(
            project_id=resolved_project_id,
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
            cursor=cursor,
        )
        offset_sql = "" if cursor else "OFFSET %s"
        query_params = tuple(params + [bounded_limit] + ([] if cursor else [max(0, int(offset))]))
        with self.connect() as connection:
            with connection.cursor() as cursor_obj:
                cursor_obj.execute(
                    f"""
                    SELECT
                        status.*,
                        assets.filename AS asset_filename,
                        assets.kind AS asset_kind
                    FROM {self.schema}.frame_processing_status status
                    JOIN {self.schema}.raw_assets assets ON assets.id = status.asset_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY status.asset_id ASC, status.frame_index ASC, status.frame_id ASC
                    LIMIT %s
                    {offset_sql}
                    """,
                    query_params,
                )
                rows = cursor_obj.fetchall()
        return {
            "frames": rows,
            "next_cursor": self._frame_status_next_cursor(rows, bounded_limit),
        }

    def list_frame_status_ids(
        self,
        *,
        project_id: str,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
        preprocessing_status: Sequence[str] | None = None,
        candidate_detection_status: Sequence[str] | None = None,
        roi_refinement_status: Sequence[str] | None = None,
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        limit: int = 5000,
        cursor: str | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        result = self.list_frame_status(
            project_id=project_id,
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
            limit=limit,
            cursor=cursor,
            offset=offset,
        )
        return {
            "frame_ids": [str(row["frame_id"]) for row in result["frames"]],
            "next_cursor": result["next_cursor"],
        }

    def get_frame_status_summary(
        self,
        *,
        project_id: str,
        run_id: str | None = None,
        asset_id: str | None = None,
        collection: str | None = None,
        preprocessing_status: Sequence[str] | None = None,
        candidate_detection_status: Sequence[str] | None = None,
        roi_refinement_status: Sequence[str] | None = None,
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> dict[str, Any]:
        resolved_project_id = self._required_project_id(project_id, "get_frame_status_summary")
        clauses, params = self._frame_status_filters(
            project_id=resolved_project_id,
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
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {self._frame_status_summary_columns()}
                    FROM {self.schema}.frame_processing_status status
                    WHERE {' AND '.join(clauses)}
                    """,
                    tuple(params),
                )
                row = cursor.fetchone()
        return self._frame_status_summary_from_row(row)

    def get_frame_status_facets(
        self,
        *,
        project_id: str,
        run_id: str | None = None,
        asset_ids: Sequence[str] | None = None,
        collections: Sequence[str] | None = None,
        preprocessing_status: Sequence[str] | None = None,
        candidate_detection_status: Sequence[str] | None = None,
        roi_refinement_status: Sequence[str] | None = None,
        has_candidates: bool | None = None,
        has_refined_rois: bool | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> dict[str, Any]:
        resolved_project_id = self._required_project_id(project_id, "get_frame_status_facets")
        clauses, params = self._frame_status_filters(
            project_id=resolved_project_id,
            run_id=run_id,
            asset_ids=asset_ids,
            collections=collections,
            preprocessing_status=preprocessing_status,
            candidate_detection_status=candidate_detection_status,
            roi_refinement_status=roi_refinement_status,
            has_candidates=has_candidates,
            has_refined_rois=has_refined_rois,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH filtered AS MATERIALIZED (
                        SELECT
                            asset_id,
                            collections,
                            preprocessing_status,
                            candidate_detection_status,
                            roi_refinement_status,
                            candidate_detection_count,
                            refined_detection_count,
                            unrefined_candidate_count,
                            updated_at
                        FROM {self.schema}.frame_processing_status status
                        WHERE {' AND '.join(clauses)}
                    ),
                    summary AS (
                        SELECT {self._frame_status_summary_columns()}
                        FROM filtered status
                    ),
                    asset_counts AS (
                        SELECT asset_id::text AS value, COUNT(*)::bigint AS unit_count
                        FROM filtered
                        GROUP BY asset_id
                    ),
                    collection_counts AS (
                        SELECT collection AS value, COUNT(*)::bigint AS unit_count
                        FROM filtered
                        CROSS JOIN LATERAL unnest(collections) AS collection
                        GROUP BY collection
                    )
                    SELECT
                        summary.*,
                        COALESCE(
                            (SELECT jsonb_object_agg(value, unit_count) FROM asset_counts),
                            '{{}}'::jsonb
                        ) AS asset_facets,
                        COALESCE(
                            (SELECT jsonb_object_agg(value, unit_count) FROM collection_counts),
                            '{{}}'::jsonb
                        ) AS collection_facets
                    FROM summary
                    """,
                    tuple(params),
                )
                row = cursor.fetchone() or {}
        summary = self._frame_status_summary_from_row(row)
        return {
            "summary": summary,
            "facets": {
                "assets": row.get("asset_facets") or {},
                "collections": row.get("collection_facets") or {},
                "preprocessing_status": summary["by_status"]["preprocessing"],
                "candidate_detection_status": summary["by_status"]["candidate_detection"],
                "roi_refinement_status": summary["by_status"]["roi_refinement"],
                "refinement_state": {
                    "refined": summary["refined_detection_count"],
                    "unrefined": summary["unrefined_candidate_count"],
                },
            },
        }

    def get_processing_status_snapshot(
        self,
        *,
        project_id: str,
    ) -> dict[str, Any] | None:
        resolved_project_id = self._required_project_id(project_id, "get_processing_status_snapshot")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM {self.schema}.project_processing_status_snapshots
                    WHERE project_id = %s
                      AND session_id IS NULL
                    """,
                    (resolved_project_id,),
                )
                return cursor.fetchone()

    def get_or_create_processing_status_snapshot(
        self,
        *,
        project_id: str,
        session_id: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_project_id = self._required_project_id(project_id, "get_or_create_processing_status_snapshot")
        summary_payload = json.dumps(json_ready(summary or {}))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if session_id is None:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.project_processing_status_snapshots
                        SET
                            status_version = CASE
                                WHEN summary IS DISTINCT FROM %s::jsonb
                                THEN status_version + 1
                                ELSE status_version
                            END,
                            generated_at = NOW(),
                            updated_at = NOW(),
                            summary = %s::jsonb
                        WHERE project_id = %s
                          AND session_id IS NULL
                        RETURNING *
                        """,
                        (summary_payload, summary_payload, resolved_project_id),
                    )
                else:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.project_processing_status_snapshots
                        SET
                            status_version = CASE
                                WHEN summary IS DISTINCT FROM %s::jsonb
                                THEN status_version + 1
                                ELSE status_version
                            END,
                            generated_at = NOW(),
                            updated_at = NOW(),
                            summary = %s::jsonb
                        WHERE project_id = %s
                          AND session_id = %s
                        RETURNING *
                        """,
                        (summary_payload, summary_payload, resolved_project_id, session_id),
                    )
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.project_processing_status_snapshots
                            (project_id, session_id, status_version, generated_at, updated_at, summary)
                        VALUES (%s, %s, 1, NOW(), NOW(), %s::jsonb)
                        RETURNING *
                        """,
                        (resolved_project_id, session_id, summary_payload),
                    )
                    row = cursor.fetchone()
            connection.commit()
        return row

    def register_model(
        self,
        model: ModelRecord,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_project_id = self._required_project_id(
            project_id or model.metadata.get("project_id"),
            "register_model",
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.models
                    (project_id, model_key, model_name, version, task, artifact_uri, labels, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (model_key) DO UPDATE SET
                        project_id = EXCLUDED.project_id,
                        model_name = EXCLUDED.model_name,
                        version = EXCLUDED.version,
                        task = EXCLUDED.task,
                        artifact_uri = EXCLUDED.artifact_uri,
                        labels = EXCLUDED.labels,
                        metadata = EXCLUDED.metadata
                    RETURNING *;
                    """,
                    (
                        resolved_project_id,
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
        project_id: str | None = None,
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
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
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

    def get_model(self, model_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["id = %s"]
                params: list[Any] = [model_id]
                if project_id:
                    clauses.append("project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"SELECT * FROM {self.schema}.models WHERE {' AND '.join(clauses)}",
                    tuple(params),
                )
                return cursor.fetchone()

    def get_model_by_key(self, model_key: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["model_key = %s"]
                params: list[Any] = [model_key]
                if project_id:
                    clauses.append("project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"SELECT * FROM {self.schema}.models WHERE {' AND '.join(clauses)}",
                    tuple(params),
                )
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

    def list_curation_labels(
        self,
        *,
        project_id: str,
        include_deprecated: bool = False,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH latest_evidence AS (
                        SELECT DISTINCT ON (refined_detection_id)
                               refined_detection_id, predicted_label_id
                        FROM {self.schema}.classification_evidence
                        ORDER BY refined_detection_id, created_at DESC, id DESC
                    )
                    SELECT labels.*,
                           count(DISTINCT annotations.id) FILTER (
                               WHERE annotations.is_current AND annotations.status <> 'deprecated'
                           ) AS annotation_count,
                           count(DISTINCT evidence.refined_detection_id) AS prediction_count
                    FROM {self.schema}.classification_labels labels
                    LEFT JOIN {self.schema}.roi_label_annotations annotations
                      ON annotations.label_id = labels.id
                    LEFT JOIN latest_evidence evidence
                      ON evidence.predicted_label_id = labels.id
                    WHERE labels.project_id = %s
                      AND (%s OR labels.deprecated_at IS NULL)
                    GROUP BY labels.id
                    ORDER BY labels.deprecated_at NULLS FIRST,
                             coalesce(labels.display_name, labels.name)
                    """,
                    (project_id, include_deprecated),
                )
                return list(cursor.fetchall())

    def create_curation_label(
        self,
        *,
        project_id: str,
        name: str,
        display_name: str | None = None,
        stable_concept_id: str | None = None,
        parent_label_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if parent_label_id is not None:
                    cursor.execute(
                        f"SELECT id FROM {self.schema}.classification_labels WHERE id = %s AND project_id = %s",
                        (parent_label_id, project_id),
                    )
                    if cursor.fetchone() is None:
                        raise KeyError(parent_label_id)
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.classification_labels
                        (project_id, name, display_name, stable_concept_id,
                         parent_label_id, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING *
                    """,
                    (
                        project_id,
                        name.strip(),
                        display_name,
                        stable_concept_id,
                        parent_label_id,
                        json.dumps(json_ready(metadata or {})),
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def import_curation_label_dictionary(
        self,
        *,
        project_id: str,
        dictionary: dict[str, Any],
    ) -> dict[str, Any]:
        """Materialize selectable taxonomy concepts as idempotent project labels."""

        vocabulary = dictionary.get("vocabulary") or {}
        nodes = dictionary.get("labels") or []
        nodes_by_id = {str(node["id"]): node for node in nodes}
        selectable_nodes = [node for node in nodes if bool(node.get("selectable", True))]
        created_count = 0
        updated_count = 0
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.schema}.classification_labels WHERE project_id = %s FOR UPDATE",
                    (project_id,),
                )
                existing_rows = list(cursor.fetchall())
                by_concept = {
                    str(row["stable_concept_id"]): row
                    for row in existing_rows
                    if row.get("stable_concept_id")
                }
                by_name = {str(row["name"]).casefold(): row for row in existing_rows}
                materialized: dict[str, str] = {
                    concept_id: str(row["id"]) for concept_id, row in by_concept.items()
                }

                def parent_label_id(node: dict[str, Any]) -> str | None:
                    parent_id = node.get("parent_id")
                    while parent_id:
                        resolved = materialized.get(str(parent_id))
                        if resolved:
                            return resolved
                        parent = nodes_by_id.get(str(parent_id)) or {}
                        parent_id = parent.get("parent_id")
                    return None

                for node in selectable_nodes:
                    concept_id = str(node["id"])
                    name = str(node["name"])
                    display_name = str(node.get("display_name") or name)
                    metadata = {
                        "label_dictionary": {
                            "key": dictionary.get("key"),
                            "filename": dictionary.get("filename"),
                            "vocabulary": vocabulary,
                            "concept": node,
                        }
                    }
                    existing = by_concept.get(concept_id) or by_name.get(name.casefold())
                    if existing is None:
                        cursor.execute(
                            f"""
                            INSERT INTO {self.schema}.classification_labels
                                (project_id, name, display_name, stable_concept_id,
                                 parent_label_id, rank, description, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                            RETURNING *
                            """,
                            (
                                project_id, name, display_name, concept_id,
                                parent_label_id(node), node.get("rank"),
                                node.get("description"), json.dumps(json_ready(metadata)),
                            ),
                        )
                        row = cursor.fetchone()
                        created_count += 1
                    else:
                        cursor.execute(
                            f"""
                            UPDATE {self.schema}.classification_labels
                            SET display_name = %s, stable_concept_id = %s,
                                parent_label_id = %s, rank = %s, description = %s,
                                metadata = metadata || %s::jsonb, deprecated_at = NULL
                            WHERE id = %s
                            RETURNING *
                            """,
                            (
                                display_name, concept_id, parent_label_id(node),
                                node.get("rank"), node.get("description"),
                                json.dumps(json_ready(metadata)), existing["id"],
                            ),
                        )
                        row = cursor.fetchone()
                        updated_count += 1
                    by_concept[concept_id] = row
                    by_name[name.casefold()] = row
                    materialized[concept_id] = str(row["id"])
            connection.commit()
        return {
            "dictionary_key": dictionary.get("key"),
            "created_count": created_count,
            "updated_count": updated_count,
            "labels": self.list_curation_labels(project_id=project_id),
        }

    def list_curation_rois(
        self,
        *,
        project_id: str,
        annotation_state: str = "all",
        review_state: str = "all",
        label_id: str | None = None,
        label_source: str = "any",
        evidence_state: str = "all",
        search: str | None = None,
        telemetry_filters: Sequence[Mapping[str, Any]] | None = None,
        sort_by: str = "oldest",
        limit: int = 120,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses = ["assets.project_id = %s"]
        params: list[Any] = [project_id]
        if annotation_state == "labeled":
            clauses.append("annotation.id IS NOT NULL AND annotation.status <> 'deprecated'")
        elif annotation_state == "unlabeled":
            clauses.append("(annotation.id IS NULL OR annotation.status = 'deprecated')")
        if review_state == "unreviewed":
            clauses.append("annotation.id IS NOT NULL AND review.id IS NULL")
        elif review_state in {"verified", "rejected", "needs_review"}:
            clauses.append("review.decision = %s")
            params.append(review_state)
        if label_id:
            if label_source == "human":
                clauses.append("annotation.label_id = %s")
                params.append(label_id)
            elif label_source == "prediction":
                clauses.append("evidence.predicted_label_id = %s")
                params.append(label_id)
            else:
                clauses.append("(annotation.label_id = %s OR evidence.predicted_label_id = %s)")
                params.extend([label_id, label_id])
        if evidence_state == "available":
            clauses.append("evidence.id IS NOT NULL")
        elif evidence_state == "missing":
            clauses.append("evidence.id IS NULL")
        elif evidence_state == "disagreement":
            clauses.append(
                "evidence.id IS NOT NULL AND ((evidence.prototype_class_index IS NOT NULL AND "
                "evidence.prototype_class_index <> evidence.predicted_class_index) OR "
                "(evidence.knn_class_index IS NOT NULL AND "
                "evidence.knn_class_index <> evidence.predicted_class_index))"
            )
        if search:
            clauses.append(
                "(refined.id::text ILIKE %s OR frames.id::text ILIKE %s "
                "OR assets.filename ILIKE %s)"
            )
            token = f"%{search}%"
            params.extend([token, token, token])
        telemetry_clauses, telemetry_params = self._telemetry_filter_clauses(
            telemetry_filters,
            timestamp_sql="frames.captured_at",
            run_sql="refined.run_id",
            project_sql="assets.project_id",
        )
        clauses.extend(telemetry_clauses)
        params.extend(telemetry_params)
        order = {
            "oldest": "refined.created_at ASC, refined.id ASC",
            "newest": "refined.created_at DESC, refined.id DESC",
            "area_asc": "refined.area ASC NULLS LAST, refined.id ASC",
            "area_desc": "refined.area DESC NULLS LAST, refined.id ASC",
            "confidence_asc": "evidence.confidence ASC NULLS FIRST, refined.id ASC",
            "confidence_desc": "evidence.confidence DESC NULLS LAST, refined.id ASC",
            "disagreement": "(coalesce(evidence.prototype_class_index <> evidence.predicted_class_index, false)::int + coalesce(evidence.knn_class_index <> evidence.predicted_class_index, false)::int) DESC, refined.id ASC",
        }.get(sort_by, "refined.created_at ASC, refined.id ASC")
        where = " AND ".join(clauses)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT refined.id, refined.run_id, refined.frame_id,
                           refined.candidate_detection_id, refined.roi_index,
                           refined.bbox_x, refined.bbox_y, refined.bbox_w, refined.bbox_h,
                           refined.crop_bbox_x, refined.crop_bbox_y,
                           refined.crop_bbox_w, refined.crop_bbox_h,
                           refined.area, refined.perimeter, refined.roi_shape,
                           refined.roi_encoding, refined.created_at,
                           frames.frame_index, frames.captured_at AS captured_at,
                           assets.id AS asset_id,
                           assets.filename AS asset_filename,
                           annotation.id AS annotation_id,
                           annotation.label_id, annotation.status AS annotation_status,
                           annotation.actor_username, annotation.created_at AS annotation_created_at,
                           label.name AS label_name,
                           coalesce(label.display_name, label.name) AS label_display_name,
                           review.decision AS review_decision,
                           evidence.id AS evidence_id,
                           evidence.predicted_label_id, evidence.predicted_label_name,
                           evidence.predicted_class_index, evidence.confidence,
                           evidence.entropy, evidence.probability_margin,
                           evidence.prototype_class_index, evidence.prototype_similarity,
                           evidence.prototype_margin, evidence.knn_class_index,
                           evidence.knn_agreement, evidence.knn_weighted_support,
                           evidence.knn_margin, evidence.inference_run_id,
                           cluster_evidence.id AS clustering_evidence_id,
                           cluster_evidence.cluster_index,
                           cluster_evidence.cluster_id,
                           cluster_evidence.similarity AS cluster_similarity,
                           cluster_evidence.novel AS cluster_novel,
                           cluster_evidence.abstained AS cluster_abstained,
                           cluster_evidence.inference_run_id AS clustering_inference_run_id,
                           count(*) OVER() AS total_count
                    FROM {self.schema}.detections_refined refined
                    JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.roi_label_annotations current_annotation
                        WHERE current_annotation.refined_detection_id = refined.id
                          AND current_annotation.is_current
                        ORDER BY current_annotation.created_at DESC, current_annotation.id DESC
                        LIMIT 1
                    ) annotation ON true
                    LEFT JOIN {self.schema}.classification_labels label
                      ON label.id = annotation.label_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.roi_annotation_reviews latest_review
                        WHERE latest_review.annotation_id = annotation.id
                        ORDER BY latest_review.created_at DESC, latest_review.id DESC
                        LIMIT 1
                    ) review ON true
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.classification_evidence latest_evidence
                        WHERE latest_evidence.refined_detection_id = refined.id
                        ORDER BY latest_evidence.created_at DESC, latest_evidence.id DESC
                        LIMIT 1
                    ) evidence ON true
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.clustering_evidence latest_cluster_evidence
                        WHERE latest_cluster_evidence.refined_detection_id = refined.id
                        ORDER BY latest_cluster_evidence.created_at DESC, latest_cluster_evidence.id DESC
                        LIMIT 1
                    ) cluster_evidence ON true
                    WHERE {where}
                    ORDER BY {order}
                    LIMIT %s OFFSET %s
                    """,
                    (*params, limit, offset),
                )
                rows = list(cursor.fetchall())
        return {
            "items": rows,
            "total": int(rows[0]["total_count"]) if rows else 0,
            "limit": limit,
            "offset": offset,
        }

    def get_curation_roi(self, roi_id: str, *, project_id: str) -> dict[str, Any] | None:
        page = self.list_curation_rois(project_id=project_id, search=roi_id, limit=10)
        row = next((item for item in page["items"] if str(item["id"]) == roi_id), None)
        if row is None:
            return None
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT annotations.*, labels.name AS label_name,
                           coalesce(labels.display_name, labels.name) AS label_display_name
                    FROM {self.schema}.roi_label_annotations annotations
                    JOIN {self.schema}.classification_labels labels ON labels.id = annotations.label_id
                    WHERE annotations.refined_detection_id = %s
                    ORDER BY annotations.created_at DESC, annotations.id DESC
                    """,
                    (roi_id,),
                )
                row["annotations"] = list(cursor.fetchall())
                cursor.execute(
                    f"""
                    SELECT reviews.*
                    FROM {self.schema}.roi_annotation_reviews reviews
                    JOIN {self.schema}.roi_label_annotations annotations
                      ON annotations.id = reviews.annotation_id
                    WHERE annotations.refined_detection_id = %s
                    ORDER BY reviews.created_at DESC, reviews.id DESC
                    """,
                    (roi_id,),
                )
                row["reviews"] = list(cursor.fetchall())
                cursor.execute(
                    f"""
                    SELECT evidence.*, runs.model_selector, artifacts.artifact_id,
                           artifacts.run_id AS model_run_id,
                           artifacts.artifact_fingerprint,
                           coalesce(json_agg(neighbors ORDER BY neighbors.rank)
                               FILTER (WHERE neighbors.evidence_id IS NOT NULL), '[]'::json) AS neighbors
                    FROM {self.schema}.classification_evidence evidence
                    JOIN {self.schema}.classification_inference_runs runs
                      ON runs.id = evidence.inference_run_id
                    LEFT JOIN {self.schema}.model_artifacts artifacts
                      ON artifacts.id = runs.model_artifact_id
                    LEFT JOIN {self.schema}.classification_evidence_neighbors neighbors
                      ON neighbors.evidence_id = evidence.id
                    WHERE evidence.refined_detection_id = %s
                    GROUP BY evidence.id, runs.model_selector, artifacts.artifact_id,
                             artifacts.run_id, artifacts.artifact_fingerprint
                    ORDER BY evidence.created_at DESC, evidence.id DESC
                """,
                    (roi_id,),
                )
                row["evidence"] = list(cursor.fetchall())
                cursor.execute(
                    f"""
                    SELECT evidence.*, runs.model_selector,
                           artifacts.artifact_id,
                           artifacts.run_id AS model_run_id,
                           artifacts.artifact_fingerprint
                    FROM {self.schema}.clustering_evidence evidence
                    JOIN {self.schema}.classification_inference_runs runs
                      ON runs.id = evidence.inference_run_id
                    LEFT JOIN {self.schema}.model_artifacts artifacts
                      ON artifacts.id = evidence.model_artifact_id
                    WHERE evidence.refined_detection_id = %s
                    ORDER BY evidence.created_at DESC, evidence.id DESC
                    """,
                    (roi_id,),
                )
                row["clustering_evidence"] = list(cursor.fetchall())
        return row

    def list_feature_space_sources(self, *, project_id: str) -> list[dict[str, Any]]:
        """List persisted embedding spaces without treating them as interchangeable.

        Every row is tied to one inference run and model artifact.  Callers must
        select one source before comparing vectors.
        """

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT 'classification' AS source_kind,
                           runs.id AS inference_run_id,
                           runs.model_selector,
                           artifacts.artifact_id,
                           artifacts.run_id AS model_run_id,
                           artifacts.artifact_fingerprint,
                           count(*) AS evidence_count,
                           count(evidence.embedding_payload_ref) AS embedding_count,
                           (jsonb_agg(evidence.embedding_shape)
                              FILTER (WHERE evidence.embedding_shape IS NOT NULL))->0
                              AS embedding_shape,
                           max(evidence.created_at) AS latest_evidence_at
                    FROM {self.schema}.classification_inference_runs runs
                    JOIN {self.schema}.classification_evidence evidence
                      ON evidence.inference_run_id = runs.id
                    LEFT JOIN {self.schema}.model_artifacts artifacts
                      ON artifacts.id = runs.model_artifact_id
                    WHERE runs.project_id = %s
                    GROUP BY runs.id, runs.model_selector, artifacts.artifact_id,
                             artifacts.run_id, artifacts.artifact_fingerprint
                    HAVING count(evidence.embedding_payload_ref) > 0
                    UNION ALL
                    SELECT 'clustering' AS source_kind,
                           runs.id AS inference_run_id,
                           runs.model_selector,
                           artifacts.artifact_id,
                           artifacts.run_id AS model_run_id,
                           artifacts.artifact_fingerprint,
                           count(*) AS evidence_count,
                           count(evidence.embedding_payload_ref) AS embedding_count,
                           (jsonb_agg(evidence.embedding_shape)
                              FILTER (WHERE evidence.embedding_shape IS NOT NULL))->0
                              AS embedding_shape,
                           max(evidence.created_at) AS latest_evidence_at
                    FROM {self.schema}.classification_inference_runs runs
                    JOIN {self.schema}.clustering_evidence evidence
                      ON evidence.inference_run_id = runs.id
                    LEFT JOIN {self.schema}.model_artifacts artifacts
                      ON artifacts.id = evidence.model_artifact_id
                    WHERE runs.project_id = %s
                    GROUP BY runs.id, runs.model_selector, artifacts.artifact_id,
                             artifacts.run_id, artifacts.artifact_fingerprint
                    HAVING count(evidence.embedding_payload_ref) > 0
                    ORDER BY latest_evidence_at DESC, source_kind, model_selector
                    """,
                    (project_id, project_id),
                )
                return list(cursor.fetchall())

    def list_feature_space_embeddings(
        self,
        *,
        project_id: str,
        source_kind: str,
        inference_run_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return embedding references for one immutable evidence source."""

        if source_kind not in {"classification", "clustering"}:
            raise ValueError(f"Unsupported feature-space source kind: {source_kind}")
        table = "classification_evidence" if source_kind == "classification" else "clustering_evidence"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT evidence.refined_detection_id,
                           evidence.embedding_payload_ref,
                           evidence.embedding_dtype,
                           evidence.embedding_shape,
                           evidence.embedding_sha256
                    FROM {self.schema}.{table} evidence
                    JOIN {self.schema}.classification_inference_runs runs
                      ON runs.id = evidence.inference_run_id
                    WHERE evidence.project_id = %s
                      AND evidence.inference_run_id = %s
                      AND runs.project_id = %s
                      AND evidence.embedding_payload_ref IS NOT NULL
                    ORDER BY evidence.refined_detection_id
                    LIMIT %s
                    """,
                    (project_id, inference_run_id, project_id, limit),
                )
                return list(cursor.fetchall())

    def count_feature_space_embeddings(
        self,
        *,
        project_id: str,
        source_kind: str,
        inference_run_id: str,
    ) -> int:
        """Count persisted ROI vectors for one immutable feature-space source."""

        if source_kind not in {"classification", "clustering"}:
            raise ValueError(f"Unsupported feature-space source kind: {source_kind}")
        table = "classification_evidence" if source_kind == "classification" else "clustering_evidence"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT count(*) AS total
                    FROM {self.schema}.{table} evidence
                    JOIN {self.schema}.classification_inference_runs runs
                      ON runs.id = evidence.inference_run_id
                    WHERE evidence.project_id = %s
                      AND evidence.inference_run_id = %s
                      AND runs.project_id = %s
                      AND evidence.embedding_payload_ref IS NOT NULL
                    """,
                    (project_id, inference_run_id, project_id),
                )
                row = cursor.fetchone()
        return int(row["total"] if row else 0)

    def list_feature_space_roi_summaries(
        self, *, project_id: str, roi_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Return compact, project-scoped ROI cards for feature-space results."""

        resolved_ids = list(dict.fromkeys(str(value) for value in roi_ids if value))
        if not resolved_ids:
            return []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT refined.id, refined.frame_id, refined.roi_index, refined.area,
                           refined.roi_shape, refined.created_at,
                           assets.id AS asset_id, assets.filename AS asset_filename,
                           coalesce(labels.display_name, labels.name) AS label_display_name,
                           review.decision AS review_decision
                    FROM {self.schema}.detections_refined refined
                    JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.roi_label_annotations annotation
                        WHERE annotation.refined_detection_id = refined.id
                          AND annotation.is_current
                        ORDER BY annotation.created_at DESC, annotation.id DESC
                        LIMIT 1
                    ) annotation ON true
                    LEFT JOIN {self.schema}.classification_labels labels
                      ON labels.id = annotation.label_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.roi_annotation_reviews latest_review
                        WHERE latest_review.annotation_id = annotation.id
                        ORDER BY latest_review.created_at DESC, latest_review.id DESC
                        LIMIT 1
                    ) review ON true
                    WHERE assets.project_id = %s
                      AND refined.id = ANY(%s::uuid[])
                    """,
                    (project_id, resolved_ids),
                )
                return list(cursor.fetchall())

    def list_feature_space_clusters(
        self, *, project_id: str, inference_run_id: str
    ) -> list[dict[str, Any]]:
        """Summarize run-local clusters with their strongest Pelagia ROI card."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH assigned AS (
                        SELECT *
                        FROM {self.schema}.clustering_evidence
                        WHERE project_id = %s
                          AND inference_run_id = %s
                          AND cluster_id IS NOT NULL
                    ), counts AS (
                        SELECT cluster_id, min(cluster_index) AS cluster_index,
                               count(*) AS roi_count,
                               avg(similarity) AS mean_similarity,
                               min(similarity) AS min_similarity,
                               max(similarity) AS max_similarity,
                               count(*) FILTER (WHERE novel OR abstained) AS novelty_count
                        FROM assigned
                        GROUP BY cluster_id
                    ), representatives AS (
                        SELECT DISTINCT ON (cluster_id)
                               cluster_id, refined_detection_id AS representative_detection_id,
                               similarity AS representative_similarity
                        FROM assigned
                        ORDER BY cluster_id, similarity DESC NULLS LAST, refined_detection_id
                    )
                    SELECT counts.*, representatives.representative_detection_id,
                           representatives.representative_similarity
                    FROM counts
                    JOIN representatives USING (cluster_id)
                    ORDER BY counts.roi_count DESC, counts.cluster_id
                    """,
                    (project_id, inference_run_id),
                )
                return list(cursor.fetchall())

    def list_feature_space_label_prototypes(
        self, *, project_id: str, inference_run_id: str
    ) -> list[dict[str, Any]]:
        """Summarize a classification run by its recorded label prototypes.

        These are model-provided prototype assignments, not self-supervised
        clusters and not human taxonomy assertions.  They remain scoped to the
        inference run and artifact that generated the classification evidence.
        """

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH assigned AS (
                        SELECT evidence.refined_detection_id,
                               evidence.prototype_class_index,
                               evidence.prototype_similarity,
                               coalesce(labels.display_name, labels.name,
                                        mappings.oracle_label_name,
                                        'Class ' || evidence.prototype_class_index::text)
                                   AS cluster_name
                        FROM {self.schema}.classification_evidence evidence
                        JOIN {self.schema}.classification_inference_runs runs
                          ON runs.id = evidence.inference_run_id
                        LEFT JOIN {self.schema}.model_class_mappings mappings
                          ON mappings.model_artifact_id = runs.model_artifact_id
                         AND mappings.class_index = evidence.prototype_class_index
                        LEFT JOIN {self.schema}.classification_labels labels
                          ON labels.id = mappings.project_label_id
                        WHERE evidence.project_id = %s
                          AND evidence.inference_run_id = %s
                          AND runs.project_id = %s
                          AND evidence.prototype_class_index IS NOT NULL
                    ), counts AS (
                        SELECT prototype_class_index,
                               max(cluster_name) AS cluster_name,
                               count(*) AS roi_count,
                               avg(prototype_similarity) AS mean_similarity,
                               min(prototype_similarity) AS min_similarity,
                               max(prototype_similarity) AS max_similarity
                        FROM assigned
                        GROUP BY prototype_class_index
                    ), representatives AS (
                        SELECT DISTINCT ON (prototype_class_index)
                               prototype_class_index,
                               refined_detection_id AS representative_detection_id,
                               prototype_similarity AS representative_similarity
                        FROM assigned
                        ORDER BY prototype_class_index,
                                 prototype_similarity DESC NULLS LAST,
                                 refined_detection_id
                    )
                    SELECT 'label-prototype:' || counts.prototype_class_index::text AS cluster_id,
                           counts.prototype_class_index AS cluster_index,
                           counts.cluster_name,
                           counts.roi_count,
                           counts.mean_similarity,
                           counts.min_similarity,
                           counts.max_similarity,
                           0::bigint AS novelty_count,
                           representatives.representative_detection_id,
                           representatives.representative_similarity
                    FROM counts
                    JOIN representatives USING (prototype_class_index)
                    ORDER BY counts.roi_count DESC, counts.prototype_class_index
                    """,
                    (project_id, inference_run_id, project_id),
                )
                return list(cursor.fetchall())

    def list_feature_space_cluster_members(
        self,
        *,
        project_id: str,
        inference_run_id: str,
        cluster_id: str,
        limit: int,
        offset: int,
        minimum: float = -1.0,
    ) -> dict[str, Any]:
        """Page one run-local cluster without collapsing it into a taxonomy label."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT refined.id, refined.frame_id, refined.roi_index, refined.area,
                           refined.roi_shape, refined.created_at,
                           assets.id AS asset_id, assets.filename AS asset_filename,
                           coalesce(labels.display_name, labels.name) AS label_display_name,
                           review.decision AS review_decision,
                           evidence.cluster_id, evidence.cluster_index,
                           evidence.similarity, evidence.novel, evidence.abstained,
                           count(*) OVER() AS total_count
                    FROM {self.schema}.clustering_evidence evidence
                    JOIN {self.schema}.detections_refined refined
                      ON refined.id = evidence.refined_detection_id
                    JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.roi_label_annotations annotation
                        WHERE annotation.refined_detection_id = refined.id
                          AND annotation.is_current
                        ORDER BY annotation.created_at DESC, annotation.id DESC
                        LIMIT 1
                    ) annotation ON true
                    LEFT JOIN {self.schema}.classification_labels labels
                      ON labels.id = annotation.label_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.roi_annotation_reviews latest_review
                        WHERE latest_review.annotation_id = annotation.id
                        ORDER BY latest_review.created_at DESC, latest_review.id DESC
                        LIMIT 1
                    ) review ON true
                    WHERE evidence.project_id = %s
                      AND evidence.inference_run_id = %s
                      AND evidence.cluster_id = %s
                      AND evidence.similarity >= %s
                      AND assets.project_id = %s
                    ORDER BY evidence.similarity DESC NULLS LAST, refined.id
                    LIMIT %s OFFSET %s
                    """,
                    (project_id, inference_run_id, cluster_id, minimum, project_id, limit, offset),
                )
                rows = list(cursor.fetchall())
        return {
            "items": rows,
            "total": int(rows[0]["total_count"]) if rows else 0,
            "limit": limit,
            "offset": offset,
        }

    def list_feature_space_label_prototype_members(
        self,
        *,
        project_id: str,
        inference_run_id: str,
        prototype_id: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        """Page one run-local classification label-prototype group."""

        prefix = "label-prototype:"
        if not prototype_id.startswith(prefix):
            raise ValueError("Label prototype ID must use the label-prototype:<class-index> form")
        try:
            prototype_class_index = int(prototype_id.removeprefix(prefix))
        except ValueError as exc:
            raise ValueError("Label prototype ID has an invalid class index") from exc
        if prototype_class_index < 0:
            raise ValueError("Label prototype class index must be non-negative")

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT refined.id, refined.frame_id, refined.roi_index, refined.area,
                           refined.roi_shape, refined.created_at,
                           assets.id AS asset_id, assets.filename AS asset_filename,
                           coalesce(annotations_labels.display_name, annotations_labels.name)
                               AS label_display_name,
                           review.decision AS review_decision,
                           'label-prototype:' || evidence.prototype_class_index::text AS cluster_id,
                           evidence.prototype_class_index AS cluster_index,
                           coalesce(prototype_labels.display_name, prototype_labels.name,
                                    mappings.oracle_label_name,
                                    'Class ' || evidence.prototype_class_index::text)
                               AS cluster_name,
                           evidence.prototype_similarity AS similarity,
                           false AS novel,
                           false AS abstained,
                           count(*) OVER() AS total_count
                    FROM {self.schema}.classification_evidence evidence
                    JOIN {self.schema}.classification_inference_runs runs
                      ON runs.id = evidence.inference_run_id
                    JOIN {self.schema}.detections_refined refined
                      ON refined.id = evidence.refined_detection_id
                    JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    LEFT JOIN {self.schema}.model_class_mappings mappings
                      ON mappings.model_artifact_id = runs.model_artifact_id
                     AND mappings.class_index = evidence.prototype_class_index
                    LEFT JOIN {self.schema}.classification_labels prototype_labels
                      ON prototype_labels.id = mappings.project_label_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.roi_label_annotations annotation
                        WHERE annotation.refined_detection_id = refined.id
                          AND annotation.is_current
                        ORDER BY annotation.created_at DESC, annotation.id DESC
                        LIMIT 1
                    ) annotation ON true
                    LEFT JOIN {self.schema}.classification_labels annotations_labels
                      ON annotations_labels.id = annotation.label_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM {self.schema}.roi_annotation_reviews latest_review
                        WHERE latest_review.annotation_id = annotation.id
                        ORDER BY latest_review.created_at DESC, latest_review.id DESC
                        LIMIT 1
                    ) review ON true
                    WHERE evidence.project_id = %s
                      AND evidence.inference_run_id = %s
                      AND runs.project_id = %s
                      AND evidence.prototype_class_index = %s
                      AND assets.project_id = %s
                    ORDER BY evidence.prototype_similarity DESC NULLS LAST, refined.id
                    LIMIT %s OFFSET %s
                    """,
                    (
                        project_id,
                        inference_run_id,
                        project_id,
                        prototype_class_index,
                        project_id,
                        limit,
                        offset,
                    ),
                )
                rows = list(cursor.fetchall())
        return {
            "items": rows,
            "total": int(rows[0]["total_count"]) if rows else 0,
            "limit": limit,
            "offset": offset,
        }

    def get_feature_space_cluster_assignment(
        self, *, project_id: str, inference_run_id: str, refined_detection_id: str
    ) -> dict[str, Any] | None:
        """Return the recorded run-local cluster for one ROI."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT cluster_id
                    FROM {self.schema}.clustering_evidence
                    WHERE project_id = %s
                      AND inference_run_id = %s
                      AND refined_detection_id = %s
                      AND cluster_id IS NOT NULL
                    """,
                    (project_id, inference_run_id, refined_detection_id),
                )
                return cursor.fetchone()

    def assign_curation_labels(
        self,
        *,
        project_id: str,
        roi_ids: Sequence[str],
        label_id: str,
        actor_user_id: str | None,
        actor_username: str,
        suggested_by_evidence_id: str | None = None,
        notes: str | None = None,
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT id FROM {self.schema}.classification_labels WHERE id = %s AND project_id = %s AND deprecated_at IS NULL",
                    (label_id, project_id),
                )
                if cursor.fetchone() is None:
                    raise KeyError(label_id)
                for roi_id in dict.fromkeys(roi_ids):
                    cursor.execute(
                        f"""
                        SELECT annotations.id
                        FROM {self.schema}.detections_refined refined
                        JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                        JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                        LEFT JOIN {self.schema}.roi_label_annotations annotations
                          ON annotations.refined_detection_id = refined.id AND annotations.is_current
                        WHERE refined.id = %s AND assets.project_id = %s
                        FOR UPDATE OF refined
                        """,
                        (roi_id, project_id),
                    )
                    current = cursor.fetchone()
                    if current is None:
                        raise KeyError(roi_id)
                    if suggested_by_evidence_id is not None:
                        cursor.execute(
                            f"""
                            SELECT id FROM {self.schema}.classification_evidence
                            WHERE id = %s AND project_id = %s
                              AND refined_detection_id = %s
                            """,
                            (suggested_by_evidence_id, project_id, roi_id),
                        )
                        if cursor.fetchone() is None:
                            raise KeyError(suggested_by_evidence_id)
                    parent_id = current.get("id")
                    if parent_id:
                        cursor.execute(
                            f"UPDATE {self.schema}.roi_label_annotations SET is_current = false WHERE id = %s",
                            (parent_id,),
                        )
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.roi_label_annotations
                            (project_id, refined_detection_id, label_id, actor_user_id,
                             actor_username, parent_annotation_id,
                             suggested_by_evidence_id, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING *
                        """,
                        (
                            project_id, roi_id, label_id, actor_user_id,
                            actor_username, parent_id, suggested_by_evidence_id, notes,
                        ),
                    )
                    created.append(cursor.fetchone())
            connection.commit()
        return created

    def review_curation_annotations(
        self,
        *,
        project_id: str,
        roi_ids: Sequence[str],
        decision: str,
        reviewer_user_id: str | None,
        reviewer_username: str,
        notes: str | None = None,
    ) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for roi_id in dict.fromkeys(roi_ids):
                    cursor.execute(
                        f"""
                        SELECT annotation.id
                        FROM {self.schema}.roi_label_annotations annotation
                        WHERE annotation.project_id = %s
                          AND annotation.refined_detection_id = %s
                          AND annotation.is_current
                          AND annotation.status <> 'deprecated'
                        """,
                        (project_id, roi_id),
                    )
                    annotation = cursor.fetchone()
                    if annotation is None:
                        raise KeyError(roi_id)
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.roi_annotation_reviews
                            (annotation_id, reviewer_user_id, reviewer_username,
                             decision, notes)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING *
                        """,
                        (
                            annotation["id"], reviewer_user_id, reviewer_username,
                            decision, notes,
                        ),
                    )
                    reviews.append(cursor.fetchone())
            connection.commit()
        return reviews

    def remove_curation_labels(
        self,
        *,
        project_id: str,
        roi_ids: Sequence[str],
        actor_username: str,
        notes: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retire current human assertions without deleting their audit history."""

        retired: list[dict[str, Any]] = []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for roi_id in dict.fromkeys(roi_ids):
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.roi_label_annotations
                        SET is_current = false,
                            status = 'deprecated',
                            metadata = metadata || %s::jsonb
                        WHERE project_id = %s
                          AND refined_detection_id = %s
                          AND is_current
                        RETURNING *
                        """,
                        (
                            json.dumps(
                                json_ready(
                                    {
                                        "retired_by": actor_username,
                                        "retired_notes": notes,
                                        "retired_at": datetime.now(timezone.utc).isoformat(),
                                    }
                                )
                            ),
                            project_id,
                            roi_id,
                        ),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise KeyError(roi_id)
                    retired.append(row)
            connection.commit()
        return retired

    def _classification_target_query(
        self,
        *,
        project_id: str,
        model_ref: str,
        evidence_kind: str = "classification",
        roi_ids: Sequence[str] = (),
        selection: Mapping[str, Any] | None = None,
    ) -> tuple[str, str, list[Any]]:
        """Build the shared target query used by previews, counts, and workers."""

        if evidence_kind not in {"classification", "clustering"}:
            raise ValueError(f"Unsupported evidence kind: {evidence_kind}")
        evidence_table = (
            "classification_evidence"
            if evidence_kind == "classification"
            else "clustering_evidence"
        )

        filters = dict(selection or {})
        clauses = ["assets.project_id = %s", "refined.roi_payload IS NOT NULL"]
        where_params: list[Any] = [project_id]
        if roi_ids:
            clauses.append("refined.id = ANY(%s::uuid[])")
            where_params.append(list(dict.fromkeys(roi_ids)))
        asset_ids = [str(value) for value in filters.get("asset_ids") or () if value]
        if asset_ids:
            clauses.append("assets.id = ANY(%s::uuid[])")
            where_params.append(list(dict.fromkeys(asset_ids)))
        collections = [str(value) for value in filters.get("collections") or () if value]
        if collections:
            clauses.append("assets.collections && %s::text[]")
            where_params.append(list(dict.fromkeys(collections)))

        annotation_state = str(filters.get("annotation_state") or "all")
        if annotation_state == "labeled":
            clauses.append("annotation.id IS NOT NULL AND annotation.status <> 'deprecated'")
        elif annotation_state == "unlabeled":
            clauses.append("(annotation.id IS NULL OR annotation.status = 'deprecated')")

        review_state = str(filters.get("review_state") or "all")
        if review_state == "unreviewed":
            clauses.append("annotation.id IS NOT NULL AND review.id IS NULL")
        elif review_state in {"verified", "rejected", "needs_review"}:
            clauses.append("review.decision = %s")
            where_params.append(review_state)

        label_id = filters.get("label_id")
        if label_id:
            if evidence_kind != "classification" and str(filters.get("label_source") or "any") != "human":
                raise ValueError("Clustering evidence queries cannot filter by predicted label")
            label_source = str(filters.get("label_source") or "any")
            if label_source == "human":
                clauses.append("annotation.label_id = %s")
                where_params.append(str(label_id))
            elif label_source == "prediction":
                clauses.append("model_evidence.predicted_label_id = %s")
                where_params.append(str(label_id))
            else:
                clauses.append(
                    "(annotation.label_id = %s OR model_evidence.predicted_label_id = %s)"
                )
                where_params.extend([str(label_id), str(label_id)])

        evidence_state = str(filters.get("evidence_state") or "missing_model")
        if evidence_state == "missing_model":
            clauses.append("model_evidence.id IS NULL")
        elif evidence_state == "available_model":
            clauses.append("model_evidence.id IS NOT NULL")
        elif evidence_state == "missing_any":
            clauses.append(
                f"NOT EXISTS (SELECT 1 FROM {self.schema}.{evidence_table} any_evidence "
                "WHERE any_evidence.refined_detection_id = refined.id)"
            )
        elif evidence_state == "available_any":
            clauses.append(
                f"EXISTS (SELECT 1 FROM {self.schema}.{evidence_table} any_evidence "
                "WHERE any_evidence.refined_detection_id = refined.id)"
            )
        elif evidence_state == "disagreement":
            if evidence_kind != "classification":
                raise ValueError("Clustering evidence does not have classification disagreement state")
            clauses.append(
                "model_evidence.id IS NOT NULL AND ((model_evidence.prototype_class_index IS NOT NULL AND "
                "model_evidence.prototype_class_index <> model_evidence.predicted_class_index) OR "
                "(model_evidence.knn_class_index IS NOT NULL AND "
                "model_evidence.knn_class_index <> model_evidence.predicted_class_index))"
            )

        min_area = filters.get("min_area")
        if min_area is not None:
            clauses.append("refined.area >= %s")
            where_params.append(float(min_area))
        max_area = filters.get("max_area")
        if max_area is not None:
            clauses.append("refined.area <= %s")
            where_params.append(float(max_area))
        search = str(filters.get("search") or "").strip()
        if search:
            token = f"%{search}%"
            clauses.append(
                "(refined.id::text ILIKE %s OR frames.id::text ILIKE %s OR assets.filename ILIKE %s)"
            )
            where_params.extend([token, token, token])

        joins = f"""
            LEFT JOIN LATERAL (
                SELECT * FROM {self.schema}.roi_label_annotations current_annotation
                WHERE current_annotation.refined_detection_id = refined.id
                  AND current_annotation.is_current
                ORDER BY current_annotation.created_at DESC, current_annotation.id DESC
                LIMIT 1
            ) annotation ON true
            LEFT JOIN LATERAL (
                SELECT * FROM {self.schema}.roi_annotation_reviews latest_review
                WHERE latest_review.annotation_id = annotation.id
                ORDER BY latest_review.created_at DESC, latest_review.id DESC
                LIMIT 1
            ) review ON true
            LEFT JOIN LATERAL (
                SELECT evidence.*
                FROM {self.schema}.{evidence_table} evidence
                JOIN {self.schema}.classification_inference_runs inference_run
                  ON inference_run.id = evidence.inference_run_id
                WHERE evidence.refined_detection_id = refined.id
                  AND inference_run.model_selector = %s
                ORDER BY evidence.created_at DESC, evidence.id DESC
                LIMIT 1
            ) model_evidence ON true
        """
        return joins, " AND ".join(clauses), [model_ref, *where_params]

    def list_classification_targets(
        self,
        *,
        project_id: str,
        model_ref: str,
        evidence_kind: str = "classification",
        roi_ids: Sequence[str] = (),
        selection: Mapping[str, Any] | None = None,
        after_created_at: datetime | None = None,
        after_id: str | None = None,
        limit: int = 128,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        joins, where, params = self._classification_target_query(
            project_id=project_id,
            model_ref=model_ref,
            evidence_kind=evidence_kind,
            roi_ids=roi_ids,
            selection=selection,
        )
        if after_created_at is not None and after_id:
            where += " AND (refined.created_at, refined.id) > (%s, %s::uuid)"
            params.extend([after_created_at, after_id])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT refined.*
                    FROM {self.schema}.detections_refined refined
                    JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    {joins}
                    WHERE {where}
                    ORDER BY refined.created_at, refined.id
                    LIMIT %s OFFSET %s
                    """,
                    (*params, limit, offset),
                )
                return list(cursor.fetchall())

    def count_classification_targets(
        self,
        *,
        project_id: str,
        model_ref: str,
        evidence_kind: str = "classification",
        roi_ids: Sequence[str] = (),
        selection: Mapping[str, Any] | None = None,
    ) -> int:
        """Count the refined ROIs that can actually be sent for classification."""

        joins, where, params = self._classification_target_query(
            project_id=project_id,
            model_ref=model_ref,
            evidence_kind=evidence_kind,
            roi_ids=roi_ids,
            selection=selection,
        )
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT count(*) AS target_count
                    FROM {self.schema}.detections_refined refined
                    JOIN {self.schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
                    {joins}
                    WHERE {where}
                    """,
                    tuple(params),
                )
                row = cursor.fetchone()
        return int((row or {}).get("target_count") or 0)

    def create_classification_inference_run(
        self,
        *,
        project_id: str,
        job_id: str | None,
        model_selector: str,
        evidence_kind: str = "classification",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.classification_inference_runs
                        (project_id, job_id, model_selector, evidence_kind, parameters)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    RETURNING *
                    """,
                    (
                        project_id,
                        job_id,
                        model_selector,
                        evidence_kind,
                        json.dumps(json_ready(parameters or {})),
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def _ensure_classification_artifact(
        self,
        cursor,
        *,
        project_id: str,
        model: dict[str, Any],
        probabilities: Sequence[dict[str, Any]],
    ) -> tuple[str, dict[int, str]]:
        cursor.execute(
            f"""
            INSERT INTO {self.schema}.model_artifacts
                (project_id, artifact_id, run_id, artifact_fingerprint,
                 task, architecture, contract_version, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (project_id, artifact_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                artifact_fingerprint = EXCLUDED.artifact_fingerprint,
                architecture = EXCLUDED.architecture,
                contract_version = EXCLUDED.contract_version
            RETURNING id
            """,
            (
                project_id,
                model["artifact_id"],
                model.get("run_id"),
                model.get("artifact_fingerprint"),
                model.get("task", "classification"),
                model.get("architecture"),
                model.get("contract_version"),
                json.dumps(json_ready(model)),
            ),
        )
        artifact_row_id = str(cursor.fetchone()["id"])
        mappings: dict[int, str] = {}
        for probability in probabilities:
            class_index = int(probability["class_index"])
            oracle_name = str(probability.get("label_name") or f"class-{class_index}")
            cursor.execute(
                f"""
                INSERT INTO {self.schema}.classification_labels
                    (project_id, name, display_name, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (project_id, name) DO UPDATE SET
                    display_name = coalesce({self.schema}.classification_labels.display_name, EXCLUDED.display_name)
                RETURNING id
                """,
                (
                    project_id,
                    oracle_name,
                    oracle_name,
                    json.dumps(
                        {
                            "source": "oracle_builder",
                            "oracle_label_id": probability.get("label_id"),
                        }
                    ),
                ),
            )
            project_label_id = str(cursor.fetchone()["id"])
            mappings[class_index] = project_label_id
            cursor.execute(
                f"""
                INSERT INTO {self.schema}.model_class_mappings
                    (model_artifact_id, class_index, oracle_label_id,
                     oracle_label_name, project_label_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (model_artifact_id, class_index) DO UPDATE SET
                    oracle_label_id = EXCLUDED.oracle_label_id,
                    oracle_label_name = EXCLUDED.oracle_label_name,
                    project_label_id = EXCLUDED.project_label_id
                """,
                (
                    artifact_row_id,
                    class_index,
                    probability.get("label_id"),
                    oracle_name,
                    project_label_id,
                ),
            )
        return artifact_row_id, mappings

    def prepare_classification_evidence_context(
        self,
        *,
        project_id: str,
        inference_run_id: str,
        model: dict[str, Any],
        probabilities: Sequence[dict[str, Any]],
    ) -> ClassificationEvidenceContext:
        """Resolve the invariant model catalog rows once for an inference run."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                artifact_row_id, mappings = self._ensure_classification_artifact(
                    cursor,
                    project_id=project_id,
                    model=model,
                    probabilities=probabilities,
                )
                cursor.execute(
                    f"UPDATE {self.schema}.classification_inference_runs SET model_artifact_id = %s WHERE id = %s",
                    (artifact_row_id, inference_run_id),
                )
            connection.commit()
        return ClassificationEvidenceContext(
            project_id=project_id,
            inference_run_id=inference_run_id,
            model_artifact_id=artifact_row_id,
            class_label_ids=mappings,
        )

    def _ensure_clustering_artifact(
        self,
        cursor,
        *,
        project_id: str,
        model: dict[str, Any],
    ) -> str:
        """Register a clustering or hybrid model without inventing labels."""

        cursor.execute(
            f"""
            INSERT INTO {self.schema}.model_artifacts
                (project_id, artifact_id, run_id, artifact_fingerprint,
                 task, architecture, contract_version, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (project_id, artifact_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                artifact_fingerprint = EXCLUDED.artifact_fingerprint,
                architecture = EXCLUDED.architecture,
                contract_version = EXCLUDED.contract_version,
                metadata = {self.schema}.model_artifacts.metadata || EXCLUDED.metadata
            RETURNING id
            """,
            (
                project_id,
                model["artifact_id"],
                model.get("run_id"),
                model.get("artifact_fingerprint"),
                model.get("task", "clustering"),
                model.get("architecture"),
                model.get("contract_version"),
                json.dumps(json_ready(model)),
            ),
        )
        return str(cursor.fetchone()["id"])

    def prepare_clustering_evidence_context(
        self,
        *,
        project_id: str,
        inference_run_id: str,
        model: dict[str, Any],
    ) -> ClusteringEvidenceContext:
        """Resolve the immutable Oracle artifact for a clustering run."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                artifact_row_id = self._ensure_clustering_artifact(
                    cursor, project_id=project_id, model=model
                )
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.classification_inference_runs
                    SET model_artifact_id = %s
                    WHERE id = %s
                    """,
                    (artifact_row_id, inference_run_id),
                )
            connection.commit()
        return ClusteringEvidenceContext(
            project_id=project_id,
            inference_run_id=inference_run_id,
            model_artifact_id=artifact_row_id,
        )

    def store_clustering_evidence_batch(
        self,
        *,
        evidence_context: ClusteringEvidenceContext,
        records: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Persist Oracle Builder's per-ROI clustering packets idempotently."""

        if not records:
            return []
        values: list[tuple[Any, ...]] = []
        for record in records:
            packet = dict(record.get("evidence_packet") or {})
            decision = dict(packet.get("decision") or {})
            values.append(
                (
                    evidence_context.project_id,
                    str(record["refined_detection_id"]),
                    evidence_context.inference_run_id,
                    evidence_context.model_artifact_id,
                    record.get("embedding_payload_ref"),
                    record.get("embedding_dtype"),
                    json.dumps(list(record.get("embedding_shape") or [])),
                    record.get("embedding_sha256"),
                    decision.get("cluster_index"),
                    decision.get("cluster_id"),
                    decision.get("similarity"),
                    decision.get("similarity_floor"),
                    decision.get("novelty_similarity_threshold"),
                    decision.get("novel"),
                    decision.get("abstained"),
                    json.dumps(json_ready(packet.get("clusters") or [])),
                    json.dumps(json_ready(packet.get("nearest_neighbors") or [])),
                    json.dumps(json_ready(packet)),
                    json.dumps(json_ready(record.get("oracle_result") or {})),
                )
            )
        columns = """
            project_id, refined_detection_id, inference_run_id, model_artifact_id,
            embedding_payload_ref, embedding_dtype, embedding_shape, embedding_sha256,
            cluster_index, cluster_id, similarity, similarity_floor,
            novelty_similarity_threshold, novel, abstained, candidate_clusters,
            nearest_neighbors, evidence_packet, oracle_result
        """
        placeholders = "(" + ", ".join(
            ["%s"] * 6 + ["%s::jsonb"] + ["%s"] * 8 + ["%s::jsonb"] * 4
        ) + ")"
        sql_text = f"""
            INSERT INTO {self.schema}.clustering_evidence ({columns})
            VALUES {", ".join(placeholders for _ in values)}
            ON CONFLICT (inference_run_id, refined_detection_id) DO UPDATE SET
                model_artifact_id = EXCLUDED.model_artifact_id,
                embedding_payload_ref = EXCLUDED.embedding_payload_ref,
                embedding_dtype = EXCLUDED.embedding_dtype,
                embedding_shape = EXCLUDED.embedding_shape,
                embedding_sha256 = EXCLUDED.embedding_sha256,
                cluster_index = EXCLUDED.cluster_index,
                cluster_id = EXCLUDED.cluster_id,
                similarity = EXCLUDED.similarity,
                similarity_floor = EXCLUDED.similarity_floor,
                novelty_similarity_threshold = EXCLUDED.novelty_similarity_threshold,
                novel = EXCLUDED.novel,
                abstained = EXCLUDED.abstained,
                candidate_clusters = EXCLUDED.candidate_clusters,
                nearest_neighbors = EXCLUDED.nearest_neighbors,
                evidence_packet = EXCLUDED.evidence_packet,
                oracle_result = EXCLUDED.oracle_result
            RETURNING id, refined_detection_id
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_text, tuple(value for row in values for value in row))
                rows = cursor.fetchall()
            connection.commit()
        return rows

    def store_classification_evidence_batch(
        self,
        *,
        evidence_context: ClassificationEvidenceContext,
        records: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Persist one Oracle response batch in one database transaction.

        ``records`` contain only values that vary by ROI.  Model catalog rows and
        class mappings are provided by ``evidence_context`` and are never written
        again here.  The method remains idempotent for a run/ROI pair.
        """
        if not records:
            return []

        values: list[tuple[Any, ...]] = []
        neighbors_by_detection: dict[str, list[tuple[Any, ...]]] = {}
        for record in records:
            output = dict(record["output"])
            probabilities = list(output.get("probabilities") or [])
            decision = dict(output.get("decision") or {})
            packet = dict(output.get("evidence") or {})
            prototype = dict(packet.get("prototype") or {})
            knn = dict(packet.get("knn") or {})
            probability_values = sorted(
                (float(value.get("probability") or 0.0) for value in probabilities),
                reverse=True,
            )
            entropy = -sum(value * math.log(value) for value in probability_values if value > 0)
            margin = (
                probability_values[0] - probability_values[1]
                if len(probability_values) > 1
                else (probability_values[0] if probability_values else 0.0)
            )
            predicted_class_index = decision.get("class_index")
            prototype_class = prototype.get("predicted_class")
            knn_class = knn.get("strongest_label")
            prototype_similarities = prototype.get("similarities") or {}
            weighted_support = knn.get("weighted_label_support") or {}
            refined_detection_id = str(record["refined_detection_id"])
            values.append(
                (
                    evidence_context.project_id,
                    refined_detection_id,
                    evidence_context.inference_run_id,
                    evidence_context.class_label_ids.get(int(predicted_class_index))
                    if predicted_class_index is not None
                    else None,
                    predicted_class_index,
                    decision.get("label_name"),
                    max(probability_values) if probability_values else None,
                    entropy,
                    margin,
                    prototype_class,
                    prototype_similarities.get(str(prototype_class)) if prototype_class is not None else None,
                    prototype.get("similarity_margin"),
                    knn_class,
                    knn.get("label_agreement"),
                    weighted_support.get(str(knn_class)) if knn_class is not None else None,
                    knn.get("label_support_margin"),
                    record.get("embedding_payload_ref"),
                    record.get("embedding_dtype"),
                    json.dumps(list(record.get("embedding_shape") or [])),
                    record.get("embedding_sha256"),
                    json.dumps(json_ready(probabilities)),
                    json.dumps(json_ready(packet)),
                    json.dumps(json_ready(record.get("oracle_result") or {})),
                )
            )
            label_names = {
                value.get("class_index"): value.get("label_name") for value in probabilities
            }
            neighbors_by_detection[refined_detection_id] = [
                (
                    rank,
                    str(neighbor["uuid"]),
                    neighbor.get("label"),
                    label_names.get(neighbor.get("label")),
                    neighbor.get("similarity"),
                )
                for rank, neighbor in enumerate(knn.get("neighbors") or [])
            ]

        columns = """
            project_id, refined_detection_id, inference_run_id,
            predicted_label_id, predicted_class_index, predicted_label_name,
            confidence, entropy, probability_margin,
            prototype_class_index, prototype_similarity, prototype_margin,
            knn_class_index, knn_agreement, knn_weighted_support, knn_margin,
            embedding_payload_ref, embedding_dtype, embedding_shape,
            embedding_sha256, probabilities, evidence_packet, oracle_result
        """
        row_placeholder = "(" + ", ".join(
            ["%s"] * 18 + ["%s::jsonb", "%s", "%s::jsonb", "%s::jsonb", "%s::jsonb"]
        ) + ")"
        evidence_sql = f"""
            INSERT INTO {self.schema}.classification_evidence ({columns})
            VALUES {", ".join(row_placeholder for _ in values)}
            ON CONFLICT (inference_run_id, refined_detection_id) DO UPDATE SET
                predicted_label_id = EXCLUDED.predicted_label_id,
                predicted_class_index = EXCLUDED.predicted_class_index,
                predicted_label_name = EXCLUDED.predicted_label_name,
                confidence = EXCLUDED.confidence,
                entropy = EXCLUDED.entropy,
                probability_margin = EXCLUDED.probability_margin,
                prototype_class_index = EXCLUDED.prototype_class_index,
                prototype_similarity = EXCLUDED.prototype_similarity,
                prototype_margin = EXCLUDED.prototype_margin,
                knn_class_index = EXCLUDED.knn_class_index,
                knn_agreement = EXCLUDED.knn_agreement,
                knn_weighted_support = EXCLUDED.knn_weighted_support,
                knn_margin = EXCLUDED.knn_margin,
                embedding_payload_ref = EXCLUDED.embedding_payload_ref,
                embedding_dtype = EXCLUDED.embedding_dtype,
                embedding_shape = EXCLUDED.embedding_shape,
                embedding_sha256 = EXCLUDED.embedding_sha256,
                probabilities = EXCLUDED.probabilities,
                evidence_packet = EXCLUDED.evidence_packet,
                oracle_result = EXCLUDED.oracle_result
            RETURNING id, refined_detection_id
        """
        parameters = tuple(value for row in values for value in row)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(evidence_sql, parameters)
                evidence_rows = cursor.fetchall()
                evidence_ids = [row["id"] for row in evidence_rows]
                cursor.execute(
                    f"DELETE FROM {self.schema}.classification_evidence_neighbors WHERE evidence_id = ANY(%s)",
                    (evidence_ids,),
                )
                evidence_ids_by_detection = {
                    str(row["refined_detection_id"]): row["id"] for row in evidence_rows
                }
                neighbor_values = [
                    (evidence_ids_by_detection[detection_id], *neighbor)
                    for detection_id, neighbors in neighbors_by_detection.items()
                    for neighbor in neighbors
                ]
                if neighbor_values:
                    neighbor_placeholder = "(" + ", ".join(["%s"] * 6) + ")"
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.classification_evidence_neighbors
                            (evidence_id, rank, exemplar_id, class_index, label_name, similarity)
                        VALUES {", ".join(neighbor_placeholder for _ in neighbor_values)}
                        """,
                        tuple(value for row in neighbor_values for value in row),
                    )
            connection.commit()
        return evidence_rows

    def complete_classification_inference_run(
        self,
        inference_run_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.classification_inference_runs
                    SET status = %s, metadata = metadata || %s::jsonb,
                        completed_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (status, json.dumps(json_ready(metadata or {})), inference_run_id),
                )
                row = cursor.fetchone()
            connection.commit()
        return row

    def _job_frame_status_ids(
        self,
        cursor,
        *,
        project_id: str,
        stage: str,
        payload_frame_ids: Sequence[str],
        payload_detection_ids: Sequence[str],
    ) -> list[str]:
        if stage in {
            PipelineStage.PREPROCESS_FRAMES.value,
            PipelineStage.SEGMENT.value,
        }:
            return list(dict.fromkeys(str(frame_id) for frame_id in payload_frame_ids if frame_id))
        if stage != PipelineStage.ROI_REFINEMENT.value or not payload_detection_ids:
            return []
        cursor.execute(
            f"""
            SELECT DISTINCT detections.frame_id
            FROM {self.schema}.detection_candidate detections
            JOIN {self.schema}.frames frames ON frames.id = detections.frame_id
            JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
            WHERE assets.project_id = %s
              AND detections.id = ANY(%s::uuid[])
            ORDER BY detections.frame_id
            """,
            (project_id, [str(detection_id) for detection_id in payload_detection_ids]),
        )
        return [str(row["frame_id"]) for row in cursor.fetchall()]

    def _ensure_frame_status_rows_in_cursor(
        self,
        cursor,
        *,
        project_id: str,
        frame_ids: Sequence[str],
    ) -> None:
        if not frame_ids:
            return
        cursor.execute(
            f"""
            INSERT INTO {self.schema}.frame_processing_status
                (project_id, frame_id, asset_id, run_id, frame_index, collections, updated_at)
            SELECT
                assets.project_id,
                frames.id,
                frames.asset_id,
                frames.run_id,
                frames.frame_index,
                assets.collections,
                NOW()
            FROM {self.schema}.frames frames
            JOIN {self.schema}.raw_assets assets ON assets.id = frames.asset_id
            WHERE assets.project_id = %s
              AND frames.id = ANY(%s::uuid[])
            ON CONFLICT (project_id, frame_id) DO UPDATE SET
                asset_id = EXCLUDED.asset_id,
                run_id = EXCLUDED.run_id,
                frame_index = EXCLUDED.frame_index,
                collections = EXCLUDED.collections,
                updated_at = NOW()
            """,
            (project_id, [str(frame_id) for frame_id in frame_ids]),
        )

    def _upsert_frame_stage_status_in_cursor(
        self,
        cursor,
        *,
        project_id: str,
        frame_ids: Sequence[str],
        stage: str,
        status: str,
        job_id: str | None = None,
    ) -> None:
        if not frame_ids:
            return
        normalized_status = self._normalize_frame_processing_status(status)
        stage_map = {
            PipelineStage.PREPROCESS_FRAMES.value: (
                "preprocessing_status",
                "preprocessing_job_id",
                "preprocessing_completed_at",
            ),
            PipelineStage.SEGMENT.value: (
                "candidate_detection_status",
                "candidate_detection_job_id",
                "candidate_detection_completed_at",
            ),
            PipelineStage.ROI_REFINEMENT.value: (
                "roi_refinement_status",
                "roi_refinement_job_id",
                "roi_refinement_completed_at",
            ),
        }
        if stage not in stage_map:
            return
        status_column, job_column, completed_column = stage_map[stage]
        completed_value = datetime.now(timezone.utc) if normalized_status == JobStatus.SUCCEEDED.value else None
        self._ensure_frame_status_rows_in_cursor(cursor, project_id=project_id, frame_ids=frame_ids)
        cursor.execute(
            f"""
            UPDATE {self.schema}.frame_processing_status
            SET
                {status_column} = %s,
                {job_column} = %s,
                {completed_column} = %s,
                updated_at = NOW()
            WHERE project_id = %s
              AND frame_id = ANY(%s::uuid[])
            """,
            (
                normalized_status,
                job_id,
                completed_value,
                project_id,
                [str(frame_id) for frame_id in frame_ids],
            ),
        )

    def _touch_processing_status_snapshot_in_cursor(self, cursor, *, project_id: str) -> None:
        cursor.execute(
            f"""
            UPDATE {self.schema}.project_processing_status_snapshots
            SET status_version = status_version + 1,
                updated_at = NOW()
            WHERE project_id = %s
              AND session_id IS NULL
            RETURNING id
            """,
            (project_id,),
        )
        if cursor.fetchone() is None:
            cursor.execute(
                f"""
                INSERT INTO {self.schema}.project_processing_status_snapshots
                    (project_id, session_id, status_version, updated_at, summary)
                VALUES (%s, NULL, 1, NOW(), '{{}}'::jsonb)
                """,
                (project_id,),
            )

    def create_job(
        self,
        stage: PipelineStage | str,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        status: JobStatus | str = JobStatus.QUEUED,
        priority: int | None = None,
        max_attempts: int | None = None,
        payload: dict[str, Any] | None = None,
        depends_on: Sequence[str] | None = None,
        summary: str | None = None,
        progress: dict[str, Any] | None = None,
        submitted_by_user_id: str | None = None,
        submitted_by_username: str | None = None,
    ) -> dict[str, Any]:
        stage_value = stage.value if isinstance(stage, PipelineStage) else stage
        status_value = status.value if isinstance(status, JobStatus) else status
        resolved_project_id = self._required_project_id(project_id, "create_job")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                resolved_payload = payload or {}
                payload_frame_ids = [
                    str(frame_id)
                    for frame_id in (resolved_payload.get("frame_ids") or [])
                    if frame_id
                ]
                if resolved_payload.get("frame_id"):
                    payload_frame_ids.append(str(resolved_payload["frame_id"]))
                payload_detection_ids = [
                    str(detection_id)
                    for detection_id in (resolved_payload.get("detection_ids") or [])
                    if detection_id
                ]
                resolved_progress = progress or _initial_job_progress(
                    stage_value,
                    status_value,
                    resolved_payload,
                )
                self._ensure_project_scope(
                    cursor,
                    resolved_project_id,
                    run_id=run_id,
                    asset_id=asset_id,
                    job_ids=depends_on,
                    frame_ids=list(dict.fromkeys(payload_frame_ids)),
                    detection_ids=list(dict.fromkeys(payload_detection_ids)),
                )
                cursor.execute(
                    f"""
                    INSERT INTO {self.schema}.processing_jobs
                    (project_id, run_id, asset_id, stage, status, priority, attempt_count, max_attempts, payload, progress, summary,
                     submitted_by_user_id, submitted_by_username)
                    VALUES (%s, %s, %s, %s::{self.schema}.stage_name, %s::{self.schema}.job_status, %s, 0, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
                    RETURNING *;
                    """,
                    (
                        resolved_project_id,
                        run_id,
                        asset_id,
                        stage_value,
                        status_value,
                        priority if priority is not None else self.config.queue.default_priority,
                        max_attempts if max_attempts is not None else self.config.queue.max_attempts,
                        json.dumps(json_ready(resolved_payload)),
                        json.dumps(json_ready(resolved_progress)),
                        summary,
                        submitted_by_user_id,
                        submitted_by_username,
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
                        "project_id": row.get("project_id"),
                        "run_id": row.get("run_id"),
                        "asset_id": row.get("asset_id"),
                        "priority": row.get("priority"),
                        "depends_on": [str(dependency) for dependency in depends_on or []],
                        "submitted_by_user_id": submitted_by_user_id,
                        "submitted_by_username": submitted_by_username,
                    },
                )
                status_frame_ids = self._job_frame_status_ids(
                    cursor,
                    project_id=resolved_project_id,
                    stage=stage_value,
                    payload_frame_ids=payload_frame_ids,
                    payload_detection_ids=payload_detection_ids,
                )
                if status_frame_ids:
                    self._upsert_frame_stage_status_in_cursor(
                        cursor,
                        project_id=resolved_project_id,
                        frame_ids=status_frame_ids,
                        stage=stage_value,
                        status=status_value,
                        job_id=str(row["id"]),
                    )
                    self._touch_processing_status_snapshot_in_cursor(
                        cursor,
                        project_id=resolved_project_id,
                    )
            connection.commit()
        return row

    # Processing series deliberately wrap, rather than alter, ordinary jobs.  This
    # keeps historical queue rows and direct job APIs compatible with series work.
    def create_processing_series(
        self,
        *,
        project_id: str,
        steps: Sequence[dict[str, Any]],
        selection: dict[str, Any] | None = None,
        preset_snapshot: dict[str, Any] | None = None,
        failure_policy: str = "fail_fast",
        priority: int | None = None,
        submitted_by_user_id: str | None = None,
        submitted_by_username: str | None = None,
    ) -> dict[str, Any]:
        if failure_policy not in {"fail_fast", "continue"}:
            raise ValueError("failure_policy must be one of: fail_fast, continue.")
        if not steps:
            raise ValueError("A processing series requires at least one step.")
        resolved_project_id = self._required_project_id(project_id, "create_processing_series")
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""INSERT INTO {self.schema}.processing_series
                    (project_id, failure_policy, priority, submitted_by_user_id, submitted_by_username, selection, preset_snapshot)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb) RETURNING *""",
                    (resolved_project_id, failure_policy, priority if priority is not None else self.config.queue.default_priority,
                     submitted_by_user_id, submitted_by_username,
                     json.dumps(json_ready(selection or {})), json.dumps(json_ready(preset_snapshot or {}))),
                )
                series = cursor.fetchone()
                for index, step in enumerate(steps):
                    stage = str(step["stage"])
                    cursor.execute(
                        f"""INSERT INTO {self.schema}.processing_series_steps
                        (series_id, step_index, stage, filters, options, failure_policy)
                        VALUES (%s, %s, %s::{self.schema}.stage_name, %s::jsonb, %s::jsonb, %s)
                        RETURNING *""",
                        (series["id"], index, stage, json.dumps(json_ready(step.get("filters") or {})),
                         json.dumps(json_ready(step.get("options") or {})), step.get("failure_policy")),
                    )
            connection.commit()
        return self.get_processing_series(str(series["id"]), project_id=resolved_project_id) or series

    def get_processing_series(self, series_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        clauses, params = ["id = %s"], [series_id]
        if project_id is not None:
            clauses.append("project_id = %s")
            params.append(project_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.processing_series WHERE {' AND '.join(clauses)}", tuple(params))
                series = cursor.fetchone()
        if series is None:
            return None
        steps = self.list_processing_series_steps(series_id, project_id=project_id)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""SELECT
                        COUNT(units.id)::bigint AS total_jobs,
                        COUNT(units.id) FILTER (WHERE jobs.status = 'succeeded')::bigint AS succeeded_jobs,
                        COUNT(units.id) FILTER (WHERE jobs.status IN ('failed', 'dead_lettered'))::bigint AS failed_jobs,
                        COUNT(units.id) FILTER (WHERE jobs.status = 'cancelled')::bigint AS cancelled_jobs,
                        COUNT(*) FILTER (WHERE steps.status = 'skipped')::bigint AS skipped_steps
                    FROM {self.schema}.processing_series_steps steps
                    LEFT JOIN {self.schema}.processing_work_units units ON units.step_id = steps.id
                    LEFT JOIN {self.schema}.processing_jobs jobs ON jobs.id = units.job_id
                    WHERE steps.series_id = %s""",
                    (series_id,),
                )
                counts = cursor.fetchone() or {}
        total = int(counts.get("total_jobs") or 0) + int(counts.get("skipped_steps") or 0)
        completed = int(counts.get("succeeded_jobs") or 0) + int(counts.get("skipped_steps") or 0)
        failed = int(counts.get("failed_jobs") or 0)
        return {
            **series,
            "steps": steps,
            "progress": {
                "unit": "jobs",
                "total": total,
                "completed": completed,
                "failed": failed,
                "skipped": int(counts.get("skipped_steps") or 0) + int(counts.get("cancelled_jobs") or 0),
                "percent": (completed / total * 100) if total else None,
            },
        }

    def list_processing_series(
        self,
        *,
        project_id: str,
        status: Sequence[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        resolved_project_id = self._required_project_id(project_id, "list_processing_series")
        bounded_limit = min(max(1, int(limit)), 1000)
        clauses = ["project_id = %s"]
        params: list[Any] = [resolved_project_id]
        statuses = [str(value) for value in (status or []) if value]
        if statuses:
            clauses.append("status = ANY(%s::text[])")
            params.append(statuses)
        params.extend([bounded_limit, max(0, int(offset))])
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""SELECT id FROM {self.schema}.processing_series
                    WHERE {' AND '.join(clauses)}
                    ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s""",
                    tuple(params),
                )
                ids = [str(row["id"]) for row in cursor.fetchall()]
        return [self.get_processing_series(series_id, project_id=resolved_project_id) for series_id in ids]

    def list_processing_series_steps(self, series_id: str, *, project_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""SELECT steps.* FROM {self.schema}.processing_series_steps steps
                    JOIN {self.schema}.processing_series series ON series.id = steps.series_id
                    WHERE steps.series_id = %s AND (%s::uuid IS NULL OR series.project_id = %s::uuid)
                    ORDER BY steps.step_index""", (series_id, project_id, project_id))
                return cursor.fetchall()

    def list_processing_work_units(self, series_id: str, *, project_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""SELECT units.*, jobs.status AS job_status, jobs.stage, jobs.error_message, jobs.result
                    FROM {self.schema}.processing_work_units units
                    JOIN {self.schema}.processing_series series ON series.id = units.series_id
                    JOIN {self.schema}.processing_jobs jobs ON jobs.id = units.job_id
                    WHERE units.series_id = %s AND (%s::uuid IS NULL OR series.project_id = %s::uuid)
                    ORDER BY units.created_at, units.id""", (series_id, project_id, project_id))
                return cursor.fetchall()

    def claim_processing_series_step(self, series_id: str, *, project_id: str) -> dict[str, Any] | None:
        """Atomically grant one director the next step; callers may safely retry."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""WITH candidate AS (
                        SELECT steps.id FROM {self.schema}.processing_series_steps steps
                        JOIN {self.schema}.processing_series series ON series.id = steps.series_id
                        WHERE steps.series_id = %s AND series.project_id = %s
                          AND series.status IN ('queued', 'active') AND steps.status = 'queued'
                        ORDER BY steps.step_index FOR UPDATE SKIP LOCKED LIMIT 1
                    ) UPDATE {self.schema}.processing_series_steps steps
                    SET status = 'planning', started_at = COALESCE(started_at, NOW())
                    FROM candidate WHERE steps.id = candidate.id
                    RETURNING steps.*, (SELECT priority FROM {self.schema}.processing_series WHERE id = steps.series_id) AS series_priority,
                    (SELECT submitted_by_user_id FROM {self.schema}.processing_series WHERE id = steps.series_id) AS submitted_by_user_id,
                    (SELECT submitted_by_username FROM {self.schema}.processing_series WHERE id = steps.series_id) AS submitted_by_username""", (series_id, project_id))
                step = cursor.fetchone()
                if step is not None:
                    cursor.execute(f"UPDATE {self.schema}.processing_series SET status = 'active', updated_at = NOW() WHERE id = %s", (series_id,))
            connection.commit()
        return step

    def attach_processing_work_units(self, *, series_id: str, step_id: str, job_ids: Sequence[str], matched_count: int) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if job_ids:
                    cursor.executemany(
                        f"INSERT INTO {self.schema}.processing_work_units (series_id, step_id, job_id) VALUES (%s, %s, %s) ON CONFLICT (job_id) DO NOTHING",
                        [(series_id, step_id, job_id) for job_id in job_ids],
                    )
                cursor.execute(
                    f"""UPDATE {self.schema}.processing_series_steps SET status = %s, matched_count = %s, job_count = %s,
                        skip_reason = CASE WHEN %s = 'skipped' THEN 'no_eligible_units' ELSE NULL END,
                        finished_at = CASE WHEN %s = 'skipped' THEN NOW() ELSE NULL END WHERE id = %s""",
                    ('active' if job_ids else 'skipped', matched_count, len(job_ids),
                     'active' if job_ids else 'skipped', 'active' if job_ids else 'skipped', step_id))
                if not job_ids:
                    cursor.execute(f"""SELECT 1 FROM {self.schema}.processing_series_steps
                        WHERE series_id = %s AND status IN ('queued', 'planning', 'active') LIMIT 1""", (series_id,))
                    if cursor.fetchone() is None:
                        cursor.execute(f"UPDATE {self.schema}.processing_series SET status = 'succeeded', finished_at = NOW(), updated_at = NOW() WHERE id = %s", (series_id,))
            connection.commit()

    def finish_processing_series_step(self, step_id: str, *, failed: bool, failure_policy: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""SELECT steps.*, series.project_id, series.failure_policy AS series_failure_policy
                    FROM {self.schema}.processing_series_steps steps JOIN {self.schema}.processing_series series ON series.id = steps.series_id
                    WHERE steps.id = %s FOR UPDATE""", (step_id,))
                step = cursor.fetchone()
                if step is None or step['status'] != 'active':
                    return step
                policy = step.get('failure_policy') or failure_policy or step['series_failure_policy']
                terminal_status = 'failed' if failed and policy == 'fail_fast' else 'succeeded'
                cursor.execute(f"UPDATE {self.schema}.processing_series_steps SET status = %s, finished_at = NOW() WHERE id = %s", (terminal_status, step_id))
                if terminal_status == 'failed':
                    cursor.execute(f"UPDATE {self.schema}.processing_series SET status = 'failed', finished_at = NOW(), updated_at = NOW() WHERE id = %s", (step['series_id'],))
                    cursor.execute(f"""UPDATE {self.schema}.processing_jobs jobs SET status = 'cancelled', finished_at = NOW(),
                        control_reason = 'series_fail_fast', updated_at = NOW()
                        FROM {self.schema}.processing_work_units units
                        WHERE units.job_id = jobs.id AND units.series_id = %s AND jobs.status IN ('queued', 'leased', 'working', 'paused')""", (step['series_id'],))
                    cursor.execute(f"UPDATE {self.schema}.processing_series_steps SET status = 'cancelled', finished_at = NOW() WHERE series_id = %s AND status IN ('queued', 'planning', 'active')", (step['series_id'],))
                else:
                    cursor.execute(f"SELECT 1 FROM {self.schema}.processing_series_steps WHERE series_id = %s AND status IN ('queued', 'planning', 'active') LIMIT 1", (step['series_id'],))
                    if cursor.fetchone() is None:
                        cursor.execute(f"UPDATE {self.schema}.processing_series SET status = 'succeeded', finished_at = NOW(), updated_at = NOW() WHERE id = %s", (step['series_id'],))
            connection.commit()
        return step

    def advance_processing_series_for_job(self, job_id: str) -> dict[str, Any] | None:
        """Mark a series step terminal only after every attached job is terminal.

        The conditional step update makes duplicate worker completion notifications harmless.
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT step_id, series_id FROM {self.schema}.processing_work_units WHERE job_id = %s", (job_id,))
                unit = cursor.fetchone()
                if unit is None:
                    return None
                cursor.execute(f"""SELECT COUNT(*) FILTER (WHERE jobs.status IN ('queued','leased','working','paused')) AS active,
                    COUNT(*) FILTER (WHERE jobs.status IN ('failed','dead_lettered','cancelled')) AS failed
                    FROM {self.schema}.processing_work_units units JOIN {self.schema}.processing_jobs jobs ON jobs.id = units.job_id
                    WHERE units.step_id = %s""", (unit['step_id'],))
                counts = cursor.fetchone()
                if int(counts['active']) > 0:
                    return {**unit, 'ready': False}
                cursor.execute(f"SELECT failure_policy FROM {self.schema}.processing_series WHERE id = %s", (unit['series_id'],))
                series = cursor.fetchone()
        if series is None:
            return None
        self.finish_processing_series_step(unit['step_id'], failed=bool(counts['failed']), failure_policy=series['failure_policy'])
        return {**unit, 'ready': True, 'failed': bool(counts['failed'])}

    def _set_processing_series_status(self, series_id: str, *, project_id: str, status: str, reason: str | None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"UPDATE {self.schema}.processing_series SET status = %s, control_reason = %s, updated_at = NOW(), finished_at = CASE WHEN %s = 'cancelled' THEN NOW() ELSE finished_at END WHERE id = %s AND project_id = %s RETURNING *", (status, reason, status, series_id, project_id))
                row = cursor.fetchone()
            connection.commit()
        return row

    def pause_processing_series(self, series_id: str, *, project_id: str, reason: str | None = None) -> dict[str, Any] | None:
        row = self._set_processing_series_status(series_id, project_id=project_id, status='paused', reason=reason)
        if row is not None:
            self.pause_jobs(project_id=project_id, job_ids=[str(unit['job_id']) for unit in self.list_processing_work_units(series_id, project_id=project_id)], reason=reason)
        return row

    def resume_processing_series(self, series_id: str, *, project_id: str, reason: str | None = None) -> dict[str, Any] | None:
        row = self._set_processing_series_status(series_id, project_id=project_id, status='active', reason=reason)
        if row is not None:
            self.resume_jobs(project_id=project_id, job_ids=[str(unit['job_id']) for unit in self.list_processing_work_units(series_id, project_id=project_id)], reason=reason)
        return row

    def cancel_processing_series(self, series_id: str, *, project_id: str, reason: str | None = None) -> dict[str, Any] | None:
        row = self._set_processing_series_status(series_id, project_id=project_id, status='cancelled', reason=reason)
        if row is not None:
            self.cancel_jobs(project_id=project_id, job_ids=[str(unit['job_id']) for unit in self.list_processing_work_units(series_id, project_id=project_id)], reason=reason)
        return row

    def retry_processing_series(self, series_id: str, *, project_id: str, reason: str | None = None) -> dict[str, Any] | None:
        """Requeue failed/cancelled units without changing the recorded plan."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {self.schema}.processing_series WHERE id = %s AND project_id = %s FOR UPDATE", (series_id, project_id))
                series = cursor.fetchone()
                if series is None:
                    return None
                cursor.execute(f"""UPDATE {self.schema}.processing_jobs jobs SET status = 'queued', lease_expires_at = NULL,
                    worker_id = NULL, error_message = NULL, finished_at = NULL, control_reason = %s, updated_at = NOW()
                    FROM {self.schema}.processing_work_units units WHERE units.job_id = jobs.id AND units.series_id = %s
                    AND jobs.status IN ('failed', 'dead_lettered', 'cancelled')""", (reason, series_id))
                cursor.execute(f"""UPDATE {self.schema}.processing_series_steps SET status = 'active', finished_at = NULL
                    WHERE series_id = %s AND status IN ('failed', 'cancelled')""", (series_id,))
                cursor.execute(f"UPDATE {self.schema}.processing_series SET status = 'active', finished_at = NULL, control_reason = %s, updated_at = NOW() WHERE id = %s RETURNING *", (reason, series_id))
                row = cursor.fetchone()
            connection.commit()
        return row

    def plan_preprocess_frames(
        self,
        *,
        project_id: str,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Resolve project frames for backend-owned preprocessing queue planning."""
        resolved_project_id = self._required_project_id(project_id, "plan_preprocess_frames")
        clauses, params = self._frame_status_filters(
            project_id=resolved_project_id,
            run_id=filters.get("run_id"),
            asset_id=filters.get("asset_id"),
            asset_ids=filters.get("asset_ids"),
            frame_ids=filters.get("frame_ids"),
            collection=None,
            preprocessing_status=filters.get("preprocessing_status"),
            start_frame=filters.get("start_frame"),
            end_frame=filters.get("end_frame"),
        )
        collections = [str(value) for value in filters.get("collection") or [] if value]
        if collections:
            clauses.append("status.collections && %s::text[]")
            params.append(collections)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT status.frame_id, status.asset_id, status.run_id, status.frame_index,
                           COALESCE(frames.payload_ref, frames.kvstore_hash) AS payload_ref
                    FROM {self.schema}.frame_processing_status status
                    JOIN {self.schema}.frames frames ON frames.id = status.frame_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY COALESCE(frames.payload_ref, frames.kvstore_hash) ASC NULLS LAST,
                             status.frame_id ASC
                    """,
                    tuple(params),
                )
                return cursor.fetchall()

    def plan_segment_frames(self, *, project_id: str, filters: dict[str, Any], payload_kind: str) -> list[dict[str, Any]]:
        resolved_project_id = self._required_project_id(project_id, "plan_segment_frames")
        clauses, params = self._frame_status_filters(
            project_id=resolved_project_id, run_id=filters.get("run_id"), asset_id=filters.get("asset_id"),
            asset_ids=filters.get("asset_ids"), frame_ids=filters.get("frame_ids"),
            collection=None, candidate_detection_status=filters.get("candidate_detection_status"),
            preprocessing_status=filters.get("preprocessing_status"), start_frame=filters.get("start_frame"), end_frame=filters.get("end_frame"),
        )
        collections = [str(value) for value in filters.get("collection") or [] if value]
        if collections:
            clauses.append("status.collections && %s::text[]")
            params.append(collections)
        payload_ref = "COALESCE(frames.preprocessed_payload_ref, frames.preprocessed_kvstore_hash)" if payload_kind in {"preprocessed", "processed", "corrected"} else "COALESCE(frames.payload_ref, frames.kvstore_hash)"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"""SELECT status.frame_id, status.asset_id, status.run_id, status.frame_index, {payload_ref} AS payload_ref
                    FROM {self.schema}.frame_processing_status status JOIN {self.schema}.frames frames ON frames.id = status.frame_id
                    WHERE {' AND '.join(clauses)} ORDER BY {payload_ref} ASC NULLS LAST, status.frame_id ASC""", tuple(params))
                return cursor.fetchall()

    def plan_roi_refinement_detections(self, *, project_id: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        resolved_project_id = self._required_project_id(project_id, "plan_roi_refinement_detections")
        clauses, params = self._frame_status_filters(
            project_id=resolved_project_id, run_id=filters.get("run_id"), asset_id=filters.get("asset_id"),
            asset_ids=filters.get("asset_ids"), frame_ids=filters.get("frame_ids"), collection=None,
            roi_refinement_status=filters.get("roi_refinement_status"), start_frame=filters.get("start_frame"), end_frame=filters.get("end_frame"),
        )
        collections = [str(value) for value in filters.get("collection") or [] if value]
        if collections:
            clauses.append("status.collections && %s::text[]")
            params.append(collections)
        refinement_clause = self._candidate_refinement_state_clause(
            schema=self.schema,
            refinement_states=filters.get("refinement_state"),
        )
        if refinement_clause:
            clauses.append(refinement_clause)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"""SELECT detections.id AS detection_id, detections.frame_id, status.asset_id, status.run_id, detections.roi_index
                    FROM {self.schema}.detection_candidate detections
                    JOIN {self.schema}.frame_processing_status status ON status.frame_id = detections.frame_id
                    WHERE {' AND '.join(clauses)} ORDER BY detections.frame_id ASC, detections.roi_index ASC, detections.id ASC""", tuple(params))
                return cursor.fetchall()

    @staticmethod
    def _candidate_refinement_state_clause(*, schema: str, refinement_states: Sequence[str] | None) -> str | None:
        states = {str(state).strip().lower() for state in (refinement_states or ["unrefined"]) if str(state).strip()}
        if not states or states == {"refined", "unrefined"}:
            return None
        exists_clause = (
            f"EXISTS (SELECT 1 FROM {schema}.detections_refined refined "
            "WHERE refined.candidate_detection_id = detections.id)"
        )
        if states == {"refined"}:
            return exists_clause
        if states == {"unrefined"}:
            return f"NOT {exists_clause}"
        return None

    def create_preprocess_jobs(
        self,
        *,
        project_id: str,
        jobs: Sequence[dict[str, Any]],
        eligible_statuses: Sequence[str],
        priority: int | None = None,
        submitted_by_user_id: str | None = None,
        submitted_by_username: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create planned preprocessing jobs and queue their frames atomically."""
        if not jobs:
            return []
        resolved_project_id = self._required_project_id(project_id, "create_preprocess_jobs")
        created: list[dict[str, Any]] = []
        with self.connect() as connection:
            with connection.cursor() as cursor:
                all_frame_ids = [frame_id for job in jobs for frame_id in job["frame_ids"]]
                self._ensure_project_scope(cursor, resolved_project_id, frame_ids=all_frame_ids)
                normalized_statuses = [self._normalize_frame_processing_status(status) for status in eligible_statuses]
                cursor.execute(
                    f"""
                    SELECT frame_id
                    FROM {self.schema}.frame_processing_status
                    WHERE project_id = %s
                      AND frame_id = ANY(%s::uuid[])
                      AND preprocessing_status = ANY(%s)
                    FOR UPDATE
                    """,
                    (resolved_project_id, all_frame_ids, normalized_statuses),
                )
                eligible_frame_ids = {str(row["frame_id"]) for row in cursor.fetchall()}
                missing = [frame_id for frame_id in all_frame_ids if frame_id not in eligible_frame_ids]
                if missing:
                    raise ValueError(
                        "Some frames are no longer eligible for preprocessing: " + ", ".join(missing[:10])
                    )
                for job in jobs:
                    progress = job.get("progress") or _initial_job_progress(
                        PipelineStage.PREPROCESS_FRAMES.value,
                        JobStatus.QUEUED.value,
                        dict(job.get("payload") or {}),
                    )
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.processing_jobs
                            (project_id, run_id, asset_id, stage, status, priority, attempt_count, max_attempts, payload, progress, summary,
                             submitted_by_user_id, submitted_by_username)
                        VALUES (%s, %s, %s, 'preprocess_frames'::{self.schema}.stage_name,
                                'queued'::{self.schema}.job_status, %s, 0, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
                        RETURNING *
                        """,
                        (
                            resolved_project_id,
                            job.get("run_id"),
                            job.get("asset_id"),
                            priority if priority is not None else self.config.queue.default_priority,
                            self.config.queue.max_attempts,
                            json.dumps(json_ready(job["payload"])),
                            json.dumps(json_ready(progress)),
                            job["summary"],
                            submitted_by_user_id,
                            submitted_by_username,
                        ),
                    )
                    row = cursor.fetchone()
                    self._append_job_event(cursor, row["id"], "job.created", {"stage": "preprocess_frames", "status": "queued", "project_id": resolved_project_id})
                    self._upsert_frame_stage_status_in_cursor(
                        cursor,
                        project_id=resolved_project_id,
                        frame_ids=job["frame_ids"],
                        stage=PipelineStage.PREPROCESS_FRAMES.value,
                        status=JobStatus.QUEUED.value,
                        job_id=str(row["id"]),
                    )
                    created.append(row)
                self._touch_processing_status_snapshot_in_cursor(cursor, project_id=resolved_project_id)
            connection.commit()
        return created

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
        logs_tail = None
        if log_message:
            current = self.get_job(job_id)
            if current is None:
                return None
            logs_tail = list(current.get("logs_tail") or [])
            logs_tail.append(log_message)
            logs_tail = logs_tail[-20:]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET progress = %s::jsonb,
                        summary = COALESCE(%s, summary),
                        logs_tail = COALESCE(%s::jsonb, logs_tail),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (
                        json.dumps(json_ready(progress)),
                        summary,
                        None if logs_tail is None else json.dumps(json_ready(logs_tail)),
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
        project_id: str | None = None,
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
                    project_id=project_id,
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
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        request_id: str | None = None,
        duration_ms: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_project_id = self._resolve_project_id(
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            job_id=job_id,
        )
        cursor.execute(
            f"""
            INSERT INTO {self.schema}.logs
            (project_id, level, logger, event_type, message, run_id, asset_id, job_id, worker_id, request_id, duration_ms, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING *;
            """,
            (
                resolved_project_id,
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
        project_id = log_payload.get("project_id")
        run_id = log_payload.get("run_id")
        asset_id = log_payload.get("asset_id")
        worker_id = log_payload.get("worker_id")
        if job_id is not None and (project_id is None or run_id is None or asset_id is None):
            cursor.execute(
                f"""
                SELECT project_id, run_id, asset_id
                FROM {self.schema}.processing_jobs
                WHERE id = %s
                """,
                (job_id,),
            )
            job_row = cursor.fetchone()
            if job_row is not None:
                project_id = project_id or job_row.get("project_id")
                run_id = run_id or job_row.get("run_id")
                asset_id = asset_id or job_row.get("asset_id")
        if project_id is not None:
            self._append_log(
                cursor,
                event_type=event_type,
                message=_event_message(event_type, log_payload),
                level=_event_level(event_type),
                logger="pelagia.jobs",
                project_id=project_id,
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
        project_id: str | None = None,
        after_id: int | None = None,
        run_id: str | None = None,
        job_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        joins = ""
        if project_id:
            joins = f"LEFT JOIN {self.schema}.processing_jobs jobs ON jobs.id = events.job_id"
            clauses.append("(jobs.project_id = %s OR (events.job_id IS NULL AND events.payload->>'project_id' = %s))")
            params.extend([project_id, project_id])
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
        project_id: str | None = None,
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
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
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

    def set_job_priority(
        self,
        job_id: str,
        priority: int,
        reason: str | None = None,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["id = %s"]
        params: list[Any] = [job_id]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        params = [priority, reason, *params]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.schema}.processing_jobs
                    SET priority = %s,
                        control_reason = COALESCE(%s, control_reason),
                        updated_at = NOW()
                    WHERE {' AND '.join(clauses)}
                    RETURNING *;
                    """,
                    tuple(params),
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

    def pause_job(
        self,
        job_id: str,
        reason: str | None = None,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_job(job_id, project_id=project_id)
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
                        WHERE id = %s AND (%s::uuid IS NULL OR project_id = %s::uuid)
                        RETURNING *;
                        """,
                        (reason, job_id, project_id, project_id),
                    )
                elif current["status"] == JobStatus.LEASED.value:
                    cursor.execute(
                        f"""
                        UPDATE {self.schema}.processing_jobs
                        SET control_reason = %s,
                            updated_at = NOW()
                        WHERE id = %s AND (%s::uuid IS NULL OR project_id = %s::uuid)
                        RETURNING *;
                        """,
                        (f"pause_requested:{reason or 'user_requested'}", job_id, project_id, project_id),
                    )
                else:
                    cursor.execute(
                        f"""
                        SELECT *
                        FROM {self.schema}.processing_jobs
                        WHERE id = %s AND (%s::uuid IS NULL OR project_id = %s::uuid)
                        """,
                        (job_id, project_id, project_id),
                    )
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

    def pause_jobs(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        stages: Sequence[str] | None = None,
        job_ids: Sequence[str] | None = None,
        worker_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Pause matching queued work and cooperatively request pauses for running work."""
        clauses, params = self._job_filter_clauses(
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            stages=stages,
            job_ids=job_ids,
            worker_id=worker_id,
        )
        clauses.append("status IN ('queued', 'leased', 'working')")
        where = f"WHERE {' AND '.join(clauses)}"
        requested_reason = f"pause_requested:{reason or 'user_requested'}"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH matched AS (
                        SELECT id, status AS previous_status
                        FROM {self.schema}.processing_jobs
                        {where}
                    )
                    UPDATE {self.schema}.processing_jobs jobs
                    SET status = CASE WHEN matched.previous_status = 'queued'
                                      THEN 'paused'::{self.schema}.job_status
                                      ELSE jobs.status END,
                        control_reason = CASE WHEN matched.previous_status = 'queued' THEN %s ELSE %s END,
                        updated_at = NOW()
                    FROM matched
                    WHERE jobs.id = matched.id
                    RETURNING jobs.*, matched.previous_status;
                    """,
                    tuple([*params, reason, requested_reason]),
                )
                rows = cursor.fetchall()
                for row in rows:
                    was_queued = row["previous_status"] == JobStatus.QUEUED.value
                    self._append_job_event(
                        cursor,
                        row["id"],
                        "job.paused" if was_queued else "job.pause_requested",
                        {"reason": reason, "previous_status": row["previous_status"]},
                    )
            connection.commit()
        paused_count = sum(row["previous_status"] == JobStatus.QUEUED.value for row in rows)
        return {
            "matched_count": len(rows),
            "paused_count": paused_count,
            "pause_requested_count": len(rows) - paused_count,
            "jobs": rows,
        }

    def resume_jobs(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        stages: Sequence[str] | None = None,
        job_ids: Sequence[str] | None = None,
        worker_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Resume matching paused work as queued work."""
        clauses, params = self._job_filter_clauses(
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            stages=stages,
            job_ids=job_ids,
            worker_id=worker_id,
        )
        clauses.append("status = 'paused'")
        where = f"WHERE {' AND '.join(clauses)}"
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
                    {where}
                    RETURNING *;
                    """,
                    tuple([reason, *params]),
                )
                rows = cursor.fetchall()
                for row in rows:
                    self._append_job_event(cursor, row["id"], "job.resumed", {"reason": reason})
            connection.commit()
        return {"matched_count": len(rows), "resumed_count": len(rows), "jobs": rows}

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

    def resume_job(
        self,
        job_id: str,
        reason: str | None = None,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
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
                      AND (%s::uuid IS NULL OR project_id = %s::uuid)
                    RETURNING *;
                    """,
                    (reason, job_id, project_id, project_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(cursor, job_id, "job.resumed", {"reason": reason})
            connection.commit()
        return row

    def get_status_summary(self, *, project_id: str | None = None) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                job_clauses: list[str] = []
                job_params: list[Any] = []
                if project_id:
                    job_clauses.append("project_id = %s")
                    job_params.append(project_id)
                job_where = f"WHERE {' AND '.join(job_clauses)}" if job_clauses else ""
                cursor.execute(
                    f"""
                    SELECT status, COUNT(*) AS count
                    FROM {self.schema}.processing_jobs
                    {job_where}
                    GROUP BY status
                    """,
                    tuple(job_params),
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
                      AND status IN ('queued', 'leased', 'working')
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
                      AND status IN ('queued', 'leased', 'working')
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

    def retry_job(self, job_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
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
                      AND (%s::uuid IS NULL OR project_id = %s::uuid)
                    RETURNING *;
                    """,
                    (job_id, project_id, project_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    self._append_job_event(cursor, job_id, "job.retried", {})
            connection.commit()
        return row

    def cancel_jobs(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        statuses: Sequence[str] | None = None,
        stages: Sequence[str] | None = None,
        job_ids: Sequence[str] | None = None,
        worker_id: str | None = None,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        clauses, params = self._job_filter_clauses(
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            statuses=statuses,
            stages=stages,
            job_ids=job_ids,
            worker_id=worker_id,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        active_status_sql = "status IN ('queued', 'leased', 'working', 'paused')"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        COUNT(*)::bigint AS matched_count,
                        COUNT(*) FILTER (WHERE {active_status_sql})::bigint AS cancellable_count
                    FROM {self.schema}.processing_jobs
                    {where};
                    """,
                    tuple(params),
                )
                counts = cursor.fetchone() or {"matched_count": 0, "cancellable_count": 0}
                matched_count = int(counts["matched_count"] or 0)
                cancellable_count = int(counts["cancellable_count"] or 0)
                if dry_run or cancellable_count == 0:
                    connection.commit()
                    return {
                        "matched_count": matched_count,
                        "cancellable_count": cancellable_count,
                        "cancelled_count": 0,
                        "jobs": [],
                        "dry_run": bool(dry_run),
                    }

                update_clauses = [*clauses, active_status_sql]
                update_where = f"WHERE {' AND '.join(update_clauses)}"
                cursor.execute(
                    f"""
                    WITH matched AS (
                        SELECT id, status AS previous_status
                        FROM {self.schema}.processing_jobs
                        {update_where}
                    )
                    UPDATE {self.schema}.processing_jobs jobs
                    SET status = 'cancelled',
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        control_reason = %s,
                        finished_at = NOW(),
                        updated_at = NOW()
                    FROM matched
                    WHERE jobs.id = matched.id
                    RETURNING jobs.*, matched.previous_status;
                    """,
                    tuple([*params, reason]),
                )
                rows = cursor.fetchall()
                filters = {
                    "project_id": project_id,
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "statuses": list(statuses or []),
                    "stages": list(stages or []),
                    "job_ids": list(job_ids or []),
                    "worker_id": worker_id,
                }
                for row in rows:
                    self._append_job_event(
                        cursor,
                        row["id"],
                        "job.cancelled",
                        {
                            "reason": reason,
                            "bulk": True,
                            "previous_status": row.get("previous_status"),
                            "filters": json_ready(filters),
                        },
                    )
            connection.commit()
        return {
            "matched_count": matched_count,
            "cancellable_count": cancellable_count,
            "cancelled_count": len(rows),
            "jobs": rows,
            "dry_run": False,
        }

    def delete_jobs(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        statuses: Sequence[str] | None = None,
        stages: Sequence[str] | None = None,
        job_ids: Sequence[str] | None = None,
        worker_id: str | None = None,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        clauses, params = self._job_filter_clauses(
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            statuses=statuses,
            stages=stages,
            job_ids=job_ids,
            worker_id=worker_id,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT COUNT(*)::bigint AS matched_count
                    FROM {self.schema}.processing_jobs
                    {where};
                    """,
                    tuple(params),
                )
                counts = cursor.fetchone() or {"matched_count": 0}
                matched_count = int(counts["matched_count"] or 0)
                if dry_run or matched_count == 0:
                    connection.commit()
                    return {
                        "matched_count": matched_count,
                        "cancellable_count": 0,
                        "cancelled_count": 0,
                        "deleted_count": 0,
                        "jobs": [],
                        "dry_run": bool(dry_run),
                    }
                cursor.execute(
                    f"""
                    DELETE FROM {self.schema}.processing_jobs
                    {where}
                    RETURNING *;
                    """,
                    tuple(params),
                )
                rows = cursor.fetchall()
                self._append_log(
                    cursor,
                    event_type="jobs.deleted",
                    message=f"Deleted {len(rows)} job records.",
                    logger="pelagia.jobs",
                    project_id=project_id,
                    payload={
                        "reason": reason,
                        "bulk": True,
                        "deleted_count": len(rows),
                        "filters": {
                            "project_id": project_id,
                            "run_id": run_id,
                            "asset_id": asset_id,
                            "statuses": list(statuses or []),
                            "stages": list(stages or []),
                            "job_ids": list(job_ids or []),
                            "worker_id": worker_id,
                        },
                    },
                )
            connection.commit()
        return {
            "matched_count": matched_count,
            "cancellable_count": 0,
            "cancelled_count": 0,
            "deleted_count": len(rows),
            "jobs": rows,
            "dry_run": False,
        }

    def cancel_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
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
                    WHERE run_id = %s AND status IN ('queued', 'leased', 'working', 'paused')
                      AND (%s::uuid IS NULL OR project_id = %s::uuid)
                    RETURNING id, status
                    """,
                    (run_id, project_id, project_id),
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
                    WHERE id = %s AND (%s::uuid IS NULL OR project_id = %s::uuid)
                    RETURNING *;
                    """,
                    (run_id, project_id, project_id),
                )
                run_row = cursor.fetchone()
            connection.commit()
        return run_row

    def reconcile_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT status, COUNT(*) AS count
                    FROM {self.schema}.processing_jobs
                    WHERE run_id = %s AND (%s::uuid IS NULL OR project_id = %s::uuid)
                    GROUP BY status
                    """,
                    (run_id, project_id, project_id),
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
                    WHERE id = %s AND (%s::uuid IS NULL OR project_id = %s::uuid)
                    RETURNING *;
                    """,
                    (run_status, run_id, project_id, project_id),
                )
                run_row = cursor.fetchone()
            connection.commit()
        return run_row

    def ingest_telemetry(
        self,
        *,
        project_id: str,
        run_id: str,
        asset: Mapping[str, Any],
        source: Mapping[str, Any],
        parameters: Sequence[Mapping[str, Any]],
        sensors: Sequence[Mapping[str, Any]],
        streams: Sequence[Mapping[str, Any]],
        observations: Iterable[tuple[str, datetime, float, int | None]] | None = None,
    ) -> dict[str, Any]:
        """Atomically publish one standardized telemetry source and its observations."""
        resolved_project_id = self._required_project_id(project_id, "ingest_telemetry")
        import_key = str(source.get("import_key") or "").strip()
        source_payload_key = str(source.get("source_payload_key") or "").strip()
        if not import_key:
            raise ValueError("Telemetry source import key must not be blank.")
        if not source_payload_key:
            raise ValueError("Telemetry source payload key must not be blank.")
        schema = self.schema
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT id FROM {schema}.runs WHERE id = %s AND project_id = %s",
                    (run_id, resolved_project_id),
                )
                if cursor.fetchone() is None:
                    raise ValueError(f"Run {run_id!r} was not found in the selected project.")

                # Serialize exact retries before creating any catalog rows. The
                # content-addressed source snapshot is safe to write before this
                # database transaction and is reused by every contender.
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"telemetry:{resolved_project_id}:{run_id}:{import_key}",),
                )
                cursor.execute(
                    f"""SELECT * FROM {schema}.telemetry_sources
                        WHERE project_id = %s AND run_id = %s AND import_key = %s""",
                    (resolved_project_id, run_id, import_key),
                )
                existing_source = cursor.fetchone()
                if existing_source is not None:
                    cursor.execute(
                        f"SELECT * FROM {schema}.raw_assets WHERE id = %s AND project_id = %s",
                        (existing_source["raw_asset_id"], resolved_project_id),
                    )
                    existing_asset = cursor.fetchone()
                    cursor.execute(
                        f"SELECT * FROM {schema}.telemetry_streams WHERE source_id = %s ORDER BY id",
                        (existing_source["id"],),
                    )
                    return {
                        "asset": existing_asset,
                        "source": existing_source,
                        "streams": cursor.fetchall(),
                    }

                cursor.execute(
                    f"""
                    INSERT INTO {schema}.raw_assets
                    (id, project_id, run_id, filename, path, kind, checksum, size_bytes,
                     collections, media_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, 'telemetry'::{schema}.asset_kind, %s, %s,
                            %s, 1, %s::jsonb)
                    RETURNING *
                    """,
                    (
                        asset["id"], resolved_project_id, run_id, asset["filename"], asset["path"],
                        asset["checksum"], int(asset["size_bytes"]),
                        normalize_collections(asset.get("collections")),
                        json.dumps(json_ready(asset.get("metadata") or {})),
                    ),
                )
                asset_row = cursor.fetchone()

                parameter_ids: dict[str, str] = {}
                for item in sorted(parameters, key=lambda value: str(value["parameter_key"])):
                    key = str(item["parameter_key"])
                    cursor.execute(
                        f"""
                        INSERT INTO {schema}.telemetry_parameters
                        (project_id, parameter_key, display_name, definition, standard_name,
                         canonical_unit, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (project_id, parameter_key) DO NOTHING
                        RETURNING id, canonical_unit
                        """,
                        (
                            resolved_project_id, key, item.get("display_name"), item.get("definition"),
                            item.get("standard_name"), item["canonical_unit"],
                            json.dumps(json_ready(item.get("metadata") or {})),
                        ),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        cursor.execute(
                            f"""SELECT id, canonical_unit, definition, standard_name
                                FROM {schema}.telemetry_parameters
                                WHERE project_id = %s AND parameter_key = %s""",
                            (resolved_project_id, key),
                        )
                        row = cursor.fetchone()
                    if row["canonical_unit"] != item["canonical_unit"]:
                        raise ValueError(
                            f"Parameter {key!r} already uses canonical unit {row['canonical_unit']!r}."
                        )
                    for field in ("definition", "standard_name"):
                        if row.get(field) and item.get(field) and row[field] != item[field]:
                            raise ValueError(
                                f"Parameter {key!r} already has a different {field.replace('_', ' ')}."
                            )
                    parameter_ids[key] = str(row["id"])

                sensor_ids: dict[str, str] = {}
                for item in sorted(sensors, key=lambda value: str(value["sensor_key"])):
                    key = str(item["sensor_key"])
                    cursor.execute(
                        f"""
                        INSERT INTO {schema}.telemetry_sensors
                        (project_id, sensor_key, display_name, manufacturer, model, serial_number, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (project_id, sensor_key) DO NOTHING
                        RETURNING id
                        """,
                        (
                            resolved_project_id, key, item.get("display_name"), item.get("manufacturer"),
                            item.get("model"), item.get("serial_number"),
                            json.dumps(json_ready(item.get("metadata") or {})),
                        ),
                    )
                    sensor_row = cursor.fetchone()
                    if sensor_row is None:
                        cursor.execute(
                            f"""SELECT id, manufacturer, model, serial_number
                                FROM {schema}.telemetry_sensors
                                WHERE project_id = %s AND sensor_key = %s
                                FOR UPDATE""",
                            (resolved_project_id, key),
                        )
                        sensor_row = cursor.fetchone()
                    if sensor_row is None:
                        raise ValueError(f"Sensor {key!r} could not be created.")
                    for field in ("manufacturer", "model", "serial_number"):
                        if (
                            sensor_row.get(field) and item.get(field)
                            and sensor_row[field] != item[field]
                        ):
                            raise ValueError(
                                f"Sensor {key!r} already has a different {field.replace('_', ' ')}."
                            )
                    sensor_ids[key] = str(sensor_row["id"])

                default_streams: dict[str, list[str]] = {}
                for item in streams:
                    parameter_key = str(item["parameter_key"])
                    if parameter_key not in parameter_ids:
                        raise ValueError(
                            f"Telemetry stream {item['stream_key']!r} references unknown "
                            f"parameter {parameter_key!r}."
                        )
                    if bool(item.get("is_default", False)):
                        default_streams.setdefault(parameter_key, []).append(str(item["stream_key"]))

                for parameter_key in sorted(default_streams):
                    requested = default_streams[parameter_key]
                    if len(requested) > 1:
                        raise ValueError(
                            f"Parameter {parameter_key!r} cannot define more than one default "
                            "telemetry stream for this run."
                        )
                    # A partial unique index is the final integrity boundary, but
                    # serializing selection here turns a concurrent contender into
                    # a stable domain conflict instead of a driver-level unique
                    # violation. Lock keys are acquired in sorted order above.
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (
                            f"telemetry-default:{resolved_project_id}:{run_id}:"
                            f"{parameter_key}",
                        ),
                    )
                    cursor.execute(
                        f"""SELECT stream_key FROM {schema}.telemetry_streams
                            WHERE project_id = %s AND run_id = %s AND parameter_id = %s
                              AND is_default
                            LIMIT 1""",
                        (resolved_project_id, run_id, parameter_ids[parameter_key]),
                    )
                    existing_default = cursor.fetchone()
                    if existing_default is not None:
                        raise ValueError(
                            f"Parameter {parameter_key!r} already has default telemetry stream "
                            f"{existing_default['stream_key']!r} for this run."
                        )

                cursor.execute(
                    f"""
                    INSERT INTO {schema}.telemetry_sources
                    (project_id, run_id, raw_asset_id, format, parser_name, parser_version,
                     import_key, source_payload_key, import_status, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'importing', %s::jsonb)
                    RETURNING *
                    """,
                    (
                        resolved_project_id, run_id, asset["id"], source["format"],
                        source["parser_name"], source["parser_version"],
                        import_key, source_payload_key,
                        json.dumps(json_ready(source.get("metadata") or {})),
                    ),
                )
                source_row = cursor.fetchone()

                stream_rows = []
                stream_ids: dict[str, int] = {}
                for item in streams:
                    cursor.execute(
                        f"""
                        INSERT INTO {schema}.telemetry_streams
                        (project_id, run_id, source_id, sensor_id, parameter_id, stream_key,
                         native_unit, sampling_rate_hz, interpolation, max_gap, priority,
                         is_default, qc_scheme, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s * interval '1 second', %s, %s, %s, %s::jsonb)
                        RETURNING *
                        """,
                        (
                            resolved_project_id, run_id, source_row["id"],
                            sensor_ids[str(item["sensor_key"])],
                            parameter_ids[str(item["parameter_key"])], item["stream_key"],
                            item["native_unit"], item.get("sampling_rate_hz"),
                            item.get("interpolation", "none"), item.get("max_gap_seconds"),
                            int(item.get("priority", 100)), bool(item.get("is_default", False)),
                            item.get("qc_scheme"), json.dumps(json_ready(item.get("metadata") or {})),
                        ),
                    )
                    stream_row = cursor.fetchone()
                    stream_rows.append(stream_row)
                    stream_ids[str(item["stream_key"])] = int(stream_row["id"])

                source_observations = observations
                if source_observations is None:
                    source_observations = (
                        (str(item["stream_key"]), observed_at, float(value), qc_flag)
                        for item in streams
                        for observed_at, value, qc_flag in item.get("observations", ())
                    )
                observation_count = 0
                observed_start_at: datetime | None = None
                observed_end_at: datetime | None = None
                if streams:
                    copy_sql = f"COPY {schema}.telemetry_observations (stream_id, observed_at, value, qc_flag) FROM STDIN"
                    with cursor.copy(copy_sql) as copy:
                        for stream_key, observed_at, value, qc_flag in source_observations:
                            if stream_key not in stream_ids:
                                raise ValueError(f"Unknown stream key {stream_key!r} in observations.")
                            copy.write_row((stream_ids[stream_key], observed_at, float(value), qc_flag))
                            observation_count += 1
                            observed_start_at = (
                                observed_at if observed_start_at is None else min(observed_start_at, observed_at)
                            )
                            observed_end_at = (
                                observed_at if observed_end_at is None else max(observed_end_at, observed_at)
                            )

                cursor.execute(
                    f"""
                    UPDATE {schema}.telemetry_sources
                    SET import_status = 'ready', observed_start_at = %s, observed_end_at = %s,
                        observation_count = %s, imported_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (observed_start_at, observed_end_at, observation_count, source_row["id"]),
                )
                source_row = cursor.fetchone()
            connection.commit()
        return {"asset": asset_row, "source": source_row, "streams": stream_rows}

    def get_telemetry_import(
        self, *, project_id: str, run_id: str, import_key: str,
    ) -> dict[str, Any] | None:
        """Return a ready import scoped to one project and run."""
        resolved_project_id = self._required_project_id(project_id, "get_telemetry_import")
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT * FROM {self.schema}.telemetry_sources
                    WHERE project_id = %s AND run_id = %s AND import_key = %s
                      AND import_status = 'ready'""",
                (resolved_project_id, run_id, import_key),
            )
            source_row = cursor.fetchone()
            if source_row is None:
                return None
            cursor.execute(
                f"SELECT * FROM {self.schema}.raw_assets WHERE id = %s AND project_id = %s",
                (source_row["raw_asset_id"], resolved_project_id),
            )
            asset_row = cursor.fetchone()
            cursor.execute(
                f"SELECT * FROM {self.schema}.telemetry_streams WHERE source_id = %s ORDER BY id",
                (source_row["id"],),
            )
            return {"asset": asset_row, "source": source_row, "streams": cursor.fetchall()}

    def list_telemetry_sources(self, *, project_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        clauses = ["sources.project_id = %s"]
        params: list[Any] = [project_id]
        if run_id is not None:
            clauses.append("sources.run_id = %s")
            params.append(run_id)
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT sources.*, assets.filename, assets.path, assets.checksum, assets.size_bytes
                    FROM {self.schema}.telemetry_sources sources
                    JOIN {self.schema}.raw_assets assets ON assets.id = sources.raw_asset_id
                    WHERE {' AND '.join(clauses)} ORDER BY sources.created_at DESC""",
                tuple(params),
            )
            return cursor.fetchall()

    def list_telemetry_parameters(self, *, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM {self.schema}.telemetry_parameters WHERE project_id = %s ORDER BY parameter_key",
                (project_id,),
            )
            return cursor.fetchall()

    def list_telemetry_sensors(self, *, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM {self.schema}.telemetry_sensors WHERE project_id = %s ORDER BY sensor_key",
                (project_id,),
            )
            return cursor.fetchall()

    def list_telemetry_streams(
        self, *, project_id: str, run_id: str | None = None, parameter_keys: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["streams.project_id = %s"]
        params: list[Any] = [project_id]
        if run_id is not None:
            clauses.append("streams.run_id = %s")
            params.append(run_id)
        if parameter_keys:
            clauses.append("parameters.parameter_key = ANY(%s)")
            params.append(list(parameter_keys))
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT streams.*, parameters.parameter_key, parameters.canonical_unit,
                           sensors.sensor_key, sensors.display_name AS sensor_display_name
                    FROM {self.schema}.telemetry_streams streams
                    JOIN {self.schema}.telemetry_parameters parameters ON parameters.id = streams.parameter_id
                    JOIN {self.schema}.telemetry_sensors sensors ON sensors.id = streams.sensor_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY streams.priority, parameters.parameter_key, streams.stream_key""",
                tuple(params),
            )
            return cursor.fetchall()

    def get_telemetry_stream(
        self, stream_id: str | int, *, project_id: str, run_id: str | None = None,
    ) -> dict[str, Any] | None:
        identifier = str(stream_id)
        if identifier.isdecimal():
            identifier_clause = "streams.id = %s"
            identifier_value: Any = int(identifier)
        else:
            try:
                import uuid

                identifier_value = uuid.UUID(identifier)
            except ValueError:
                return None
            identifier_clause = "streams.public_id = %s"
        with self.connect() as connection, connection.cursor() as cursor:
            run_clause = " AND streams.run_id = %s" if run_id is not None else ""
            cursor.execute(
                f"""SELECT streams.*, parameters.parameter_key, parameters.canonical_unit,
                           sensors.sensor_key
                    FROM {self.schema}.telemetry_streams streams
                    JOIN {self.schema}.telemetry_parameters parameters ON parameters.id = streams.parameter_id
                    JOIN {self.schema}.telemetry_sensors sensors ON sensors.id = streams.sensor_id
                    WHERE streams.project_id = %s AND {identifier_clause}{run_clause}""",
                (project_id, identifier_value, run_id) if run_id is not None else (project_id, identifier_value),
            )
            return cursor.fetchone()

    def telemetry_observations_around(
        self, *, project_id: str, stream_id: int, observed_at: datetime,
        excluded_qc_flags: Sequence[int] | None = None,
    ) -> dict[str, dict[str, Any] | None]:
        stream = self.get_telemetry_stream(stream_id, project_id=project_id)
        if stream is None:
            return {"previous": None, "next": None}
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""(SELECT * FROM {self.schema}.telemetry_observations
                       WHERE stream_id = %s AND observed_at <= %s ORDER BY observed_at DESC LIMIT 1)
                    UNION ALL
                    (SELECT * FROM {self.schema}.telemetry_observations
                     WHERE stream_id = %s AND observed_at > %s ORDER BY observed_at ASC LIMIT 1)""",
                (stream["id"], observed_at, stream["id"], observed_at),
            )
            rows = cursor.fetchall()
        previous = next((row for row in rows if row["observed_at"] <= observed_at), None)
        following = next((row for row in rows if row["observed_at"] > observed_at), None)
        # Fetching the valid brackets separately keeps QC filtering out of the
        # raw bracket query: an exact excluded observation must remain visible
        # to callers, while previous/nearest interpolation can skip over it.
        excluded_qc_flags = tuple(excluded_qc_flags or ())
        valid = {"previous": previous, "next": following}
        if excluded_qc_flags:
            with self.connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"""(SELECT * FROM {self.schema}.telemetry_observations
                           WHERE stream_id = %s AND observed_at <= %s
                             AND (qc_flag IS NULL OR NOT (qc_flag = ANY(%s)))
                           ORDER BY observed_at DESC LIMIT 1)
                        UNION ALL
                        (SELECT * FROM {self.schema}.telemetry_observations
                         WHERE stream_id = %s AND observed_at > %s
                           AND (qc_flag IS NULL OR NOT (qc_flag = ANY(%s)))
                         ORDER BY observed_at ASC LIMIT 1)""",
                    (stream["id"], observed_at, list(excluded_qc_flags),
                     stream["id"], observed_at, list(excluded_qc_flags)),
                )
                valid_rows = cursor.fetchall()
            valid["previous_valid"] = next(
                (row for row in valid_rows if row["observed_at"] <= observed_at), None
            )
            valid["next_valid"] = next(
                (row for row in valid_rows if row["observed_at"] > observed_at), None
            )
        return {"previous": previous, "next": following, **valid}

    def list_telemetry_observations(
        self, *, project_id: str, stream_id: int, start_at: datetime | None = None,
        end_at: datetime | None = None, run_id: str | None = None,
        limit: int | None = None, offset: int = 0,
    ) -> list[dict[str, Any]]:
        if limit is not None and limit < 1:
            raise ValueError("Telemetry observation limit must be positive.")
        if offset < 0:
            raise ValueError("Telemetry observation offset must not be negative.")
        stream = self.get_telemetry_stream(stream_id, project_id=project_id, run_id=run_id)
        if stream is None:
            return []
        clauses = ["stream_id = %s"]
        params: list[Any] = [stream["id"]]
        if start_at is not None:
            clauses.append("observed_at >= %s")
            params.append(start_at)
        if end_at is not None:
            clauses.append("observed_at <= %s")
            params.append(end_at)
        with self.connect() as connection, connection.cursor() as cursor:
            pagination = ""
            if limit is not None:
                pagination = " LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            cursor.execute(
                f"SELECT * FROM {self.schema}.telemetry_observations WHERE {' AND '.join(clauses)} ORDER BY observed_at{pagination}",
                tuple(params),
            )
            return cursor.fetchall()

    def create_timeline_event_type(
        self, *, project_id: str, event_type_key: str, display_name: str | None = None,
        description: str | None = None, metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO {self.schema}.timeline_event_types
                    (project_id, event_type_key, display_name, description, metadata)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (project_id, event_type_key) DO NOTHING
                    RETURNING *""",
                (project_id, event_type_key, display_name, description, json.dumps(json_ready(metadata or {}))),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    f"""SELECT * FROM {self.schema}.timeline_event_types
                        WHERE project_id = %s AND event_type_key = %s""",
                    (project_id, event_type_key),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("Timeline event type could not be created.")
                for field_name in ("display_name", "description"):
                    supplied = locals()[field_name]
                    if supplied is not None and row.get(field_name) not in (None, supplied):
                        raise ValueError(
                            f"Timeline event type {event_type_key!r} already has a different {field_name}."
                        )
                if metadata and dict(row.get("metadata") or {}) != dict(metadata):
                    raise ValueError(
                        f"Timeline event type {event_type_key!r} already has different metadata."
                    )
            connection.commit()
            return row

    def list_timeline_event_types(self, *, project_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM {self.schema}.timeline_event_types WHERE project_id = %s ORDER BY event_type_key",
                (project_id,),
            )
            return cursor.fetchall()

    def create_timeline_event(
        self, *, project_id: str, run_id: str, event_type_id: str, start_at: datetime,
        end_at: datetime | None = None, source_id: str | None = None,
        value: str | None = None, created_by: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if start_at.tzinfo is None or start_at.utcoffset() is None:
            raise ValueError("Timeline event start_at must include a timezone.")
        if end_at is not None and (end_at.tzinfo is None or end_at.utcoffset() is None):
            raise ValueError("Timeline event end_at must include a timezone.")
        start_at = start_at.astimezone(timezone.utc)
        end_at = None if end_at is None else end_at.astimezone(timezone.utc)
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO {self.schema}.timeline_events
                    (project_id, run_id, event_type_id, start_at, end_at, source_id, value, created_by, metadata)
                    SELECT %s, runs.id, types.id, %s, %s, %s, %s, %s, %s::jsonb
                    FROM {self.schema}.runs runs
                    JOIN {self.schema}.timeline_event_types types ON types.id = %s AND types.project_id = runs.project_id
                    LEFT JOIN {self.schema}.telemetry_sources sources
                        ON sources.id = %s AND sources.project_id = runs.project_id AND sources.run_id = runs.id
                    WHERE runs.id = %s AND runs.project_id = %s
                      AND (%s::uuid IS NULL OR sources.id IS NOT NULL)
                    RETURNING *""",
                (
                    project_id, start_at, end_at, source_id, value, created_by,
                    json.dumps(json_ready(metadata or {})), event_type_id, source_id, run_id, project_id,
                    source_id,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError("Run or event type was not found in the selected project.")
            connection.commit()
            return row

    def get_timeline_event(
        self, *, project_id: str, run_id: str, event_id: str,
    ) -> dict[str, Any] | None:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT events.*, types.event_type_key, types.display_name
                    FROM {self.schema}.timeline_events events
                    JOIN {self.schema}.timeline_event_types types ON types.id = events.event_type_id
                    WHERE events.id = %s AND events.project_id = %s AND events.run_id = %s""",
                (event_id, project_id, run_id),
            )
            return cursor.fetchone()

    def list_timeline_events(
        self, *, project_id: str, run_id: str, start_at: datetime | None = None,
        end_at: datetime | None = None, event_type_id: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["events.project_id = %s", "events.run_id = %s"]
        params: list[Any] = [project_id, run_id]
        if start_at is not None and end_at is not None:
            clauses.append(
                "tstzrange(events.start_at, COALESCE(events.end_at, events.start_at), '[]') "
                "&& tstzrange(%s::timestamptz, %s::timestamptz, '[]')"
            )
            params.extend((start_at, end_at))
        elif start_at is not None:
            clauses.append(
                "tstzrange(events.start_at, COALESCE(events.end_at, events.start_at), '[]') "
                "&& tstzrange(%s::timestamptz, NULL, '[)')"
            )
            params.append(start_at)
        elif end_at is not None:
            clauses.append(
                "tstzrange(events.start_at, COALESCE(events.end_at, events.start_at), '[]') "
                "&& tstzrange(NULL, %s::timestamptz, '(]')"
            )
            params.append(end_at)
        if event_type_id is not None:
            clauses.append("events.event_type_id = %s")
            params.append(event_type_id)
        if source_id is not None:
            clauses.append("events.source_id = %s")
            params.append(source_id)
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT events.*, types.event_type_key, types.display_name
                    FROM {self.schema}.timeline_events events
                    JOIN {self.schema}.timeline_event_types types ON types.id = events.event_type_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY events.start_at, types.event_type_key, events.id""",
                tuple(params),
            )
            return cursor.fetchall()

    def update_timeline_event(
        self, *, project_id: str, run_id: str, event_id: str,
        updates: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        allowed = {"event_type_id", "source_id", "start_at", "end_at", "value", "metadata"}
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unsupported timeline event fields: {', '.join(sorted(unknown))}.")
        if not updates:
            raise ValueError("Provide at least one timeline event field to update.")

        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT * FROM {self.schema}.timeline_events
                    WHERE id = %s AND project_id = %s AND run_id = %s FOR UPDATE""",
                (event_id, project_id, run_id),
            )
            current = cursor.fetchone()
            if current is None:
                return None

            resolved = dict(updates)
            start_at = resolved.get("start_at", current["start_at"])
            end_at = resolved.get("end_at", current["end_at"])
            if start_at.tzinfo is None or start_at.utcoffset() is None:
                raise ValueError("Timeline event start_at must include a timezone.")
            if end_at is not None and (end_at.tzinfo is None or end_at.utcoffset() is None):
                raise ValueError("Timeline event end_at must include a timezone.")
            if end_at is not None and end_at < start_at:
                raise ValueError("Timeline event end_at must not precede start_at.")
            if "event_type_id" in resolved:
                cursor.execute(
                    f"""SELECT 1 FROM {self.schema}.timeline_event_types
                        WHERE id = %s AND project_id = %s""",
                    (resolved["event_type_id"], project_id),
                )
                if cursor.fetchone() is None:
                    raise ValueError("Timeline event type was not found in the selected project.")
            if resolved.get("source_id") is not None:
                cursor.execute(
                    f"""SELECT 1 FROM {self.schema}.telemetry_sources
                        WHERE id = %s AND project_id = %s AND run_id = %s""",
                    (resolved["source_id"], project_id, run_id),
                )
                if cursor.fetchone() is None:
                    raise ValueError("Telemetry source was not found for the selected run and project.")

            assignments: list[str] = []
            params: list[Any] = []
            for field_name in ("event_type_id", "source_id", "start_at", "end_at", "value", "metadata"):
                if field_name not in resolved:
                    continue
                if field_name == "metadata":
                    assignments.append("metadata = %s::jsonb")
                    params.append(json.dumps(json_ready(resolved[field_name] or {})))
                else:
                    assignments.append(f"{field_name} = %s")
                    params.append(resolved[field_name])
            assignments.append("updated_at = NOW()")
            params.extend([event_id, project_id, run_id])
            cursor.execute(
                f"""UPDATE {self.schema}.timeline_events
                    SET {', '.join(assignments)}
                    WHERE id = %s AND project_id = %s AND run_id = %s
                    RETURNING *""",
                tuple(params),
            )
            row = cursor.fetchone()
            connection.commit()
            return row

    def delete_timeline_event(
        self, *, project_id: str, run_id: str, event_id: str,
    ) -> dict[str, Any] | None:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""DELETE FROM {self.schema}.timeline_events
                    WHERE id = %s AND project_id = %s AND run_id = %s
                    RETURNING *""",
                (event_id, project_id, run_id),
            )
            row = cursor.fetchone()
            connection.commit()
            return row

    def list_timeline_events_at(
        self, *, project_id: str, run_id: str, observed_at: datetime,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT events.*, types.event_type_key, types.display_name
                    FROM {self.schema}.timeline_events events
                    JOIN {self.schema}.timeline_event_types types ON types.id = events.event_type_id
                    WHERE events.project_id = %s AND events.run_id = %s
                      AND tstzrange(
                            events.start_at, COALESCE(events.end_at, events.start_at), '[]'
                          ) @> %s::timestamptz
                    ORDER BY events.start_at, types.event_type_key""",
                (project_id, run_id, observed_at),
            )
            return cursor.fetchall()
