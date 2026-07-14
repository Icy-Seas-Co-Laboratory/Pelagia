from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
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
    "schema_migrations",
    "users",
    "projects",
    "project_memberships",
    "user_sessions",
    "runs",
    "raw_assets",
    "frames",
    "detection_candidate",
    "detections_refined",
    "models",
    "classification_results",
    "processing_jobs",
    "processing_job_dependencies",
    "project_processing_status_snapshots",
    "frame_processing_status",
    "worker_sessions",
    "job_events",
    "logs",
)

DEFAULT_PROJECT_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_PROJECT_KEY = "default"
PROJECT_ROLES = {"viewer", "editor", "manager", "admin"}
FRAME_PROCESSING_STATUSES = {"unknown", "queued", "leased", "working", "succeeded", "failed", "cancelled", "dead_lettered"}
PASSWORD_HASH_ITERATIONS = 260_000
DEFAULT_SESSION_TTL_SECONDS = 7 * 24 * 60 * 60


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
        # Scoped views are the preferred application-facing dependencies.  Keep
        # this facade intact while the underlying SQL moves out incrementally.
        from .scoped import CatalogRepository, FrameRepository, IdentityRepository, JobRepository

        self.identity = IdentityRepository(self)
        self.catalog = CatalogRepository(self)
        self.frames = FrameRepository(self)
        self.jobs = JobRepository(self)

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

    def purge_all(self) -> dict[str, Any]:
        """Delete all Pelagia rows while preserving the schema, indexes, and functions."""
        tables = [table for table in REQUIRED_SCHEMA_TABLES if table != "schema_migrations"]
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
            "preserved_tables": ["schema_migrations"],
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
                     background_metadata, metadata)
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
        refinement_state: str | None = None,
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

    def get_detection(self, detection_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                clauses = ["detections.id = %s"]
                params: list[Any] = [detection_id]
                if project_id:
                    clauses.append("assets.project_id = %s")
                    params.append(project_id)
                cursor.execute(
                    f"""
                    SELECT
                        detections.*,
                        frames.asset_id,
                        frames.frame_index,
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
                return cursor.fetchone()

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
        collection: str | None = None,
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
        if collection:
            clauses.append("%s = ANY(status.collections)")
            params.append(collection)
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
                    WITH filtered AS (
                        SELECT *
                        FROM {self.schema}.frame_processing_status status
                        WHERE {' AND '.join(clauses)}
                    ),
                    totals AS (
                        SELECT
                            COUNT(*)::bigint AS total_frame_count,
                            COUNT(*) FILTER (WHERE preprocessing_status = 'succeeded')::bigint AS preprocessing_succeeded_count,
                            COUNT(*) FILTER (WHERE candidate_detection_status = 'succeeded')::bigint AS candidate_detection_succeeded_count,
                            COUNT(*) FILTER (WHERE roi_refinement_status = 'succeeded')::bigint AS roi_refinement_succeeded_count,
                            COUNT(*) FILTER (WHERE candidate_detection_count > 0)::bigint AS frames_with_candidates_count,
                            COUNT(*) FILTER (WHERE refined_detection_count > 0)::bigint AS frames_with_refined_rois_count,
                            COALESCE(SUM(candidate_detection_count), 0)::bigint AS candidate_detection_count,
                            COALESCE(SUM(refined_detection_count), 0)::bigint AS refined_detection_count,
                            COALESCE(SUM(unrefined_candidate_count), 0)::bigint AS unrefined_candidate_count,
                            MAX(updated_at) AS updated_at
                        FROM filtered
                    ),
                    preprocessing_counts AS (
                        SELECT preprocessing_status AS status, COUNT(*)::bigint AS frame_count
                        FROM filtered
                        GROUP BY preprocessing_status
                    ),
                    candidate_detection_counts AS (
                        SELECT candidate_detection_status AS status, COUNT(*)::bigint AS frame_count
                        FROM filtered
                        GROUP BY candidate_detection_status
                    ),
                    roi_refinement_counts AS (
                        SELECT roi_refinement_status AS status, COUNT(*)::bigint AS frame_count
                        FROM filtered
                        GROUP BY roi_refinement_status
                    )
                    SELECT
                        totals.*,
                        jsonb_build_object(
                            'preprocessing',
                            COALESCE((SELECT jsonb_object_agg(status, frame_count) FROM preprocessing_counts), '{{}}'::jsonb),
                            'candidate_detection',
                            COALESCE((SELECT jsonb_object_agg(status, frame_count) FROM candidate_detection_counts), '{{}}'::jsonb),
                            'roi_refinement',
                            COALESCE((SELECT jsonb_object_agg(status, frame_count) FROM roi_refinement_counts), '{{}}'::jsonb)
                        ) AS by_status
                    FROM totals
                    """,
                    tuple(params),
                )
                row = cursor.fetchone() or {}
        by_status = row.get("by_status") or {}
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
                    (project_id, run_id, asset_id, stage, status, priority, attempt_count, max_attempts, payload, summary)
                    VALUES (%s, %s, %s, %s::{self.schema}.stage_name, %s::{self.schema}.job_status, %s, 0, %s, %s::jsonb, %s)
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
                        "project_id": row.get("project_id"),
                        "run_id": row.get("run_id"),
                        "asset_id": row.get("asset_id"),
                        "priority": row.get("priority"),
                        "depends_on": [str(dependency) for dependency in depends_on or []],
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
            project_id=resolved_project_id, run_id=filters.get("run_id"), asset_id=filters.get("asset_id"), collection=None,
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
                    cursor.execute(
                        f"""
                        INSERT INTO {self.schema}.processing_jobs
                            (project_id, run_id, asset_id, stage, status, priority, attempt_count, max_attempts, payload, summary)
                        VALUES (%s, %s, %s, 'preprocess_frames'::{self.schema}.stage_name,
                                'queued'::{self.schema}.job_status, %s, 0, %s, %s::jsonb, %s)
                        RETURNING *
                        """,
                        (
                            resolved_project_id,
                            job.get("run_id"),
                            job.get("asset_id"),
                            priority if priority is not None else self.config.queue.default_priority,
                            self.config.queue.max_attempts,
                            json.dumps(json_ready(job["payload"])),
                            job["summary"],
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
