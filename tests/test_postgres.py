from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import sqlite3
from threading import Barrier
import uuid

import pytest

from Pelagia.config import CoreConfig
from Pelagia.domain import (
    AssetKind,
    DetectionRecord,
    FrameRecord,
    JobStatus,
    ModelRecord,
    PlannedRun,
    PipelineStage,
    RawAssetManifest,
    RunManifest,
    WorkItem,
)
from Pelagia.storage import postgres
from Pelagia.storage.postgres import DEFAULT_PROJECT_ID, PostgresRepository, hash_session_token
from Pelagia.services.taxonomy import default_taxonomy_dictionary
from Pelagia.services.registry_transfer import export_sqlite_workspace, load_sqlite_workspace
from Pelagia.services.registry_workspace import RegistryWorkspaceService
from Pelagia.services.telemetry import TelemetryColumn, TelemetryCsvSpec, TelemetryIngestionService, TelemetryResolver
from oracle_data_contracts.datasets import initialize_database, validate_database


POSTGRES_TEST_DSN = os.getenv(
    "PELAGIA_TEST_DATABASE_DSN",
    "postgresql://postgres:postgres@localhost:5432/pelagia",
)


class _MemoryBlobStore:
    def __init__(self):
        self.payloads = {}

    def put_store(self, payload):
        payload = bytes(payload)
        key = hashlib.sha256(payload).hexdigest()
        self.payloads[key] = payload
        return key

    def get_store(self, key):
        return self.payloads[key]


class _DetectionInsertCursor:
    def __init__(self):
        self.executions = []
        self.fetchall_calls = 0
        self._result_sets = []
        self._rows = []
        self._result_index = 0

    def executemany(self, statement, parameters, *, returning=False):
        parameters = list(parameters)
        self.executions.append((statement, parameters))
        self._result_sets = [
            [{"id": f"det-{len(self.executions)}-{index}"}]
            for index in range(len(parameters))
        ] if returning else []
        self._result_index = 0
        self._rows = self._result_sets[0] if self._result_sets else []

    def nextset(self):
        self._result_index += 1
        if self._result_index >= len(self._result_sets):
            return None
        self._rows = self._result_sets[self._result_index]
        return True

    def fetchall(self):
        self.fetchall_calls += 1
        return self._rows


class _ExistingRowCursor:
    def __init__(self, row):
        self.row = row
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, parameters=()):
        self.executions.append((statement, parameters))

    def fetchone(self):
        return self.row


class _ExistingRowConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1


class _RefinedRetryCursor:
    def __init__(self):
        self.executions = []
        self.rows = {}
        self.insert_count = 0
        self._current = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, parameters=()):
        self.executions.append((statement, parameters))
        normalized = " ".join(statement.split())
        if normalized.startswith("SELECT pg_advisory_xact_lock"):
            self._current = None
            return
        if normalized.startswith("SELECT * FROM"):
            job_id, candidate_detection_id, roi_index = parameters
            self._current = self.rows.get(
                (str(job_id), str(candidate_detection_id), int(roi_index))
            )
            return
        if normalized.startswith("INSERT INTO"):
            candidate_detection_id, job_id, _, _, roi_index = parameters[:5]
            key = (str(job_id), str(candidate_detection_id), int(roi_index))
            self.insert_count += 1
            self._current = {
                "id": f"refined-{self.insert_count}",
                "job_id": job_id,
                "candidate_detection_id": candidate_detection_id,
                "roi_index": roi_index,
            }
            self.rows[key] = self._current
            return
        raise AssertionError(f"Unexpected SQL in retry test: {normalized}")

    def fetchone(self):
        return self._current


def _unit_detection(index: int, *, payload_bytes: int = 1) -> DetectionRecord:
    return DetectionRecord(
        run_id="run-1",
        frame_id=f"frame-{index}",
        roi_index=index,
        bbox_x=0,
        bbox_y=0,
        bbox_w=1,
        bbox_h=1,
        area=1.0,
        perimeter=4.0,
        major_axis_length=1.0,
        minor_axis_length=1.0,
        min_gray_value=0,
        mean_gray_value=0.0,
        roi_payload=b"r" * payload_bytes,
        mask_payload=b"m" * payload_bytes,
        metadata={"index": index},
    )


def test_detection_insert_batches_by_row_count_and_returns_ids_only():
    repository = object.__new__(PostgresRepository)
    repository.schema = "pelagia_unit"
    cursor = _DetectionInsertCursor()

    rows = repository._insert_detection_rows(
        cursor,
        "run-1",
        [_unit_detection(index) for index in range(5)],
        returning="ids",
        batch_rows=2,
        batch_bytes=1024 * 1024,
    )

    assert [len(parameters) for _, parameters in cursor.executions] == [2, 2, 1]
    assert all("RETURNING id" in statement for statement, _ in cursor.executions)
    assert all("RETURNING *" not in statement for statement, _ in cursor.executions)
    assert len(rows) == 5


def test_detection_insert_batches_by_payload_bytes_and_can_skip_returning():
    repository = object.__new__(PostgresRepository)
    repository.schema = "pelagia_unit"
    cursor = _DetectionInsertCursor()

    rows = repository._insert_detection_rows(
        cursor,
        "run-1",
        [_unit_detection(index, payload_bytes=4_000) for index in range(2)],
        returning="none",
        batch_rows=250,
        batch_bytes=9_000,
    )

    assert [len(parameters) for _, parameters in cursor.executions] == [1, 1]
    assert all("RETURNING" not in statement for statement, _ in cursor.executions)
    assert cursor.fetchall_calls == 0
    assert rows == []


def test_refinement_retry_reuses_row_for_same_job_candidate_and_roi():
    repository = object.__new__(PostgresRepository)
    repository.schema = "pelagia_unit"
    repository._ensure_project_scope = lambda *_args, **_kwargs: None
    existing = {"id": "refined-existing", "candidate_detection_id": "candidate-1"}
    cursor = _ExistingRowCursor(existing)
    connection = _ExistingRowConnection(cursor)
    repository.connect = lambda: connection

    rows = repository.upsert_refined_detections(
        [("candidate-1", _unit_detection(1))],
        job_id="job-1",
        project_id="project-1",
    )

    assert rows == [existing]
    assert len(cursor.executions) == 2
    assert "pg_advisory_xact_lock" in cursor.executions[0][0]
    assert cursor.executions[0][1] == ("roi-refinement:job-1:candidate-1:1",)
    assert (
        "WHERE job_id = %s AND candidate_detection_id = %s AND roi_index = %s"
        in cursor.executions[1][0]
    )
    assert cursor.executions[1][1] == ("job-1", "candidate-1", 1)
    assert all("INSERT INTO" not in statement for statement, _ in cursor.executions)


def test_refinement_retry_preserves_parent_and_residual_children():
    repository = object.__new__(PostgresRepository)
    repository.schema = "pelagia_unit"
    repository._ensure_project_scope = lambda *_args, **_kwargs: None
    cursor = _RefinedRetryCursor()
    connection = _ExistingRowConnection(cursor)
    repository.connect = lambda: connection
    parent = _unit_detection(7)
    residual_one = _unit_detection(7101)
    residual_two = _unit_detection(7102)
    residual_one.frame_id = parent.frame_id
    residual_two.frame_id = parent.frame_id
    residual_one.metadata.update(
        {"synthetic_candidate": True, "residual_component_index": 1}
    )
    residual_two.metadata.update(
        {"synthetic_candidate": True, "residual_component_index": 2}
    )
    outputs = [
        ("candidate-1", parent),
        ("candidate-1", residual_one),
        ("candidate-1", residual_two),
    ]

    first = repository.upsert_refined_detections(
        outputs,
        job_id="job-1",
        project_id="project-1",
    )
    retried = repository.upsert_refined_detections(
        outputs,
        job_id="job-1",
        project_id="project-1",
    )

    assert [row["roi_index"] for row in first] == [7, 7101, 7102]
    assert [row["id"] for row in retried] == [row["id"] for row in first]
    assert cursor.insert_count == 3
    assert len(cursor.rows) == 3


def test_continuity_retry_reuses_run_for_same_job_asset_and_method():
    repository = object.__new__(PostgresRepository)
    repository.schema = "pelagia_unit"
    repository._ensure_project_scope = lambda *_args, **_kwargs: None
    existing = {"id": "continuity-existing"}
    cursor = _ExistingRowCursor(existing)
    connection = _ExistingRowConnection(cursor)
    repository.connect = lambda: connection

    stored = repository.record_roi_continuity(
        project_id="project-1",
        asset_id="asset-1",
        run_id="run-1",
        job_id="job-1",
        method="line_scan_continuity_v1",
        parameters={"scan_axis": "y"},
        assemblies=[],
        decisions=[],
    )

    assert stored == existing
    assert len(cursor.executions) == 2
    assert "pg_advisory_xact_lock" in cursor.executions[0][0]
    assert "roi_continuity_runs" in cursor.executions[1][0]
    assert all("INSERT INTO" not in statement for statement, _ in cursor.executions)


def test_render_schema_loads_sql_resource():
    rendered = postgres.render_schema("pelagia_unit")

    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.users" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.projects" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.project_memberships" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.user_sessions" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.schema_migrations" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.frames" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.logs" in rendered
    assert "'telemetry_import'" in rendered
    assert "project_id uuid" in rendered
    assert "payload_ref text" in rendered
    assert "project_id uuid REFERENCES pelagia_unit.projects(id) ON DELETE CASCADE" in rendered
    assert "job_id uuid REFERENCES pelagia_unit.processing_jobs(id) ON DELETE SET NULL" in rendered
    assert "UNIQUE (candidate_detection_id)" not in rendered
    assert "DROP CONSTRAINT IF EXISTS detections_refined_candidate_detection_id_key" in rendered
    assert "{schema}" not in rendered


def test_packaged_migrations_are_discoverable_and_rendered():
    migrations = postgres.available_migrations()

    assert [migration["migration_id"] for migration in migrations] == [
        "0001_processing_status",
        "0002_projectless_admin_sessions",
        "0003_processing_status_summary_indexes",
        "0004_frame_flatfield_profiles",
        "0005_roi_curation",
        "0006_registry_workspaces",
        "0007_registry_generation",
        "0008_registry_item_geometry",
        "0009_interchange_ingestion",
        "0010_telemetry",
        "0011_telemetry_ingest_durability",
        "0012_telemetry_reference_indexes",
        "0013_telemetry_timeline_overlap",
        "0014_telemetry_reference_index_names",
        "0015_telemetry_import_stage",
        "0016_processing_series",
        "0017_processing_series_snapshots",
        "0018_roi_clustering_evidence",
        "0019_feature_space_analysis",
        "0020_roi_embedding_evidence",
        "0021_roi_continuity",
            "0022_job_lease_fencing",
            "0023_job_retry_dispatch",
            "0024_export_bundles",
            "0025_export_lifecycle",
            "0026_clear_series_work_units",
            "0027_roi_orientation",
            "0028_oracle_v2_inference_provenance",
            "0029_operational_log_retention",
            "0030_registry_mask_revision_integrity",
            "0031_processing_submissions",
            "0032_incremental_processing_submissions",
            "0033_storage_reencode",
    ]
    rendered = postgres.render_migration(migrations[0], "pelagia_unit")
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.frame_processing_status" in rendered
    assert "{schema}" not in rendered


    projectless_sessions = postgres.render_migration(migrations[1], "pelagia_unit")
    assert "ALTER COLUMN project_id DROP NOT NULL" in projectless_sessions
    assert "{schema}" not in projectless_sessions
    summary_indexes = postgres.render_migration(migrations[2], "pelagia_unit")
    assert "frame_processing_status_has_candidates" in summary_indexes
    assert "{schema}" not in summary_indexes
    flatfield_columns = postgres.render_migration(migrations[3], "pelagia_unit")
    assert "ADD COLUMN IF NOT EXISTS flatfield_profile real[]" in flatfield_columns
    assert "ADD COLUMN IF NOT EXISTS flatfield_metadata jsonb" in flatfield_columns
    assert "{schema}" not in flatfield_columns
    curation_schema = postgres.render_migration(migrations[4], "pelagia_unit")
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.classification_evidence" in curation_schema
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.roi_label_annotations" in curation_schema
    assert "{schema}" not in curation_schema
    telemetry_schema = postgres.render_migration(migrations[9], "pelagia_unit")
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.telemetry_observations" in telemetry_schema
    assert "PARTITION BY HASH (stream_id)" in telemetry_schema
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.timeline_events" in telemetry_schema
    assert "{schema}" not in telemetry_schema
    durability_schema = postgres.render_migration(migrations[10], "pelagia_unit")
    assert "ADD COLUMN IF NOT EXISTS import_key text" in durability_schema
    assert "telemetry_sources_import_key" in durability_schema
    assert "{schema}" not in durability_schema
    index_schema = postgres.render_migration(migrations[11], "pelagia_unit")
    assert "telemetry_streams_sensor_project" in index_schema
    assert "{schema}" not in index_schema
    overlap_schema = postgres.render_migration(migrations[12], "pelagia_unit")
    assert "USING gist" in overlap_schema
    assert "tstzrange(start_at, COALESCE(end_at, start_at), '[]')" in overlap_schema
    assert "{schema}" not in overlap_schema
    reference_name_schema = postgres.render_migration(migrations[13], "pelagia_unit")
    assert "telemetry_streams_sensor_project_idx" in reference_name_schema
    assert "{schema}" not in reference_name_schema
    telemetry_stage_schema = postgres.render_migration(migrations[14], "pelagia_unit")
    assert "ADD VALUE IF NOT EXISTS 'telemetry_import'" in telemetry_stage_schema
    assert "{schema}" not in telemetry_stage_schema
    series_schema = postgres.render_migration(migrations[15], "pelagia_unit")
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.processing_series" in series_schema
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.processing_work_units" in series_schema
    assert "{schema}" not in series_schema
    analysis_schema = postgres.render_migration(migrations[18], "pelagia_unit")
    assert "ADD VALUE IF NOT EXISTS 'feature_space_analysis'" in analysis_schema
    assert "{schema}" not in analysis_schema
    clustering_schema = postgres.render_migration(migrations[17], "pelagia_unit")
    embedding_schema = postgres.render_migration(migrations[19], "pelagia_unit")
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.clustering_evidence" in clustering_schema
    assert "evidence_kind" in clustering_schema
    assert "{schema}" not in clustering_schema
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.embedding_evidence" in embedding_schema
    assert "'embedding'" in embedding_schema
    assert "{schema}" not in embedding_schema
    inference_v2_schema = postgres.render_migration(migrations[27], "pelagia_unit")
    assert "classification_inference_batches" in inference_v2_schema
    assert "oracle_artifact_fingerprint_sha256" in inference_v2_schema
    assert "classification_inference_batch_items" in inference_v2_schema
    assert "{schema}" not in inference_v2_schema
    incremental_schema = postgres.render_migration(migrations[31], "pelagia_unit")
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.processing_submission_units" in incremental_schema
    assert "idx_pelagia_unit_psu_active" in incremental_schema
    assert "idx_pelagia_unit_ps_series_unique" in incremental_schema
    assert "planning_complete" in incremental_schema
    assert "{schema}" not in incremental_schema


def test_project_storage_reencode_job_is_deduplicated_while_active(postgres_repo):
    project = postgres_repo.create_project(f"storage-update-{uuid.uuid4().hex}")
    policy = {
        "frame_encoding": "jpg", "frame_quality": 75,
        "small_roi_encoding": "zstd", "large_roi_encoding": "jpg",
        "large_roi_min_pixels": 1000, "roi_quality": 75,
        "mask_encoding": "zstd",
    }

    first, created = postgres_repo.enqueue_project_storage_reencode(
        project_id=str(project["id"]), policy=policy,
    )
    second, second_created = postgres_repo.enqueue_project_storage_reencode(
        project_id=str(project["id"]), policy=policy,
    )

    assert created is True
    assert second_created is False
    assert first["id"] == second["id"]
    assert first["payload"]["policy"] == policy
    assert postgres_repo.latest_project_storage_reencode_job(str(project["id"]))["id"] == first["id"]


def test_storage_reencode_worker_availability_uses_capability_and_recent_heartbeat(postgres_repo):
    assert postgres_repo.has_online_worker_capability("storage_reencode") is False
    postgres_repo.touch_worker("storage-capable", "idle", capabilities=["storage_reencode"])
    assert postgres_repo.has_online_worker_capability("storage_reencode") is True
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {postgres_repo.schema}.worker_sessions "
                "SET last_heartbeat = NOW() - INTERVAL '1 hour' "
                "WHERE worker_id = 'storage-capable'"
            )
        connection.commit()
    assert postgres_repo.has_online_worker_capability("storage_reencode") is False


def test_project_storage_reencode_rows_are_scoped_and_compare_and_swap(postgres_repo):
    project = postgres_repo.create_project(f"storage-rows-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    other = postgres_repo.create_project(f"storage-other-{uuid.uuid4().hex}")
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.runs
                (project_id, run_key, instrument, source_path, source_type)
                VALUES (%s, %s, 'test', '/tmp/test', 'image') RETURNING id""",
                (project_id, f"storage-run-{uuid.uuid4().hex}"),
            )
            run_id = str(cursor.fetchone()["id"])
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.raw_assets
                (project_id, run_id, filename, path, kind, checksum, size_bytes)
                VALUES (%s, %s, 'frame.png', '/tmp/frame.png', 'image', 'sha256:test', 4)
                RETURNING id""",
                (project_id, run_id),
            )
            asset_id = str(cursor.fetchone()["id"])
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.frames
                (run_id, asset_id, frame_index, width, height, kvstore_hash,
                 preview_thumbhash, payload_ref, payload_encoding, payload_format,
                 payload_dtype, payload_shape)
                VALUES (%s, %s, 1, 2, 2, 'old-ref', %s, 'old-ref', 'raw',
                        'raw_ndarray_c_order', 'uint8', '[2,2]'::jsonb) RETURNING id""",
                (run_id, asset_id, b"preview"),
            )
            frame_id = str(cursor.fetchone()["id"])
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.detection_candidate
                (run_id, frame_id, roi_index, bbox_x, bbox_y, bbox_w, bbox_h,
                 roi_payload, mask_payload, roi_encoding, roi_format, roi_dtype,
                 roi_shape, mask_encoding, mask_format, mask_dtype, mask_shape)
                VALUES (%s, %s, 1, 0, 0, 2, 2, %s, %s, 'raw',
                        'raw_ndarray_c_order', 'uint8', '[2,2]'::jsonb, 'raw',
                        'raw_ndarray_c_order', 'uint8', '[2,2]'::jsonb) RETURNING id""",
                (run_id, frame_id, b"roi!", b"mask"),
            )
            detection_id = str(cursor.fetchone()["id"])
        connection.commit()

    boundary = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    page = postgres_repo.list_storage_reencode_page(
        project_id=project_id, resource="frames", created_before=boundary,
    )
    assert [str(row["id"]) for row in page] == [frame_id]
    assert postgres_repo.list_storage_reencode_page(
        project_id=str(other["id"]), resource="frames", created_before=boundary,
    ) == []
    frame = postgres_repo.get_storage_reencode_row(
        project_id=project_id, resource="frames", resource_id=frame_id,
    )
    assert postgres_repo.get_storage_reencode_row(
        project_id=str(other["id"]), resource="frames", resource_id=frame_id,
    ) is None
    assert postgres_repo.update_storage_reencode_frame(
        project_id=project_id, frame_id=frame_id, row_version=frame["row_version"],
        variant="original", old_ref="old-ref", new_ref="new-ref", encoding="zstd",
        format="zstd_ndarray_c_order", dtype="uint8", shape=[2, 2],
        metadata={"kvstore_key": "new-ref"}, preview_thumbhash=b"new-preview",
    ) is True
    assert postgres_repo.update_storage_reencode_frame(
        project_id=project_id, frame_id=frame_id, row_version=frame["row_version"],
        variant="original", old_ref="old-ref", new_ref="bad-ref", encoding="zstd",
        format="zstd_ndarray_c_order", dtype="uint8", shape=[2, 2], metadata={},
    ) is False
    detection = postgres_repo.get_storage_reencode_row(
        project_id=project_id, resource="detection_candidate", resource_id=detection_id,
    )
    assert postgres_repo.update_storage_reencode_detection(
        project_id=project_id, resource="detection_candidate", resource_id=detection_id,
        row_version=detection["row_version"], roi_payload=b"new!", roi_encoding="zstd",
        roi_format="zstd_ndarray_c_order", mask_payload=b"mask",
        mask_encoding="raw", mask_format="raw_ndarray_c_order", metadata={"rewritten": True},
    ) is True
    assert postgres_repo.get_storage_reencode_row(
        project_id=project_id, resource="detection_candidate", resource_id=detection_id,
    )["metadata"]["rewritten"] is True


def test_postgres_project_columns_are_mandatory_without_defaults(postgres_repo):
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND column_name = 'project_id'
                  AND table_name = ANY(%s)
                """,
                (
                    postgres_repo.schema,
                    ["runs", "raw_assets", "models", "processing_jobs", "logs"],
                ),
            )
            rows = {row["table_name"]: row for row in cursor.fetchall()}

    assert set(rows) == {"runs", "raw_assets", "models", "processing_jobs", "logs"}
    for row in rows.values():
        assert row["is_nullable"] == "NO"
        assert row["column_default"] is None


def test_postgres_schema_status_reports_applied_migrations(postgres_repo):
    status = postgres_repo.schema_status()

    assert status["ready"] is True
    assert "schema_migrations" in status["existing_tables"]
    assert status["migrations"]["available_count"] == 33
    assert status["migrations"]["applied_count"] == 33
    assert status["migrations"]["pending_count"] == 0
    assert status["migrations"]["applied"][0]["migration_id"] == "0001_processing_status"


def test_incremental_submission_indexes_remain_distinct_in_long_test_schema(postgres_repo):
    configured_names = {
        f"idx_{postgres_repo.schema}_psu_active",
        f"idx_{postgres_repo.schema}_psu_submission",
        f"idx_{postgres_repo.schema}_psu_unreleased",
        f"idx_{postgres_repo.schema}_psj_submission",
        f"idx_{postgres_repo.schema}_ps_fair",
        f"idx_{postgres_repo.schema}_ps_series_unique",
        f"idx_{postgres_repo.schema}_frames_seek",
        f"idx_{postgres_repo.schema}_dc_seek",
    }
    # PostgreSQL's ``name`` type truncates identifiers to 63 bytes. The short
    # suffixes must remain distinct even with the deliberately long test schema.
    expected = {name[:63] for name in configured_names}
    assert len(expected) == len(configured_names)
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT indexname, indexdef
               FROM pg_indexes
               WHERE schemaname = %s AND indexname = ANY(%s)""",
            (postgres_repo.schema, sorted(expected)),
        )
        indexes = {row["indexname"]: row["indexdef"] for row in cursor.fetchall()}

    assert set(indexes) == expected
    active_name = f"idx_{postgres_repo.schema}_psu_active"[:63]
    assert "CREATE UNIQUE INDEX" in indexes[active_name]
    assert "WHERE (released_at IS NULL)" in indexes[active_name]


def test_postgres_reuses_oracle_v2_inference_run_for_job(postgres_repo):
    project = postgres_repo.create_project(f"oracle-v2-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    job = postgres_repo.create_job(
        PipelineStage.CLASSIFY,
        project_id=project_id,
        payload={},
    )
    artifact_id = str(uuid.uuid4())
    arguments = {
        "project_id": project_id,
        "job_id": str(job["id"]),
        "model_selector": artifact_id,
        "evidence_kind": "classification",
        "artifact_id": artifact_id,
        "artifact_fingerprint": "a" * 64,
        "output_schema_version": "image_analysis.v1",
    }

    first = postgres_repo.create_classification_inference_run(**arguments)
    resumed = postgres_repo.create_classification_inference_run(**arguments)

    assert resumed["id"] == first["id"]
    assert resumed["oracle_artifact_id"] == artifact_id
    assert resumed["oracle_artifact_fingerprint_sha256"] == "a" * 64
    assert postgres_repo.list_classification_inference_batches(
        project_id=project_id,
        inference_run_id=str(first["id"]),
    ) == []
    with pytest.raises(ValueError, match="different Oracle V2 contract"):
        postgres_repo.create_classification_inference_run(
            **(arguments | {"artifact_fingerprint": "b" * 64})
        )


def test_postgres_atomically_commits_oracle_ledger_and_evidence(postgres_repo, monkeypatch):
    project = postgres_repo.create_project(f"oracle-atomic-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    frame_id = str(uuid.uuid4())
    candidate_id = str(uuid.uuid4())
    refined_id = str(uuid.uuid4())
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.runs
                    (id, project_id, run_key, instrument, source_path, source_type)
                    VALUES (%s, %s, %s, 'test', '/tmp/test', 'video')""",
                (run_id, project_id, f"oracle-atomic-{uuid.uuid4().hex}"),
            )
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.raw_assets
                    (id, project_id, run_id, filename, path, kind, checksum, size_bytes)
                    VALUES (%s, %s, %s, 'test.avi', '/tmp/test.avi', 'video', 'sha256:test', 1)""",
                (asset_id, project_id, run_id),
            )
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.frames
                    (id, run_id, asset_id, frame_index, width, height, kvstore_hash, preview_thumbhash)
                    VALUES (%s, %s, %s, 0, 1, 1, 'frame-key', %s)""",
                (frame_id, run_id, asset_id, b"thumb"),
            )
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.detection_candidate
                    (id, run_id, frame_id, roi_index, bbox_x, bbox_y, bbox_w, bbox_h)
                    VALUES (%s, %s, %s, 0, 0, 0, 1, 1)""",
                (candidate_id, run_id, frame_id),
            )
            cursor.execute(
                f"""INSERT INTO {postgres_repo.schema}.detections_refined
                    (id, candidate_detection_id, run_id, frame_id, roi_index,
                     bbox_x, bbox_y, bbox_w, bbox_h)
                    VALUES (%s, %s, %s, %s, 0, 0, 0, 1, 1)""",
                (refined_id, candidate_id, run_id, frame_id),
            )
        connection.commit()

    artifact_id = str(uuid.uuid4())
    inference_run = postgres_repo.create_classification_inference_run(
        project_id=project_id,
        job_id=None,
        model_selector=artifact_id,
        evidence_kind="classification",
        artifact_id=artifact_id,
        artifact_fingerprint="a" * 64,
        output_schema_version="1.0.0",
    )
    inference_run_id = str(inference_run["id"])
    request_id = str(uuid.uuid4())
    item_id = str(uuid.uuid4())
    input_sha256 = "b" * 64

    other_project = postgres_repo.create_project(f"oracle-atomic-other-{uuid.uuid4().hex}")
    other_artifact_id = str(uuid.uuid4())
    other_inference_run = postgres_repo.create_classification_inference_run(
        project_id=str(other_project["id"]),
        job_id=None,
        model_selector=other_artifact_id,
        evidence_kind="classification",
        artifact_id=other_artifact_id,
        artifact_fingerprint="c" * 64,
        output_schema_version="1.0.0",
    )
    with pytest.raises(ValueError, match="belong to the inference run project"):
        postgres_repo.reserve_classification_inference_batch(
            project_id=str(other_project["id"]),
            inference_run_id=str(other_inference_run["id"]),
            request_id=str(uuid.uuid4()),
            batch_index=0,
            items=[{
                "refined_detection_id": refined_id,
                "item_id": str(uuid.uuid4()),
                "input_sha256": input_sha256,
            }],
        )

    postgres_repo.reserve_classification_inference_batch(
        project_id=project_id,
        inference_run_id=inference_run_id,
        request_id=request_id,
        batch_index=0,
        items=[{
            "refined_detection_id": refined_id,
            "item_id": item_id,
            "input_sha256": input_sha256,
        }],
    )
    model = {
        "artifact_id": artifact_id,
        "artifact_fingerprint": "a" * 64,
        "task": "classification",
    }
    probabilities = [
        {"class_index": 0, "label_name": "copepod", "probability": 0.9},
        {"class_index": 1, "label_name": "other", "probability": 0.1},
    ]
    evidence_context = postgres_repo.prepare_classification_evidence_context(
        project_id=project_id,
        inference_run_id=inference_run_id,
        model=model,
        probabilities=probabilities,
    )
    result = {
        "item_id": item_id,
        "refined_detection_id": refined_id,
        "input_sha256": input_sha256,
        "status": "succeeded",
    }
    record = {
        "refined_detection_id": refined_id,
        "output": {
            "decision": {"class_index": 0, "label_name": "copepod"},
            "probabilities": probabilities,
            "evidence": {},
        },
        "oracle_result": {"item_id": item_id, "input_sha256": input_sha256},
    }

    original_store = postgres_repo.store_classification_evidence_batch

    def fail_evidence_write(**kwargs):
        assert kwargs["_connection"] is not None
        raise RuntimeError("simulated evidence insert failure")

    monkeypatch.setattr(postgres_repo, "store_classification_evidence_batch", fail_evidence_write)
    with pytest.raises(RuntimeError, match="simulated evidence insert failure"):
        postgres_repo.finalize_classification_inference_batch(
            project_id=project_id,
            inference_run_id=inference_run_id,
            request_id=request_id,
            evidence_kind="classification",
            results=[result],
            classification_context=evidence_context,
            classification_records=[record],
        )
    assert postgres_repo.list_classification_inference_batches(
        project_id=project_id, inference_run_id=inference_run_id
    )[0]["items"][0]["status"] == "reserved"

    monkeypatch.setattr(postgres_repo, "store_classification_evidence_batch", original_store)
    postgres_repo.finalize_classification_inference_batch(
        project_id=project_id,
        inference_run_id=inference_run_id,
        request_id=request_id,
        evidence_kind="classification",
        results=[result],
        classification_context=evidence_context,
        classification_records=[record],
    )
    batch = postgres_repo.list_classification_inference_batches(
        project_id=project_id, inference_run_id=inference_run_id
    )[0]
    assert batch["status"] == "complete"
    assert batch["items"][0]["status"] == "succeeded"

    later_record = {
        **record,
        "output": {
            **record["output"],
            "decision": {"class_index": 1, "label_name": "other"},
        },
    }
    postgres_repo.finalize_classification_inference_batch(
        project_id=project_id,
        inference_run_id=inference_run_id,
        request_id=request_id,
        evidence_kind="classification",
        results=[result],
        classification_context=evidence_context,
        classification_records=[later_record],
    )
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT count(*) AS count, min(predicted_class_index) AS predicted_class_index
                    FROM {postgres_repo.schema}.classification_evidence
                    WHERE project_id = %s AND inference_run_id = %s
                      AND refined_detection_id = %s""",
                (project_id, inference_run_id, refined_id),
            )
            evidence = cursor.fetchone()
            assert evidence["count"] == 1
            assert evidence["predicted_class_index"] == 0


def test_postgres_export_artifacts_are_project_scoped_and_attach_jobs(postgres_repo):
    project = postgres_repo.create_project(f"export-{uuid.uuid4().hex}")
    other = postgres_repo.create_project(f"export-other-{uuid.uuid4().hex}")
    artifact = postgres_repo.create_export_artifact(
        project_id=str(project["id"]), request={"products": ["raw_roi_statistics"]},
    )
    job = postgres_repo.create_job(
        PipelineStage.EXPORT_BUNDLE, project_id=str(project["id"]), payload={"export_id": str(artifact["id"])},
    )
    attached = postgres_repo.attach_export_artifact_job(
        str(artifact["id"]), str(job["id"]), project_id=str(project["id"]),
    )
    assert attached is not None and str(attached["job_id"]) == str(job["id"])
    assert postgres_repo.get_export_artifact(str(artifact["id"]), project_id=str(other["id"])) is None
    completed = postgres_repo.update_export_artifact(
        str(artifact["id"]), project_id=str(project["id"]), status="succeeded",
        manifest={"export_id": str(artifact["id"])}, artifact_path="/not/exposed.zip",
        artifact_sha256="a" * 64, size_bytes=12,
        input_snapshot={"large_membership": ["snapshot-marker"]},
        attempts=[{"attempt_number": 1, "details": "attempt-marker"}],
        estimate={"estimated_bundle_bytes": 1024},
        progress={"completed": 1, "total": 1},
    )
    assert completed is not None and completed["status"] == "succeeded"
    history = postgres_repo.list_export_artifacts(project_id=str(project["id"]))
    assert len(history) == 1
    assert not {"manifest", "input_snapshot", "attempts", "estimate", "progress"}.intersection(history[0])
    assert postgres_repo.get_export_artifact(str(artifact["id"]), project_id=str(project["id"]))["manifest"] == {
        "export_id": str(artifact["id"])
    }


def test_initialize_schema_upgrades_database_missing_latest_job_columns(postgres_repo):
    """The base schema must not index 0023 columns before its migration runs."""

    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"ALTER TABLE {postgres_repo.schema}.processing_jobs "
                "DROP COLUMN lease_token, "
                "DROP COLUMN available_at, "
                "DROP COLUMN failure_category, "
                "DROP COLUMN idempotency_key"
            )
            cursor.execute(
                f"DROP TABLE {postgres_repo.schema}.processing_job_dispatches"
            )
            cursor.execute(
                f"DELETE FROM {postgres_repo.schema}.schema_migrations "
                "WHERE migration_id IN ('0022_job_lease_fencing', '0023_job_retry_dispatch')"
            )
        connection.commit()

    postgres_repo.initialize_schema()

    status = postgres_repo.migration_status()
    assert status["ready"] is True
    assert status["pending_count"] == 0
    assert {
        row["migration_id"] for row in status["applied"]
    } >= {"0022_job_lease_fencing", "0023_job_retry_dispatch"}


def test_initialize_schema_adds_planner_columns_before_indexing_them(postgres_repo):
    """Existing submissions must reach migration 0032 before its fair index."""

    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"ALTER TABLE {postgres_repo.schema}.processing_submissions "
                "DROP COLUMN last_planned_at CASCADE"
            )
            cursor.execute(
                f"DELETE FROM {postgres_repo.schema}.schema_migrations "
                "WHERE migration_id = '0032_incremental_processing_submissions'"
            )
        connection.commit()

    postgres_repo.initialize_schema()

    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'processing_submissions' "
                "AND column_name = 'last_planned_at'",
                (postgres_repo.schema,),
            )
            assert cursor.fetchone() is not None
            cursor.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = %s AND indexname = %s",
                (postgres_repo.schema, f"idx_{postgres_repo.schema}_ps_fair"),
            )
            assert cursor.fetchone() is not None


def test_processing_series_is_project_scoped_and_advances_terminal_steps(postgres_repo):
    project = postgres_repo.create_project(f"series-{uuid.uuid4().hex}")
    other_project = postgres_repo.create_project(f"series-other-{uuid.uuid4().hex}")
    series = postgres_repo.create_processing_series(
        project_id=str(project["id"]),
        steps=[
            {"stage": PipelineStage.SEGMENT.value, "filters": {}, "options": {}},
            {"stage": PipelineStage.ROI_REFINEMENT.value, "filters": {}, "options": {}},
        ],
        failure_policy="continue",
    )
    assert postgres_repo.get_processing_series(str(series["id"]), project_id=str(other_project["id"])) is None
    first = postgres_repo.claim_processing_series_step(str(series["id"]), project_id=str(project["id"]))
    assert first is not None
    job = postgres_repo.create_job(PipelineStage.SEGMENT, project_id=str(project["id"]))
    postgres_repo.attach_processing_work_units(series_id=str(series["id"]), step_id=str(first["id"]), job_ids=[str(job["id"])], matched_count=1)
    postgres_repo.complete_job(str(job["id"]))
    advanced = postgres_repo.advance_processing_series_for_job(str(job["id"]))
    assert advanced is not None and advanced["ready"] is True
    steps = postgres_repo.list_processing_series_steps(str(series["id"]), project_id=str(project["id"]))
    assert steps[0]["status"] == "succeeded"
    second = postgres_repo.claim_processing_series_step(str(series["id"]), project_id=str(project["id"]))
    assert second is not None
    postgres_repo.attach_processing_work_units(series_id=str(series["id"]), step_id=str(second["id"]), job_ids=[], matched_count=0)
    assert postgres_repo.get_processing_series(str(series["id"]), project_id=str(project["id"]))["status"] == "succeeded"


def test_processing_submission_is_idempotent_scoped_and_fenced(postgres_repo):
    project = postgres_repo.create_project(f"submission-{uuid.uuid4().hex}")
    other = postgres_repo.create_project(f"submission-other-{uuid.uuid4().hex}")
    request_id = str(uuid.uuid4())
    request = {"stage": "segment", "filters": {"collection": ["tow-a"]}, "options": {}}

    first, created = postgres_repo.create_processing_submission(
        project_id=str(project["id"]), request_id=request_id, kind="queue", request=request,
    )
    repeated, created_again = postgres_repo.create_processing_submission(
        project_id=str(project["id"]), request_id=request_id, kind="queue", request=request,
    )
    assert created is True
    assert created_again is False
    assert repeated["id"] == first["id"]
    with pytest.raises(ValueError, match="different processing submission"):
        postgres_repo.create_processing_submission(
            project_id=str(project["id"]), request_id=request_id, kind="queue", request={**request, "priority": 2},
        )
    scoped_other, other_created = postgres_repo.create_processing_submission(
        project_id=str(other["id"]), request_id=request_id, kind="queue", request=request,
    )
    assert other_created is True
    assert scoped_other["id"] != first["id"]

    claimed = postgres_repo.claim_processing_submission()
    assert claimed is not None and claimed["id"] == first["id"]
    stale = postgres_repo.complete_processing_submission(
        str(claimed["id"]), claim_token=str(uuid.uuid4()), result={"job_count": 0},
    )
    assert stale is None
    with postgres_repo.processing_submission_transaction(
        str(claimed["id"]), claim_token=str(claimed["claim_token"]), project_id=str(project["id"]),
    ) as connection:
        assert connection is not None
        completed = postgres_repo.complete_processing_submission(
            str(claimed["id"]), claim_token=str(claimed["claim_token"]), result={"job_count": 0}, _connection=connection,
        )
        assert completed is not None
    assert postgres_repo.get_processing_submission(str(first["id"]), project_id=str(project["id"]))["status"] == "succeeded"


def test_processing_series_failure_policy_and_retry(postgres_repo):
    project = postgres_repo.create_project(f"series-failure-{uuid.uuid4().hex}")
    def failed_series(policy):
        series = postgres_repo.create_processing_series(project_id=str(project["id"]), steps=[{"stage": PipelineStage.SEGMENT.value}], failure_policy=policy)
        step = postgres_repo.claim_processing_series_step(str(series["id"]), project_id=str(project["id"]))
        job = postgres_repo.create_job(PipelineStage.SEGMENT, project_id=str(project["id"]))
        postgres_repo.attach_processing_work_units(series_id=str(series["id"]), step_id=str(step["id"]), job_ids=[str(job["id"])], matched_count=1)
        postgres_repo.fail_job(str(job["id"]), "fixture failure")
        postgres_repo.advance_processing_series_for_job(str(job["id"]))
        return series, step, job

    fail_fast, _, job = failed_series("fail_fast")
    assert postgres_repo.get_processing_series(str(fail_fast["id"]), project_id=str(project["id"]))["status"] == "failed"
    postgres_repo.retry_processing_series(str(fail_fast["id"]), project_id=str(project["id"]))
    assert postgres_repo.get_job(str(job["id"]), project_id=str(project["id"]))["status"] == "queued"
    continuing, _, _ = failed_series("continue")
    assert postgres_repo.get_processing_series(str(continuing["id"]), project_id=str(project["id"]))["status"] == "succeeded"


def test_postgres_telemetry_import_lookup_events_and_project_scope(postgres_repo, tmp_path):
    project = postgres_repo.create_project(f"telemetry-{uuid.uuid4().hex}")
    other_project = postgres_repo.create_project(f"telemetry-other-{uuid.uuid4().hex}")
    event_author = postgres_repo.create_user(f"telemetry-author-{uuid.uuid4().hex}")
    run_id = str(uuid.uuid4())
    postgres_repo.register_planned_run(
        PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=f"telemetry-run-{uuid.uuid4().hex}",
                instrument="shipboard",
                source_path=str(tmp_path),
                source_type="telemetry",
                created_at=datetime.now(timezone.utc),
            )
        ),
        project_id=str(project["id"]),
    )
    source = tmp_path / "telemetry.csv"
    source.write_text(
        "time,temp\n"
        "2026-08-21T00:00:00Z,10\n"
        "2026-08-21T00:00:02Z,14\n",
        encoding="utf-8",
    )

    blob_store = _MemoryBlobStore()
    service = TelemetryIngestionService(postgres_repo, blob_store)
    imported = service.import_csv(
        source,
        project_id=str(project["id"]),
        run_id=run_id,
        spec=TelemetryCsvSpec(
            timestamp_column="time",
            streams=[
                TelemetryColumn(
                    column="temp", stream_key="sensor.temperature", sensor_key="sensor",
                    parameter_key="temperature", native_unit="degC", canonical_unit="degC",
                    interpolation="linear", max_gap_seconds=5,
                )
            ],
        ),
    )

    assert imported["source"]["observation_count"] == 2
    assert blob_store.get_store(imported["source"]["source_payload_key"]) == source.read_bytes()
    retried = service.import_csv(
        source,
        project_id=str(project["id"]),
        run_id=run_id,
        spec=TelemetryCsvSpec(
            timestamp_column="time",
            streams=[
                TelemetryColumn(
                    column="temp", stream_key="sensor.temperature", sensor_key="sensor",
                    parameter_key="temperature", native_unit="degC", canonical_unit="degC",
                    interpolation="linear", max_gap_seconds=5,
                )
            ],
        ),
    )
    assert retried["source"]["id"] == imported["source"]["id"]
    assert len(postgres_repo.telemetry.list_telemetry_sources(
        project_id=str(project["id"]), run_id=run_id,
    )) == 1
    assert postgres_repo.telemetry.get_telemetry_import(
        project_id=str(other_project["id"]), run_id=run_id,
        import_key=imported["source"]["import_key"],
    ) is None
    resolved = TelemetryResolver(postgres_repo).at(
        project_id=str(project["id"]), run_id=run_id,
        observed_at=datetime(2026, 8, 21, 0, 0, 1, tzinfo=timezone.utc),
    )
    assert resolved["telemetry"]["temperature"]["value"] == pytest.approx(12.0)
    assert postgres_repo.telemetry.list_telemetry_streams(
        project_id=str(other_project["id"]), run_id=run_id,
    ) == []

    secondary_source = tmp_path / "telemetry-secondary.csv"
    secondary_source.write_text(
        "time,temp\n2026-08-21T00:00:00Z,9\n2026-08-21T00:00:02Z,13\n",
        encoding="utf-8",
    )
    TelemetryIngestionService(postgres_repo, blob_store).import_csv(
        secondary_source,
        project_id=str(project["id"]),
        run_id=run_id,
        spec=TelemetryCsvSpec(
            timestamp_column="time",
            streams=[
                TelemetryColumn(
                    column="temp", stream_key="secondary.temperature", sensor_key="secondary",
                    parameter_key="temperature", native_unit="degC", canonical_unit="degC",
                    interpolation="linear", max_gap_seconds=5,
                )
            ],
        ),
    )
    streams = postgres_repo.telemetry.list_telemetry_streams(
        project_id=str(project["id"]), run_id=run_id,
    )
    assert len(streams) == 2
    assert sum(stream["is_default"] for stream in streams) == 1

    event_type = postgres_repo.telemetry.create_timeline_event_type(
        project_id=str(project["id"]), event_type_key="station",
    )
    event = postgres_repo.telemetry.create_timeline_event(
        project_id=str(project["id"]), run_id=run_id, event_type_id=str(event_type["id"]),
        start_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
        end_at=datetime(2026, 8, 21, 0, 10, tzinfo=timezone.utc), value="DBO3.4",
        source_id=str(imported["source"]["id"]), created_by=str(event_author["id"]),
    )
    assert event["value"] == "DBO3.4"
    assert str(event["source_id"]) == str(imported["source"]["id"])
    assert str(event["created_by"]) == str(event_author["id"])
    assert postgres_repo.telemetry.get_timeline_event(
        project_id=str(project["id"]), run_id=run_id, event_id=str(event["id"]),
    )["id"] == event["id"]
    assert postgres_repo.telemetry.list_timeline_events(
        project_id=str(project["id"]), run_id=run_id,
        start_at=datetime(2026, 8, 21, 0, 5, tzinfo=timezone.utc),
        end_at=datetime(2026, 8, 21, 0, 15, tzinfo=timezone.utc),
    )[0]["id"] == event["id"]
    updated = postgres_repo.telemetry.update_timeline_event(
        project_id=str(project["id"]), run_id=run_id, event_id=str(event["id"]),
        updates={"value": "DBO3.5", "source_id": None},
    )
    assert updated["value"] == "DBO3.5"
    assert updated["source_id"] is None
    assert postgres_repo.telemetry.list_timeline_events_at(
        project_id=str(project["id"]), run_id=run_id,
        observed_at=datetime(2026, 8, 21, 0, 5, tzinfo=timezone.utc),
    )[0]["event_type_key"] == "station"

    point_type = postgres_repo.telemetry.create_timeline_event_type(
        project_id=str(project["id"]), event_type_key="camera_cleaned",
    )
    point_at = datetime(2026, 8, 21, 0, 20, tzinfo=timezone.utc)
    postgres_repo.telemetry.create_timeline_event(
        project_id=str(project["id"]), run_id=run_id,
        event_type_id=str(point_type["id"]), start_at=point_at,
    )
    assert any(
        row["event_type_key"] == "camera_cleaned"
        for row in postgres_repo.telemetry.list_timeline_events_at(
            project_id=str(project["id"]), run_id=run_id, observed_at=point_at,
        )
    )
    assert not any(
        row["event_type_key"] == "camera_cleaned"
        for row in postgres_repo.telemetry.list_timeline_events_at(
            project_id=str(project["id"]), run_id=run_id,
            observed_at=point_at + timedelta(milliseconds=1),
        )
    )

    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT indexdef FROM pg_indexes
                WHERE schemaname = %s AND indexname = %s""",
            (
                postgres_repo.schema,
                f"idx_tl_period_{postgres_repo.schema}",
            ),
        )
        overlap_index = cursor.fetchone()
        assert overlap_index is not None
        assert "USING gist" in overlap_index["indexdef"]
        cursor.execute("SET LOCAL enable_seqscan = off")
        cursor.execute(
            f"""EXPLAIN (FORMAT JSON)
                SELECT id FROM {postgres_repo.schema}.timeline_events
                WHERE tstzrange(start_at, COALESCE(end_at, start_at), '[]')
                      @> %s::timestamptz""",
            (datetime(2026, 8, 21, 0, 5, tzinfo=timezone.utc),),
        )
        plan = cursor.fetchone()["QUERY PLAN"]
        assert f"idx_tl_period_{postgres_repo.schema}" in str(plan)

    with pytest.raises(postgres.psycopg.errors.ForeignKeyViolation):
        with postgres_repo.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {postgres_repo.schema}.telemetry_sources SET project_id = %s WHERE id = %s",
                (other_project["id"], imported["source"]["id"]),
            )
            connection.commit()

    failed_asset_id = str(uuid.uuid4())

    def failing_observations():
        yield "failed.temperature", datetime(2026, 8, 21, tzinfo=timezone.utc), 1.0, None
        raise ValueError("simulated parser failure")

    with pytest.raises(ValueError, match="simulated parser failure"):
        postgres_repo.telemetry.ingest_telemetry(
            project_id=str(project["id"]),
            run_id=run_id,
            asset={
                "id": failed_asset_id, "filename": "failed.csv", "path": "/tmp/failed.csv",
                "checksum": "sha256:failed", "size_bytes": 1,
            },
            source={
                "format": "delimited", "parser_name": "test", "parser_version": "1",
                "import_key": "failed-import", "source_payload_key": "failed-payload",
            },
            parameters=[{"parameter_key": "temperature", "canonical_unit": "degC"}],
            sensors=[{"sensor_key": "failed-sensor"}],
            streams=[
                {
                    "stream_key": "failed.temperature", "sensor_key": "failed-sensor",
                    "parameter_key": "temperature", "native_unit": "degC",
                    "interpolation": "none", "is_default": False,
                }
            ],
            observations=failing_observations(),
        )
    assert postgres_repo.get_asset(failed_asset_id, project_id=str(project["id"])) is None
    assert postgres_repo.telemetry.get_telemetry_import(
        project_id=str(project["id"]), run_id=run_id, import_key="failed-import",
    ) is None


def test_postgres_concurrent_telemetry_defaults_return_domain_conflict(postgres_repo, tmp_path):
    project = postgres_repo.create_project(f"telemetry-default-race-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    run_id = str(uuid.uuid4())
    postgres_repo.register_planned_run(
        PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=f"telemetry-default-race-{uuid.uuid4().hex}",
                instrument="shipboard",
                source_path=str(tmp_path),
                source_type="telemetry",
                created_at=datetime.now(timezone.utc),
            )
        ),
        project_id=project_id,
    )

    barrier = Barrier(2)

    def ingest_contender(suffix: str):
        barrier.wait(timeout=5)
        return postgres_repo.telemetry.ingest_telemetry(
            project_id=project_id,
            run_id=run_id,
            asset={
                "id": str(uuid.uuid4()),
                "filename": f"{suffix}.csv",
                "path": f"/tmp/{suffix}.csv",
                "checksum": f"sha256:{suffix}",
                "size_bytes": 1,
            },
            source={
                "format": "delimited",
                "parser_name": "test",
                "parser_version": "1",
                "import_key": f"import-{suffix}",
                "source_payload_key": f"payload-{suffix}",
            },
            parameters=[{"parameter_key": "temperature", "canonical_unit": "degC"}],
            sensors=[{"sensor_key": f"sensor-{suffix}"}],
            streams=[
                {
                    "stream_key": f"{suffix}.temperature",
                    "sensor_key": f"sensor-{suffix}",
                    "parameter_key": "temperature",
                    "native_unit": "degC",
                    "interpolation": "none",
                    "is_default": True,
                }
            ],
        )

    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ingest_contender, suffix) for suffix in ("primary", "secondary")]
        for future in futures:
            try:
                results.append(future.result(timeout=15))
            except ValueError as exc:
                errors.append(exc)

    assert len(results) == 1
    assert len(errors) == 1
    assert "already has default telemetry stream" in str(errors[0])
    sources = postgres_repo.telemetry.list_telemetry_sources(
        project_id=project_id, run_id=run_id,
    )
    streams = postgres_repo.telemetry.list_telemetry_streams(
        project_id=project_id, run_id=run_id,
    )
    assert len(sources) == 1
    assert len(streams) == 1
    assert streams[0]["is_default"] is True


def test_postgres_worker_can_start_without_any_projects(postgres_repo):
    assert postgres_repo.list_projects(active_only=False) == []

    worker = postgres_repo.touch_worker(
        "projectless-worker",
        status="idle",
        capabilities=[PipelineStage.EXTRACT_FRAMES.value],
        pid=12345,
        shutdown_requested=False,
    )

    assert worker["worker_id"] == "projectless-worker"
    assert worker["status"] == "idle"
    shutdown = postgres_repo.request_worker_shutdown("projectless-worker", reason="test")
    assert shutdown is not None
    stopped = postgres_repo.touch_worker(
        "projectless-worker",
        status="stopped",
        capabilities=[PipelineStage.EXTRACT_FRAMES.value],
        pid=12345,
        shutdown_requested=False,
    )
    assert stopped["status"] == "stopped"
    events = postgres_repo.list_job_events(limit=100)
    event_types = {
        event["event_type"]
        for event in events
        if event["payload"].get("worker_id") == "projectless-worker"
    }
    assert event_types == {"worker.shutdown_requested"}
    assert postgres_repo.list_logs(worker_id="projectless-worker") == []


@pytest.fixture()
def postgres_repo():
    if postgres.psycopg is None:
        pytest.skip("psycopg is not installed.")

    schema = f"pelagia_test_{uuid.uuid4().hex}"
    config = CoreConfig()
    config.database.dsn = POSTGRES_TEST_DSN
    config.database.schema_name = schema

    repo = PostgresRepository(config)
    try:
        with repo.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
    except Exception as exc:
        pytest.skip(f"Postgres is not reachable at {POSTGRES_TEST_DSN}: {exc}")

    try:
        repo.initialize_schema()
    except Exception as exc:
        try:
            with repo.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"DROP SCHEMA IF EXISTS {repo.schema} CASCADE")
                connection.commit()
        except Exception:
            pass
        pytest.skip(f"Postgres test schema could not be initialized at {POSTGRES_TEST_DSN}: {exc}")

    try:
        yield repo
    finally:
        with repo.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {repo.schema} CASCADE")
            connection.commit()
        repo.close()


def test_postgres_imports_default_curation_labels_idempotently(postgres_repo):
    project = postgres_repo.create_project(f"taxonomy-{uuid.uuid4().hex}")
    dictionary = default_taxonomy_dictionary()

    first = postgres_repo.import_curation_label_dictionary(
        project_id=str(project["id"]),
        dictionary=dictionary,
    )
    second = postgres_repo.import_curation_label_dictionary(
        project_id=str(project["id"]),
        dictionary=dictionary,
    )

    assert first["dictionary_key"] == "pelagia-core@0.1.1"
    assert first["created_count"] == dictionary["selectable_count"]
    assert first["updated_count"] == 0
    assert second["created_count"] == 0
    assert second["updated_count"] == dictionary["selectable_count"]
    labels = {label["stable_concept_id"]: label for label in second["labels"]}
    assert labels["taxon_crustacea"]["parent_label_id"] == labels["taxon_arthropoda"]["id"]
    assert labels["taxon_copepoda"]["display_name"] == "Copepod"


@pytest.mark.parametrize("dataset_type", ["classification", "mask_refinement"])
def test_registry_workspace_round_trip_purge_and_reload(postgres_repo, tmp_path, dataset_type):
    import cv2
    import numpy as np

    source = tmp_path / f"{dataset_type}-source.sqlite"
    destination = tmp_path / f"{dataset_type}-export.sqlite"
    with sqlite3.connect(source) as connection:
        info = initialize_database(connection, dataset_type, name="registry-test")
        asset_id, item_id = str(uuid.uuid4()), str(uuid.uuid4())
        payload = b"registry-geometry"
        connection.execute(
            """INSERT INTO assets (
               asset_id,dataset_id,content_sha256,payload,encoding,shape_json,metadata_json,created_at
               ) VALUES (?,?,?,?,?,'[12, 16, 3]','{}','2026-08-19T00:00:00Z')""",
            (asset_id, info["dataset_id"], hashlib.sha256(payload).hexdigest(), payload, "raw"),
        )
        connection.execute(
            """INSERT INTO dataset_items (
               item_id,dataset_id,source_key,metadata_json,created_at,updated_at
               ) VALUES (?,?,?,'{}','2026-08-19T00:00:00Z','2026-08-19T00:00:00Z')""",
            (item_id, info["dataset_id"], "geometry-test"),
        )
        connection.execute(
            """INSERT INTO item_geometry (
               item_id,coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
               crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h,metadata_json
               ) VALUES (?,'source_frame_pixels',102,203,5,6,100,200,16,12,'{}')""",
            (item_id,),
        )
        relation = "classification_items" if dataset_type == "classification" else "mask_refinement_items"
        columns = "item_id,image_asset_id" if dataset_type == "classification" else "item_id,image_asset_id,candidate_mask_asset_id"
        candidate_id = None
        if dataset_type == "mask_refinement":
            candidate_id = str(uuid.uuid4())
            candidate_payload = np.ones((12, 16), dtype=np.uint8).tobytes()
            connection.execute(
                """INSERT INTO assets (
                   asset_id,dataset_id,content_sha256,payload,encoding,media_type,shape_json,dtype,metadata_json,created_at
                   ) VALUES (?,?,?,?,'raw','application/octet-stream','[12, 16]','uint8','{}','2026-08-19T00:00:00Z')""",
                (candidate_id, info["dataset_id"], hashlib.sha256(candidate_payload).hexdigest(), candidate_payload),
            )
        values = (item_id, asset_id) if dataset_type == "classification" else (item_id, asset_id, candidate_id)
        connection.execute(
            f"INSERT INTO {relation} ({columns}) VALUES ({','.join('?' for _ in values)})", values
        )
        connection.commit()
    project = postgres_repo.create_project(f"registry-{uuid.uuid4().hex}")
    scope = {"project_id": str(project["id"]), "owner_username": "registry-tester"}

    loaded = load_sqlite_workspace(postgres_repo, source, **scope)
    workspace = RegistryWorkspaceService(postgres_repo, scope["project_id"], scope["owner_username"])
    registry_item = workspace.list_items(limit=10, offset=0)["items"][0]
    assert registry_item["bbox"] == {"x": 102, "y": 203, "w": 5, "h": 6}
    assert registry_item["crop_bbox"] == {"x": 100, "y": 200, "w": 16, "h": 12}
    updated = workspace.update_geometry(
        item_id, {"x": 103, "y": 204, "w": 6, "h": 5}, expected_updated_at=registry_item["updated_at"],
    )
    assert updated["bbox"] == {"x": 103, "y": 204, "w": 6, "h": 5}
    if dataset_type == "mask_refinement":
        assert workspace.mask_layers(item_id)["candidate"]["asset_id"] == candidate_id
        candidate_display, media_type = workspace.mask_image(item_id, "candidate")
        assert media_type == "image/png"
        assert np.all(cv2.imdecode(np.frombuffer(candidate_display, dtype=np.uint8), cv2.IMREAD_UNCHANGED) == 255)
        edited_mask = np.full((12, 16), 255, dtype=np.uint8)
        edited_mask[2:5, 3:7] = 0
        ok, encoded = cv2.imencode(".png", edited_mask)
        assert ok
        saved = workspace.save_mask_revision(
            item_id, encoded.tobytes(), expected_current_annotation_id=None,
            method="manual-erase", parameters={"brush_radius_px": 2}, notes=None,
        )
        repeated = workspace.save_mask_revision(
            item_id, encoded.tobytes(), expected_current_annotation_id=saved["annotation_id"],
            method="manual-erase", parameters={"brush_radius_px": 2}, notes=None,
        )
        assert repeated["asset_id"] == saved["asset_id"]
        assert repeated["parent_annotation_id"] == saved["annotation_id"]
        assert workspace.mask_layers(item_id)["current"]["annotation_id"] == repeated["annotation_id"]
        assert workspace.mask_layers(item_id)["candidate"]["asset_id"] == candidate_id
    assert RegistryWorkspaceService(postgres_repo, scope["project_id"], "other-user").active_workspace() is None
    label = workspace.add_label({"name": "copepod", "display_name": "Copepod"})
    assert label["name"] == "copepod"

    exported = export_sqlite_workspace(
        postgres_repo, loaded["workspace_id"], destination, operation_id="export-job-1", **scope,
    )
    with sqlite3.connect(destination) as connection:
        assert validate_database(connection)["valid"] is True
        assert connection.execute(
            "SELECT bbox_x,bbox_y,bbox_w,bbox_h,crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h FROM item_geometry"
        ).fetchone() == (103, 204, 6, 5, 100, 200, 16, 12)
        if dataset_type == "mask_refinement":
            assert connection.execute("SELECT candidate_mask_asset_id FROM mask_refinement_items").fetchone()[0] == candidate_id
            assert connection.execute("SELECT count(*) FROM mask_annotations WHERE is_current=1").fetchone()[0] == 1
            assert connection.execute("SELECT count(*) FROM mask_annotations").fetchone()[0] == 2
        label_table = "classification_labels" if dataset_type == "classification" else "annotation_labels"
        assert connection.execute(f"SELECT count(*) FROM {label_table}").fetchone()[0] == 1
    assert exported["parent_revision_id"]
    retried = export_sqlite_workspace(
        postgres_repo, loaded["workspace_id"], destination, operation_id="export-job-1", **scope,
    )
    assert retried["reused"] is True
    assert retried["revision_id"] == exported["revision_id"]

    purged = workspace.purge()
    assert purged["status"] == "purged"
    reloaded = load_sqlite_workspace(postgres_repo, destination, **scope)
    assert reloaded["reused"] is False
    assert reloaded["workspace_id"] != loaded["workspace_id"]
    reloaded_workspace = RegistryWorkspaceService(postgres_repo, scope["project_id"], scope["owner_username"])
    assert reloaded_workspace.item_detail(item_id)["bbox"] == {"x": 103, "y": 204, "w": 6, "h": 5}
    if dataset_type == "mask_refinement":
        assert reloaded_workspace.mask_layers(item_id)["candidate"]["asset_id"] == candidate_id
        assert reloaded_workspace.mask_layers(item_id)["current"]["annotation_id"] == repeated["annotation_id"]


def test_registry_workspace_has_one_active_owner_project_workspace(postgres_repo):
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT indexdef FROM pg_indexes WHERE schemaname=%s
            AND tablename='registry_workspaces' AND indexdef ILIKE 'CREATE UNIQUE INDEX%%'
            AND indexdef LIKE '%%WHERE is_active%%'""",
            (postgres_repo.schema,),
        )
        definition = cursor.fetchone()["indexdef"]
    assert "UNIQUE" in definition
    assert "WHERE is_active" in definition


def test_registry_workspace_catalog_is_scoped_and_can_reactivate(postgres_repo, tmp_path):
    project = postgres_repo.create_project(f"registry-catalog-{uuid.uuid4().hex}")
    scope = {"project_id": str(project["id"]), "owner_username": "registry-tester"}
    sources = []
    for index in range(2):
        source = tmp_path / f"catalog-{index}.sqlite"
        with sqlite3.connect(source) as connection:
            initialize_database(connection, "classification", name=f"catalog-{index}")
            connection.commit()
        sources.append(source)

    first = load_sqlite_workspace(postgres_repo, sources[0], **scope)
    second = load_sqlite_workspace(postgres_repo, sources[1], **scope)
    workspace = RegistryWorkspaceService(postgres_repo, **scope)

    catalog = workspace.list_workspaces()
    assert [entry["workspace_id"] for entry in catalog] == [second["workspace_id"], first["workspace_id"]]
    assert catalog[0]["is_active"] is True
    assert catalog[1]["is_active"] is False
    assert catalog[0]["item_count"] == 0
    assert RegistryWorkspaceService(postgres_repo, scope["project_id"], "other-user").list_workspaces() == []

    activated = workspace.activate_workspace(first["workspace_id"])
    assert activated["workspace_id"] == first["workspace_id"]
    assert [entry["workspace_id"] for entry in workspace.list_workspaces()] == [first["workspace_id"], second["workspace_id"]]

    with pytest.raises(KeyError):
        RegistryWorkspaceService(postgres_repo, scope["project_id"], "other-user").activate_workspace(first["workspace_id"])


def test_registry_workspace_details_include_pelagia_inference_contract(postgres_repo, tmp_path):
    source = tmp_path / "pelagia-derived.sqlite"
    with sqlite3.connect(source) as connection:
        initialize_database(
            connection, "classification", name="pelagia-derived",
            metadata={"pelagia": {"selection_fingerprint_sha256": "selection-digest"}},
        )
        connection.commit()
    project = postgres_repo.create_project(f"registry-details-{uuid.uuid4().hex}")
    scope = {"project_id": str(project["id"]), "owner_username": "registry-tester"}
    loaded = load_sqlite_workspace(postgres_repo, source, **scope)
    run_id = "pelagia-inference-run"
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""INSERT INTO {postgres_repo.schema}.registry_inference_runs
            (workspace_id,inference_run_id,dataset_fingerprint_sha256,name,status,created_at,metadata)
            VALUES (%s,%s,'dataset-digest','Pelagia inference','complete','2026-08-19T00:00:00Z',
              '{{"evaluation_metrics":{{"accuracy":0.88}}}}'::jsonb)""",
            (loaded["workspace_id"], run_id),
        )
        cursor.execute(
            f"""INSERT INTO {postgres_repo.schema}.registry_model_evidence
            (workspace_id,evidence_id,inference_run_id,item_id,prediction_confidence,
             nearest_neighbor_similarity,top_k_label_agreement,weighted_label_support,label_margin,created_at)
            VALUES (%s,'evidence-1',%s,'roi-1',0.75,0.9,0.6,0.7,0.2,'2026-08-19T00:00:00Z')""",
            (loaded["workspace_id"], run_id),
        )
        connection.commit()

    details = RegistryWorkspaceService(postgres_repo, **scope).details()

    assert details["dataset"]["lifecycle"] == "working"
    assert details["dataset"]["metadata"]["pelagia"]["selection_fingerprint_sha256"] == "selection-digest"
    assert len(details["inference_sources"]) == 1
    inference = details["inference_sources"][0]
    assert inference["source_key"] == f"registry:{run_id}"
    assert inference["evidence"]["item_count"] == 1
    assert inference["evidence"]["aggregates"]["mean_confidence"] == pytest.approx(0.75)
    assert inference["performance_metrics"]["run · metadata · evaluation_metrics · accuracy"] == 0.88
    assert inference["run"]["status"] == "complete"


def test_registry_source_overwrite_rejects_changed_bytes(postgres_repo, tmp_path):
    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as connection:
        initialize_database(connection, "classification", name="hash-guard")
        connection.commit()
    project = postgres_repo.create_project(f"registry-hash-{uuid.uuid4().hex}")
    scope = {"project_id": str(project["id"]), "owner_username": "registry-tester"}
    loaded = load_sqlite_workspace(postgres_repo, source, **scope)
    with source.open("ab") as stream:
        stream.write(b"externally-changed")

    with pytest.raises(ValueError, match="changed after it was loaded"):
        export_sqlite_workspace(postgres_repo, loaded["workspace_id"], source, replace_source=True, **scope)

    workspace = RegistryWorkspaceService(postgres_repo, scope["project_id"], scope["owner_username"])
    assert workspace.active_workspace()["status"] == "loaded"


def test_queue_status_snapshot_is_complete_bounded_and_consistent_during_writes(postgres_repo):
    project = postgres_repo.create_project(f"queue-status-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    rows = []
    for index in range(1001):
        unit = "frames" if index % 2 == 0 else "rois"
        stage = PipelineStage.SEGMENT.value if unit == "frames" else PipelineStage.ROI_REFINEMENT.value
        progress = {
            "schema_version": 2,
            "attempt_count": 0,
            "unit": unit,
            "total": 10,
            "total_known": True,
            "completed": 10 if index < 1000 else 0,
            "failed": 0,
            "skipped": 0,
        }
        rows.append((project_id, stage, "succeeded" if index < 1000 else "queued", json.dumps(progress)))
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                f"""INSERT INTO {postgres_repo.schema}.processing_jobs (project_id, stage, status, progress)
                VALUES (%s::uuid, %s::{postgres_repo.schema}.stage_name,
                    %s::{postgres_repo.schema}.job_status, %s::jsonb)""",
                rows,
            )
        connection.commit()

    snapshot = postgres_repo.get_queue_status_snapshot(project_id=project_id, job_limit=7)

    assert snapshot["total"]["job_count"] == 1001
    assert len(snapshot["jobs"]) == 7
    assert sum(row["job_count"] for row in snapshot["by_stage"]) == 1001
    assert sum(row["job_count"] for row in snapshot["work_groups"]) == 1001
    assert {(row["stage"], row["unit"]) for row in snapshot["work_groups"]} == {
        (PipelineStage.SEGMENT.value, "frames"),
        (PipelineStage.ROI_REFINEMENT.value, "rois"),
    }
    # Active rows sort ahead of recent history even with a small display bound.
    assert snapshot["jobs"][0]["status"] == JobStatus.QUEUED.value

    def writer():
        for _ in range(20):
            postgres_repo.create_job(PipelineStage.SEGMENT, project_id=project_id, payload={})

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(writer)
        while not future.done():
            concurrent = postgres_repo.get_queue_status_snapshot(project_id=project_id, job_limit=3)
            total = concurrent["total"]["job_count"]
            assert sum(row["job_count"] for row in concurrent["by_stage"]) == total
            assert sum(row["job_count"] for row in concurrent["work_groups"]) == total
        future.result()

    with ThreadPoolExecutor(max_workers=4) as executor:
        snapshots = list(executor.map(
            lambda _: postgres_repo.get_queue_status_snapshot(project_id=project_id, job_limit=3),
            range(4),
        ))
    assert {item["total"]["job_count"] for item in snapshots} == {1021}


def test_queue_status_job_and_group_estimates_agree_on_unavailable_work(postgres_repo):
    from Pelagia.services.queue_status import QueueStatusService

    project = postgres_repo.create_project(f"queue-estimates-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    service = QueueStatusService(postgres_repo)
    now = datetime.now(timezone.utc)
    for scenario in ("healthy", "expired_lease", "stale_progress", "finalizing", "unknown_total"):
        job_id = str(uuid.uuid4())
        progress = {
            "unit": "frames", "total": 10, "completed": 5,
            "attempt_count": 0, "rates": {"units_per_second": 2},
            "recorded_at": (now - timedelta(minutes=20) if scenario == "stale_progress" else now).isoformat(),
        }
        if scenario == "finalizing":
            progress["completed"] = 10
        if scenario == "unknown_total":
            # Legacy workers used zero before learning the total. Group and
            # individual normalization must interpret that identically.
            progress["total"] = 0
        expiry = now - timedelta(seconds=5) if scenario == "expired_lease" else now + timedelta(minutes=10)
        with postgres_repo.connect() as connection:
            connection.execute(
                f"""INSERT INTO {postgres_repo.schema}.processing_jobs
                    (id, project_id, stage, status, progress, lease_expires_at)
                    VALUES (%s, %s, 'segment', 'working', %s::jsonb, %s)""",
                (job_id, project_id, json.dumps(progress), expiry),
            )
            connection.commit()
        result = service.snapshot(project_id=project_id, job_ids=[job_id])
        job_progress = result["jobs"]["items"][0]["progress"]
        group = result["work_groups"][0]
        if scenario == "healthy":
            assert job_progress["rate"]["units_per_second"] == group["rate"]["units_per_second"] == 2
            assert job_progress["eta"]["seconds"] == group["eta"]["seconds"] == 2.5
        else:
            assert job_progress["eta"]["reason"] == group["eta"]["reason"] == scenario
            assert job_progress["eta"]["seconds"] is None
            assert group["eta"]["seconds"] is None
        if scenario == "unknown_total":
            assert group["active_progress"]["percent"] is None
            assert result["health"]["progress_coverage_percent"] == 0


def test_postgres_repository_registers_frames_and_jobs(postgres_repo):
    postgres_repo.ensure_default_project()
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    frame_payload = b"frame-bytes"

    planned_run = PlannedRun(
        manifest=RunManifest(
            run_id=run_id,
            run_key=f"test-run-{uuid.uuid4().hex}",
            instrument="pytest",
            source_path="/tmp/example.avi",
            source_type=AssetKind.VIDEO.value,
            created_at=datetime.now(timezone.utc),
            metadata={"suite": "postgres"},
            assets=[
                RawAssetManifest(
                    asset_id=asset_id,
                    filename="example.avi",
                    path="/tmp/example.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=123,
                    checksum="sha256:test",
                    collections=["skq202510S-T1", "test"],
                    media_count=1,
                    metadata={"kind": "fixture"},
                )
            ],
        ),
        jobs=[
            WorkItem(
                job_id=job_id,
                run_id=run_id,
                stage=PipelineStage.EXTRACT_FRAMES,
                asset_id=asset_id,
                payload={"source_path": "/tmp/example.avi"},
            )
        ],
    )

    run_row = postgres_repo.register_planned_run(planned_run, project_id=DEFAULT_PROJECT_ID)
    assert str(run_row["run"]["id"]) == run_id
    assert str(run_row["run"]["project_id"]) == DEFAULT_PROJECT_ID
    assert run_row["asset_count"] == 1
    assert run_row["job_count"] == 1
    summary_job = postgres_repo.create_job(
        PipelineStage.SEGMENT,
        project_id=DEFAULT_PROJECT_ID,
        run_id=run_id,
        asset_id=asset_id,
        payload={"requested_frame_ids": [str(uuid.uuid4()) for _ in range(20)]},
    )
    assert str(summary_job["project_id"]) == DEFAULT_PROJECT_ID
    completed_summary_job = postgres_repo.complete_job(
        str(summary_job["id"]),
        result={"detection_ids": [str(uuid.uuid4()) for _ in range(20)]},
    )
    assert completed_summary_job is not None
    summary_jobs = postgres_repo.list_jobs(stage=PipelineStage.SEGMENT.value, limit=1, include_details=False)
    assert "payload" not in summary_jobs[0]
    assert "result" not in summary_jobs[0]
    assert summary_jobs[0]["payload_bytes"] > 0
    assert summary_jobs[0]["result_bytes"] > 0
    detail_jobs = postgres_repo.list_jobs(stage=PipelineStage.SEGMENT.value, limit=1, include_details=True)
    assert len(detail_jobs[0]["payload"]["requested_frame_ids"]) == 20
    assert len(detail_jobs[0]["result"]["detection_ids"]) == 20

    assets = postgres_repo.list_assets(run_id)
    assert len(assets) == 1
    assert str(assets[0]["id"]) == asset_id
    assert str(assets[0]["project_id"]) == DEFAULT_PROJECT_ID
    assert assets[0]["filename"] == "example.avi"
    assert assets[0]["collections"] == ["skq202510S-T1", "test"]
    assert postgres_repo.list_collections() == [
        {"collection": "skq202510S-T1", "asset_count": 1},
        {"collection": "test", "asset_count": 1},
    ]
    assert postgres_repo.list_assets(collection="test")[0]["filename"] == "example.avi"

    inserted_frames = postgres_repo.replace_frames(
        run_id,
        [
            FrameRecord(
                asset_id=asset_id,
                frame_index=1,
                width=4,
                height=3,
                preview_thumbhash=frame_payload,
                kvstore_hash="kvstore-key-1",
                captured_at=datetime(2026, 8, 21, 18, 4, 5, 125000, tzinfo=timezone.utc),
                source_ref="/tmp/example.avi",
                bbox_x=7,
                bbox_y=8,
                payload_ref="kvstore-key-1",
                payload_encoding="zstd",
                payload_format="zstd_ndarray_c_order",
                payload_dtype="uint8",
                payload_shape=[3, 4],
                metadata={"kvstore_key": "kvstore-key-1"},
            ),
            FrameRecord(
                asset_id=asset_id,
                frame_index=2,
                width=4,
                height=3,
                preview_thumbhash=frame_payload,
                kvstore_hash="kvstore-key-2",
                source_ref="/tmp/example.avi",
                payload_ref="kvstore-key-2",
                payload_encoding="zstd",
                payload_format="zstd_ndarray_c_order",
                payload_dtype="uint8",
                payload_shape=[3, 4],
                metadata={"kvstore_key": "kvstore-key-2"},
            ),
        ],
    )
    assert len(inserted_frames) == 2
    assert inserted_frames[0]["kvstore_hash"] == "kvstore-key-1"
    assert inserted_frames[0]["payload_ref"] == "kvstore-key-1"
    assert inserted_frames[0]["payload_encoding"] == "zstd"
    assert inserted_frames[0]["payload_shape"] == [3, 4]
    assert inserted_frames[0]["bbox_x"] == 7
    assert inserted_frames[0]["bbox_y"] == 8

    frame_record = FrameRecord.from_row(inserted_frames[0])
    assert frame_record.id == inserted_frames[0]["id"]
    assert frame_record.run_id == run_id
    assert frame_record.payload_ref == "kvstore-key-1"
    assert frame_record.payload_shape == [3, 4]

    frames = postgres_repo.list_frames(asset_id)
    assert [frame["frame_index"] for frame in frames] == [2, 1]
    assert postgres_repo.list_frames(asset_id, limit=1, offset=1)[0]["frame_index"] == 1
    assert frames[1]["metadata"]["kvstore_key"] == "kvstore-key-1"
    assert postgres_repo.count_frames(asset_id) == 2
    assert postgres_repo.get_frame_record(inserted_frames[0]["id"]).payload_ref == "kvstore-key-1"
    assert postgres_repo.list_frame_records(asset_id)[1].bbox_x == 7

    detection_records = [
        DetectionRecord(
            run_id=run_id,
            frame_id=inserted_frames[0]["id"],
            roi_index=1,
            bbox_x=1,
            bbox_y=2,
            bbox_w=3,
            bbox_h=4,
            area=12.0,
            perimeter=14.0,
            major_axis_length=4.0,
            minor_axis_length=3.0,
            min_gray_value=5,
            mean_gray_value=6.5,
            roi_payload=b"roi-bytes",
            mask_payload=b"mask-bytes",
            crop_bbox_x=0,
            crop_bbox_y=1,
            crop_bbox_w=5,
            crop_bbox_h=6,
            roi_encoding="raw",
            roi_format="raw_ndarray_c_order",
            roi_dtype="uint8",
            roi_shape=[4, 5],
            mask_encoding="raw",
            mask_format="raw_ndarray_c_order",
            mask_dtype="uint8",
            mask_shape=[4, 5],
            metadata={"kind": "roi"},
        ),
        DetectionRecord(
            run_id=run_id,
            frame_id=inserted_frames[1]["id"],
            roi_index=1,
            bbox_x=10,
            bbox_y=20,
            bbox_w=30,
            bbox_h=40,
            area=1200.0,
            perimeter=140.0,
            major_axis_length=40.0,
            minor_axis_length=30.0,
            min_gray_value=8,
            mean_gray_value=9.5,
            roi_payload=b"other-roi",
            mask_payload=b"other-mask",
            roi_encoding="png",
            roi_format="png",
            mask_encoding="png",
            mask_format="png",
            metadata={"kind": "roi"},
        ),
    ]
    inserted_detections = postgres_repo.replace_detections(run_id, asset_id, detection_records)
    assert len(inserted_detections) == 2
    assert inserted_detections[0]["roi_payload"] == b"roi-bytes"
    assert inserted_detections[0]["mask_payload"] == b"mask-bytes"
    assert inserted_detections[0]["crop_bbox_w"] == 5
    assert inserted_detections[0]["roi_encoding"] == "raw"
    assert inserted_detections[0]["mask_shape"] == [4, 5]

    inserted_ids = postgres_repo.replace_frame_detections(
        run_id,
        [str(frame["id"]) for frame in inserted_frames],
        detection_records,
        returning="ids",
    )
    assert len(inserted_ids) == 2
    assert all(set(row) == {"id"} for row in inserted_ids)
    inserted_detections = postgres_repo.replace_detections(run_id, asset_id, detection_records)

    rollback_records = [
        DetectionRecord(
            run_id=run_id,
            frame_id=inserted_frames[index % 2]["id"],
            roi_index=index + 10,
            bbox_x=0,
            bbox_y=0,
            bbox_w=0 if index == 250 else 1,
            bbox_h=1,
            area=1.0,
            perimeter=4.0,
            major_axis_length=1.0,
            minor_axis_length=1.0,
            min_gray_value=0,
            mean_gray_value=0.0,
            roi_payload=None,
        )
        for index in range(251)
    ]
    with pytest.raises(postgres.psycopg.errors.CheckViolation):
        postgres_repo.replace_frame_detections(
            run_id,
            [str(frame["id"]) for frame in inserted_frames],
            rollback_records,
            returning="ids",
        )

    detections = postgres_repo.list_detections(asset_id)
    assert len(detections) == 2
    assert [detection["frame_id"] for detection in detections] == [
        inserted_frames[1]["id"],
        inserted_frames[0]["id"],
    ]
    assert postgres_repo.list_detections(asset_id, limit=1, offset=1)[0]["frame_id"] == inserted_frames[0]["id"]
    assert detections[1]["mask_payload"] == b"mask-bytes"
    filtered_detections = postgres_repo.list_detections(
        asset_id,
        start_frame=1,
        end_frame=1,
        roi_encoding="raw",
        min_area=10,
        max_area=20,
        limit=1,
    )
    assert len(filtered_detections) == 1
    assert filtered_detections[0]["mask_payload"] == b"mask-bytes"
    assert postgres_repo.list_detections(asset_id, roi_encoding="raw", max_bbox_w=2) == []
    global_detections = postgres_repo.list_detections(collection="test", roi_encoding="raw", limit=10)
    assert len(global_detections) == 1
    assert str(global_detections[0]["asset_id"]) == asset_id
    assert global_detections[0]["frame_index"] == 1
    detection_lookup = postgres_repo.get_detection(str(filtered_detections[0]["id"]))
    assert detection_lookup is not None
    assert detection_lookup["roi_payload"] == b"roi-bytes"
    assert str(detection_lookup["asset_id"]) == asset_id
    assert detection_lookup["captured_at"] == datetime(2026, 8, 21, 18, 4, 5, 125000, tzinfo=timezone.utc)
    bulk_detection_lookup = postgres_repo.get_detections(
        [
            str(detections[1]["id"]),
            str(detections[0]["id"]),
            str(uuid.uuid4()),
        ]
    )
    assert [str(row["id"]) for row in bulk_detection_lookup] == [
        str(detections[1]["id"]),
        str(detections[0]["id"]),
    ]
    detection_record = DetectionRecord.from_row(filtered_detections[0])
    assert detection_record.id == str(filtered_detections[0]["id"])
    assert detection_record.roi_payload == b"roi-bytes"
    assert detection_record.mask_shape == [4, 5]
    assert postgres_repo.list_detection_records(asset_id)[1].mask_payload == b"mask-bytes"

    refined = postgres_repo.upsert_refined_detections(
        [
            (
                str(filtered_detections[0]["id"]),
                DetectionRecord(
                    run_id=run_id,
                    frame_id=inserted_frames[0]["id"],
                    roi_index=1,
                    bbox_x=1,
                    bbox_y=2,
                    bbox_w=3,
                    bbox_h=4,
                    area=12.0,
                    perimeter=14.0,
                    major_axis_length=4.0,
                    minor_axis_length=3.0,
                    min_gray_value=5,
                    mean_gray_value=6.5,
                    roi_payload=b"refined-roi",
                    mask_payload=b"refined-mask",
                    crop_bbox_x=0,
                    crop_bbox_y=1,
                    crop_bbox_w=5,
                    crop_bbox_h=6,
                    roi_encoding="raw",
                    roi_format="raw_ndarray_c_order",
                    roi_dtype="uint8",
                    roi_shape=[4, 5],
                    mask_encoding="raw",
                    mask_format="raw_ndarray_c_order",
                    mask_dtype="uint8",
                    mask_shape=[4, 5],
                    metadata={"detection_stage": "refined", "refinement_method": "identity"},
                ),
            )
        ]
    )
    assert len(refined) == 1
    assert refined[0]["candidate_detection_id"] == filtered_detections[0]["id"]
    assert refined[0]["roi_dtype"] == "uint8"
    assert refined[0]["roi_shape"] == [4, 5]
    assert refined[0]["mask_dtype"] == "uint8"
    assert refined[0]["metadata"]["refinement_method"] == "identity"
    refined_again = postgres_repo.upsert_refined_detections(
        [
            (
                str(filtered_detections[0]["id"]),
                DetectionRecord(
                    run_id=run_id,
                    frame_id=inserted_frames[0]["id"],
                    roi_index=2,
                    bbox_x=2,
                    bbox_y=3,
                    bbox_w=4,
                    bbox_h=5,
                    area=20.0,
                    perimeter=18.0,
                    major_axis_length=5.0,
                    minor_axis_length=4.0,
                    min_gray_value=7,
                    mean_gray_value=8.5,
                    roi_payload=b"refined-roi-child",
                    mask_payload=b"refined-mask-child",
                    crop_bbox_x=1,
                    crop_bbox_y=2,
                    crop_bbox_w=6,
                    crop_bbox_h=7,
                    roi_encoding="raw",
                    roi_format="raw_ndarray_c_order",
                    roi_dtype="uint8",
                    roi_shape=[5, 4],
                    mask_encoding="raw",
                    mask_format="raw_ndarray_c_order",
                    mask_dtype="uint8",
                    mask_shape=[5, 4],
                    metadata={
                        "detection_stage": "refined",
                        "refinement_method": "identity",
                        "refinement_role": "residual_child",
                        "roi_measurements": {"version": 1, "area_method": "opencv_external_contours"},
                    },
                ),
            )
        ]
    )
    assert len(refined_again) == 1
    assert refined_again[0]["candidate_detection_id"] == filtered_detections[0]["id"]
    assert refined_again[0]["id"] != refined[0]["id"]
    listed_refined = postgres_repo.list_refined_detections(str(inserted_frames[0]["id"]), limit=10)
    assert [row["id"] for row in listed_refined] == [refined[0]["id"], refined_again[0]["id"]]
    assert all("roi_payload" not in row for row in listed_refined)
    assert listed_refined[0]["mask_payload"] == b"refined-mask"
    assert listed_refined[1]["mask_payload"] is None
    assert listed_refined[1]["mask_payload_bytes"] == len(b"refined-mask-child")
    assert postgres_repo.list_refined_detections(
        str(inserted_frames[0]["id"]), limit=1, offset=1,
    )[0]["id"] == refined_again[0]["id"]
    assert postgres_repo.list_refined_detections(
        str(inserted_frames[0]["id"]), project_id=str(uuid.uuid4()),
    ) == []
    latest_refined = postgres_repo.get_refined_detection_for_candidate(str(filtered_detections[0]["id"]))
    assert latest_refined is not None
    assert latest_refined["candidate_detection_id"] == filtered_detections[0]["id"]
    assert latest_refined["captured_at"] == datetime(2026, 8, 21, 18, 4, 5, 125000, tzinfo=timezone.utc)

    detection_stats = postgres_repo.list_asset_detection_stats(collection="test", limit=10)
    assert detection_stats["summary"] == {
        "total_asset_count": 1,
        "identified_asset_count": 1,
        "total_detection_count": 2,
    }
    assert len(detection_stats["assets"]) == 1
    assert str(detection_stats["assets"][0]["asset_id"]) == asset_id
    assert detection_stats["assets"][0]["filename"] == "example.avi"
    assert detection_stats["assets"][0]["frame_count"] == 2
    assert detection_stats["assets"][0]["detection_count"] == 2

    queued = postgres_repo.get_job(job_id)
    assert queued is not None
    assert queued["status"] == "queued"
    project_status = postgres_repo.get_status_summary(project_id=DEFAULT_PROJECT_ID)
    assert project_status["queue"] == {"queued": 1, "succeeded": 1}
    assert postgres_repo.get_status_summary(project_id=str(uuid.uuid4()))["queue"] == {}
    events = postgres_repo.list_job_events(job_id=job_id)
    assert [event["event_type"] for event in events] == ["job.created"]
    assert events[0]["payload"]["stage"] == PipelineStage.EXTRACT_FRAMES.value
    mirrored_logs = postgres_repo.list_logs(job_id=job_id, event_type="job.created")
    assert len(mirrored_logs) == 1
    assert mirrored_logs[0]["event_type"] == "job.created"
    assert mirrored_logs[0]["level"] == "info"
    assert mirrored_logs[0]["payload"]["stage"] == PipelineStage.EXTRACT_FRAMES.value

    claimed = postgres_repo.claim_jobs("pytest-worker", stages=[PipelineStage.EXTRACT_FRAMES], limit=1)
    assert len(claimed) == 1
    assert str(claimed[0]["id"]) == job_id
    assert str(claimed[0]["project_id"]) == DEFAULT_PROJECT_ID
    assert claimed[0]["status"] == "leased"
    events = postgres_repo.list_job_events(job_id=job_id)
    assert [event["event_type"] for event in events] == ["job.leased", "job.created"]
    assert events[0]["payload"]["worker_id"] == "pytest-worker"

    completed = postgres_repo.complete_job(job_id, result={"frames": len(frames)})
    assert completed is not None
    assert completed["status"] == "succeeded"
    assert completed["result"]["frames"] == 2
    events = postgres_repo.list_job_events(job_id=job_id)
    assert [event["event_type"] for event in events] == [
        "job.completed",
        "job.leased",
        "job.created",
    ]
    assert events[0]["payload"]["result"] == {"frames": 2}

    postgres_repo.touch_worker(
        "pytest-worker",
        status="idle",
        capabilities=[PipelineStage.EXTRACT_FRAMES.value],
        pid=12345,
    )
    postgres_repo.request_worker_shutdown("pytest-worker", reason="test")
    worker_events = [
        event
        for event in postgres_repo.list_job_events(limit=100)
        if event["payload"].get("worker_id") == "pytest-worker"
    ]
    assert "worker.touched" not in [event["event_type"] for event in worker_events]

    log_row = postgres_repo.append_log(
        event_type="pipeline.extract_timed",
        message="Extracted frames",
        level="info",
        logger="pelagia.tests",
        run_id=run_id,
        asset_id=asset_id,
        job_id=job_id,
        worker_id="pytest-worker",
        duration_ms=42.5,
        payload={"frame_count": 2},
    )
    assert log_row["event_type"] == "pipeline.extract_timed"
    assert str(log_row["project_id"]) == DEFAULT_PROJECT_ID
    assert log_row["duration_ms"] == 42.5
    logs = postgres_repo.list_logs(
        run_id=run_id,
        worker_id="pytest-worker",
        event_type="pipeline.extract_timed",
    )
    assert logs[0]["payload"] == {"frame_count": 2}
    assert "worker.shutdown_requested" in [event["event_type"] for event in worker_events]


def test_postgres_operational_log_cleanup_is_bounded_and_preserves_recent_rows(postgres_repo):
    project = postgres_repo.ensure_default_project()
    project_id = str(project["id"])
    old_progress = postgres_repo.append_job_event(None, "job.progress_updated", {"worker_id": "old"})
    old_event = postgres_repo.append_job_event(None, "job.completed", {})
    old_debug = postgres_repo.append_log(project_id=project_id, event_type="job.progress_updated", level="debug")
    old_routine = postgres_repo.append_log(project_id=project_id, event_type="pipeline.completed", level="info")
    old_routine_2 = postgres_repo.append_log(project_id=project_id, event_type="pipeline.completed", level="info")
    old_important = postgres_repo.append_log(project_id=project_id, event_type="jobs.deleted", level="info")
    recent = postgres_repo.append_log(project_id=project_id, event_type="pipeline.recent", level="info")
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            for table, row_id, age_days in (
                ("job_events", old_progress["id"], 10),
                ("job_events", old_event["id"], 100),
                ("logs", old_debug["id"], 10),
                ("logs", old_routine["id"], 100),
                ("logs", old_routine_2["id"], 100),
                ("logs", old_important["id"], 400),
            ):
                cursor.execute(
                    f"UPDATE {postgres_repo.schema}.{table} SET created_at = NOW() - (%s * INTERVAL '1 day') WHERE id = %s",
                    (age_days, row_id),
                )
        connection.commit()

    postgres_repo.config.logging.cleanup_batch_size = 1
    preview = postgres_repo.prune_operational_logs(dry_run=True)
    assert preview["deleted"]["logs_routine"] == 2
    assert all(count == 1 for name, count in preview["deleted"].items() if name != "logs_routine")
    first_pass = postgres_repo.prune_operational_logs()
    assert all(count == 1 for count in first_pass["deleted"].values())
    assert postgres_repo.prune_operational_logs()["deleted"]["logs_routine"] == 1
    assert postgres_repo.prune_operational_logs(dry_run=True)["deleted"] == {
        name: 0 for name in preview["deleted"]
    }
    assert postgres_repo.list_logs(event_type="pipeline.recent")[0]["id"] == recent["id"]


def test_postgres_job_telemetry_avoids_redundant_log_rows(postgres_repo):
    project = postgres_repo.create_project(f"telemetry-{uuid.uuid4().hex}")
    job = postgres_repo.create_job(PipelineStage.EXTRACT_FRAMES, project_id=str(project["id"]))
    claim = postgres_repo.claim_jobs("telemetry-worker", stages=[PipelineStage.EXTRACT_FRAMES], limit=1)[0]
    job_id = str(job["id"])
    lease_token = str(claim["lease_token"])

    assert postgres_repo.heartbeat("telemetry-worker", job_id, lease_token=lease_token) is not None
    assert postgres_repo.update_job_progress(
        job_id, {"completed": 1}, worker_id="telemetry-worker", lease_token=lease_token,
    ) is not None
    postgres_repo.touch_worker("telemetry-worker", status="working", leased_job_id=job_id)

    event_types = [row["event_type"] for row in postgres_repo.list_job_events(job_id=job_id)]
    assert "job.progress_updated" in event_types
    assert "job.heartbeat" not in event_types
    assert postgres_repo.list_logs(job_id=job_id, event_type="job.progress_updated") == []
    worker_events = postgres_repo.list_job_events(limit=100)
    assert not any(row["event_type"] in {"worker.heartbeat", "worker.touched"} for row in worker_events)


def test_postgres_frame_processing_status_projection_tracks_stage_and_counts(postgres_repo):
    postgres_repo.ensure_default_project()
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    planned_run = PlannedRun(
        manifest=RunManifest(
            run_id=run_id,
            run_key=f"status-run-{uuid.uuid4().hex}",
            instrument="pytest",
            source_path="/tmp/status.avi",
            source_type=AssetKind.VIDEO.value,
            created_at=datetime.now(timezone.utc),
            assets=[
                RawAssetManifest(
                    asset_id=asset_id,
                    filename="status.avi",
                    path="/tmp/status.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=456,
                    checksum="sha256:status",
                    collections=["status-test"],
                )
            ],
        ),
        jobs=[],
    )
    postgres_repo.register_planned_run(planned_run, project_id=DEFAULT_PROJECT_ID)
    frames = postgres_repo.replace_frames(
        run_id,
        [
            FrameRecord(
                asset_id=asset_id,
                frame_index=1,
                width=4,
                height=4,
                preview_thumbhash=b"thumb-1",
                kvstore_hash="frame-1",
                payload_ref="frame-1",
                payload_encoding="zstd",
                payload_format="zstd_ndarray_c_order",
                payload_dtype="uint8",
                payload_shape=[4, 4],
            ),
            FrameRecord(
                asset_id=asset_id,
                frame_index=2,
                width=4,
                height=4,
                preview_thumbhash=b"thumb-2",
                kvstore_hash="frame-2",
                payload_ref="frame-2",
                payload_encoding="zstd",
                payload_format="zstd_ndarray_c_order",
                payload_dtype="uint8",
                payload_shape=[4, 4],
            ),
        ],
    )
    frame_ids = [str(row["id"]) for row in frames]

    assert postgres_repo.ensure_frame_status_rows(project_id=DEFAULT_PROJECT_ID, asset_id=asset_id) == 2
    preprocess_job = postgres_repo.create_job(
        PipelineStage.PREPROCESS_FRAMES,
        project_id=DEFAULT_PROJECT_ID,
        run_id=run_id,
        asset_id=asset_id,
        payload={"frame_ids": [frame_ids[0]]},
    )
    queued_preprocess_rows = postgres_repo.list_frame_status(
        project_id=DEFAULT_PROJECT_ID,
        preprocessing_status=[JobStatus.QUEUED.value],
    )
    assert [str(row["frame_id"]) for row in queued_preprocess_rows["frames"]] == [frame_ids[0]]
    assert (
        postgres_repo.upsert_frame_stage_status(
            project_id=DEFAULT_PROJECT_ID,
            frame_ids=[frame_ids[0]],
            stage=PipelineStage.PREPROCESS_FRAMES,
            status=JobStatus.WORKING,
            job_id=str(preprocess_job["id"]),
        )
        == 1
    )
    working_rows = postgres_repo.list_frame_status(
        project_id=DEFAULT_PROJECT_ID,
        preprocessing_status=[JobStatus.WORKING.value],
    )
    assert [str(row["frame_id"]) for row in working_rows["frames"]] == [frame_ids[0]]

    detections = postgres_repo.replace_detections(
        run_id,
        asset_id,
        [
            DetectionRecord(
                run_id=run_id,
                frame_id=frame_ids[0],
                roi_index=1,
                bbox_x=0,
                bbox_y=0,
                bbox_w=2,
                bbox_h=2,
                area=4,
                perimeter=8,
                major_axis_length=2,
                minor_axis_length=2,
                min_gray_value=1,
                mean_gray_value=2.0,
                roi_payload=None,
            ),
            DetectionRecord(
                run_id=run_id,
                frame_id=frame_ids[1],
                roi_index=1,
                bbox_x=0,
                bbox_y=0,
                bbox_w=2,
                bbox_h=2,
                area=4,
                perimeter=8,
                major_axis_length=2,
                minor_axis_length=2,
                min_gray_value=1,
                mean_gray_value=2.0,
                roi_payload=None,
            ),
        ],
    )
    refined_job = postgres_repo.create_job(
        PipelineStage.ROI_REFINEMENT,
        project_id=DEFAULT_PROJECT_ID,
        run_id=run_id,
        asset_id=asset_id,
        payload={"detection_ids": [str(detections[0]["id"])]},
    )
    queued_refinement_rows = postgres_repo.list_frame_status(
        project_id=DEFAULT_PROJECT_ID,
        roi_refinement_status=[JobStatus.QUEUED.value],
    )
    assert [str(row["frame_id"]) for row in queued_refinement_rows["frames"]] == [frame_ids[0]]
    postgres_repo.upsert_refined_detections(
        [
            (
                str(detections[0]["id"]),
                DetectionRecord(
                    run_id=run_id,
                    frame_id=frame_ids[0],
                    roi_index=1,
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=2,
                    bbox_h=2,
                    area=4,
                    perimeter=8,
                    major_axis_length=2,
                    minor_axis_length=2,
                    min_gray_value=1,
                    mean_gray_value=2.0,
                    roi_payload=None,
                ),
            )
        ],
        job_id=str(refined_job["id"]),
        project_id=DEFAULT_PROJECT_ID,
    )
    assert postgres_repo.refresh_frame_status_counts(project_id=DEFAULT_PROJECT_ID, asset_id=asset_id) == 2

    refined_ids = postgres_repo.list_frame_status_ids(project_id=DEFAULT_PROJECT_ID, has_refined_rois=True)
    assert refined_ids["frame_ids"] == [frame_ids[0]]
    summary = postgres_repo.get_frame_status_summary(project_id=DEFAULT_PROJECT_ID, asset_id=asset_id)
    assert summary["total_frame_count"] == 2
    assert summary["candidate_detection_count"] == 2
    assert summary["refined_detection_count"] == 1
    assert summary["unrefined_candidate_count"] == 1
    assert summary["by_status"]["preprocessing"] == {"unknown": 1, "working": 1}

    facets = postgres_repo.get_frame_status_facets(project_id=DEFAULT_PROJECT_ID)
    assert facets["summary"]["total_frame_count"] == 2
    assert facets["facets"]["assets"] == {asset_id: 2}
    assert facets["facets"]["preprocessing_status"] == {"unknown": 1, "working": 1}
    assert facets["facets"]["refinement_state"] == {"refined": 1, "unrefined": 1}

    snapshot = postgres_repo.get_or_create_processing_status_snapshot(
        project_id=DEFAULT_PROJECT_ID,
        summary=summary,
    )
    unchanged_snapshot = postgres_repo.get_or_create_processing_status_snapshot(
        project_id=DEFAULT_PROJECT_ID,
        summary=summary,
    )
    changed_snapshot = postgres_repo.get_or_create_processing_status_snapshot(
        project_id=DEFAULT_PROJECT_ID,
        summary={**summary, "candidate_detection_count": 3},
    )
    assert unchanged_snapshot["status_version"] == snapshot["status_version"]
    assert changed_snapshot["status_version"] == snapshot["status_version"] + 1

    first_touch = postgres_repo.touch_processing_status_snapshot(project_id=DEFAULT_PROJECT_ID)
    second_touch = postgres_repo.touch_processing_status_snapshot(project_id=DEFAULT_PROJECT_ID)
    assert second_touch["status_version"] == first_touch["status_version"] + 1


def test_postgres_repository_manages_users_projects_memberships_and_sessions(postgres_repo):
    assert postgres_repo.get_project_by_key("default") is None

    user = postgres_repo.create_user(
        "Ada",
        password="secret-passphrase",
        display_name="Ada Lovelace",
    )
    assert user["username"] == "ada"
    assert user["password_hash"] != "secret-passphrase"
    assert postgres_repo.verify_user_password("ada", "secret-passphrase")["id"] == user["id"]
    assert postgres_repo.verify_user_password("ada", "wrong") is None

    project = postgres_repo.create_project(
        "Ocean Lab",
        project_name="Ocean Lab",
        description="Shared analysis project",
        kvstore_root_path="./data/kvstore/projects/ocean-lab",
        metadata={"collection": "lab"},
    )
    assert project["project_key"] == "ocean lab"
    assert project["kvstore_root_path"].endswith("ocean-lab")

    membership = postgres_repo.add_project_member(
        str(user["id"]),
        str(project["id"]),
        role="editor",
        metadata={"invited_by": "pytest"},
    )
    assert membership["role"] == "editor"
    assert membership["metadata"]["invited_by"] == "pytest"
    assert postgres_repo.get_project_membership(str(user["id"]), str(project["id"]))["role"] == "editor"

    memberships = postgres_repo.list_user_project_memberships(str(user["id"]))
    assert [(row["project_key"], row["role"] ) for row in memberships] == [("ocean lab", "editor")]

    user_projects = postgres_repo.list_user_projects(str(user["id"]))
    assert [row["project_key"] for row in user_projects] == ["ocean lab"]
    assert user_projects[0]["membership_role"] == "editor"

    project_users = postgres_repo.list_users(project_id=str(project["id"]))
    assert [row["username"] for row in project_users] == ["ada"]
    assert project_users[0]["project_role"] == "editor"
    assert "password_hash" not in project_users[0]

    session_result = postgres_repo.create_session(
        str(user["id"]),
        str(project["id"]),
        ttl_seconds=3600,
        user_agent="pytest",
        remote_addr="127.0.0.1",
    )
    token = session_result["token"]
    session = session_result["session"]
    assert token
    assert session["token_hash"] == hash_session_token(token)
    assert token != session["token_hash"]

    resolved = postgres_repo.get_session(token)
    assert resolved is not None
    assert resolved["username"] == "ada"
    assert resolved["project_key"] == "ocean lab"
    assert resolved["project_role"] == "editor"
    assert resolved["user_agent"] == "pytest"

    revoked = postgres_repo.revoke_session(token)
    assert revoked is not None
    assert postgres_repo.get_session(token) is None

    removable_session = postgres_repo.create_session(str(user["id"]), str(project["id"]))
    removed = postgres_repo.remove_project_member(str(user["id"]), str(project["id"]))
    assert removed is not None
    assert postgres_repo.get_project_membership(str(user["id"]), str(project["id"])) is None
    assert postgres_repo.get_session(removable_session["token"]) is None

    other_project = postgres_repo.create_project("other-project")
    with pytest.raises(PermissionError):
        postgres_repo.create_session(str(user["id"]), str(other_project["id"]))

    admin = postgres_repo.create_user("Admin", password="secret", is_admin=True)
    projectless_admin_session = postgres_repo.create_session(str(admin["id"]), None)
    resolved_projectless = postgres_repo.get_session(projectless_admin_session["token"])
    assert resolved_projectless is not None
    assert resolved_projectless["is_admin"] is True
    assert resolved_projectless["project_id"] is None
    assert resolved_projectless["project_key"] is None
    assert resolved_projectless["project_role"] is None
    with pytest.raises(PermissionError):
        postgres_repo.create_session(str(user["id"]), None)

    admin_session = postgres_repo.create_session(str(admin["id"]), str(other_project["id"]))
    assert postgres_repo.get_session(admin_session["token"])["is_admin"] is True
    deactivated = postgres_repo.deactivate_project(
        str(other_project["id"]),
        metadata={"deleted_by_user_id": str(admin["id"])},
    )
    assert deactivated["is_active"] is False
    assert deactivated["metadata"]["deleted_by_user_id"] == str(admin["id"])
    assert postgres_repo.get_session(admin_session["token"]) is None

    managed_user = postgres_repo.create_user("Managed", password="old-password")
    postgres_repo.add_project_member(str(managed_user["id"]), str(project["id"]), role="viewer")
    managed_session = postgres_repo.create_session(str(managed_user["id"]), str(project["id"]))
    reset_user = postgres_repo.reset_user_password(
        str(managed_user["id"]),
        "new-password",
        metadata={"password_reset_by_user_id": str(admin["id"])},
    )
    assert reset_user["password_hash"] != managed_user["password_hash"]
    assert reset_user["metadata"]["password_reset_by_user_id"] == str(admin["id"])
    assert postgres_repo.verify_user_password("managed", "new-password")["id"] == managed_user["id"]
    assert postgres_repo.get_session(managed_session["token"]) is None

    active_session = postgres_repo.create_session(str(managed_user["id"]), str(project["id"]))
    inactive_user = postgres_repo.deactivate_user(
        str(managed_user["id"]),
        metadata={"deactivated_by_user_id": str(admin["id"])},
    )
    assert inactive_user["is_active"] is False
    assert inactive_user["metadata"]["deactivated_by_user_id"] == str(admin["id"])
    assert postgres_repo.verify_user_password("managed", "new-password") is None
    assert postgres_repo.get_session(active_session["token"]) is None

    deleted_user = postgres_repo.create_user("Deleted", password="secret")
    postgres_repo.add_project_member(str(deleted_user["id"]), str(project["id"]), role="viewer")
    assert postgres_repo.delete_user(str(deleted_user["id"]))["username"] == "deleted"
    assert postgres_repo.get_user(str(deleted_user["id"])) is None


def test_postgres_repository_filters_core_reads_by_project(postgres_repo):
    project = postgres_repo.create_project(f"scope-{uuid.uuid4().hex}")
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    planned_run = PlannedRun(
        manifest=RunManifest(
            run_id=run_id,
            run_key=f"scope-run-{uuid.uuid4().hex}",
            instrument="pytest",
            source_path="/tmp/scope.avi",
            source_type=AssetKind.VIDEO.value,
            created_at=datetime.now(timezone.utc),
            assets=[
                RawAssetManifest(
                    asset_id=asset_id,
                    filename="scope.avi",
                    path="/tmp/scope.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=123,
                    checksum="sha256:scope",
                    collections=["scope"],
                )
            ],
        ),
        jobs=[
            WorkItem(
                job_id=job_id,
                run_id=run_id,
                stage=PipelineStage.EXTRACT_FRAMES,
                asset_id=asset_id,
            )
        ],
    )
    postgres_repo.register_planned_run(planned_run, project_id=str(project["id"]))
    model = postgres_repo.register_model(
        ModelRecord(
            model_key=f"scope-model-{uuid.uuid4().hex}",
            model_name="Scope Model",
            version="1",
        ),
        project_id=str(project["id"]),
    )
    log = postgres_repo.append_log(
        project_id=str(project["id"]),
        event_type="scope.test",
        message="scoped",
    )

    assert [str(row["id"]) for row in postgres_repo.list_runs(project_id=str(project["id"]))] == [run_id]
    assert postgres_repo.list_runs(project_id=DEFAULT_PROJECT_ID) == []
    assert [str(row["id"]) for row in postgres_repo.list_assets(project_id=str(project["id"]))] == [asset_id]
    assert postgres_repo.list_assets(project_id=DEFAULT_PROJECT_ID) == []
    assert [str(row["id"]) for row in postgres_repo.list_jobs(project_id=str(project["id"]))] == [job_id]
    assert postgres_repo.list_jobs(project_id=DEFAULT_PROJECT_ID) == []
    assert postgres_repo.get_run(run_id, project_id=DEFAULT_PROJECT_ID) is None
    assert postgres_repo.get_asset(asset_id, project_id=DEFAULT_PROJECT_ID) is None
    assert postgres_repo.get_job(job_id, project_id=DEFAULT_PROJECT_ID) is None
    assert postgres_repo.get_model(str(model["id"]), project_id=str(project["id"])) is not None
    assert postgres_repo.get_model(str(model["id"]), project_id=DEFAULT_PROJECT_ID) is None
    assert postgres_repo.list_logs(project_id=str(project["id"]))[0]["id"] == log["id"]
    assert postgres_repo.list_logs(project_id=DEFAULT_PROJECT_ID) == []


def test_postgres_repository_delete_asset_removes_candidate_and_refined_detections(postgres_repo):
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    project = postgres_repo.create_project(f"delete-asset-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    postgres_repo.register_planned_run(
        PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=f"delete-asset-{uuid.uuid4().hex}",
                instrument="pytest",
                source_path="/tmp/delete-asset.avi",
                source_type=AssetKind.VIDEO.value,
                created_at=datetime.now(timezone.utc),
                assets=[
                    RawAssetManifest(
                        asset_id=asset_id,
                        filename="delete-asset.avi",
                        path="/tmp/delete-asset.avi",
                        kind=AssetKind.VIDEO,
                        size_bytes=123,
                        checksum="sha256:delete-asset",
                    )
                ],
            ),
            jobs=[],
        ),
        project_id=project_id,
    )
    frame = postgres_repo.replace_frames(
        run_id,
        [
            FrameRecord(
                asset_id=asset_id,
                frame_index=1,
                width=4,
                height=3,
                kvstore_hash="delete-asset-frame-key",
                preview_thumbhash=b"thumb",
                payload_ref="delete-asset-frame-key",
            )
        ],
    )[0]
    candidate = postgres_repo.replace_detections(
        run_id,
        asset_id,
        [
            DetectionRecord(
                run_id=run_id,
                frame_id=str(frame["id"]),
                roi_index=1,
                bbox_x=0,
                bbox_y=0,
                bbox_w=2,
                bbox_h=2,
                area=4,
                perimeter=8,
                major_axis_length=2,
                minor_axis_length=2,
                min_gray_value=1,
                mean_gray_value=2.0,
                roi_payload=b"candidate-roi",
                mask_payload=b"candidate-mask",
                centroid_x=0.75,
                centroid_y=1.25,
                major_axis_center_x=0.5,
                major_axis_center_y=1.0,
                major_axis_orientation_degrees=30.0,
                major_axis_direction_degrees=210.0,
            )
        ],
    )[0]
    assert candidate["centroid_x"] == 0.75
    assert candidate["major_axis_direction_degrees"] == 210.0
    refined = postgres_repo.upsert_refined_detections(
        [
            (
                str(candidate["id"]),
                DetectionRecord(
                    run_id=run_id,
                    frame_id=str(frame["id"]),
                    roi_index=1,
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=2,
                    bbox_h=2,
                    area=4,
                    perimeter=8,
                    major_axis_length=2,
                    minor_axis_length=2,
                    centroid_x=0.75,
                    centroid_y=1.25,
                    major_axis_center_x=0.5,
                    major_axis_center_y=1.0,
                    major_axis_orientation_degrees=30.0,
                    major_axis_direction_degrees=210.0,
                    min_gray_value=1,
                    mean_gray_value=2.0,
                    roi_payload=b"refined-roi",
                    mask_payload=b"refined-mask",
                ),
            )
        ],
        project_id=project_id,
    )[0]
    assert refined["centroid_y"] == 1.25
    assert refined["major_axis_orientation_degrees"] == 30.0

    deleted = postgres_repo.delete_asset(asset_id, project_id=project_id)

    assert deleted is not None
    assert deleted["candidate_detection_count"] == 1
    assert deleted["refined_detection_count"] == 1
    assert postgres_repo.get_detection(str(candidate["id"]), project_id=project_id) is None
    assert postgres_repo.get_refined_detection(str(refined["id"]), project_id=project_id) is None
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) AS count FROM {postgres_repo.schema}.detection_candidate WHERE id = %s",
                (candidate["id"],),
            )
            assert cursor.fetchone()["count"] == 0
            cursor.execute(
                f"SELECT COUNT(*) AS count FROM {postgres_repo.schema}.detections_refined WHERE id = %s",
                (refined["id"],),
            )
            assert cursor.fetchone()["count"] == 0


def test_postgres_repository_creates_and_deletes_project_scoped_live_frame_copy(postgres_repo):
    postgres_repo.ensure_default_project()
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    planned_run = PlannedRun(
        manifest=RunManifest(
            run_id=run_id,
            run_key=f"live-sandbox-{uuid.uuid4().hex}",
            instrument="pytest",
            source_path="/tmp/live.avi",
            source_type=AssetKind.VIDEO.value,
            created_at=datetime.now(timezone.utc),
            assets=[
                RawAssetManifest(
                    asset_id=asset_id,
                    filename="live.avi",
                    path="/tmp/live.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=123,
                    checksum="sha256:live",
                    media_count=1,
                )
            ],
        ),
        jobs=[],
    )
    postgres_repo.register_planned_run(planned_run, project_id=DEFAULT_PROJECT_ID)
    frame = postgres_repo.replace_frames(
        run_id,
        [
            FrameRecord(
                asset_id=asset_id,
                frame_index=1,
                width=4,
                height=3,
                kvstore_hash="live-frame-key",
                preview_thumbhash=b"thumb",
                payload_ref="live-frame-key",
                payload_encoding="zstd",
                payload_format="zstd_ndarray_c_order",
                payload_dtype="uint8",
                payload_shape=[3, 4],
            )
        ],
    )[0]

    sandbox = postgres_repo.create_live_frame_copy(
        str(frame["id"]),
        operation="preprocess",
        project_id=DEFAULT_PROJECT_ID,
    )

    assert sandbox["frame_index"] == -1
    assert str(sandbox["parent_frame_id"]) == str(frame["id"])
    assert sandbox["metadata"]["live_preview"]["is_sandbox"] is True
    assert sandbox["metadata"]["live_preview"]["operation"] == "preprocess"
    assert sandbox["payload_ref"] == "live-frame-key"

    deleted = postgres_repo.delete_live_frame_copy(str(sandbox["id"]), project_id=DEFAULT_PROJECT_ID)
    assert deleted is not None
    assert str(deleted["frame"]["id"]) == str(sandbox["id"])


def test_postgres_repository_enforces_project_scope_on_job_creation(postgres_repo):
    project = postgres_repo.create_project(f"write-scope-{uuid.uuid4().hex}")
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    planned_run = PlannedRun(
        manifest=RunManifest(
            run_id=run_id,
            run_key=f"write-scope-run-{uuid.uuid4().hex}",
            instrument="pytest",
            source_path="/tmp/write-scope.avi",
            source_type=AssetKind.VIDEO.value,
            created_at=datetime.now(timezone.utc),
            assets=[
                RawAssetManifest(
                    asset_id=asset_id,
                    filename="write-scope.avi",
                    path="/tmp/write-scope.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=123,
                    checksum="sha256:write-scope",
                    collections=["write-scope"],
                )
            ],
        ),
    )
    postgres_repo.register_planned_run(planned_run, project_id=str(project["id"]))
    frame = FrameRecord(
        run_id=run_id,
        asset_id=asset_id,
        frame_index=1,
        width=2,
        height=2,
        kvstore_hash="frame-key",
        preview_thumbhash=b"thumb",
        payload_ref="frame-key",
        payload_encoding="raw",
        payload_format="raw_ndarray_c_order",
        payload_dtype="uint8",
        payload_shape=[2, 2],
    )
    frame_row = postgres_repo.replace_frames(run_id, [frame])[0]

    job = postgres_repo.create_job(
        PipelineStage.PREPROCESS_FRAMES,
        project_id=str(project["id"]),
        run_id=run_id,
        asset_id=asset_id,
        payload={"frame_ids": [str(frame_row["id"])]},
    )

    assert str(job["project_id"]) == str(project["id"])
    assert str(postgres_repo.claim_jobs("project-worker", stages=[PipelineStage.PREPROCESS_FRAMES], limit=1)[0]["project_id"]) == str(project["id"])
    with pytest.raises(KeyError):
        postgres_repo.create_job(
            PipelineStage.PREPROCESS_FRAMES,
            project_id=DEFAULT_PROJECT_ID,
            run_id=run_id,
            asset_id=asset_id,
            payload={"frame_ids": [str(frame_row["id"])]},
        )
    with pytest.raises(KeyError):
        postgres_repo.create_job(
            PipelineStage.PREPROCESS_FRAMES,
            project_id=str(project["id"]),
            run_id=run_id,
            asset_id=asset_id,
            payload={"frame_ids": [str(uuid.uuid4())]},
        )


def test_postgres_repository_lease_token_fences_stale_worker_updates(postgres_repo):
    project = postgres_repo.create_project(f"lease-fencing-{uuid.uuid4().hex}")
    job = postgres_repo.create_job(
        PipelineStage.EXTRACT_FRAMES,
        project_id=str(project["id"]),
        summary="fenced lease",
    )
    first_claim = postgres_repo.claim_jobs("first-worker", stages=[PipelineStage.EXTRACT_FRAMES], limit=1)[0]
    first_token = str(first_claim["lease_token"])
    assert first_claim["lease_token"] is not None

    # Simulate an old worker that continues after its lease was reclaimed.
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {postgres_repo.schema}.processing_jobs SET lease_expires_at = NOW() - INTERVAL '1 second' WHERE id = %s",
                (job["id"],),
            )
        connection.commit()
    assert postgres_repo.requeue_expired_jobs()["queued"] == 1
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {postgres_repo.schema}.processing_jobs SET available_at = NOW() - INTERVAL '1 second' WHERE id = %s",
                (job["id"],),
            )
        connection.commit()
    second_claim = postgres_repo.claim_jobs("second-worker", stages=[PipelineStage.EXTRACT_FRAMES], limit=1)[0]
    second_token = str(second_claim["lease_token"])
    assert second_token != first_token

    assert postgres_repo.heartbeat("first-worker", str(job["id"]), lease_token=first_token) is None
    assert postgres_repo.update_job_progress(
        str(job["id"]), {"completed": 99}, worker_id="first-worker", lease_token=first_token
    ) is None
    assert postgres_repo.complete_job(
        str(job["id"]), {"stale": True}, worker_id="first-worker", lease_token=first_token
    ) is None
    assert postgres_repo.record_failure(
        str(job["id"]), "stale failure", worker_id="first-worker", lease_token=first_token
    ) is None

    completed = postgres_repo.complete_job(
        str(job["id"]), {"current": True}, worker_id="second-worker", lease_token=second_token
    )
    assert completed is not None
    assert completed["status"] == JobStatus.SUCCEEDED.value
    assert completed["result"] == {"current": True}


def test_postgres_repository_schedules_retry_and_replays_dead_letter(postgres_repo):
    project = postgres_repo.create_project(f"retry-schedule-{uuid.uuid4().hex}")
    job = postgres_repo.create_job(PipelineStage.SEGMENT, project_id=str(project["id"]))
    claim = postgres_repo.claim_jobs("retry-worker")[0]
    retried = postgres_repo.record_failure(
        str(job["id"]), "temporary storage error", retryable=True,
        failure_category="infrastructure", worker_id="retry-worker", lease_token=str(claim["lease_token"]),
    )
    assert retried["status"] == JobStatus.QUEUED.value
    assert retried["failure_category"] == "infrastructure"
    assert retried["available_at"] > retried["updated_at"]
    assert postgres_repo.claim_jobs("another-worker") == []
    signals = postgres_repo.get_job_supervisor_signals()
    assert signals["retry_scheduled"] >= 1
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {postgres_repo.schema}.processing_jobs SET available_at = NOW() - INTERVAL '1 second' WHERE id = %s",
                (job["id"],),
            )
        connection.commit()
    assert postgres_repo.claim_jobs("another-worker")[0]["id"] == job["id"]


def test_postgres_repository_defers_unstarted_job_without_spending_attempt(postgres_repo):
    project = postgres_repo.create_project(f"memory-defer-{uuid.uuid4().hex}")
    job = postgres_repo.create_job(PipelineStage.SEGMENT, project_id=str(project["id"]))
    claim = postgres_repo.claim_jobs("memory-worker", stages=[PipelineStage.SEGMENT])[0]
    token = str(claim["lease_token"])

    assert postgres_repo.defer_unstarted_job(
        str(job["id"]), worker_id="other-worker", lease_token=token,
    ) is None
    deferred = postgres_repo.defer_unstarted_job(
        str(job["id"]), worker_id="memory-worker", lease_token=token,
    )

    assert deferred is not None
    assert deferred["status"] == JobStatus.QUEUED.value
    assert deferred["attempt_count"] == 0
    assert deferred["worker_id"] is None
    assert deferred["lease_token"] is None
    assert deferred["available_at"] > deferred["updated_at"]
    assert postgres_repo.claim_jobs("another-worker", stages=[PipelineStage.SEGMENT]) == []
    assert postgres_repo.list_job_events(job_id=str(job["id"]))[0]["event_type"] == "job.deferred"

    paused_job = postgres_repo.create_job(PipelineStage.SEGMENT, project_id=str(project["id"]))
    paused_claim = postgres_repo.claim_jobs("memory-worker", stages=[PipelineStage.SEGMENT])[0]
    postgres_repo.pause_job(str(paused_job["id"]), reason="operator")
    assert postgres_repo.defer_unstarted_job(
        str(paused_job["id"]), worker_id="memory-worker",
        lease_token=str(paused_claim["lease_token"]),
    ) is None
    assert postgres_repo.get_job(str(paused_job["id"]))["control_reason"] == "pause_requested:operator"


def test_postgres_repository_materializes_idempotent_successor_dispatch(postgres_repo):
    project = postgres_repo.create_project(f"successor-dispatch-{uuid.uuid4().hex}")
    parent = postgres_repo.create_job(PipelineStage.EXTRACT_FRAMES, project_id=str(project["id"]))
    claim = postgres_repo.claim_jobs("dispatch-worker")[0]
    dispatch = postgres_repo.enqueue_successor_dispatch(
        parent_job_id=str(parent["id"]), worker_id="dispatch-worker", lease_token=str(claim["lease_token"]),
        project_id=str(project["id"]), stage=PipelineStage.SEGMENT,
        idempotency_key=f"successor:{parent['id']}:segment", payload={"frame_ids": []},
    )
    assert dispatch is not None and dispatch["status"] == "pending"
    assert postgres_repo.materialize_pending_dispatches(parent_job_id=str(parent["id"]))["materialized"] == 0
    assert postgres_repo.complete_job(str(parent["id"]), worker_id="dispatch-worker", lease_token=str(claim["lease_token"]))
    assert postgres_repo.materialize_pending_dispatches(parent_job_id=str(parent["id"]))["materialized"] == 1
    assert postgres_repo.materialize_pending_dispatches(parent_job_id=str(parent["id"]))["materialized"] == 0
    children = postgres_repo.list_jobs(project_id=str(project["id"]), stage=PipelineStage.SEGMENT.value)
    assert len(children) == 1


def test_postgres_repository_recovers_dispatch_after_materialization_interruption(postgres_repo, monkeypatch):
    """A crash after child creation must leave a replayable, idempotent outbox row."""
    project = postgres_repo.create_project(f"dispatch-recovery-{uuid.uuid4().hex}")
    parent = postgres_repo.create_job(PipelineStage.EXTRACT_FRAMES, project_id=str(project["id"]))
    claim = postgres_repo.claim_jobs("dispatch-recovery-worker")[0]
    key = f"successor:{parent['id']}:segment"
    dispatch = postgres_repo.enqueue_successor_dispatch(
        parent_job_id=str(parent["id"]), worker_id="dispatch-recovery-worker",
        lease_token=str(claim["lease_token"]), project_id=str(project["id"]),
        stage=PipelineStage.SEGMENT, idempotency_key=key, payload={"frame_ids": []},
    )
    assert dispatch is not None
    assert postgres_repo.complete_job(
        str(parent["id"]), worker_id="dispatch-recovery-worker", lease_token=str(claim["lease_token"])
    ) is not None

    original_create_job = postgres_repo.create_job
    interrupted = False

    def create_then_interrupt(*args, **kwargs):
        nonlocal interrupted
        created = original_create_job(*args, **kwargs)
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated process interruption after child creation")
        return created

    monkeypatch.setattr(postgres_repo, "create_job", create_then_interrupt)
    first = postgres_repo.materialize_pending_dispatches(parent_job_id=str(parent["id"]))
    assert first == {"materialized": 0, "failed": 1, "pending": 0}

    monkeypatch.setattr(postgres_repo, "create_job", original_create_job)
    recovered = postgres_repo.materialize_pending_dispatches(parent_job_id=str(parent["id"]))
    assert recovered == {"materialized": 1, "failed": 0, "pending": 0}
    children = postgres_repo.list_jobs(project_id=str(project["id"]), stage=PipelineStage.SEGMENT.value)
    assert len(children) == 1
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT status, child_job_id, error_message FROM {postgres_repo.schema}.processing_job_dispatches WHERE id = %s",
                (dispatch["id"],),
            )
            persisted = cursor.fetchone()
    assert persisted["status"] == "materialized"
    assert str(persisted["child_job_id"]) == str(children[0]["id"])
    assert persisted["error_message"] is None


def test_postgres_repository_supervisor_lock_is_singleton(postgres_repo):
    with postgres_repo.job_supervisor_lock() as first:
        assert first is True
        with postgres_repo.job_supervisor_lock() as second:
            assert second is False


def test_postgres_repository_reclaims_only_empty_stale_series_planning_steps(postgres_repo):
    project = postgres_repo.create_project(f"series-reclaim-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    series = postgres_repo.create_processing_series(
        project_id=project_id,
        steps=[{"stage": PipelineStage.SEGMENT.value, "filters": {}, "options": {}}],
    )
    step = postgres_repo.claim_processing_series_step(str(series["id"]), project_id=project_id)
    assert step is not None and step["status"] == "planning"

    reclaimed = postgres_repo.reclaim_stale_processing_series_planning_steps(
        project_id=project_id, older_than_seconds=-1, limit=1
    )
    assert [str(row["id"]) for row in reclaimed] == [str(step["id"])]
    assert postgres_repo.list_processing_series_steps(str(series["id"]), project_id=project_id)[0]["status"] == "queued"


def test_postgres_repository_cancel_jobs_filters_and_scopes(postgres_repo):
    postgres_repo.ensure_default_project()
    other_project = postgres_repo.create_project(f"cancel-scope-{uuid.uuid4().hex}")
    queued_job = postgres_repo.create_job(
        PipelineStage.EXTRACT_FRAMES,
        project_id=DEFAULT_PROJECT_ID,
        status=JobStatus.QUEUED,
        summary="queued default",
    )
    leased_job = postgres_repo.create_job(
        PipelineStage.PREPROCESS_FRAMES,
        project_id=DEFAULT_PROJECT_ID,
        status=JobStatus.QUEUED,
        summary="leased default",
    )
    succeeded_job = postgres_repo.create_job(
        PipelineStage.SEGMENT,
        project_id=DEFAULT_PROJECT_ID,
        status=JobStatus.QUEUED,
        summary="succeeded default",
    )
    other_job = postgres_repo.create_job(
        PipelineStage.EXTRACT_FRAMES,
        project_id=str(other_project["id"]),
        status=JobStatus.QUEUED,
        summary="queued other",
    )
    claimed = postgres_repo.claim_jobs(
        "cancel-worker",
        stages=[PipelineStage.PREPROCESS_FRAMES],
        limit=1,
    )
    assert str(claimed[0]["id"]) == str(leased_job["id"])
    postgres_repo.complete_job(str(succeeded_job["id"]), result={"done": True})

    preview = postgres_repo.cancel_jobs(project_id=DEFAULT_PROJECT_ID, dry_run=True)

    assert preview["dry_run"] is True
    assert preview["matched_count"] == 3
    assert preview["cancellable_count"] == 2
    assert preview["cancelled_count"] == 0
    assert postgres_repo.get_job(str(queued_job["id"]))["status"] == JobStatus.QUEUED.value

    result = postgres_repo.cancel_jobs(
        project_id=DEFAULT_PROJECT_ID,
        stages=[PipelineStage.EXTRACT_FRAMES.value, PipelineStage.PREPROCESS_FRAMES.value],
        reason="clear test queue",
    )

    assert result["matched_count"] == 2
    assert result["cancellable_count"] == 2
    assert result["cancelled_count"] == 2
    assert {str(row["id"]) for row in result["jobs"]} == {
        str(queued_job["id"]),
        str(leased_job["id"]),
    }
    assert postgres_repo.get_job(str(queued_job["id"]))["status"] == JobStatus.CANCELLED.value
    assert postgres_repo.get_job(str(leased_job["id"]))["status"] == JobStatus.CANCELLED.value
    assert postgres_repo.get_job(str(succeeded_job["id"]))["status"] == JobStatus.SUCCEEDED.value
    assert postgres_repo.get_job(str(other_job["id"]))["status"] == JobStatus.QUEUED.value
    assert postgres_repo.get_job(str(queued_job["id"]))["control_reason"] == "clear test queue"
    assert postgres_repo.complete_job(str(leased_job["id"]), result={"late": True}) is None
    assert postgres_repo.get_job(str(leased_job["id"]))["status"] == JobStatus.CANCELLED.value
    assert postgres_repo.record_failure(str(leased_job["id"]), "late failure") is None
    assert postgres_repo.get_job(str(leased_job["id"]))["status"] == JobStatus.CANCELLED.value

    events = postgres_repo.list_job_events(job_id=str(queued_job["id"]))
    assert events[0]["event_type"] == "job.cancelled"
    assert events[0]["payload"]["bulk"] is True
    assert events[0]["payload"]["previous_status"] == JobStatus.QUEUED.value


def test_postgres_repository_clears_terminal_series_jobs_and_work_units(postgres_repo):
    project = postgres_repo.create_project(f"clear-series-job-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    series = postgres_repo.create_processing_series(
        project_id=project_id,
        steps=[{"stage": PipelineStage.SEGMENT.value, "filters": {}, "options": {}}],
    )
    step = postgres_repo.claim_processing_series_step(str(series["id"]), project_id=project_id)
    assert step is not None
    job = postgres_repo.create_job(
        PipelineStage.SEGMENT,
        project_id=project_id,
        status=JobStatus.CANCELLED,
    )
    postgres_repo.attach_processing_work_units(
        series_id=str(series["id"]),
        step_id=str(step["id"]),
        job_ids=[str(job["id"])],
        matched_count=1,
    )

    result = postgres_repo.delete_jobs(
        project_id=project_id,
        statuses=[JobStatus.CANCELLED.value],
    )

    assert result["deleted_count"] == 1
    assert postgres_repo.get_job(str(job["id"]), project_id=project_id) is None
    assert postgres_repo.list_processing_work_units(str(series["id"]), project_id=project_id) == []


def test_postgres_repository_bulk_pause_and_resume_are_stage_scoped(postgres_repo):
    project = postgres_repo.create_project(f"bulk-control-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    queued_job = postgres_repo.create_job(
        PipelineStage.CLASSIFY,
        project_id=project_id,
        status=JobStatus.QUEUED,
        summary="queued classification",
    )
    leased_job = postgres_repo.create_job(
        PipelineStage.CLASSIFY,
        project_id=project_id,
        status=JobStatus.QUEUED,
        summary="leased classification",
    )
    unrelated_job = postgres_repo.create_job(
        PipelineStage.SEGMENT,
        project_id=project_id,
        status=JobStatus.QUEUED,
        summary="unrelated segmentation",
    )
    claimed = postgres_repo.claim_jobs("classification-worker", stages=[PipelineStage.CLASSIFY], limit=1)
    leased_id = str(claimed[0]["id"])
    paused_id = str(leased_job["id"] if leased_id == str(queued_job["id"]) else queued_job["id"])

    paused = postgres_repo.pause_jobs(
        project_id=project_id,
        stages=[PipelineStage.CLASSIFY.value],
        reason="hold classification",
    )

    assert paused["matched_count"] == 2
    assert paused["paused_count"] == 1
    assert paused["pause_requested_count"] == 1
    paused_row = postgres_repo.get_job(paused_id)
    leased_row = postgres_repo.get_job(leased_id)
    assert paused_row["status"] == JobStatus.PAUSED.value
    assert leased_row["status"] == JobStatus.LEASED.value
    assert leased_row["control_reason"].startswith("pause_requested:")
    assert postgres_repo.get_job(str(unrelated_job["id"]))["status"] == JobStatus.QUEUED.value

    resumed = postgres_repo.resume_jobs(
        project_id=project_id,
        stages=[PipelineStage.CLASSIFY.value],
        reason="resume classification",
    )

    assert resumed["resumed_count"] == 1
    assert postgres_repo.get_job(paused_id)["status"] == JobStatus.QUEUED.value

def test_postgres_repository_purge_all_deletes_rows(postgres_repo):
    postgres_repo.ensure_default_project()
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    planned_run = PlannedRun(
        manifest=RunManifest(
            run_id=run_id,
            run_key=f"purge-test-{uuid.uuid4().hex}",
            instrument="pytest",
            source_path="/tmp/purge.avi",
            source_type=AssetKind.VIDEO.value,
            created_at=datetime.now(timezone.utc),
            assets=[
                RawAssetManifest(
                    asset_id=asset_id,
                    filename="purge.avi",
                    path="/tmp/purge.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=456,
                    checksum="sha256:purge",
                    collections=["purge"],
                )
            ],
        ),
        jobs=[
            WorkItem(
                job_id=job_id,
                run_id=run_id,
                stage=PipelineStage.EXTRACT_FRAMES,
                asset_id=asset_id,
                payload={"source_path": "/tmp/purge.avi"},
            )
        ],
    )
    postgres_repo.register_planned_run(planned_run, project_id=DEFAULT_PROJECT_ID)
    postgres_repo.touch_worker(
        "purge-worker",
        status="idle",
        capabilities=[PipelineStage.EXTRACT_FRAMES.value],
        pid=23456,
    )
    postgres_repo.register_model(
        ModelRecord(
            model_key=f"purge-model-{uuid.uuid4().hex}",
            model_name="Purge Model",
            version="0",
        ),
        project_id=DEFAULT_PROJECT_ID,
    )

    result = postgres_repo.purge_all()

    assert result["tables"]["runs"] == 1
    assert result["tables"]["raw_assets"] == 1
    assert result["tables"]["processing_jobs"] == 1
    assert result["tables"]["worker_sessions"] == 1
    assert result["tables"]["models"] == 1
    assert result["tables"]["job_events"] >= 1
    assert result["exact_counts_collected"] is True
    assert postgres_repo.list_runs() == []
    assert postgres_repo.list_assets() == []
    assert postgres_repo.list_jobs() == []
    assert postgres_repo.list_worker_sessions() == []
    assert postgres_repo.list_collections() == []
    assert postgres_repo.list_models() == []
    assert postgres_repo.list_job_events() == []


def test_postgres_repository_purge_all_can_skip_exact_counts(postgres_repo):
    result = postgres_repo.purge_all(exact_counts=False)

    assert result["tables"] is None
    assert result["total_rows_deleted"] is None
    assert result["exact_counts_collected"] is False
    assert result["missing_tables"] == []
    assert set(result["purged_tables"]) == set(postgres.REQUIRED_SCHEMA_TABLES) - {"schema_migrations"}


def test_postgres_large_reset_sequence_repairs_incomplete_schema(postgres_repo):
    missing_table = "classification_evidence"
    with postgres_repo.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP TABLE {postgres_repo.schema}.{missing_table} CASCADE")
        connection.commit()

    status = postgres_repo.schema_status()
    assert missing_table in status["missing_tables"]

    purge_result = postgres_repo.purge_all(
        exact_counts=False,
        preserve_migrations=False,
    )
    assert missing_table in purge_result["missing_tables"]
    assert purge_result["preserved_tables"] == []

    postgres_repo.initialize_schema(statement_timeout_ms=0)

    repaired = postgres_repo.schema_status()
    assert repaired["missing_tables"] == []
    assert repaired["ready"] is True
