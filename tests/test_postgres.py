from __future__ import annotations

from datetime import datetime, timezone
import os
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


POSTGRES_TEST_DSN = os.getenv(
    "PELAGIA_TEST_DATABASE_DSN",
    "postgresql://postgres:postgres@localhost:5432/pelagia",
)


def test_render_schema_loads_sql_resource():
    rendered = postgres.render_schema("pelagia_unit")

    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.users" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.projects" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.project_memberships" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.user_sessions" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.schema_migrations" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.frames" in rendered
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.logs" in rendered
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
    assert status["migrations"]["available_count"] == 3
    assert status["migrations"]["applied_count"] == 3
    assert status["migrations"]["pending_count"] == 0
    assert status["migrations"]["applied"][0]["migration_id"] == "0001_processing_status"


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
    assert {"worker.touched", "worker.shutdown_requested"} <= event_types
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


def test_postgres_repository_registers_frames_and_jobs(postgres_repo):
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

    inserted_detections = postgres_repo.replace_detections(
        run_id,
        asset_id,
        [
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
            )
        ],
    )
    assert len(inserted_detections) == 2
    assert inserted_detections[0]["roi_payload"] == b"roi-bytes"
    assert inserted_detections[0]["mask_payload"] == b"mask-bytes"
    assert inserted_detections[0]["crop_bbox_w"] == 5
    assert inserted_detections[0]["roi_encoding"] == "raw"
    assert inserted_detections[0]["mask_shape"] == [4, 5]

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
                    },
                ),
            )
        ]
    )
    assert len(refined_again) == 1
    assert refined_again[0]["candidate_detection_id"] == filtered_detections[0]["id"]
    assert refined_again[0]["id"] != refined[0]["id"]
    latest_refined = postgres_repo.get_refined_detection_for_candidate(str(filtered_detections[0]["id"]))
    assert latest_refined is not None
    assert latest_refined["candidate_detection_id"] == filtered_detections[0]["id"]

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
    assert "worker.touched" in [event["event_type"] for event in worker_events]

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


def test_postgres_frame_processing_status_projection_tracks_stage_and_counts(postgres_repo):
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
            )
        ],
    )[0]
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
                    min_gray_value=1,
                    mean_gray_value=2.0,
                    roi_payload=b"refined-roi",
                    mask_payload=b"refined-mask",
                ),
            )
        ],
        project_id=project_id,
    )[0]

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


def test_postgres_repository_cancel_jobs_filters_and_scopes(postgres_repo):
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

    events = postgres_repo.list_job_events(job_id=str(queued_job["id"]))
    assert events[0]["event_type"] == "job.cancelled"
    assert events[0]["payload"]["bulk"] is True
    assert events[0]["payload"]["previous_status"] == JobStatus.QUEUED.value


def test_postgres_repository_purge_all_deletes_rows(postgres_repo):
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
    assert result["tables"]["job_events"] >= 2
    assert postgres_repo.list_runs() == []
    assert postgres_repo.list_assets() == []
    assert postgres_repo.list_jobs() == []
    assert postgres_repo.list_worker_sessions() == []
    assert postgres_repo.list_collections() == []
    assert postgres_repo.list_models() == []
    assert postgres_repo.list_job_events() == []
