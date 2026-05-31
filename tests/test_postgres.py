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
    PlannedRun,
    PipelineStage,
    RawAssetManifest,
    RunManifest,
    WorkItem,
)
from Pelagia.storage import postgres
from Pelagia.storage.postgres import PostgresRepository


POSTGRES_TEST_DSN = os.getenv(
    "PELAGIA_TEST_DATABASE_DSN",
    "postgresql://postgres:postgres@localhost:5432/pelagia",
)


def test_render_schema_loads_sql_resource():
    rendered = postgres.render_schema("pelagia_unit")

    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.frames" in rendered
    assert "payload_ref text" in rendered
    assert "{schema}" not in rendered


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
                    asset_key="example.avi",
                    path="/tmp/example.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=123,
                    checksum="sha256:test",
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

    run_row = postgres_repo.register_planned_run(planned_run)
    assert str(run_row["run"]["id"]) == run_id
    assert run_row["asset_count"] == 1
    assert run_row["job_count"] == 1

    assets = postgres_repo.list_assets(run_id)
    assert len(assets) == 1
    assert str(assets[0]["id"]) == asset_id
    assert assets[0]["asset_key"] == "example.avi"

    inserted_frames = postgres_repo.replace_frames(
        run_id,
        [
            FrameRecord(
                asset_id=asset_id,
                frame_index=1,
                width=4,
                height=3,
                frame_png=frame_payload,
                frame_hash="kvstore-key-1",
                source_ref="/tmp/example.avi",
                bbox_x=7,
                bbox_y=8,
                payload_ref="kvstore-key-1",
                payload_encoding="zstd",
                payload_format="zstd_ndarray_c_order",
                payload_dtype="uint8",
                payload_shape=[3, 4],
                metadata={"kvstore_key": "kvstore-key-1"},
            )
        ],
    )
    assert len(inserted_frames) == 1
    assert inserted_frames[0]["frame_hash"] == "kvstore-key-1"
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
    assert len(frames) == 1
    assert frames[0]["frame_index"] == 1
    assert frames[0]["metadata"]["kvstore_key"] == "kvstore-key-1"
    assert postgres_repo.get_frame_record(inserted_frames[0]["id"]).payload_ref == "kvstore-key-1"
    assert postgres_repo.list_frame_records(asset_id)[0].bbox_x == 7

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
            )
        ],
    )
    assert len(inserted_detections) == 1
    assert inserted_detections[0]["roi_payload"] == b"roi-bytes"
    assert inserted_detections[0]["mask_payload"] == b"mask-bytes"
    assert inserted_detections[0]["crop_bbox_w"] == 5
    assert inserted_detections[0]["roi_encoding"] == "raw"
    assert inserted_detections[0]["mask_shape"] == [4, 5]

    detections = postgres_repo.list_detections(asset_id)
    assert len(detections) == 1
    assert detections[0]["mask_payload"] == b"mask-bytes"
    detection_record = DetectionRecord.from_row(detections[0])
    assert detection_record.id == str(detections[0]["id"])
    assert detection_record.roi_payload == b"roi-bytes"
    assert detection_record.mask_shape == [4, 5]
    assert postgres_repo.list_detection_records(asset_id)[0].mask_payload == b"mask-bytes"

    queued = postgres_repo.get_job(job_id)
    assert queued is not None
    assert queued["status"] == "queued"
    events = postgres_repo.list_job_events(job_id=job_id)
    assert [event["event_type"] for event in events] == ["job.created"]
    assert events[0]["payload"]["stage"] == PipelineStage.EXTRACT_FRAMES.value

    claimed = postgres_repo.claim_jobs("pytest-worker", stages=[PipelineStage.EXTRACT_FRAMES], limit=1)
    assert len(claimed) == 1
    assert str(claimed[0]["id"]) == job_id
    assert claimed[0]["status"] == "leased"
    events = postgres_repo.list_job_events(job_id=job_id)
    assert [event["event_type"] for event in events] == ["job.created", "job.leased"]
    assert events[-1]["payload"]["worker_id"] == "pytest-worker"

    completed = postgres_repo.complete_job(job_id, result={"frames": len(frames)})
    assert completed is not None
    assert completed["status"] == "succeeded"
    assert completed["result"]["frames"] == 1
    events = postgres_repo.list_job_events(job_id=job_id)
    assert [event["event_type"] for event in events] == [
        "job.created",
        "job.leased",
        "job.completed",
    ]
    assert events[-1]["payload"]["result"] == {"frames": 1}

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
    assert "worker.shutdown_requested" in [event["event_type"] for event in worker_events]
