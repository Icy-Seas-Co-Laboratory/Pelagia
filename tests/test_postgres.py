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
    ModelRecord,
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
    assert "CREATE TABLE IF NOT EXISTS pelagia_unit.logs" in rendered
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

    run_row = postgres_repo.register_planned_run(planned_run)
    assert str(run_row["run"]["id"]) == run_id
    assert run_row["asset_count"] == 1
    assert run_row["job_count"] == 1
    summary_job = postgres_repo.create_job(
        PipelineStage.SEGMENT,
        run_id=run_id,
        asset_id=asset_id,
        payload={"frame_ids": [str(uuid.uuid4()) for _ in range(20)]},
    )
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
    assert len(detail_jobs[0]["payload"]["frame_ids"]) == 20
    assert len(detail_jobs[0]["result"]["detection_ids"]) == 20

    assets = postgres_repo.list_assets(run_id)
    assert len(assets) == 1
    assert str(assets[0]["id"]) == asset_id
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
    assert log_row["duration_ms"] == 42.5
    logs = postgres_repo.list_logs(
        run_id=run_id,
        worker_id="pytest-worker",
        event_type="pipeline.extract_timed",
    )
    assert logs[0]["payload"] == {"frame_count": 2}
    assert "worker.shutdown_requested" in [event["event_type"] for event in worker_events]


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
    postgres_repo.register_planned_run(planned_run)
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
        )
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
