from __future__ import annotations

from datetime import datetime, timezone
import os
import uuid

import pytest

from Pelagia.config import CoreConfig
from Pelagia.domain import (
    AssetKind,
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
    "postgresql://postgres:postgres@localhost:5432/postgres",
)


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
                metadata={"kvstore_key": "kvstore-key-1"},
            )
        ],
    )
    assert len(inserted_frames) == 1
    assert inserted_frames[0]["frame_hash"] == "kvstore-key-1"

    frames = postgres_repo.list_frames(asset_id)
    assert len(frames) == 1
    assert frames[0]["frame_index"] == 1
    assert frames[0]["metadata"]["kvstore_key"] == "kvstore-key-1"

    queued = postgres_repo.get_job(job_id)
    assert queued is not None
    assert queued["status"] == "queued"

    claimed = postgres_repo.claim_jobs("pytest-worker", stages=[PipelineStage.EXTRACT_FRAMES], limit=1)
    assert len(claimed) == 1
    assert str(claimed[0]["id"]) == job_id
    assert claimed[0]["status"] == "leased"

    completed = postgres_repo.complete_job(job_id, result={"frames": len(frames)})
    assert completed is not None
    assert completed["status"] == "succeeded"
    assert completed["result"]["frames"] == 1
