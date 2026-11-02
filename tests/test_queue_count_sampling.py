from __future__ import annotations

from datetime import datetime, timezone
import os
import uuid

import pytest

from Pelagia.config import CoreConfig
from Pelagia.storage import postgres
from Pelagia.storage.count_sampling import CountSample, count_sample
from Pelagia.storage.postgres import PostgresRepository


class SizeCursor:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def execute(self, query, params):
        self.calls.append((query, params))

    def fetchone(self):
        return {"pages": self.pages}


def test_auto_sampling_uses_physical_pages_and_exact_mode_needs_no_probe():
    cursor = SizeCursor(8192)
    assert count_sample(cursor, "catalog", "detection_candidate", "exact").fraction == 1
    assert not cursor.calls
    sample = count_sample(cursor, "catalog", "detection_candidate", "auto")
    assert sample.fraction == 256 / 8192
    assert "TABLESAMPLE SYSTEM" in sample.sql
    assert "REPEATABLE (0)" in sample.sql
    assert cursor.calls[0][1] == ("catalog.detection_candidate",)
    assert sample.count(20) == 640
    assert not sample.metadata(0)["exact"]
    assert sample.metadata(0)["matching_sample_rows"] == 0
    assert count_sample(SizeCursor(1024), "catalog", "frames", "auto").sql == ""
    with pytest.raises(ValueError, match="count_mode"):
        count_sample(cursor, "catalog", "frames", "invalid")


@pytest.fixture
def count_repository():
    if postgres.psycopg is None:
        pytest.skip("psycopg is not installed")
    config = CoreConfig()
    config.database.dsn = os.getenv(
        "PELAGIA_TEST_DATABASE_DSN", "postgresql://postgres:postgres@localhost:5432/pelagia",
    )
    config.database.schema_name = f"pelagia_count_test_{uuid.uuid4().hex}"
    repo = PostgresRepository(config)
    try:
        with repo.connect() as connection:
            connection.execute("SELECT 1")
    except postgres.psycopg.OperationalError:
        pytest.skip("PostgreSQL test database unavailable")
    # Only the projection and candidate fields used by these read paths are
    # needed. No hot store, processing jobs, or external services are involved.
    with repo.connect() as connection:
        connection.execute(f"CREATE SCHEMA {repo.schema}")
        connection.execute(f"""CREATE TABLE {repo.schema}.frame_processing_status (
            project_id uuid NOT NULL, frame_id uuid PRIMARY KEY, asset_id uuid,
            collections text[] NOT NULL, preprocessing_status text,
            candidate_detection_status text, roi_refinement_status text,
            candidate_detection_count integer, refined_detection_count integer,
            unrefined_candidate_count integer, refinement_queue_candidate_count integer,
            refinement_queue_refined_candidate_count integer, updated_at timestamptz
        )""")
        connection.execute(f"""CREATE TABLE {repo.schema}.detection_candidate (
            id uuid PRIMARY KEY, frame_id uuid, bbox_w integer, bbox_h integer,
            roi_payload_ref text
        )""")
        connection.execute(f"""CREATE TABLE {repo.schema}.detections_refined (
            id uuid PRIMARY KEY, candidate_detection_id uuid
        )""")
        connection.commit()
    try:
        yield repo
    finally:
        with repo.connect() as connection:
            connection.execute(f"DROP SCHEMA {repo.schema} CASCADE")
            connection.commit()
        repo.close()


def test_count_previews_scale_all_facets_and_preserve_project_scope(count_repository, monkeypatch):
    repo = count_repository
    project, other_project, asset, frame, other_frame = [str(uuid.uuid4()) for _ in range(5)]
    now = datetime.now(timezone.utc)
    with repo.connect() as connection:
        for owner, frame_id, candidates in [(project, frame, 3), (other_project, other_frame, 999)]:
            connection.execute(
                f"INSERT INTO {repo.schema}.frame_processing_status VALUES "
                "(%s,%s,%s,%s,'succeeded','succeeded','unknown',%s,2,1,%s,1,%s)",
                (owner, frame_id, asset, ["a", "b"], candidates, candidates, now),
            )
        candidates = [str(uuid.uuid4()) for _ in range(3)]
        for index, candidate in enumerate(candidates):
            connection.execute(
                f"INSERT INTO {repo.schema}.detection_candidate VALUES (%s,%s,%s,10,'crop')",
                (candidate, frame, index + 1),
            )
        # Multiple outputs still cover one candidate parent.
        for _ in range(2):
            connection.execute(
                f"INSERT INTO {repo.schema}.detections_refined VALUES (%s,%s)",
                (str(uuid.uuid4()), candidates[1]),
            )
        connection.execute(
            f"INSERT INTO {repo.schema}.detection_candidate VALUES (%s,%s,2,10,'crop')",
            (str(uuid.uuid4()), other_frame),
        )
        connection.commit()

    exact = repo.get_refinement_queue_counts(project_id=project)
    assert exact == {"candidates": 3, "refined": 1, "unrefined": 2}
    auto = repo.get_refinement_queue_counts(project_id=project, count_mode="auto")
    assert auto["counts"]["exact"]
    assert auto["candidates"] == 3
    assert repo.get_refinement_queue_counts(project_id=project, min_area=20, max_area=20) == {
        "candidates": 1, "refined": 1, "unrefined": 0,
    }

    class WholeTableHalfWeight(CountSample):
        @property
        def sql(self):
            return ""

    # Deterministic sample contents make extrapolation assertions independent
    # of PostgreSQL block placement. Real TABLESAMPLE SQL is exercised below.
    monkeypatch.setattr(postgres, "count_sample", lambda *args: WholeTableHalfWeight(0.5))
    facets = repo.get_frame_status_facets(project_id=project, collections=["a", "b"], count_mode="auto")
    assert facets["summary"]["total_frame_count"] == 2
    assert facets["summary"]["candidate_detection_count"] == 6
    assert facets["summary"]["updated_at"] == now
    assert facets["facets"]["assets"] == {asset: 2}
    assert facets["facets"]["collections"] == {"a": 2, "b": 2}
    assert facets["facets"]["preprocessing_status"] == {"succeeded": 2}
    assert facets["counts"]["matching_sample_rows"] == 1
    ranged = repo.get_refinement_queue_counts(project_id=project, min_area=20, max_area=30, count_mode="auto")
    assert {key: ranged[key] for key in exact} == {"candidates": 4, "refined": 2, "unrefined": 2}

    monkeypatch.setattr(postgres, "count_sample", lambda *args: CountSample(0.5))
    sampled = repo.get_frame_status_facets(project_id=project, count_mode="auto")
    assert not sampled["counts"]["exact"]
    assert sampled["summary"]["total_frame_count"] in {0, 2}
    sampled_rois = repo.get_refinement_queue_counts(project_id=project, min_area=20, count_mode="auto")
    assert not sampled_rois["counts"]["exact"]
    assert sampled_rois["candidates"] in {0, 4}
