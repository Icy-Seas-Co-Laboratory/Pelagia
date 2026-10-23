from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier, Event
from time import monotonic
import uuid

import pytest

from Pelagia.config import CoreConfig
from Pelagia.domain import AssetKind, FrameRecord, PlannedRun, RawAssetManifest, RunManifest
from Pelagia.services.context import AppContext
from Pelagia.services.processing_queue import ProcessingQueueService, ProcessingSubmissionService
from test_postgres import postgres_repo


class _SimulatedProcessCrash(BaseException):
    pass


def _project_with_frames(repository, *, count: int = 2) -> tuple[str, str]:
    project = repository.create_project(f"submission-integration-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    repository.register_planned_run(
        PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=f"submission-run-{uuid.uuid4().hex}",
                instrument="pytest",
                source_path="/tmp/submission.avi",
                source_type=AssetKind.VIDEO.value,
                created_at=datetime.now(timezone.utc),
                assets=[
                    RawAssetManifest(
                        asset_id=asset_id,
                        filename="submission.avi",
                        path="/tmp/submission.avi",
                        kind=AssetKind.VIDEO,
                        size_bytes=1,
                        checksum=f"sha256:{uuid.uuid4().hex}",
                        collections=["submission-test"],
                    )
                ],
            ),
            jobs=[],
        ),
        project_id=project_id,
    )
    repository.replace_frames(
        run_id,
        [
            FrameRecord(
                asset_id=asset_id,
                frame_index=index,
                width=4,
                height=4,
                preview_thumbhash=f"thumb-{index}".encode(),
                kvstore_hash=f"payload-{uuid.uuid4().hex}",
                payload_ref=f"payload-{uuid.uuid4().hex}",
                payload_encoding="zstd",
                payload_format="zstd_ndarray_c_order",
                payload_dtype="uint8",
                payload_shape=[4, 4],
            )
            for index in range(count)
        ],
    )
    assert repository.ensure_frame_status_rows(project_id=project_id, asset_id=asset_id) == count
    return project_id, asset_id


def _service(repository) -> ProcessingSubmissionService:
    return ProcessingSubmissionService(AppContext(config=CoreConfig(), repository=repository))


def _expire_claim(repository, submission_id: str) -> None:
    with repository.connect() as connection:
        connection.execute(
            f"UPDATE {repository.schema}.processing_submissions "
            "SET claim_expires_at = NOW() - INTERVAL '1 second' WHERE id = %s",
            (submission_id,),
        )
        connection.commit()


def _materialized_counts(repository, project_id: str) -> tuple[int, int, int]:
    with repository.connect() as connection:
        job_count = connection.execute(
            f"SELECT COUNT(*) AS count FROM {repository.schema}.processing_jobs WHERE project_id = %s",
            (project_id,),
        ).fetchone()["count"]
        series_count = connection.execute(
            f"SELECT COUNT(*) AS count FROM {repository.schema}.processing_series WHERE project_id = %s",
            (project_id,),
        ).fetchone()["count"]
        unit_count = connection.execute(
            f"""SELECT COUNT(*) AS count
            FROM {repository.schema}.processing_work_units units
            JOIN {repository.schema}.processing_series series ON series.id = units.series_id
            WHERE series.project_id = %s""",
            (project_id,),
        ).fetchone()["count"]
    return int(job_count), int(series_count), int(unit_count)


def test_submission_crash_after_first_job_rolls_back_and_replays_once(
    postgres_repo, monkeypatch,
):
    import Pelagia.services.processing_queue as processing_queue

    project_id, asset_id = _project_with_frames(postgres_repo, count=2)
    service = _service(postgres_repo)
    submission = service.submit(
        kind="queue",
        request={
            "stage": "segment",
            "filters": {"asset_ids": [asset_id]},
            "options": {},
            "priority": None,
            "dry_run": False,
        },
        request_id=str(uuid.uuid4()),
        project_id=project_id,
    )
    assert submission["request"]["filters"]["candidate_detection_status"] == ["cancelled", "failed", "unknown"]
    claimed = postgres_repo.claim_processing_submission(lease_seconds=60)
    assert claimed is not None and claimed["id"] == submission["id"]

    monkeypatch.setattr(processing_queue, "INCREMENTAL_FRAME_PAGE_SIZE", 1)
    original_create_job = postgres_repo.create_job
    calls = 0

    def crash_on_first_page(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise _SimulatedProcessCrash()
        return original_create_job(*args, **kwargs)

    monkeypatch.setattr(postgres_repo, "create_job", crash_on_first_page)
    with pytest.raises(_SimulatedProcessCrash):
        service._dispatch_claimed(claimed)

    assert _materialized_counts(postgres_repo, project_id) == (0, 0, 0)
    assert postgres_repo.get_processing_submission(
        str(submission["id"]), project_id=project_id,
    )["status"] == "planning"

    monkeypatch.setattr(postgres_repo, "create_job", original_create_job)
    _expire_claim(postgres_repo, str(submission["id"]))
    assert service.dispatch(limit=1) == 1
    first_checkpoint = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    assert first_checkpoint["status"] == "planning"
    assert first_checkpoint["progress"]["scheduled_count"] == 1
    assert first_checkpoint["planning_cursor"]["frame_id"]
    claimed_job = postgres_repo.claim_jobs("incremental-worker", stages=[], limit=1)
    assert len(claimed_job) == 1

    # A restarted planner resumes strictly after the committed cursor while
    # the worker is already able to process the first page.
    assert service.dispatch(limit=1) == 1
    assert _materialized_counts(postgres_repo, project_id) == (2, 0, 0)
    completed = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    assert completed["status"] == "processing"
    assert completed["result"]["job_count"] == 2
    with postgres_repo.connect() as connection:
        reservations = connection.execute(
            f"SELECT unit_id FROM {postgres_repo.schema}.processing_submission_units WHERE submission_id = %s",
            (submission["id"],),
        ).fetchall()
    assert len({str(row["unit_id"]) for row in reservations}) == 2


def test_series_crash_after_attachment_rolls_back_and_replays_once(postgres_repo, monkeypatch):
    project_id, asset_id = _project_with_frames(postgres_repo, count=1)
    service = _service(postgres_repo)
    submission = service.submit(
        kind="series",
        request={
            "steps": [{"stage": "segment", "filters": {"asset_ids": [asset_id]}, "options": {}}],
            "selection": {"asset_ids": [asset_id]},
            "preset_snapshot": {},
            "failure_policy": "fail_fast",
            "priority": None,
            "dry_run": False,
        },
        request_id=str(uuid.uuid4()),
        project_id=project_id,
    )
    assert submission["request"]["steps"][0]["filters"]["candidate_detection_status"] == [
        "cancelled", "failed", "unknown",
    ]
    claimed = postgres_repo.claim_processing_submission(lease_seconds=60)
    assert claimed is not None and claimed["id"] == submission["id"]

    original_checkpoint = postgres_repo.checkpoint_processing_submission

    def crash_before_submission_checkpoint(*args, **kwargs):
        raise _SimulatedProcessCrash()

    monkeypatch.setattr(postgres_repo, "checkpoint_processing_submission", crash_before_submission_checkpoint)
    with pytest.raises(_SimulatedProcessCrash):
        service._dispatch_claimed(claimed)
    assert _materialized_counts(postgres_repo, project_id) == (0, 0, 0)

    monkeypatch.setattr(postgres_repo, "checkpoint_processing_submission", original_checkpoint)
    _expire_claim(postgres_repo, str(submission["id"]))
    assert service.dispatch(limit=1) == 1
    assert _materialized_counts(postgres_repo, project_id) == (1, 1, 1)
    planned = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    assert planned["status"] == "processing"
    assert planned["result"]["series"]["id"]


def test_incremental_series_checkpoints_each_step_when_it_becomes_eligible(postgres_repo, monkeypatch):
    import Pelagia.services.processing_queue as processing_queue

    project_id, asset_id = _project_with_frames(postgres_repo, count=2)
    service = _service(postgres_repo)
    series_director = ProcessingQueueService(AppContext(config=CoreConfig(), repository=postgres_repo))
    monkeypatch.setattr(processing_queue, "INCREMENTAL_FRAME_PAGE_SIZE", 1)
    submission = service.submit(
        kind="series",
        request={
            "steps": [
                {"stage": "preprocess_frames", "filters": {"asset_ids": [asset_id]}, "options": {}},
                {"stage": "segment", "filters": {"asset_ids": [asset_id]}, "options": {}},
            ],
            "selection": {"asset_ids": [asset_id]}, "preset_snapshot": {},
            "failure_policy": "fail_fast", "priority": 7,
        },
        request_id=str(uuid.uuid4()), project_id=project_id,
        submitted_by_user_id="scientist-1", submitted_by_username="Scientist",
    )

    assert service.dispatch(limit=1) == 1
    first = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    first_step_id = first["planning_cursor"]["step_id"]
    assert first["status"] == "planning"
    assert service.dispatch(limit=1) == 1
    planned_first = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    assert planned_first["status"] == "processing"
    series_id = planned_first["result"]["series"]["id"]
    first_units = postgres_repo.list_processing_work_units(series_id, project_id=project_id)
    assert len(first_units) == 2
    for unit in first_units:
        job = postgres_repo.get_job(str(unit["job_id"]), project_id=project_id)
        assert job["priority"] == 7
        assert job["submitted_by_user_id"] == "scientist-1"
        postgres_repo.complete_job(str(unit["job_id"]), result={})
        series_director.advance_series_for_job(str(unit["job_id"]))

    awakened = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    assert awakened["status"] == "planning"
    assert service.dispatch(limit=1) == 1
    second = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    assert second["planning_cursor"]["step_id"] != first_step_id
    assert second["selection_boundary"]["step_id"] == second["planning_cursor"]["step_id"]


def test_paused_incremental_series_does_not_report_reservation_waiting_or_future_planning(postgres_repo):
    project_id, asset_id = _project_with_frames(postgres_repo, count=1)
    service = _service(postgres_repo)
    submission = service.submit(
        kind="series",
        request={
            "steps": [
                {"stage": "preprocess_frames", "filters": {"asset_ids": [asset_id]}, "options": {}},
                {"stage": "segment", "filters": {"asset_ids": [asset_id]}, "options": {}},
            ],
            "selection": {"asset_ids": [asset_id]}, "preset_snapshot": {},
            "failure_policy": "fail_fast", "priority": None,
        },
        request_id=str(uuid.uuid4()), project_id=project_id,
    )
    assert service.dispatch(limit=1) == 1
    active = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    series_id = str(active["result"]["series"]["id"])

    assert postgres_repo.pause_processing_series(
        series_id, project_id=project_id, reason="operator review",
    )["status"] == "paused"
    postgres_repo.wake_processing_submission_for_series(series_id)
    paused = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)

    assert paused["status"] == "processing"
    assert paused["progress"]["phase"] == "processing"
    assert paused["progress"]["more_jobs_being_prepared"] is False
    assert paused["progress"]["waiting_for_submission_id"] is None
    assert paused["progress"]["paused_count"] == 1


def test_pausing_during_incremental_planning_stops_new_pages_until_resume(postgres_repo, monkeypatch):
    import Pelagia.services.processing_queue as processing_queue

    project_id, asset_id = _project_with_frames(postgres_repo, count=2)
    monkeypatch.setattr(processing_queue, "INCREMENTAL_FRAME_PAGE_SIZE", 1)
    service = _service(postgres_repo)
    submission = service.submit(
        kind="series",
        request={
            "steps": [{"stage": "preprocess_frames", "filters": {"asset_ids": [asset_id]}, "options": {}}],
            "selection": {"asset_ids": [asset_id]}, "preset_snapshot": {},
            "failure_policy": "fail_fast", "priority": None,
        },
        request_id=str(uuid.uuid4()), project_id=project_id,
    )
    assert service.dispatch(limit=1) == 1
    first = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    series_id = str(first["result"]["series"]["id"])
    assert first["status"] == "planning"
    assert first["progress"]["job_count"] == 1

    director = ProcessingQueueService(AppContext(config=CoreConfig(), repository=postgres_repo))
    assert postgres_repo.pause_processing_series(series_id, project_id=project_id)["status"] == "paused"
    assert service.dispatch(limit=1) == 1
    paused = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    assert paused["status"] == "processing"
    assert paused["progress"]["job_count"] == 1
    assert paused["progress"]["more_jobs_being_prepared"] is False

    assert director.resume_series(series_id, project_id=project_id)["status"] == "active"
    assert service.dispatch(limit=1) == 1
    resumed = postgres_repo.get_processing_submission(str(submission["id"]), project_id=project_id)
    assert resumed["progress"]["job_count"] == 2


def test_overlapping_submission_waits_on_named_reservation_without_hot_spin(postgres_repo):
    project_id, asset_id = _project_with_frames(postgres_repo, count=1)
    service = _service(postgres_repo)
    first = service.submit(
        kind="queue",
        request={"stage": "segment", "filters": {"asset_ids": [asset_id]}, "options": {"padding": 1}},
        request_id=str(uuid.uuid4()), project_id=project_id,
    )
    assert service.dispatch(limit=1) == 1
    assert postgres_repo.get_processing_submission(str(first["id"]), project_id=project_id)["status"] == "processing"

    second = service.submit(
        kind="queue",
        request={"stage": "segment", "filters": {"asset_ids": [asset_id]}, "options": {"padding": 2}},
        request_id=str(uuid.uuid4()), project_id=project_id,
    )
    assert service.dispatch(limit=1) == 1
    waiting = postgres_repo.get_processing_submission(str(second["id"]), project_id=project_id)
    assert waiting["status"] == "planning"
    assert waiting["progress"]["phase"] == "waiting"
    assert waiting["progress"]["waiting_for_submission_id"] == str(first["id"])
    assert waiting["next_plan_at"] is not None
    assert postgres_repo.claim_processing_submission(lease_seconds=60) is None

    cancelled = postgres_repo.cancel_processing_submission(
        str(second["id"]), project_id=project_id, reason="cancel blocked request",
    )
    assert cancelled["status"] == "cancelled"
    assert cancelled["progress"]["phase"] == "cancelled"
    assert cancelled["progress"]["waiting_for_submission_id"] is None


def test_expired_claim_is_fenced_and_locked_materialization_cannot_be_reclaimed(postgres_repo):
    project_id, _ = _project_with_frames(postgres_repo, count=0)
    request = {"stage": "segment", "filters": {"collection": ["none"]}, "options": {}}
    submission, _ = postgres_repo.create_processing_submission(
        project_id=project_id, request_id=str(uuid.uuid4()), kind="queue", request=request,
    )
    first = postgres_repo.claim_processing_submission(lease_seconds=60)
    assert first is not None
    _expire_claim(postgres_repo, str(submission["id"]))
    second = postgres_repo.claim_processing_submission(lease_seconds=60)
    assert second is not None and second["claim_token"] != first["claim_token"]

    with postgres_repo.processing_submission_transaction(
        str(submission["id"]), claim_token=str(first["claim_token"]), project_id=project_id,
    ) as stale_connection:
        assert stale_connection is None
    assert postgres_repo.complete_processing_submission(
        str(submission["id"]), claim_token=str(first["claim_token"]), result={},
    ) is None
    assert postgres_repo.fail_processing_submission(
        str(submission["id"]), claim_token=str(first["claim_token"]), error="stale",
    ) is None

    entered, release = Event(), Event()

    def hold_current_claim():
        with postgres_repo.processing_submission_transaction(
            str(submission["id"]), claim_token=str(second["claim_token"]), project_id=project_id,
        ) as connection:
            assert connection is not None
            connection.execute(
                f"UPDATE {postgres_repo.schema}.processing_submissions "
                "SET claim_expires_at = NOW() - INTERVAL '1 second' WHERE id = %s",
                (submission["id"],),
            )
            entered.set()
            assert release.wait(5)
            assert postgres_repo.complete_processing_submission(
                str(submission["id"]), claim_token=str(second["claim_token"]), result={},
                _connection=connection,
            ) is not None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(hold_current_claim)
        assert entered.wait(5)
        started = monotonic()
        assert postgres_repo.claim_processing_submission(lease_seconds=60) is None
        assert monotonic() - started < 1.0
        release.set()
        future.result(timeout=5)


def test_submission_idempotency_is_concurrent_and_acceptance_does_not_wait_for_planning(postgres_repo):
    project_id, _ = _project_with_frames(postgres_repo, count=0)
    request_id = str(uuid.uuid4())
    request = {"stage": "segment", "filters": {"collection": ["none"]}, "options": {}}
    barrier = Barrier(8)

    def submit_same_request(_index: int):
        barrier.wait(timeout=5)
        return postgres_repo.create_processing_submission(
            project_id=project_id, request_id=request_id, kind="queue", request=request,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(submit_same_request, range(8)))
    assert len({str(row["id"]) for row, _created in results}) == 1
    assert sum(1 for _row, created in results if created) == 1

    submission = results[0][0]
    claimed = postgres_repo.claim_processing_submission(lease_seconds=60)
    assert claimed is not None and claimed["id"] == submission["id"]
    entered, release = Event(), Event()

    def hold_materialization():
        with postgres_repo.processing_submission_transaction(
            str(submission["id"]), claim_token=str(claimed["claim_token"]), project_id=project_id,
        ) as connection:
            assert connection is not None
            entered.set()
            assert release.wait(5)
            assert postgres_repo.complete_processing_submission(
                str(submission["id"]), claim_token=str(claimed["claim_token"]), result={},
                _connection=connection,
            ) is not None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(hold_materialization)
        assert entered.wait(5)
        started = monotonic()
        repeated, created = postgres_repo.create_processing_submission(
            project_id=project_id, request_id=request_id, kind="queue", request=request,
        )
        retry_elapsed = monotonic() - started
        started = monotonic()
        new_submission, new_created = postgres_repo.create_processing_submission(
            project_id=project_id, request_id=str(uuid.uuid4()), kind="queue", request=request,
        )
        new_elapsed = monotonic() - started
        release.set()
        future.result(timeout=5)

    assert repeated["id"] == submission["id"] and created is False
    assert new_submission["id"] == submission["id"] and new_created is False
    assert new_submission["_reused_active_submission"] is True
    assert retry_elapsed < 1.0
    assert new_elapsed < 1.0
