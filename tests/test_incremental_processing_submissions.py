"""PostgreSQL regressions for checkpointed processing-submission planning."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid

import pytest

from Pelagia.config import CoreConfig
from Pelagia.domain import AssetKind, DetectionRecord, FrameRecord, PlannedRun, RawAssetManifest, RunManifest
from Pelagia.services.context import AppContext
from Pelagia.services.processing_queue import ProcessingSubmissionService
from test_postgres import postgres_repo


def _project_with_frames(repository, *, count: int) -> tuple[str, str]:
    project = repository.create_project(f"incremental-submission-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    run_id, asset_id = str(uuid.uuid4()), str(uuid.uuid4())
    repository.register_planned_run(
        PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=f"incremental-run-{uuid.uuid4().hex}",
                instrument="pytest",
                source_path="/tmp/incremental.avi",
                source_type=AssetKind.VIDEO.value,
                created_at=datetime.now(timezone.utc),
                assets=[RawAssetManifest(
                    asset_id=asset_id,
                    filename="incremental.avi",
                    path="/tmp/incremental.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=1,
                    checksum=f"sha256:{uuid.uuid4().hex}",
                )],
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


def _submission(service, *, project_id: str, asset_id: str, request_id: str | None = None, options=None):
    return service.submit(
        kind="queue",
        request={
            "stage": "segment",
            "filters": {"asset_ids": [asset_id]},
            "options": options or {},
            "priority": None,
            "dry_run": False,
        },
        request_id=request_id or str(uuid.uuid4()),
        project_id=project_id,
    )


def _submission_jobs(repository, submission_id: str) -> list[dict]:
    with repository.connect() as connection:
        return connection.execute(
            f"""SELECT jobs.* FROM {repository.schema}.processing_jobs jobs
            JOIN {repository.schema}.processing_submission_jobs linked ON linked.job_id = jobs.id
            WHERE linked.submission_id = %s ORDER BY jobs.created_at, jobs.id""",
            (submission_id,),
        ).fetchall()


def test_first_page_is_claimable_before_later_pages_and_restart_has_no_duplicate_units(postgres_repo, monkeypatch):
    """Workers can consume a committed page while a new service resumes planning."""
    import Pelagia.services.processing_queue as processing_queue

    project_id, asset_id = _project_with_frames(postgres_repo, count=3)
    monkeypatch.setattr(processing_queue, "INCREMENTAL_FRAME_PAGE_SIZE", 1)
    first_service = _service(postgres_repo)
    submitted = _submission(first_service, project_id=project_id, asset_id=asset_id)

    assert first_service.dispatch(limit=1) == 1
    checkpoint = postgres_repo.get_processing_submission(str(submitted["id"]), project_id=project_id)
    assert checkpoint["status"] == "planning"
    assert checkpoint["progress"]["scheduled_count"] == 1
    assert checkpoint["progress"]["more_jobs_being_prepared"] is True
    assert checkpoint["planning_cursor"]["frame_id"]

    # A worker need not wait for the selection to be fully enumerated.
    claimed = postgres_repo.claim_jobs("incremental-test-worker", stages=[], limit=1)
    assert len(claimed) == 1
    assert claimed[0]["payload"]["frame_ids"]

    # Simulate a new dispatcher process after the first page's transaction committed.
    restarted_service = _service(postgres_repo)
    assert restarted_service.dispatch(limit=1) == 1
    assert restarted_service.dispatch(limit=1) == 1
    finished_planning = postgres_repo.get_processing_submission(str(submitted["id"]), project_id=project_id)
    assert finished_planning["status"] == "processing"
    assert finished_planning["progress"]["scheduled_count"] == 3
    assert finished_planning["progress"]["job_count"] == 3

    jobs = _submission_jobs(postgres_repo, str(submitted["id"]))
    frame_ids = [frame_id for job in jobs for frame_id in job["payload"]["frame_ids"]]
    assert len(frame_ids) == 3
    assert len(set(frame_ids)) == 3


def test_frame_byte_limit_creates_checkpointed_pages_before_frame_count_limit(postgres_repo, monkeypatch):
    """Decoded input size can split durable pages without losing the seek cursor."""
    import Pelagia.services.processing_queue as processing_queue

    project_id, asset_id = _project_with_frames(postgres_repo, count=3)
    # The fixture frames are 4x4 uint8 arrays (16 decoded bytes each).  A
    # 20-byte cap therefore permits one frame despite the 1,000-frame cap.
    monkeypatch.setattr(processing_queue, "SEGMENT_INPUT_BYTES_PER_JOB", 20)
    service = _service(postgres_repo)
    submitted = _submission(service, project_id=project_id, asset_id=asset_id)

    assert service.dispatch(limit=1) == 1
    first = postgres_repo.get_processing_submission(str(submitted["id"]), project_id=project_id)
    assert first["status"] == "planning"
    assert first["progress"]["scheduled_count"] == 1
    assert first["planning_cursor"]["frame_id"]
    assert len(_submission_jobs(postgres_repo, str(submitted["id"]))) == 1

    assert service.dispatch(limit=1) == 1
    assert service.dispatch(limit=1) == 1
    finished = postgres_repo.get_processing_submission(str(submitted["id"]), project_id=project_id)
    assert finished["status"] == "processing"
    jobs = _submission_jobs(postgres_repo, str(submitted["id"]))
    assert [len(job["payload"]["frame_ids"]) for job in jobs] == [1, 1, 1]


def test_roi_byte_limit_pages_exclude_expansion_frames(postgres_repo):
    """Refinement budgets crops and masks, not their source frames."""
    project_id, asset_id = _project_with_frames(postgres_repo, count=2)
    frames = postgres_repo.list_frames(asset_id)
    run_id = str(frames[0]["run_id"])
    with postgres_repo.connect() as connection:
        connection.execute(
            f"""UPDATE {postgres_repo.schema}.frames
            SET preprocessed_payload_shape = %s::jsonb, preprocessed_payload_dtype = 'uint8'
            WHERE id = ANY(%s::uuid[])""",
            (json.dumps([30_000_000]), [str(frame["id"]) for frame in frames]),
        )
        connection.commit()

    records = [
        DetectionRecord(
            run_id=run_id,
            frame_id=str(frame["id"]),
            roi_index=index,
            bbox_x=0, bbox_y=0, bbox_w=20_000_000, bbox_h=1,
            area=1.0, perimeter=4.0, major_axis_length=1.0, minor_axis_length=1.0,
            min_gray_value=0, mean_gray_value=0.0,
            roi_payload=b"r", mask_payload=b"m",
            crop_bbox_x=0, crop_bbox_y=0, crop_bbox_w=20_000_000, crop_bbox_h=1,
            roi_encoding="raw", roi_format="raw_ndarray_c_order", roi_dtype="uint8",
            roi_shape=[20_000_000],
            mask_encoding="raw", mask_format="raw_ndarray_c_order", mask_dtype="uint8",
            mask_shape=[10_000_000],
        )
        for frame, index in ((frames[0], 0), (frames[0], 1), (frames[1], 0))
    ]
    inserted = postgres_repo.replace_detections(run_id, asset_id, records)
    inserted_ids = {str(record["id"]) for record in inserted}
    # The durable seek order is creation time, so make the shared-frame pair
    # adjacent regardless of database insert timing or UUID order.
    with postgres_repo.connect() as connection:
        connection.execute(
            f"""UPDATE {postgres_repo.schema}.detection_candidate
            SET created_at = NOW() - INTERVAL '30 seconds'
                + CASE WHEN frame_id = %s::uuid THEN roi_index * INTERVAL '1 second'
                       ELSE INTERVAL '2 seconds' END
            WHERE id = ANY(%s::uuid[])""",
            (str(frames[0]["id"]), list(inserted_ids)),
        )
        connection.commit()

    service = _service(postgres_repo)
    submitted = service.submit(
        kind="queue",
        request={
            "stage": "roi_refinement",
            "filters": {"asset_ids": [asset_id]},
            "options": {"allow_frame_expansion": True},
            "priority": None,
            "dry_run": False,
        },
        request_id=str(uuid.uuid4()),
        project_id=project_id,
    )

    assert service.dispatch(limit=1) == 1
    finished = postgres_repo.get_processing_submission(str(submitted["id"]), project_id=project_id)
    assert finished["status"] == "processing"
    jobs = _submission_jobs(postgres_repo, str(submitted["id"]))
    scheduled_ids = [detection_id for job in jobs for detection_id in job["payload"]["detection_ids"]]
    assert len(jobs) == 1
    assert set(scheduled_ids) == inserted_ids
    assert len(scheduled_ids) == len(set(scheduled_ids)) == 3


def test_equivalent_active_selection_is_reused_but_settings_change_creates_new_submission(postgres_repo):
    project_id, asset_id = _project_with_frames(postgres_repo, count=0)
    service = _service(postgres_repo)
    other_asset = str(uuid.uuid4())
    first = service.submit(
        kind="queue",
        request={
            "stage": "segment",
            "filters": {"asset_ids": [asset_id, other_asset], "collection": ["z", "a"]},
            "options": {}, "priority": None, "dry_run": False,
        },
        request_id=str(uuid.uuid4()), project_id=project_id,
    )
    legacy_request = dict(first["request"])
    legacy_request["filters"] = {
        **legacy_request["filters"],
        "asset_ids": list(reversed(legacy_request["filters"]["asset_ids"])),
        "collection": list(reversed(legacy_request["filters"]["collection"])),
    }
    with postgres_repo.connect() as connection:
        connection.execute(
            f"UPDATE {postgres_repo.schema}.processing_submissions "
            "SET request = %s::jsonb, request_hash = 'legacy-pre-0032-hash' WHERE id = %s",
            (json.dumps(legacy_request), first["id"]),
        )
        connection.commit()
    idempotent_retry = service.submit(
        kind="queue", request={
            "stage": "segment",
            "filters": {"asset_ids": [asset_id, other_asset], "collection": ["z", "a"]},
            "options": {}, "priority": None, "dry_run": False,
        },
        request_id=str(first["request_id"]), project_id=project_id,
    )
    equivalent = service.submit(
        kind="queue",
        request={
            "stage": "segment",
            "filters": {"collection": ["a", "z"], "asset_ids": [other_asset, asset_id]},
            "options": {}, "priority": None, "dry_run": False,
        },
        request_id=str(uuid.uuid4()), project_id=project_id,
    )
    changed = _submission(
        service, project_id=project_id, asset_id=asset_id,
        options={"apply_mask": True},
    )

    assert idempotent_retry["id"] == first["id"]
    assert equivalent["id"] == first["id"]
    assert equivalent["_reused_active_submission"] is True
    assert changed["id"] != first["id"]


def test_equivalent_selection_can_be_submitted_again_after_terminal_completion(postgres_repo):
    project_id, asset_id = _project_with_frames(postgres_repo, count=0)
    service = _service(postgres_repo)
    first = _submission(service, project_id=project_id, asset_id=asset_id)
    assert service.dispatch(limit=1) == 1
    assert postgres_repo.get_processing_submission(
        str(first["id"]), project_id=project_id,
    )["status"] == "succeeded"

    repeated = _submission(service, project_id=project_id, asset_id=asset_id)

    assert repeated["id"] != first["id"]
    assert not repeated.get("_reused_active_submission", False)


def test_submission_progress_separates_paused_from_running_jobs(postgres_repo):
    project_id, asset_id = _project_with_frames(postgres_repo, count=1)
    service = _service(postgres_repo)
    submitted = _submission(service, project_id=project_id, asset_id=asset_id)
    assert service.dispatch(limit=1) == 1
    job = _submission_jobs(postgres_repo, str(submitted["id"]))[0]

    paused = postgres_repo.pause_job(
        str(job["id"]), project_id=project_id, reason="operator review",
    )
    assert paused is not None and paused["status"] == "paused"
    progress = postgres_repo.get_processing_submission(
        str(submitted["id"]), project_id=project_id,
    )["progress"]
    assert progress["paused_count"] == 1
    assert progress["running_count"] == 0
    assert progress["queued_count"] == 0


def test_cancellation_after_first_page_prevents_future_planning_and_cancels_linked_job(postgres_repo, monkeypatch):
    import Pelagia.services.processing_queue as processing_queue

    project_id, asset_id = _project_with_frames(postgres_repo, count=3)
    monkeypatch.setattr(processing_queue, "INCREMENTAL_FRAME_PAGE_SIZE", 1)
    service = _service(postgres_repo)
    submitted = _submission(service, project_id=project_id, asset_id=asset_id)
    assert service.dispatch(limit=1) == 1
    first_job = _submission_jobs(postgres_repo, str(submitted["id"]))[0]

    cancelled = postgres_repo.cancel_processing_submission(
        str(submitted["id"]), project_id=project_id, reason="test cancellation",
    )
    assert cancelled is not None and cancelled["status"] == "cancelled"
    assert cancelled["progress"]["phase"] == "cancelled"
    assert cancelled["progress"]["cancelled_count"] == 1
    assert cancelled["progress"]["failed_count"] == 0
    assert cancelled["progress"]["running_count"] == 0
    assert postgres_repo.get_job(str(first_job["id"]), project_id=project_id)["status"] == "cancelled"
    assert service.dispatch(limit=3) == 0
    assert len(_submission_jobs(postgres_repo, str(submitted["id"]))) == 1

    with postgres_repo.connect() as connection:
        statuses = connection.execute(
            f"""SELECT candidate_detection_status FROM {postgres_repo.schema}.frame_processing_status
            WHERE project_id = %s ORDER BY frame_index""",
            (project_id,),
        ).fetchall()
    assert [row["candidate_detection_status"] for row in statuses].count("cancelled") == 1

    retried = _submission(service, project_id=project_id, asset_id=asset_id)
    assert retried["id"] != submitted["id"]
    assert service.dispatch(limit=1) == 1
    assert len(_submission_jobs(postgres_repo, str(retried["id"]))) == 1


def test_refinement_seek_preserves_generation_order_and_legacy_cursors(postgres_repo):
    project_id, asset_id = _project_with_frames(postgres_repo, count=2)
    frames = sorted(postgres_repo.list_frames(asset_id), key=lambda row: row["frame_index"])
    run_id = str(frames[0]["run_id"])
    records = [
        DetectionRecord(
            run_id=run_id, frame_id=str(frame["id"]), roi_index=index,
            bbox_x=0, bbox_y=0, bbox_w=2, bbox_h=2, area=4,
            perimeter=8, major_axis_length=2, minor_axis_length=2,
            min_gray_value=0, mean_gray_value=0, roi_payload=bytes([index + 1]),
            roi_encoding="raw", roi_format="raw_ndarray_c_order", roi_dtype="uint8", roi_shape=[2, 2],
        )
        for frame in frames for index in range(3)
    ]
    inserted = postgres_repo.replace_detections(run_id, asset_id, records)
    expected = [str(row["id"]) for row in inserted]
    boundary, cursor, actual = None, None, []
    while True:
        page = postgres_repo.plan_processing_submission_page(
            submission_id=str(uuid.uuid4()), project_id=project_id, stage="roi_refinement",
            filters={"asset_ids": [asset_id], "refinement_state": ["refined", "unrefined"]},
            boundary=boundary, planning_cursor=cursor, limit=1,
        )
        actual.extend(str(row["detection_id"]) for row in page["rows"])
        boundary, cursor = page["boundary"], page["cursor"]
        if not page["has_more"]:
            break
    assert boundary["roi_order"] == "generation_v1"
    assert actual == expected

    legacy_boundary = {"created_before": boundary["created_before"]}
    cursor, legacy_ids = None, []
    while True:
        page = postgres_repo.plan_processing_submission_page(
            submission_id=str(uuid.uuid4()), project_id=project_id, stage="roi_refinement",
            filters={"asset_ids": [asset_id], "refinement_state": ["refined", "unrefined"]},
            boundary=legacy_boundary, planning_cursor=cursor, limit=1,
        )
        legacy_ids.extend(str(row["detection_id"]) for row in page["rows"])
        cursor = page["cursor"]
        if not page["has_more"]:
            break
    assert set(legacy_ids) == set(expected)
    assert len(legacy_ids) == len(expected)
    assert "frame_index" not in cursor


@pytest.mark.parametrize("stage", ["preprocess_frames", "segment"])
def test_frame_seek_uses_asset_frame_order_and_legacy_cursor(postgres_repo, stage):
    project_id, asset_id = _project_with_frames(postgres_repo, count=4)
    frames = sorted(postgres_repo.list_frames(asset_id), key=lambda row: row["frame_index"])
    expected = [str(row["id"]) for row in frames]
    boundary, cursor, actual = None, None, []
    while True:
        page = postgres_repo.plan_processing_submission_page(
            submission_id=str(uuid.uuid4()), project_id=project_id, stage=stage,
            filters={"asset_ids": [asset_id]}, boundary=boundary, planning_cursor=cursor, limit=1,
        )
        actual.extend(str(row["frame_id"]) for row in page["rows"])
        boundary, cursor = page["boundary"], page["cursor"]
        if not page["has_more"]:
            break
    assert boundary["frame_order"] == "asset_frame_v1"
    assert actual == expected

    legacy_boundary = {"created_before": boundary["created_before"]}
    cursor, actual = None, []
    while True:
        page = postgres_repo.plan_processing_submission_page(
            submission_id=str(uuid.uuid4()), project_id=project_id, stage=stage,
            filters={"asset_ids": [asset_id]}, boundary=legacy_boundary, planning_cursor=cursor, limit=1,
        )
        actual.extend(str(row["frame_id"]) for row in page["rows"])
        cursor = page["cursor"]
        if not page["has_more"]:
            break
    assert set(actual) == set(expected)
    assert len(actual) == len(expected)
    assert "asset_id" not in cursor


def test_refinement_queue_counts_only_candidates_with_crops(postgres_repo):
    project_id, asset_id = _project_with_frames(postgres_repo, count=1)
    frame = postgres_repo.list_frames(asset_id)[0]
    run_id, frame_id = str(frame["run_id"]), str(frame["id"])
    def record(index, payload):
        return DetectionRecord(
            run_id=run_id, frame_id=frame_id, roi_index=index,
            bbox_x=0, bbox_y=0, bbox_w=2, bbox_h=2, area=4,
            perimeter=8, major_axis_length=2, minor_axis_length=2,
            min_gray_value=0, mean_gray_value=0, roi_payload=payload,
            roi_encoding="raw", roi_format="raw_ndarray_c_order", roi_dtype="uint8", roi_shape=[2, 2],
        )
    rows = postgres_repo.replace_detections(run_id, asset_id, [record(0, b"crop1"), record(1, b"crop2"), record(2, None)])
    postgres_repo.upsert_refined_detections(
        [(str(rows[0]["id"]), record(0, b"refined")), (str(rows[0]["id"]), record(1, b"refined2")),
         (str(rows[2]["id"]), record(2, b"old-refined"))], project_id=project_id,
    )
    assert postgres_repo.get_refinement_queue_counts(project_id=project_id) == {
        "candidates": 2, "refined": 1, "unrefined": 1,
    }
    assert postgres_repo.get_refinement_queue_counts(project_id=project_id, asset_ids=[str(uuid.uuid4())]) == {
        "candidates": 0, "refined": 0, "unrefined": 0,
    }
    other_project, _ = _project_with_frames(postgres_repo, count=0)
    assert postgres_repo.get_refinement_queue_counts(project_id=other_project) == {
        "candidates": 0, "refined": 0, "unrefined": 0,
    }
