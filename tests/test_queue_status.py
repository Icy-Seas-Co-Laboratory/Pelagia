from __future__ import annotations

from datetime import datetime, timezone

import pytest

from Pelagia.services.queue_status import QueueStatusService, normalize_job


AS_OF = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


class SnapshotRepository:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = []

    def get_queue_status_snapshot(self, **kwargs):
        self.calls.append(kwargs)
        return self.snapshot


def _counts(**overrides):
    row = {
        "job_count": 3,
        "active": 2,
        "historical": 1,
        "queued": 1,
        "leased": 1,
        "working": 0,
        "paused": 0,
        "succeeded": 1,
        "failed": 0,
        "cancelled": 0,
        "dead_lettered": 0,
        "known_total_jobs": 3,
        "known_total_units": 30,
        "completed_units": 15,
        "failed_units": 1,
        "skipped_units": 2,
    }
    row.update(overrides)
    return row


def test_queue_status_keeps_mixed_units_separate_and_marks_combined_progress_unavailable():
    snapshot = {
        "as_of": AS_OF,
        "total": _counts(expired_leases=1, stale_active_jobs=2, pending_controls=1),
        "by_stage": [_counts(stage="segment")],
        "work_groups": [
            _counts(stage="segment", unit="frames", job_count=2, known_total_jobs=2,
                    unknown_total_jobs=0, unknown_active_jobs=0, known_total_units=20,
                    completed_units=10, remaining_units=10, units_per_second=2),
            _counts(stage="segment", unit="rois", job_count=1, active=0, historical=1,
                    known_total_jobs=1, unknown_total_jobs=0, unknown_active_jobs=0,
                    known_total_units=10, completed_units=5, remaining_units=0,
                    units_per_second=0),
        ],
        "jobs": [],
        "recent_failures": [],
    }
    repository = SnapshotRepository(snapshot)

    status = QueueStatusService(repository).snapshot(project_id="project-1")

    assert [(group["stage"], group["unit"]) for group in status["work_groups"]] == [
        ("segment", "frames"),
        ("segment", "rois"),
    ]
    assert status["work_groups"][0]["progress"]["percent"] == pytest.approx(50)
    assert status["work_groups"][0]["eta"]["seconds"] == pytest.approx(5)
    assert status["total"]["progress"]["percent"] is None
    assert status["total"]["progress"]["completed_units"] is None
    assert status["total"]["progress"]["reason"] == "mixed_units"
    assert status["by_stage"][0]["progress"]["reason"] == "mixed_units"
    assert status["health"] == {
        "expired_leases": 1, "stale_active_jobs": 2, "pending_controls": 1,
        "progress_coverage_percent": 0, "oldest_queued_age_seconds": None,
    }


def test_queue_status_preserves_scope_and_bounded_row_metadata():
    snapshot = {
        "as_of": AS_OF,
        "total": _counts(job_count=4),
        "by_stage": [],
        "work_groups": [],
        "jobs": [{"id": "job-1", "status": "queued", "progress": {}}],
        "recent_failures": [],
    }
    repository = SnapshotRepository(snapshot)

    status = QueueStatusService(repository).snapshot(
        project_id="project-1",
        run_id="run-1",
        statuses=["queued"],
        stages=["segment"],
        job_ids=["job-1"],
        job_limit=1,
    )

    assert status["scope"] == {
        "project_id": "project-1", "run_id": "run-1", "asset_id": None,
        "status": ["queued"], "stage": ["segment"], "ids": ["job-1"], "worker_id": None,
    }
    assert repository.calls[0]["project_id"] == "project-1"
    assert status["jobs"]["matching_count"] == 4
    assert status["jobs"]["truncated"] is True
    assert status["jobs"]["limit"] == 1


def test_unknown_total_jobs_suppress_cohort_percent():
    snapshot = {
        "as_of": AS_OF,
        "total": _counts(unknown_total_jobs=1),
        "by_stage": [],
        "work_groups": [],
        "jobs": [],
        "recent_failures": [],
    }

    status = QueueStatusService(SnapshotRepository(snapshot)).snapshot(project_id="project-1")

    assert status["total"]["progress"]["percent"] is None
    assert status["total"]["progress"]["reason"] == "unknown_total"


def test_normalized_progress_rejects_rate_from_a_previous_retry_attempt():
    job = normalize_job({
        "id": "job-1",
        "status": "leased",
        "attempt_count": 2,
        "progress": {
            "attempt_count": 1,
            "total": 10,
            "total_known": True,
            "completed": 8,
            "rates": {"units_per_second": 4, "method": "recent_progress", "window_seconds": 2},
        },
    })

    assert job["progress"]["attempt_current"] is False
    assert job["progress"]["completed"] == 0
    assert job["progress"]["total"] is None
    assert job["progress"]["percent"] is None
    assert job["progress"]["rate"] == {
        "units_per_second": None,
        "method": "recent_progress",
        "window_seconds": 2.0,
        "available": False,
        "reason": "awaiting_attempt_progress",
    }
    assert job["progress"]["eta"]["reason"] == "awaiting_attempt_progress"


def test_normalized_progress_calculates_rate_from_durable_sample():
    job = normalize_job({
        "status": "leased",
        "attempt_count": 1,
        "lease_expires_at": "2026-10-22T12:01:00+00:00",
        "progress": {
            "attempt_count": 1,
            "observed_at": "2026-10-22T12:00:00+00:00",
            "total": 20,
            "total_known": True,
            "completed": 10,
            "rate_sample": {"completed_delta": 6, "elapsed_seconds": 3, "sample_count": 2},
            "rates": {"units_per_second": 999, "method": "recent_progress", "window_seconds": 3},
        },
    }, as_of=AS_OF)

    assert job["progress"]["rate"]["units_per_second"] == pytest.approx(2)
    assert job["progress"]["eta"]["seconds"] == pytest.approx(5)


def test_terminal_job_overrides_finalizing_phase():
    job = normalize_job({
        "status": "succeeded",
        "attempt_count": 1,
        "progress": {"attempt_count": 1, "phase": "finalizing", "total": 1, "total_known": True, "completed": 1},
    }, as_of=AS_OF)

    assert job["progress"]["phase"] == "complete"
    assert job["progress"]["rate"]["reason"] == "job_not_active"


@pytest.mark.parametrize(
    ("status", "total_known", "expected_reason"),
    [("leased", False, "insufficient_progress_samples"), ("paused", True, "paused")],
)
def test_normalized_progress_explains_unavailable_eta(status, total_known, expected_reason):
    job = normalize_job({
        "status": status,
        "attempt_count": 1,
        "progress": {"attempt_count": 1, "total": 0, "total_known": total_known, "completed": 0},
    })

    assert job["progress"]["eta"]["available"] is False
    assert job["progress"]["eta"]["reason"] == expected_reason
