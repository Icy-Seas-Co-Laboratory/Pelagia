from datetime import datetime
from types import SimpleNamespace

import pytest

from Pelagia.workers.progress import JobProgressReporter


@pytest.fixture
def reporting(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    writes = []

    def persist(job_id, progress, **kwargs):
        writes.append(progress)
        return {"id": job_id, "progress": progress}

    monkeypatch.setattr("Pelagia.workers.progress.monotonic", lambda: clock.now)
    context = SimpleNamespace(repository=SimpleNamespace(update_job_progress=persist))
    return clock, writes, context


def test_reporter_uses_recent_observations_and_expires_old_throughput(reporting):
    clock, writes, context = reporting
    reporter = JobProgressReporter({"id": "job", "attempt_count": 1}, context, stage="segment", unit="frames", total=100)
    reporter.start()
    assert writes[-1]["rates"]["units_per_second"] is None
    clock.now = 10
    reporter.update(20, force=True)
    assert writes[-1]["rates"]["units_per_second"] == 2
    clock.now = 40
    reporter.update(30, force=True)
    assert writes[-1]["rates"]["units_per_second"] == pytest.approx(1 / 3)
    clock.now = 80
    reporter.update(30, force=True)
    assert writes[-1]["rates"]["units_per_second"] == 0
    assert writes[-1]["rates"]["method"] == "recent_progress"
    assert datetime.fromisoformat(writes[-1]["observed_at"]).tzinfo is not None


def test_resumed_progress_does_not_count_existing_work_as_new_throughput(reporting):
    clock, writes, context = reporting
    first = JobProgressReporter({"id": "job", "attempt_count": 1}, context, stage="segment", unit="frames", total=100)
    first.start()
    clock.now = 10
    first.update(50, force=True)
    resumed = JobProgressReporter({"id": "job", "attempt_count": 2}, context, stage="segment", unit="frames", total=100)
    resumed.update(50, force=True)
    assert writes[-1]["rates"]["units_per_second"] is None
    assert writes[-1]["attempt_id"] != writes[0]["attempt_id"]
    assert writes[-1]["attempt_count"] == 2
    clock.now = 20
    resumed.update(60, force=True)
    assert writes[-1]["rates"]["units_per_second"] == 1
    clock.now = 30
    resumed.update(0, force=True)
    assert writes[-1]["rates"]["units_per_second"] is None


def test_discovered_total_and_finalization_remain_distinct_from_job_success(reporting):
    clock, writes, context = reporting
    job = {"id": "job", "status": "working"}
    reporter = JobProgressReporter(job, context, stage="extract_frames", unit="frames", total=0)
    reporter.start()
    clock.now = 10
    reporter.update(10, force=True)
    assert writes[-1]["total_known"] is False
    assert writes[-1]["percent"] is None
    reporter.total = 20
    clock.now = 20
    reporter.finish()
    assert writes[-1]["total_known"] is True
    assert writes[-1]["percent"] == 100
    assert writes[-1]["phase"] == "finalizing"
    assert job["status"] == "working"


def test_unknown_total_finish_preserves_observed_work_and_unit_change_resets_rate(reporting):
    clock, writes, context = reporting
    reporter = JobProgressReporter({"id": "job"}, context, stage="export_bundle", unit="input_record", total=None)
    reporter.start()
    clock.now = 10
    reporter.update(10, force=True)
    clock.now = 20
    reporter.unit = "bytes"
    reporter.update(1000, force=True)
    assert writes[-1]["rates"]["units_per_second"] is None
    reporter.finish()
    assert writes[-1]["completed"] == 1000
    assert writes[-1]["total_known"] is False


def test_explicit_empty_workload_is_distinct_from_unknown_total(reporting):
    _, writes, context = reporting
    reporter = JobProgressReporter({"id": "job"}, context, stage="segment", unit="frames", total=0, total_known=True)
    reporter.finish()
    assert writes[-1]["total"] == 0
    assert writes[-1]["total_known"] is True


def test_recovered_outputs_advance_progress_without_inflating_rate(reporting):
    clock, writes, context = reporting
    reporter = JobProgressReporter({"id": "job"}, context, stage="classify", unit="rois", total=1000)
    reporter.start()
    clock.now = 1
    reporter.update(900, reset_rate=True, current={"phase": "resuming"})
    assert writes[-1]["completed"] == 900
    assert writes[-1]["rates"]["units_per_second"] is None
    clock.now = 11
    reporter.update(910, force=True)
    assert writes[-1]["rates"]["units_per_second"] == 1
