from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4


ACTIVE_STATUSES = frozenset({"queued", "leased", "working", "paused"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "dead_lettered"})


def _float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        resolved = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return resolved.astimezone(timezone.utc).isoformat()
    return str(value)


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


def _progress_totals(row: Mapping[str, Any]) -> dict[str, Any]:
    total = _float(row.get("known_total_units"))
    completed = _float(row.get("completed_units")) or 0.0
    failed = _float(row.get("failed_units")) or 0.0
    skipped = _float(row.get("skipped_units")) or 0.0
    unknown_jobs = int(row.get("unknown_total_jobs") or 0)
    total = None if total is None or int(row.get("known_total_jobs") or 0) <= 0 else total
    result = {
        "known_total_units": total,
        "completed_units": completed,
        "failed_units": failed,
        "skipped_units": skipped,
        "unknown_total_jobs": unknown_jobs,
        "percent": min(100.0, max(0.0, completed / total * 100.0)) if not unknown_jobs and total and total > 0 else None,
    }
    if unknown_jobs:
        result["reason"] = "unknown_total"
    return result


def _job_counts(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "job_count": int(row.get("job_count") or 0),
        "active": int(row.get("active") or 0),
        "historical": int(row.get("historical") or 0),
    }
    for status in ("queued", "leased", "working", "paused", "succeeded", "failed", "cancelled", "dead_lettered"):
        result[status] = int(row.get(status) or 0)
    result["progress"] = _progress_totals(row)
    return result


def _attempt_is_current(job: Mapping[str, Any], progress: Mapping[str, Any]) -> bool:
    progress_attempt = progress.get("attempt_count")
    if progress_attempt is None:
        return True
    try:
        return int(progress_attempt) == int(job.get("attempt_count") or 0)
    except (TypeError, ValueError):
        return False


def normalize_job(
    job: Mapping[str, Any],
    *,
    as_of: datetime | None = None,
    stale_after_seconds: int = 120,
) -> dict[str, Any]:
    """Return a compact job row with one normalized progress/rate/ETA shape."""
    normalized = dict(job)
    progress = dict(normalized.get("progress") or {})
    total = _float(progress.get("total"))
    completed = _float(progress.get("completed")) or 0.0
    total_known = total is not None and total >= 0 and progress.get("total_known", total > 0) is True
    attempt_current = _attempt_is_current(normalized, progress)
    failed = _float(progress.get("failed")) or 0.0
    skipped = _float(progress.get("skipped")) or 0.0
    if not attempt_current:
        # A requeued job retains the prior attempt's durable progress until its
        # new reporter starts.  Do not present those counters as current work.
        total = None
        completed = 0.0
        failed = 0.0
        skipped = 0.0
        total_known = False
    status = str(normalized.get("status") or "")
    active = status in ACTIVE_STATUSES
    executing = status in {"leased", "working"}
    phase = str(progress.get("phase") or ("waiting" if status == "queued" else "processing" if active else "complete"))
    if status in TERMINAL_STATUSES:
        phase = "complete"
    elif executing and total_known and total is not None and completed >= total:
        phase = "finalizing"
    observed = _datetime(progress.get("recorded_at") or progress.get("observed_at") or normalized.get("updated_at"))
    snapshot_time = as_of or datetime.now(timezone.utc)
    stale = bool(observed and (snapshot_time - observed).total_seconds() > max(1, stale_after_seconds))
    lease_expires = _datetime(normalized.get("lease_expires_at"))
    expired_lease = bool(executing and lease_expires and lease_expires < snapshot_time)

    rates = progress.get("rates") if isinstance(progress.get("rates"), Mapping) else {}
    rate_sample = progress.get("rate_sample") if isinstance(progress.get("rate_sample"), Mapping) else {}
    completed_delta = _float(rate_sample.get("completed_delta"))
    sample_elapsed = _float(rate_sample.get("elapsed_seconds"))
    if completed_delta is not None and completed_delta >= 0 and sample_elapsed is not None and sample_elapsed > 0:
        rate_value = completed_delta / sample_elapsed
        rate_method = "recent_progress"
    else:
        rate_value = _float(rates.get("units_per_second"))
        rate_method = rates.get("method") or ("worker_lifetime" if rate_value is not None else None)
    if not attempt_current:
        rate_reason = "awaiting_attempt_progress"
    elif status == "paused":
        rate_reason = "paused"
    elif phase == "finalizing":
        rate_reason = "finalizing"
    elif status == "queued":
        rate_reason = "waiting_to_start"
    elif not active:
        rate_reason = "job_not_active"
    elif expired_lease:
        rate_reason = "expired_lease"
    elif stale:
        rate_reason = "stale_progress"
    elif not executing:
        rate_reason = "job_not_executing"
    elif rate_value is None or rate_value <= 0:
        rate_reason = "insufficient_progress_samples"
    else:
        rate_reason = None

    remaining = max(0.0, total - completed) if total_known and total is not None else None
    if rate_reason is not None:
        eta_reason = rate_reason
    elif remaining is None:
        eta_reason = "unknown_total"
    else:
        eta_reason = None

    progress.update(
        {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total_known": total_known,
            "attempt_current": attempt_current,
            "phase": phase,
            "current": progress.get("current") if attempt_current else {},
            "secondary": progress.get("secondary") if attempt_current else {},
            "observed_at": _iso(progress.get("observed_at")),
            "stale": stale,
            "expired_lease": expired_lease,
            "rate": {
                "units_per_second": rate_value if rate_reason is None else None,
                "method": rate_method,
                "window_seconds": _float(rates.get("window_seconds")),
                "available": rate_reason is None,
                "reason": rate_reason,
            },
            "eta": {
                "seconds": remaining / rate_value if eta_reason is None and rate_value else None,
                "available": eta_reason is None,
                "reason": eta_reason,
            },
        }
    )
    progress["percent"] = min(100.0, max(0.0, completed / total * 100.0)) if total_known and total and total > 0 else None
    normalized["progress"] = progress
    return normalized


class QueueStatusService:
    """Build the versioned, project-scoped queue status read model."""

    def __init__(self, repository: Any):
        self.repository = repository
        self.jobs = getattr(repository, "jobs", repository)

    def snapshot(
        self,
        *,
        project_id: str,
        run_id: str | None = None,
        asset_id: str | None = None,
        statuses: Sequence[str] = (),
        stages: Sequence[str] = (),
        job_ids: Sequence[str] = (),
        worker_id: str | None = None,
        job_limit: int = 50,
        recent_failure_limit: int = 10,
    ) -> dict[str, Any]:
        raw = self.jobs.get_queue_status_snapshot(
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            statuses=statuses,
            stages=stages,
            job_ids=job_ids,
            worker_id=worker_id,
            job_limit=job_limit,
            recent_failure_limit=recent_failure_limit,
        )
        as_of = raw.get("as_of") or datetime.now(timezone.utc)
        as_of_datetime = _datetime(as_of) or datetime.now(timezone.utc)
        stale_after_seconds = int(raw.get("stale_after_seconds") or 120)
        groups = [self._work_group(row) for row in raw.get("work_groups") or []]
        unit_count = len({str(group["unit"]) for group in groups})
        stage_count = len({str(group["stage"]) for group in groups})

        total = _job_counts(raw.get("total") or {})
        if unit_count > 1 or stage_count > 1:
            reason = "mixed_units" if unit_count > 1 else "multiple_stages"
            total["progress"] = self._mixed_progress(total["progress"], reason=reason)

        by_stage = []
        for row in raw.get("by_stage") or []:
            item = {"stage": str(row.get("stage")), **_job_counts(row)}
            stage_units = {str(group["unit"]) for group in groups if group["stage"] == item["stage"]}
            if len(stage_units) > 1:
                item["progress"] = self._mixed_progress(item["progress"])
            by_stage.append(item)

        matching_count = int((raw.get("total") or {}).get("job_count") or 0)
        active_count = int((raw.get("total") or {}).get("active") or 0)
        active_known_count = int((raw.get("total") or {}).get("active_known_total_jobs") or 0)
        jobs = [normalize_job(job, as_of=as_of_datetime, stale_after_seconds=stale_after_seconds) for job in raw.get("jobs") or []]
        failures = [
            normalize_job(job, as_of=as_of_datetime, stale_after_seconds=stale_after_seconds)
            for job in raw.get("recent_failures") or []
        ]
        return {
            "schema_version": 1,
            "snapshot_id": str(uuid4()),
            "as_of": _iso(as_of),
            "scope": {
                "project_id": str(project_id),
                "run_id": run_id,
                "asset_id": asset_id,
                "status": list(statuses),
                "stage": list(stages),
                "ids": list(job_ids),
                "worker_id": worker_id,
            },
            "total": total,
            "by_stage": by_stage,
            "work_groups": groups,
            "health": {
                "expired_leases": int((raw.get("total") or {}).get("expired_leases") or 0),
                "stale_active_jobs": int((raw.get("total") or {}).get("stale_active_jobs") or 0),
                "pending_controls": int((raw.get("total") or {}).get("pending_controls") or 0),
                "progress_coverage_percent": (100.0 * active_known_count / active_count) if active_count else None,
                "oldest_queued_age_seconds": _float((raw.get("total") or {}).get("oldest_queued_age_seconds")),
            },
            "jobs": {
                "items": jobs,
                "limit": int(job_limit),
                "matching_count": matching_count,
                "truncated": matching_count > len(jobs),
            },
            "recent_failures": failures,
        }

    @staticmethod
    def _mixed_progress(progress: Mapping[str, Any], *, reason: str = "mixed_units") -> dict[str, Any]:
        return {
            **progress,
            "known_total_units": None,
            "completed_units": None,
            "failed_units": None,
            "skipped_units": None,
            "percent": None,
            "reason": reason,
        }

    def _work_group(self, row: Mapping[str, Any]) -> dict[str, Any]:
        progress = _progress_totals(row)
        rate = _float(row.get("units_per_second"))
        active = int(row.get("active") or 0)
        unknown_active = int(row.get("unknown_active_jobs") or 0)
        paused = int(row.get("paused") or 0)
        finalizing = int(row.get("finalizing_jobs") or 0)
        blocked = int(row.get("blocked_jobs") or 0)
        expired = int(row.get("expired_lease_jobs") or 0)
        stale = int(row.get("stale_progress_jobs") or 0)
        if active <= 0:
            rate_reason = "no_active_jobs"
        elif paused > 0:
            rate_reason = "paused"
        elif finalizing > 0:
            rate_reason = "finalizing"
        elif blocked > 0:
            rate_reason = "blocked"
        elif expired > 0:
            rate_reason = "expired_lease"
        elif stale > 0:
            rate_reason = "stale_progress"
        elif rate is None or rate <= 0:
            rate_reason = "insufficient_progress_samples"
        else:
            rate_reason = None
        remaining = _float(row.get("remaining_units"))
        if rate_reason is not None:
            eta_reason = rate_reason
        elif unknown_active > 0:
            eta_reason = "unknown_total"
        elif remaining is None:
            eta_reason = "unknown_total"
        else:
            eta_reason = None
        return {
            "stage": str(row.get("stage")),
            "unit": str(row.get("unit") or "unknown"),
            "job_count": int(row.get("job_count") or 0),
            "active": active,
            "historical": int(row.get("historical") or 0),
            "known_total_jobs": int(row.get("known_total_jobs") or 0),
            "unknown_total_jobs": int(row.get("unknown_total_jobs") or 0),
            "progress": progress,
            "active_progress": _progress_totals({
                "known_total_jobs": row.get("active_known_total_jobs"),
                "unknown_total_jobs": row.get("unknown_active_jobs"),
                "known_total_units": row.get("active_known_total_units"),
                "completed_units": row.get("active_completed_units"),
                "failed_units": row.get("active_failed_units"),
                "skipped_units": row.get("active_skipped_units"),
            }),
            "historical_progress": _progress_totals({
                "known_total_jobs": row.get("historical_known_total_jobs"),
                "unknown_total_jobs": row.get("historical_unknown_total_jobs"),
                "known_total_units": row.get("historical_known_total_units"),
                "completed_units": row.get("historical_completed_units"),
                "failed_units": row.get("historical_failed_units"),
                "skipped_units": row.get("historical_skipped_units"),
            }),
            "paused_jobs": paused,
            "finalizing_jobs": finalizing,
            "blocked_jobs": blocked,
            "rate": {
                "units_per_second": rate if rate_reason is None else None,
                "method": "sum_active_job_rates" if rate_reason is None else None,
                "observed_at": _iso(row.get("rate_observed_at")),
                "available": rate_reason is None,
                "reason": rate_reason,
            },
            "eta": {
                "seconds": remaining / rate if eta_reason is None and rate else None,
                "available": eta_reason is None,
                "reason": eta_reason,
            },
        }
