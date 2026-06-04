from __future__ import annotations

from typing import Any

from ..observability import get_core_logger


def processing_core_logger(name: str):
    return get_core_logger(f"processing.{name}")


def log_processing_event(
    context: Any,
    level: str,
    event_type: str,
    message: str,
    *,
    run_id: str | None = None,
    asset_id: str | None = None,
    job_id: str | None = None,
    worker_id: str | None = None,
    duration_ms: float | None = None,
    payload: dict[str, Any] | None = None,
    logger: str = "pelagia.processing",
    core_logger=None,
) -> None:
    """Write a structured processing event when an AppContext logger exists."""
    database_logger = None if context is None else getattr(context, "logger", None)
    if database_logger is None:
        return
    try:
        database_logger.log(
            event_type=event_type,
            message=message,
            level=level,
            run_id=run_id,
            asset_id=asset_id,
            job_id=job_id,
            worker_id=worker_id,
            duration_ms=duration_ms,
            payload=payload or {},
            logger=logger,
        )
    except Exception:
        fallback = core_logger or processing_core_logger("logging")
        fallback.exception("Failed to write processing database log event %s", event_type)
