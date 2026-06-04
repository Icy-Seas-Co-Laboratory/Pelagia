from __future__ import annotations

import logging
import os
import socket
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Any, Iterator

from .config import CoreConfig
from .utils.serialization import json_ready


CORE_LOGGER_NAME = "pelagia"
_CORE_LOGGING_CONFIGURED = False
_CORE_LOGGING_SIGNATURE: tuple[object, ...] | None = None


def configure_core_logging(config: CoreConfig, *, force: bool = False) -> logging.Logger:
    """Configure Pelagia's operational logger.

    Core logging must work when PostgreSQL is unavailable, so it writes to a
    rotating local file and optionally stderr.
    """
    global _CORE_LOGGING_CONFIGURED, _CORE_LOGGING_SIGNATURE

    logger = logging.getLogger(CORE_LOGGER_NAME)
    signature = (
        config.logging.log_path,
        config.logging.file_name,
        config.logging.level,
        config.logging.console,
        config.logging.max_bytes,
        config.logging.backup_count,
    )
    if _CORE_LOGGING_CONFIGURED and _CORE_LOGGING_SIGNATURE == signature and not force:
        return logger

    level = _coerce_log_level(config.logging.level)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(process)d] %(name)s - %(message)s"
    )

    try:
        config.logging.log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.logging.log_path / config.logging.file_name,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    except OSError as exc:
        fallback = logging.StreamHandler()
        fallback.setFormatter(formatter)
        fallback.setLevel(level)
        logger.addHandler(fallback)
        logger.error("Unable to initialize file logging: %s", exc)

    if config.logging.console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    _CORE_LOGGING_CONFIGURED = True
    _CORE_LOGGING_SIGNATURE = signature
    return logger


def get_core_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return logging.getLogger(CORE_LOGGER_NAME)
    return logging.getLogger(f"{CORE_LOGGER_NAME}.{name}")


def _coerce_log_level(value: str) -> int:
    return int(getattr(logging, str(value).upper(), logging.INFO))


class DatabaseLogger:
    """Structured PostgreSQL-backed log writer with file-log fallback."""

    def __init__(self, repository, *, logger_name: str = CORE_LOGGER_NAME):
        self.repository = repository
        self.core_logger = logging.getLogger(logger_name)

    def log(
        self,
        *,
        event_type: str,
        message: str | None = None,
        level: str = "info",
        run_id: str | None = None,
        asset_id: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        request_id: str | None = None,
        duration_ms: float | None = None,
        payload: dict[str, Any] | None = None,
        logger: str = CORE_LOGGER_NAME,
    ) -> dict[str, Any] | None:
        try:
            return self.repository.append_log(
                event_type=event_type,
                message=message,
                level=level,
                logger=logger,
                run_id=run_id,
                asset_id=asset_id,
                job_id=job_id,
                worker_id=worker_id,
                request_id=request_id,
                duration_ms=duration_ms,
                payload=_base_payload(payload),
            )
        except Exception:
            self.core_logger.exception(
                "Database log write failed for event_type=%s level=%s message=%r",
                event_type,
                level,
                message,
            )
            return None

    def debug(self, event_type: str, message: str | None = None, **kwargs) -> dict[str, Any] | None:
        return self.log(event_type=event_type, message=message, level="debug", **kwargs)

    def info(self, event_type: str, message: str | None = None, **kwargs) -> dict[str, Any] | None:
        return self.log(event_type=event_type, message=message, level="info", **kwargs)

    def warning(self, event_type: str, message: str | None = None, **kwargs) -> dict[str, Any] | None:
        return self.log(event_type=event_type, message=message, level="warning", **kwargs)

    def error(self, event_type: str, message: str | None = None, **kwargs) -> dict[str, Any] | None:
        return self.log(event_type=event_type, message=message, level="error", **kwargs)

    @contextmanager
    def timed(
        self,
        event_type: str,
        *,
        message: str | None = None,
        level: str = "info",
        payload: dict[str, Any] | None = None,
        **kwargs,
    ) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            error_payload = dict(payload or {})
            error_payload["error_type"] = type(exc).__name__
            error_payload["error_message"] = str(exc)
            self.log(
                event_type=f"{event_type}.failed",
                message=message or str(exc),
                level="error",
                duration_ms=duration_ms,
                payload=error_payload,
                **kwargs,
            )
            raise
        else:
            duration_ms = (time.perf_counter() - started) * 1000
            self.log(
                event_type=event_type,
                message=message,
                level=level,
                duration_ms=duration_ms,
                payload=payload,
                **kwargs,
            )


def _base_payload(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    base.update(payload or {})
    return json_ready(base)
