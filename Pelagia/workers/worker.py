from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from threading import Event

from ..domain import PipelineStage
from ..observability import get_core_logger
from ..services.context import AppContext
from .handlers import HandlerRegistry, mark_job_frame_stage_failed


@dataclass(slots=True)
class Worker:
    """Simple single-process worker loop skeleton."""

    context: AppContext
    handlers: HandlerRegistry
    worker_id: str = field(default_factory=lambda: f"worker-{socket.gethostname()}")

    def _capabilities(self, stages: list[PipelineStage] | None = None) -> list[str]:
        return [stage.value for stage in stages or []]

    def _touch(
        self,
        status: str,
        *,
        stages: list[PipelineStage] | None = None,
        leased_job_id: str | None = None,
        shutdown_requested: bool | None = None,
    ) -> dict | None:
        if self.context.repository is None:
            return None
        return self.context.repository.touch_worker(
            self.worker_id,
            status=status,
            leased_job_id=leased_job_id,
            capabilities=self._capabilities(stages),
            metadata={"hostname": socket.gethostname()},
            pid=os.getpid(),
            shutdown_requested=shutdown_requested,
        )

    def shutdown_requested(self) -> bool:
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")
        session = self.context.repository.get_worker_session(self.worker_id)
        return bool(session and session.get("shutdown_requested"))

    def run_once(self, stages: list[PipelineStage] | None = None) -> int:
        """Claim and process currently available jobs once."""
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")

        if self.shutdown_requested():
            self._touch("stopped", stages=stages)
            return 0

        self._touch("idle", stages=stages)
        jobs = self.context.repository.claim_jobs(self.worker_id, stages=stages)
        for job in jobs:
            project_id = None if job.get("project_id") is None else str(job.get("project_id"))
            job_context = self.context.for_project(project_id)
            self._touch("working", stages=stages, leased_job_id=str(job["id"]))
            started = time.perf_counter()
            job_id = str(job["id"])
            stage = job.get("stage")
            run_id = None if job.get("run_id") is None else str(job.get("run_id"))
            asset_id = None if job.get("asset_id") is None else str(job.get("asset_id"))
            if self.context.logger is not None:
                self.context.logger.debug(
                    "job.handler_started",
                    "Worker started job handler",
                    job_id=job_id,
                    run_id=run_id,
                    asset_id=asset_id,
                    worker_id=self.worker_id,
                    payload={"stage": stage},
                )
            try:
                result = self.handlers.handle(job, job_context)
                self.context.repository.complete_job(job["id"], result=result)
                duration_ms = (time.perf_counter() - started) * 1000
                if self.context.logger is not None:
                    self.context.logger.info(
                        "job.handler_completed",
                        "Worker completed job handler",
                        job_id=job_id,
                        run_id=run_id,
                        asset_id=asset_id,
                        worker_id=self.worker_id,
                        duration_ms=duration_ms,
                        payload={
                            "stage": stage,
                            "result_keys": sorted((result or {}).keys()),
                        },
                    )
            except Exception as exc:
                duration_ms = (time.perf_counter() - started) * 1000
                mark_job_frame_stage_failed(job, job_context)
                get_core_logger("worker").exception(
                    "Worker %s failed job %s stage=%s",
                    self.worker_id,
                    job_id,
                    stage,
                )
                if self.context.logger is not None:
                    self.context.logger.error(
                        "job.handler_failed",
                        "Worker failed job handler",
                        job_id=job_id,
                        run_id=run_id,
                        asset_id=asset_id,
                        worker_id=self.worker_id,
                        duration_ms=duration_ms,
                        payload={
                            "stage": stage,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                self.context.repository.record_failure(job["id"], str(exc), retryable=True)
            finally:
                self._touch("idle", stages=stages)
        return len(jobs)

    def run_forever(
        self,
        stages: list[PipelineStage] | None = None,
        *,
        idle_sleep_seconds: float = 2.0,
        requeue_interval_seconds: float = 30.0,
        stop_event: Event | None = None,
    ) -> None:
        """Run this worker until signaled or externally requested to shut down."""
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")

        stop = stop_event or Event()
        last_requeue_at = 0.0
        self._touch("idle", stages=stages, shutdown_requested=False)
        try:
            while not stop.is_set():
                if self.shutdown_requested():
                    break

                now = time.monotonic()
                if now - last_requeue_at >= requeue_interval_seconds:
                    self.context.repository.requeue_expired_jobs()
                    last_requeue_at = now

                claimed = self.run_once(stages=stages)
                if claimed == 0:
                    stop.wait(idle_sleep_seconds)
        finally:
            self._touch("stopped", stages=stages, shutdown_requested=False)
