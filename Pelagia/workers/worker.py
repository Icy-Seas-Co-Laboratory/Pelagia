from __future__ import annotations

import os
import sys
import socket
import time
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Iterator

from ..domain import PipelineStage
from ..observability import get_core_logger
from ..services.context import AppContext
from .common import mark_job_frame_stage_failed
from .progress import JobCancellationRequested, JobLeaseLost, JobPauseRequested
from .registry import HandlerRegistry
from .runtime import worker_runtime_profile


@dataclass(slots=True)
class Worker:
    """Simple single-process worker loop skeleton."""

    context: AppContext
    handlers: HandlerRegistry
    worker_id: str = field(default_factory=lambda: f"worker-{socket.gethostname()}")

    def _capabilities(self, stages: list[PipelineStage] | None = None) -> list[str]:
        return [stage.value for stage in stages or []]

    def _runtime_metadata(self, stages: list[PipelineStage] | None) -> dict[str, str]:
        return {
            "hostname": socket.gethostname(),
            "runtime_profile": worker_runtime_profile(stages),
            "python_executable": sys.executable,
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        }

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
            metadata=self._runtime_metadata(stages),
            pid=os.getpid(),
            shutdown_requested=shutdown_requested,
        )

    def shutdown_requested(self) -> bool:
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")
        session = self.context.repository.get_worker_session(self.worker_id)
        return bool(session and session.get("shutdown_requested"))

    @staticmethod
    def _call_claim_aware(method, *args, worker_id: str, lease_token: str | None, **kwargs):
        """Call fenced repository methods without breaking lightweight repositories.

        Production repository methods accept the current claim identity.  Small
        test and extension repositories from before lease fencing may not yet
        declare those keyword arguments, so only omit them when their Python
        signature proves they cannot accept arbitrary keyword arguments.
        """
        if lease_token:
            try:
                parameters = inspect.signature(method).parameters.values()
                accepts_keywords = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
                names = {parameter.name for parameter in parameters}
            except (TypeError, ValueError):
                accepts_keywords, names = True, set()
            if accepts_keywords or {"worker_id", "lease_token"} <= names:
                kwargs.update(worker_id=worker_id, lease_token=lease_token)
        return method(*args, **kwargs)

    @staticmethod
    def _call_heartbeat(method, worker_id: str, job_id: str, lease_token: str | None):
        """Renew a lease without passing ``worker_id`` twice.

        Unlike other fenced repository methods, ``heartbeat`` takes the worker
        ID as its first positional argument.  It must therefore receive only
        the optional lease token as a keyword.
        """
        if lease_token:
            try:
                parameters = inspect.signature(method).parameters.values()
                accepts_keywords = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
                names = {parameter.name for parameter in parameters}
            except (TypeError, ValueError):
                accepts_keywords, names = True, set()
            if accepts_keywords or "lease_token" in names:
                return method(worker_id, job_id, lease_token=lease_token)
        return method(worker_id, job_id)

    @contextmanager
    def _maintain_job_lease(self, job_id: str, lease_token: str | None) -> Iterator[None]:
        """Renew a claimed job lease while its handler is running.

        Handlers may block in external services for longer than the queue lease.
        Lease ownership therefore belongs to the worker runtime, independently of
        whether a particular handler is currently able to report useful progress.
        """

        repository = self.context.repository
        if repository is None:
            yield
            return
        interval = max(1.0, float(self.context.config.queue.heartbeat_interval_seconds))
        stopped = Event()

        def renew() -> None:
            try:
                self._call_heartbeat(
                    repository.heartbeat,
                    self.worker_id,
                    job_id,
                    lease_token=lease_token,
                )
            except Exception:
                get_core_logger("worker").exception(
                    "Worker %s could not renew the lease for job %s",
                    self.worker_id,
                    job_id,
                )

        def heartbeat() -> None:
            while not stopped.wait(interval):
                renew()

        # Renew immediately, then continue independently while the handler runs.
        renew()
        thread = Thread(
            target=heartbeat,
            name=f"pelagia-heartbeat-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=min(interval, 2.0))

    def _acknowledge_requested_pause(self, job_id: str, lease_token: str | None) -> bool:
        """Pause before handler entry when an operator raced with the claim.

        Stage handlers remain responsible for their own batch-level checkpoints;
        this runtime check prevents a newly leased job from starting expensive
        work after a pause request was already persisted.
        """
        repository = self.context.repository
        if repository is None or not lease_token:
            return False
        control = repository.get_job_control(
            job_id,
            worker_id=self.worker_id,
            lease_token=lease_token,
        )
        if control is None:
            return True
        if not str(control.get("control_reason") or "").startswith("pause_requested:"):
            return False
        repository.finalize_paused_job(
            job_id,
            worker_id=self.worker_id,
            lease_token=lease_token,
        )
        get_core_logger("worker").info("Worker %s acknowledged pause for job %s", self.worker_id, job_id)
        return True

    def _run_post_completion_actions(self, job_id: str) -> None:
        """Best-effort accelerators for work already made durable on completion.

        These actions must not turn an already-succeeded parent into a failure.
        Both the dispatch outbox and series director have independent scheduled
        reconciliation paths, so an exception here is safe to retry later.
        """
        repository = self.context.repository
        if repository is None:
            return
        materialize = getattr(repository, "materialize_pending_dispatches", None)
        if callable(materialize):
            try:
                materialize(parent_job_id=job_id)
            except Exception:
                get_core_logger("worker").exception(
                    "Could not materialize successor dispatches for completed job %s",
                    job_id,
                )
        if hasattr(repository, "advance_processing_series_for_job"):
            try:
                from ..services.processing_queue import ProcessingQueueService

                ProcessingQueueService(self.context).advance_series_for_job(job_id)
            except Exception:
                get_core_logger("worker").exception(
                    "Could not advance processing series for completed job %s",
                    job_id,
                )

    def run_once(self, stages: list[PipelineStage] | None = None) -> int:
        """Claim and process currently available jobs once."""
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")
        worker_runtime_profile(stages)

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
            lease_token = None if job.get("lease_token") is None else str(job["lease_token"])
            # Keep claim identity out of the persisted payload while allowing
            # progress reporters and future checkpoint-aware handlers to fence
            # their updates against a re-claimed lease.
            job["_worker_id"] = self.worker_id
            job["_lease_token"] = lease_token
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
                if self._acknowledge_requested_pause(job_id, lease_token):
                    continue
                with self._maintain_job_lease(job_id, lease_token):
                    result = self.handlers.handle(job, job_context)
                # Close the race between the handler's final checkpoint and
                # terminal publication. This also observes immediate cancel,
                # which invalidates the claim and therefore returns no control.
                if self._acknowledge_requested_pause(job_id, lease_token):
                    continue
                completed = self._call_claim_aware(
                    self.context.repository.complete_job,
                    job["id"],
                    result=result,
                    worker_id=self.worker_id,
                    lease_token=lease_token,
                )
                if completed is None and lease_token:
                    get_core_logger("worker").warning(
                        "Worker %s lost lease before completing job %s; output was not published.",
                        self.worker_id,
                        job_id,
                    )
                    continue
                self._run_post_completion_actions(job_id)
                duration_ms = (time.perf_counter() - started) * 1000
                timings = (result or {}).get("timings")
                if timings:
                    get_core_logger("worker").info(
                        "Worker %s completed job %s stage=%s timings=%s",
                        self.worker_id,
                        job_id,
                        stage,
                        timings,
                    )
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
                            "timings": timings,
                        },
                    )
            except (JobPauseRequested, JobCancellationRequested, JobLeaseLost) as exc:
                get_core_logger("worker").info(
                    "Worker %s stopped job %s at a cooperative checkpoint: %s",
                    self.worker_id,
                    job_id,
                    exc,
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - started) * 1000
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
                failed = self._call_claim_aware(
                    self.context.repository.record_failure,
                    job["id"],
                    str(exc),
                    retryable=True,
                    worker_id=self.worker_id,
                    lease_token=lease_token,
                )
                # Do not let a stale worker overwrite per-frame status after a
                # newer lease has started work on the same job.
                if failed is not None or not lease_token:
                    mark_job_frame_stage_failed(job, job_context)
                if failed is None and lease_token:
                    get_core_logger("worker").warning(
                        "Worker %s lost lease before recording failure for job %s.",
                        self.worker_id,
                        job_id,
                    )
                    continue
                if hasattr(self.context.repository, "advance_processing_series_for_job"):
                    from ..services.processing_queue import ProcessingQueueService
                    ProcessingQueueService(self.context).advance_series_for_job(job_id)
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
        worker_runtime_profile(stages)

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
