"""Bounded ephemeral feature-space analysis worker stage."""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Iterator

from ...services.context import AppContext
from ...services.feature_space import FeatureSpaceService
from ..progress import JobProgressReporter

ANALYSIS_TIMEOUT_SECONDS = 30


@contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    """Interrupt a runaway analysis in the worker's main thread."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def expired(_signum, _frame):
        raise TimeoutError(f"Feature-space analysis exceeded its {seconds}-second limit.")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def handle(job: dict, context: AppContext) -> dict:
    payload = dict(job.get("payload") or {})
    reporter = JobProgressReporter(job, context, stage="feature_space_analysis", unit="analysis", total=1, emit_every=1)
    reporter.start("Computing UMAP and HDBSCAN")
    with _deadline(ANALYSIS_TIMEOUT_SECONDS):
        result = FeatureSpaceService(context, project_id=str(job["project_id"])).umap_rois(
            source_key=str(payload["source_key"]),
            min_cluster_size=int(payload.get("min_cluster_size", 5)),
            min_samples=payload.get("min_samples"),
            cluster_selection_epsilon=float(payload.get("cluster_selection_epsilon", 0.0)),
        )
    # Job results are retrieved outside the curation route, so retain the same
    # authenticated-image paths the synchronous response previously supplied.
    result["items"] = [
        {
            **item,
            "roi_url": f"/refined-detections/{item['id']}/roi?format=jpg&width=320",
            "thumbnail_url": f"/refined-detections/{item['id']}/roi?format=jpg&width=180",
        }
        for item in result.get("items", [])
    ]
    reporter.finish(message="Feature-space analysis complete")
    return {"operation": "feature_space_analysis", "cache_key": payload["cache_key"], "ephemeral": True, **result}
