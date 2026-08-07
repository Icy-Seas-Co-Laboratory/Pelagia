from __future__ import annotations

from collections.abc import Iterable

from ..domain import PipelineStage


def worker_runtime_profile(stages: Iterable[PipelineStage] | None) -> str:
    """Return the runtime profile; ML execution is hosted by Oracle Builder."""
    if stages is None:
        raise ValueError("Workers must declare explicit stages.")
    set(stages)
    return "cpu"
