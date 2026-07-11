from __future__ import annotations

from collections.abc import Iterable

from ..domain import PipelineStage


GPU_ML_STAGES = frozenset({PipelineStage.ROI_REFINEMENT})


def worker_runtime_profile(stages: Iterable[PipelineStage] | None) -> str:
    """Return the required runtime profile for a worker's stages."""
    if stages is None:
        raise ValueError("Workers must declare explicit stages so GPU/ML work stays isolated.")
    selected_stages = set(stages)
    gpu_ml_stages = selected_stages & GPU_ML_STAGES
    if gpu_ml_stages and selected_stages - GPU_ML_STAGES:
        raise ValueError(
            "GPU/ML stages must run in a dedicated worker. "
            f"Do not combine {', '.join(sorted(stage.value for stage in gpu_ml_stages))} "
            "with non-ML stages."
        )
    return "gpu-ml" if gpu_ml_stages else "cpu"
