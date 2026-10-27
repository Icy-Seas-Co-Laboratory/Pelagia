"""Create a portable interchange dataset from server-side videos."""
from pathlib import Path
from typing import Any

from ...domain import PipelineStage
from ...services.context import AppContext
from ..progress import JobProgressReporter


def handle(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    from pelagia_interchange import ingest_video_directory

    payload = dict(job.get("payload") or {})
    source = Path(payload["source_path"])
    output = Path(payload["output_path"])
    if source.resolve() != source or output.resolve() != output:
        raise ValueError("Source or output path changed since submission.")
    if not source.is_dir() or output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directories, and source must exist.")
    # Reserve the destination exclusively. A second job or a retry must never
    # append to or overwrite a dataset created by another attempt.
    output.mkdir(exist_ok=False)
    reporter = JobProgressReporter(job, context, stage=PipelineStage.INTERCHANGE_CREATE.value,
                                   unit="step", total=None, emit_every=1)
    reporter.start("Preparing interchange dataset")
    count = 0
    def progress(message: str) -> None:
        nonlocal count
        count += 1
        reporter.update(count, message=message)
    result = ingest_video_directory(
        source, output, title=payload.get("title"), description=payload.get("description", ""),
        stream=payload.get("stream", "camera"), recursive=payload.get("recursive", False),
        shard_target_size=payload.get("shard_target_size", "10GB"),
        jpeg_quality=payload.get("jpeg_quality", 90), jpeg_subsampling=payload.get("jpeg_subsampling", "444"),
        grayscale=payload.get("grayscale", False), source_file_boundary=payload.get("source_file_boundary", False),
        generate_previews=payload.get("generate_previews", True), progress=progress,
    )
    reporter.finish(message="Interchange dataset created")
    return {"path": str(output), "frames": result.frames, "acquisition_segments": result.acquisition_segments,
            "previews": result.previews}
