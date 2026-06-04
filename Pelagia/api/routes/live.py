from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "is_dir": path.is_dir(),
        "size_bytes": None if path.is_dir() else stat.st_size,
        "modified_at": stat.st_mtime,
    }


if APIRouter is not None:
    from ...processing.segmentation import live_segment_wrapper
    from ._common import as_response, detection_summary, get_context, get_repository

    router = APIRouter(prefix="/live", tags=["live"])

    @router.get("/files")
    def list_server_files(
        directory: str,
        recursive: bool = False,
        include_hidden: bool = False,
        limit: int = 500,
    ) -> dict:
        if limit < 1:
            raise HTTPException(status_code=422, detail="limit must be >= 1.")

        root = Path(directory).expanduser().resolve()
        if not root.exists():
            raise HTTPException(status_code=404, detail=f"Directory {str(root)!r} was not found.")
        if not root.is_dir():
            raise HTTPException(status_code=422, detail=f"{str(root)!r} is not a directory.")

        iterator = root.rglob("*") if recursive else root.iterdir()
        entries = []
        for path in iterator:
            try:
                if not include_hidden and any(part.startswith(".") for part in path.relative_to(root).parts):
                    continue
                entries.append(_file_entry(path, root))
            except OSError:
                continue
            if len(entries) >= limit:
                break

        entries.sort(key=lambda item: (not item["is_dir"], item["relative_path"].lower()))
        return as_response(
            {
                "directory": str(root),
                "recursive": recursive,
                "include_hidden": include_hidden,
                "limit": limit,
                "count": len(entries),
                "entries": entries,
            }
        )

    @router.get("/segment")
    def segment_live_frame(
        request: Request,
        frame_id: str,
        threshold: int | float | None = None,
        min_perimeter: int | float | None = None,
        max_perimeter: int | float | None = None,
        padding: int | None = None,
    ) -> dict:
        context = get_context(request)
        repository = get_repository(request)
        frame_record = repository.get_frame_record(frame_id)
        if frame_record is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_id!r} was not found.")

        defaults = context.config.processing.segmentation
        try:
            detections = live_segment_wrapper(
                frame_id,
                threshold=threshold,
                min_perimeter=defaults.min_perimeter if min_perimeter is None else min_perimeter,
                max_perimeter=defaults.max_perimeter if max_perimeter is None else max_perimeter,
                padding=defaults.padding if padding is None else padding,
                context=context,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        detection_rows = [detection_summary(detection) for detection in detections]

        response = {
            "frame_id": frame_id,
            "run_id": getattr(frame_record, "run_id", None),
            "asset_id": getattr(frame_record, "asset_id", None),
            "saved": False,
            "detection_count": len(detections),
            "detections": detection_rows,
        }

        return as_response(response)
else:
    router = None
