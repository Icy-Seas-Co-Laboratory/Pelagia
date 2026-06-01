from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...domain import AssetKind, PipelineStage, PlannedRun, RawAssetManifest, RunManifest, normalize_collections

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if APIRouter is not None:
    from ._common import as_response, get_repository

    class QueueVideoRequest(BaseModel):
        source_path: str
        n_tile: int = 1
        enqueue_segment: bool = False
        segmentation_padding: int = 0
        roi_encoding: str = "zstd"
        collections: str | list[str] | None = None
        run_id: str | None = None
        asset_id: str | None = None
        run_key: str | None = None
        instrument: str = "api"
        metadata: dict[str, Any] = Field(default_factory=dict)

    router = APIRouter(prefix="/ingestion", tags=["ingestion"])

    @router.post("/videos")
    def queue_video_ingestion(request: Request, body: QueueVideoRequest) -> dict:
        if body.n_tile < 1:
            raise HTTPException(status_code=422, detail="n_tile must be >= 1.")

        repository = get_repository(request)
        source_path = Path(body.source_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Source video {str(source_path)!r} was not found.",
            )

        run_id = body.run_id or str(uuid.uuid4())
        asset_id = body.asset_id or str(uuid.uuid4())
        run_key = body.run_key or f"video:{source_path.stem}:{uuid.uuid4().hex[:12]}"
        collections = normalize_collections(body.collections)
        metadata = dict(body.metadata)
        metadata.setdefault("api_endpoint", "POST /ingestion/videos")
        metadata.setdefault("collections", collections)

        planned_run = PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=run_key,
                instrument=body.instrument,
                source_path=str(source_path),
                source_type=AssetKind.VIDEO.value,
                created_at=datetime.now(timezone.utc),
                metadata=metadata,
                assets=[
                    RawAssetManifest(
                        asset_id=asset_id,
                        filename=source_path.name,
                        path=str(source_path),
                        kind=AssetKind.VIDEO,
                        size_bytes=source_path.stat().st_size,
                        checksum=_sha256_file(source_path),
                        collections=collections,
                        metadata={"api_endpoint": "POST /ingestion/videos", "collections": collections},
                    )
                ],
            )
        )
        registration = repository.register_planned_run(planned_run)
        job = repository.create_job(
            PipelineStage.EXTRACT_FRAMES,
            run_id=run_id,
            asset_id=asset_id,
            payload={
                "source_path": str(source_path),
                "n_tile": body.n_tile,
                "enqueue_segment": body.enqueue_segment,
                "segmentation_padding": body.segmentation_padding,
                "roi_encoding": body.roi_encoding,
                "collections": collections,
            },
            summary=f"extract_frames queued for {source_path.name}",
        )
        return as_response(
            {
                "run_id": run_id,
                "asset_id": asset_id,
                "run_key": run_key,
                "collections": collections,
                "registration": registration,
                "job": job,
            }
        )
else:
    router = None
