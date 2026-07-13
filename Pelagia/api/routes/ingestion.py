from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...domain import AssetKind, PipelineStage, PlannedRun, RawAssetManifest, RunManifest, normalize_collections
from ...processing.ingest_analysis import analyze_ingest_path

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field, field_validator
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if APIRouter is not None:
    from ..auth import require_project_write
    from ...services.project_settings import resolve_project_storage_settings
    from ...services.job_commands import ExtractFramesCommand
    from ...processing.codec_registry import normalize_image_encoding
    from ._common import as_response, get_repository

    class QueueVideoRequest(BaseModel):
        source_path: str
        n_tile: int | None = None
        image_encoding: str | None = None
        image_quality: int | None = None
        adaptive_background_subtraction: bool | None = None
        adaptive_background_period: int | None = None
        apply_mask: bool | None = None
        mask_path: str | None = None
        enqueue_segment: bool = False
        roi_padding: int | None = None
        roi_encoding: str | None = None
        collections: str | list[str] | None = None
        run_id: str | None = None
        asset_id: str | None = None
        run_key: str | None = None
        instrument: str = "api"
        compute_checksum: bool = False
        metadata: dict[str, Any] = Field(default_factory=dict)

    class AnalyzeIngestionRequest(BaseModel):
        source_path: str
        kind: str = "auto"
        recursive: bool = False
        compute_checksum: bool = False
        collections: str | list[str] | None = None
        n_tile: int | None = None
        image_encoding: str | None = None
        image_quality: int | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class IngestionAssetRequest(BaseModel):
        asset_id: str | None = None
        filename: str | None = None
        path: str
        kind: str
        size_bytes: int | None = None
        checksum: str | None = None
        checksum_status: str | None = None
        collections: str | list[str] | None = None
        media_count: int | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)
        n_tile: int | None = None
        image_encoding: str | None = None
        image_quality: int | None = None
        recursive: bool | None = None
        adaptive_background_subtraction: bool | None = None
        adaptive_background_period: int | None = None
        apply_mask: bool | None = None
        mask_path: str | None = None
        enqueue_segment: bool | None = None
        roi_padding: int | None = None
        roi_encoding: str | None = None

        @field_validator("kind")
        @classmethod
        def _validate_kind(cls, value: str) -> str:
            normalized = str(value).lower()
            if normalized not in {AssetKind.VIDEO.value, AssetKind.IMAGE_SEQUENCE.value}:
                raise ValueError("kind must be one of: video, image_sequence.")
            return normalized

    class QueueAssetsRequest(BaseModel):
        assets: list[IngestionAssetRequest]
        run_id: str | None = None
        run_key: str | None = None
        instrument: str = "api"
        source_path: str | None = None
        source_type: str | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)
        n_tile: int | None = None
        image_encoding: str | None = None
        image_quality: int | None = None
        adaptive_background_subtraction: bool | None = None
        adaptive_background_period: int | None = None
        apply_mask: bool | None = None
        mask_path: str | None = None
        enqueue_segment: bool = False
        roi_padding: int | None = None
        roi_encoding: str | None = None

    router = APIRouter(prefix="/ingestion", tags=["ingestion"])

    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _configured_import_roots(request: Request) -> list[Path]:
        config = request.app.state.context.config
        browser = config.file_browser
        roots = [browser.root_path_import_dir, *browser.allowed_root_paths]
        resolved: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            path = Path(root).expanduser().resolve()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)
        return resolved

    def _resolve_allowed_import_path(request: Request, source_path: str) -> Path:
        resolved = Path(source_path).expanduser().resolve()
        roots = _configured_import_roots(request)
        if not roots:
            return resolved
        for root in roots:
            if resolved == root or _is_relative_to(resolved, root):
                return resolved
        raise HTTPException(
            status_code=403,
            detail="Source path is outside the configured import roots.",
        )

    def _checksum_for_asset(path: Path, *, compute_checksum: bool, provided_checksum: str | None = None) -> tuple[str, str]:
        if provided_checksum:
            return provided_checksum, "provided"
        stat = path.stat()
        if compute_checksum and path.is_file():
            return f"sha256:{_sha256_file(path)}", "computed"
        return f"uncomputed:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}", "deferred"

    def _normalize_image_encoding(value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return normalize_image_encoding(value)
        except ValueError:
            raise HTTPException(status_code=422, detail="image_encoding must be one of: zstd, jpg, jxl, jxs, png, raw.")

    def _normalize_image_quality(value: int | None, default: int) -> int:
        quality = int(default if value is None else value)
        if quality < 0 or quality > 100:
            raise HTTPException(status_code=422, detail="image_quality must be between 0 and 100.")
        return quality

    def _resolved_ingest_defaults(
        request: Request,
        *,
        project_id: str | None = None,
        n_tile: int | None = None,
        image_encoding: str | None = None,
        image_quality: int | None = None,
    ) -> dict[str, Any]:
        defaults = request.app.state.context.config.processing
        ingest_defaults = defaults.video_ingest
        preprocessing_defaults = defaults.preprocessing
        roi_recording_defaults = defaults.roi_recording
        storage_settings = resolve_project_storage_settings(
            request.app.state.context,
            project_id,
            frame_encoding=image_encoding,
            frame_quality=image_quality,
        )
        return {
            "n_tile": ingest_defaults.n_tile if n_tile is None else n_tile,
            "image_encoding": storage_settings.frame_encoding,
            "image_quality": storage_settings.frame_quality,
            "adaptive_background_subtraction": preprocessing_defaults.adaptive_background_subtraction,
            "adaptive_background_period": preprocessing_defaults.adaptive_background_period,
            "apply_mask": preprocessing_defaults.apply_mask,
            "mask_path": preprocessing_defaults.mask_path,
            "roi_padding": roi_recording_defaults.padding,
            "roi_encoding": storage_settings.roi_encoding,
        }

    def _job_payload_for_asset(
        request: Request,
        asset: IngestionAssetRequest,
        *,
        global_body: QueueAssetsRequest | QueueVideoRequest,
        project_id: str | None,
        collections: list[str],
        checksum_status: str,
    ) -> dict[str, Any]:
        defaults = _resolved_ingest_defaults(
            request,
            project_id=project_id,
            n_tile=getattr(global_body, "n_tile", None),
            image_encoding=getattr(global_body, "image_encoding", None),
            image_quality=getattr(global_body, "image_quality", None),
        )
        n_tile = defaults["n_tile"] if asset.n_tile is None else asset.n_tile
        image_encoding = _normalize_image_encoding(asset.image_encoding) or defaults["image_encoding"]
        image_quality = _normalize_image_quality(asset.image_quality, defaults["image_quality"])
        adaptive_background_period = (
            asset.adaptive_background_period
            if asset.adaptive_background_period is not None
            else (
                getattr(global_body, "adaptive_background_period", None)
                if getattr(global_body, "adaptive_background_period", None) is not None
                else defaults["adaptive_background_period"]
            )
        )
        if n_tile < 1:
            raise HTTPException(status_code=422, detail="n_tile must be >= 1.")
        if adaptive_background_period < 1:
            raise HTTPException(status_code=422, detail="adaptive_background_period must be >= 1.")

        def resolve_option(asset_value: Any, body_name: str, default_name: str) -> Any:
            if asset_value is not None:
                return asset_value
            body_value = getattr(global_body, body_name, None)
            if body_value is not None:
                return body_value
            return defaults[default_name]

        return ExtractFramesCommand.from_payload({
            "source_path": str(Path(asset.path).expanduser().resolve()),
            "kind": asset.kind,
            "n_tile": n_tile,
            "recursive": bool(asset.recursive) if asset.recursive is not None else False,
            "adaptive_background_subtraction": resolve_option(
                asset.adaptive_background_subtraction,
                "adaptive_background_subtraction",
                "adaptive_background_subtraction",
            ),
            "adaptive_background_period": adaptive_background_period,
            "apply_mask": resolve_option(asset.apply_mask, "apply_mask", "apply_mask"),
            "mask_path": resolve_option(asset.mask_path, "mask_path", "mask_path"),
            "enqueue_segment": (
                getattr(global_body, "enqueue_segment", False)
                if asset.enqueue_segment is None
                else asset.enqueue_segment
            ),
            "padding": resolve_option(asset.roi_padding, "roi_padding", "roi_padding"),
            "roi_encoding": resolve_option(asset.roi_encoding, "roi_encoding", "roi_encoding"),
            "collections": collections,
            "checksum_status": checksum_status,
            "metadata": {
                "kvstore_encoding": image_encoding,
                "array_encoding": image_encoding,
                "kvstore_quality": image_quality,
                "array_quality": image_quality,
            },
        }).to_payload()

    @router.post("/analyze")
    def analyze_ingestion_source(request: Request, body: AnalyzeIngestionRequest) -> dict:
        auth = require_project_write(request)
        source_path = _resolve_allowed_import_path(request, body.source_path)
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source path {str(source_path)!r} was not found.")
        try:
            assets = analyze_ingest_path(
                source_path,
                kind=body.kind,
                recursive=body.recursive,
                collections=body.collections,
                compute_checksum=body.compute_checksum,
                metadata=body.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Source path {str(exc)!r} was not found.") from exc
        defaults = _resolved_ingest_defaults(
            request,
            project_id=auth.project_id,
            n_tile=body.n_tile,
            image_encoding=body.image_encoding,
            image_quality=body.image_quality,
        )
        asset_payloads = [asset.as_dict() for asset in assets]
        return as_response(
            {
                "source_path": str(source_path),
                "kind": body.kind,
                "recursive": body.recursive,
                "asset_count": len(asset_payloads),
                "assets": asset_payloads,
                "defaults": defaults,
                "suggested_ingestion_request": {
                    "source_path": str(source_path),
                    "assets": asset_payloads,
                    "n_tile": defaults["n_tile"],
                    "image_encoding": defaults["image_encoding"],
                    "image_quality": defaults["image_quality"],
                    "adaptive_background_subtraction": defaults["adaptive_background_subtraction"],
                    "adaptive_background_period": defaults["adaptive_background_period"],
                    "apply_mask": defaults["apply_mask"],
                    "mask_path": defaults["mask_path"],
                    "roi_padding": defaults["roi_padding"],
                    "roi_encoding": defaults["roi_encoding"],
                },
            }
        )

    @router.post("/assets")
    def queue_asset_ingestion(request: Request, body: QueueAssetsRequest) -> dict:
        if not body.assets:
            raise HTTPException(status_code=422, detail="Provide at least one asset.")
        repository = get_repository(request)
        auth = require_project_write(request)
        run_id = body.run_id or str(uuid.uuid4())
        first_path = Path(body.assets[0].path).expanduser().resolve()
        run_key = body.run_key or f"assets:{first_path.stem}:{uuid.uuid4().hex[:12]}"
        run_source_path = body.source_path or str(first_path.parent if len(body.assets) == 1 else first_path)
        run_source_type = body.source_type or (
            body.assets[0].kind if len({asset.kind for asset in body.assets}) == 1 else "mixed"
        )

        manifests: list[RawAssetManifest] = []
        resolved_assets: list[tuple[IngestionAssetRequest, str, list[str], str]] = []
        for asset in body.assets:
            path = _resolve_allowed_import_path(request, asset.path)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Source asset path {str(path)!r} was not found.")
            if asset.kind == AssetKind.VIDEO.value and not path.is_file():
                raise HTTPException(status_code=422, detail=f"Video asset path must be a file: {str(path)!r}.")
            if asset.kind == AssetKind.IMAGE_SEQUENCE.value and not path.is_dir():
                raise HTTPException(status_code=422, detail=f"Image sequence asset path must be a folder: {str(path)!r}.")
            stat_size = path.stat().st_size if path.is_file() else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            checksum, checksum_status = _checksum_for_asset(
                path,
                compute_checksum=False,
                provided_checksum=asset.checksum,
            )
            if asset.checksum_status:
                checksum_status = asset.checksum_status
            collections = normalize_collections(asset.collections)
            metadata = dict(asset.metadata or {})
            metadata["project_id"] = auth.project_id
            metadata.setdefault("api_endpoint", "POST /ingestion/assets")
            metadata.setdefault("collections", collections)
            metadata.setdefault("checksum_status", checksum_status)
            manifest = RawAssetManifest(
                asset_id=asset.asset_id or str(uuid.uuid4()),
                filename=asset.filename or path.name,
                path=str(path),
                kind=AssetKind(asset.kind),
                size_bytes=asset.size_bytes if asset.size_bytes is not None else stat_size,
                checksum=checksum,
                collections=collections,
                media_count=asset.media_count if asset.media_count is not None else int(metadata.get("source_frame_count") or 1),
                metadata=metadata,
            )
            manifests.append(manifest)
            resolved_assets.append((asset, manifest.asset_id, collections, checksum_status))

        planned_run = PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=run_key,
                instrument=body.instrument,
                source_path=run_source_path,
                source_type=run_source_type,
                created_at=datetime.now(timezone.utc),
                metadata={
                    **dict(body.metadata or {}),
                    "project_id": auth.project_id,
                    "api_endpoint": "POST /ingestion/assets",
                    "asset_count": len(manifests),
                },
                assets=manifests,
            )
        )
        registration = repository.register_planned_run(planned_run, project_id=auth.project_id)
        jobs = []
        for asset, asset_id, collections, checksum_status in resolved_assets:
            payload = _job_payload_for_asset(
                request,
                asset,
                global_body=body,
                project_id=auth.project_id,
                collections=collections,
                checksum_status=checksum_status,
            )
            job = repository.create_job(
                PipelineStage.EXTRACT_FRAMES,
                project_id=auth.project_id,
                run_id=run_id,
                asset_id=asset_id,
                payload=payload,
                summary=f"extract_frames queued for {Path(asset.path).name}",
            )
            jobs.append(job)

        return as_response(
            {
                "run_id": run_id,
                "run_key": run_key,
                "asset_count": len(manifests),
                "assets": [asdict(manifest) for manifest in manifests],
                "registration": registration,
                "jobs": jobs,
            }
        )

    @router.post("/videos")
    def queue_video_ingestion(request: Request, body: QueueVideoRequest) -> dict:
        repository = get_repository(request)
        auth = require_project_write(request)
        defaults = _resolved_ingest_defaults(
            request,
            project_id=auth.project_id,
            n_tile=body.n_tile,
            image_encoding=body.image_encoding,
            image_quality=body.image_quality,
        )
        processing_defaults = request.app.state.context.config.processing
        preprocessing_defaults = processing_defaults.preprocessing
        roi_recording_defaults = processing_defaults.roi_recording
        n_tile = defaults["n_tile"]
        image_encoding = defaults["image_encoding"]
        image_quality = defaults["image_quality"]
        adaptive_background_subtraction = (
            preprocessing_defaults.adaptive_background_subtraction
            if body.adaptive_background_subtraction is None
            else body.adaptive_background_subtraction
        )
        adaptive_background_period = (
            preprocessing_defaults.adaptive_background_period
            if body.adaptive_background_period is None
            else body.adaptive_background_period
        )
        apply_mask = preprocessing_defaults.apply_mask if body.apply_mask is None else body.apply_mask
        mask_path = preprocessing_defaults.mask_path if body.mask_path is None else body.mask_path
        roi_padding = (
            roi_recording_defaults.padding
            if body.roi_padding is None
            else body.roi_padding
        )
        roi_encoding = defaults["roi_encoding"] if body.roi_encoding is None else body.roi_encoding
        if n_tile < 1:
            raise HTTPException(status_code=422, detail="n_tile must be >= 1.")
        if adaptive_background_period < 1:
            raise HTTPException(status_code=422, detail="adaptive_background_period must be >= 1.")

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
        run_metadata = {
            "project_id": auth.project_id,
            "api_endpoint": "POST /ingestion/videos",
            "asset_count": 1,
        }
        stat = source_path.stat()
        if body.compute_checksum:
            checksum = f"sha256:{_sha256_file(source_path)}"
            checksum_status = "computed"
        else:
            checksum = f"uncomputed:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
            checksum_status = "deferred"
        asset_metadata = dict(body.metadata)
        asset_metadata["project_id"] = auth.project_id
        asset_metadata.setdefault("api_endpoint", "POST /ingestion/videos")
        asset_metadata.setdefault("collections", collections)
        asset_metadata.setdefault("checksum_status", checksum_status)

        planned_run = PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=run_key,
                instrument=body.instrument,
                source_path=str(source_path),
                source_type=AssetKind.VIDEO.value,
                created_at=datetime.now(timezone.utc),
                metadata=run_metadata,
                assets=[
                    RawAssetManifest(
                        asset_id=asset_id,
                        filename=source_path.name,
                        path=str(source_path),
                        kind=AssetKind.VIDEO,
                        size_bytes=stat.st_size,
                        checksum=checksum,
                        collections=collections,
                        metadata={
                            **asset_metadata,
                            "collections": collections,
                            "checksum_status": checksum_status,
                        },
                    )
                ],
            )
        )
        registration = repository.register_planned_run(planned_run, project_id=auth.project_id)
        job = repository.create_job(
            PipelineStage.EXTRACT_FRAMES,
            project_id=auth.project_id,
            run_id=run_id,
            asset_id=asset_id,
            payload=ExtractFramesCommand.from_payload({
                "source_path": str(source_path),
                "n_tile": n_tile,
                "adaptive_background_subtraction": adaptive_background_subtraction,
                "adaptive_background_period": adaptive_background_period,
                "apply_mask": apply_mask,
                "mask_path": mask_path,
                "enqueue_segment": body.enqueue_segment,
                "metadata": {
                    "kvstore_encoding": image_encoding,
                    "array_encoding": image_encoding,
                    "kvstore_quality": image_quality,
                    "array_quality": image_quality,
                },
                "padding": roi_padding,
                "roi_encoding": roi_encoding,
                "collections": collections,
                "checksum_status": checksum_status,
            }).to_payload(),
            summary=f"extract_frames queued for {source_path.name}",
        )
        return as_response(
            {
                "run_id": run_id,
                "asset_id": asset_id,
                "run_key": run_key,
                "collections": collections,
                "checksum_status": checksum_status,
                "registration": registration,
                "job": job,
            }
        )
else:
    router = None
