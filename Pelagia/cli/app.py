from __future__ import annotations

import hashlib
import json
import signal
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import CoreConfig
from ..domain import AssetKind, JobStatus, PipelineStage, PlannedRun, RawAssetManifest, RunManifest, normalize_collections
from ..observability import configure_core_logging
from ..services.context import AppContext
from ..services.stores import StoreService
from ..utils.serialization import json_ready


try:
    import typer
except ImportError:  # pragma: no cover - argparse fallback covers no-typer envs
    typer = None


if typer is not None:
    app = typer.Typer(help="Pelagia command line tools.")

    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _path_size_bytes(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def _config_from_options(
        kvstore_root: Optional[Path],
        database_dsn: Optional[str],
        schema: Optional[str],
    ) -> CoreConfig:
        config = CoreConfig.load()
        if kvstore_root is not None:
            config.kvstore.root_path = kvstore_root
        if database_dsn is not None:
            config.database.dsn = database_dsn
        if schema is not None:
            config.database.schema_name = schema
        return config

    def _context_from_options(
        kvstore_root: Optional[Path],
        database_dsn: Optional[str],
        schema: Optional[str],
        *,
        initialize_schema: bool = False,
    ) -> AppContext:
        config = _config_from_options(kvstore_root, database_dsn, schema)
        configure_core_logging(config)
        context = AppContext.from_config(config)
        if context.kvstore is not None and not context.kvstore.initialized:
            context.kvstore.initialize(
                hash_algorithm=config.kvstore.hash_algorithm,
                prefix_length=config.kvstore.prefix_length,
                max_db_bytes=config.kvstore.max_db_bytes,
                max_rows=config.kvstore.max_rows,
            )
        if initialize_schema and context.repository is not None:
            context.repository.initialize_schema()
        return context

    def _repository_from_options(
        kvstore_root: Optional[Path],
        database_dsn: Optional[str],
        schema: Optional[str],
    ):
        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required for this command.")
        return context.repository

    def _echo_json(payload: object) -> None:
        typer.echo(json.dumps(json_ready(payload), indent=2, sort_keys=True))

    def _split_csv(value: Optional[str]) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def _segment_payload(
        frame_ids: Optional[str],
        start_frame: Optional[int],
        end_frame: Optional[int],
        limit: Optional[int],
        threshold: Optional[float],
        frame_payload_kind: Optional[str],
        apply_preprocessing: Optional[bool],
        flatfield_correction: Optional[bool],
        flatfield_q: Optional[float],
        flatfield_axis: Optional[int],
        apply_mask: Optional[bool],
        crop_enabled: Optional[bool],
        crop_x: Optional[int],
        crop_y: Optional[int],
        crop_w: Optional[int],
        crop_h: Optional[int],
        background_correction: Optional[bool],
        invert_intensity: Optional[bool],
        min_perimeter: Optional[float],
        max_perimeter: Optional[float],
        padding: Optional[int],
        roi_encoding: Optional[str],
        zstd_min_bytes: Optional[int],
        processing_defaults,
    ) -> dict:
        roi_filter_defaults = processing_defaults.roi_filter
        roi_recording_defaults = processing_defaults.roi_recording
        flatfield_defaults = processing_defaults.flatfield
        preprocessing_defaults = processing_defaults.preprocessing
        return {
            "frame_ids": _split_csv(frame_ids),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "limit": limit,
            "threshold": threshold,
            "frame_payload_kind": frame_payload_kind or "original",
            "apply_preprocessing": (
                (frame_payload_kind or "original") in {"original", "raw"}
                if apply_preprocessing is None
                else apply_preprocessing
            ),
            "flatfield_correction": (
                flatfield_defaults.flatfield_correction
                if flatfield_correction is None
                else flatfield_correction
            ),
            "flatfield_q": flatfield_defaults.flatfield_q if flatfield_q is None else flatfield_q,
            "flatfield_axis": flatfield_defaults.flatfield_axis if flatfield_axis is None else flatfield_axis,
            "apply_mask": preprocessing_defaults.apply_mask if apply_mask is None else apply_mask,
            "crop_enabled": (
                preprocessing_defaults.crop_enabled
                if crop_enabled is None
                else crop_enabled
            ),
            "crop_x": preprocessing_defaults.crop_x if crop_x is None else crop_x,
            "crop_y": preprocessing_defaults.crop_y if crop_y is None else crop_y,
            "crop_w": preprocessing_defaults.crop_w if crop_w is None else crop_w,
            "crop_h": preprocessing_defaults.crop_h if crop_h is None else crop_h,
            "background_correction": (
                preprocessing_defaults.background_correction
                if background_correction is None
                else background_correction
            ),
            "invert_intensity": (
                preprocessing_defaults.invert_intensity
                if invert_intensity is None
                else invert_intensity
            ),
            "min_perimeter": roi_filter_defaults.min_perimeter if min_perimeter is None else min_perimeter,
            "max_perimeter": roi_filter_defaults.max_perimeter if max_perimeter is None else max_perimeter,
            "padding": roi_recording_defaults.padding if padding is None else padding,
            "roi_encoding": roi_recording_defaults.roi_encoding if roi_encoding is None else roi_encoding,
            "zstd_min_bytes": roi_recording_defaults.zstd_min_bytes if zstd_min_bytes is None else zstd_min_bytes,
        }

    def _preprocess_payload(
        frame_ids: Optional[str],
        start_frame: Optional[int],
        end_frame: Optional[int],
        limit: Optional[int],
        flatfield_correction: Optional[bool],
        flatfield_q: Optional[float],
        flatfield_axis: Optional[int],
        apply_mask: Optional[bool],
        crop_enabled: Optional[bool],
        crop_x: Optional[int],
        crop_y: Optional[int],
        crop_w: Optional[int],
        crop_h: Optional[int],
        background_correction: Optional[bool],
        invert_intensity: Optional[bool],
        encoding: Optional[str],
        processing_defaults,
    ) -> dict:
        flatfield_defaults = processing_defaults.flatfield
        preprocessing_defaults = processing_defaults.preprocessing
        return {
            "frame_ids": _split_csv(frame_ids),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "limit": limit,
            "flatfield_correction": (
                flatfield_defaults.flatfield_correction
                if flatfield_correction is None
                else flatfield_correction
            ),
            "flatfield_q": flatfield_defaults.flatfield_q if flatfield_q is None else flatfield_q,
            "flatfield_axis": flatfield_defaults.flatfield_axis if flatfield_axis is None else flatfield_axis,
            "apply_mask": preprocessing_defaults.apply_mask if apply_mask is None else apply_mask,
            "crop_enabled": (
                preprocessing_defaults.crop_enabled
                if crop_enabled is None
                else crop_enabled
            ),
            "crop_x": preprocessing_defaults.crop_x if crop_x is None else crop_x,
            "crop_y": preprocessing_defaults.crop_y if crop_y is None else crop_y,
            "crop_w": preprocessing_defaults.crop_w if crop_w is None else crop_w,
            "crop_h": preprocessing_defaults.crop_h if crop_h is None else crop_h,
            "background_correction": (
                preprocessing_defaults.background_correction
                if background_correction is None
                else background_correction
            ),
            "invert_intensity": (
                preprocessing_defaults.invert_intensity
                if invert_intensity is None
                else invert_intensity
            ),
            "encoding": encoding,
        }

    def _resolve_segment_target(
        repository,
        run_id: Optional[str],
        asset_id: Optional[str],
        payload: dict,
    ) -> tuple[Optional[str], str]:
        resolved_run_id = run_id
        resolved_asset_id = asset_id

        if resolved_asset_id is None and payload["frame_ids"]:
            first_frame = repository.get_frame_record(payload["frame_ids"][0])
            if first_frame is None:
                raise RuntimeError(f"Frame {payload['frame_ids'][0]!r} was not found.")
            resolved_run_id = resolved_run_id or first_frame.run_id
            resolved_asset_id = first_frame.asset_id

        if resolved_asset_id is None:
            raise RuntimeError("Segmentation requires --asset-id or --frame-ids.")

        if resolved_run_id is None:
            asset = repository.get_asset(resolved_asset_id)
            if asset is None:
                raise RuntimeError(f"Asset {resolved_asset_id!r} was not found.")
            resolved_run_id = asset.get("run_id")

        return resolved_run_id, resolved_asset_id

    def _sorted_jobs(rows: list[dict], limit: int) -> list[dict]:
        def sort_key(row: dict) -> tuple[object, str]:
            fallback = datetime.min.replace(tzinfo=timezone.utc)
            return (row.get("created_at") or fallback, str(row.get("id") or ""))

        return sorted(rows, key=sort_key, reverse=True)[:limit]

    def _register_video(
        context: AppContext,
        input_path: Path,
        run_id: Optional[str],
        asset_id: Optional[str],
        run_key: Optional[str],
        instrument: str,
        collections: Optional[str],
        *,
        cli_command: str,
        compute_checksum: bool = True,
    ) -> tuple[str, str, str]:
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to register video assets.")

        resolved = input_path.expanduser().resolve()
        resolved_run_id = run_id or str(uuid.uuid4())
        resolved_asset_id = asset_id or str(uuid.uuid4())
        is_image_sequence = resolved.is_dir()
        asset_kind = AssetKind.IMAGE_SEQUENCE if is_image_sequence else AssetKind.VIDEO
        source_type = asset_kind.value
        run_key_prefix = "image_sequence" if is_image_sequence else "video"
        resolved_run_key = run_key or f"{run_key_prefix}:{resolved.stem}:{uuid.uuid4().hex[:12]}"
        filename = resolved.name
        stat = resolved.stat()
        normalized_collections = normalize_collections(collections)
        checksum = (
            f"sha256:{_sha256_file(resolved)}"
            if compute_checksum and resolved.is_file()
            else f"uncomputed:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
        )
        if compute_checksum and resolved.is_dir():
            checksum = f"directory:size={_path_size_bytes(resolved)}:mtime_ns={stat.st_mtime_ns}"

        planned_run = PlannedRun(
            manifest=RunManifest(
                run_id=resolved_run_id,
                run_key=resolved_run_key,
                instrument=instrument,
                source_path=str(resolved),
                source_type=source_type,
                created_at=datetime.now(timezone.utc),
                metadata={
                    "cli_command": cli_command,
                    "checksum_status": "computed" if compute_checksum else "deferred",
                },
                assets=[
                    RawAssetManifest(
                        asset_id=resolved_asset_id,
                        filename=filename,
                        path=str(resolved),
                        kind=asset_kind,
                        size_bytes=_path_size_bytes(resolved),
                        checksum=checksum,
                        collections=normalized_collections,
                        metadata={
                            "cli_command": cli_command,
                            "checksum_status": "computed" if compute_checksum else "deferred",
                        },
                    )
                ],
            )
        )
        context.repository.register_planned_run(planned_run)
        return resolved_run_id, resolved_asset_id, resolved_run_key

    def _extract_frames_common(
        input_path: Path,
        n_tile: Optional[int],
        kvstore_root: Optional[Path],
        database_dsn: Optional[str],
        schema: Optional[str],
        run_id: Optional[str],
        asset_id: Optional[str],
        run_key: Optional[str],
        instrument: str,
        collections: Optional[str],
        adaptive_background_subtraction: Optional[bool] = None,
        adaptive_background_period: Optional[int] = None,
        apply_mask: Optional[bool] = None,
        mask_path: Optional[str] = None,
    ) -> tuple[AppContext, dict]:
        from ..processing.ingest import ingest_image_folder, ingest_video_file

        resolved = input_path.expanduser().resolve()
        context = _context_from_options(kvstore_root, database_dsn, schema)
        resolved_run_id, resolved_asset_id, resolved_run_key = _register_video(
            context,
            resolved,
            run_id,
            asset_id,
            run_key,
            instrument,
            collections,
            cli_command="extract_frames",
            compute_checksum=True,
        )
        normalized_collections = normalize_collections(collections)
        ingest_defaults = context.config.processing.video_ingest
        preprocessing_defaults = context.config.processing.preprocessing
        resolved_n_tile = ingest_defaults.n_tile if n_tile is None else n_tile
        resolved_adaptive_background_subtraction = (
            preprocessing_defaults.adaptive_background_subtraction
            if adaptive_background_subtraction is None
            else adaptive_background_subtraction
        )
        resolved_adaptive_background_period = (
            preprocessing_defaults.adaptive_background_period
            if adaptive_background_period is None
            else adaptive_background_period
        )
        resolved_apply_mask = preprocessing_defaults.apply_mask if apply_mask is None else apply_mask
        resolved_mask_path = preprocessing_defaults.mask_path if mask_path is None else mask_path
        if resolved.is_dir():
            frame_rows = ingest_image_folder(
                resolved,
                context=context,
                run_id=resolved_run_id,
                asset_id=resolved_asset_id,
                metadata={"cli_command": "extract_frames", "collections": normalized_collections},
            )
        else:
            frame_rows = ingest_video_file(
                resolved,
                n_tile=resolved_n_tile,
                context=context,
                run_id=resolved_run_id,
                asset_id=resolved_asset_id,
                metadata={"cli_command": "extract_frames", "collections": normalized_collections},
                adaptive_background_subtraction=resolved_adaptive_background_subtraction,
                adaptive_background_period=resolved_adaptive_background_period,
                apply_mask=resolved_apply_mask,
                mask_path=resolved_mask_path,
            )
        return context, {
            "run_id": resolved_run_id,
            "asset_id": resolved_asset_id,
            "run_key": resolved_run_key,
            "source_path": str(resolved),
            "collections": normalized_collections,
            "frame_count": len(frame_rows),
            "frame_ids": [row["id"] for row in frame_rows],
        }

    @app.command("init-kvstore")
    def init_kvstore(root: Optional[Path] = None) -> None:
        config = CoreConfig.load()
        if root is not None:
            config.kvstore.root_path = root
        service = StoreService.from_config(config.kvstore)
        service.ensure_initialized(config.kvstore)
        typer.echo(f"KVStore ready at {service.store.root_path}")

    @app.command("init-system")
    def init_system(
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(
            kvstore_root,
            database_dsn,
            schema,
            initialize_schema=True,
        )
        result = {
            "database": {
                "schema": context.repository.schema if context.repository is not None else None,
                "initialized": context.repository is not None,
            },
            "kvstore": context.kvstore.status() if context.kvstore is not None else None,
        }
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("check-system")
    def check_system(
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        config = _config_from_options(kvstore_root, database_dsn, schema)
        context = AppContext.from_config(config)
        database_status = (
            context.repository.schema_status()
            if context.repository is not None
            else {"ready": False, "missing_tables": ["repository"]}
        )
        kvstore_status = (
            context.kvstore.status(deep=False)
            if context.kvstore is not None
            else {"initialized": False}
        )
        ready = bool(database_status.get("ready")) and bool(kvstore_status.get("initialized"))
        result = {
            "ready": ready,
            "database": database_status,
            "kvstore": kvstore_status,
        }
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))
        if not ready:
            raise typer.Exit(code=1)

    @app.command("list_asset_ids")
    def list_asset_ids(
        run_id: Optional[str] = None,
        collection: Optional[str] = None,
        kind: Optional[str] = None,
        filename: Optional[str] = None,
        limit: int = 100,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        repository = _repository_from_options(kvstore_root, database_dsn, schema)
        assets = repository.list_assets(
            run_id=run_id,
            collection=collection,
            kind=kind,
            filename=filename,
            limit=limit,
        )
        _echo_json(
            {
                "count": len(assets),
                "asset_ids": [asset["id"] for asset in assets],
                "assets": assets,
            }
        )

    @app.command("list_frame_ids")
    def list_frame_ids(
        asset_id: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        limit: Optional[int] = 100,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        repository = _repository_from_options(kvstore_root, database_dsn, schema)
        frames = repository.list_frames(
            asset_id,
            start_frame=start_frame,
            end_frame=end_frame,
            limit=limit,
        )
        _echo_json(
            {
                "asset_id": asset_id,
                "count": len(frames),
                "frame_ids": [frame["id"] for frame in frames],
                "frames": frames,
            }
        )

    @app.command("list_jobs")
    def list_jobs(
        state: str = "all",
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        worker_id: Optional[str] = None,
        limit: int = 100,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        repository = _repository_from_options(kvstore_root, database_dsn, schema)
        if status:
            statuses: list[str | None] = [JobStatus(status).value]
        else:
            normalized_state = state.strip().lower()
            if normalized_state == "all":
                statuses = [None]
            elif normalized_state == "active":
                statuses = [JobStatus.QUEUED.value, JobStatus.LEASED.value, JobStatus.PAUSED.value]
            elif normalized_state == "inactive":
                statuses = [
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                    JobStatus.DEAD_LETTERED.value,
                ]
            else:
                raise RuntimeError("--state must be one of: active, inactive, all.")

        selected_stage = None if stage is None else PipelineStage(stage).value
        jobs: list[dict] = []
        for selected_status in statuses:
            jobs.extend(
                repository.list_jobs(
                    run_id=run_id,
                    asset_id=asset_id,
                    status=selected_status,
                    stage=selected_stage,
                    worker_id=worker_id,
                    limit=limit,
                )
            )
        jobs = _sorted_jobs(jobs, limit)
        _echo_json(
            {
                "count": len(jobs),
                "job_ids": [job["id"] for job in jobs],
                "jobs": jobs,
            }
        )

    @app.command("extract_frames")
    def extract_frames(
        input_path: Path,
        n_tile: Optional[int] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        run_key: Optional[str] = None,
        instrument: str = "cli",
        collections: Optional[str] = None,
        adaptive_background_subtraction: Optional[bool] = None,
        adaptive_background_period: Optional[int] = None,
        apply_mask: Optional[bool] = None,
        mask_path: Optional[str] = None,
    ) -> None:
        _, result = _extract_frames_common(
            input_path,
            n_tile,
            kvstore_root,
            database_dsn,
            schema,
            run_id,
            asset_id,
            run_key,
            instrument,
            collections,
            adaptive_background_subtraction,
            adaptive_background_period,
            apply_mask,
            mask_path,
        )
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("preprocess")
    def preprocess(
        asset_id: Optional[str] = None,
        run_id: Optional[str] = None,
        frame_ids: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        limit: Optional[int] = None,
        flatfield_correction: Optional[bool] = None,
        flatfield_q: Optional[float] = None,
        flatfield_axis: Optional[int] = None,
        apply_mask: Optional[bool] = None,
        crop_enabled: Optional[bool] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
        background_correction: Optional[bool] = None,
        invert_intensity: Optional[bool] = None,
        encoding: Optional[str] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        from ..workers.handlers import preprocess_frames_handler

        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to run preprocessing.")

        payload = _preprocess_payload(
            frame_ids,
            start_frame,
            end_frame,
            limit,
            flatfield_correction,
            flatfield_q,
            flatfield_axis,
            apply_mask,
            crop_enabled,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            background_correction,
            invert_intensity,
            encoding,
            context.config.processing,
        )
        resolved_run_id, resolved_asset_id = _resolve_segment_target(
            context.repository,
            run_id,
            asset_id,
            payload,
        )
        result = preprocess_frames_handler(
            {
                "id": "cli-preprocess",
                "stage": PipelineStage.PREPROCESS_FRAMES.value,
                "run_id": resolved_run_id,
                "asset_id": resolved_asset_id,
                "payload": payload,
            },
            context,
        )
        _echo_json(result)

    @app.command("queue_preprocess")
    def queue_preprocess(
        asset_id: Optional[str] = None,
        run_id: Optional[str] = None,
        frame_ids: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        limit: Optional[int] = None,
        flatfield_correction: Optional[bool] = None,
        flatfield_q: Optional[float] = None,
        flatfield_axis: Optional[int] = None,
        apply_mask: Optional[bool] = None,
        crop_enabled: Optional[bool] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
        background_correction: Optional[bool] = None,
        invert_intensity: Optional[bool] = None,
        encoding: Optional[str] = None,
        priority: Optional[int] = None,
        depends_on: Optional[str] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to queue preprocessing.")

        payload = _preprocess_payload(
            frame_ids,
            start_frame,
            end_frame,
            limit,
            flatfield_correction,
            flatfield_q,
            flatfield_axis,
            apply_mask,
            crop_enabled,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            background_correction,
            invert_intensity,
            encoding,
            context.config.processing,
        )
        resolved_run_id, resolved_asset_id = _resolve_segment_target(
            context.repository,
            run_id,
            asset_id,
            payload,
        )
        job = context.repository.create_job(
            PipelineStage.PREPROCESS_FRAMES,
            run_id=resolved_run_id,
            asset_id=resolved_asset_id,
            priority=priority,
            payload=payload,
            depends_on=_split_csv(depends_on),
            summary=f"preprocess queued for asset {resolved_asset_id}",
        )
        _echo_json({"job": job})

    @app.command("segment")
    def segment(
        asset_id: Optional[str] = None,
        run_id: Optional[str] = None,
        frame_ids: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        frame_payload_kind: Optional[str] = None,
        apply_preprocessing: Optional[bool] = None,
        flatfield_correction: Optional[bool] = None,
        flatfield_q: Optional[float] = None,
        flatfield_axis: Optional[int] = None,
        apply_mask: Optional[bool] = None,
        crop_enabled: Optional[bool] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
        background_correction: Optional[bool] = None,
        invert_intensity: Optional[bool] = None,
        min_perimeter: Optional[float] = None,
        max_perimeter: Optional[float] = None,
        padding: Optional[int] = None,
        roi_encoding: Optional[str] = None,
        zstd_min_bytes: Optional[int] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        from ..workers.handlers import roi_detection_handler

        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to run segmentation.")

        payload = _segment_payload(
            frame_ids,
            start_frame,
            end_frame,
            limit,
            threshold,
            frame_payload_kind,
            apply_preprocessing,
            flatfield_correction,
            flatfield_q,
            flatfield_axis,
            apply_mask,
            crop_enabled,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            background_correction,
            invert_intensity,
            min_perimeter,
            max_perimeter,
            padding,
            roi_encoding,
            zstd_min_bytes,
            context.config.processing,
        )
        resolved_run_id, resolved_asset_id = _resolve_segment_target(
            context.repository,
            run_id,
            asset_id,
            payload,
        )
        result = roi_detection_handler(
            {
                "id": "cli-segment",
                "stage": PipelineStage.SEGMENT.value,
                "run_id": resolved_run_id,
                "asset_id": resolved_asset_id,
                "payload": payload,
            },
            context,
        )
        _echo_json(result)

    @app.command("queue_segment")
    def queue_segment(
        asset_id: Optional[str] = None,
        run_id: Optional[str] = None,
        frame_ids: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
        frame_payload_kind: Optional[str] = None,
        apply_preprocessing: Optional[bool] = None,
        flatfield_correction: Optional[bool] = None,
        flatfield_q: Optional[float] = None,
        flatfield_axis: Optional[int] = None,
        apply_mask: Optional[bool] = None,
        crop_enabled: Optional[bool] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
        background_correction: Optional[bool] = None,
        invert_intensity: Optional[bool] = None,
        min_perimeter: Optional[float] = None,
        max_perimeter: Optional[float] = None,
        padding: Optional[int] = None,
        roi_encoding: Optional[str] = None,
        zstd_min_bytes: Optional[int] = None,
        priority: Optional[int] = None,
        depends_on: Optional[str] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to queue segmentation.")

        payload = _segment_payload(
            frame_ids,
            start_frame,
            end_frame,
            limit,
            threshold,
            frame_payload_kind,
            apply_preprocessing,
            flatfield_correction,
            flatfield_q,
            flatfield_axis,
            apply_mask,
            crop_enabled,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            background_correction,
            invert_intensity,
            min_perimeter,
            max_perimeter,
            padding,
            roi_encoding,
            zstd_min_bytes,
            context.config.processing,
        )
        resolved_run_id, resolved_asset_id = _resolve_segment_target(
            context.repository,
            run_id,
            asset_id,
            payload,
        )
        job = context.repository.create_job(
            PipelineStage.SEGMENT,
            run_id=resolved_run_id,
            asset_id=resolved_asset_id,
            priority=priority,
            payload=payload,
            depends_on=_split_csv(depends_on),
            summary=f"segment queued for asset {resolved_asset_id}",
        )
        _echo_json({"job": job})

    @app.command("queue_extract_frames")
    def queue_extract_frames(
        input_path: Path,
        n_tile: Optional[int] = None,
        enqueue_segment: bool = False,
        roi_padding: Optional[int] = None,
        roi_encoding: Optional[str] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        run_key: Optional[str] = None,
        instrument: str = "cli",
        collections: Optional[str] = None,
        adaptive_background_subtraction: Optional[bool] = None,
        adaptive_background_period: Optional[int] = None,
        apply_mask: Optional[bool] = None,
        mask_path: Optional[str] = None,
        compute_checksum: bool = False,
    ) -> None:
        resolved = input_path.expanduser().resolve()
        context = _context_from_options(kvstore_root, database_dsn, schema)
        ingest_defaults = context.config.processing.video_ingest
        preprocessing_defaults = context.config.processing.preprocessing
        roi_recording_defaults = context.config.processing.roi_recording
        resolved_n_tile = ingest_defaults.n_tile if n_tile is None else n_tile
        resolved_adaptive_background_subtraction = (
            preprocessing_defaults.adaptive_background_subtraction
            if adaptive_background_subtraction is None
            else adaptive_background_subtraction
        )
        resolved_adaptive_background_period = (
            preprocessing_defaults.adaptive_background_period
            if adaptive_background_period is None
            else adaptive_background_period
        )
        resolved_apply_mask = preprocessing_defaults.apply_mask if apply_mask is None else apply_mask
        resolved_mask_path = preprocessing_defaults.mask_path if mask_path is None else mask_path
        resolved_roi_padding = (
            roi_recording_defaults.padding if roi_padding is None else roi_padding
        )
        resolved_roi_encoding = roi_recording_defaults.roi_encoding if roi_encoding is None else roi_encoding
        normalized_collections = normalize_collections(collections)
        resolved_run_id, resolved_asset_id, resolved_run_key = _register_video(
            context,
            resolved,
            run_id,
            asset_id,
            run_key,
            instrument,
            collections,
            cli_command="queue_extract_frames",
            compute_checksum=compute_checksum,
        )
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to queue frame extraction.")
        job = context.repository.create_job(
            PipelineStage.EXTRACT_FRAMES,
            run_id=resolved_run_id,
            asset_id=resolved_asset_id,
            payload={
                "source_path": str(resolved),
                "kind": AssetKind.IMAGE_SEQUENCE.value if resolved.is_dir() else AssetKind.VIDEO.value,
                "n_tile": resolved_n_tile,
                "adaptive_background_subtraction": resolved_adaptive_background_subtraction,
                "adaptive_background_period": resolved_adaptive_background_period,
                "apply_mask": resolved_apply_mask,
                "mask_path": resolved_mask_path,
                "enqueue_segment": enqueue_segment,
                "padding": resolved_roi_padding,
                "roi_encoding": resolved_roi_encoding,
                "collections": normalized_collections,
                "checksum_status": "computed" if compute_checksum else "deferred",
            },
            summary=f"extract_frames queued for {resolved.name}",
        )
        result = {
            "run_id": resolved_run_id,
            "asset_id": resolved_asset_id,
            "run_key": resolved_run_key,
            "job_id": job["id"],
            "queue_stage": PipelineStage.EXTRACT_FRAMES.value,
            "source_path": str(resolved),
            "collections": normalized_collections,
        }
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("queue_ingest")
    def queue_ingest(
        input_path: Path,
        recursive: bool = True,
        n_tile: Optional[int] = None,
        enqueue_segment: bool = False,
        roi_padding: Optional[int] = None,
        roi_encoding: Optional[str] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        instrument: str = "cli",
        collections: Optional[str] = None,
        adaptive_background_subtraction: Optional[bool] = None,
        adaptive_background_period: Optional[int] = None,
        apply_mask: Optional[bool] = None,
        mask_path: Optional[str] = None,
        compute_checksum: bool = False,
    ) -> None:
        from ..processing.ingest import discover_ingest_sources

        resolved = input_path.expanduser().resolve()
        sources = discover_ingest_sources(resolved, recursive=recursive)
        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to queue ingestion.")

        ingest_defaults = context.config.processing.video_ingest
        preprocessing_defaults = context.config.processing.preprocessing
        roi_recording_defaults = context.config.processing.roi_recording
        resolved_n_tile = ingest_defaults.n_tile if n_tile is None else n_tile
        resolved_adaptive_background_subtraction = (
            preprocessing_defaults.adaptive_background_subtraction
            if adaptive_background_subtraction is None
            else adaptive_background_subtraction
        )
        resolved_adaptive_background_period = (
            preprocessing_defaults.adaptive_background_period
            if adaptive_background_period is None
            else adaptive_background_period
        )
        resolved_apply_mask = preprocessing_defaults.apply_mask if apply_mask is None else apply_mask
        resolved_mask_path = preprocessing_defaults.mask_path if mask_path is None else mask_path
        resolved_roi_padding = (
            roi_recording_defaults.padding if roi_padding is None else roi_padding
        )
        resolved_roi_encoding = roi_recording_defaults.roi_encoding if roi_encoding is None else roi_encoding
        normalized_collections = normalize_collections(collections)

        queued = []
        for source in sources:
            source_path = source.path
            resolved_run_id, resolved_asset_id, resolved_run_key = _register_video(
                context,
                source_path,
                None,
                None,
                None,
                instrument,
                collections,
                cli_command="queue_ingest",
                compute_checksum=compute_checksum,
            )
            job = context.repository.create_job(
                PipelineStage.EXTRACT_FRAMES,
                run_id=resolved_run_id,
                asset_id=resolved_asset_id,
                payload={
                    "source_path": str(source_path),
                    "kind": source.kind,
                    "recursive": source.recursive,
                    "n_tile": resolved_n_tile,
                    "adaptive_background_subtraction": resolved_adaptive_background_subtraction,
                    "adaptive_background_period": resolved_adaptive_background_period,
                    "apply_mask": resolved_apply_mask,
                    "mask_path": resolved_mask_path,
                    "enqueue_segment": enqueue_segment,
                    "padding": resolved_roi_padding,
                    "roi_encoding": resolved_roi_encoding,
                    "collections": normalized_collections,
                    "checksum_status": "computed" if compute_checksum else "deferred",
                },
                summary=f"extract_frames queued for {source_path.name}",
            )
            queued.append(
                {
                    "run_id": resolved_run_id,
                    "asset_id": resolved_asset_id,
                    "run_key": resolved_run_key,
                    "job_id": job["id"],
                    "queue_stage": PipelineStage.EXTRACT_FRAMES.value,
                    "source_path": str(source_path),
                    "kind": source.kind,
                    "collections": normalized_collections,
                }
            )

        _echo_json(
            {
                "source_path": str(resolved),
                "recursive": recursive,
                "source_count": len(sources),
                "queued_count": len(queued),
                "queued": queued,
            }
        )

    @app.command("reset")
    def reset(
        delete: bool = False,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        if not delete:
            typer.echo("Refusing to reset Pelagia storage without --delete.")
            raise typer.Exit(code=2)

        config = _config_from_options(kvstore_root, database_dsn, schema)
        context = AppContext.from_config(config)
        if context.repository is None or context.kvstore is None:
            raise RuntimeError("Reset requires both a PostgresRepository and KVStore.")

        context.repository.initialize_schema()
        database_result = context.repository.purge_all()
        kvstore_result = context.kvstore.reset(
            hash_algorithm=config.kvstore.hash_algorithm,
            prefix_length=config.kvstore.prefix_length,
            max_db_bytes=config.kvstore.max_db_bytes,
            max_rows=config.kvstore.max_rows,
        )
        typer.echo(
            json.dumps(
                json_ready(
                    {
                        "deleted": True,
                        "database": database_result,
                        "kvstore": kvstore_result,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )

    @app.command("worker_run_once")
    def worker_run_once(
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        worker_id: Optional[str] = None,
        stages: Optional[str] = None,
    ) -> None:
        from ..workers import Worker, default_handler_registry

        context = _context_from_options(kvstore_root, database_dsn, schema)
        selected_stages = None
        if stages:
            selected_stages = [
                PipelineStage(stage.strip())
                for stage in stages.split(",")
                if stage.strip()
            ]
        handlers = default_handler_registry()
        if worker_id:
            worker = Worker(context=context, handlers=handlers, worker_id=worker_id)
        else:
            worker = Worker(context=context, handlers=handlers)
        claimed = worker.run_once(stages=selected_stages)
        typer.echo(json.dumps({"worker_id": worker.worker_id, "claimed": claimed}, sort_keys=True))

    @app.command("worker_run")
    def worker_run(
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        worker_id: Optional[str] = None,
        stages: Optional[str] = None,
        idle_sleep_seconds: float = 2.0,
        requeue_interval_seconds: float = 30.0,
    ) -> None:
        from ..workers import Worker, default_handler_registry

        context = _context_from_options(kvstore_root, database_dsn, schema)
        selected_stages = None
        if stages:
            selected_stages = [
                PipelineStage(stage.strip())
                for stage in stages.split(",")
                if stage.strip()
            ]
        handlers = default_handler_registry()
        if worker_id:
            worker = Worker(context=context, handlers=handlers, worker_id=worker_id)
        else:
            worker = Worker(context=context, handlers=handlers)

        stop_event = threading.Event()

        def stop_worker(signum, _frame):
            typer.echo(f"Worker {worker.worker_id} received signal {signum}; stopping after current job.")
            stop_event.set()

        signal.signal(signal.SIGINT, stop_worker)
        signal.signal(signal.SIGTERM, stop_worker)

        typer.echo(json.dumps({"worker_id": worker.worker_id, "status": "starting"}, sort_keys=True))
        worker.run_forever(
            stages=selected_stages,
            idle_sleep_seconds=idle_sleep_seconds,
            requeue_interval_seconds=requeue_interval_seconds,
            stop_event=stop_event,
        )
        typer.echo(json.dumps({"worker_id": worker.worker_id, "status": "stopped"}, sort_keys=True))

    @app.command("worker_shutdown")
    def worker_shutdown(
        worker_id: str,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        config = CoreConfig.load()
        if database_dsn is not None:
            config.database.dsn = database_dsn
        if schema is not None:
            config.database.schema_name = schema
        configure_core_logging(config)
        context = AppContext.from_config(config)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to request worker shutdown.")
        row = context.repository.request_worker_shutdown(worker_id, reason=reason)
        typer.echo(json.dumps(json_ready(row or {"worker_id": worker_id, "found": False}), indent=2, sort_keys=True))

    def main() -> None:
        app()
else:
    app = None

    def main() -> None:
        raise RuntimeError("Install typer to run the Pelagia CLI.")


if __name__ == "__main__":
    main()
