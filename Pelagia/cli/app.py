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
        min_perimeter: Optional[float],
        max_perimeter: Optional[float],
        padding: Optional[int],
        roi_encoding: Optional[str],
        zstd_min_bytes: Optional[int],
        defaults,
    ) -> dict:
        return {
            "frame_ids": _split_csv(frame_ids),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "limit": limit,
            "threshold": threshold,
            "min_perimeter": defaults.min_perimeter if min_perimeter is None else min_perimeter,
            "max_perimeter": defaults.max_perimeter if max_perimeter is None else max_perimeter,
            "padding": defaults.padding if padding is None else padding,
            "roi_encoding": defaults.roi_encoding if roi_encoding is None else roi_encoding,
            "zstd_min_bytes": defaults.zstd_min_bytes if zstd_min_bytes is None else zstd_min_bytes,
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
        resolved_run_key = run_key or f"video:{resolved.stem}:{uuid.uuid4().hex[:12]}"
        filename = resolved.name
        stat = resolved.stat()
        normalized_collections = normalize_collections(collections)
        checksum = (
            f"sha256:{_sha256_file(resolved)}"
            if compute_checksum
            else f"uncomputed:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
        )

        planned_run = PlannedRun(
            manifest=RunManifest(
                run_id=resolved_run_id,
                run_key=resolved_run_key,
                instrument=instrument,
                source_path=str(resolved),
                source_type=AssetKind.VIDEO.value,
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
                        kind=AssetKind.VIDEO,
                        size_bytes=stat.st_size,
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
        flatfield_correction: Optional[bool] = None,
        flatfield_q: Optional[float] = None,
        flatfield_axis: Optional[int] = None,
        adaptive_background_subtraction: Optional[bool] = None,
        adaptive_background_period: Optional[int] = None,
        frame_mask: Optional[bool] = None,
        frame_mask_path: Optional[str] = None,
    ) -> tuple[AppContext, dict]:
        from ..processing.video_ingest import ingest_video_file

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
        resolved_n_tile = ingest_defaults.n_tile if n_tile is None else n_tile
        resolved_flatfield_correction = (
            ingest_defaults.flatfield_correction
            if flatfield_correction is None
            else flatfield_correction
        )
        resolved_flatfield_q = ingest_defaults.flatfield_q if flatfield_q is None else flatfield_q
        resolved_flatfield_axis = (
            ingest_defaults.flatfield_axis if flatfield_axis is None else flatfield_axis
        )
        resolved_adaptive_background_subtraction = (
            ingest_defaults.adaptive_background_subtraction
            if adaptive_background_subtraction is None
            else adaptive_background_subtraction
        )
        resolved_adaptive_background_period = (
            ingest_defaults.adaptive_background_period
            if adaptive_background_period is None
            else adaptive_background_period
        )
        resolved_frame_mask = ingest_defaults.frame_mask if frame_mask is None else frame_mask
        resolved_frame_mask_path = (
            ingest_defaults.frame_mask_path if frame_mask_path is None else frame_mask_path
        )
        frame_rows = ingest_video_file(
            resolved,
            n_tile=resolved_n_tile,
            context=context,
            run_id=resolved_run_id,
            asset_id=resolved_asset_id,
            metadata={"cli_command": "extract_frames", "collections": normalized_collections},
            flatfield_correction=resolved_flatfield_correction,
            flatfield_q=resolved_flatfield_q,
            flatfield_axis=resolved_flatfield_axis,
            adaptive_background_subtraction=resolved_adaptive_background_subtraction,
            adaptive_background_period=resolved_adaptive_background_period,
            frame_mask=resolved_frame_mask,
            frame_mask_path=resolved_frame_mask_path,
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
        flatfield_correction: Optional[bool] = None,
        flatfield_q: Optional[float] = None,
        flatfield_axis: Optional[int] = None,
        adaptive_background_subtraction: Optional[bool] = None,
        adaptive_background_period: Optional[int] = None,
        frame_mask: Optional[bool] = None,
        frame_mask_path: Optional[str] = None,
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
            flatfield_correction,
            flatfield_q,
            flatfield_axis,
            adaptive_background_subtraction,
            adaptive_background_period,
            frame_mask,
            frame_mask_path,
        )
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("segment")
    def segment(
        asset_id: Optional[str] = None,
        run_id: Optional[str] = None,
        frame_ids: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
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
            min_perimeter,
            max_perimeter,
            padding,
            roi_encoding,
            zstd_min_bytes,
            context.config.processing.segmentation,
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
            min_perimeter,
            max_perimeter,
            padding,
            roi_encoding,
            zstd_min_bytes,
            context.config.processing.segmentation,
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
        segmentation_padding: Optional[int] = None,
        roi_encoding: Optional[str] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        run_key: Optional[str] = None,
        instrument: str = "cli",
        collections: Optional[str] = None,
        flatfield_correction: Optional[bool] = None,
        flatfield_q: Optional[float] = None,
        flatfield_axis: Optional[int] = None,
        adaptive_background_subtraction: Optional[bool] = None,
        adaptive_background_period: Optional[int] = None,
        frame_mask: Optional[bool] = None,
        frame_mask_path: Optional[str] = None,
        compute_checksum: bool = False,
    ) -> None:
        resolved = input_path.expanduser().resolve()
        context = _context_from_options(kvstore_root, database_dsn, schema)
        ingest_defaults = context.config.processing.video_ingest
        segment_defaults = context.config.processing.segmentation
        resolved_n_tile = ingest_defaults.n_tile if n_tile is None else n_tile
        resolved_flatfield_correction = (
            ingest_defaults.flatfield_correction
            if flatfield_correction is None
            else flatfield_correction
        )
        resolved_flatfield_q = ingest_defaults.flatfield_q if flatfield_q is None else flatfield_q
        resolved_flatfield_axis = (
            ingest_defaults.flatfield_axis if flatfield_axis is None else flatfield_axis
        )
        resolved_adaptive_background_subtraction = (
            ingest_defaults.adaptive_background_subtraction
            if adaptive_background_subtraction is None
            else adaptive_background_subtraction
        )
        resolved_adaptive_background_period = (
            ingest_defaults.adaptive_background_period
            if adaptive_background_period is None
            else adaptive_background_period
        )
        resolved_frame_mask = ingest_defaults.frame_mask if frame_mask is None else frame_mask
        resolved_frame_mask_path = (
            ingest_defaults.frame_mask_path if frame_mask_path is None else frame_mask_path
        )
        resolved_segmentation_padding = (
            segment_defaults.padding if segmentation_padding is None else segmentation_padding
        )
        resolved_roi_encoding = segment_defaults.roi_encoding if roi_encoding is None else roi_encoding
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
                "n_tile": resolved_n_tile,
                "flatfield_correction": resolved_flatfield_correction,
                "flatfield_q": resolved_flatfield_q,
                "flatfield_axis": resolved_flatfield_axis,
                "adaptive_background_subtraction": resolved_adaptive_background_subtraction,
                "adaptive_background_period": resolved_adaptive_background_period,
                "frame_mask": resolved_frame_mask,
                "frame_mask_path": resolved_frame_mask_path,
                "enqueue_segment": enqueue_segment,
                "segmentation_padding": resolved_segmentation_padding,
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
