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
from ..domain import AssetKind, PipelineStage, PlannedRun, RawAssetManifest, RunManifest, normalize_collections
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
    ) -> AppContext:
        config = _config_from_options(kvstore_root, database_dsn, schema)
        context = AppContext.from_config(config)
        if context.kvstore is not None and not context.kvstore.initialized:
            context.kvstore.initialize(
                hash_algorithm=config.kvstore.hash_algorithm,
                prefix_length=config.kvstore.prefix_length,
                max_db_bytes=config.kvstore.max_db_bytes,
                max_rows=config.kvstore.max_rows,
            )
        if context.repository is not None:
            context.repository.initialize_schema()
        return context

    def _register_video(
        context: AppContext,
        input_path: Path,
        run_id: Optional[str],
        asset_id: Optional[str],
        run_key: Optional[str],
        instrument: str,
        collections: Optional[str],
    ) -> tuple[str, str, str]:
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to register video ingestion.")

        resolved = input_path.expanduser().resolve()
        resolved_run_id = run_id or str(uuid.uuid4())
        resolved_asset_id = asset_id or str(uuid.uuid4())
        resolved_run_key = run_key or f"video:{resolved.stem}:{uuid.uuid4().hex[:12]}"
        filename = resolved.name
        normalized_collections = normalize_collections(collections)

        planned_run = PlannedRun(
            manifest=RunManifest(
                run_id=resolved_run_id,
                run_key=resolved_run_key,
                instrument=instrument,
                source_path=str(resolved),
                source_type=AssetKind.VIDEO.value,
                created_at=datetime.now(timezone.utc),
                metadata={"cli_command": "ingest_video"},
                assets=[
                    RawAssetManifest(
                        asset_id=resolved_asset_id,
                        filename=filename,
                        path=str(resolved),
                        kind=AssetKind.VIDEO,
                        size_bytes=resolved.stat().st_size,
                        checksum=_sha256_file(resolved),
                        collections=normalized_collections,
                    )
                ],
            )
        )
        context.repository.register_planned_run(planned_run)
        return resolved_run_id, resolved_asset_id, resolved_run_key

    def _ingest_video_common(
        input_path: Path,
        n_tile: int,
        kvstore_root: Optional[Path],
        database_dsn: Optional[str],
        schema: Optional[str],
        run_id: Optional[str],
        asset_id: Optional[str],
        run_key: Optional[str],
        instrument: str,
        collections: Optional[str],
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
        )
        normalized_collections = normalize_collections(collections)
        frame_rows = ingest_video_file(
            resolved,
            n_tile=n_tile,
            context=context,
            run_id=resolved_run_id,
            asset_id=resolved_asset_id,
            metadata={"cli_command": "ingest_video", "collections": normalized_collections},
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
        context = _context_from_options(kvstore_root, database_dsn, schema)
        result = {
            "database": {
                "schema": context.repository.schema if context.repository is not None else None,
                "initialized": context.repository is not None,
            },
            "kvstore": context.kvstore.status() if context.kvstore is not None else None,
        }
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("ingest_video")
    def ingest_video(
        input_path: Path,
        n_tile: int = 1,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        run_key: Optional[str] = None,
        instrument: str = "cli",
        collections: Optional[str] = None,
    ) -> None:
        _, result = _ingest_video_common(
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
        )
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("ingest_video_to_queue")
    def ingest_video_to_queue(
        input_path: Path,
        n_tile: int = 1,
        queue_stage: str = PipelineStage.SEGMENT.value,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        run_key: Optional[str] = None,
        instrument: str = "cli",
        collections: Optional[str] = None,
    ) -> None:
        context, result = _ingest_video_common(
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
        )
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to enqueue video ingestion.")
        stage = PipelineStage(queue_stage)
        job = context.repository.create_job(
            stage,
            run_id=result["run_id"],
            asset_id=result["asset_id"],
            payload={
                "source_path": result["source_path"],
                "frame_count": result["frame_count"],
                "frame_ids": result["frame_ids"],
                "n_tile": n_tile,
                "collections": result["collections"],
            },
            summary=f"{stage.value} queued for {result['frame_count']} ingested frames",
        )
        result["job_id"] = job["id"]
        result["queue_stage"] = stage.value
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("queue_extract_frames")
    def queue_extract_frames(
        input_path: Path,
        n_tile: int = 1,
        enqueue_segment: bool = False,
        segmentation_padding: int = 0,
        roi_encoding: str = "zstd",
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        run_key: Optional[str] = None,
        instrument: str = "cli",
        collections: Optional[str] = None,
    ) -> None:
        resolved = input_path.expanduser().resolve()
        context = _context_from_options(kvstore_root, database_dsn, schema)
        normalized_collections = normalize_collections(collections)
        resolved_run_id, resolved_asset_id, resolved_run_key = _register_video(
            context,
            resolved,
            run_id,
            asset_id,
            run_key,
            instrument,
            collections,
        )
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to queue frame extraction.")
        job = context.repository.create_job(
            PipelineStage.EXTRACT_FRAMES,
            run_id=resolved_run_id,
            asset_id=resolved_asset_id,
            payload={
                "source_path": str(resolved),
                "n_tile": n_tile,
                "enqueue_segment": enqueue_segment,
                "segmentation_padding": segmentation_padding,
                "roi_encoding": roi_encoding,
                "collections": normalized_collections,
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
