from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import CoreConfig
from ..domain import AssetKind, PipelineStage, PlannedRun, RawAssetManifest, RunManifest
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

    def _context_from_options(
        kvstore_root: Optional[Path],
        database_dsn: Optional[str],
        schema: Optional[str],
    ) -> AppContext:
        config = CoreConfig.from_env()
        if kvstore_root is not None:
            config.kvstore.root_path = kvstore_root
        if database_dsn is not None:
            config.database.dsn = database_dsn
        if schema is not None:
            config.database.schema_name = schema

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
    ) -> tuple[str, str, str]:
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to register video ingestion.")

        resolved = input_path.expanduser().resolve()
        resolved_run_id = run_id or str(uuid.uuid4())
        resolved_asset_id = asset_id or str(uuid.uuid4())
        resolved_run_key = run_key or f"video:{resolved.stem}:{uuid.uuid4().hex[:12]}"
        asset_key = resolved.name

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
                        asset_key=asset_key,
                        path=str(resolved),
                        kind=AssetKind.VIDEO,
                        size_bytes=resolved.stat().st_size,
                        checksum=_sha256_file(resolved),
                    )
                ],
            )
        )
        context.repository.register_planned_run(planned_run)
        return resolved_run_id, resolved_asset_id, resolved_run_key

    def _ingest_video_common(
        input_path: Path,
        n_tile: int,
        dest_path: Optional[Path],
        kvstore_root: Optional[Path],
        database_dsn: Optional[str],
        schema: Optional[str],
        run_id: Optional[str],
        asset_id: Optional[str],
        run_key: Optional[str],
        instrument: str,
    ) -> tuple[AppContext, dict]:
        from ..processing.frames import ingest_video_file

        resolved = input_path.expanduser().resolve()
        context = _context_from_options(kvstore_root, database_dsn, schema)
        resolved_run_id, resolved_asset_id, resolved_run_key = _register_video(
            context,
            resolved,
            run_id,
            asset_id,
            run_key,
            instrument,
        )
        frame_rows = ingest_video_file(
            resolved,
            n_tile=n_tile,
            context=context,
            run_id=resolved_run_id,
            asset_id=resolved_asset_id,
            dest_path=dest_path,
            metadata={"cli_command": "ingest_video"},
        )
        return context, {
            "run_id": resolved_run_id,
            "asset_id": resolved_asset_id,
            "run_key": resolved_run_key,
            "source_path": str(resolved),
            "frame_count": len(frame_rows),
            "frame_ids": [row["id"] for row in frame_rows],
        }

    @app.command("init-kvstore")
    def init_kvstore(root: Optional[Path] = None) -> None:
        config = CoreConfig.from_env()
        if root is not None:
            config.kvstore.root_path = root
        service = StoreService.from_config(config.kvstore)
        service.ensure_initialized(config.kvstore)
        typer.echo(f"KVStore ready at {service.store.root_path}")

    @app.command("ingest_video")
    def ingest_video(
        input_path: Path,
        n_tile: int = 1,
        dest_path: Optional[Path] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        run_key: Optional[str] = None,
        instrument: str = "cli",
    ) -> None:
        _, result = _ingest_video_common(
            input_path,
            n_tile,
            dest_path,
            kvstore_root,
            database_dsn,
            schema,
            run_id,
            asset_id,
            run_key,
            instrument,
        )
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("ingest_video_to_queue")
    def ingest_video_to_queue(
        input_path: Path,
        n_tile: int = 1,
        dest_path: Optional[Path] = None,
        queue_stage: str = PipelineStage.SEGMENT.value,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        run_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        run_key: Optional[str] = None,
        instrument: str = "cli",
    ) -> None:
        context, result = _ingest_video_common(
            input_path,
            n_tile,
            dest_path,
            kvstore_root,
            database_dsn,
            schema,
            run_id,
            asset_id,
            run_key,
            instrument,
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
            },
            summary=f"{stage.value} queued for {result['frame_count']} ingested frames",
        )
        result["job_id"] = job["id"]
        result["queue_stage"] = stage.value
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    def main() -> None:
        app()
else:
    app = None

    def main() -> None:
        raise RuntimeError("Install typer to run the Pelagia CLI.")
