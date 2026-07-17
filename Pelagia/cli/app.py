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
from ..services.projects import initialize_project_kvstore
from ..services.job_commands import ExtractFramesCommand, PreprocessFramesCommand, SegmentFramesCommand
from ..services.pipeline import PipelineService
from ..storage.blob_store import create_kvstore, create_named_kvstore, initialize_kvstore, named_kvstore_path, reset_kvstore
from ..utils.serialization import json_ready
from ..version import build_info


try:
    import typer
except ImportError:  # pragma: no cover - argparse fallback covers no-typer envs
    typer = None


if typer is not None:
    app = typer.Typer(help="Pelagia command line tools.")
    environment_app = typer.Typer(help="Create, synchronize, and inspect worker environments.")
    app.add_typer(environment_app, name="env")

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

    def _queued_path_size_bytes(path: Path, stat_size_bytes: int) -> int:
        if path.is_file():
            return int(stat_size_bytes)
        return 0

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
            initialize_kvstore(context.kvstore, config.kvstore)
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

    def _list_all_projects(repository, *, active_only: bool) -> list[dict]:
        projects = []
        offset = 0
        while True:
            page = repository.list_projects(active_only=active_only, limit=100, offset=offset)
            projects.extend(page)
            if len(page) < 100:
                return projects
            offset += len(page)

    def _echo_json(payload: object) -> None:
        typer.echo(json.dumps(json_ready(payload), indent=2, sort_keys=True))

    @environment_app.command("sync")
    def environment_sync(
        profile: str,
        root: Path = Path("."),
        python: Optional[str] = None,
        uv: Optional[Path] = None,
        imagecodecs_wheel: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        """Create and install a named worker environment."""
        from .environments import sync_profile

        try:
            result = sync_profile(
                profile,
                root=root,
                python=python,
                uv=uv,
                imagecodecs_wheel=imagecodecs_wheel,
                dry_run=dry_run,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        _echo_json(result)

    @environment_app.command("doctor")
    def environment_doctor(
        profile: str = "all",
        root: Path = Path("."),
        require_gpu: bool = False,
        require_jpegxs: bool = False,
    ) -> None:
        """Report whether named worker environments are ready to run."""
        from .environments import doctor_profiles

        try:
            result = doctor_profiles(
                profile,
                root=root,
                require_gpu=require_gpu,
                require_jpegxs=require_jpegxs,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        _echo_json(result)
        if not result["healthy"]:
            raise typer.Exit(code=1)

    def _split_csv(value: Optional[str]) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def _project_id_from_key(context: AppContext, project_key: Optional[str] = None) -> str:
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required for project lookup.")
        resolved_key = project_key or context.config.auth.dev_project_key
        project = context.repository.get_project_by_key(resolved_key)
        if project is None:
            raise RuntimeError(
                f"Project {resolved_key!r} was not found. Create it with `create-project` or `create-dev-login`."
            )
        return str(project["id"])

    def _segment_payload(
        frame_ids: Optional[str],
        start_frame: Optional[int],
        end_frame: Optional[int],
        limit: Optional[int],
        threshold: Optional[float],
        frame_payload_kind: Optional[str],
        apply_preprocessing: Optional[bool],
        min_field_value: Optional[float],
        max_field_value: Optional[float],
        apply_mask: Optional[bool],
        crop_enabled: Optional[bool],
        crop_x: Optional[int],
        crop_y: Optional[int],
        crop_w: Optional[int],
        crop_h: Optional[int],
        min_perimeter: Optional[float],
        max_perimeter: Optional[float],
        padding: Optional[int],
        roi_encoding: Optional[str],
        zstd_min_bytes: Optional[int],
        processing_defaults,
    ) -> dict:
        roi_filter_defaults = processing_defaults.roi_filter
        roi_recording_defaults = processing_defaults.roi_recording
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
            "min_field_value": min_field_value,
            "max_field_value": max_field_value,
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
            "min_perimeter": roi_filter_defaults.min_perimeter if min_perimeter is None else min_perimeter,
            "max_perimeter": roi_filter_defaults.max_perimeter if max_perimeter is None else max_perimeter,
            "padding": roi_recording_defaults.padding if padding is None else padding,
            "roi_encoding": roi_encoding,
            "zstd_min_bytes": roi_recording_defaults.zstd_min_bytes if zstd_min_bytes is None else zstd_min_bytes,
        }

    def _preprocess_payload(
        frame_ids: Optional[str],
        start_frame: Optional[int],
        end_frame: Optional[int],
        limit: Optional[int],
        min_field_value: Optional[float],
        max_field_value: Optional[float],
        apply_mask: Optional[bool],
        crop_enabled: Optional[bool],
        crop_x: Optional[int],
        crop_y: Optional[int],
        crop_w: Optional[int],
        crop_h: Optional[int],
        encoding: Optional[str],
        processing_defaults,
    ) -> dict:
        preprocessing_defaults = processing_defaults.preprocessing
        return {
            "frame_ids": _split_csv(frame_ids),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "limit": limit,
            "min_field_value": min_field_value,
            "max_field_value": max_field_value,
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
            "encoding": encoding,
        }

    def _resolve_segment_target(
        repository,
        run_id: Optional[str],
        asset_id: Optional[str],
        payload: dict,
        *,
        project_id: Optional[str] = None,
    ) -> tuple[Optional[str], str]:
        resolved_run_id = run_id
        resolved_asset_id = asset_id

        if resolved_asset_id is None and payload["frame_ids"]:
            first_frame = repository.get_frame_record(payload["frame_ids"][0], project_id=project_id)
            if first_frame is None:
                raise RuntimeError(f"Frame {payload['frame_ids'][0]!r} was not found.")
            resolved_run_id = resolved_run_id or first_frame.run_id
            resolved_asset_id = first_frame.asset_id

        if resolved_asset_id is None:
            raise RuntimeError("Segmentation requires --asset-id or --frame-ids.")

        if resolved_run_id is None:
            asset = repository.get_asset(resolved_asset_id, project_id=project_id)
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
        project_key: Optional[str] = None,
        compute_checksum: bool = True,
    ) -> tuple[str, str, str]:
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to register video assets.")

        project_id = _project_id_from_key(context, project_key)
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
        size_bytes = (
            _path_size_bytes(resolved)
            if compute_checksum
            else _queued_path_size_bytes(resolved, stat.st_size)
        )

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
                    "project_id": project_id,
                },
                assets=[
                    RawAssetManifest(
                        asset_id=resolved_asset_id,
                        filename=filename,
                        path=str(resolved),
                        kind=asset_kind,
                        size_bytes=size_bytes,
                        checksum=checksum,
                        collections=normalized_collections,
                        metadata={
                            "cli_command": cli_command,
                            "checksum_status": "computed" if compute_checksum else "deferred",
                            "project_id": project_id,
                        },
                    )
                ],
            )
        )
        context.repository.register_planned_run(planned_run, project_id=project_id)
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
        project_key: Optional[str] = None,
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
            project_key=project_key,
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
    def init_kvstore(directory: Path, store_name: str) -> None:
        config = CoreConfig.load()
        store = create_named_kvstore(directory, store_name, config.kvstore)
        if not store.initialized:
            initialize_kvstore(store, config.kvstore)
        typer.echo(f"KVStore ready at {store.root_path}")

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
            "build": build_info(),
            "database": {
                "schema": context.repository.schema if context.repository is not None else None,
                "initialized": context.repository is not None,
                "schema_status": (
                    context.repository.schema_status()
                    if context.repository is not None
                    else None
                ),
            },
            "kvstore": context.kvstore.status() if context.kvstore is not None else None,
        }
        typer.echo(json.dumps(json_ready(result), indent=2, sort_keys=True))

    @app.command("migrate-db")
    def migrate_db(
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        config = _config_from_options(None, database_dsn, schema)
        configure_core_logging(config)
        context = AppContext.from_config(config)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required for database migrations.")
        context.repository.initialize_schema()
        _echo_json(
            {
                "build": build_info(),
                "database": context.repository.schema_status(),
            }
        )

    @app.command("create-user")
    def create_user(
        username: str,
        password: Optional[str] = None,
        display_name: Optional[str] = None,
        admin: bool = False,
        inactive: bool = False,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema, initialize_schema=True)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to create users.")
        user = context.repository.create_user(
            username,
            password=password,
            display_name=display_name,
            is_admin=admin,
            is_active=not inactive,
        )
        _echo_json({"user": user})

    @app.command("create-project")
    def create_project(
        project_key: str,
        project_name: Optional[str] = None,
        description: Optional[str] = None,
        kvstore_directory: Optional[Path] = None,
        kvstore_name: Optional[str] = None,
        inactive: bool = False,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema, initialize_schema=True)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to create projects.")
        directory = kvstore_directory or context.config.kvstore.directory
        store_name = kvstore_name or project_key
        project = context.repository.create_project(
            project_key,
            project_name=project_name,
            description=description,
            kvstore_root_path=str(named_kvstore_path(directory, store_name)),
            is_active=not inactive,
            metadata={"kvstore": {"directory": str(directory), "store_name": store_name}},
        )
        kvstore = initialize_project_kvstore(context, project)
        _echo_json({"project": project, "kvstore": kvstore})

    @app.command("add-project-user")
    def add_project_user(
        username: str,
        project_key: str,
        role: str = "editor",
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema, initialize_schema=True)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to update project membership.")
        user = context.repository.get_user_by_username(username)
        if user is None:
            raise RuntimeError(f"User {username!r} was not found.")
        project = context.repository.get_project_by_key(project_key)
        if project is None:
            raise RuntimeError(f"Project {project_key!r} was not found.")
        membership = context.repository.add_project_member(str(user["id"]), str(project["id"]), role=role)
        _echo_json({"user": user, "project": project, "membership": membership})

    @app.command("list-projects")
    def list_projects(
        username: Optional[str] = None,
        all_projects: bool = False,
        limit: int = 100,
        offset: int = 0,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema, initialize_schema=True)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to list projects.")
        user = context.repository.get_user_by_username(username) if username else None
        if username and user is None:
            raise RuntimeError(f"User {username!r} was not found.")
        projects = (
            context.repository.list_user_projects(str(user["id"]))
            if user is not None and not all_projects
            else context.repository.list_projects(active_only=not all_projects, limit=limit, offset=offset)
        )
        _echo_json({"count": len(projects), "projects": projects})

    @app.command("create-dev-login")
    def create_dev_login(
        username: Optional[str] = None,
        password: Optional[str] = None,
        project_key: Optional[str] = None,
        project_name: Optional[str] = None,
        role: str = "admin",
        ttl_seconds: Optional[int] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema, initialize_schema=True)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to create a dev login.")
        auth_config = context.config.auth
        resolved_username = username or auth_config.bootstrap_admin_username or "dev-admin"
        resolved_password = password or auth_config.bootstrap_admin_password or "pelagia-dev"
        resolved_project_key = project_key or auth_config.dev_project_key
        user = context.repository.get_user_by_username(resolved_username)
        user_created = user is None
        if user is None:
            user = context.repository.create_user(
                resolved_username,
                password=resolved_password,
                display_name=auth_config.bootstrap_admin_display_name or "Pelagia Dev Admin",
                is_admin=True,
            )
        if resolved_project_key is None:
            _echo_json(
                {
                    "username": resolved_username,
                    "password": resolved_password if user_created else None,
                    "password_applied": user_created,
                    "project_creation_required": True,
                    "project": None,
                    "kvstore": None,
                    "membership": None,
                    "token": None,
                    "session": None,
                }
            )
            return
        project = context.repository.get_project_by_key(resolved_project_key)
        project_kvstore = None
        if project is None:
            store_name = resolved_project_key
            project = context.repository.create_project(
                resolved_project_key,
                project_name=project_name or resolved_project_key.title(),
                kvstore_root_path=str(
                    named_kvstore_path(context.config.kvstore.directory, store_name)
                ),
                metadata={
                    "kvstore": {
                        "directory": str(context.config.kvstore.directory),
                        "store_name": store_name,
                    }
                },
            )
            project_kvstore = initialize_project_kvstore(context, project)
        membership = context.repository.add_project_member(str(user["id"]), str(project["id"]), role=role)
        session = context.repository.create_session(
            str(user["id"]),
            str(project["id"]),
            ttl_seconds=ttl_seconds or auth_config.session_ttl_seconds,
            metadata={"cli_command": "create-dev-login"},
        )
        _echo_json(
            {
                "username": resolved_username,
                "password": resolved_password if user_created else None,
                "password_applied": user_created,
                "project": project,
                "kvstore": project_kvstore,
                "membership": membership,
                "token": session["token"],
                "session": session,
            }
        )

    @app.command("check-system")
    def check_system(
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema)
        database_status = (
            context.repository.schema_status()
            if context.repository is not None
            else {"ready": False, "missing_tables": ["repository"]}
        )
        projects = _list_all_projects(context.repository, active_only=True) if context.repository is not None else []
        project_stores = []
        for project in projects:
            store = context.kvstore_for_project(str(project["id"]), initialize=False)
            project_stores.append(
                {
                    "project_id": project["id"],
                    "project_key": project.get("project_key"),
                    "root_path": project.get("kvstore_root_path"),
                    "initialized": bool(store is not None and store.initialized),
                }
            )
        kvstore_status = {
            "required": bool(projects),
            "initialized": all(store["initialized"] for store in project_stores),
            "project_count": len(projects),
            "stores": project_stores,
        }
        ready = bool(database_status.get("ready")) and bool(kvstore_status.get("initialized"))
        result = {
            "build": build_info(),
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
                statuses = [JobStatus.QUEUED.value, JobStatus.LEASED.value, JobStatus.WORKING.value, JobStatus.PAUSED.value]
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
        project_key: Optional[str] = None,
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
            project_key,
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
        min_field_value: Optional[float] = None,
        max_field_value: Optional[float] = None,
        apply_mask: Optional[bool] = None,
        crop_enabled: Optional[bool] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
        encoding: Optional[str] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        project_key: Optional[str] = None,
    ) -> None:
        from ..workers.handlers import preprocess_frames_handler

        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to run preprocessing.")
        project_id = _project_id_from_key(context, project_key)

        payload = _preprocess_payload(
            frame_ids,
            start_frame,
            end_frame,
            limit,
            min_field_value,
            max_field_value,
            apply_mask,
            crop_enabled,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            encoding,
            context.config.processing,
        )
        resolved_run_id, resolved_asset_id = _resolve_segment_target(
            context.repository,
            run_id,
            asset_id,
            payload,
            project_id=project_id,
        )
        project_context = context.for_project(project_id)
        result = preprocess_frames_handler(
            {
                "id": "cli-preprocess",
                "project_id": project_id,
                "stage": PipelineStage.PREPROCESS_FRAMES.value,
                "run_id": resolved_run_id,
                "asset_id": resolved_asset_id,
                "payload": payload,
            },
            project_context,
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
        min_field_value: Optional[float] = None,
        max_field_value: Optional[float] = None,
        apply_mask: Optional[bool] = None,
        crop_enabled: Optional[bool] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
        encoding: Optional[str] = None,
        priority: Optional[int] = None,
        depends_on: Optional[str] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        project_key: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to queue preprocessing.")
        project_id = _project_id_from_key(context, project_key)

        payload = _preprocess_payload(
            frame_ids,
            start_frame,
            end_frame,
            limit,
            min_field_value,
            max_field_value,
            apply_mask,
            crop_enabled,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            encoding,
            context.config.processing,
        )
        resolved_run_id, resolved_asset_id = _resolve_segment_target(
            context.repository,
            run_id,
            asset_id,
            payload,
            project_id=project_id,
        )
        payload = PreprocessFramesCommand.from_payload(payload).to_payload()
        job = PipelineService(context).queue(
            PipelineStage.PREPROCESS_FRAMES,
            project_id=project_id,
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
        min_field_value: Optional[float] = None,
        max_field_value: Optional[float] = None,
        apply_mask: Optional[bool] = None,
        crop_enabled: Optional[bool] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
        min_perimeter: Optional[float] = None,
        max_perimeter: Optional[float] = None,
        padding: Optional[int] = None,
        roi_encoding: Optional[str] = None,
        zstd_min_bytes: Optional[int] = None,
        kvstore_root: Optional[Path] = None,
        database_dsn: Optional[str] = None,
        schema: Optional[str] = None,
        project_key: Optional[str] = None,
    ) -> None:
        from ..workers.handlers import roi_detection_handler

        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to run segmentation.")
        project_id = _project_id_from_key(context, project_key)

        payload = _segment_payload(
            frame_ids,
            start_frame,
            end_frame,
            limit,
            threshold,
            frame_payload_kind,
            apply_preprocessing,
            min_field_value,
            max_field_value,
            apply_mask,
            crop_enabled,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
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
            project_id=project_id,
        )
        project_context = context.for_project(project_id)
        result = roi_detection_handler(
            {
                "id": "cli-segment",
                "project_id": project_id,
                "stage": PipelineStage.SEGMENT.value,
                "run_id": resolved_run_id,
                "asset_id": resolved_asset_id,
                "payload": payload,
            },
            project_context,
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
        min_field_value: Optional[float] = None,
        max_field_value: Optional[float] = None,
        apply_mask: Optional[bool] = None,
        crop_enabled: Optional[bool] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_w: Optional[int] = None,
        crop_h: Optional[int] = None,
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
        project_key: Optional[str] = None,
    ) -> None:
        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to queue segmentation.")
        project_id = _project_id_from_key(context, project_key)

        payload = _segment_payload(
            frame_ids,
            start_frame,
            end_frame,
            limit,
            threshold,
            frame_payload_kind,
            apply_preprocessing,
            min_field_value,
            max_field_value,
            apply_mask,
            crop_enabled,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
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
            project_id=project_id,
        )
        payload = SegmentFramesCommand.from_payload(payload).to_payload()
        job = PipelineService(context).queue(
            PipelineStage.SEGMENT,
            project_id=project_id,
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
        project_key: Optional[str] = None,
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
            project_key=project_key,
            compute_checksum=compute_checksum,
        )
        if context.repository is None:
            raise RuntimeError("A PostgresRepository is required to queue frame extraction.")
        project_id = _project_id_from_key(context, project_key)
        job = PipelineService(context).queue(
            PipelineStage.EXTRACT_FRAMES,
            project_id=project_id,
            run_id=resolved_run_id,
            asset_id=resolved_asset_id,
            payload=ExtractFramesCommand.from_payload({
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
            }).to_payload(),
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
        project_key: Optional[str] = None,
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
        project_id = _project_id_from_key(context, project_key)

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
                project_key=project_key,
                compute_checksum=compute_checksum,
            )
            job = PipelineService(context).queue(
                PipelineStage.EXTRACT_FRAMES,
                project_id=project_id,
                run_id=resolved_run_id,
                asset_id=resolved_asset_id,
                payload=ExtractFramesCommand.from_payload({
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
                }).to_payload(),
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

        context = _context_from_options(kvstore_root, database_dsn, schema)
        if context.repository is None:
            raise RuntimeError("Reset requires a PostgresRepository.")
        config = context.config
        context.repository.initialize_schema()
        projects = _list_all_projects(context.repository, active_only=False)

        kvstore_results = []
        seen_roots: set[str] = set()
        for project in projects:
            root_path = project.get("kvstore_root_path")
            if not root_path:
                kvstore_results.append(
                    {
                        "project_id": project.get("id"),
                        "project_key": project.get("project_key"),
                        "root_path": None,
                        "status": "not_configured",
                    }
                )
                continue
            resolved_root = str(Path(root_path).expanduser().resolve(strict=False))
            if resolved_root in seen_roots:
                kvstore_results.append(
                    {
                        "project_id": project.get("id"),
                        "project_key": project.get("project_key"),
                        "root_path": resolved_root,
                        "status": "shared_store_already_reset",
                    }
                )
                continue
            seen_roots.add(resolved_root)
            store = create_kvstore(resolved_root, config.kvstore)
            if not store.initialized:
                kvstore_results.append(
                    {
                        "project_id": project.get("id"),
                        "project_key": project.get("project_key"),
                        "root_path": resolved_root,
                        "status": "missing",
                    }
                )
                continue
            try:
                reset_result = reset_kvstore(store, config.kvstore)
            except Exception as exc:
                kvstore_results.append(
                    {
                        "project_id": project.get("id"),
                        "project_key": project.get("project_key"),
                        "root_path": resolved_root,
                        "status": "error",
                        "error": str(exc),
                    }
                )
            else:
                kvstore_results.append(
                    {
                        "project_id": project.get("id"),
                        "project_key": project.get("project_key"),
                        "root_path": resolved_root,
                        "status": "reset",
                        "result": reset_result,
                    }
                )
        database_result = context.repository.purge_all()
        typer.echo(
            json.dumps(
                json_ready(
                    {
                        "deleted": True,
                        "database": database_result,
                        "kvstores": {
                            "project_count": len(projects),
                            "reset_count": sum(result["status"] == "reset" for result in kvstore_results),
                            "results": kvstore_results,
                        },
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
        from ..workers import Worker, default_handler_registry, worker_runtime_profile

        context = _context_from_options(kvstore_root, database_dsn, schema)
        selected_stages = None
        if stages:
            selected_stages = [
                PipelineStage(stage.strip())
                for stage in stages.split(",")
                if stage.strip()
            ]
        worker_runtime_profile(selected_stages)
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
        from ..workers import Worker, default_handler_registry, worker_runtime_profile

        context = _context_from_options(kvstore_root, database_dsn, schema)
        selected_stages = None
        if stages:
            selected_stages = [
                PipelineStage(stage.strip())
                for stage in stages.split(",")
                if stage.strip()
            ]
        worker_runtime_profile(selected_stages)
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
