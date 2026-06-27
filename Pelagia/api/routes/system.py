from __future__ import annotations

from typing import Any
from uuid import UUID

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_auth
    from ..schemas import OptionsResponse, SystemCapabilitiesResponse
    from ...config import CoreConfig
    from ...processing.capabilities import preprocessing_capabilities, system_capabilities
    from ...storage.blob_store import initialize_kvstore
    from ._common import as_response, get_context, get_kvstore, get_repository, kvstore_status, postgres_ping

    router = APIRouter(prefix="/system", tags=["system"])
    preprocessing_router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])
    routers = [preprocessing_router]

    @router.get("")
    def get_system(request: Request) -> dict:
        context = get_context(request)
        config = context.config
        return as_response(
            {
                "name": "Pelagia",
                "version": "0.0.1",
                "database": {
                    "schema": config.database.schema_name,
                    "connect_timeout_s": config.database.connect_timeout_s,
                    "statement_timeout_ms": config.database.statement_timeout_ms,
                },
                "queue": {
                    "default_priority": config.queue.default_priority,
                    "max_attempts": config.queue.max_attempts,
                    "max_claim_count": config.queue.max_claim_count,
                    "lease_seconds": config.queue.lease_seconds,
                    "heartbeat_interval_seconds": config.queue.heartbeat_interval_seconds,
                },
                "kvstore": {
                    "backend": config.kvstore.backend,
                    "root_path": config.kvstore.root_path,
                    "hash_algorithm": config.kvstore.hash_algorithm,
                    "prefix_length": config.kvstore.prefix_length,
                    "max_db_bytes": config.kvstore.max_db_bytes,
                    "max_rows": config.kvstore.max_rows,
                    "max_blob_bytes": config.kvstore.max_blob_bytes,
                },
                "image_data_storage": {
                    "encoding": config.image_data_storage.encoding,
                },
                "logging": {
                    "log_path": config.logging.log_path,
                    "file_name": config.logging.file_name,
                    "level": config.logging.level,
                    "console": config.logging.console,
                    "max_bytes": config.logging.max_bytes,
                    "backup_count": config.logging.backup_count,
                },
            }
        )

    @router.get("/config")
    def get_system_config(request: Request) -> dict:
        context = get_context(request)
        defaults = CoreConfig.load(local_config_path=None, use_env=False)
        return as_response(
            {
                "effective": context.config,
                "defaults": defaults,
            }
        )

    @router.get("/capabilities", response_model=SystemCapabilitiesResponse)
    def get_system_capabilities(request: Request) -> dict:
        return as_response(system_capabilities(get_context(request).config))

    @preprocessing_router.get("/options", response_model=OptionsResponse)
    def get_preprocessing_options(request: Request) -> dict:
        return as_response(preprocessing_capabilities(get_context(request).config.processing))

    @router.get("/status")
    def get_system_status(request: Request, deep_kvstore: bool = False) -> dict:
        repository = get_repository(request)
        kvstore = get_kvstore(request)
        postgres = {"healthy": False}
        try:
            postgres = postgres_ping(repository)
        except Exception as exc:
            postgres = {"healthy": False, "error": str(exc), "schema": repository.schema}
        queue = repository.get_status_summary()
        return as_response(
            {
                "postgres": postgres,
                "kvstore": kvstore_status(kvstore, deep=deep_kvstore),
                "queue": queue.get("queue", {}),
                "workers": queue.get("workers", {}),
            }
        )

    def _is_uuid(value: str | None) -> bool:
        if not value:
            return False
        try:
            UUID(str(value))
        except ValueError:
            return False
        return True

    def _project_by_id_or_key(repository, project_ref: str) -> dict[str, Any] | None:
        if _is_uuid(project_ref):
            project = repository.get_project(project_ref)
            if project is not None:
                return project
        return repository.get_project_by_key(project_ref)

    @router.get("/status/{project_ref}")
    def get_project_system_status(request: Request, project_ref: str, deep_kvstore: bool = False) -> dict:
        auth = require_auth(request)
        context = get_context(request)
        repository = get_repository(request)
        project = _project_by_id_or_key(repository, project_ref)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project {project_ref!r} was not found.")
        if not project.get("is_active", True):
            raise HTTPException(status_code=404, detail=f"Project {project_ref!r} was not found.")
        project_id = str(project["id"])
        if not auth.is_admin and repository.get_project_membership(auth.user_id, project_id) is None:
            raise HTTPException(status_code=403, detail="Project read permission is required.")

        kvstore = context.kvstore_for_project(project_id, initialize=False)
        if kvstore is None:
            raise HTTPException(status_code=503, detail="KVStore is not configured.")
        postgres = {"healthy": False}
        try:
            postgres = postgres_ping(repository)
        except Exception as exc:
            postgres = {"healthy": False, "error": str(exc), "schema": repository.schema}
        queue = repository.get_status_summary(project_id=project_id)
        return as_response(
            {
                "project": project,
                "postgres": postgres,
                "kvstore": kvstore_status(kvstore, deep=deep_kvstore),
                "queue": queue.get("queue", {}),
                "workers": queue.get("workers", {}),
            }
        )

    @router.get("/use")
    def get_system_use() -> dict:
        return {
            "capabilities": [
                "register video assets and queue frame extraction",
                "run synchronous thresholding and candidate detection for stored frames",
                "inspect and control queued jobs",
                "monitor worker sessions and request worker shutdown",
                "inspect KVStore and PostgreSQL health",
                "browse runs, assets, frames, detections, and registered models",
            ],
            "common_flows": {
                "queue_video_ingestion": "POST /ingestion/videos",
                "system_capabilities": "GET /system/capabilities",
                "preprocessing_options": "GET /preprocessing/options",
                "segmentation_options": "GET /segmentation/options",
                "preprocess_frame_now": "POST /frame/preprocess",
                "queue_preprocessing": "POST /frame/preprocess/jobs",
                "live_threshold": "GET /live/threshold",
                "live_detection_candidate": "GET /live/detection-candidate",
                "live_sandbox": "GET /live/sandbox",
                "delete_live_sandbox": "DELETE /live/sandbox/{sandbox_frame_id}",
                "segment_frame_now": "POST /segmentation/frames/{frame_id}",
                "queue_job": "POST /jobs",
                "worker_status": "GET /workers",
                "system_status": "GET /system/status",
                "project_system_status": "GET /system/status/{project_id_or_key}",
            },
        }

    @router.post("/initialize")
    def initialize_system(request: Request) -> dict:
        context = get_context(request)
        initialized: dict[str, bool] = {"postgres": False, "kvstore": False}
        if context.repository is not None:
            context.repository.initialize_schema()
            initialized["postgres"] = True
        if context.kvstore is not None and not context.kvstore.initialized:
            initialize_kvstore(context.kvstore, context.config.kvstore)
            initialized["kvstore"] = True
        if context.repository is None and context.kvstore is None:
            raise HTTPException(status_code=503, detail="No system stores are configured.")
        if context.logger is not None and initialized["postgres"]:
            context.logger.info(
                "system.initialized",
                "Pelagia stores initialized",
                payload=initialized,
            )
        return initialized
else:
    router = None
    preprocessing_router = None
    routers = []
