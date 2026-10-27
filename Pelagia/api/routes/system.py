from __future__ import annotations

from typing import Any
from uuid import UUID

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_admin, require_auth
    from ..schemas import OptionsResponse, SystemCapabilitiesResponse
    from ...config import CoreConfig
    from ...processing.capabilities import preprocessing_capabilities, system_capabilities
    from ...services.system_usage import SystemUsageService
    from ...version import __version__, build_info
    from ._common import as_response, get_context, get_repository, kvstore_status, postgres_ping

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
                "version": __version__,
                "build": build_info(),
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
                "storage": {
                    "cold": {
                        "backend": "skerrykv",
                        "endpoint": config.cold_storage.endpoint,
                        "timeout_seconds": config.cold_storage.timeout_seconds,
                    },
                    "hot": {
                        "backend": "skerrykv",
                        "endpoint": config.hot_storage.endpoint,
                        "timeout_seconds": config.hot_storage.timeout_seconds,
                    },
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
        context = get_context(request)
        cold_storage = context.kvstore
        hot_storage = context.hot_storage
        postgres = {"healthy": False}
        try:
            postgres = postgres_ping(repository)
        except Exception as exc:
            postgres = {"healthy": False, "error": str(exc), "schema": repository.schema}
        queue = repository.get_status_summary()
        database_status = repository.schema_status()
        return as_response(
            {
                "build": build_info(),
                "postgres": postgres,
                "database": database_status,
                "storage": {
                    "cold": (
                        kvstore_status(cold_storage, deep=deep_kvstore)
                        if cold_storage is not None
                        else {"configured": False, "initialized": False}
                    ),
                    "hot": (
                        kvstore_status(hot_storage, deep=deep_kvstore)
                        if hot_storage is not None
                        else {"configured": False, "initialized": False}
                    ),
                },
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

        cold_storage = context.kvstore_for_project(project_id, initialize=False)
        hot_storage = context.hot_storage or cold_storage
        if cold_storage is None:
            raise HTTPException(status_code=503, detail="SkerryKV storage is not configured.")
        postgres = {"healthy": False}
        try:
            postgres = postgres_ping(repository)
        except Exception as exc:
            postgres = {"healthy": False, "error": str(exc), "schema": repository.schema}
        queue = repository.get_status_summary(project_id=project_id)
        database_status = repository.schema_status()
        return as_response(
            {
                "build": build_info(),
                "project": project,
                "postgres": postgres,
                "database": database_status,
                "storage": {
                    "cold": kvstore_status(cold_storage, deep=deep_kvstore),
                    "hot": kvstore_status(hot_storage, deep=deep_kvstore),
                },
                "queue": queue.get("queue", {}),
                "workers": queue.get("workers", {}),
            }
        )

    def _system_discovery() -> dict:
        return {
            "capabilities": [
                "register video assets and queue frame extraction",
                "run synchronous thresholding and candidate detection for stored frames",
                "inspect and control queued jobs",
                "monitor worker sessions and request worker shutdown",
                "inspect SkerryKV and PostgreSQL health",
                "browse runs, assets, frames, detections, and registered models",
            ],
            "common_flows": {
                "analyze_ingestion_source": "POST /ingestion/analyze",
                "queue_analyzed_assets": "POST /ingestion/assets",
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
                "system_usage": "GET /system/usage",
            },
        }

    def _system_usage(request: Request) -> dict:
        require_auth(request)
        return as_response(SystemUsageService(get_context(request)).snapshot())

    @router.get("/usage")
    def get_system_usage(request: Request) -> dict:
        return _system_usage(request)

    @router.get("/use")
    def get_system_use(request: Request) -> dict:
        return {**_system_usage(request), **_system_discovery()}

    @router.post("/initialize")
    def initialize_system(request: Request) -> dict:
        context = get_context(request)
        initialized: dict[str, bool] = {"postgres": False, "cold_storage": False, "hot_storage": False}
        if context.repository is not None:
            context.repository.initialize_schema()
            initialized["postgres"] = True
        initialized["cold_storage"] = context.kvstore is not None
        initialized["hot_storage"] = context.hot_storage is not None
        if context.repository is None:
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
