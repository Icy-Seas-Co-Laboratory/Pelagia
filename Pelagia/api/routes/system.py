from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ...config import CoreConfig
    from ...processing.capabilities import preprocessing_capabilities, system_capabilities
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
                    "root_path": config.kvstore.root_path,
                    "hash_algorithm": config.kvstore.hash_algorithm,
                    "prefix_length": config.kvstore.prefix_length,
                    "max_db_bytes": config.kvstore.max_db_bytes,
                    "max_rows": config.kvstore.max_rows,
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

    @router.get("/capabilities")
    def get_system_capabilities(request: Request) -> dict:
        return as_response(system_capabilities(get_context(request).config))

    @preprocessing_router.get("/options")
    def get_preprocessing_options(request: Request) -> dict:
        return as_response(preprocessing_capabilities(get_context(request).config.processing))

    @router.get("/status")
    def get_system_status(request: Request) -> dict:
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
                "kvstore": kvstore_status(kvstore),
                "queue": queue.get("queue", {}),
                "workers": queue.get("workers", {}),
            }
        )

    @router.get("/use")
    def get_system_use() -> dict:
        return {
            "capabilities": [
                "register video assets and queue frame extraction",
                "run synchronous segmentation for stored frames",
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
                "live_segmentation": "GET /live/segmentation",
                "segment_frame_now": "POST /segmentation/frames/{frame_id}",
                "queue_job": "POST /jobs",
                "worker_status": "GET /workers",
                "system_status": "GET /system/status",
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
            config = context.config.kvstore
            context.kvstore.initialize(
                hash_algorithm=config.hash_algorithm,
                prefix_length=config.prefix_length,
                max_db_bytes=config.max_db_bytes,
                max_rows=config.max_rows,
            )
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
