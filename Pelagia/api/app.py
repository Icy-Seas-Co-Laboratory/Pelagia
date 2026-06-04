from __future__ import annotations

from ..config import CoreConfig
from ..observability import configure_core_logging
from ..services.context import AppContext
from .routes import assets, collections, detections, health, ingestion, jobs, kvstore, live, logs, models, runs, segmentation, system, workers


def create_app(config: CoreConfig | None = None):
    """Create the FastAPI app.

    FastAPI is optional during early development; importing this module raises a
    clear error if the dependency has not been installed.
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise RuntimeError("Install FastAPI to run the Pelagia API.") from exc

    resolved_config = config or CoreConfig.load()
    core_logger = configure_core_logging(resolved_config)
    core_logger.info("Starting Pelagia API")
    app = FastAPI(title="Pelagia", version="0.0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = resolved_config
    app.state.context = AppContext.from_config(resolved_config)
    for route_module in (
        health,
        system,
        ingestion,
        segmentation,
        runs,
        jobs,
        workers,
        assets,
        detections,
        collections,
        live,
        kvstore,
        logs,
        models,
    ):
        if route_module.router is not None:
            app.include_router(route_module.router)
    return app
