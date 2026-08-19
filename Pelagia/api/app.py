from __future__ import annotations

from contextlib import asynccontextmanager

from ..config import CoreConfig
from ..observability import configure_core_logging
from ..services.context import AppContext
from ..version import __version__
from .routes import assets, auth, collections, curation, detections, frame, health, ingestion, io, jobs, kvstore, live, live_preview, live_sandbox, logs, models, processing, processing_status, registry, roi_refinement, runs, segmentation, system, workers


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
    context = AppContext.from_config(resolved_config)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            context.close()

    app = FastAPI(
        title="Pelagia",
        version=__version__,
        lifespan=lifespan,
        root_path=resolved_config.api.root_path,
    )
    exposed_headers = [
        "X-Pelagia-Frame-Id",
        "X-Pelagia-Payload-Kind",
        "X-Pelagia-Source-Width",
        "X-Pelagia-Source-Height",
        "X-Pelagia-Image-Width",
        "X-Pelagia-Image-Height",
        "X-Pelagia-Scale-X",
        "X-Pelagia-Scale-Y",
        "X-Pelagia-Width",
        "X-Pelagia-Height",
        "X-Pelagia-Resize-Width",
        "X-Pelagia-Resize-Height",
        "X-Pelagia-Scale",
    ]
    app.state.config = resolved_config
    app.state.context = context
    for route_module in (
        health,
        auth,
        system,
        ingestion,
        io,
        segmentation,
        runs,
        jobs,
        processing,
        processing_status,
        workers,
        frame,
        assets,
        detections,
        collections,
        live,
        live_preview,
        live_sandbox,
        kvstore,
        logs,
        models,
        curation,
        roi_refinement,
        registry,
    ):
        if route_module.router is not None:
            app.include_router(route_module.router)
        for extra_router in getattr(route_module, "routers", []):
            if extra_router is not None:
                app.include_router(extra_router)
    cors_app = CORSMiddleware(
        app,
        allow_origin_regex=resolved_config.api.cors_allow_origin_regex,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=exposed_headers,
    )
    cors_app.state = app.state
    return cors_app
