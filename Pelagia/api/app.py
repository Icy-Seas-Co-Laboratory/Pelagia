from __future__ import annotations

from ..config import CoreConfig
from ..services.context import AppContext
from .routes import assets, collections, health, ingestion, jobs, kvstore, models, runs, segmentation, system, workers


def create_app(config: CoreConfig | None = None):
    """Create the FastAPI app.

    FastAPI is optional during early development; importing this module raises a
    clear error if the dependency has not been installed.
    """
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install FastAPI to run the Pelagia API.") from exc

    resolved_config = config or CoreConfig.load()
    app = FastAPI(title="Pelagia", version="0.0.1")
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
        collections,
        kvstore,
        models,
    ):
        if route_module.router is not None:
            app.include_router(route_module.router)
    return app
