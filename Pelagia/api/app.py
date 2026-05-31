from __future__ import annotations

from ..config import CoreConfig
from .routes import assets, health, jobs, models, runs


def create_app(config: CoreConfig | None = None):
    """Create the FastAPI app.

    FastAPI is optional during early development; importing this module raises a
    clear error if the dependency has not been installed.
    """
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install FastAPI to run the Pelagia API.") from exc

    app = FastAPI(title="Pelagia", version="0.0.1")
    app.state.config = config or CoreConfig.load()
    for route_module in (health, runs, jobs, assets, models):
        if route_module.router is not None:
            app.include_router(route_module.router)
    return app
