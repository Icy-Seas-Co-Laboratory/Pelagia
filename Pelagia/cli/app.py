from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import CoreConfig
from ..services.stores import StoreService


try:
    import typer
except ImportError:  # pragma: no cover - argparse fallback covers no-typer envs
    typer = None


if typer is not None:
    app = typer.Typer(help="Pelagia command line tools.")

    @app.command("init-kvstore")
    def init_kvstore(root: Optional[Path] = None) -> None:
        config = CoreConfig.from_env()
        if root is not None:
            config.kvstore.root_path = root
        service = StoreService.from_config(config.kvstore)
        service.ensure_initialized(config.kvstore)
        typer.echo(f"KVStore ready at {service.store.root_path}")

    def main() -> None:
        app()
else:
    app = None

    def main() -> None:
        raise RuntimeError("Install typer to run the Pelagia CLI.")
