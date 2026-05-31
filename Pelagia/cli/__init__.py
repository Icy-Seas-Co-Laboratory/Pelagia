"""Command-line interface package."""

__all__ = ["app", "main"]


def main() -> None:
    from .app import main as app_main

    app_main()


def __getattr__(name: str):
    if name == "app":
        from .app import app

        return app
    raise AttributeError(name)
