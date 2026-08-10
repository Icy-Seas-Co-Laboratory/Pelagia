from __future__ import annotations

from typing import TYPE_CHECKING

from .version import __version__, build_info

if TYPE_CHECKING:
    from .config import CoreConfig

__all__ = ["CoreConfig", "__version__", "build_info"]


def __getattr__(name: str):
    if name == "CoreConfig":
        from .config import CoreConfig

        return CoreConfig
    raise AttributeError(name)
