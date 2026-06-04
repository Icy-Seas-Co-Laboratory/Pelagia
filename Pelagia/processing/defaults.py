from __future__ import annotations

from ..config import CoreConfig, ProcessingConfig


def default_processing_config() -> ProcessingConfig:
    """Return processing defaults for direct calls without an AppContext."""
    return CoreConfig.load().processing
