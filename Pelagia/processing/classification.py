from __future__ import annotations

from ..domain import ClassificationResultRecord
from ._logging import processing_core_logger


_CORE_LOGGER = processing_core_logger("classification")


def classify_detection(detection_id: str, model_id: str, roi_payload: bytes) -> ClassificationResultRecord:
    """Template classification routine for one detected crop."""
    _CORE_LOGGER.warning(
        "Classification routine is not implemented detection_id=%s model_id=%s payload_bytes=%s",
        detection_id,
        model_id,
        len(roi_payload),
    )
    raise NotImplementedError("Implement project-specific classification here.")
