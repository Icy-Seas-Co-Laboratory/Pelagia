from __future__ import annotations

from ..domain import ClassificationResultRecord


def classify_detection(detection_id: str, model_id: str, roi_payload: bytes) -> ClassificationResultRecord:
    """Template classification routine for one detected crop."""
    raise NotImplementedError("Implement project-specific classification here.")
