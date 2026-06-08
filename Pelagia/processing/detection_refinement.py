from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..domain import DetectionRecord
from .frame_codec import decode_array_payload, encode_array_payload
from .frame_preprocess import as_binary_mask, as_grayscale_array


RefinementFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(slots=True)
class DetectionRefinementResult:
    """Runtime result for moving a candidate ROI mask toward a refined ROI mask."""

    candidate_detection: DetectionRecord
    roi: np.ndarray
    candidate_mask: np.ndarray
    refined_mask: np.ndarray
    method: str = "identity"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_detection_record(self, *, encoding: str | None = None) -> DetectionRecord:
        """Return a copy of the candidate record annotated as refined metadata."""
        metadata = dict(self.candidate_detection.metadata or {})
        metadata.update(self.metadata)
        metadata.update(
            {
                "detection_stage": "refined",
                "candidate_detection_id": self.candidate_detection.id,
                "refinement_method": self.method,
            }
        )
        requested_encoding = encoding or self.candidate_detection.roi_encoding or "png"
        roi_payload, roi_encoding, roi_format = encode_array_payload(self.roi, requested_encoding)
        mask_payload, mask_encoding, mask_format = encode_array_payload(self.refined_mask, requested_encoding)
        return replace(
            self.candidate_detection,
            roi_payload=roi_payload,
            mask_payload=mask_payload,
            roi_encoding=roi_encoding,
            roi_format=roi_format,
            roi_dtype=str(self.roi.dtype),
            roi_shape=list(self.roi.shape),
            mask_encoding=mask_encoding,
            mask_format=mask_format,
            mask_dtype=str(self.refined_mask.dtype),
            mask_shape=list(self.refined_mask.shape),
            metadata=metadata,
        )


def decode_detection_roi(detection: DetectionRecord) -> np.ndarray:
    """Decode a stored detection ROI payload into a contiguous numpy array."""
    if detection.roi_payload is None:
        raise ValueError("Detection does not include ROI payload data.")
    metadata = {
        "array_encoding": detection.roi_encoding,
        "dtype": detection.roi_dtype,
        "shape": detection.roi_shape,
    }
    return np.ascontiguousarray(decode_array_payload(detection.roi_payload, metadata))


def decode_detection_candidate_mask(detection: DetectionRecord) -> np.ndarray:
    """Decode the candidate ROI mask, or derive one from non-zero ROI pixels."""
    if detection.mask_payload is not None:
        metadata = {
            "array_encoding": detection.mask_encoding,
            "dtype": detection.mask_dtype,
            "shape": detection.mask_shape,
        }
        return as_binary_mask(decode_array_payload(detection.mask_payload, metadata))

    roi = decode_detection_roi(detection)
    return as_binary_mask(as_grayscale_array(roi))


def refine_detection(
    detection: DetectionRecord,
    *,
    refiner: RefinementFn | None = None,
    method: str | None = None,
) -> DetectionRefinementResult:
    """
    Refine one candidate detection ROI.

    ``refiner`` is intentionally model-shaped for a future U-net hook: it receives
    ``(roi, candidate_mask)`` and returns a refined mask-like array.
    """
    roi = decode_detection_roi(detection)
    candidate_mask = decode_detection_candidate_mask(detection)
    if candidate_mask.shape[:2] != roi.shape[:2]:
        raise ValueError(
            f"Candidate mask shape {candidate_mask.shape[:2]} does not match ROI shape {roi.shape[:2]}."
        )

    if refiner is None:
        refined_mask = candidate_mask
        resolved_method = method or "identity"
    else:
        refined_mask = as_binary_mask(refiner(roi, candidate_mask))
        resolved_method = method or getattr(refiner, "__name__", "custom_refiner")
    refined_mask = as_binary_mask(refined_mask)
    if refined_mask.shape[:2] != roi.shape[:2]:
        raise ValueError(
            f"Refined mask shape {refined_mask.shape[:2]} does not match ROI shape {roi.shape[:2]}."
        )

    return DetectionRefinementResult(
        candidate_detection=detection,
        roi=roi,
        candidate_mask=candidate_mask,
        refined_mask=refined_mask,
        method=resolved_method,
        metadata={
            "candidate_mask_kind": "candidate",
            "refined_mask_kind": "refined",
            "candidate_shape": list(candidate_mask.shape),
            "refined_shape": list(refined_mask.shape),
        },
    )


def refine_detections(
    detections: Iterable[DetectionRecord],
    *,
    refiner: RefinementFn | None = None,
    method: str | None = None,
) -> list[DetectionRefinementResult]:
    """Refine a batch of candidate detections."""
    return [
        refine_detection(detection, refiner=refiner, method=method)
        for detection in detections
    ]
