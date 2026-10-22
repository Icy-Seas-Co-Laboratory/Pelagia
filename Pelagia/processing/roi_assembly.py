"""Assemble binary-mask components into stable candidate ROIs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import time
from typing import Any

import cv2
import numpy as np

from .mask_augmentation import as_binary_mask
from .roi_filter import component_statistics_pass_filters
from .timing import measure_phase


@dataclass(slots=True)
class CandidateRoi:
    """A candidate object assembled from a binary mask."""

    roi_index: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    area: float
    perimeter: float
    mask: np.ndarray
    contour: np.ndarray
    mask_x: int = 0
    mask_y: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width_plus_height(self) -> int:
        return int(self.bbox_w + self.bbox_h)

    @property
    def bbox_perimeter(self) -> int:
        return int((2 * self.bbox_w) + (2 * self.bbox_h))


def assemble_candidate_rois(
    mask: np.ndarray,
    *,
    method: str = "connected_components",
    connectivity: int = 8,
    filter_options: dict[str, int | float | None] | None = None,
    assembly_stats: dict[str, int | float] | None = None,
) -> list[CandidateRoi]:
    """Assemble candidate ROI objects from a binary mask."""
    resolved_method = str(method).replace("-", "_").lower()
    binary = as_binary_mask(mask)
    if resolved_method in {"connected_components", "components", "connected_component"}:
        return assemble_connected_components(
            binary,
            connectivity=connectivity,
            filter_options=filter_options,
            assembly_stats=assembly_stats,
        )
    if resolved_method in {"contours", "find_contours"}:
        with (
            measure_phase("segmentation.contour_assembly")
            if assembly_stats is not None
            else nullcontext()
        ):
            candidates = assemble_contours(binary)
        if assembly_stats is not None:
            assembly_stats.update(
                source_component_count=len(candidates),
                materialized_candidate_count=len(candidates),
                filtered_before_materialization_count=0,
            )
        return candidates
    raise ValueError(f"Unsupported ROI assembly method {method!r}.")


def assemble_connected_components(
    mask: np.ndarray,
    *,
    connectivity: int = 8,
    filter_options: dict[str, int | float | None] | None = None,
    assembly_stats: dict[str, int | float] | None = None,
) -> list[CandidateRoi]:
    """Assemble ROIs using OpenCV connected components."""
    binary = as_binary_mask(mask)
    label_started = time.perf_counter()
    with (
        measure_phase("segmentation.connected_component_labeling")
        if assembly_stats is not None
        else nullcontext()
    ):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8),
            connectivity=int(connectivity),
        )
    label_duration_ms = (time.perf_counter() - label_started) * 1000
    source_component_count = int(num - 1)
    options = filter_options or {}
    candidates: list[CandidateRoi] = []
    materialization_started = time.perf_counter()
    with (
        measure_phase("segmentation.connected_component_materialization")
        if assembly_stats is not None
        else nullcontext()
    ):
        for label in range(1, num):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            # Candidate area has historically been its bounding-box area, not
            # its foreground-pixel count. Keep that public semantics here.
            area = float(width * height)
            if not component_statistics_pass_filters(
                area=area,
                width=width,
                height=height,
                **options,
            ):
                continue
            # Candidate masks are bbox-local even though bbox coordinates are frame-relative.
            component_mask = np.ascontiguousarray(
                (labels[y:y + height, x:x + width] == label).astype(np.uint8) * 255
            )
            contour = _largest_contour(component_mask)
            if contour is None:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            candidates.append(
                CandidateRoi(
                    roi_index=len(candidates) + 1,
                    bbox_x=x,
                    bbox_y=y,
                    bbox_w=width,
                    bbox_h=height,
                    area=area,
                    perimeter=perimeter,
                    mask=np.ascontiguousarray(component_mask),
                    contour=contour,
                    mask_x=x,
                    mask_y=y,
                    metadata={
                        "assembly_method": "connected_components",
                        "component_label": int(label),
                        "mask_coordinate_space": "bbox_local",
                    },
                )
            )
    if assembly_stats is not None:
        assembly_stats.update(
            source_component_count=source_component_count,
            materialized_candidate_count=len(candidates),
            filtered_before_materialization_count=source_component_count - len(candidates),
            connected_component_labeling_ms=label_duration_ms,
            connected_component_materialization_ms=(time.perf_counter() - materialization_started) * 1000,
        )
    return candidates


def assemble_contours(mask: np.ndarray) -> list[CandidateRoi]:
    """Assemble ROIs from external contours."""
    binary = as_binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[CandidateRoi] = []
    for contour in contours:
        if contour.size == 0:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        # Shift only the contour used to draw the local mask; retain the frame contour below.
        local_contour = np.ascontiguousarray(contour - np.array([[[x, y]]], dtype=contour.dtype))
        component_mask = np.zeros((int(height), int(width)), dtype=np.uint8)
        cv2.drawContours(component_mask, [local_contour], -1, 255, thickness=cv2.FILLED)
        candidates.append(
            CandidateRoi(
                roi_index=len(candidates) + 1,
                bbox_x=int(x),
                bbox_y=int(y),
                bbox_w=int(width),
                bbox_h=int(height),
                area=float(cv2.contourArea(contour)),
                perimeter=float(cv2.arcLength(contour, True)),
                mask=np.ascontiguousarray(component_mask),
                contour=np.ascontiguousarray(contour),
                mask_x=int(x),
                mask_y=int(y),
                metadata={
                    "assembly_method": "contours",
                    "mask_coordinate_space": "bbox_local",
                },
            )
        )
    return candidates


def renumber_candidate_rois(candidates: list[CandidateRoi]) -> list[CandidateRoi]:
    """Return candidates with stable one-based ROI indices."""
    for index, candidate in enumerate(candidates, start=1):
        candidate.roi_index = index
    return candidates


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return np.ascontiguousarray(max(contours, key=cv2.contourArea))
