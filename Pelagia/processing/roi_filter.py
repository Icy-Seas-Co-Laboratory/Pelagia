"""Apply geometric and payload-storage filters to candidate ROIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .roi_assembly import CandidateRoi


def filter_candidate_rois(
    candidates: list[CandidateRoi],
    *,
    min_area: int | float | None = None,
    max_area: int | float | None = None,
    min_perimeter: int | float | None = None,
    max_perimeter: int | float | None = None,
    min_width: int | float | None = None,
    max_width: int | float | None = None,
    min_height: int | float | None = None,
    max_height: int | float | None = None,
    min_width_plus_height: int | float | None = None,
    max_width_plus_height: int | float | None = None,
    reindex: bool = True,
) -> list[CandidateRoi]:
    """Filter assembled candidate ROIs by explicit geometric criteria."""
    kept = [
        candidate
        for candidate in candidates
        if candidate_passes_filters(
            candidate,
            min_area=min_area,
            max_area=max_area,
            min_perimeter=min_perimeter,
            max_perimeter=max_perimeter,
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            min_width_plus_height=min_width_plus_height,
            max_width_plus_height=max_width_plus_height,
        )
    ]
    if not reindex:
        return kept
    # Import lazily so connected-component assembly can share the statistics
    # predicate without making this module part of an import cycle.
    from .roi_assembly import renumber_candidate_rois

    return renumber_candidate_rois(kept)


def candidate_passes_filters(
    candidate: CandidateRoi,
    *,
    min_area: int | float | None = None,
    max_area: int | float | None = None,
    min_perimeter: int | float | None = None,
    max_perimeter: int | float | None = None,
    min_width: int | float | None = None,
    max_width: int | float | None = None,
    min_height: int | float | None = None,
    max_height: int | float | None = None,
    min_width_plus_height: int | float | None = None,
    max_width_plus_height: int | float | None = None,
) -> bool:
    """Return true when one candidate satisfies all configured filters."""
    return _geometric_measurements_pass_filters(
        area=int(candidate.bbox_w) * int(candidate.bbox_h),
        perimeter=candidate.perimeter,
        width=candidate.bbox_w,
        height=candidate.bbox_h,
        min_area=min_area,
        max_area=max_area,
        min_perimeter=min_perimeter,
        max_perimeter=max_perimeter,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        min_width_plus_height=min_width_plus_height,
        max_width_plus_height=max_width_plus_height,
    )


def component_statistics_pass_filters(
    *,
    area: int | float,
    width: int,
    height: int,
    min_area: int | float | None = None,
    max_area: int | float | None = None,
    min_perimeter: int | float | None = None,
    max_perimeter: int | float | None = None,
    min_width: int | float | None = None,
    max_width: int | float | None = None,
    min_height: int | float | None = None,
    max_height: int | float | None = None,
    min_width_plus_height: int | float | None = None,
    max_width_plus_height: int | float | None = None,
) -> bool:
    """Apply the fast generation gate to connected-component statistics.

    Both area and perimeter use bounding-box measurements here so components
    can be rejected before allocating a local mask or extracting a contour.
    Explicit calls that filter materialized ROIs use contour perimeter.
    """
    return _geometric_measurements_pass_filters(
        area=area,
        perimeter=(2 * width) + (2 * height),
        width=width,
        height=height,
        min_area=min_area,
        max_area=max_area,
        min_perimeter=min_perimeter,
        max_perimeter=max_perimeter,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        min_width_plus_height=min_width_plus_height,
        max_width_plus_height=max_width_plus_height,
    )


def _geometric_measurements_pass_filters(
    *,
    area: int | float,
    perimeter: int | float,
    width: int,
    height: int,
    min_area: int | float | None = None,
    max_area: int | float | None = None,
    min_perimeter: int | float | None = None,
    max_perimeter: int | float | None = None,
    min_width: int | float | None = None,
    max_width: int | float | None = None,
    min_height: int | float | None = None,
    max_height: int | float | None = None,
    min_width_plus_height: int | float | None = None,
    max_width_plus_height: int | float | None = None,
) -> bool:
    checks = (
        (area, min_area, max_area),
        (perimeter, min_perimeter, max_perimeter),
        (width, min_width, max_width),
        (height, min_height, max_height),
        (width + height, min_width_plus_height, max_width_plus_height),
    )
    for value, minimum, maximum in checks:
        if minimum is not None and float(value) < float(minimum):
            return False
        if maximum is not None and float(value) > float(maximum):
            return False
    return True


def should_store_roi_payload(
    candidate: CandidateRoi,
    *,
    min_area: int | float | None = None,
    min_width: int | float | None = None,
    min_height: int | float | None = None,
    min_width_plus_height: int | float | None = None,
) -> bool:
    """Return true when a candidate is large enough to store its image payload."""
    return candidate_passes_filters(
        candidate,
        min_area=min_area,
        min_width=min_width,
        min_height=min_height,
        min_width_plus_height=min_width_plus_height,
    )
