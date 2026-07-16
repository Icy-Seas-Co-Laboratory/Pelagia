"""Apply geometric and payload-storage filters to candidate ROIs."""

from __future__ import annotations

from .roi_assembly import CandidateRoi, renumber_candidate_rois


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
    return renumber_candidate_rois(kept) if reindex else kept


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
    checks = (
        (candidate.area, min_area, max_area),
        (candidate.bbox_perimeter, min_perimeter, max_perimeter),
        (candidate.bbox_w, min_width, max_width),
        (candidate.bbox_h, min_height, max_height),
        (candidate.width_plus_height, min_width_plus_height, max_width_plus_height),
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
