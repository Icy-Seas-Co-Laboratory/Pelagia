"""Audit-friendly assembly of frame-local ROI segments from line-scan cameras.

This module deliberately does not merge pixel payloads or mutate detections.
Line-scan continuity is a logical relationship between immutable, frame-local
refined ROI segments.  Persistence and job scheduling are intentionally left to
the caller so this deterministic stage can be replayed and tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

from ..domain import DetectionRecord


ScanAxis = Literal["x", "y"]


@dataclass(frozen=True, slots=True)
class LineScanGeometry:
    """Explicit geometry declaration required before continuity can run.

    ``scan_axis`` is the dimension on which successive frames are appended;
    an increasing frame index therefore joins the source's positive boundary
    to the target's zero boundary.
    """

    declared: bool
    scan_axis: ScanAxis

    def __post_init__(self) -> None:
        if not self.declared:
            raise ValueError("ROI continuity requires declared line-scan geometry.")
        if self.scan_axis not in {"x", "y"}:
            raise ValueError("scan_axis must be 'x' or 'y'.")


@dataclass(frozen=True, slots=True)
class FrameLocalRoiSegment:
    """The minimum frame-local context needed to link a refined ROI."""

    segment_id: str
    frame_id: str
    asset_id: str
    frame_index: int
    frame_width: int
    frame_height: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    edge_angle_degrees: float | None = None
    edge_strength: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_detection(
        cls,
        detection: DetectionRecord,
        *,
        asset_id: str,
        frame_index: int,
        frame_width: int,
        frame_height: int,
        edge_angle_degrees: float | None = None,
        edge_strength: float | None = None,
    ) -> "FrameLocalRoiSegment":
        """Adapt an existing frame-local detection without changing its schema."""
        if detection.id is None:
            raise ValueError("Continuity segments require persisted detection ids.")
        metadata = dict(detection.metadata)
        return cls(
            segment_id=str(detection.id), frame_id=str(detection.frame_id), asset_id=str(asset_id),
            frame_index=int(frame_index), frame_width=int(frame_width), frame_height=int(frame_height),
            bbox_x=int(detection.bbox_x), bbox_y=int(detection.bbox_y),
            bbox_w=int(detection.bbox_w), bbox_h=int(detection.bbox_h),
            edge_angle_degrees=(
                edge_angle_degrees
                if edge_angle_degrees is not None
                else _optional_float(metadata.get("dominant_edge_angle_degrees"))
            ),
            edge_strength=(
                edge_strength if edge_strength is not None else _optional_float(metadata.get("edge_strength"))
            ),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class RoiContinuityOptions:
    """Deterministic association limits for adjacent line-scan frames."""

    boundary_band_pixels: int = 32
    min_boundary_overlap: float = 0.25
    max_center_displacement_pixels: float = 64.0
    min_link_score: float = 0.7
    axis_alignment_tolerance_degrees: float = 1.0
    min_edge_strength: float | None = None

    def __post_init__(self) -> None:
        if self.boundary_band_pixels < 0:
            raise ValueError("boundary_band_pixels must be >= 0.")
        if not 0 <= self.min_boundary_overlap <= 1:
            raise ValueError("min_boundary_overlap must be between 0 and 1.")
        if self.max_center_displacement_pixels <= 0:
            raise ValueError("max_center_displacement_pixels must be > 0.")
        if not 0 <= self.min_link_score <= 1:
            raise ValueError("min_link_score must be between 0 and 1.")
        if not 0 <= self.axis_alignment_tolerance_degrees <= 45:
            raise ValueError("axis_alignment_tolerance_degrees must be between 0 and 45.")


@dataclass(frozen=True, slots=True)
class RoiContinuityDecision:
    """One accepted or rejected association with its complete score evidence."""

    source_segment_id: str
    target_segment_id: str
    source_frame_index: int
    target_frame_index: int
    score: float
    accepted: bool
    reason: str | None
    features: dict[str, float | bool | None]


@dataclass(frozen=True, slots=True)
class RoiContinuityAssembly:
    """A logical parent ROI and references to its unchanged local segments."""

    assembly_id: str
    asset_id: str
    segment_ids: tuple[str, ...]
    link_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RoiContinuityResult:
    decisions: tuple[RoiContinuityDecision, ...]
    assemblies: tuple[RoiContinuityAssembly, ...]

    @property
    def accepted_links(self) -> tuple[RoiContinuityDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.accepted)


def assemble_line_scan_continuity(
    segments: list[FrameLocalRoiSegment],
    *,
    geometry: LineScanGeometry,
    options: RoiContinuityOptions | None = None,
) -> RoiContinuityResult:
    """Link compatible refined ROI segments in adjacent frames only.

    Associations are one-to-one for each pair of frames.  Links are selected in
    a stable score order, then assembled transitively into logical parents.
    Every considered adjacent-frame pair is emitted as a decision, including
    orientation, boundary, score, and competing-link rejections.
    """
    resolved = options or RoiContinuityOptions()
    _validate_segments(segments)
    ordered = sorted(segments, key=lambda item: (item.asset_id, item.frame_index, item.segment_id))
    decisions: list[RoiContinuityDecision] = []
    accepted_pairs: list[tuple[str, str]] = []
    for asset_id in sorted({segment.asset_id for segment in ordered}):
        asset_segments = [segment for segment in ordered if segment.asset_id == asset_id]
        by_index: dict[int, list[FrameLocalRoiSegment]] = {}
        for segment in asset_segments:
            by_index.setdefault(segment.frame_index, []).append(segment)
        for frame_index in sorted(by_index):
            sources = by_index[frame_index]
            targets = by_index.get(frame_index + 1, [])
            candidates = [
                _score_pair(source, target, geometry=geometry, options=resolved)
                for source in sources for target in targets
            ]
            viable = [item for item in candidates if item.accepted]
            used_sources: set[str] = set()
            used_targets: set[str] = set()
            for candidate in sorted(viable, key=lambda item: (-item.score, item.source_segment_id, item.target_segment_id)):
                if candidate.source_segment_id in used_sources or candidate.target_segment_id in used_targets:
                    decisions.append(_replace_reason(candidate, "competing_association"))
                    continue
                used_sources.add(candidate.source_segment_id)
                used_targets.add(candidate.target_segment_id)
                decisions.append(candidate)
                accepted_pairs.append((candidate.source_segment_id, candidate.target_segment_id))
            decisions.extend(item for item in candidates if not item.accepted)
    decisions.sort(key=lambda item: (item.source_frame_index, item.target_frame_index, item.source_segment_id, item.target_segment_id))
    assemblies = _assemble_components(ordered, accepted_pairs)
    return RoiContinuityResult(decisions=tuple(decisions), assemblies=tuple(assemblies))


def _score_pair(source: FrameLocalRoiSegment, target: FrameLocalRoiSegment, *, geometry: LineScanGeometry, options: RoiContinuityOptions) -> RoiContinuityDecision:
    features: dict[str, float | bool | None] = {
        "source_boundary_distance": _positive_boundary_distance(source, geometry.scan_axis),
        "target_boundary_distance": _zero_boundary_distance(target, geometry.scan_axis),
        "source_edge_angle_degrees": source.edge_angle_degrees,
        "target_edge_angle_degrees": target.edge_angle_degrees,
        "source_axis_aligned": _is_axis_aligned(source.edge_angle_degrees, options.axis_alignment_tolerance_degrees),
        "target_axis_aligned": _is_axis_aligned(target.edge_angle_degrees, options.axis_alignment_tolerance_degrees),
        "source_edge_strength": source.edge_strength,
        "target_edge_strength": target.edge_strength,
    }
    if bool(features["source_axis_aligned"]) or bool(features["target_axis_aligned"]):
        return _decision(source, target, 0.0, False, "axis_aligned_edge", features)
    if options.min_edge_strength is not None and any(
        value is not None and float(value) < options.min_edge_strength
        for value in (source.edge_strength, target.edge_strength)
    ):
        return _decision(source, target, 0.0, False, "weak_edge", features)
    if float(features["source_boundary_distance"]) > options.boundary_band_pixels or float(features["target_boundary_distance"]) > options.boundary_band_pixels:
        return _decision(source, target, 0.0, False, "outside_boundary_band", features)
    source_interval = _transverse_interval(source, geometry.scan_axis)
    target_interval = _transverse_interval(target, geometry.scan_axis)
    overlap = _interval_overlap_ratio(source_interval, target_interval)
    displacement = abs(_interval_center(source_interval) - _interval_center(target_interval))
    positional = max(0.0, 1.0 - (displacement / options.max_center_displacement_pixels))
    score = (0.75 * overlap) + (0.25 * positional)
    features.update({"transverse_overlap": overlap, "center_displacement": displacement, "positional_score": positional})
    if overlap < options.min_boundary_overlap:
        return _decision(source, target, score, False, "insufficient_boundary_overlap", features)
    if score < options.min_link_score:
        return _decision(source, target, score, False, "below_link_score", features)
    return _decision(source, target, score, True, None, features)


def _decision(source: FrameLocalRoiSegment, target: FrameLocalRoiSegment, score: float, accepted: bool, reason: str | None, features: dict[str, float | bool | None]) -> RoiContinuityDecision:
    return RoiContinuityDecision(source.segment_id, target.segment_id, source.frame_index, target.frame_index, round(score, 8), accepted, reason, features)


def _replace_reason(decision: RoiContinuityDecision, reason: str) -> RoiContinuityDecision:
    return RoiContinuityDecision(decision.source_segment_id, decision.target_segment_id, decision.source_frame_index, decision.target_frame_index, decision.score, False, reason, decision.features)


def _assemble_components(segments: list[FrameLocalRoiSegment], links: list[tuple[str, str]]) -> list[RoiContinuityAssembly]:
    by_id = {segment.segment_id: segment for segment in segments}
    neighbours = {segment_id: set() for segment_id in by_id}
    for source, target in links:
        neighbours[source].add(target)
        neighbours[target].add(source)
    result: list[RoiContinuityAssembly] = []
    seen: set[str] = set()
    for segment_id in sorted(neighbours):
        if segment_id in seen or not neighbours[segment_id]:
            continue
        component: set[str] = set()
        pending = [segment_id]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(neighbours[current] - component)
        seen.update(component)
        segment_ids = tuple(sorted(component, key=lambda item: (by_id[item].frame_index, item)))
        asset_id = by_id[segment_ids[0]].asset_id
        pairs = tuple(sorted((pair for pair in links if pair[0] in component and pair[1] in component)))
        digest = sha256("|".join(segment_ids).encode("utf-8")).hexdigest()[:20]
        result.append(RoiContinuityAssembly(f"line-scan:{asset_id}:{digest}", asset_id, segment_ids, pairs))
    return result


def _validate_segments(segments: list[FrameLocalRoiSegment]) -> None:
    ids: set[str] = set()
    for segment in segments:
        if segment.segment_id in ids:
            raise ValueError(f"Duplicate continuity segment id {segment.segment_id!r}.")
        ids.add(segment.segment_id)
        if segment.frame_width <= 0 or segment.frame_height <= 0:
            raise ValueError("Continuity segment frame dimensions must be positive.")
        if segment.bbox_w <= 0 or segment.bbox_h <= 0:
            raise ValueError("Continuity segment bounding-box dimensions must be positive.")
        if (
            segment.bbox_x < 0
            or segment.bbox_y < 0
            or segment.bbox_x + segment.bbox_w > segment.frame_width
            or segment.bbox_y + segment.bbox_h > segment.frame_height
        ):
            raise ValueError("Continuity segment bounding box must lie within its frame.")


def _positive_boundary_distance(segment: FrameLocalRoiSegment, axis: ScanAxis) -> float:
    return float(segment.frame_height - (segment.bbox_y + segment.bbox_h) if axis == "y" else segment.frame_width - (segment.bbox_x + segment.bbox_w))


def _zero_boundary_distance(segment: FrameLocalRoiSegment, axis: ScanAxis) -> float:
    return float(segment.bbox_y if axis == "y" else segment.bbox_x)


def _transverse_interval(segment: FrameLocalRoiSegment, axis: ScanAxis) -> tuple[float, float]:
    return (float(segment.bbox_x), float(segment.bbox_x + segment.bbox_w)) if axis == "y" else (float(segment.bbox_y), float(segment.bbox_y + segment.bbox_h))


def _interval_overlap_ratio(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    return overlap / min(left[1] - left[0], right[1] - right[0])


def _interval_center(interval: tuple[float, float]) -> float:
    return (interval[0] + interval[1]) / 2.0


def _is_axis_aligned(angle: float | None, tolerance: float) -> bool:
    if angle is None:
        return False
    normalized = float(angle) % 180.0
    return min(normalized, abs(normalized - 90.0), abs(normalized - 180.0)) <= tolerance


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
