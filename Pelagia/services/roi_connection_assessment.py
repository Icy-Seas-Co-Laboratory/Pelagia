"""Project-scoped reassessment of line-scan ROI continuity after refinement.

Continuity is derived from the current edge-touching refined detections for an
asset.  Each invocation records an audit run so a later refinement can be
reassessed without mutating an earlier result.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

from ..config import RoiContinuityProcessingConfig
from ..processing.roi_continuity import (
    FrameLocalRoiSegment,
    LineScanGeometry,
    RoiContinuityOptions,
    assemble_line_scan_continuity,
)


def assess_refined_roi_connections(
    *,
    repository: Any,
    project_id: str,
    asset_ids: Iterable[str],
    defaults: RoiContinuityProcessingConfig,
    run_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Recompute and persist line-scan continuity for the supplied assets.

    Only assets that explicitly declare ``line_scan`` or
    ``line_scan_geometry`` in metadata are eligible.  Eligible assets are
    always recorded, including when no refined detections exist, so the audit
    trail represents the complete state after the triggering refinement.
    """
    requested_asset_ids = list(dict.fromkeys(str(asset_id) for asset_id in asset_ids))
    summary: dict[str, Any] = {
        "asset_count": len(requested_asset_ids),
        "assessed_asset_count": 0,
        "skipped_asset_ids": [],
        "runs": [],
    }
    if not defaults.enabled:
        summary["skipped_asset_ids"] = requested_asset_ids
        summary["disabled"] = True
        return summary

    options = RoiContinuityOptions(min_link_score=float(defaults.min_link_score))
    for asset_id in requested_asset_ids:
        asset = repository.get_asset(asset_id, project_id=project_id)
        metadata = dict((asset or {}).get("metadata") or {})
        declared = bool(metadata.get("line_scan") or metadata.get("line_scan_geometry"))
        if not declared:
            summary["skipped_asset_ids"].append(asset_id)
            continue

        geometry_metadata = metadata.get("line_scan_geometry") or metadata.get("line_scan")
        axis = (
            geometry_metadata.get("scan_axis", geometry_metadata.get("axis", defaults.scan_axis))
            if isinstance(geometry_metadata, dict)
            else defaults.scan_axis
        )
        geometry = LineScanGeometry(declared=True, scan_axis=str(axis))
        assessed = assess_asset_roi_connections(
            repository=repository,
            project_id=project_id,
            asset_id=asset_id,
            geometry=geometry,
            options=options,
            line_scan_declared=True,
            run_id=run_id,
            job_id=job_id,
        )
        summary["assessed_asset_count"] += 1
        summary["runs"].append(assessed)
    return summary


def assess_asset_roi_connections(
    *,
    repository: Any,
    project_id: str,
    asset_id: str,
    geometry: LineScanGeometry,
    options: RoiContinuityOptions,
    line_scan_declared: bool,
    run_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Record selected links and a compact identity of the assessed input.

    Automatic refinement and explicit continuity retries share this path so
    geometry, effective parameters, and provenance have the same meaning.
    The fingerprint covers algorithm inputs, not unused image payloads.
    """
    rows = repository.list_refined_detections_for_continuity(
        asset_id, project_id=project_id, scan_axis=geometry.scan_axis,
    )
    segments = [_segment_from_row(row) for row in rows]
    result = assemble_line_scan_continuity(segments, geometry=geometry, options=options)
    inputs = [
        (
            item.segment_id, item.frame_id, item.asset_id, item.frame_index,
            item.frame_width, item.frame_height,
            item.bbox_x, item.bbox_y, item.bbox_w, item.bbox_h,
            item.edge_angle_degrees, item.edge_strength,
        )
        for item in sorted(segments, key=lambda item: item.segment_id)
    ]
    fingerprint = sha256(
        json.dumps(inputs, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    counts = {
        "segment_count": len(segments),
        "accepted_link_count": len(result.accepted_links),
        "assembly_count": len(result.assemblies),
    }
    stored = repository.record_roi_continuity(
        project_id=project_id,
        asset_id=asset_id,
        run_id=run_id,
        job_id=job_id,
        method="line_scan_continuity_v2",
        parameters={
            **asdict(options),
            "scan_axis": geometry.scan_axis,
            # Legacy options cannot relax exact edge contact or adjacency.
            "boundary_band_pixels": 0,
            "max_frame_gap": 1,
        },
        assemblies=[
            {
                "assembly_id": item.assembly_id,
                "segment_ids": list(item.segment_ids),
                "link_pairs": [list(pair) for pair in item.link_pairs],
            }
            for item in result.assemblies
        ],
        decisions=[
            {
                "source_segment_id": item.source_segment_id,
                "target_segment_id": item.target_segment_id,
                "score": item.score,
                "accepted": True,
            }
            for item in result.accepted_links
        ],
        metadata={
            "line_scan_declared": line_scan_declared,
            "input_scope": "asset_frame_edges",
            "input_fingerprint": fingerprint,
            **counts,
        },
    )
    # A job retry may reuse its already-published run after more refinement has
    # arrived. Report that run's counts rather than the new calculation's counts.
    stored_metadata = stored.get("metadata") or {}
    return {
        "asset_id": asset_id,
        "continuity_run_id": stored.get("id"),
        **{name: stored_metadata.get(name, value) for name, value in counts.items()},
    }


def _segment_from_row(row: dict[str, Any]) -> FrameLocalRoiSegment:
    metadata = dict(row.get("metadata") or {})
    return FrameLocalRoiSegment(
        segment_id=str(row["id"]),
        frame_id=str(row["frame_id"]),
        asset_id=str(row["asset_id"]),
        frame_index=int(row["frame_index"]),
        frame_width=int(row["frame_width"]),
        frame_height=int(row["frame_height"]),
        bbox_x=int(row["bbox_x"]),
        bbox_y=int(row["bbox_y"]),
        bbox_w=int(row["bbox_w"]),
        bbox_h=int(row["bbox_h"]),
        edge_angle_degrees=metadata.get("dominant_edge_angle_degrees"),
        edge_strength=metadata.get("edge_strength"),
        metadata=metadata,
    )
