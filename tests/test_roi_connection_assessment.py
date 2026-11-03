from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from Pelagia.config import RoiContinuityProcessingConfig
from Pelagia.services.roi_connection_assessment import assess_refined_roi_connections


class FakeRepository:
    def __init__(self, assets: dict[str, dict[str, Any]], rows: dict[str, list[dict[str, Any]]]) -> None:
        self.assets = assets
        self.rows = rows
        self.listed_asset_ids: list[str] = []
        self.recorded: list[dict[str, Any]] = []

    def get_asset(self, asset_id: str, *, project_id: str) -> dict[str, Any] | None:
        assert project_id == "project-1"
        return self.assets.get(asset_id)

    def list_refined_detections_for_continuity(
        self, asset_id: str, *, project_id: str, scan_axis: str,
    ) -> list[dict[str, Any]]:
        assert project_id == "project-1"
        self.listed_asset_ids.append(asset_id)
        coord, extent, dimension = (
            ("bbox_y", "bbox_h", "frame_height") if scan_axis == "y"
            else ("bbox_x", "bbox_w", "frame_width")
        )
        return [row for row in self.rows.get(asset_id, [])
                if row[coord] == 0 or row[coord] + row[extent] == row[dimension]]

    def record_roi_continuity(self, **kwargs: Any) -> dict[str, Any]:
        self.recorded.append(kwargs)
        return {"id": f"continuity-{len(self.recorded)}"}


def _refined_row(segment_id: str, frame_index: int, *, y: int) -> dict[str, Any]:
    return {
        "id": segment_id,
        "frame_id": f"frame-{frame_index}",
        "asset_id": "line-scan-asset",
        "frame_index": frame_index,
        "frame_width": 100,
        "frame_height": 100,
        "bbox_x": 40,
        "bbox_y": y,
        "bbox_w": 20,
        "bbox_h": 10,
        "metadata": {"dominant_edge_angle_degrees": 45.0, "edge_strength": 0.9},
    }


def test_skips_assets_without_line_scan_metadata():
    repository = FakeRepository({"ordinary": {"metadata": {}}}, {})

    summary = assess_refined_roi_connections(
        repository=repository,
        project_id="project-1",
        asset_ids=["ordinary"],
        defaults=RoiContinuityProcessingConfig(),
    )

    assert summary == {
        "asset_count": 1,
        "assessed_asset_count": 0,
        "skipped_asset_ids": ["ordinary"],
        "runs": [],
    }
    assert repository.listed_asset_ids == []
    assert repository.recorded == []


def test_assesses_complete_line_scan_asset_and_records_audit_run():
    repository = FakeRepository(
        {"line-scan-asset": {"metadata": {"line_scan_geometry": {"axis": "y"}}}},
        {"line-scan-asset": [
            _refined_row("refined-a", 0, y=90),
            _refined_row("refined-b", 1, y=0),
        ]},
    )

    summary = assess_refined_roi_connections(
        repository=repository,
        project_id="project-1",
        asset_ids=["line-scan-asset"],
        defaults=RoiContinuityProcessingConfig(scan_axis="x"),
        run_id="run-1",
        job_id="job-1",
    )

    assert repository.listed_asset_ids == ["line-scan-asset"]
    assert len(repository.recorded) == 1
    recorded = repository.recorded[0]
    assert recorded["project_id"] == "project-1"
    assert recorded["run_id"] == "run-1"
    assert recorded["job_id"] == "job-1"
    assert recorded["method"] == "line_scan_continuity_v2"
    assert recorded["parameters"]["scan_axis"] == "y"
    assert recorded["parameters"]["boundary_band_pixels"] == 0
    assert recorded["parameters"]["min_boundary_overlap"] == 0.25
    assert recorded["parameters"]["max_frame_gap"] == 1
    assert recorded["metadata"]["line_scan_declared"] is True
    assert recorded["metadata"]["input_scope"] == "asset_frame_edges"
    assert len(recorded["metadata"]["input_fingerprint"]) == 64
    assert recorded["decisions"] == [{
        "source_segment_id": "refined-a", "target_segment_id": "refined-b",
        "score": 1.0, "accepted": True,
    }]
    assert summary["runs"] == [{
        "asset_id": "line-scan-asset",
        "continuity_run_id": "continuity-1",
        "segment_count": 2,
        "accepted_link_count": 1,
        "assembly_count": 1,
    }]


def test_records_empty_audit_run_for_eligible_asset_without_refined_detections():
    repository = FakeRepository({"line-scan-asset": {"metadata": {"line_scan": True}}}, {})

    summary = assess_refined_roi_connections(
        repository=repository,
        project_id="project-1",
        asset_ids=["line-scan-asset", "line-scan-asset"],
        defaults=RoiContinuityProcessingConfig(),
    )

    assert repository.listed_asset_ids == ["line-scan-asset"]
    assert len(repository.recorded) == 1
    assert repository.recorded[0]["assemblies"] == []
    assert repository.recorded[0]["decisions"] == []
    assert summary["asset_count"] == 1
    assert summary["runs"][0]["segment_count"] == 0


def test_only_selected_links_are_recorded_and_legacy_band_does_not_admit_near_edge():
    rows = [
        _refined_row("source", 0, y=90),
        _refined_row("target-b", 1, y=0),
        _refined_row("target-a", 1, y=0),
        _refined_row("near-edge", 0, y=89),
    ]
    repository = FakeRepository(
        {"line-scan-asset": {"metadata": {"line_scan": True}}},
        {"line-scan-asset": rows},
    )
    assess_refined_roi_connections(
        repository=repository, project_id="project-1", asset_ids=["line-scan-asset"],
        defaults=RoiContinuityProcessingConfig(boundary_band_pixels=32),
    )
    selected = repository.recorded[0]["decisions"]
    assert len(selected) == 1
    assert selected[0]["source_segment_id"] == "source"
    # Equal scores use deterministic ID ordering.
    assert selected[0]["target_segment_id"] == "target-a"
    assert repository.recorded[0]["metadata"]["accepted_link_count"] == 1


def test_input_fingerprint_is_order_independent_and_tracks_geometry():
    rows = [_refined_row("a", 0, y=90), _refined_row("b", 1, y=0)]
    repository = FakeRepository(
        {"line-scan-asset": {"metadata": {"line_scan": True}}},
        {"line-scan-asset": rows},
    )

    def fingerprint():
        assess_refined_roi_connections(
            repository=repository, project_id="project-1", asset_ids=["line-scan-asset"],
            defaults=RoiContinuityProcessingConfig(),
        )
        return repository.recorded[-1]["metadata"]["input_fingerprint"]

    first = fingerprint()
    rows.reverse()
    rows[0]["metadata"]["unused_label"] = "not an algorithm input"
    assert fingerprint() == first
    rows[0]["bbox_x"] += 1
    assert fingerprint() != first


def test_retry_summary_uses_published_run_counts():
    class ExistingRunRepository(FakeRepository):
        def record_roi_continuity(self, **kwargs):
            return {"id": "existing", "metadata": {
                "segment_count": 20, "accepted_link_count": 3, "assembly_count": 2,
            }}

    repository = ExistingRunRepository(
        {"line-scan-asset": {"metadata": {"line_scan": True}}}, {},
    )
    summary = assess_refined_roi_connections(
        repository=repository, project_id="project-1", asset_ids=["line-scan-asset"],
        defaults=RoiContinuityProcessingConfig(), job_id="job-1",
    )
    assert summary["runs"] == [{
        "asset_id": "line-scan-asset", "continuity_run_id": "existing",
        "segment_count": 20, "accepted_link_count": 3, "assembly_count": 2,
    }]


@pytest.mark.parametrize("explicit_axis, expected_links", [(None, 1), ("y", 0)])
def test_continuity_worker_uses_asset_axis_unless_explicitly_overridden(explicit_axis, expected_links):
    from Pelagia.workers.handlers import roi_continuity_handler

    rows = [_refined_row("a", 0, y=40), _refined_row("b", 1, y=40)]
    for row, x in zip(rows, (90, 0)):
        row.update(bbox_x=x, bbox_w=10, bbox_h=20)
    repository = FakeRepository(
        {"line-scan-asset": {"metadata": {"line_scan_geometry": {"scan_axis": "x"}}}},
        {"line-scan-asset": rows},
    )
    context = SimpleNamespace(repository=repository, config=SimpleNamespace(
        processing=SimpleNamespace(roi_continuity=RoiContinuityProcessingConfig(scan_axis="y")),
    ))
    payload = {"asset_id": "line-scan-asset", "boundary_band_pixels": 32}
    if explicit_axis is not None:
        payload["scan_axis"] = explicit_axis
    result = roi_continuity_handler({
        "id": "job-1", "run_id": "run-1", "project_id": "project-1", "payload": payload,
    }, context)
    assert result["accepted_link_count"] == expected_links
    assert repository.recorded[0]["parameters"]["scan_axis"] == (explicit_axis or "x")
    assert repository.recorded[0]["parameters"]["boundary_band_pixels"] == 0
