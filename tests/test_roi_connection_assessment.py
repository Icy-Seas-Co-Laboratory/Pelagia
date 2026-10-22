from __future__ import annotations

from typing import Any

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
        self, asset_id: str, *, project_id: str,
    ) -> list[dict[str, Any]]:
        assert project_id == "project-1"
        self.listed_asset_ids.append(asset_id)
        return self.rows.get(asset_id, [])

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
    assert recorded["method"] == "line_scan_continuity_v1"
    assert recorded["parameters"]["scan_axis"] == "y"
    assert recorded["metadata"] == {"line_scan_declared": True, "recomputed": True}
    assert recorded["decisions"][0]["accepted"] is True
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
