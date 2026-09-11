from __future__ import annotations

import json
import sqlite3
import uuid

import numpy as np

from Pelagia.processing.frame_codec import encode_array_payload
from Pelagia.services.exports import roi


def _row(*, area: int = 100, asset_id: str | None = None) -> dict:
    payload, encoding, format_name = encode_array_payload(np.arange(6, dtype=np.uint8).reshape(2, 3), "png")
    return {
        "id": str(uuid.uuid4()), "candidate_detection_id": str(uuid.uuid4()), "source_asset_id": asset_id or str(uuid.uuid4()),
        "source_asset_filename": "source image.tif", "source_asset_collections": ["survey"],
        "source_asset_metadata": {"instrument_name": "Camera A", "depth_m": 12.5, "opaque": {"source": "lab"}},
        "frame_id": str(uuid.uuid4()), "frame_index": 4, "frame_width": 100, "frame_height": 80,
        "frame_metadata": {"station": "A01"}, "bbox_x": 2, "bbox_y": 3, "bbox_w": 10, "bbox_h": area // 10,
        "area": 19.5, "perimeter": 20, "major_axis_length": 8, "minor_axis_length": 3,
        "min_gray_value": 1, "mean_gray_value": 3.2, "refinement_method": "identity",
        "metadata": {"detection_method": "threshold", "unmapped": "retained"}, "run_id": str(uuid.uuid4()),
        "roi_payload": payload, "roi_encoding": encoding, "roi_format": format_name, "roi_dtype": "uint8", "roi_shape": [2, 3],
    }


def test_statistics_row_is_analyst_facing_and_entity_metadata_is_not_repeated():
    row = roi.statistics_row(_row())
    assert row["image_name"] == "source image.tif"
    assert row["bounding_box_area_px2"] == 100
    assert row["instrument"] == "Camera A"
    assert row["station"] == "A01"
    assert not any("metadata" in name for name in row)
    extra = roi.additional_metadata_rows(_row())
    assert {item["field_name"] for item in extra} >= {"opaque", "unmapped"}
    first, second = _row(), _row()
    second["source_asset_id"] = first["source_asset_id"]
    second["frame_id"] = first["frame_id"]
    tables = roi.export_metadata_tables([first, second])
    assert len(tables["asset_metadata"]) == 1
    assert len(tables["frame_metadata"]) == 0  # station is already a statistics column.
    assert len(tables["roi_metadata"]) == 2


def test_bbox_bins_follow_documented_boundaries():
    assert roi.bbox_area_bins(1001)[:12] == [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 200), (200, 300)]
    rows = [_row(area=value) for value in (10, 100, 1000, 1001)]
    bins = roi.binned_statistics(rows)
    assert {(item["area_bin_lower_px2"], item["roi_count"]) for item in bins} == {(10, 1), (100, 1), (1000, 2)}


def test_writers_use_uuid_paths_and_include_evidence(tmp_path, monkeypatch):
    asset_id = str(uuid.uuid4()); row = _row(asset_id=asset_id)
    monkeypatch.setattr(roi, "_selected_rows", lambda *args: [row])
    class Repository:
        def get_curation_roi(self, roi_id, *, project_id):
            return {"id": roi_id, "evidence": [{"id": "old"}, {"id": "new"}]}
    result = roi.write_raw_roi_statistics(Repository(), project_id=str(uuid.uuid4()), selection={}, output_root=tmp_path, file_format="sqlite")
    statistics = tmp_path / result["paths"][0]
    assert f"assets/{asset_id}" in str(statistics)
    with sqlite3.connect(statistics) as db:
        assert db.execute("SELECT count(*) FROM roi_statistics").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM asset_metadata").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM frame_metadata").fetchone()[0] == 0
    evidence = roi.write_roi_evidence(Repository(), project_id=str(uuid.uuid4()), selection={}, output_root=tmp_path)
    sidecar = tmp_path / [path for path in evidence["paths"] if path.endswith("metadata.json")][0]
    payload = json.loads(sidecar.read_text())
    assert len(payload["evidence"]["evidence"]) == 2
    assert (sidecar.parent / "image.png").exists()


def test_roi_row_collection_reports_bounded_progress(monkeypatch):
    rows = [_row() for _ in range(1_001)]
    monkeypatch.setattr(roi.registry_generation, "_selected_rows", lambda *_args: iter(rows))
    progress: list[tuple[int, str]] = []

    collected = roi._selected_rows(
        object(), str(uuid.uuid4()), {"roi_stage": "refined"},
        lambda completed, message: progress.append((completed, message)),
    )

    assert len(collected) == 1_001
    assert [completed for completed, _message in progress] == [1_000, 1_001]
