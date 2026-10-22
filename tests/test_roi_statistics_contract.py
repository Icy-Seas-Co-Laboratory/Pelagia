"""Explicit ROI metrics must never reinterpret historical pixel counts."""
from contextlib import contextmanager

import numpy as np
import pytest

from Pelagia.api.routes._common import detection_summary
from Pelagia.storage.postgres import PostgresRepository


def test_legacy_refined_area_is_remeasured_without_changing_stored_area():
    mask = np.full((3, 4), 255, dtype=np.uint8)
    row = dict(id="refined", candidate_detection_id="candidate", bbox_w=4, bbox_h=3,
               area=12, perimeter=10, mask_payload=mask.tobytes(), mask_encoding="raw",
               mask_format="raw_ndarray_c_order", mask_dtype="uint8", mask_shape=[3, 4])
    result = detection_summary(row)
    assert result["bbox_area"] == 12
    assert result["contour_area"] == 6
    assert result["contour_perimeter"] == 10
    assert result["area"] == row["area"] == 12
    assert "mask_payload" not in result


@pytest.mark.parametrize("payload", [None, b"corrupt"])
def test_unavailable_legacy_mask_does_not_become_zero_or_pixel_area(payload):
    result = detection_summary(dict(candidate_detection_id="candidate", area=42,
                                    mask_payload=payload, mask_encoding="png"))
    assert result["contour_area"] is None
    assert result["bbox_area"] is None


def test_versioned_refined_area_needs_no_mask_decode():
    result = detection_summary(dict(area=6, perimeter=10, candidate_detection_id="candidate",
        metadata={"roi_measurements": {"version": 1, "area_method": "opencv_external_contours"}}))
    assert result["contour_area"] == 6
    assert result["contour_perimeter"] == 10


def test_candidate_bbox_area_ignores_historical_contour_area():
    result = detection_summary(dict(area=6, perimeter=10, bbox_w=4, bbox_h=3))
    assert result["bbox_area"] == 12
    assert result["contour_area"] is None


class CaptureCursor:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def execute(self, sql, params): self.sql, self.params = sql, params
    def fetchall(self): return []


class CaptureConnection:
    def __init__(self, cursor): self._cursor = cursor
    def cursor(self): return self._cursor


def repository_with_cursor(monkeypatch):
    repository = object.__new__(PostgresRepository)
    repository.schema = "pelagia"
    cursor = CaptureCursor()
    @contextmanager
    def connect():
        yield CaptureConnection(cursor)
    monkeypatch.setattr(repository, "connect", connect)
    return repository, cursor


def test_roi_browser_filters_bbox_area_and_stored_contour_perimeter(monkeypatch):
    repository, cursor = repository_with_cursor(monkeypatch)
    repository.list_detections(project_id="project", min_area=12, min_perimeter=10, sort_by="area")
    assert "(detections.bbox_w::double precision * detections.bbox_h) >= %s" in cursor.sql
    assert "detections.perimeter >= %s" in cursor.sql
    assert "ORDER BY (detections.bbox_w::double precision * detections.bbox_h)" in cursor.sql
    assert "assets.project_id = %s" in cursor.sql


def test_refined_frame_listing_scopes_and_paginates_all_identities(monkeypatch):
    repository, cursor = repository_with_cursor(monkeypatch)
    repository.list_refined_detections("frame", project_id="project", limit=2, offset=2)
    assert cursor.params == ("frame", "project", 2, 2)
    assert "assets.project_id = %s" in cursor.sql
    assert "refined.frame_id = %s" in cursor.sql
    assert "ORDER BY refined.roi_index ASC, refined.id ASC" in cursor.sql
    assert "DISTINCT" not in cursor.sql
    assert "SELECT refined.*" not in cursor.sql
    assert "THEN NULL ELSE refined.mask_payload" in cursor.sql
    assert "octet_length(refined.roi_payload) AS roi_payload_bytes" in cursor.sql


@pytest.mark.parametrize("version", [None, 0, 2])
def test_unversioned_or_unknown_area_method_is_not_trusted(version):
    result = detection_summary(dict(area=42, candidate_detection_id="candidate",
        metadata={"roi_measurements": {"version": version, "area_method": "opencv_external_contours"}}))
    assert result["contour_area"] is None


def test_corrupt_legacy_zstd_mask_returns_unknown_contour_area():
    result = detection_summary(dict(area=42, candidate_detection_id="candidate",
        mask_payload=b"corrupt", mask_encoding="zstd", mask_dtype="uint8", mask_shape=[3, 4]))
    assert result["contour_area"] is None
