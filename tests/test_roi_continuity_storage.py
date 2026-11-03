from __future__ import annotations

from typing import Any

import pytest

from Pelagia.storage.postgres import PostgresRepository


class _Cursor:
    def __init__(self, *, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.executions: list[tuple[str, tuple[Any, ...]]] = []
        self.many: list[tuple[str, list[tuple[Any, ...]]]] = []
        self._assembly_index = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> None:
        self.executions.append((statement, parameters))

    def executemany(self, statement: str, parameters: list[tuple[Any, ...]]) -> None:
        self.many.append((statement, list(parameters)))

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows

    def fetchone(self) -> dict[str, Any] | None:
        statement = self.executions[-1][0]
        if "roi_continuity_runs" in statement and "SELECT" in statement:
            return None
        if "roi_continuity_runs" in statement:
            return {"id": "run-1"}
        self._assembly_index += 1
        return {"id": f"assembly-{self._assembly_index}"}


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def commit(self) -> None:
        self.commits += 1

    def cursor(self) -> _Cursor:
        return self._cursor


def _repository(cursor: _Cursor) -> PostgresRepository:
    repository = object.__new__(PostgresRepository)
    repository.schema = "pelagia_unit"
    repository.connect = lambda: _Connection(cursor)
    repository._ensure_project_scope = lambda *_args, **_kwargs: None
    return repository


def test_continuity_query_selects_only_geometry_and_never_hydrates_payloads():
    row = {"id": "roi-1", "bbox_x": 1}
    cursor = _Cursor(rows=[row])
    repository = _repository(cursor)
    repository.hot_store = object()

    rows = repository.list_refined_detections_for_continuity(
        "asset-1", project_id="project-1", start_frame=4, end_frame=8,
    )

    assert rows == [row]
    statement, parameters = cursor.executions[0]
    assert "refined.*" not in statement
    assert "roi_payload" not in statement
    assert "mask_payload" not in statement
    assert "refined.metadata" in statement
    assert "jsonb_build_object" in statement
    assert "refined.metadata," not in statement
    assert "frames.frame_index >= %s" in statement
    assert "frames.frame_index <= %s" in statement
    assert parameters == ("asset-1", "project-1", 4, 8)


@pytest.mark.parametrize(
    ("axis", "expected_clause"),
    [
        ("x", "(refined.bbox_x = 0 OR refined.bbox_x + refined.bbox_w = frames.width)"),
        ("y", "(refined.bbox_y = 0 OR refined.bbox_y + refined.bbox_h = frames.height)"),
    ],
)
def test_continuity_query_filters_to_foreground_frame_edges_when_axis_is_supplied(
    axis: str, expected_clause: str,
):
    cursor = _Cursor()
    repository = _repository(cursor)

    repository.list_refined_detections_for_continuity(
        "asset-1", project_id="project-1", scan_axis=axis,
    )

    statement, parameters = cursor.executions[0]
    assert expected_clause in statement
    assert parameters == ("asset-1", "project-1")


def test_continuity_query_rejects_unknown_scan_axis_before_database_access():
    cursor = _Cursor()
    repository = _repository(cursor)

    with pytest.raises(ValueError, match="scan_axis"):
        repository.list_refined_detections_for_continuity(
            "asset-1", project_id="project-1", scan_axis="z",
        )

    assert cursor.executions == []


def test_continuity_persists_only_selected_links_and_accepts_serialized_link_pairs():
    cursor = _Cursor()
    repository = _repository(cursor)

    stored = repository.record_roi_continuity(
        project_id="project-1", asset_id="asset-1", run_id=None, job_id=None,
        method="line_scan_continuity_v2", parameters={"scan_axis": "y"},
        assemblies=[{"assembly_id": "assembly-key", "link_pairs": [["roi-a", "roi-b"]]}],
        decisions=[
            {"source_segment_id": "roi-a", "target_segment_id": "roi-b", "score": 0.9, "accepted": True},
            {"source_segment_id": "roi-a", "target_segment_id": "roi-c", "score": 0.8, "accepted": False,
             "reason": "competing_association", "features": {"unused": True}},
        ],
    )

    assert stored == {"id": "run-1"}
    assert len(cursor.many) == 1
    _, values = cursor.many[0]
    assert values == [("run-1", "assembly-1", "roi-a", "roi-b", 0.9, True, None, "{}")]


def test_continuity_persists_empty_assessment_without_link_insert():
    cursor = _Cursor()
    repository = _repository(cursor)

    stored = repository.record_roi_continuity(
        project_id="project-1", asset_id="asset-1", run_id=None, job_id=None,
        method="line_scan_continuity_v2", parameters={"scan_axis": "y"},
        assemblies=[], decisions=[],
    )

    assert stored == {"id": "run-1"}
    assert cursor.many == []
