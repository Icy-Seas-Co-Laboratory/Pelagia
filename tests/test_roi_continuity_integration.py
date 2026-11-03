from __future__ import annotations

from datetime import datetime, timezone
import uuid

from Pelagia.config import RoiContinuityProcessingConfig
from Pelagia.domain import (
    AssetKind,
    DetectionRecord,
    FrameRecord,
    PlannedRun,
    RawAssetManifest,
    RunManifest,
)
from Pelagia.services.roi_connection_assessment import assess_refined_roi_connections
from test_postgres import postgres_repo  # noqa: F401 -- shared live PostgreSQL fixture


def _record(*, run_id: str, frame_id: str, roi_index: int, x: int, y: int, width: int, height: int) -> DetectionRecord:
    """Create a payload-bearing record so continuity proves it does not hydrate it."""
    return DetectionRecord(
        run_id=run_id,
        frame_id=frame_id,
        roi_index=roi_index,
        bbox_x=x,
        bbox_y=y,
        bbox_w=width,
        bbox_h=height,
        crop_bbox_x=x,
        crop_bbox_y=y,
        crop_bbox_w=width,
        crop_bbox_h=height,
        area=float(width * height),
        perimeter=float(2 * (width + height)),
        major_axis_length=float(max(width, height)),
        minor_axis_length=float(min(width, height)),
        min_gray_value=1,
        mean_gray_value=2.0,
        roi_payload=b"roi payload that continuity must not read",
        mask_payload=b"mask payload that continuity must not read",
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype="uint8",
        roi_shape=[height, width],
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype="uint8",
        mask_shape=[height, width],
        metadata={
            "detection_stage": "refined",
            "refinement_method": "identity",
            "dominant_edge_angle_degrees": 45.0,
            "edge_strength": 0.9,
            "unused_refinement_audit": "x" * 10_000,
        },
    )


def _line_scan_asset(repository) -> tuple[str, str, str, list[dict[str, object]]]:
    project = repository.create_project(f"continuity-{uuid.uuid4().hex}")
    project_id = str(project["id"])
    run_id = str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    repository.register_planned_run(
        PlannedRun(
            manifest=RunManifest(
                run_id=run_id,
                run_key=f"continuity-run-{uuid.uuid4().hex}",
                instrument="pytest",
                source_path="/tmp/continuity.avi",
                source_type=AssetKind.VIDEO.value,
                created_at=datetime.now(timezone.utc),
                assets=[RawAssetManifest(
                    asset_id=asset_id,
                    filename="continuity.avi",
                    path="/tmp/continuity.avi",
                    kind=AssetKind.VIDEO,
                    size_bytes=1,
                    checksum=f"sha256:{uuid.uuid4().hex}",
                    metadata={"line_scan_geometry": {"scan_axis": "y"}},
                )],
            ),
            jobs=[],
        ),
        project_id=project_id,
    )
    frames = repository.replace_frames(
        run_id,
        [
            FrameRecord(
                asset_id=asset_id,
                frame_index=index,
                width=100,
                height=100,
                preview_thumbhash=b"thumb",
                kvstore_hash=f"frame-{index}",
                payload_ref=f"frame-{index}",
                payload_encoding="raw",
                payload_format="raw_ndarray_c_order",
                payload_dtype="uint8",
                payload_shape=[100, 100],
            )
            for index in range(2)
        ],
    )
    return project_id, run_id, asset_id, frames


def test_continuity_service_persists_only_selected_edge_link_without_payload_hydration(postgres_repo, monkeypatch):
    project_id, run_id, asset_id, frames = _line_scan_asset(postgres_repo)
    frame_zero, frame_one = (str(frame["id"]) for frame in frames)
    candidates = [
        _record(run_id=run_id, frame_id=frame_zero, roi_index=1, x=40, y=90, width=20, height=10),
        _record(run_id=run_id, frame_id=frame_one, roi_index=1, x=43, y=0, width=20, height=10),
        # Neither belongs in the continuity input: one is interior and one is near,
        # rather than on, the joining edge.
        _record(run_id=run_id, frame_id=frame_zero, roi_index=2, x=1, y=40, width=10, height=10),
        _record(run_id=run_id, frame_id=frame_zero, roi_index=3, x=70, y=89, width=10, height=10),
    ]
    candidate_rows = postgres_repo.replace_frame_detections(
        run_id, [frame_zero, frame_one], candidates, project_id=project_id,
    )
    refined = postgres_repo.upsert_refined_detections(
        [(str(candidate["id"]), record) for candidate, record in zip(candidate_rows, candidates)],
        project_id=project_id,
    )
    source_id, target_id = (str(refined[index]["id"]) for index in (0, 1))

    other_project = postgres_repo.create_project(f"other-{uuid.uuid4().hex}")
    assert postgres_repo.list_refined_detections_for_continuity(
        asset_id, project_id=str(other_project["id"]), scan_axis="y",
    ) == []
    edge_rows = postgres_repo.list_refined_detections_for_continuity(
        asset_id, project_id=project_id, scan_axis="y",
    )
    assert len(edge_rows) == 2
    assert all(set(row["metadata"]) <= {"dominant_edge_angle_degrees", "edge_strength"} for row in edge_rows)

    def unexpected_payload_read(_key: str) -> bytes:
        raise AssertionError("continuity must not read ROI or mask payload objects")

    monkeypatch.setattr(postgres_repo.hot_store, "get_store", unexpected_payload_read)
    summary = assess_refined_roi_connections(
        repository=postgres_repo,
        project_id=project_id,
        asset_ids=[asset_id],
        defaults=RoiContinuityProcessingConfig(),
        run_id=run_id,
    )

    assert summary["runs"][0]["segment_count"] == 2
    assert summary["runs"][0]["accepted_link_count"] == 1
    continuity_run_id = summary["runs"][0]["continuity_run_id"]
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT method, metadata FROM {postgres_repo.schema}.roi_continuity_runs WHERE id = %s",
            (continuity_run_id,),
        )
        run = cursor.fetchone()
        cursor.execute(
            f"""SELECT source_refined_detection_id::text, target_refined_detection_id::text,
                       score, accepted, reason, features
                FROM {postgres_repo.schema}.roi_continuity_links
                WHERE continuity_run_id = %s""",
            (continuity_run_id,),
        )
        links = cursor.fetchall()
        cursor.execute(
            f"SELECT metadata FROM {postgres_repo.schema}.roi_continuity_assemblies WHERE continuity_run_id = %s",
            (continuity_run_id,),
        )
        assemblies = cursor.fetchall()

    assert run["method"] == "line_scan_continuity_v2"
    assert run["metadata"]["input_scope"] == "asset_frame_edges"
    assert links == [{
        "source_refined_detection_id": source_id,
        "target_refined_detection_id": target_id,
        "score": 0.87578125,
        "accepted": True,
        "reason": None,
        "features": {},
    }]
    assert len(assemblies) == 1
    assert assemblies[0]["metadata"]["segment_ids"] == [source_id, target_id]
    assert assemblies[0]["metadata"]["link_pairs"] == [[source_id, target_id]]
