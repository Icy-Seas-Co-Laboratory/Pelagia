import pytest

from Pelagia.processing.roi_continuity import (
    FrameLocalRoiSegment,
    LineScanGeometry,
    RoiContinuityOptions,
    assemble_line_scan_continuity,
)


def _segment(
    segment_id: str,
    frame_index: int,
    *,
    x: int = 40,
    y: int = 0,
    width: int = 20,
    height: int = 10,
    angle: float | None = 45.0,
    strength: float | None = 0.9,
) -> FrameLocalRoiSegment:
    return FrameLocalRoiSegment(
        segment_id=segment_id,
        frame_id=f"frame-{frame_index}",
        asset_id="asset-1",
        frame_index=frame_index,
        frame_width=100,
        frame_height=100,
        bbox_x=x,
        bbox_y=y,
        bbox_w=width,
        bbox_h=height,
        edge_angle_degrees=angle,
        edge_strength=strength,
    )


def test_continuity_links_adjacent_boundary_segments_and_retains_audit_evidence():
    source = _segment("refined-a", 10, y=90, x=40)
    target = _segment("refined-b", 11, y=0, x=43)

    result = assemble_line_scan_continuity(
        [target, source],
        geometry=LineScanGeometry(declared=True, scan_axis="y"),
        options=RoiContinuityOptions(min_link_score=0.7),
    )

    assert len(result.accepted_links) == 1
    link = result.accepted_links[0]
    assert (link.source_segment_id, link.target_segment_id) == ("refined-a", "refined-b")
    assert link.features["transverse_overlap"] == pytest.approx(0.85)
    assert link.features["source_boundary_distance"] == 0
    assert link.features["target_boundary_distance"] == 0
    assert len(result.assemblies) == 1
    assert result.assemblies[0].segment_ids == ("refined-a", "refined-b")
    assert result.assemblies[0].link_pairs == (("refined-a", "refined-b"),)


@pytest.mark.parametrize("angle", [0.0, 90.0, 180.0])
def test_continuity_rejects_literal_axis_aligned_edges(angle):
    result = assemble_line_scan_continuity(
        [_segment("refined-a", 0, y=90, angle=angle), _segment("refined-b", 1, y=0)],
        geometry=LineScanGeometry(declared=True, scan_axis="y"),
    )

    assert result.accepted_links == ()
    assert result.decisions[0].reason == "axis_aligned_edge"
    assert result.decisions[0].features["source_axis_aligned"] is True


def test_continuity_only_considers_adjacent_frames():
    result = assemble_line_scan_continuity(
        [_segment("refined-a", 0, y=90), _segment("refined-c", 2, y=0)],
        geometry=LineScanGeometry(declared=True, scan_axis="y"),
    )

    assert result.decisions == ()
    assert result.assemblies == ()


def test_continuity_emits_rejected_pair_audit_for_boundary_mismatch():
    result = assemble_line_scan_continuity(
        [_segment("refined-a", 0, y=50), _segment("refined-b", 1, y=0)],
        geometry=LineScanGeometry(declared=True, scan_axis="y"),
    )

    assert result.accepted_links == ()
    assert result.decisions[0].reason == "outside_boundary_band"
    assert result.decisions[0].features["source_boundary_distance"] == 40


def test_continuity_requires_an_explicit_line_scan_declaration():
    with pytest.raises(ValueError, match="declared line-scan geometry"):
        LineScanGeometry(declared=False, scan_axis="y")


def test_continuity_association_is_one_to_one_and_audits_competing_link():
    source = _segment("refined-a", 0, y=90, x=40)
    stronger_target = _segment("refined-b", 1, y=0, x=40)
    competing_target = _segment("refined-c", 1, y=0, x=42)

    result = assemble_line_scan_continuity(
        [source, competing_target, stronger_target],
        geometry=LineScanGeometry(declared=True, scan_axis="y"),
    )

    assert [(link.source_segment_id, link.target_segment_id) for link in result.accepted_links] == [
        ("refined-a", "refined-b")
    ]
    assert any(decision.reason == "competing_association" for decision in result.decisions)
