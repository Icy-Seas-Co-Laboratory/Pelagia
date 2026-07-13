import numpy as np
import pytest

from Pelagia.domain import PipelineStage
from Pelagia.processing.codec_registry import (
    encode_image_response,
    image_extension,
    image_media_type,
    normalize_image_encoding,
)
from Pelagia.services.job_commands import (
    COMMAND_TYPE_KEY,
    COMMAND_VERSION_KEY,
    FrameSelection,
    SegmentFramesCommand,
    command_from_payload,
)


def test_segment_command_upgrades_a_legacy_flat_payload():
    command = command_from_payload(
        PipelineStage.SEGMENT,
        {"frame_ids": ["frame-1"], "padding": 8, "roi_encoding": "png"},
    )

    payload = command.to_payload()

    assert isinstance(command, SegmentFramesCommand)
    assert payload[COMMAND_TYPE_KEY] == "segment_frames"
    assert payload[COMMAND_VERSION_KEY] == 1
    assert payload["frame_ids"] == ["frame-1"]
    assert payload["padding"] == 8
    assert payload["roi_encoding"] == "png"


def test_command_rejects_a_payload_for_another_stage():
    with pytest.raises(ValueError, match="Expected 'segment_frames' command payload"):
        SegmentFramesCommand.from_payload({COMMAND_TYPE_KEY: "preprocess_frames"})


def test_codec_registry_normalizes_aliases_and_encodes_png():
    assert normalize_image_encoding("jpeg-xs") == "jxs"
    assert image_extension("jpeg") == "jpg"
    assert image_media_type("jpeg") == "image/jpeg"

    payload, media_type = encode_image_response(np.array([[0, 255]], dtype=np.uint8), "png")

    assert media_type == "image/png"
    assert payload.startswith(b"\x89PNG")


def test_segment_command_serializes_explicit_selection():
    payload = SegmentFramesCommand(
        selection=FrameSelection(frame_ids=("frame-1",), asset_id="asset-1"),
        options={"roi_encoding": "zstd"},
    ).to_payload()

    assert payload["asset_id"] == "asset-1"
    assert payload["roi_encoding"] == "zstd"
