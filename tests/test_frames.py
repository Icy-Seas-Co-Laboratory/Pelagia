import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from Pelagia.config import CoreConfig
from Pelagia.domain import FrameRecord
from Pelagia.processing import ingest as ingest_module
from Pelagia.processing.frame_codec import decode_array_payload
from Pelagia.processing.frame_correction import (
    apply_flatfield_correction,
    flatfield_correction_for_framedata,
    flatfield_global_correction_for_framedata,
)
from Pelagia.processing.frame_model import FrameData
from Pelagia.processing.frame_store import retrieve_frame, store_frame
from Pelagia.processing.frame_time import parse_filename_timestamp_utc
from Pelagia.processing.thumbhash import compute_thumbhash
from Pelagia.processing.ingest import convert_frame_to_grayscale, discover_ingest_sources
from Pelagia.processing.ingest import ingest_image_folder, ingest_video_file


FRAME_ID = "00000000-0000-7000-8000-000000000042"
PARENT_FRAME_ID = "00000000-0000-7000-8000-000000000099"


class FakeKVStore:
    initialized = True

    def __init__(self):
        self.payload = None

    def put_store(self, payload):
        self.payload = payload
        return "fake-kv-key"

    def get_store(self, key):
        if key != "fake-kv-key":
            raise KeyError(key)
        return self.payload


class FakeCursor:
    def __init__(self):
        self.params = None
        self.params_history = []
        self.row = {"id": FRAME_ID, "kvstore_hash": "fake-kv-key"}

    def execute(self, query, params):
        self.params = params
        self.params_history.append(params)

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeRepository:
    schema = "pelagia"

    def __init__(self):
        self.cursor_obj = FakeCursor()

    def connect(self):
        return FakeConnection(self.cursor_obj)


class FakeContext:
    def __init__(self):
        self.kvstore = FakeKVStore()
        self.repository = FakeRepository()
        self.config = CoreConfig()


class FakeDatabaseLogger:
    def __init__(self):
        self.events = []

    def log(self, **kwargs):
        self.events.append(kwargs)
        return {"id": len(self.events), **kwargs}


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "Camera-00002-2025-11-10 02-21-32.482.mkv",
            datetime(2025, 11, 10, 2, 21, 32, 482000, tzinfo=timezone.utc),
        ),
        (
            "Camera3_VIPF-342-2022-07-22-22-00-05.291.avi",
            datetime(2022, 7, 22, 22, 0, 5, 291000, tzinfo=timezone.utc),
        ),
        (
            "Camera2_VIPF-256-2022-07-22-02-57-53.315-16-av1.mkv",
            datetime(2022, 7, 22, 2, 57, 53, 315000, tzinfo=timezone.utc),
        ),
        (
            "Camera2_VIPF-256-2022-07-22-02-57-53.315-6-av1.mkv",
            datetime(2022, 7, 22, 2, 57, 53, 315000, tzinfo=timezone.utc),
        ),
        (
            "Camera-00006-2025-11-14 05-45-57.098.mkv",
            datetime(2025, 11, 14, 5, 45, 57, 98000, tzinfo=timezone.utc),
        ),
    ],
)
def test_parse_filename_timestamp_utc_handles_camera_names(filename, expected):
    assert parse_filename_timestamp_utc(filename) == expected


def test_flatfield_correction_uses_column_profile_for_grayscale_data():
    data = np.array([[10, 20], [30, 80]], dtype=np.uint8)

    corrected = flatfield_correction_for_framedata(data, q=0.5)
    global_corrected = flatfield_global_correction_for_framedata(data, q=0.5, axis=0)

    np.testing.assert_array_equal(corrected, np.array([[25, 20], [75, 80]], dtype=np.uint8))
    np.testing.assert_array_equal(corrected, global_corrected)


def test_frame_infers_full_frame_geometry_from_data():
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
    )

    assert frame.width == 4
    assert frame.height == 3
    assert frame.bbox_x == 0
    assert frame.bbox_y == 0
    assert frame.get_size() == (4, 3)
    assert frame.get_bbox() == (0, 0, 4, 3)
    assert frame.bounds == (0, 0, 4, 3)


def test_frame_preserves_roi_geometry():
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    mask = np.full((2, 3), 255, dtype=np.uint8)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        mask=mask,
        width=3,
        height=2,
        bbox_x=10,
        bbox_y=20,
        parent_frame_id=42,
    )

    assert frame.get_size() == (3, 2)
    assert frame.get_bbox() == (10, 20, 3, 2)
    assert frame.get_bounds() == (10, 20, 13, 22)
    assert frame.parent_frame_id == 42
    np.testing.assert_array_equal(frame.get_mask(), mask)


def test_frame_validate_mask_rejects_mismatched_dimensions():
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    mask = np.full((2, 4), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="mask width"):
        FrameData(
            sourcePath="/tmp/",
            filename="frame.png",
            frameNumber=7,
            data=data,
            mask=mask,
        )


def test_frame_validate_geometry_rejects_mismatched_dimensions():
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        width=4,
        height=2,
    )

    with pytest.raises(ValueError, match="width"):
        frame.validate_geometry()


def test_store_frame_writes_numpy_payload_and_metadata():
    ctx = FakeContext()
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "kvstore_encoding": "png",
        },
    )

    row = store_frame(frame, context=ctx)

    assert row["kvstore_hash"] == "fake-kv-key"
    assert ctx.kvstore.payload.startswith(b"\x89PNG\r\n\x1a\n")

    params = ctx.repository.cursor_obj.params
    assert params[2] == 7          # frame_index
    assert params[4] == 4          # width
    assert params[5] == 3          # height
    assert params[10] == "fake-kv-key"
    assert params[11] == compute_thumbhash(data)
    assert params[12] == "fake-kv-key"
    assert params[13] == "png"

    metadata = json.loads(params[17])
    assert metadata["kvstore_encoding"] == "png"
    assert metadata["kvstore_format"] == "png"
    assert metadata["width"] == 4
    assert metadata["height"] == 3
    assert metadata["bbox_x"] == 0
    assert metadata["bbox_y"] == 0


def test_store_frame_writes_roi_geometry_metadata():
    ctx = FakeContext()
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        width=3,
        height=2,
        bbox_x=10,
        bbox_y=20,
        parent_frame_id=42,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
        },
    )

    store_frame(frame, context=ctx)

    params = ctx.repository.cursor_obj.params
    assert params[4] == 3
    assert params[5] == 2
    assert params[6] == 10
    assert params[7] == 20
    assert params[8] == 42

    metadata = json.loads(params[17])
    assert metadata["width"] == 3
    assert metadata["height"] == 2
    assert metadata["bbox_x"] == 10
    assert metadata["bbox_y"] == 20
    assert metadata["parent_frame_id"] == 42


def test_store_frame_can_write_raw_numpy_payload():
    ctx = FakeContext()
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "kvstore_encoding": "raw",
        },
    )

    store_frame(frame, context=ctx)

    assert ctx.kvstore.payload == data.tobytes(order="C")
    metadata = json.loads(ctx.repository.cursor_obj.params[17])
    assert metadata["kvstore_encoding"] == "raw"
    assert metadata["kvstore_format"] == "raw_ndarray_c_order"


def test_store_frame_can_write_jpg_payload():
    ctx = FakeContext()
    data = np.arange(100, dtype=np.uint8).reshape(10, 10)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "kvstore_encoding": "jpg",
        },
    )

    store_frame(frame, context=ctx)

    assert ctx.kvstore.payload.startswith(b"\xff\xd8")
    metadata = json.loads(ctx.repository.cursor_obj.params[17])
    assert metadata["kvstore_encoding"] == "jpg"
    assert metadata["kvstore_format"] == "jpg"
    decoded = decode_array_payload(ctx.kvstore.payload, metadata)
    assert decoded.shape == data.shape
    assert decoded.dtype == np.uint8


def test_store_frame_uses_configured_default_image_data_storage_encoding():
    ctx = FakeContext()
    ctx.config.processing.frame_storage.image_encoding = "raw"
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
        },
    )

    store_frame(frame, context=ctx)

    assert ctx.kvstore.payload == data.tobytes(order="C")
    metadata = json.loads(ctx.repository.cursor_obj.params[17])
    assert metadata["kvstore_encoding"] == "raw"
    assert metadata["kvstore_format"] == "raw_ndarray_c_order"


def test_store_frame_preserves_data_when_flatfield_metadata_is_present():
    ctx = FakeContext()
    data = np.array([[10, 20], [30, 80]], dtype=np.uint8)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "kvstore_encoding": "raw",
            "flatfield_correction": True,
            "flatfield_q": 0.5,
        },
    )

    store_frame(frame, context=ctx)

    np.testing.assert_array_equal(
        np.frombuffer(ctx.kvstore.payload, dtype=np.uint8).reshape(2, 2),
        data,
    )
    metadata = json.loads(ctx.repository.cursor_obj.params[17])
    assert metadata["flatfield_correction"] is True
    assert metadata["flatfield_q"] == 0.5
    assert "flatfield_maximum_value" not in metadata


def test_apply_flatfield_correction_returns_corrected_runtime_frame():
    data = np.array([[10, 20], [30, 80]], dtype=np.uint8)
    record = FrameRecord(
        id=FRAME_ID,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        frame_index=7,
        width=2,
        height=2,
        kvstore_hash="kvstore-key",
        preview_thumbhash=b"",
    )
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": record.run_id,
            "asset_id": record.asset_id,
            "frame_id": record.id,
        },
    )

    corrected = apply_flatfield_correction(record, frame=frame, q=0.5, axis=0)

    np.testing.assert_array_equal(
        corrected.read(),
        np.array([[25, 20], [75, 80]], dtype=np.uint8),
    )
    assert corrected.metadata["flatfield_correction"] is True
    assert corrected.metadata["flatfield_q"] == 0.5
    assert corrected.metadata["flatfield_axis"] == 0
    assert corrected.metadata["flatfield_source_frame_id"] == FRAME_ID


def test_convert_frame_to_grayscale_keeps_existing_grayscale_frame():
    frame = np.arange(6, dtype=np.uint8).reshape(2, 3)

    converted = convert_frame_to_grayscale(frame)

    np.testing.assert_array_equal(converted, frame)
    assert converted.flags.c_contiguous


def test_convert_frame_to_grayscale_converts_bgr_and_bgra_frames():
    bgr = np.array([[[10, 20, 30], [30, 20, 10]]], dtype=np.uint8)
    bgra = np.array([[[10, 20, 30, 255], [30, 20, 10, 0]]], dtype=np.uint8)

    np.testing.assert_array_equal(
        convert_frame_to_grayscale(bgr),
        ingest_module.cv2.cvtColor(bgr, ingest_module.cv2.COLOR_BGR2GRAY),
    )
    np.testing.assert_array_equal(
        convert_frame_to_grayscale(bgra),
        ingest_module.cv2.cvtColor(bgra, ingest_module.cv2.COLOR_BGRA2GRAY),
    )


def test_store_frame_can_write_zstd_numpy_payload():
    zstd = pytest.importorskip("zstandard")

    ctx = FakeContext()
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "kvstore_encoding": "zstd",
        },
    )

    store_frame(frame, context=ctx)

    assert zstd.ZstdDecompressor().decompress(ctx.kvstore.payload) == data.tobytes(order="C")
    metadata = json.loads(ctx.repository.cursor_obj.params[17])
    assert metadata["kvstore_encoding"] == "zstd"
    assert metadata["kvstore_format"] == "zstd_ndarray_c_order"


def test_retrieve_frame_reconstructs_frame_from_metadata_and_kvstore():
    ctx = FakeContext()
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    stored = FrameData(
        sourcePath="/tmp/source",
        filename="frame.png",
        frameNumber=7,
        data=data,
        tileNumber=3,
        sourceFrameStart=5,
        sourceFrameEnd=7,
        frameType="line",
        channel=2,
        bbox_x=10,
        bbox_y=20,
        parent_frame_id=PARENT_FRAME_ID,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "kvstore_encoding": "raw",
        },
    )
    store_frame(stored, context=ctx)
    metadata = json.loads(ctx.repository.cursor_obj.params[17])
    ctx.repository.cursor_obj.row = {
        "id": FRAME_ID,
        "run_id": "00000000-0000-0000-0000-000000000001",
        "asset_id": "00000000-0000-0000-0000-000000000002",
        "frame_index": 3,
        "captured_at": None,
        "width": 4,
        "height": 3,
        "source_ref": "/tmp/source/frame.png",
        "kvstore_hash": "fake-kv-key",
        "metadata": metadata,
    }

    retrieved = retrieve_frame(FRAME_ID, context=ctx)

    assert retrieved.frameNumber == 7
    assert retrieved.tileNumber == 3
    assert retrieved.sourceFrameStart == 5
    assert retrieved.sourceFrameEnd == 7
    assert retrieved.frameType == "line"
    assert retrieved.channel == 2
    assert retrieved.width == 4
    assert retrieved.height == 3
    assert retrieved.get_bbox() == (10, 20, 4, 3)
    assert retrieved.parent_frame_id == PARENT_FRAME_ID
    assert retrieved.sourcePath == "/tmp/source"
    assert retrieved.filename == "frame.png"
    np.testing.assert_array_equal(retrieved.data, data)
    assert retrieved.metadata["frame_id"] == FRAME_ID


def test_frame_data_from_record_maps_row_model_to_runtime_container():
    record = FrameRecord(
        id=FRAME_ID,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        frame_index=3,
        captured_at=None,
        width=4,
        height=3,
        bbox_x=10,
        bbox_y=20,
        parent_frame_id=PARENT_FRAME_ID,
        source_ref="/tmp/source/frame.png",
        kvstore_hash="fake-kv-key",
        preview_thumbhash=b"PTH1preview",
        payload_ref="fake-kv-key",
        payload_encoding="raw",
        payload_format="raw_ndarray_c_order",
        payload_dtype="uint8",
        payload_shape=[3, 4],
        metadata={
            "frame_number": 7,
            "tile_number": 3,
            "source_frame_start": 5,
            "source_frame_end": 7,
            "frame_type": "line",
            "channel": 2,
        },
    )
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame_data = FrameData.from_record(record, data=data)

    assert frame_data.frameNumber == 7
    assert frame_data.tileNumber == 3
    assert frame_data.sourceFrameStart == 5
    assert frame_data.sourceFrameEnd == 7
    assert frame_data.frameType == "line"
    assert frame_data.channel == 2
    assert frame_data.get_bbox() == (10, 20, 4, 3)
    assert frame_data.parent_frame_id == PARENT_FRAME_ID
    assert frame_data.sourcePath == "/tmp/source"
    assert frame_data.filename == "frame.png"
    assert frame_data.metadata["frame_id"] == FRAME_ID
    assert frame_data.metadata["kvstore_encoding"] == "raw"
    assert frame_data.metadata["shape"] == [3, 4]
    np.testing.assert_array_equal(frame_data.data, data)


def test_ingest_video_file_timestamps_frames_from_filename_and_fps(monkeypatch):
    class FakeVideoCapture:
        def __init__(self, path):
            self.frames = [
                np.zeros((2, 2), dtype=np.uint8),
                np.ones((2, 2), dtype=np.uint8),
            ]
            self.index = 0

        def isOpened(self):
            return self.index <= len(self.frames)

        def get(self, prop):
            if prop == ingest_module.cv2.CAP_PROP_FPS:
                return 20.0
            return 0.0

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(ingest_module.cv2, "VideoCapture", FakeVideoCapture)
    ctx = FakeContext()

    ingest_video_file(
        "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
        n_tile=1,
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw"},
    )

    first_params, second_params = ctx.repository.cursor_obj.params_history
    start = datetime(2025, 11, 10, 2, 21, 32, 482000, tzinfo=timezone.utc)
    assert first_params[3] == start
    assert second_params[3] == start + timedelta(seconds=0.05)

    first_metadata = json.loads(first_params[17])
    assert first_metadata["source_timestamp_utc"] == "2025-11-10T02:21:32.482000+00:00"
    assert first_metadata["fps"] == 20.0
    assert first_metadata["frame_interval_seconds"] == 0.05


def test_ingest_video_file_writes_structured_log_events(monkeypatch):
    class FakeVideoCapture:
        def __init__(self, path):
            self.frames = [
                np.zeros((2, 2), dtype=np.uint8),
                np.ones((2, 2), dtype=np.uint8),
            ]
            self.index = 0

        def isOpened(self):
            return self.index <= len(self.frames)

        def get(self, prop):
            return 20.0 if prop == ingest_module.cv2.CAP_PROP_FPS else 0.0

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(ingest_module.cv2, "VideoCapture", FakeVideoCapture)
    ctx = FakeContext()
    ctx.logger = FakeDatabaseLogger()

    ingest_video_file(
        "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
        n_tile=2,
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw"},
    )

    event_types = [event["event_type"] for event in ctx.logger.events]
    assert event_types == [
        "video_ingest.started",
        "video_ingest.video_opened",
        "video_ingest.tile_stored",
        "video_ingest.completed",
    ]
    completed = ctx.logger.events[-1]
    assert completed["duration_ms"] >= 0
    assert completed["run_id"] == "00000000-0000-0000-0000-000000000001"
    assert completed["asset_id"] == "00000000-0000-0000-0000-000000000002"
    assert completed["payload"]["source_frame_count"] == 2
    assert completed["payload"]["stored_tile_count"] == 1


def test_ingest_video_file_converts_color_frames_to_grayscale(monkeypatch):
    class FakeVideoCapture:
        def __init__(self, path):
            self.frames = [
                np.array([[[10, 20, 30], [30, 20, 10]]], dtype=np.uint8),
            ]
            self.index = 0

        def isOpened(self):
            return self.index <= len(self.frames)

        def get(self, prop):
            return 20.0 if prop == ingest_module.cv2.CAP_PROP_FPS else 0.0

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(ingest_module.cv2, "VideoCapture", FakeVideoCapture)
    ctx = FakeContext()

    ingest_video_file(
        "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw"},
    )

    assert len(ctx.kvstore.payload) == 2


def test_ingest_image_folder_stores_supported_images_in_order(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    ingest_module.cv2.imwrite(str(image_dir / "frame_002.jpg"), np.full((2, 2), 20, dtype=np.uint8))
    ingest_module.cv2.imwrite(str(image_dir / "frame_001.png"), np.full((2, 2), 10, dtype=np.uint8))
    (image_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    ctx = FakeContext()

    rows = ingest_image_folder(
        image_dir,
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw"},
    )

    assert len(rows) == 2
    first_params, second_params = ctx.repository.cursor_obj.params_history
    first_metadata = json.loads(first_params[17])
    second_metadata = json.loads(second_params[17])
    assert first_params[2] == 1
    assert second_params[2] == 2
    assert first_metadata["source_image_filename"] == "frame_001.png"
    assert second_metadata["source_image_filename"] == "frame_002.jpg"
    assert first_metadata["source_type"] == "image_folder"


def test_discover_ingest_sources_finds_image_folders_and_video_files(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    (image_dir / "frame_001.png").write_bytes(b"not-a-real-image")
    video_path = tmp_path / "Camera-00001-2025-11-10 02-21-32.482.mkv"
    video_path.write_bytes(b"not-a-real-video")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    sources = discover_ingest_sources(tmp_path)

    assert [source.kind for source in sources] == ["image_sequence", "video"]
    assert [source.path for source in sources] == [image_dir.resolve(), video_path.resolve()]
