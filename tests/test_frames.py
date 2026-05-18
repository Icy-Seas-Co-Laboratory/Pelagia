import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from Pelagia.config import CoreConfig, ImageDataStorageConfig
from Pelagia.processing import video_ingest as video_ingest_module
from Pelagia.processing.frame_correction import _flatfield_correction_for_framedata
from Pelagia.processing.frame_time import _parse_filename_timestamp_utc
from Pelagia.processing.video_ingest import convert_frame_to_grayscale
from Pelagia.processing.frames import (
    Frame,
    ingest_video_file,
    retrieve_frame,
    store_frame,
)


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
        self.row = {"id": 1, "frame_hash": "fake-kv-key"}

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
    assert _parse_filename_timestamp_utc(filename) == expected


def test_flatfield_correction_uses_column_profile_for_grayscale_data():
    data = np.array([[10, 20], [30, 80]], dtype=np.uint8)

    corrected = _flatfield_correction_for_framedata(data, q=0.5)

    np.testing.assert_array_equal(corrected, np.array([[127, 102], [255, 255]], dtype=np.uint8))


def test_store_frame_writes_numpy_payload_and_metadata():
    ctx = FakeContext()
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = Frame(
        sourcePath="/tmp/",
        destPath="/tmp/out",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
        },
    )

    row = store_frame(frame, context=ctx)

    assert row["frame_hash"] == "fake-kv-key"
    assert ctx.kvstore.payload.startswith(b"\x89PNG\r\n\x1a\n")

    params = ctx.repository.cursor_obj.params
    assert params[2] == 7          # frame_index
    assert params[4] == 4          # width
    assert params[5] == 3          # height
    assert params[7] == "fake-kv-key"

    metadata = json.loads(params[9])
    assert metadata["kvstore_encoding"] == "png"
    assert metadata["kvstore_format"] == "png"


def test_store_frame_can_write_raw_numpy_payload():
    ctx = FakeContext()
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = Frame(
        sourcePath="/tmp/",
        destPath="/tmp/out",
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
    metadata = json.loads(ctx.repository.cursor_obj.params[9])
    assert metadata["kvstore_encoding"] == "raw"
    assert metadata["kvstore_format"] == "raw_ndarray_c_order"


def test_store_frame_uses_configured_default_image_data_storage_encoding():
    ctx = FakeContext()
    ctx.config.image_data_storage = ImageDataStorageConfig(encoding="raw")
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = Frame(
        sourcePath="/tmp/",
        destPath="/tmp/out",
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
    metadata = json.loads(ctx.repository.cursor_obj.params[9])
    assert metadata["kvstore_encoding"] == "raw"
    assert metadata["kvstore_format"] == "raw_ndarray_c_order"


def test_store_frame_applies_flatfield_correction_when_requested():
    ctx = FakeContext()
    data = np.array([[10, 20], [30, 80]], dtype=np.uint8)

    frame = Frame(
        sourcePath="/tmp/",
        destPath="/tmp/out",
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
        np.array([[127, 102], [255, 255]], dtype=np.uint8),
    )
    metadata = json.loads(ctx.repository.cursor_obj.params[9])
    assert metadata["flatfield_correction"] is True
    assert metadata["flatfield_q"] == 0.5


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
        video_ingest_module.cv2.cvtColor(bgr, video_ingest_module.cv2.COLOR_BGR2GRAY),
    )
    np.testing.assert_array_equal(
        convert_frame_to_grayscale(bgra),
        video_ingest_module.cv2.cvtColor(bgra, video_ingest_module.cv2.COLOR_BGRA2GRAY),
    )


def test_store_frame_can_write_zstd_numpy_payload():
    zstd = pytest.importorskip("zstandard")

    ctx = FakeContext()
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    frame = Frame(
        sourcePath="/tmp/",
        destPath="/tmp/out",
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
    metadata = json.loads(ctx.repository.cursor_obj.params[9])
    assert metadata["kvstore_encoding"] == "zstd"
    assert metadata["kvstore_format"] == "zstd_ndarray_c_order"


def test_retrieve_frame_reconstructs_frame_from_metadata_and_kvstore():
    ctx = FakeContext()
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    stored = Frame(
        sourcePath="/tmp/source",
        destPath="/tmp/out",
        filename="frame.png",
        frameNumber=7,
        data=data,
        tileNumber=3,
        sourceFrameStart=5,
        sourceFrameEnd=7,
        frameType="line",
        channel=2,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "kvstore_encoding": "raw",
        },
    )
    store_frame(stored, context=ctx)
    metadata = json.loads(ctx.repository.cursor_obj.params[9])
    ctx.repository.cursor_obj.row = {
        "id": 42,
        "run_id": "00000000-0000-0000-0000-000000000001",
        "asset_id": "00000000-0000-0000-0000-000000000002",
        "frame_index": 3,
        "captured_at": None,
        "source_ref": "/tmp/source/frame.png",
        "frame_hash": "fake-kv-key",
        "metadata": metadata,
    }

    retrieved = retrieve_frame(42, context=ctx)

    assert retrieved.frameNumber == 7
    assert retrieved.tileNumber == 3
    assert retrieved.sourceFrameStart == 5
    assert retrieved.sourceFrameEnd == 7
    assert retrieved.frameType == "line"
    assert retrieved.channel == 2
    assert retrieved.sourcePath == "/tmp/source"
    assert retrieved.destPath == "/tmp/out"
    assert retrieved.filename == "frame.png"
    np.testing.assert_array_equal(retrieved.data, data)
    assert retrieved.metadata["frame_id"] == 42


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
            if prop == video_ingest_module.cv2.CAP_PROP_FPS:
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

    monkeypatch.setattr(video_ingest_module.cv2, "VideoCapture", FakeVideoCapture)
    ctx = FakeContext()

    ingest_video_file(
        "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw"},
        flatfield_correction=False,
    )

    first_params, second_params = ctx.repository.cursor_obj.params_history
    start = datetime(2025, 11, 10, 2, 21, 32, 482000, tzinfo=timezone.utc)
    assert first_params[3] == start
    assert second_params[3] == start + timedelta(seconds=0.05)

    first_metadata = json.loads(first_params[9])
    assert first_metadata["source_timestamp_utc"] == "2025-11-10T02:21:32.482000+00:00"
    assert first_metadata["fps"] == 20.0
    assert first_metadata["frame_interval_seconds"] == 0.05


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
            return 20.0 if prop == video_ingest_module.cv2.CAP_PROP_FPS else 0.0

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(video_ingest_module.cv2, "VideoCapture", FakeVideoCapture)
    ctx = FakeContext()

    ingest_video_file(
        "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw", "flatfield_correction": False},
        flatfield_correction=False,
    )

    assert len(ctx.kvstore.payload) == 2
