import io
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from Pelagia.config import CoreConfig
from Pelagia.domain import FrameRecord
from Pelagia.processing import frame_codec
from Pelagia.processing import ingest as ingest_module
from Pelagia.processing.frame_codec import decode_array_payload, encode_array_payload
from Pelagia.processing.frame_correction import (
    _bounded_field,
    _divide_by_field,
    apply_flatfield_correction,
    divide_background,
    ensure_asset_background_windows,
    flatfield_correction,
    flatfield_profile_correction,
    generate_background_for_frames,
)
from Pelagia.processing.frame_model import FrameData
from Pelagia.processing.frame_preprocess import preprocess_frame_for_segmentation
from Pelagia.processing.frame_store import _fit_background_to_frame, retrieve_frame, store_frame
from Pelagia.processing.frame_time import parse_filename_timestamp_utc
from Pelagia.processing.thumbhash import compute_thumbhash
from Pelagia.services.context import AppContext
from Pelagia.storage.kvstore import KVStore
from Pelagia.storage.kvstore2 import KVStore2
from Pelagia.storage.postgres import DEFAULT_PROJECT_ID
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


class PartitionedCursor:
    def __init__(self, repository):
        self.repository = repository
        self.params = None
        self.row = None

    def execute(self, query, params):
        self.params = params
        normalized_query = " ".join(str(query).split()).lower()
        if normalized_query.startswith("insert into"):
            frame_id = f"00000000-0000-7000-8000-{len(self.repository.frames) + 1:012d}"
            asset = self.repository.assets[str(params[1])]
            row = {
                "id": frame_id,
                "run_id": params[0],
                "asset_id": params[1],
                "frame_index": params[2],
                "captured_at": params[3],
                "width": params[4],
                "height": params[5],
                "bbox_x": params[6],
                "bbox_y": params[7],
                "parent_frame_id": params[8],
                "source_ref": params[9],
                "kvstore_hash": params[10],
                "preview_thumbhash": params[11],
                "payload_ref": params[12],
                "payload_encoding": params[13],
                "payload_format": params[14],
                "payload_dtype": params[15],
                "payload_shape": json.loads(params[16]),
                "metadata": json.loads(params[17]),
                "project_id": asset["project_id"],
            }
            self.repository.frames[frame_id] = row
            self.row = row
            return
        if "from pelagia.frames frames" in normalized_query:
            frame_id = str(params[0])
            row = self.repository.frames.get(frame_id)
            if row is not None and len(params) > 1 and str(row["project_id"]) != str(params[1]):
                row = None
            self.row = row
            return
        self.row = None

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class PartitionedConnection:
    def __init__(self, repository):
        self.repository = repository

    def cursor(self):
        return PartitionedCursor(self.repository)

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class PartitionedRepository:
    schema = "pelagia"

    def __init__(self, project_id, project_root=None):
        self.project_id = str(project_id)
        self.project_root = project_root
        self.assets = {
            "asset-project": {"id": "asset-project", "project_id": self.project_id},
            "asset-default": {"id": "asset-default", "project_id": DEFAULT_PROJECT_ID},
        }
        self.frames = {}

    def get_project(self, project_id):
        if str(project_id) == self.project_id:
            return {
                "id": self.project_id,
                "kvstore_root_path": None if self.project_root is None else str(self.project_root),
            }
        if str(project_id) == DEFAULT_PROJECT_ID:
            return {"id": DEFAULT_PROJECT_ID, "kvstore_root_path": None}
        return None

    def get_asset(self, asset_id, *, project_id=None):
        asset = self.assets.get(str(asset_id))
        if asset is None:
            return None
        if project_id is not None and str(asset["project_id"]) != str(project_id):
            return None
        return dict(asset)

    def get_frame(self, frame_id, *, project_id=None):
        row = self.frames.get(str(frame_id))
        if row is None:
            return None
        if project_id is not None and str(row["project_id"]) != str(project_id):
            return None
        return dict(row)

    def connect(self):
        return PartitionedConnection(self)


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

    corrected = flatfield_correction(data, q=0.5, axis=0)
    row_corrected = flatfield_correction(data, q=0.5, axis=1)

    np.testing.assert_array_equal(corrected, np.array([[127, 102], [255, 255]], dtype=np.uint8))
    np.testing.assert_array_equal(row_corrected, np.array([[170, 255], [139, 255]], dtype=np.uint8))


def test_stored_flatfield_profile_correction_uses_column_mean_vector():
    corrected = flatfield_profile_correction(
        np.array([[10, 20], [30, 40]], dtype=np.uint8),
        [20.0, 30.0],
    )

    np.testing.assert_array_equal(
        corrected,
        np.array([[127, 170], [255, 255]], dtype=np.uint8),
    )


def test_stored_flatfield_profile_correction_uses_row_mean_vector():
    corrected = flatfield_profile_correction(
        np.array([[10, 20], [30, 40]], dtype=np.uint8),
        [15.0, 35.0],
        axis=1,
    )

    np.testing.assert_array_equal(
        corrected,
        np.array([[170, 255], [218, 255]], dtype=np.uint8),
    )

def test_preprocessing_prefers_stored_flatfield_profile():
    frame = FrameData(
        sourcePath="/tmp",
        filename="frame.png",
        frameNumber=1,
        data=np.array([[10, 20], [30, 40]], dtype=np.uint8),
        metadata={"flatfield_profile": [20.0, 30.0]},
    )

    processed = preprocess_frame_for_segmentation(
        frame,
        flatfield_correction=True,
        flatfield_min_field_value=1,
        background_correction=False,
        invert_intensity=False,
    )

    np.testing.assert_array_equal(
        processed.read(),
        np.array([[127, 170], [255, 255]], dtype=np.uint8),
    )
    assert processed.metadata["flatfield_profile_method"] == "window_column_mean"


def test_preprocessing_uses_stored_row_profile_for_axis_one():
    frame = FrameData(
        sourcePath="/tmp",
        filename="frame.png",
        frameNumber=1,
        data=np.array([[10, 20], [30, 40]], dtype=np.uint8),
        metadata={
            "flatfield_profile": [15.0, 35.0],
            "flatfield_metadata": {"flatfield_axis": 1},
        },
    )

    processed = preprocess_frame_for_segmentation(
        frame,
        flatfield_correction=True,
        flatfield_min_field_value=1,
        background_correction=False,
        invert_intensity=False,
    )

    np.testing.assert_array_equal(
        processed.read(),
        np.array([[170, 255], [218, 255]], dtype=np.uint8),
    )
    assert processed.metadata["flatfield_profile_method"] == "window_row_mean"
    assert processed.metadata["flatfield_axis"] == 1


def test_bounded_field_marks_out_of_range_values_with_sentinel_divisors():
    field = np.array([5, 20, 300], dtype=np.float32)

    bounded = _bounded_field(field, min_field_value=10, max_field_value=200)

    np.testing.assert_array_equal(
        bounded,
        np.array([0, 20, 255], dtype=np.float32),
    )


def test_divide_by_field_maps_zero_divisor_to_255():
    data = np.array([[10, 10, 10]], dtype=np.uint8)
    field = np.array([0, 10, 255], dtype=np.float32)

    corrected = _divide_by_field(data, field)

    np.testing.assert_array_equal(corrected, np.array([[255, 255, 10]], dtype=np.uint8))


def test_divide_by_field_maps_zero_divisor_to_255_for_float_data():
    corrected = _divide_by_field(
        np.array([10.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(corrected, np.array([255.0], dtype=np.float32))


def test_divide_background_promotes_uint8_field_for_calculation():
    data = np.array([[50, 100, 200]], dtype=np.uint8)
    background = np.array([[50, 100, 200]], dtype=np.uint8)

    corrected = divide_background(data, background=background)

    assert corrected.dtype == np.uint8
    np.testing.assert_array_equal(corrected, np.full((1, 3), 255, dtype=np.uint8))


def test_generate_background_for_frames_stores_mean_field(monkeypatch):
    class BackgroundKVStore:
        def __init__(self):
            self.payload = None

        def put_store(self, payload):
            self.payload = payload
            return "background-key"

    class BackgroundRepository:
        def __init__(self):
            self.updated = None

        def update_frame_background_payloads(self, frame_ids, **kwargs):
            self.updated = (frame_ids, kwargs)
            return [{"id": frame_id, **kwargs} for frame_id in frame_ids]

    class BackgroundContext:
        def __init__(self):
            self.kvstore = BackgroundKVStore()
            self.repository = BackgroundRepository()
            self.config = CoreConfig()

    frames = {
        "frame-1": FrameData("/tmp/", "a.png", 1, data=np.array([[10, 20], [30, 40]], dtype=np.uint8)),
        "frame-2": FrameData("/tmp/", "b.png", 2, data=np.array([[33, 43], [53, 63]], dtype=np.uint8)),
    }

    def fake_retrieve_frame(frame_id, *, context, payload_kind):
        assert payload_kind == "original"
        return frames[frame_id]

    monkeypatch.setattr("Pelagia.processing.frame_store.retrieve_frame", fake_retrieve_frame)
    ctx = BackgroundContext()

    result = generate_background_for_frames(["frame-1", "frame-2"], context=ctx)

    assert result["background_payload_ref"] == "background-key"
    assert result["updated_frame_count"] == 2
    assert ctx.repository.updated[0] == ["frame-1", "frame-2"]
    assert ctx.repository.updated[1]["payload_dtype"] == "uint8"
    decoded = decode_array_payload(
        ctx.kvstore.payload,
        {"kvstore_encoding": "zstd", "dtype": "uint8", "shape": [2, 2]},
    )
    np.testing.assert_array_equal(
        decoded,
        np.array([[22, 32], [42, 52]], dtype=np.uint8),
    )


def test_asset_background_windows_use_fixed_stride_and_wider_sources(monkeypatch):
    records = {
        f"frame-{index}": FrameRecord(
            id=f"frame-{index}",
            asset_id="asset-1",
            run_id="run-1",
            frame_index=index - 1,
            width=2,
            height=2,
            kvstore_hash=None,
            preview_thumbhash=None,
        )
        for index in range(1, 18)
    }

    class Repository:
        def get_frame_record(self, frame_id, *, project_id=None):
            return records.get(frame_id)

        def list_frames(self, asset_id, *, project_id=None, limit=None):
            assert asset_id == "asset-1"
            return [
                {
                    "id": record.id,
                    "asset_id": record.asset_id,
                    "run_id": record.run_id,
                    "frame_index": record.frame_index,
                    "width": record.width,
                    "height": record.height,
                }
                for record in records.values()
            ]

    calls = []

    def fake_generate(frame_ids, **kwargs):
        calls.append((frame_ids, kwargs))
        return {"background_payload_ref": "background-key", "updated_frame_count": len(frame_ids)}

    context = SimpleNamespace(
        repository=Repository(),
        config=CoreConfig(),
        active_project_id="project-1",
    )
    monkeypatch.setattr("Pelagia.processing.frame_correction.generate_background_for_frames", fake_generate)

    result = ensure_asset_background_windows(
        ["frame-12"],
        context=context,
        window_stride=5,
        window_width=5,
    )

    assert calls[0][0] == ["frame-9", "frame-10", "frame-11", "frame-12", "frame-13"]
    metadata = calls[0][1]["metadata"]
    assert metadata["background_window_center"] == 10
    assert metadata["background_application_start"] == 8
    assert metadata["background_application_end"] == 12
    assert result["windows"][0]["center"] == 10


def test_asset_background_windows_exclude_partial_frames_and_assign_nominal_background(monkeypatch):
    records = {
        f"frame-{index}": FrameRecord(
            id=f"frame-{index}",
            asset_id="asset-1",
            run_id="run-1",
            frame_index=index,
            width=2048,
            height=4096 if index == 12 else 8192,
            kvstore_hash=f"kv-{index}",
            preview_thumbhash=b"thumb",
        )
        for index in range(8, 13)
    }

    class Repository:
        def __init__(self):
            self.background_updates = []

        def get_frame_records(self, frame_ids, *, project_id=None):
            return [records[frame_id] for frame_id in frame_ids]

        def list_frames(self, asset_id, *, project_id=None, limit=None):
            return [
                {
                    "id": record.id,
                    "asset_id": record.asset_id,
                    "run_id": record.run_id,
                    "frame_index": record.frame_index,
                    "width": record.width,
                    "height": record.height,
                }
                for record in records.values()
            ]

        def update_frame_background_payloads(self, frame_ids, **kwargs):
            self.background_updates.append((list(frame_ids), kwargs))
            return [{"id": frame_id} for frame_id in frame_ids]

    calls = []

    def fake_generate(frame_ids, **kwargs):
        calls.append((list(frame_ids), kwargs["metadata"]))
        metadata = {"background_layout": "nominal_frame", **kwargs["metadata"]}
        return {
            "background_payload_ref": "background-key",
            "background_payload_encoding": "zstd",
            "background_payload_format": "zstd_ndarray_c_order",
            "background_payload_dtype": "uint8",
            "background_payload_shape": [8192, 2048],
            "background_metadata": metadata,
            "updated_frame_count": len(frame_ids),
        }

    repository = Repository()
    context = SimpleNamespace(
        repository=repository,
        config=CoreConfig(),
        active_project_id="project-1",
    )
    monkeypatch.setattr("Pelagia.processing.frame_correction.generate_background_for_frames", fake_generate)

    result = ensure_asset_background_windows(
        ["frame-10", "frame-12"],
        context=context,
        window_stride=5,
        window_width=5,
    )

    assert calls == [
        (
            ["frame-8", "frame-9", "frame-10", "frame-11"],
            {
                "background_window_center": 10,
                "background_window_start": 8,
                "background_window_end": 12,
                "background_window_stride": 5,
                "background_window_width": 5,
                "background_application_start": 8,
                "background_application_end": 12,
                "background_nominal_width": 2048,
                "background_nominal_height": 8192,
            },
        )
    ]
    assert repository.background_updates[0][0] == ["frame-12"]
    assert repository.background_updates[0][1]["payload_shape"] == [8192, 2048]
    assert result["windows"][0]["nominal_height"] == 8192


def test_nominal_background_is_trimmed_to_partial_frame_shape():
    background = np.arange(8 * 2, dtype=np.uint8).reshape(8, 2)

    trimmed = _fit_background_to_frame(
        background,
        (4, 2),
        {"background_layout": "nominal_frame"},
    )

    np.testing.assert_array_equal(trimmed, background[:4, :])


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


def test_project_kvstore_roots_partition_frame_writes_and_reads(tmp_path):
    project_id = "11111111-1111-1111-1111-111111111111"
    config = CoreConfig()
    config.kvstore.root_path = tmp_path / "kvstore"
    config.kvstore.prefix_length = 1
    default_store = KVStore(config.kvstore.root_path)
    default_store.initialize(prefix_length=1)
    repository = PartitionedRepository(project_id)
    context = AppContext(config=config, repository=repository, kvstore=default_store)
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)

    row = store_frame(
        FrameData(
            sourcePath="/tmp/",
            filename="frame.png",
            frameNumber=1,
            data=data,
            metadata={
                "run_id": "run-1",
                "asset_id": "asset-project",
                "kvstore_encoding": "raw",
            },
        ),
        context=context.for_project(project_id),
    )

    project_store = context.kvstore_for_project(project_id)
    assert project_store is not None
    assert project_store.root_path == (tmp_path / "kvstore" / "projects" / project_id).resolve(strict=False)
    assert project_store.key_exists(row["payload_ref"])
    assert not default_store.key_exists(row["payload_ref"])

    retrieved = retrieve_frame(str(row["id"]), context=context)
    np.testing.assert_array_equal(retrieved.read(), data)

    other_context = context.for_project("22222222-2222-2222-2222-222222222222")
    with pytest.raises(KeyError):
        retrieve_frame(str(row["id"]), context=other_context)


def test_default_project_kvstore_uses_existing_root(tmp_path):
    config = CoreConfig()
    config.kvstore.root_path = tmp_path / "kvstore"
    config.kvstore.prefix_length = 1
    default_store = KVStore(config.kvstore.root_path)
    default_store.initialize(prefix_length=1)
    repository = PartitionedRepository("11111111-1111-1111-1111-111111111111")
    context = AppContext(config=config, repository=repository, kvstore=default_store)
    data = np.arange(4, dtype=np.uint8).reshape(2, 2)

    row = store_frame(
        FrameData(
            sourcePath="/tmp/",
            filename="frame.png",
            frameNumber=1,
            data=data,
            metadata={
                "run_id": "run-default",
                "asset_id": "asset-default",
                "kvstore_encoding": "raw",
            },
        ),
        context=context.for_project(DEFAULT_PROJECT_ID),
    )

    assert context.kvstore_for_project(DEFAULT_PROJECT_ID) is default_store
    assert default_store.key_exists(row["payload_ref"])


def test_project_kvstore_root_can_come_from_project_row(tmp_path):
    project_id = "11111111-1111-1111-1111-111111111111"
    custom_root = tmp_path / "custom-project-kv"
    config = CoreConfig()
    config.kvstore.root_path = tmp_path / "kvstore"
    config.kvstore.prefix_length = 1
    default_store = KVStore(config.kvstore.root_path)
    default_store.initialize(prefix_length=1)
    context = AppContext(
        config=config,
        repository=PartitionedRepository(project_id, project_root=custom_root),
        kvstore=default_store,
    )

    project_store = context.kvstore_for_project(project_id)

    assert project_store is not None
    assert project_store.root_path == custom_root.resolve(strict=False)
    assert project_store.initialized is True


def test_project_kvstore_can_use_kvstore2_backend(tmp_path):
    project_id = "11111111-1111-1111-1111-111111111111"
    config = CoreConfig()
    config.kvstore.backend = "kvstore2"
    config.kvstore.root_path = tmp_path / "kvstore"
    config.kvstore.prefix_length = 1
    config.kvstore.max_blob_bytes = 1024
    default_store = KVStore2(config.kvstore.root_path)
    default_store.initialize(prefix_length=1, max_blob_bytes=1024)
    context = AppContext(
        config=config,
        repository=PartitionedRepository(project_id),
        kvstore=default_store,
    )

    project_store = context.kvstore_for_project(project_id)
    key = project_store.put_store(b"kvstore2 payload")

    assert isinstance(project_store, KVStore2)
    assert project_store.root_path == (tmp_path / "kvstore" / "projects" / project_id).resolve(strict=False)
    assert project_store.get_store(key) == b"kvstore2 payload"
    assert project_store.config["layout"] == "sqlite-index-blob-shard"


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
    assert metadata["kvstore_quality"] == 90
    decoded = decode_array_payload(ctx.kvstore.payload, metadata)
    assert decoded.shape == data.shape
    assert decoded.dtype == np.uint8


def test_jpg_codec_uses_native_grayscale_imagecodecs_path(monkeypatch):
    encode_calls: list[dict[str, object]] = []
    grayscale = np.arange(16, dtype=np.uint8).reshape(4, 4)

    def fake_encode(data, **kwargs):
        encode_calls.append({"data": data.copy(), **kwargs})
        return b"jpg"

    codec = SimpleNamespace(
        JPEG=SimpleNamespace(available=True),
        jpeg_encode=fake_encode,
        jpeg_decode=lambda payload: grayscale.copy(),
    )
    monkeypatch.setattr(frame_codec, "_imagecodecs", lambda: codec)

    payload, encoding, payload_format = encode_array_payload(grayscale, "jpg", quality=87)
    decoded = decode_array_payload(payload, {"kvstore_encoding": "jpg"})

    assert payload == b"jpg"
    assert encoding == "jpg"
    assert payload_format == "jpg"
    assert encode_calls[0]["level"] == 87
    assert np.array_equal(encode_calls[0]["data"], grayscale)
    assert np.array_equal(decoded, grayscale)


def test_jpg_codec_accepts_color_and_preserves_bgr_channels(monkeypatch):
    encode_calls: list[np.ndarray] = []
    rgb = np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8)

    def fake_encode(data, **kwargs):
        encode_calls.append(data.copy())
        return b"jpg"

    codec = SimpleNamespace(
        JPEG=SimpleNamespace(available=True),
        jpeg_encode=fake_encode,
        jpeg_decode=lambda payload: rgb.copy(),
    )
    monkeypatch.setattr(frame_codec, "_imagecodecs", lambda: codec)

    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    payload, _, _ = encode_array_payload(bgr, "jpg")
    decoded = decode_array_payload(payload, {"kvstore_encoding": "jpg"})

    assert np.array_equal(encode_calls[0], rgb)
    assert np.array_equal(decoded, bgr)
    assert decoded.flags.c_contiguous


def test_imagecodecs_decodes_legacy_opencv_jpg_as_bgr():
    pytest.importorskip("imagecodecs")
    bgr = np.zeros((32, 96, 3), dtype=np.uint8)
    bgr[:, :32] = (255, 0, 0)
    bgr[:, 32:64] = (0, 255, 0)
    bgr[:, 64:] = (0, 0, 255)
    ok, encoded = frame_codec.cv2.imencode(
        ".jpg",
        bgr,
        [frame_codec.cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    assert ok

    expected = frame_codec.cv2.imdecode(encoded, frame_codec.cv2.IMREAD_UNCHANGED)
    decoded = decode_array_payload(encoded.tobytes(), {"kvstore_encoding": "jpg"})

    np.testing.assert_allclose(decoded, expected, atol=2)


def test_store_frame_can_write_jxl_payload():
    pytest.importorskip("imagecodecs")

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
            "kvstore_encoding": "jxl",
            "kvstore_quality": 90,
        },
    )

    store_frame(frame, context=ctx)

    metadata = json.loads(ctx.repository.cursor_obj.params[17])
    assert metadata["kvstore_encoding"] == "jxl"
    assert metadata["kvstore_format"] == "jxl"
    assert metadata["kvstore_quality"] == 90
    decoded = decode_array_payload(ctx.kvstore.payload, metadata)
    assert decoded.shape == data.shape
    assert decoded.dtype == np.uint8


def test_store_frame_can_write_jxs_payload():
    imagecodecs = pytest.importorskip("imagecodecs")
    if not imagecodecs.JPEGXS.available:
        pytest.skip("imagecodecs was not built with JPEG XS support")

    ctx = FakeContext()
    data = np.repeat(np.arange(32, dtype=np.uint8)[np.newaxis, :, np.newaxis], 32, axis=0)
    data = np.repeat(data, 3, axis=2)

    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "kvstore_encoding": "jxs",
        },
    )

    store_frame(frame, context=ctx)

    metadata = json.loads(ctx.repository.cursor_obj.params[17])
    assert metadata["kvstore_encoding"] == "jxs"
    assert metadata["kvstore_format"] == "jxs"
    decoded = decode_array_payload(ctx.kvstore.payload, metadata)
    assert decoded.shape == data.shape
    assert decoded.dtype == data.dtype


def test_jxl_encoder_uses_imagecodecs_in_memory_with_fast_effort(monkeypatch):
    encode_calls: list[dict[str, object]] = []

    def fake_encode(data, **kwargs):
        encode_calls.append({"shape": data.shape, "dtype": data.dtype, **kwargs})
        return b"jxl"

    monkeypatch.setattr("imagecodecs.jpegxl_encode", fake_encode)

    payload, encoding, payload_format = encode_array_payload(
        np.arange(16, dtype=np.uint8).reshape(4, 4),
        "jxl",
    )

    assert payload == b"jxl"
    assert encoding == "jxl"
    assert payload_format == "jxl"
    assert encode_calls == [
        {
            "shape": (4, 4),
            "dtype": np.dtype("uint8"),
            "level": 90,
            "effort": 1,
            "numthreads": 4,
        }
    ]


def test_jxs_encoder_uses_imagecodecs_and_preserves_grayscale_shape(monkeypatch):
    encode_calls: list[np.ndarray] = []

    def fake_encode(data):
        encode_calls.append(data.copy())
        return b"jxs"

    codec = SimpleNamespace(
        JPEGXS=SimpleNamespace(available=True),
        jpegxs_encode=fake_encode,
        jpegxs_decode=lambda payload: np.full((4, 4, 3), 17, dtype=np.uint8),
    )
    monkeypatch.setattr(frame_codec, "_imagecodecs", lambda: codec)

    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    payload, encoding, payload_format = encode_array_payload(data, "jpeg-xs")
    decoded = decode_array_payload(payload, {"kvstore_encoding": "jxs", "shape": [4, 4]})

    assert payload == b"jxs"
    assert encoding == "jxs"
    assert payload_format == "jxs"
    assert encode_calls[0].shape == (4, 4, 3)
    assert np.array_equal(encode_calls[0][:, :, 0], data)
    assert np.array_equal(encode_calls[0][:, :, 1], data)
    assert np.array_equal(encode_calls[0][:, :, 2], data)
    assert decoded.shape == data.shape
    assert np.array_equal(decoded, np.full((4, 4), 17, dtype=np.uint8))


def test_jxs_encoder_requires_an_imagecodecs_build_with_jpeg_xs(monkeypatch):
    codec = SimpleNamespace(JPEGXS=SimpleNamespace(available=False))
    monkeypatch.setattr(frame_codec, "_imagecodecs", lambda: codec)

    with pytest.raises(RuntimeError, match="JPEG XS support is not available"):
        encode_array_payload(np.zeros((4, 4, 3), dtype=np.uint8), "jxs")


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


def test_store_frame_uses_project_storage_encoding_when_not_overridden():
    ctx = FakeContext()
    ctx.active_project_id = "project-1"
    ctx.repository.get_project = lambda project_id: {
        "settings": {"storage": {"frame": {"encoding": "raw"}}}
    }
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
        np.array([[127, 102], [255, 255]], dtype=np.uint8),
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


def test_ingest_video_file_emits_progress_callback(monkeypatch):
    class FakeVideoCapture:
        def __init__(self, path):
            self.frames = [
                np.zeros((2, 2), dtype=np.uint8),
                np.ones((2, 2), dtype=np.uint8),
                np.full((2, 2), 2, dtype=np.uint8),
            ]
            self.index = 0

        def isOpened(self):
            return self.index <= len(self.frames)

        def get(self, prop):
            if prop == ingest_module.cv2.CAP_PROP_FPS:
                return 10.0
            if prop == ingest_module.cv2.CAP_PROP_FRAME_COUNT:
                return 3.0
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
    updates = []

    ingest_video_file(
        "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
        n_tile=2,
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw"},
        progress_callback=updates.append,
    )

    assert [update["event"] for update in updates] == [
        "started",
        "video_opened",
        "tile_stored",
        "tile_stored",
        "completed",
    ]
    assert updates[1]["source_frame_count"] == 3
    assert updates[1]["estimated_tile_count"] == 2
    assert updates[2]["source_frames_read"] == 2
    assert updates[2]["stored_tile_count"] == 1
    assert updates[3]["partial_tile"] is True
    assert updates[3]["source_frames_read"] == 3
    assert updates[-1]["stored_tile_count"] == 2


def test_ingest_video_file_prefers_software_decode_when_configured(monkeypatch):
    calls = []
    thread_calls = []

    class FakeVideoCapture:
        def __init__(self, *args):
            calls.append((args, os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")))
            self.frames = [np.zeros((2, 2), dtype=np.uint8)]
            self.index = 0

        def isOpened(self):
            return self.index <= len(self.frames)

        def get(self, prop):
            if prop == ingest_module.cv2.CAP_PROP_FPS:
                return 20.0
            if prop == ingest_module.cv2.CAP_PROP_FRAME_COUNT:
                return 1.0
            return 0.0

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(ingest_module.cv2, "CAP_FFMPEG", 1900, raising=False)
    monkeypatch.setattr(ingest_module.cv2, "CAP_PROP_N_THREADS", 998, raising=False)
    monkeypatch.setattr(ingest_module.cv2, "CAP_PROP_HW_ACCELERATION", 999, raising=False)
    monkeypatch.setattr(ingest_module.cv2, "VIDEO_ACCELERATION_NONE", 0, raising=False)
    monkeypatch.setattr(ingest_module.cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(ingest_module.cv2, "setNumThreads", thread_calls.append)
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    ctx = FakeContext()
    ctx.logger = FakeDatabaseLogger()
    ctx.config.processing.video_ingest.prefer_software_decode = True

    ingest_video_file(
        "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
        n_tile=1,
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw"},
    )

    assert calls[0] == (
        ("/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv", 1900, [998, 1]),
        "threads;1|hwaccel;none",
    )
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ
    assert thread_calls == [1]
    opened = [event for event in ctx.logger.events if event["event_type"] == "video_ingest.video_opened"][0]
    assert opened["payload"]["video_decode_mode"] == "software_ffmpeg_options"
    assert opened["payload"]["opencv_threads"] == 1
    assert opened["payload"]["decoder_threads"] == 1


def test_ingest_video_file_uses_ffmpeg_fallback_when_opencv_decodes_zero_frames(monkeypatch):
    frame = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    popen_calls = []

    class FakeVideoCapture:
        def __init__(self, *args):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == ingest_module.cv2.CAP_PROP_FPS:
                return 4.0
            if prop == ingest_module.cv2.CAP_PROP_FRAME_COUNT:
                return 1.0
            if prop == ingest_module.cv2.CAP_PROP_FRAME_WIDTH:
                return 2.0
            if prop == ingest_module.cv2.CAP_PROP_FRAME_HEIGHT:
                return 2.0
            return 0.0

        def read(self):
            return False, None

        def release(self):
            pass

    class FakeProcess:
        def __init__(self, command, stdout=None, stderr=None):
            popen_calls.append(command)
            self.stdout = io.BytesIO(frame.tobytes())
            self.stderr = io.BytesIO(b"")

        def wait(self):
            return 0

    monkeypatch.setattr(ingest_module.cv2, "CAP_FFMPEG", 1900, raising=False)
    monkeypatch.setattr(ingest_module.cv2, "CAP_PROP_HW_ACCELERATION", 999, raising=False)
    monkeypatch.setattr(ingest_module.cv2, "VIDEO_ACCELERATION_NONE", 0, raising=False)
    monkeypatch.setattr(ingest_module.cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(ingest_module.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "ffmpeg" else None)
    monkeypatch.setattr(ingest_module.subprocess, "Popen", FakeProcess)
    ctx = FakeContext()
    ctx.logger = FakeDatabaseLogger()
    ctx.config.processing.video_ingest.prefer_software_decode = True

    rows = ingest_video_file(
        "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
        n_tile=1,
        context=ctx,
        run_id="00000000-0000-0000-0000-000000000001",
        asset_id="00000000-0000-0000-0000-000000000002",
        metadata={"kvstore_encoding": "raw"},
    )

    assert len(rows) == 1
    assert popen_calls[0][:9] == [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-filter_threads",
        "1",
        "-hwaccel",
        "none",
    ]
    assert popen_calls[0][9:11] == ["-threads:v", "1"]
    assert "none" in popen_calls[0]
    metadata = json.loads(ctx.repository.cursor_obj.params_history[0][17])
    decoded = decode_array_payload(ctx.kvstore.payload, metadata)
    assert decoded.tolist() == frame.tolist()
    event_types = [event["event_type"] for event in ctx.logger.events]
    assert "video_ingest.decoder_fallback" in event_types
    assert "video_ingest.decoder_fallback_completed" in event_types
    assert ctx.logger.events[-1]["payload"]["video_decode_mode"] == "ffmpeg_cli_software_fallback"
    assert ctx.logger.events[-1]["payload"]["stored_tile_count"] == 1


def test_ingest_video_file_fails_when_video_decodes_zero_frames(monkeypatch):
    class FakeVideoCapture:
        def __init__(self, path):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == ingest_module.cv2.CAP_PROP_FPS:
                return 4.0
            if prop == ingest_module.cv2.CAP_PROP_FRAME_COUNT:
                return 1.0
            return 0.0

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(ingest_module.cv2, "VideoCapture", FakeVideoCapture)
    ctx = FakeContext()
    ctx.logger = FakeDatabaseLogger()
    ctx.config.processing.video_ingest.prefer_software_decode = False

    with pytest.raises(ValueError, match="Decoded zero frames from video"):
        ingest_video_file(
            "/tmp/Camera-00002-2025-11-10 02-21-32.482.mkv",
            n_tile=1,
            context=ctx,
            run_id="00000000-0000-0000-0000-000000000001",
            asset_id="00000000-0000-0000-0000-000000000002",
            metadata={"kvstore_encoding": "raw"},
        )

    assert ctx.logger.events[-1]["event_type"] == "video_ingest.failed"
    assert ctx.logger.events[-1]["payload"]["error_type"] == "ValueError"


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
