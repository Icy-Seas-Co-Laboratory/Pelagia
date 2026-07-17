import numpy as np
import pytest

from Pelagia.config import CoreConfig
from Pelagia.processing.frame_codec import decode_array_payload
from Pelagia.processing.ingest_background import MeanFieldIngestAddon
from Pelagia.services.context import AppContext


class BackgroundKVStore:
    initialized = True

    def __init__(self):
        self.payloads = {}

    def put_store(self, payload):
        key = f"background-{len(self.payloads) + 1}"
        self.payloads[key] = payload
        return key


class BackgroundRepository:
    def __init__(self):
        self.calls = []

    def update_frame_background_payload_assignments(self, assignments, *, project_id=None):
        self.calls.append((list(assignments), project_id))
        return [{"id": assignment["frame_id"]} for assignment in assignments]


def test_ingest_background_addon_streams_mean_windows_and_batches_assignments():
    repository = BackgroundRepository()
    kvstore = BackgroundKVStore()
    context = AppContext(config=CoreConfig(), repository=repository, kvstore=kvstore)
    addon = MeanFieldIngestAddon(
        context=context,
        project_id=None,
        window_stride=3,
        window_width=3,
        encoding="raw",
    )

    arrays = {
        1: np.full((2, 2), 10, dtype=np.uint8),
        2: np.full((2, 2), 20, dtype=np.uint8),
        3: np.full((2, 2), 40, dtype=np.uint8),
        4: np.full((1, 2), 200, dtype=np.uint8),
        5: np.full((2, 2), 80, dtype=np.uint8),
    }
    for frame_index, array in arrays.items():
        addon.consume({"id": f"frame-{frame_index}", "frame_index": frame_index}, array)

    result = addon.finalize()

    assert result == {
        "method": "mean",
        "window_stride": 3,
        "window_width": 3,
        "window_count": 3,
        "stored_window_count": 3,
        "updated_frame_count": 5,
        "skipped_frame_count": 1,
        "skipped_background_frame_count": 1,
        "skipped_flatfield_frame_count": 0,
        "nominal_shape": [2, 2],
        "flatfield_axis": 0,
        "flatfield_profile_length": 2,
        "backgrounds_generated": True,
        "flatfield_profiles_generated": True,
    }
    assert len(repository.calls) == 1
    assignments, project_id = repository.calls[0]
    assert project_id is None
    assert [assignment["frame_id"] for assignment in assignments] == [
        "frame-1",
        "frame-2",
        "frame-3",
        "frame-4",
        "frame-5",
    ]
    assert assignments[1]["payload_ref"] == assignments[2]["payload_ref"]
    assert assignments[2]["payload_ref"] == assignments[3]["payload_ref"]
    middle = assignments[2]
    decoded = decode_array_payload(kvstore.payloads[middle["payload_ref"]], middle["metadata"])
    np.testing.assert_array_equal(decoded, np.full((2, 2), 30, dtype=np.uint8))
    assert middle["metadata"]["background_source_frame_ids"] == ["frame-2", "frame-3"]
    assert middle["flatfield_profile"] == [64.0, 64.0]
    assert middle["flatfield_metadata"]["flatfield_method"] == "column_mean"
    assert middle["flatfield_metadata"]["stack_axis"] == 0
    assert middle["flatfield_metadata"]["source_frame_ids"] == [
        "frame-2",
        "frame-3",
        "frame-4",
    ]


def test_ingest_background_addon_can_generate_profiles_without_kvstore_writes():
    repository = BackgroundRepository()
    kvstore = BackgroundKVStore()
    addon = MeanFieldIngestAddon(
        context=AppContext(config=CoreConfig(), repository=repository, kvstore=kvstore),
        project_id=None,
        window_stride=5,
        window_width=5,
        generate_backgrounds=False,
        generate_flatfield_profiles=True,
    )
    addon.consume(
        {"id": "frame-1", "frame_index": 1},
        np.array([[10, 20], [30, 40]], dtype=np.uint8),
    )
    addon.consume(
        {"id": "frame-2", "frame_index": 2},
        np.array([[50, 70]], dtype=np.uint8),
    )

    result = addon.finalize()

    assert result["backgrounds_generated"] is False
    assert result["flatfield_profiles_generated"] is True
    assert kvstore.payloads == {}
    np.testing.assert_allclose(
        repository.calls[0][0][0]["flatfield_profile"],
        [30.0, 43.333332],
        rtol=1e-6,
    )
    assert result["skipped_frame_count"] == 0
    assert result["skipped_flatfield_frame_count"] == 0
    assert "payload_ref" not in repository.calls[0][0][0]


def test_ingest_background_addon_stacks_horizontally_for_row_profiles():
    repository = BackgroundRepository()
    addon = MeanFieldIngestAddon(
        context=AppContext(
            config=CoreConfig(),
            repository=repository,
            kvstore=BackgroundKVStore(),
        ),
        project_id=None,
        window_stride=5,
        window_width=5,
        generate_backgrounds=False,
        generate_flatfield_profiles=True,
        flatfield_axis=1,
    )
    addon.consume(
        {"id": "frame-1", "frame_index": 1},
        np.array([[10, 30], [20, 40]], dtype=np.uint8),
    )
    addon.consume(
        {"id": "frame-2", "frame_index": 2},
        np.array([[50], [70]], dtype=np.uint8),
    )

    result = addon.finalize()
    assignment = repository.calls[0][0][0]

    assert result["flatfield_axis"] == 1
    assert result["flatfield_profile_length"] == 2
    assert result["skipped_flatfield_frame_count"] == 0
    np.testing.assert_allclose(assignment["flatfield_profile"], [30.0, 43.333332], rtol=1e-6)
    assert assignment["flatfield_metadata"]["flatfield_method"] == "row_mean"
    assert assignment["flatfield_metadata"]["stack_axis"] == 1


def test_ingest_background_addon_supports_independent_output_windows():
    repository = BackgroundRepository()
    addon = MeanFieldIngestAddon(
        context=AppContext(
            config=CoreConfig(),
            repository=repository,
            kvstore=BackgroundKVStore(),
        ),
        project_id=None,
        window_stride=3,
        window_width=3,
        flatfield_window_stride=1,
        flatfield_window_width=1,
        encoding="raw",
        generate_backgrounds=True,
        generate_flatfield_profiles=True,
    )
    for frame_index in (1, 2, 3):
        addon.consume(
            {"id": f"frame-{frame_index}", "frame_index": frame_index},
            np.full((2, 2), frame_index * 10, dtype=np.uint8),
        )

    result = addon.finalize()

    assert result["background"]["window_stride"] == 3
    assert result["background"]["window_width"] == 3
    assert result["flatfield"]["window_stride"] == 1
    assert result["flatfield"]["window_width"] == 1
    assert result["window_count"] == (
        result["background"]["window_count"] + result["flatfield"]["window_count"]
    )
    assert result["background_window_count"] == result["background"]["window_count"]
    assert result["flatfield_window_count"] == result["flatfield"]["window_count"]
    assert len(repository.calls) == 1
    assert "payload_ref" in repository.calls[0][0][0]
    assert "flatfield_profile" in repository.calls[0][0][0]


def test_ingest_background_addon_rejects_non_monotonic_frame_order():
    addon = MeanFieldIngestAddon(
        context=AppContext(
            config=CoreConfig(),
            repository=BackgroundRepository(),
            kvstore=BackgroundKVStore(),
        ),
        project_id=None,
        window_stride=3,
        window_width=3,
        encoding="raw",
    )
    addon.consume({"id": "frame-2", "frame_index": 2}, np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="increasing frame-index order"):
        addon.consume({"id": "frame-1", "frame_index": 1}, np.zeros((2, 2), dtype=np.uint8))
