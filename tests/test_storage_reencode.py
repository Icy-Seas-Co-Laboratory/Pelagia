from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from Pelagia.config import CoreConfig
from Pelagia.processing.frame_codec import decode_array_payload, encode_array_payload
from Pelagia.services.storage_reencode import _encode_verified, reencode_project_storage


class _BlobStore:
    def __init__(self):
        self.blobs = {}

    def put_store(self, payload):
        payload = bytes(payload)
        ref = hashlib.sha256(payload).hexdigest()
        self.blobs[ref] = payload
        return ref

    def get_store(self, ref):
        return self.blobs[ref]


def test_lossless_reencoding_accepts_identical_nan_bits():
    array = np.array([[float("nan"), 1.25]], dtype=np.float32)
    payload, encoding, _ = _encode_verified(array, "zstd", 75)
    assert encoding == "zstd"
    assert decode_array_payload(payload, {
        "kvstore_encoding": "zstd", "dtype": "float32", "shape": [1, 2],
    }).tobytes() == array.tobytes()


class _Repository:
    def __init__(self, rows, job):
        self.rows = rows
        self.job = job
        self.pages = []

    def list_storage_reencode_page(self, *, project_id, resource, created_before,
                                   after_created_at=None, after_id=None, limit=25):
        self.pages.append(resource)
        boundary = datetime.fromisoformat(created_before)
        after = (datetime.fromisoformat(after_created_at), after_id) if after_created_at else None
        return [
            {"id": row["id"], "created_at": row["created_at"]}
            for row in sorted(self.rows[resource].values(), key=lambda item: (item["created_at"], item["id"]))
            if row["project_id"] == project_id and row["created_at"] <= boundary
            and (after is None or (row["created_at"], row["id"]) > after)
        ][:limit]

    def get_storage_reencode_row(self, *, project_id, resource, resource_id):
        row = self.rows[resource].get(resource_id)
        return dict(row) if row and row["project_id"] == project_id else None

    def update_storage_reencode_frame(self, *, project_id, frame_id, row_version,
                                      variant, old_ref, new_ref, encoding, format,
                                      dtype, shape, metadata, preview_thumbhash=None):
        row = self.rows["frames"][frame_id]
        if row["project_id"] != project_id or row["row_version"] != row_version:
            return False
        assert variant == "original"
        assert row["payload_ref"] == old_ref
        row.update(kvstore_hash=new_ref, payload_ref=new_ref, payload_encoding=encoding,
                   payload_format=format, payload_dtype=dtype, payload_shape=shape,
                   metadata=metadata, preview_thumbhash=preview_thumbhash,
                   row_version=str(int(row_version) + 1))
        return True

    def update_storage_reencode_detection(self, *, project_id, resource, resource_id,
                                          row_version, roi_payload, roi_encoding, roi_format,
                                          mask_payload, mask_encoding, mask_format, metadata):
        row = self.rows[resource][resource_id]
        if row["project_id"] != project_id or row["row_version"] != row_version:
            return False
        row.update(roi_payload=roi_payload, roi_encoding=roi_encoding, roi_format=roi_format,
                   mask_payload=mask_payload, mask_encoding=mask_encoding,
                   mask_format=mask_format, metadata=metadata,
                   row_version=str(int(row_version) + 1))
        return True

    def update_job_progress(self, job_id, progress, **kwargs):
        assert job_id == self.job["id"]
        self.job["progress"] = progress
        return self.job


def _roi_row(row_id, array, created_at, *, project_id="project-1"):
    mask = np.full(array.shape, 255, dtype=np.uint8)
    return {
        "id": row_id, "project_id": project_id, "created_at": created_at,
        "row_version": "1", "roi_payload": array.tobytes(), "roi_encoding": "raw",
        "roi_format": "raw_ndarray_c_order", "roi_dtype": str(array.dtype),
        "roi_shape": list(array.shape), "mask_payload": mask.tobytes(),
        "mask_encoding": "raw", "mask_format": "raw_ndarray_c_order",
        "mask_dtype": "uint8", "mask_shape": list(mask.shape), "metadata": {},
    }


def test_project_storage_reencode_applies_cutoff_and_keeps_project_scope():
    pytest.importorskip("imagecodecs")
    now = datetime.now(timezone.utc)
    store = _BlobStore()
    frame = np.full((8, 8), 93, dtype=np.uint8)
    original_ref = store.put_store(frame.tobytes())
    rows = {
        "frames": {"frame-1": {
            "id": "frame-1", "project_id": "project-1", "created_at": now - timedelta(seconds=2),
            "row_version": "1", "kvstore_hash": original_ref, "payload_ref": original_ref,
            "payload_encoding": "raw", "payload_format": "raw_ndarray_c_order",
            "payload_dtype": "uint8", "payload_shape": [8, 8], "metadata": {},
        }},
        "detection_candidate": {
            "small": _roi_row("small", np.full((25, 39), 127, dtype=np.uint8), now - timedelta(seconds=1)),
            "other": _roi_row("other", np.full((25, 40), 127, dtype=np.uint8), now - timedelta(seconds=1), project_id="project-2"),
        },
        "detections_refined": {
            "large": _roi_row("large", np.full((25, 40), 127, dtype=np.uint8), now - timedelta(seconds=1)),
        },
    }
    job = {"id": "job-1", "project_id": "project-1", "payload": {
        "created_before": now.isoformat(),
        "policy": {"frame_encoding": "jpg", "frame_quality": 75,
                   "small_roi_encoding": "zstd", "large_roi_encoding": "jpg",
                   "large_roi_min_pixels": 1000, "roi_quality": 75,
                   "mask_encoding": "zstd"},
    }}
    repository = _Repository(rows, job)
    context = SimpleNamespace(repository=repository, config=CoreConfig(),
                              kvstore_for_project=lambda project_id: store if project_id == "project-1" else None)

    result = reencode_project_storage(job, context)

    assert result["examined"] == 3
    assert result["rewritten"] == 5
    assert rows["frames"]["frame-1"]["payload_encoding"] == "jpg"
    assert rows["frames"]["frame-1"]["metadata"]["kvstore_quality"] == 75
    assert original_ref in store.blobs
    assert rows["detection_candidate"]["small"]["roi_encoding"] == "zstd"
    assert rows["detections_refined"]["large"]["roi_encoding"] == "jpg"
    assert rows["detection_candidate"]["small"]["mask_encoding"] == "zstd"
    assert rows["detection_candidate"]["other"]["roi_encoding"] == "raw"
    small = rows["detection_candidate"]["small"]
    assert np.array_equal(
        decode_array_payload(small["roi_payload"], {
            "kvstore_encoding": small["roi_encoding"], "dtype": small["roi_dtype"],
            "shape": small["roi_shape"],
        }), np.full((25, 39), 127, dtype=np.uint8),
    )

    assert reencode_project_storage(job, context)["rewritten"] == 5

    previous_ref = rows["frames"]["frame-1"]["payload_ref"]
    reencode_project_storage({**job, "progress": {}}, context)
    assert rows["frames"]["frame-1"]["payload_ref"] == previous_ref
    assert len(rows["detection_candidate"]["small"]["metadata"]["storage_reencode_history"]) == 1


def test_storage_reencode_retry_does_not_encode_committed_roi_twice():
    now = datetime.now(timezone.utc)
    rows = {
        "frames": {},
        "detection_candidate": {
            "a": _roi_row("a", np.full((4, 4), 8, dtype=np.uint8), now - timedelta(seconds=1)),
            "b": _roi_row("b", np.full((4, 4), 9, dtype=np.uint8), now - timedelta(seconds=1)),
        },
        "detections_refined": {},
    }
    job = {"id": "job-retry", "project_id": "project-1", "payload": {
        "created_before": now.isoformat(),
        "policy": {"frame_encoding": "zstd", "frame_quality": 75,
                   "small_roi_encoding": "zstd", "large_roi_encoding": "zstd",
                   "large_roi_min_pixels": 1000, "roi_quality": 75,
                   "mask_encoding": "zstd"},
    }}

    class FailingRepository(_Repository):
        fail_once = True

        def get_storage_reencode_row(self, *, project_id, resource, resource_id):
            if resource_id == "b" and self.fail_once:
                self.fail_once = False
                raise RuntimeError("simulated interruption")
            return super().get_storage_reencode_row(
                project_id=project_id, resource=resource, resource_id=resource_id,
            )

    repository = FailingRepository(rows, job)
    context = SimpleNamespace(repository=repository, config=CoreConfig(),
                              kvstore_for_project=lambda _project_id: _BlobStore())

    with pytest.raises(RuntimeError, match="simulated interruption"):
        reencode_project_storage(job, context)
    first_payload = rows["detection_candidate"]["a"]["roi_payload"]

    result = reencode_project_storage(job, context)

    assert rows["detection_candidate"]["a"]["roi_payload"] == first_payload
    assert len(rows["detection_candidate"]["a"]["metadata"]["storage_reencode_history"]) == 1
    assert rows["detection_candidate"]["b"]["roi_encoding"] == "zstd"
    assert result["examined"] == 2


def test_storage_reencode_skips_candidate_already_at_saved_policy():
    from Pelagia.services.storage_reencode import _reencode_detection

    now = datetime.now(timezone.utc)
    array = np.full((4, 4), 12, dtype=np.uint8)
    row = _roi_row("current", array, now)
    row["roi_payload"] = encode_array_payload(array, "zstd")[0]
    row["mask_payload"] = encode_array_payload(np.full((4, 4), 255, dtype=np.uint8), "zstd")[0]
    row["roi_encoding"] = row["mask_encoding"] = "zstd"
    row["roi_format"] = row["mask_format"] = "zstd_ndarray_c_order"
    row["metadata"] = {"roi_storage_policy": {"quality": 75}}
    repository = _Repository({"frames": {}, "detection_candidate": {"current": row}, "detections_refined": {}}, {"id": "job"})
    policy = {"frame_encoding": "jpg", "frame_quality": 75,
              "small_roi_encoding": "zstd", "large_roi_encoding": "jpg",
              "large_roi_min_pixels": 1000, "roi_quality": 75, "mask_encoding": "zstd"}

    assert _reencode_detection(repository, project_id="project-1", resource="detection_candidate",
                               resource_id="current", policy=policy, fingerprint="current",
                               job_id="job") == (0, 0, 0)
    assert row["row_version"] == "1"
