from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np

from oracle_data_contracts.datasets import validate_database

from Pelagia.services import registry_generation


def test_generate_registry_dataset_writes_valid_contract_and_loads_workspace(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    dataset_id = str(uuid.uuid4())
    revision_id = str(uuid.uuid4())
    label_id = str(uuid.uuid4())
    roi_id = str(uuid.uuid4())
    inference_run_id = str(uuid.uuid4())
    evidence_id = str(uuid.uuid4())
    annotation_id = str(uuid.uuid4())
    review_id = str(uuid.uuid4())
    source_asset_id = str(uuid.uuid4())
    frame_id = str(uuid.uuid4())
    payload = b"portable-roi-payload"

    row = {
        "id": roi_id, "roi_payload": payload, "roi_encoding": "png", "roi_format": "png",
        "roi_dtype": "uint8", "roi_shape": [3, 4], "source_asset_id": source_asset_id,
        "source_asset_filename": "source.tif", "frame_id": frame_id, "frame_index": 7,
        "bbox_x": 102, "bbox_y": 203, "bbox_w": 2, "bbox_h": 1,
        "crop_bbox_x": 101, "crop_bbox_y": 201, "crop_bbox_w": 4, "crop_bbox_h": 3,
        "roi_index": 2, "area": 12.0, "selection_ordinal": 1, "created_at": now,
        "annotation_id": annotation_id, "label_id": label_id, "actor_username": "curator",
        "annotation_method": "human", "annotation_status": "accepted",
        "parent_annotation_id": None, "annotation_notes": None, "annotation_metadata": {},
        "annotation_created_at": now, "review_id": review_id, "reviewer_username": "reviewer",
        "review_decision": "verified", "review_notes": None, "review_metadata": {},
        "review_created_at": now, "evidence_id": evidence_id, "inference_run_id": inference_run_id,
        "predicted_label_id": label_id, "predicted_label_name": "copepod", "confidence": 0.91,
        "prototype_similarity": 0.8, "knn_agreement": 0.75, "knn_weighted_support": 0.7,
        "probability_margin": 0.4, "evidence_packet": {"source": "oracle"},
        "probabilities": [{"label_name": "copepod", "probability": 0.91}],
        "evidence_created_at": now, "model_selector": "oracle-test", "inference_status": "complete",
        "inference_parameters": {}, "inference_metadata": {}, "inference_created_at": now,
        "inference_completed_at": now, "artifact_id": str(uuid.uuid4()), "model_run_id": None,
        "artifact_fingerprint": "abc123",
    }
    label = {
        "id": label_id, "name": "copepod", "display_name": "Copepod", "parent_label_id": None,
        "rank": None, "description": None, "metadata": {}, "created_at": now, "deprecated_at": None,
    }
    monkeypatch.setattr(registry_generation, "_selected_rows", lambda *args, **kwargs: [row])
    monkeypatch.setattr(registry_generation, "_labels", lambda *args, **kwargs: [label])
    monkeypatch.setattr(registry_generation, "preview_registry_dataset", lambda *args, **kwargs: {
        "matching_count": 1, "selected_count": 1, "payload_bytes": len(payload),
        "estimated_sqlite_bytes": 2 * 1024 * 1024, "subsample_ratio": 500,
    })
    loaded = {}

    def fake_load(repository, source, **scope):
        loaded.update({"source": source, **scope})
        return {"workspace_id": str(uuid.uuid4()), "reused": False}

    monkeypatch.setattr(registry_generation, "load_sqlite_workspace", fake_load)
    destination = tmp_path / "curation.sqlite"
    result = registry_generation.generate_and_load_registry_dataset(
        object(), destination, project_id=str(uuid.uuid4()), owner_username="curator",
        name="Curation export", selection={"annotation_state": "all"}, subsample_ratio=500,
        dataset_id=dataset_id, revision_id=revision_id,
    )

    assert result["selected_count"] == 1
    assert result["subsample_ratio"] == 500
    assert loaded["source"] == destination
    with sqlite3.connect(destination) as connection:
        report = validate_database(connection)
        assert report["valid"] is True
        assert connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM item_label_annotations").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM model_evidence").fetchone()[0] == 1
        geometry = connection.execute(
            """SELECT coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
               crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h
               FROM item_geometry WHERE item_id=?""",
            (roi_id,),
        ).fetchone()
        assert geometry == ("source_frame_pixels", 102, 203, 2, 1, 101, 201, 4, 3)
        item_metadata = connection.execute(
            "SELECT metadata_json FROM dataset_items WHERE item_id=?", (roi_id,)
        ).fetchone()[0]
        assert '"detection_metadata"' in item_metadata
        assert '"crop_bbox"' in item_metadata
        metadata = connection.execute("SELECT metadata_json FROM dataset WHERE singleton=1").fetchone()[0]
        assert '"subsample_ratio": 500' in metadata
        assert '"spatial_contract"' in metadata

    resumed = registry_generation.generate_and_load_registry_dataset(
        object(), destination, project_id=loaded["project_id"], owner_username="curator",
        name="Curation export", selection={"annotation_state": "all"}, subsample_ratio=500,
        dataset_id=dataset_id, revision_id=revision_id,
    )
    assert resumed["resumed"] is True
    assert resumed["selected_count"] == 1


def test_preview_registry_dataset_applies_exact_stride_count():
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def execute(self, query, params):
            assert "selection_ordinal %%" in query
            assert params[-2:] == (500, 500)
        def fetchone(self):
            return {"matching_count": 1001, "selected_count": 2, "payload_bytes": 600}

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def cursor(self): return Cursor()

    class Repository:
        schema = "pelagia"
        def connect(self): return Connection()

    preview = registry_generation.preview_registry_dataset(
        Repository(), project_id=str(uuid.uuid4()), selection={}, subsample_ratio=500
    )
    assert preview["matching_count"] == 1001
    assert preview["selected_count"] == 2
    assert preview["subsample_ratio"] == 500


def test_non_browser_roi_encoding_is_normalized_to_png():
    source = np.arange(12, dtype=np.uint8).reshape(3, 4)
    payload, encoding, media_type, shape, dtype = registry_generation._portable_image({
        "roi_payload": source.tobytes(), "roi_encoding": "raw",
        "roi_format": "raw_ndarray_c_order", "roi_shape": [3, 4], "roi_dtype": "uint8",
    })
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert encoding == "png"
    assert media_type == "image/png"
    assert shape == [3, 4]
    assert dtype == "uint8"
    np.testing.assert_array_equal(decoded, source)
