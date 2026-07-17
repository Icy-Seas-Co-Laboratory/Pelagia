import numpy as np
import pytest

from Pelagia.config import CoreConfig
from Pelagia.domain import DetectionRecord, FrameRecord, PipelineStage
from Pelagia.processing.frame_model import FrameData
from Pelagia.services.context import AppContext
from Pelagia.workers.handlers import (
    HandlerRegistry,
    background_frames_handler,
    default_handler_registry,
    extract_frames_handler,
    preprocess_frames_handler,
    roi_detection_handler,
    roi_refinement_handler,
)
from Pelagia.workers.worker import Worker
from Pelagia.workers.runtime import worker_runtime_profile


class FakeRepository:
    def __init__(self):
        self.assets = {
            "asset-1": {
                "id": "asset-1",
                "path": "/tmp/source.avi",
                "collections": ["test"],
            }
        }
        self.claimed_jobs = []
        self.completed = []
        self.failures = []
        self.created_jobs = []
        self.replaced_detections = []
        self.refined_detections = []
        self.background_calls = []
        self.touches = []
        self.requeued = 0
        self.progress_updates = []
        self.frame_status_rows = []
        self.frame_stage_status_updates = []
        self.frame_status_count_refreshes = []
        self.processing_status_snapshot_touches = []
        self.frames_with_background = set()
        self.shutdown_requested = False
        self.project_calls = []

    def get_asset(self, asset_id, **kwargs):
        self.project_calls.append(("get_asset", kwargs.get("project_id")))
        return self.assets.get(asset_id)

    def get_project(self, project_id):
        return {"id": project_id, "kvstore_root_path": None}

    def claim_jobs(self, worker_id, stages=None):
        return list(self.claimed_jobs)

    def complete_job(self, job_id, result=None):
        self.completed.append((job_id, result))
        return {"id": job_id, "status": "succeeded", "result": result}

    def record_failure(self, job_id, error_message, retryable=True):
        self.failures.append((job_id, error_message, retryable))
        return {"id": job_id, "status": "queued", "error_message": error_message}

    def create_job(self, stage, **kwargs):
        job = {"id": "segment-job-1", "stage": stage.value, **kwargs}
        self.created_jobs.append(job)
        return job

    def update_job_progress(self, job_id, progress, summary=None, log_message=None):
        self.progress_updates.append(
            {
                "job_id": job_id,
                "progress": progress,
                "summary": summary,
                "log_message": log_message,
            }
        )
        return {"id": job_id, "progress": progress}

    def ensure_frame_status_rows(self, **kwargs):
        self.frame_status_rows.append(kwargs)
        return len(kwargs.get("frame_ids") or [])

    def upsert_frame_stage_status(self, **kwargs):
        self.frame_stage_status_updates.append(kwargs)
        return len(kwargs.get("frame_ids") or [])

    def refresh_frame_status_counts(self, **kwargs):
        self.frame_status_count_refreshes.append(kwargs)
        return len(kwargs.get("frame_ids") or [])

    def touch_processing_status_snapshot(self, **kwargs):
        self.processing_status_snapshot_touches.append(kwargs)
        return {"status_version": len(self.processing_status_snapshot_touches)}

    def list_frames(self, asset_id, **kwargs):
        self.project_calls.append(("list_frames", kwargs.get("project_id")))
        return [{"id": "frame-1", "asset_id": asset_id, "frame_index": 1, **kwargs}]

    def get_frame_record(self, frame_id, **kwargs):
        self.project_calls.append(("get_frame_record", kwargs.get("project_id")))
        if frame_id != "frame-1":
            return None
        return FrameRecord(
            id="frame-1",
            run_id="run-1",
            asset_id="asset-1",
            frame_index=1,
            width=4,
            height=4,
            kvstore_hash="kvstore-key",
            preview_thumbhash=b"thumb",
            background_kvstore_hash=(
                "background-key" if frame_id in self.frames_with_background else None
            ),
            background_payload_ref=(
                "background-key" if frame_id in self.frames_with_background else None
            ),
            background_payload_encoding=(
                "raw" if frame_id in self.frames_with_background else None
            ),
            background_payload_format=(
                "raw_ndarray_c_order" if frame_id in self.frames_with_background else None
            ),
            background_payload_dtype=(
                "float32" if frame_id in self.frames_with_background else None
            ),
            background_payload_shape=(
                [4, 4] if frame_id in self.frames_with_background else []
            ),
        )

    def get_frame_records(self, frame_ids, **kwargs):
        return [
            record
            for frame_id in frame_ids
            if (record := self.get_frame_record(frame_id, **kwargs)) is not None
        ]

    def replace_frame_detections(self, run_id, frame_ids, detections, **kwargs):
        self.project_calls.append(("replace_frame_detections", kwargs.get("project_id")))
        rows = [
            {"id": f"det-{index}", "frame_id": detection.frame_id}
            for index, detection in enumerate(detections, start=1)
        ]
        self.replaced_detections.append((run_id, frame_ids, detections))
        return rows

    def get_detection(self, detection_id, **kwargs):
        self.project_calls.append(("get_detection", kwargs.get("project_id")))
        if detection_id not in {"det-1", "det-no-roi"}:
            return None
        roi = np.array([[0, 20], [40, 80]], dtype=np.uint8)
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        return {
            "id": detection_id,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "frame_id": "frame-1",
            "roi_index": 1,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": 2,
            "bbox_h": 2,
            "crop_bbox_x": 0,
            "crop_bbox_y": 0,
            "crop_bbox_w": 2,
            "crop_bbox_h": 2,
            "area": 2,
            "perimeter": 4,
            "major_axis_length": 2,
            "minor_axis_length": 2,
            "min_gray_value": 20,
            "mean_gray_value": 30,
            "roi_payload": None if detection_id == "det-no-roi" else roi.tobytes(order="C"),
            "mask_payload": mask.tobytes(order="C"),
            "roi_encoding": "raw",
            "roi_format": "raw_ndarray_c_order",
            "roi_dtype": "uint8",
            "roi_shape": [2, 2],
            "mask_encoding": "raw",
            "mask_format": "raw_ndarray_c_order",
            "mask_dtype": "uint8",
            "mask_shape": [2, 2],
            "metadata": {},
        }

    def upsert_refined_detections(self, refined_detections, *, job_id=None, project_id=None):
        self.project_calls.append(("upsert_refined_detections", project_id))
        rows = []
        for candidate_detection_id, detection in refined_detections:
            row = {
                "id": f"refined-{candidate_detection_id}",
                "candidate_detection_id": candidate_detection_id,
                "job_id": job_id,
                "run_id": detection.run_id,
                "frame_id": detection.frame_id,
                "roi_index": detection.roi_index,
                "metadata": detection.metadata,
            }
            rows.append(row)
        self.refined_detections.append(refined_detections)
        return rows

    def touch_worker(self, worker_id, **kwargs):
        row = {"worker_id": worker_id, **kwargs}
        self.touches.append(row)
        return row

    def get_worker_session(self, worker_id):
        return {"worker_id": worker_id, "shutdown_requested": self.shutdown_requested}

    def requeue_expired_jobs(self):
        self.requeued += 1
        return {"queued": 0, "dead_lettered": 0}


def make_context(repository):
    return AppContext(config=CoreConfig(), repository=repository, kvstore=None)


def test_worker_runtime_profile_requires_explicit_non_mixed_stages():
    with pytest.raises(ValueError, match="explicit stages"):
        worker_runtime_profile(None)
    assert worker_runtime_profile([PipelineStage.SEGMENT]) == "cpu"
    assert worker_runtime_profile([PipelineStage.ROI_REFINEMENT]) == "gpu-ml"
    with pytest.raises(ValueError, match="dedicated worker"):
        worker_runtime_profile([PipelineStage.SEGMENT, PipelineStage.ROI_REFINEMENT])


def test_extract_frames_handler_ingests_registered_asset(monkeypatch):
    repo = FakeRepository()
    repo.assets["asset-1"]["metadata"] = {"analysis": "generated", "source": "asset"}
    context = make_context(repo)
    calls = []

    def fake_ingest_video_file(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"id": 10}, {"id": 11}]

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ingest_module.ingest_video_file",
        fake_ingest_video_file,
    )

    job = {
        "id": "job-1",
        "stage": PipelineStage.EXTRACT_FRAMES.value,
        "run_id": "run-1",
        "asset_id": "asset-1",
        "payload": {
            "n_tile": 2,
            "metadata": {"source": "test"},
        },
    }

    result = extract_frames_handler(job, context)
    timings = result.pop("timings")

    assert result == {
        "stage": PipelineStage.EXTRACT_FRAMES.value,
        "project_id": None,
        "run_id": "run-1",
        "asset_id": "asset-1",
        "source_path": "/tmp/source.avi",
        "frame_count": 2,
        "frame_ids": [10, 11],
    }
    assert timings["schema_version"] == 1
    assert timings["unit_count"] == 2

    args, kwargs = calls[0]
    assert args == ("/tmp/source.avi",)
    assert kwargs["n_tile"] == 2
    assert kwargs["context"] is context
    assert kwargs["run_id"] == "run-1"
    assert kwargs["asset_id"] == "asset-1"
    assert "flatfield_correction" not in kwargs
    assert "flatfield_q" not in kwargs
    assert "flatfield_axis" not in kwargs
    assert "flatfield_maximum_value" not in kwargs
    assert kwargs["adaptive_background_subtraction"] is False
    assert kwargs["adaptive_background_period"] == 50
    assert kwargs["apply_mask"] is False
    assert kwargs["mask_path"] is None
    assert kwargs["metadata"]["source"] == "test"
    assert kwargs["metadata"]["analysis"] == "generated"
    assert kwargs["metadata"]["collections"] == ["test"]
    assert kwargs["metadata"]["worker_job_id"] == "job-1"


def test_extract_frames_handler_reports_video_ingest_progress(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo)

    def fake_ingest_video_file(*args, **kwargs):
        progress_callback = kwargs["progress_callback"]
        progress_callback(
            {
                "event": "video_opened",
                "filename": "source.avi",
                "source_frame_count": 50,
                "source_frames_read": 0,
                "stored_tile_count": 0,
                "n_tile": 5,
                "estimated_tile_count": 10,
                "fps": 20.0,
            }
        )
        progress_callback(
            {
                "event": "tile_stored",
                "filename": "source.avi",
                "source_frame_count": 50,
                "source_frames_read": 25,
                "source_frame_start": 21,
                "source_frame_end": 25,
                "stored_tile_count": 5,
                "tile_number": 5,
                "n_tile": 5,
                "estimated_tile_count": 10,
                "fps": 20.0,
            }
        )
        return [
            {"id": "tile-1"},
            {"id": "tile-2"},
            {"id": "tile-3"},
            {"id": "tile-4"},
            {"id": "tile-5"},
        ]

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ingest_module.ingest_video_file",
        fake_ingest_video_file,
    )

    result = extract_frames_handler(
        {
            "id": "job-1",
            "stage": PipelineStage.EXTRACT_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"n_tile": 5},
        },
        context,
    )

    assert result["frame_count"] == 5
    progress_payloads = [update["progress"] for update in repo.progress_updates]
    assert progress_payloads[0]["completed"] == 0
    assert any(payload["completed"] == 25 for payload in progress_payloads)
    assert progress_payloads[-1]["completed"] == 50
    assert progress_payloads[-1]["total"] == 50
    assert progress_payloads[-1]["secondary"]["stored_frame_count"] == 5
    mid_progress = next(payload for payload in progress_payloads if payload["completed"] == 25)
    assert mid_progress["unit"] == "frames"
    assert mid_progress["current"]["tile_number"] == 5
    assert mid_progress["secondary"]["stored_tile_count"] == 5
    assert mid_progress["secondary"]["estimated_tile_count"] == 10


def test_extract_frames_handler_can_enqueue_segment_job(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo)

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ingest_module.ingest_video_file",
        lambda *args, **kwargs: [{"id": 10}],
    )

    result = extract_frames_handler(
        {
            "id": "job-1",
            "stage": PipelineStage.EXTRACT_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "enqueue_segment": True,
                "padding": 4,
                "roi_encoding": "raw",
            },
        },
        context,
    )

    assert result["segment_job_id"] == "segment-job-1"
    assert repo.created_jobs[0]["stage"] == PipelineStage.SEGMENT.value
    assert repo.created_jobs[0]["payload"] == {
        "command_type": "segment_frames",
        "command_version": 1,
        "frame_ids": [10],
        "padding": 4,
        "roi_encoding": "raw",
        "collections": ["test"],
    }
    assert repo.created_jobs[0]["depends_on"] == ["job-1"]


def test_extract_frames_handler_preserves_project_when_enqueuing_segment(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo).for_project("project-1")

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ingest_module.ingest_video_file",
        lambda *args, **kwargs: [{"id": "frame-1"}],
    )

    result = extract_frames_handler(
        {
            "id": "job-1",
            "project_id": "project-1",
            "stage": PipelineStage.EXTRACT_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"enqueue_segment": True},
        },
        context,
    )

    assert result["project_id"] == "project-1"
    assert repo.created_jobs[0]["project_id"] == "project-1"
    assert repo.created_jobs[0]["stage"] == PipelineStage.SEGMENT.value
    assert ("get_asset", "project-1") in repo.project_calls
    assert repo.frame_status_rows == [
        {
            "project_id": "project-1",
            "frame_ids": ["frame-1"],
            "asset_id": "asset-1",
        }
    ]
    assert repo.processing_status_snapshot_touches == [{"project_id": "project-1"}]


def test_extract_frames_handler_ingests_image_sequence_folder(monkeypatch, tmp_path):
    repo = FakeRepository()
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    repo.assets["asset-1"]["path"] = str(image_dir)
    repo.assets["asset-1"]["kind"] = "image_sequence"
    context = make_context(repo)
    calls = []

    def fake_ingest_image_folder(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"id": 20}, {"id": 21}]

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ingest_module.ingest_image_folder",
        fake_ingest_image_folder,
    )

    result = extract_frames_handler(
        {
            "id": "job-1",
            "stage": PipelineStage.EXTRACT_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"recursive": True},
        },
        context,
    )

    assert result["frame_count"] == 2
    assert result["frame_ids"] == [20, 21]
    args, kwargs = calls[0]
    assert args == (str(image_dir),)
    assert kwargs["recursive"] is True
    assert kwargs["context"] is context
    assert kwargs["run_id"] == "run-1"
    assert kwargs["asset_id"] == "asset-1"


def test_preprocess_frames_handler_stores_preprocessed_payloads(monkeypatch):
    repo = FakeRepository()
    repo.frames_with_background.add("frame-1")
    context = make_context(repo)
    retrieved = []
    preprocessed = []
    stored = []

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original", frame_record=None):
        retrieved.append((frame_id, context, payload_kind))
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=np.zeros((4, 4), dtype=np.uint8),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    def fake_preprocess_frame(frame, **kwargs):
        preprocessed.append((frame, kwargs))
        return frame

    def fake_store_preprocessed_frames(frames, **kwargs):
        stored.append((list(frames), kwargs))
        return [{"id": frame_id} for frame_id, _ in frames]

    monkeypatch.setattr("Pelagia.workers.handlers.retrieve_frame", fake_retrieve_frame)
    monkeypatch.setattr("Pelagia.workers.handlers.preprocess_frame_for_segmentation", fake_preprocess_frame)
    monkeypatch.setattr("Pelagia.workers.handlers.store_preprocessed_frames", fake_store_preprocessed_frames)

    result = preprocess_frames_handler(
        {
            "id": "job-preprocess",
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "frame_ids": ["frame-1"],
                "min_field_value": 2,
                "max_field_value": 200,
                "encoding": "jpg",
            },
        },
        context,
    )

    assert result["stage"] == PipelineStage.PREPROCESS_FRAMES.value
    assert result["frame_count"] == 1
    assert result["preprocessed_frame_ids"] == ["frame-1"]
    assert result["timings"]["schema_version"] == 1
    assert result["timings"]["unit_count"] == 1
    assert result["timings"]["phase_counts"]["selection.frame_metadata_lookup"] == 1
    assert retrieved == [("frame-1", context, "original")]
    assert preprocessed[0][1]["min_field_value"] == 2
    assert preprocessed[0][1]["max_field_value"] == 200
    assert stored[0][1]["encoding"] == "jpg"
    assert stored[0][0][0][0] == "frame-1"


def test_preprocess_frames_handler_processes_database_work_units(monkeypatch):
    class WorkUnitRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.record_batches = []

        def get_frame_records(self, frame_ids, **kwargs):
            self.record_batches.append(list(frame_ids))
            return [
                FrameRecord(
                    id=frame_id,
                    run_id="run-1",
                    asset_id="asset-1",
                    frame_index=index,
                    width=4,
                    height=4,
                    kvstore_hash=f"kv-{frame_id}",
                    preview_thumbhash=b"thumb",
                )
                for index, frame_id in enumerate(frame_ids)
            ]

    repository = WorkUnitRepository()
    context = make_context(repository)
    frame_ids = [f"frame-{index}" for index in range(26)]
    stored_batches = []

    monkeypatch.setattr(
        "Pelagia.workers.handlers.retrieve_frame",
        lambda frame_id, **kwargs: FrameData(
            sourcePath="/tmp",
            filename=f"{frame_id}.png",
            frameNumber=1,
            data=np.zeros((4, 4), dtype=np.uint8),
            metadata={"frame_id": frame_id},
        ),
    )
    monkeypatch.setattr(
        "Pelagia.workers.handlers.preprocess_frame_for_segmentation",
        lambda frame, **kwargs: frame,
    )

    def fake_store(frames, **kwargs):
        stored_batches.append([frame_id for frame_id, _ in frames])
        return [{"id": frame_id} for frame_id, _ in frames]

    monkeypatch.setattr("Pelagia.workers.handlers.store_preprocessed_frames", fake_store)

    result = preprocess_frames_handler(
        {
            "id": "job-preprocess-units",
            "project_id": "project-1",
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "payload": {"frame_ids": frame_ids, "background_correction": False},
        },
        context,
    )

    assert repository.record_batches == [frame_ids[:25], frame_ids[25:]]
    assert stored_batches == [frame_ids[:25], frame_ids[25:]]
    assert result["frame_count"] == 26
    assert result["timings"]["phase_counts"]["selection.frame_metadata_lookup"] == 2


def test_preprocess_frames_handler_accepts_frame_ids_from_multiple_assets(monkeypatch):
    class MixedAssetRepository(FakeRepository):
        def get_frame_record(self, frame_id, **kwargs):
            records = {
                "frame-1": ("asset-1", "run-1"),
                "frame-2": ("asset-2", "run-2"),
            }
            asset_id, run_id = records.get(frame_id, (None, None))
            if asset_id is None:
                return None
            return FrameRecord(
                id=frame_id,
                run_id=run_id,
                asset_id=asset_id,
                frame_index=1,
                width=4,
                height=4,
                kvstore_hash="kvstore-key",
                preview_thumbhash=b"thumb",
                background_kvstore_hash="background-key",
                background_payload_ref="background-key",
            )

    repository = MixedAssetRepository()
    context = make_context(repository)
    monkeypatch.setattr(
        "Pelagia.workers.handlers.retrieve_frame",
        lambda frame_id, **kwargs: FrameData(
            sourcePath="/tmp",
            filename=f"{frame_id}.png",
            frameNumber=1,
            data=np.zeros((4, 4), dtype=np.uint8),
            metadata={"frame_id": frame_id},
        ),
    )
    monkeypatch.setattr("Pelagia.workers.handlers.preprocess_frame_for_segmentation", lambda frame, **kwargs: frame)
    monkeypatch.setattr(
        "Pelagia.workers.handlers.store_preprocessed_frames",
        lambda frames, **kwargs: [{"id": frame_id} for frame_id, _ in frames],
    )

    result = preprocess_frames_handler(
        {
            "id": "job-preprocess-mixed",
            "project_id": "project-1",
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "payload": {"frame_ids": ["frame-1", "frame-2"], "background_correction": False},
        },
        context,
    )

    assert result["asset_id"] is None
    assert result["run_id"] is None
    assert result["asset_ids"] == ["asset-1", "asset-2"]
    assert result["run_ids"] == ["run-1", "run-2"]
    assert result["preprocessed_frame_ids"] == ["frame-1", "frame-2"]


def test_roi_detection_handler_accepts_frame_ids_from_multiple_assets(monkeypatch):
    class MixedAssetRepository(FakeRepository):
        def get_frame_record(self, frame_id, **kwargs):
            records = {
                "frame-1": ("asset-1", "run-1"),
                "frame-2": ("asset-2", "run-2"),
            }
            asset_id, run_id = records.get(frame_id, (None, None))
            if asset_id is None:
                return None
            return FrameRecord(
                id=frame_id,
                run_id=run_id,
                asset_id=asset_id,
                frame_index=1,
                width=4,
                height=4,
                kvstore_hash="kvstore-key",
                preview_thumbhash=b"thumb",
            )

    repository = MixedAssetRepository()
    context = make_context(repository).for_project("project-1")
    monkeypatch.setattr(
        "Pelagia.workers.handlers.retrieve_frame",
        lambda frame_id, **kwargs: FrameData(
            sourcePath="/tmp",
            filename=f"{frame_id}.png",
            frameNumber=1,
            data=np.zeros((4, 4), dtype=np.uint8),
            metadata={"frame_id": frame_id},
        ),
    )

    def fake_segment_frame(frame, *, frame_record, **kwargs):
        return [
            DetectionRecord(
                run_id=frame_record.run_id,
                frame_id=frame_record.id,
                roi_index=0,
                bbox_x=0,
                bbox_y=0,
                bbox_w=1,
                bbox_h=1,
                area=1.0,
                perimeter=4.0,
                major_axis_length=1.0,
                minor_axis_length=1.0,
                min_gray_value=0,
                mean_gray_value=0.0,
                roi_payload=b"roi",
            )
        ]

    monkeypatch.setattr("Pelagia.workers.handlers.segment_frame", fake_segment_frame)

    result = roi_detection_handler(
        {
            "id": "job-segment-mixed",
            "project_id": "project-1",
            "stage": PipelineStage.SEGMENT.value,
            "payload": {"frame_ids": ["frame-1", "frame-2"]},
        },
        context,
    )

    assert result["asset_id"] is None
    assert result["run_id"] is None
    assert result["asset_ids"] == ["asset-1", "asset-2"]
    assert result["run_ids"] == ["run-1", "run-2"]
    assert [(run_id, frame_ids) for run_id, frame_ids, _ in repository.replaced_detections] == [
        ("run-1", ["frame-1"]),
        ("run-2", ["frame-2"]),
    ]
    assert repository.frame_status_count_refreshes == [
        {"project_id": "project-1", "frame_ids": ["frame-1", "frame-2"], "asset_id": None}
    ]


def test_preprocess_frames_handler_allows_frames_without_correction_fields(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo)
    monkeypatch.setattr(
        "Pelagia.workers.handlers.retrieve_frame",
        lambda frame_id, **kwargs: FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=np.zeros((4, 4), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        "Pelagia.workers.handlers.store_preprocessed_frames",
        lambda frames, **kwargs: [{"id": frame_id} for frame_id, _ in frames],
    )

    result = preprocess_frames_handler(
        {
            "id": "job-preprocess",
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"frame_ids": ["frame-1"]},
        },
        context,
    )

    assert result["frame_count"] == 1


def test_preprocess_frames_handler_skips_background_generation_when_present(monkeypatch):
    repo = FakeRepository()
    repo.frames_with_background.add("frame-1")
    context = make_context(repo)
    calls = []

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ensure_asset_background_windows",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "Pelagia.workers.handlers.retrieve_frame",
        lambda frame_id, context=None, payload_kind="original", frame_record=None: FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=np.zeros((4, 4), dtype=np.uint8),
            bkg=np.ones((4, 4), dtype=np.float32),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        ),
    )
    monkeypatch.setattr("Pelagia.workers.handlers.preprocess_frame_for_segmentation", lambda frame, **kwargs: frame)
    monkeypatch.setattr(
        "Pelagia.workers.handlers.store_preprocessed_frames",
        lambda frames, **kwargs: [{"id": frame_id} for frame_id, _ in frames],
    )

    result = preprocess_frames_handler(
        {
            "id": "job-preprocess",
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "frame_ids": ["frame-1"],
                "background_correction": True,
            },
        },
        context,
    )

    assert calls == []
    assert "background_generation" not in result


def test_preprocess_frames_handler_does_not_regenerate_stale_background_reference(monkeypatch):
    repo = FakeRepository()
    repo.frames_with_background.add("frame-1")
    context = make_context(repo)
    def fake_retrieve_frame(frame_id, context=None, payload_kind="original", frame_record=None):
        raise KeyError("stale-background-key")

    monkeypatch.setattr("Pelagia.workers.handlers.retrieve_frame", fake_retrieve_frame)

    with pytest.raises(KeyError, match="stale-background-key"):
        preprocess_frames_handler(
            {
                "id": "job-preprocess",
                "stage": PipelineStage.PREPROCESS_FRAMES.value,
                "run_id": "run-1",
                "asset_id": "asset-1",
                "payload": {"frame_ids": ["frame-1"], "background_correction": True},
            },
            context,
        )


def test_preprocess_frames_handler_uses_project_context(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo).for_project("project-1")

    monkeypatch.setattr(
        "Pelagia.workers.handlers.retrieve_frame",
        lambda frame_id, context=None, payload_kind="original", frame_record=None: FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=np.zeros((4, 4), dtype=np.uint8),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        ),
    )
    monkeypatch.setattr("Pelagia.workers.handlers.preprocess_frame_for_segmentation", lambda frame, **kwargs: frame)
    monkeypatch.setattr(
        "Pelagia.workers.handlers.store_preprocessed_frames",
        lambda frames, **kwargs: [{"id": frame_id} for frame_id, _ in frames],
    )

    result = preprocess_frames_handler(
        {
            "id": "job-preprocess",
            "project_id": "project-1",
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"frame_ids": ["frame-1"]},
        },
        context,
    )

    assert result["project_id"] == "project-1"
    assert ("get_frame_record", "project-1") in repo.project_calls
    assert repo.frame_stage_status_updates == [
        {
            "project_id": "project-1",
            "frame_ids": ["frame-1"],
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "status": "working",
            "job_id": "job-preprocess",
        },
        {
            "project_id": "project-1",
            "frame_ids": ["frame-1"],
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "status": "succeeded",
            "job_id": "job-preprocess",
        },
    ]
    assert repo.processing_status_snapshot_touches == [{"project_id": "project-1"}]


def test_background_frames_handler_generates_background_for_frame_batch(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo)
    calls = []

    def fake_generate_background_for_frames(frame_ids, **kwargs):
        calls.append((frame_ids, kwargs))
        return {
            "background_payload_ref": "background-key",
            "frame_ids": frame_ids,
            "frame_count": len(frame_ids),
            "updated_frame_count": len(frame_ids),
        }

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ensure_asset_background_windows",
        fake_generate_background_for_frames,
    )

    result = background_frames_handler(
        {
            "id": "job-background",
            "stage": PipelineStage.BACKGROUND_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "frame_ids": ["frame-1"],
                "payload_kind": "original",
                "encoding": "raw",
            },
        },
        context,
    )

    assert result["stage"] == PipelineStage.BACKGROUND_FRAMES.value
    assert result["run_id"] == "run-1"
    assert result["asset_id"] == "asset-1"
    assert result["background_payload_ref"] == "background-key"
    assert result["timings"]["unit_count"] == 1
    assert result["timings"]["phase_counts"]["selection.frame_metadata_lookup"] == 1
    assert calls == [
        (
            ["frame-1"],
            {"context": context, "payload_kind": "original", "encoding": "raw"},
        )
    ]


def test_roi_detection_handler_segments_frames_and_stores_detections(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo)
    retrieved = []
    segmented = []

    def fake_retrieve_frame(frame_id, context=None):
        retrieved.append((frame_id, context))
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=None,
            width=4,
            height=4,
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    def fake_segment_frame(frame, **kwargs):
        segmented.append((frame, kwargs))
        return [
            DetectionRecord(
                run_id="run-1",
                frame_id=frame.metadata["frame_id"],
                roi_index=1,
                bbox_x=1,
                bbox_y=1,
                bbox_w=2,
                bbox_h=2,
                area=4.0,
                perimeter=8.0,
                major_axis_length=2.0,
                minor_axis_length=2.0,
                min_gray_value=0,
                mean_gray_value=1.0,
                roi_payload=b"roi",
            )
        ]

    monkeypatch.setattr("Pelagia.workers.handlers.retrieve_frame", fake_retrieve_frame)
    monkeypatch.setattr("Pelagia.workers.handlers.segment_frame", fake_segment_frame)

    result = roi_detection_handler(
        {
            "id": "job-2",
            "stage": PipelineStage.SEGMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "frame_ids": ["frame-1"],
                "threshold": 100,
                "min_perimeter": 3,
                "padding": 5,
                "roi_encoding": "raw",
                "mask_augmentation_steps": ["erode"],
                "erode_iterations": 2,
                "roi_assembly_method": "contours",
                "min_area": 4,
                "store_roi_payload_min_area": 10,
                "always_store_mask": False,
            },
        },
        context,
    )

    assert result["stage"] == PipelineStage.SEGMENT.value
    assert result["run_id"] == "run-1"
    assert result["asset_id"] == "asset-1"
    assert result["frame_count"] == 1
    assert result["detection_count"] == 1
    assert result["detection_ids"] == ["det-1"]
    assert result["timings"]["unit_count"] == 1
    assert result["timings"]["phase_counts"]["segmentation.database_update"] == 1
    assert retrieved == [("frame-1", context)]
    assert segmented[0][1]["threshold"] == 100
    assert segmented[0][1]["min_perimeter"] == 3
    assert segmented[0][1]["padding"] == 5
    assert segmented[0][1]["roi_encoding"] == "raw"
    assert segmented[0][1]["mask_augmentation_steps"] == ["erode"]
    assert segmented[0][1]["erode_iterations"] == 2
    assert segmented[0][1]["roi_assembly_method"] == "contours"
    assert segmented[0][1]["min_area"] == 4
    assert segmented[0][1]["store_roi_payload_min_area"] == 10
    assert segmented[0][1]["always_store_mask"] is False
    assert result["resolved_options"]["roi_assembly"]["roi_assembly_method"] == "contours"
    assert repo.replaced_detections[0][0] == "run-1"
    assert repo.replaced_detections[0][1] == ["frame-1"]


def test_roi_detection_handler_preserves_project_id(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo).for_project("project-1")

    monkeypatch.setattr(
        "Pelagia.workers.handlers.retrieve_frame",
        lambda frame_id, context=None, payload_kind="original": FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=None,
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        ),
    )
    monkeypatch.setattr("Pelagia.workers.handlers.segment_frame", lambda frame, **kwargs: [])

    result = roi_detection_handler(
        {
            "id": "job-segment",
            "project_id": "project-1",
            "stage": PipelineStage.SEGMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"frame_ids": ["frame-1"]},
        },
        context,
    )

    assert result["project_id"] == "project-1"
    assert ("get_frame_record", "project-1") in repo.project_calls
    assert ("replace_frame_detections", "project-1") in repo.project_calls
    assert repo.frame_stage_status_updates == [
        {
            "project_id": "project-1",
            "frame_ids": ["frame-1"],
            "stage": PipelineStage.SEGMENT.value,
            "status": "working",
            "job_id": "job-segment",
        },
        {
            "project_id": "project-1",
            "frame_ids": ["frame-1"],
            "stage": PipelineStage.SEGMENT.value,
            "status": "succeeded",
            "job_id": "job-segment",
        },
    ]
    assert repo.frame_status_count_refreshes == [
        {"project_id": "project-1", "frame_ids": ["frame-1"], "asset_id": "asset-1"}
    ]
    assert repo.processing_status_snapshot_touches == [{"project_id": "project-1"}]


def test_roi_refinement_handler_refines_and_stores_candidate_rois():
    repo = FakeRepository()
    context = make_context(repo)

    result = roi_refinement_handler(
        {
            "id": "job-refine",
            "stage": PipelineStage.ROI_REFINEMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "detection_ids": ["det-1"],
                "model_kind": "identity",
                "allow_frame_expansion": False,
                "batch_size": 1,
                "encoding": "raw",
            },
        },
        context,
    )

    assert result["stage"] == PipelineStage.ROI_REFINEMENT.value
    assert result["run_id"] == "run-1"
    assert result["asset_id"] == "asset-1"
    assert result["detection_count"] == 1
    assert result["refined_count"] == 1
    assert result["detection_ids"] == ["det-1"]
    assert result["refined_detection_ids"] == ["refined-det-1"]
    assert result["refinement_method"] == "identity"
    assert result["timings"]["unit_count"] == 1
    assert result["timings"]["phase_counts"]["refinement.model_inference"] == 1
    assert result["timings"]["phase_counts"]["refinement.database_update"] == 1
    assert result["resolved_options"]["batch_size"] == 1
    assert result["resolved_options"]["allow_frame_expansion"] is False
    assert repo.refined_detections[0][0][0] == "det-1"
    assert repo.refined_detections[0][0][1].metadata["detection_stage"] == "refined"


def test_roi_refinement_handler_preserves_project_id():
    repo = FakeRepository()
    context = make_context(repo).for_project("project-1")

    result = roi_refinement_handler(
        {
            "id": "job-refine",
            "project_id": "project-1",
            "stage": PipelineStage.ROI_REFINEMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "detection_ids": ["det-1"],
                "model_kind": "identity",
                "allow_frame_expansion": False,
                "encoding": "raw",
            },
        },
        context,
    )

    assert result["project_id"] == "project-1"
    assert ("get_detection", "project-1") in repo.project_calls
    assert ("upsert_refined_detections", "project-1") in repo.project_calls
    assert repo.frame_stage_status_updates == [
        {
            "project_id": "project-1",
            "frame_ids": ["frame-1"],
            "stage": PipelineStage.ROI_REFINEMENT.value,
            "status": "working",
            "job_id": "job-refine",
        },
        {
            "project_id": "project-1",
            "frame_ids": ["frame-1"],
            "stage": PipelineStage.ROI_REFINEMENT.value,
            "status": "succeeded",
            "job_id": "job-refine",
        },
    ]
    assert repo.frame_status_count_refreshes == [
        {"project_id": "project-1", "frame_ids": ["frame-1"], "asset_id": None}
    ]
    assert repo.processing_status_snapshot_touches == [{"project_id": "project-1"}]


def test_roi_refinement_handler_auto_encoding_reuses_candidate_encoding():
    repo = FakeRepository()
    context = make_context(repo)

    result = roi_refinement_handler(
        {
            "id": "job-refine-auto",
            "stage": PipelineStage.ROI_REFINEMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "detection_ids": ["det-1"],
                "model_kind": "identity",
                "allow_frame_expansion": False,
                "encoding": "auto",
            },
        },
        context,
    )

    assert result["refined_count"] == 1
    assert result["resolved_options"]["encoding"] is None
    assert repo.refined_detections[0][0][1].roi_encoding == "raw"


def test_roi_refinement_handler_loads_frame_crop_when_roi_payload_is_missing(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo)
    loaded = []

    def fake_retrieve_frame(frame_id, *, context, payload_kind):
        loaded.append((frame_id, payload_kind))
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=np.array([[0, 20], [40, 80]], dtype=np.uint8),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    monkeypatch.setattr("Pelagia.workers.handlers.retrieve_frame", fake_retrieve_frame)

    result = roi_refinement_handler(
        {
            "id": "job-refine-no-roi",
            "stage": PipelineStage.ROI_REFINEMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "detection_ids": ["det-no-roi"],
                "model_kind": "identity",
                "encoding": "raw",
            },
        },
        context,
    )

    assert result["refined_count"] == 1
    assert loaded == [("frame-1", "preprocessed")]
    assert repo.refined_detections[0][0][1].metadata["refinement_initial_roi_source"] == "frame"


def test_default_registry_includes_roi_detection_handler(monkeypatch):
    repo = FakeRepository()
    repo.claimed_jobs = [
        {
            "id": "job-2",
            "stage": PipelineStage.SEGMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"frame_ids": ["frame-1"]},
        }
    ]
    context = make_context(repo)

    monkeypatch.setattr(
        "Pelagia.workers.handlers.retrieve_frame",
        lambda frame_id, context=None: FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=None,
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        ),
    )
    monkeypatch.setattr("Pelagia.workers.handlers.segment_frame", lambda frame, **kwargs: [])

    worker = Worker(
        context=context,
        handlers=default_handler_registry(),
        worker_id="pytest-worker",
    )

    assert worker.run_once(stages=[PipelineStage.SEGMENT]) == 1
    assert repo.completed[0][1]["stage"] == PipelineStage.SEGMENT.value
    assert repo.completed[0][1]["detection_count"] == 0


def test_default_registry_includes_roi_refinement_handler():
    repo = FakeRepository()
    repo.claimed_jobs = [
        {
            "id": "job-refine",
            "stage": PipelineStage.ROI_REFINEMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "detection_ids": ["det-1"],
                "model_kind": "identity",
                "allow_frame_expansion": False,
                "encoding": "raw",
            },
        }
    ]
    worker = Worker(
        context=make_context(repo),
        handlers=default_handler_registry(),
        worker_id="pytest-worker",
    )

    assert worker.run_once(stages=[PipelineStage.ROI_REFINEMENT]) == 1
    assert repo.completed[0][1]["stage"] == PipelineStage.ROI_REFINEMENT.value
    assert repo.completed[0][1]["refined_count"] == 1


def test_default_registry_includes_background_frames_handler(monkeypatch):
    repo = FakeRepository()
    repo.claimed_jobs = [
        {
            "id": "job-background",
            "stage": PipelineStage.BACKGROUND_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"frame_ids": ["frame-1"], "payload_kind": "original"},
        }
    ]

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ensure_asset_background_windows",
        lambda frame_ids, **kwargs: {
            "background_payload_ref": "background-key",
            "frame_ids": frame_ids,
            "frame_count": len(frame_ids),
            "updated_frame_count": len(frame_ids),
        },
    )

    worker = Worker(
        context=make_context(repo),
        handlers=default_handler_registry(),
        worker_id="pytest-worker",
    )

    assert worker.run_once(stages=[PipelineStage.BACKGROUND_FRAMES]) == 1
    assert repo.completed[0][1]["stage"] == PipelineStage.BACKGROUND_FRAMES.value
    assert repo.completed[0][1]["background_payload_ref"] == "background-key"
    assert repo.completed[0][1]["frame_count"] == 1


def test_worker_run_once_uses_default_extract_frames_handler(monkeypatch):
    repo = FakeRepository()
    repo.claimed_jobs = [
        {
            "id": "job-1",
            "stage": PipelineStage.EXTRACT_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {},
        }
    ]
    context = make_context(repo)

    monkeypatch.setattr(
        "Pelagia.workers.handlers.ingest_module.ingest_video_file",
        lambda *args, **kwargs: [{"id": 10}, {"id": 11}],
    )

    worker = Worker(
        context=context,
        handlers=default_handler_registry(),
        worker_id="pytest-worker",
    )

    assert worker.run_once(stages=[PipelineStage.EXTRACT_FRAMES]) == 1
    assert repo.completed[0][0] == "job-1"
    assert repo.completed[0][1]["frame_count"] == 2
    assert repo.failures == []


def test_worker_run_once_marks_frame_stage_failed_for_project_job():
    repo = FakeRepository()
    repo.claimed_jobs = [
        {
            "id": "job-fail",
            "project_id": "project-1",
            "stage": PipelineStage.SEGMENT.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"frame_ids": ["frame-1"]},
        }
    ]
    registry = HandlerRegistry()

    def failing_handler(job, context):
        raise RuntimeError("decode failed")

    registry.register(PipelineStage.SEGMENT, failing_handler)
    worker = Worker(
        context=make_context(repo),
        handlers=registry,
        worker_id="pytest-worker",
    )

    assert worker.run_once(stages=[PipelineStage.SEGMENT]) == 1
    assert repo.completed == []
    assert repo.failures[0][0] == "job-fail"
    assert "decode failed" in repo.failures[0][1]
    assert repo.frame_stage_status_updates == [
        {
            "project_id": "project-1",
            "frame_ids": ["frame-1"],
            "stage": PipelineStage.SEGMENT.value,
            "status": "failed",
            "job_id": "job-fail",
        }
    ]
    assert repo.processing_status_snapshot_touches == [{"project_id": "project-1"}]


def test_worker_run_once_stops_when_shutdown_requested():
    repo = FakeRepository()
    repo.shutdown_requested = True
    worker = Worker(
        context=make_context(repo),
        handlers=default_handler_registry(),
        worker_id="pytest-worker",
    )

    assert worker.run_once(stages=[PipelineStage.EXTRACT_FRAMES]) == 0
    assert repo.touches[-1]["status"] == "stopped"


def test_worker_run_forever_requeues_and_stops_from_event():
    repo = FakeRepository()
    stop_after_one_loop = []

    class StopEvent:
        def is_set(self):
            return bool(stop_after_one_loop)

        def wait(self, _seconds):
            stop_after_one_loop.append(True)

    worker = Worker(
        context=make_context(repo),
        handlers=default_handler_registry(),
        worker_id="pytest-worker",
    )

    worker.run_forever(
        stages=[PipelineStage.EXTRACT_FRAMES],
        idle_sleep_seconds=0,
        requeue_interval_seconds=0,
        stop_event=StopEvent(),
    )

    assert repo.requeued >= 1
    assert repo.touches[0]["status"] == "idle"
    assert repo.touches[-1]["status"] == "stopped"
    assert repo.touches[-1]["shutdown_requested"] is False
