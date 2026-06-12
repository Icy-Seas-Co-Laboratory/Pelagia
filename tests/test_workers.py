import numpy as np

from Pelagia.config import CoreConfig
from Pelagia.domain import DetectionRecord, FrameRecord, PipelineStage
from Pelagia.processing.frame_model import FrameData
from Pelagia.services.context import AppContext
from Pelagia.workers.handlers import (
    background_frames_handler,
    default_handler_registry,
    extract_frames_handler,
    preprocess_frames_handler,
    roi_detection_handler,
    roi_refinement_handler,
)
from Pelagia.workers.worker import Worker


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
        self.shutdown_requested = False

    def get_asset(self, asset_id):
        return self.assets.get(asset_id)

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

    def list_frames(self, asset_id, **kwargs):
        return [{"id": "frame-1", "asset_id": asset_id, "frame_index": 1, **kwargs}]

    def get_frame_record(self, frame_id):
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
        )

    def replace_frame_detections(self, run_id, frame_ids, detections):
        rows = [
            {"id": f"det-{index}", "frame_id": detection.frame_id}
            for index, detection in enumerate(detections, start=1)
        ]
        self.replaced_detections.append((run_id, frame_ids, detections))
        return rows

    def get_detection(self, detection_id):
        if detection_id != "det-1":
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
            "roi_payload": roi.tobytes(order="C"),
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

    def upsert_refined_detections(self, refined_detections, *, job_id=None):
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


def test_extract_frames_handler_ingests_registered_asset(monkeypatch):
    repo = FakeRepository()
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

    assert result == {
        "stage": PipelineStage.EXTRACT_FRAMES.value,
        "run_id": "run-1",
        "asset_id": "asset-1",
        "source_path": "/tmp/source.avi",
        "frame_count": 2,
        "frame_ids": [10, 11],
    }

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
    assert kwargs["metadata"]["collections"] == ["test"]
    assert kwargs["metadata"]["worker_job_id"] == "job-1"


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
        "frame_ids": [10],
        "padding": 4,
        "roi_encoding": "raw",
        "collections": ["test"],
    }
    assert repo.created_jobs[0]["depends_on"] == ["job-1"]


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
    context = make_context(repo)
    retrieved = []
    preprocessed = []
    stored = []

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original"):
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

    def fake_store_preprocessed_frame(frame_id, frame, **kwargs):
        stored.append((frame_id, frame, kwargs))
        return {"id": frame_id}

    monkeypatch.setattr("Pelagia.workers.handlers.retrieve_frame", fake_retrieve_frame)
    monkeypatch.setattr("Pelagia.workers.handlers.preprocess_frame_for_segmentation", fake_preprocess_frame)
    monkeypatch.setattr("Pelagia.workers.handlers.store_preprocessed_frame", fake_store_preprocessed_frame)

    result = preprocess_frames_handler(
        {
            "id": "job-preprocess",
            "stage": PipelineStage.PREPROCESS_FRAMES.value,
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {
                "frame_ids": ["frame-1"],
                "flatfield_correction": False,
                "background_correction": True,
                "encoding": "jpg",
            },
        },
        context,
    )

    assert result["stage"] == PipelineStage.PREPROCESS_FRAMES.value
    assert result["frame_count"] == 1
    assert result["preprocessed_frame_ids"] == ["frame-1"]
    assert retrieved == [("frame-1", context, "original")]
    assert preprocessed[0][1]["flatfield_correction"] is False
    assert preprocessed[0][1]["background_correction"] is True
    assert stored[0][2]["encoding"] == "jpg"


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
        "Pelagia.workers.handlers.generate_background_for_frames",
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
    assert result["resolved_options"]["batch_size"] == 1
    assert result["resolved_options"]["allow_frame_expansion"] is False
    assert repo.refined_detections[0][0][0] == "det-1"
    assert repo.refined_detections[0][0][1].metadata["detection_stage"] == "refined"


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
        "Pelagia.workers.handlers.generate_background_for_frames",
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
