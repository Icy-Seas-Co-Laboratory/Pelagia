from Pelagia.config import CoreConfig
from Pelagia.domain import PipelineStage
from Pelagia.services.context import AppContext
from Pelagia.workers.handlers import default_handler_registry, extract_frames_handler
from Pelagia.workers.worker import Worker


class FakeRepository:
    def __init__(self):
        self.assets = {"asset-1": {"id": "asset-1", "path": "/tmp/source.avi"}}
        self.claimed_jobs = []
        self.completed = []
        self.failures = []
        self.created_jobs = []
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
        "Pelagia.workers.handlers.video_ingest_module.ingest_video_file",
        fake_ingest_video_file,
    )

    job = {
        "id": "job-1",
        "stage": PipelineStage.EXTRACT_FRAMES.value,
        "run_id": "run-1",
        "asset_id": "asset-1",
        "payload": {
            "n_tile": 2,
            "dest_path": "/tmp/out",
            "flatfield_correction": False,
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
    assert kwargs["dest_path"] == "/tmp/out"
    assert kwargs["flatfield_correction"] is False
    assert kwargs["metadata"]["source"] == "test"
    assert kwargs["metadata"]["worker_job_id"] == "job-1"


def test_extract_frames_handler_can_enqueue_segment_job(monkeypatch):
    repo = FakeRepository()
    context = make_context(repo)

    monkeypatch.setattr(
        "Pelagia.workers.handlers.video_ingest_module.ingest_video_file",
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
                "segmentation_padding": 4,
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
    }
    assert repo.created_jobs[0]["depends_on"] == ["job-1"]


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
        "Pelagia.workers.handlers.video_ingest_module.ingest_video_file",
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
