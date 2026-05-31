from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from Pelagia.api import create_app
from Pelagia.config import CoreConfig
from Pelagia.domain import PipelineStage
from Pelagia.services.context import AppContext


class FakeRepository:
    schema = "pelagia"

    def __init__(self):
        self.created_jobs = []
        self.registered_runs = []
        self.shutdown_requests = []
        self.priority_updates = []

    def list_runs(self, **kwargs):
        return [{"id": "run-1", **kwargs}]

    def get_run(self, run_id):
        if run_id != "run-1":
            return None
        return {"id": run_id, "status": "queued", "job_summary": []}

    def list_assets(self, **kwargs):
        return [
            {
                "id": "asset-1",
                "run_id": kwargs.get("run_id"),
                "kind": "video",
                "collections": ["skq202510S-T1", "test"],
                **kwargs,
            }
        ]

    def get_asset(self, asset_id):
        if asset_id != "asset-1":
            return None
        return {"id": asset_id, "run_id": "run-1", "kind": "video", "collections": ["test"]}

    def list_frames(self, asset_id, **kwargs):
        return [{"id": 1, "asset_id": asset_id, "frame_index": 2, "frame_png": b"abc", **kwargs}]

    def get_frame_by_asset_index(self, asset_id, frame_index):
        if asset_id != "asset-1" or frame_index != 2:
            return None
        return {"id": 1, "asset_id": asset_id, "frame_index": frame_index}

    def list_detections(self, asset_id):
        return [
            {
                "id": "det-1",
                "asset_id": asset_id,
                "roi_payload": b"roi",
                "mask_payload": b"mask",
            }
        ]

    def list_models(self, **kwargs):
        return [{"id": "model-1", "model_key": "demo", **kwargs}]

    def list_collections(self, **kwargs):
        return [{"collection": kwargs.get("collection") or "test", "asset_count": 1, "limit": kwargs.get("limit")}]

    def get_model(self, model_id):
        return {"id": model_id, "model_key": "demo"} if model_id == "model-1" else None

    def list_jobs(self, **kwargs):
        return [{"id": "job-1", "stage": PipelineStage.EXTRACT_FRAMES.value, **kwargs}]

    def get_job(self, job_id):
        return {"id": job_id, "status": "queued"} if job_id == "job-1" else None

    def create_job(self, stage, **kwargs):
        stage_value = stage.value if hasattr(stage, "value") else stage
        job = {"id": "job-new", "stage": stage_value, **kwargs}
        self.created_jobs.append(job)
        return job

    def list_job_events(self, **kwargs):
        return [{"id": 1, "event_type": "job.created", **kwargs}]

    def pause_job(self, job_id, reason=None):
        return {"id": job_id, "status": "paused", "reason": reason}

    def resume_job(self, job_id, reason=None):
        return {"id": job_id, "status": "queued", "reason": reason}

    def retry_job(self, job_id):
        return {"id": job_id, "status": "queued"}

    def set_job_priority(self, job_id, priority, reason=None):
        self.priority_updates.append((job_id, priority, reason))
        return {"id": job_id, "priority": priority, "reason": reason}

    def list_worker_sessions(self, **kwargs):
        return [{"worker_id": "extract-1", "status": kwargs.get("status") or "idle", **kwargs}]

    def get_worker_session(self, worker_id):
        return {"worker_id": worker_id, "status": "idle"} if worker_id == "extract-1" else None

    def request_worker_shutdown(self, worker_id, reason=None):
        self.shutdown_requests.append((worker_id, reason))
        return {"worker_id": worker_id, "shutdown_requested": True, "reason": reason}

    def get_status_summary(self):
        return {"queue": {"queued": 2}, "workers": {"total": 1, "online": 1, "busy": 0}}

    def register_planned_run(self, planned_run):
        self.registered_runs.append(planned_run)
        return {"run": {"id": planned_run.manifest.run_id}, "asset_count": 1, "job_count": 0}

    def connect(self):
        raise AssertionError("API unit tests should not open a database connection.")


class FakeKVStore:
    initialized = True

    def status(self):
        return {
            "root_path": "/tmp/pelagia-kv",
            "initialized": True,
            "total_stored_blobs": 3,
        }

    def check_health(self):
        return {"healthy": True, "errors": [], "warnings": []}


def make_client():
    app = create_app(CoreConfig())
    repository = FakeRepository()
    kvstore = FakeKVStore()
    app.state.context = AppContext(config=CoreConfig(), repository=repository, kvstore=kvstore)
    return TestClient(app), repository, kvstore


def test_api_lists_system_status_without_live_database():
    client, _, _ = make_client()

    response = client.get("/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["queue"] == {"queued": 2}
    assert body["workers"]["online"] == 1
    assert body["kvstore"]["initialized"] is True


def test_api_kvstore_includes_status_and_health():
    client, _, _ = make_client()

    response = client.get("/kvstore")

    assert response.status_code == 200
    body = response.json()
    assert body["status"]["initialized"] is True
    assert body["status"]["total_stored_blobs"] == 3
    assert body["health"]["healthy"] is True


def test_api_can_create_queue_job():
    client, repository, _ = make_client()

    response = client.post(
        "/jobs",
        json={
            "stage": "extract_frames",
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"source_path": "/tmp/source.avi"},
        },
    )

    assert response.status_code == 200
    assert response.json()["job"]["stage"] == "extract_frames"
    assert repository.created_jobs[0]["run_id"] == "run-1"


def test_api_can_request_worker_shutdown():
    client, repository, _ = make_client()

    response = client.post("/workers/extract-1/shutdown", json={"reason": "maintenance"})

    assert response.status_code == 200
    assert response.json()["worker"]["shutdown_requested"] is True
    assert repository.shutdown_requests == [("extract-1", "maintenance")]


def test_api_asset_views_summarize_payload_bytes():
    client, _, _ = make_client()

    frame_response = client.get("/assets/asset-1/frames")
    detection_response = client.get("/assets/asset-1/detections")

    assert frame_response.json()["frames"][0]["frame_png_bytes"] == 3
    assert "frame_png" not in frame_response.json()["frames"][0]
    assert detection_response.json()["detections"][0]["roi_payload_bytes"] == 3
    assert detection_response.json()["detections"][0]["mask_payload_bytes"] == 4


def test_api_asset_frames_accepts_range_filters():
    client, _, _ = make_client()

    response = client.get("/assets/asset-1/frames?start_frame=2&end_frame=5&limit=10")

    assert response.status_code == 200
    frame = response.json()["frames"][0]
    assert frame["start_frame"] == 2
    assert frame["end_frame"] == 5
    assert frame["limit"] == 10


def test_api_framedata_returns_matrix_and_png(monkeypatch):
    from Pelagia.api.routes import assets

    class FakeFrame:
        def read(self):
            return np.array([[0, 128], [255, 64]], dtype=np.uint8)

    monkeypatch.setattr(assets, "retrieve_frame", lambda frame_id, context: FakeFrame())
    client, _, _ = make_client()

    matrix_response = client.get("/assets/asset-1/framedata/2?format=matrix")
    png_response = client.get("/assets/asset-1/framedata/2?format=png")

    assert matrix_response.status_code == 200
    assert matrix_response.json()["shape"] == [2, 2]
    assert matrix_response.json()["data"] == [[0, 128], [255, 64]]
    assert png_response.status_code == 200
    assert png_response.headers["content-type"] == "image/png"
    assert png_response.content.startswith(b"\x89PNG")


def test_api_framedata_returns_small_preview(monkeypatch):
    from Pelagia.api.routes import assets

    class FakeFrame:
        def read(self):
            return np.arange(50 * 100, dtype=np.uint8).reshape((50, 100))

    monkeypatch.setattr(assets, "retrieve_frame", lambda frame_id, context: FakeFrame())
    client, _, _ = make_client()

    response = client.get("/assets/asset-1/framedata/2?format=preview&preview_max_dim=16")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-pelagia-preview"] == "true"
    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (8, 16)


def test_api_queues_video_ingestion(tmp_path):
    client, repository, _ = make_client()
    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"not-a-real-video")

    response = client.post(
        "/ingestion/videos",
        json={
            "source_path": str(video_path),
            "n_tile": 2,
            "enqueue_segment": True,
            "segmentation_padding": 4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"]
    assert body["run_id"]
    assert body["job"]["stage"] == "extract_frames"
    assert repository.registered_runs[0].manifest.assets[0].path == str(video_path.resolve())
    assert repository.registered_runs[0].manifest.assets[0].collections == ["none"]
    assert repository.created_jobs[0]["payload"]["enqueue_segment"] is True


def test_api_queues_video_ingestion_with_collections(tmp_path):
    client, repository, _ = make_client()
    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"not-a-real-video")

    response = client.post(
        "/ingestion/videos",
        json={
            "source_path": str(video_path),
            "collections": "skq202510S-T1, test, transect1",
        },
    )

    assert response.status_code == 200
    assert response.json()["collections"] == ["skq202510S-T1", "test", "transect1"]
    assert repository.registered_runs[0].manifest.assets[0].collections == [
        "skq202510S-T1",
        "test",
        "transect1",
    ]
    assert repository.created_jobs[0]["payload"]["collections"] == [
        "skq202510S-T1",
        "test",
        "transect1",
    ]


def test_api_lists_collections_and_filters_assets():
    client, _, _ = make_client()

    collections_response = client.get("/collections")
    assets_response = client.get("/assets?collection=test")

    assert collections_response.status_code == 200
    assert collections_response.json()["collections"] == [{"collection": "test", "asset_count": 1, "limit": 100}]
    assert assets_response.json()["assets"][0]["collection"] == "test"


def test_api_search_endpoints_forward_optional_filters():
    client, _, _ = make_client()

    assets_response = client.get(
        "/assets?collection=test&kind=video&asset_key=sample&limit=5"
    )
    runs_response = client.get(
        "/runs?collection=test&instrument=api&source_type=video&status=registered&limit=7"
    )
    models_response = client.get("/models?task=classification&model_key=demo&limit=3")
    workers_response = client.get("/workers?status=idle&capability=extract_frames&limit=4")

    asset = assets_response.json()["assets"][0]
    assert asset["collection"] == "test"
    assert asset["kind"] == "video"
    assert asset["asset_key"] == "sample"
    assert asset["limit"] == 5
    run = runs_response.json()["runs"][0]
    assert run["collection"] == "test"
    assert run["instrument"] == "api"
    assert run["source_type"] == "video"
    assert run["status"] == "registered"
    assert run["limit"] == 7
    model = models_response.json()["models"][0]
    assert model["task"] == "classification"
    assert model["model_key"] == "demo"
    assert model["limit"] == 3
    worker = workers_response.json()["workers"][0]
    assert worker["status"] == "idle"
    assert worker["capability"] == "extract_frames"
    assert worker["limit"] == 4
