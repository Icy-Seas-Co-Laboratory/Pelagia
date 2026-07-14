from Pelagia.config import CoreConfig
from Pelagia.services.context import AppContext
from Pelagia.services.processing_queue import PreprocessQueueRequest, ProcessingQueueService


class QueueRepository:
    def __init__(self):
        self.created = []

    def plan_preprocess_frames(self, *, project_id, filters):
        assert project_id == "project-1"
        return [
            {"frame_id": "frame-b", "asset_id": "asset-2", "run_id": "run-2", "frame_index": 1, "payload_ref": "b-key"},
            {"frame_id": "frame-a", "asset_id": "asset-1", "run_id": "run-1", "frame_index": 1, "payload_ref": "a-key"},
            {"frame_id": "frame-c", "asset_id": "asset-1", "run_id": "run-1", "frame_index": 2, "payload_ref": "a-key"},
        ]

    def create_preprocess_jobs(self, *, project_id, jobs, eligible_statuses, priority):
        assert eligible_statuses == ["unknown", "failed"]
        self.created = jobs
        return [{"id": f"job-{index}"} for index, _ in enumerate(jobs, start=1)]

    def plan_segment_frames(self, *, project_id, filters, payload_kind):
        return self.plan_preprocess_frames(project_id=project_id, filters=filters)

    def plan_roi_refinement_detections(self, *, project_id, filters):
        return [
            {"detection_id": "det-2", "frame_id": "frame-b", "asset_id": "asset-2", "run_id": "run-2", "roi_index": 0},
            {"detection_id": "det-1", "frame_id": "frame-a", "asset_id": "asset-1", "run_id": "run-1", "roi_index": 1},
        ]

    def create_job(self, stage, **kwargs):
        self.created.append({"stage": stage, **kwargs})
        return {"id": f"job-{len(self.created)}"}


def test_preprocess_queue_orders_and_chunks_frames_by_payload_ref():
    repository = QueueRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))

    result = service.queue_preprocess(
        PreprocessQueueRequest(filters={}, options={"flatfield_correction": True}, max_units=2),
        project_id="project-1",
    )

    assert result["ordering"] == "kvstore_hash"
    assert result["batch_sizes"] == [2, 1]
    assert result["job_ids"] == ["job-1", "job-2"]
    assert repository.created[0]["frame_ids"] == ["frame-a", "frame-c"]
    assert repository.created[0]["asset_id"] == "asset-1"
    assert repository.created[1]["asset_id"] == "asset-2"


def test_preprocess_queue_dry_run_does_not_create_jobs():
    repository = QueueRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))

    result = service.queue_preprocess(
        PreprocessQueueRequest(filters={}, options={}, max_units=250, dry_run=True),
        project_id="project-1",
    )

    assert result["matched_count"] == 3
    assert result["job_count"] == 1
    assert "job_ids" not in result
    assert repository.created == []


def test_preprocess_queue_resolves_each_requested_asset():
    class MultiAssetRepository(QueueRepository):
        def __init__(self):
            super().__init__()
            self.asset_filters = []

        def plan_preprocess_frames(self, *, project_id, filters):
            self.asset_filters.append(filters.get("asset_id"))
            return []

    repository = MultiAssetRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))

    service.queue_preprocess(
        PreprocessQueueRequest(filters={"asset_ids": ["asset-1", "asset-2"]}, options={}, dry_run=True),
        project_id="project-1",
    )

    assert repository.asset_filters == ["asset-1", "asset-2"]


def test_segment_and_refinement_queue_apply_stage_specific_ordering():
    repository = QueueRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))

    segment = service.queue_segment(
        PreprocessQueueRequest(filters={}, options={"frame_payload_kind": "preprocessed"}, max_units=2),
        project_id="project-1",
    )
    refinement = service.queue_roi_refinement(
        PreprocessQueueRequest(filters={}, options={}, max_units=1),
        project_id="project-1",
    )

    assert segment["ordering"] == "kvstore_hash"
    assert refinement["ordering"] == "frame_id"
    assert refinement["job_count"] == 2
    refinement_payloads = [job["payload"] for job in repository.created if job["stage"] == "roi_refinement"]
    assert refinement_payloads[0]["detection_ids"] == ["det-1"]
