import pytest

from Pelagia.config import CoreConfig
from Pelagia.services.context import AppContext
from Pelagia.services.processing_queue import (
    PREPROCESS_FRAMES_PER_JOB,
    ROI_REFINEMENT_DETECTIONS_PER_JOB,
    SEGMENT_FRAMES_PER_JOB,
    PreprocessQueueRequest,
    ProcessingQueueService,
)
from Pelagia.storage.postgres import PostgresRepository


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


def test_processing_queue_api_request_rejects_client_batch_controls():
    from Pelagia.api.routes.processing import ProcessingQueueRequest

    with pytest.raises(ValueError, match="batch"):
        ProcessingQueueRequest(stage="preprocess_frames", batch={"max_units": 1})


def test_processing_queue_refinement_state_filters_match_frontend_contract():
    refined = PostgresRepository._candidate_refinement_state_clause(
        schema="pelagia",
        refinement_states=["refined"],
    )
    unrefined = PostgresRepository._candidate_refinement_state_clause(
        schema="pelagia",
        refinement_states=["unrefined"],
    )
    both = PostgresRepository._candidate_refinement_state_clause(
        schema="pelagia",
        refinement_states=["refined", "unrefined"],
    )
    default = PostgresRepository._candidate_refinement_state_clause(
        schema="pelagia",
        refinement_states=None,
    )

    assert refined is not None
    assert refined.startswith("EXISTS")
    assert unrefined is not None
    assert unrefined.startswith("NOT EXISTS")
    assert both is None
    assert default == unrefined


def test_preprocess_queue_orders_frames_by_payload_ref_with_backend_batch_limit():
    repository = QueueRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))

    result = service.queue_preprocess(
        PreprocessQueueRequest(filters={}, options={"flatfield_correction": True}),
        project_id="project-1",
    )

    assert result["ordering"] == "kvstore_hash"
    assert result["max_units_per_job"] == PREPROCESS_FRAMES_PER_JOB
    assert result["batch_sizes"] == [3]
    assert result["job_ids"] == ["job-1"]
    assert repository.created[0]["frame_ids"] == ["frame-a", "frame-c", "frame-b"]
    assert repository.created[0]["asset_id"] is None
    assert repository.created[0]["run_id"] is None


def test_preprocess_queue_dry_run_does_not_create_jobs():
    repository = QueueRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))

    result = service.queue_preprocess(
        PreprocessQueueRequest(filters={}, options={}, dry_run=True),
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
        PreprocessQueueRequest(filters={}, options={"frame_payload_kind": "preprocessed"}),
        project_id="project-1",
    )
    refinement = service.queue_roi_refinement(
        PreprocessQueueRequest(filters={}, options={}),
        project_id="project-1",
    )

    assert segment["ordering"] == "kvstore_hash"
    assert segment["max_units_per_job"] == SEGMENT_FRAMES_PER_JOB
    assert refinement["ordering"] == "frame_id"
    assert refinement["max_units_per_job"] == ROI_REFINEMENT_DETECTIONS_PER_JOB
    assert refinement["job_count"] == 1
    refinement_payloads = [job["payload"] for job in repository.created if job["stage"] == "roi_refinement"]
    assert refinement_payloads[0]["detection_ids"] == ["det-1", "det-2"]


def test_queue_stage_batch_limits_are_controlled_by_the_backend():
    class LargeQueueRepository(QueueRepository):
        def plan_preprocess_frames(self, *, project_id, filters):
            return [
                {
                    "frame_id": f"frame-{index}",
                    "asset_id": "asset-1",
                    "run_id": "run-1",
                    "frame_index": index,
                    "payload_ref": "payload-1",
                }
                for index in range(PREPROCESS_FRAMES_PER_JOB + 1)
            ]

        def plan_segment_frames(self, *, project_id, filters, payload_kind):
            return self.plan_preprocess_frames(project_id=project_id, filters=filters)

        def plan_roi_refinement_detections(self, *, project_id, filters):
            return [
                {
                    "detection_id": f"detection-{index}",
                    "frame_id": f"frame-{index}",
                    "asset_id": "asset-1",
                    "run_id": "run-1",
                    "roi_index": 0,
                }
                for index in range(ROI_REFINEMENT_DETECTIONS_PER_JOB + 1)
            ]

    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=LargeQueueRepository()))
    request = PreprocessQueueRequest(filters={}, options={}, dry_run=True)

    preprocess = service.queue_preprocess(request, project_id="project-1")
    segment = service.queue_segment(request, project_id="project-1")
    refinement = service.queue_roi_refinement(request, project_id="project-1")

    assert preprocess["batch_sizes"] == [PREPROCESS_FRAMES_PER_JOB, 1]
    assert segment["batch_sizes"] == [SEGMENT_FRAMES_PER_JOB, 1]
    assert refinement["batch_sizes"] == [ROI_REFINEMENT_DETECTIONS_PER_JOB, 1]
