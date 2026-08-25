import pytest

from Pelagia.config import CoreConfig
from Pelagia.services.context import AppContext
from Pelagia.services.processing_queue import (
    PREPROCESS_FRAMES_PER_JOB,
    ROI_REFINEMENT_DETECTIONS_PER_JOB,
    SEGMENT_FRAMES_PER_JOB,
    PreprocessQueueRequest,
    ProcessingSeriesRequest,
    ProcessingQueueService,
)
from Pelagia.storage.postgres import PostgresRepository, _initial_job_progress


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

    def create_preprocess_jobs(
        self,
        *,
        project_id,
        jobs,
        eligible_statuses,
        priority,
        submitted_by_user_id=None,
        submitted_by_username=None,
    ):
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


def test_initial_job_progress_counts_queued_frame_and_roi_units():
    frame_progress = _initial_job_progress(
        "segment",
        "queued",
        {"frame_ids": ["frame-1", "frame-2", "frame-1"]},
    )
    roi_progress = _initial_job_progress(
        "roi_refinement",
        "queued",
        {"detection_ids": ["roi-1", "roi-2", "roi-2"]},
    )

    assert frame_progress["unit"] == "frames"
    assert frame_progress["total"] == 2
    assert frame_progress["completed"] == 0
    assert roi_progress["unit"] == "rois"
    assert roi_progress["total"] == 2
    assert roi_progress["completed"] == 0


class SeriesRepository(QueueRepository):
    def __init__(self, *, empty_first=False):
        super().__init__()
        self.empty_first = empty_first
        self.series = None
        self.steps = []
        self.attached = []
        self.claims = 0
        self.project_ids = []

    def create_processing_series(self, *, project_id, steps, **kwargs):
        self.project_ids.append(project_id)
        self.steps = [dict(step, id=f"step-{index}", series_id="series-1", status="queued") for index, step in enumerate(steps)]
        self.series = {"id": "series-1", "project_id": project_id, "status": "queued", "failure_policy": kwargs["failure_policy"]}
        return {**self.series, "steps": self.steps}

    def get_processing_series(self, series_id, *, project_id=None):
        if self.series is None or series_id != "series-1" or (project_id and project_id != self.series["project_id"]):
            return None
        return {**self.series, "steps": self.steps}

    def claim_processing_series_step(self, series_id, *, project_id):
        self.project_ids.append(project_id)
        if project_id != self.series["project_id"]:
            return None
        for step in self.steps:
            if step["status"] == "queued":
                step["status"] = "planning"
                self.claims += 1
                return step
        return None

    def attach_processing_work_units(self, *, series_id, step_id, job_ids, matched_count):
        self.attached.append((step_id, list(job_ids), matched_count))
        step = next(step for step in self.steps if step["id"] == step_id)
        step["status"] = "active" if job_ids else "skipped"

    def advance_processing_series_for_job(self, job_id):
        return {"series_id": "series-1", "step_id": "step-0", "ready": True, "failed": False}


def test_series_plans_first_step_and_preserves_project_scope():
    repository = SeriesRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))

    series = service.create_series(
        ProcessingSeriesRequest(steps=({"stage": "preprocess_frames", "filters": {}, "options": {}},)),
        project_id="project-1",
    )

    assert series["id"] == "series-1"
    assert repository.project_ids == ["project-1", "project-1"]
    assert repository.attached == [("step-0", ["job-1"], 3)]
    assert service.advance_series("series-1", project_id="other-project") is None


def test_series_no_candidate_step_is_skipped_and_next_step_is_planned():
    class EmptyPreprocessSeriesRepository(SeriesRepository):
        def plan_preprocess_frames(self, *, project_id, filters):
            return []

    repository = EmptyPreprocessSeriesRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))
    service.create_series(
        ProcessingSeriesRequest(steps=(
            {"stage": "preprocess_frames", "filters": {}, "options": {}},
            {"stage": "segment", "filters": {}, "options": {}},
        )), project_id="project-1",
    )

    assert repository.attached[0] == ("step-0", [], 0)
    assert repository.attached[1][0] == "step-1"
    assert repository.steps[0]["status"] == "skipped"


def test_series_rejects_noncanonical_processing_order():
    repository = SeriesRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))

    with pytest.raises(ValueError, match="canonical Pelagia order"):
        service.create_series(
            ProcessingSeriesRequest(steps=(
                {"stage": "segment", "filters": {}, "options": {}},
                {"stage": "preprocess_frames", "filters": {}, "options": {}},
            )),
            project_id="project-1",
        )


def test_completion_director_is_idempotent_after_a_series_job():
    repository = SeriesRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))
    service.create_series(ProcessingSeriesRequest(steps=({"stage": "preprocess_frames", "filters": {}, "options": {}},)), project_id="project-1")

    service.advance_series_for_job("job-1")
    service.advance_series_for_job("job-1")

    assert repository.claims == 1


def test_series_request_validates_failure_policy_and_dry_run_plans_without_writes():
    repository = SeriesRepository()
    service = ProcessingQueueService(AppContext(config=CoreConfig(), repository=repository))
    result = service.create_series(
        ProcessingSeriesRequest(steps=({"stage": "segment", "filters": {}, "options": {}},), dry_run=True, failure_policy="continue"),
        project_id="project-1",
    )
    assert result["dry_run"] is True
    assert repository.series is None
    with pytest.raises(ValueError, match="failure_policy"):
        service.create_series(ProcessingSeriesRequest(steps=({"stage": "segment"},), failure_policy="ignore"), project_id="project-1")
