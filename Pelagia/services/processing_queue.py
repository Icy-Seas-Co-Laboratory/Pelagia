"""Backend-owned planning and batching for processing jobs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..services.context import AppContext
from .job_commands import PreprocessFramesCommand, RoiRefinementCommand, SegmentFramesCommand, FrameSelection


PREPROCESS_FRAMES_PER_JOB = 1_000
SEGMENT_FRAMES_PER_JOB = 1_000
ROI_REFINEMENT_DETECTIONS_PER_JOB = 10_000
# These caps use decoded input array sizes.  They deliberately use decimal
# units because they are operational job limits (1 GB / 100 MB), not storage
# allocation limits expressed in GiB/MiB.
PREPROCESS_INPUT_BYTES_PER_JOB = 1_000_000_000
SEGMENT_INPUT_BYTES_PER_JOB = 1_000_000_000
ROI_REFINEMENT_INPUT_BYTES_PER_JOB = 100_000_000
INCREMENTAL_FRAME_PAGE_SIZE = 1_000
INCREMENTAL_REFINEMENT_PAGE_SIZE = 10_000
# A planning step has no durable director lease in the initial series schema.
# Keep the recovery timeout deliberately conservative: a second director must
# never reclaim work that a healthy planner is still materialising.
PROCESSING_SERIES_PLANNING_TIMEOUT_SECONDS = 15 * 60
# A claim now owns only one bounded database page.  A short lease recovers an
# API crash quickly; the row/advisory transaction fence protects a live page.
PROCESSING_SUBMISSION_CLAIM_SECONDS = 2 * 60
_TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "dead_lettered", "cancelled"})


def decoded_array_input_bytes(shape: Any, dtype: Any) -> int | None:
    """Return decoded array bytes, or ``None`` when its metadata is not usable.

    Queue planning must not turn missing metadata into a zero-byte estimate:
    callers use ``None`` to isolate that work into a one-unit job.
    """
    if not shape or dtype is None:
        return None
    try:
        dimensions = tuple(int(dimension) for dimension in shape)
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            return None
        itemsize = int(np.dtype(dtype).itemsize)
        if itemsize <= 0:
            return None
    except (TypeError, ValueError):
        return None
    total = itemsize
    for dimension in dimensions:
        total *= dimension
    return total


def batch_processing_units_by_limit(
    units: list[dict[str, Any]],
    max_units: int,
    max_bytes: int,
    extra_bytes_for_batch: Callable[[set[str], dict[str, Any]], int | None] | None = None,
) -> list[list[dict[str, Any]]]:
    """Batch ordered units without exceeding their count or decoded-byte cap.

    A unit larger than the byte cap is still scheduled by itself.  The same
    conservative treatment applies when the planner cannot establish its input
    size from durable metadata.
    """
    if max_units < 1 or max_bytes < 1:
        raise ValueError("Processing batch limits must be positive.")
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    current_frame_ids: set[str] = set()

    def flush() -> None:
        nonlocal current, current_bytes, current_frame_ids
        if current:
            batches.append(current)
        current = []
        current_bytes = 0
        current_frame_ids = set()

    for unit in units:
        value = unit.get("input_bytes")
        try:
            input_bytes = int(value)
            valid_size = not isinstance(value, bool) and input_bytes >= 0
        except (TypeError, ValueError, OverflowError):
            input_bytes = 0
            valid_size = False
        extra_bytes = 0 if extra_bytes_for_batch is None else extra_bytes_for_batch(current_frame_ids, unit)
        if extra_bytes is not None:
            try:
                extra_bytes = int(extra_bytes)
                valid_extra = extra_bytes >= 0
            except (TypeError, ValueError, OverflowError):
                valid_extra = False
        else:
            valid_extra = False
        if not valid_size or not valid_extra:
            flush()
            batches.append([unit])
            continue
        required_bytes = input_bytes + extra_bytes
        if required_bytes > max_bytes:
            flush()
            batches.append([unit])
            continue
        if current and (len(current) >= max_units or current_bytes + required_bytes > max_bytes):
            flush()
            extra_bytes = 0 if extra_bytes_for_batch is None else extra_bytes_for_batch(current_frame_ids, unit)
            if extra_bytes is None:
                batches.append([unit])
                continue
            try:
                extra_bytes = int(extra_bytes)
            except (TypeError, ValueError, OverflowError):
                batches.append([unit])
                continue
            if extra_bytes < 0 or input_bytes + extra_bytes > max_bytes:
                batches.append([unit])
                continue
        current.append(unit)
        current_bytes += input_bytes + extra_bytes
        if unit.get("frame_id"):
            current_frame_ids.add(str(unit["frame_id"]))
        if len(current) >= max_units:
            flush()
    flush()
    return batches


def _submission_page_batch(
    rows: list[dict[str, Any]], *, stage: str, options: dict[str, Any], max_units: int,
) -> list[dict[str, Any]]:
    """Select one durable planning page under count and decoded-input limits.

    Refinement budgets decoded ROI crops and masks; optional source-frame
    reads for expansion do not determine batch size.
    """
    if stage in {"preprocess_frames", "segment"}:
        max_bytes = (
            PREPROCESS_INPUT_BYTES_PER_JOB
            if stage == "preprocess_frames"
            else SEGMENT_INPUT_BYTES_PER_JOB
        )
        prepared = [{**row, "input_bytes": _frame_input_bytes(row)} for row in rows]
        batches = batch_processing_units_by_limit(prepared, max_units, max_bytes)
        return batches[0] if batches else []

    if stage != "roi_refinement":
        raise ValueError(f"Unsupported processing stage: {stage}.")

    prepared = [{**row, "input_bytes": _roi_input_bytes(row)} for row in rows]
    batches = batch_processing_units_by_limit(
        prepared, max_units, ROI_REFINEMENT_INPUT_BYTES_PER_JOB,
    )
    return batches[0] if batches else []


def _frame_input_bytes(row: dict[str, Any]) -> int | None:
    return decoded_array_input_bytes(row.get("payload_shape"), row.get("payload_dtype"))


def _roi_input_bytes(row: dict[str, Any]) -> int | None:
    """Decoded candidate crop/mask bytes, apart from an expansion frame."""
    has_roi = row.get("has_roi_payload")
    has_mask = row.get("has_mask_payload")
    if not isinstance(has_roi, bool) or not isinstance(has_mask, bool):
        return None
    total = 0
    if has_roi:
        roi_bytes = decoded_array_input_bytes(row.get("roi_shape"), row.get("roi_dtype"))
        if roi_bytes is None:
            return None
        total += roi_bytes
    if has_mask:
        mask_bytes = decoded_array_input_bytes(row.get("mask_shape"), row.get("mask_dtype"))
        if mask_bytes is None:
            return None
        total += mask_bytes
    else:
        # Refinement always works with a binary mask.  When it is not persisted
        # alongside the candidate, it is reconstructed from the crop geometry.
        # Binary masks are uint8, one byte per crop pixel.
        try:
            mask_width = int(row.get("crop_bbox_w") or row.get("bbox_w"))
            mask_height = int(row.get("crop_bbox_h") or row.get("bbox_h"))
        except (TypeError, ValueError):
            return None
        if mask_width <= 0 or mask_height <= 0:
            return None
        total += mask_width * mask_height
    return total


# Processing presets are a UI-facing flat document. Queue payloads are also
# flat but stage-specific, so resolve only the documented preset fields here
# rather than forwarding an opaque settings blob to workers.
_PRESET_OPTION_KEYS = {
    "preprocess_frames": {
        "min_field_value", "max_field_value", "apply_mask", "crop_enabled", "crop_x", "crop_y", "crop_w", "crop_h",
    },
    "segment": {
        "min_field_value", "max_field_value", "apply_mask", "crop_enabled", "crop_x", "crop_y", "crop_w", "crop_h",
        "frame_payload_kind", "apply_preprocessing",
        "threshold_method", "manual_threshold", "thresholding_maximum_value", "bounded_otsu_min_contrast",
        "bounded_otsu_max_foreground_fraction", "canny_enabled", "canny_low_threshold", "canny_high_threshold",
        "canny_blur_kernel", "adaptive_block_size", "adaptive_c", "percentile_background_percentile",
        "percentile_min_contrast", "hysteresis_low_threshold", "hysteresis_high_threshold", "hysteresis_connectivity",
        "sobel_percentile", "sobel_threshold", "sobel_kernel_size", "mask_augmentation_enabled",
        "mask_augmentation_steps", "dilate_kernel_w", "dilate_kernel_h", "dilate_iterations", "erode_kernel_w",
        "erode_kernel_h", "erode_iterations", "open_kernel_w", "open_kernel_h", "open_iterations", "close_kernel_w",
        "close_kernel_h", "close_iterations", "fill_holes", "remove_small_components", "min_component_area",
        "clear_border", "roi_assembly_method", "roi_assembly_connectivity", "min_area", "max_area", "min_perimeter",
        "max_perimeter", "min_width", "max_width", "min_height", "max_height", "min_width_plus_height",
        "max_width_plus_height", "padding", "store_roi_payload_min_area", "store_roi_payload_min_width",
        "store_roi_payload_min_height", "store_roi_payload_min_width_plus_height",
    },
    "roi_refinement": {
        "method", "model_ref", "max_iterations", "expansion_pixels", "edge_touch_margin", "encoding",
        "allow_frame_expansion", "store",
    },
}
_PRESET_KEY_ALIASES = {
    "refinement_model_kind": "method",
    "refinement_model_ref": "model_ref",
    "refinement_max_iterations": "max_iterations",
    "refinement_expansion_pixels": "expansion_pixels",
    "refinement_edge_touch_margin": "edge_touch_margin",
    "refinement_encoding": "encoding",
    "refinement_allow_frame_expansion": "allow_frame_expansion",
    "refinement_store": "store",
}

_UNORDERED_REQUEST_LIST_KEYS = frozenset({
    "asset_ids", "frame_ids", "collection", "collections", "preprocessing_status",
    "candidate_detection_status", "roi_refinement_status", "refinement_state",
})


def _canonical_submission_value(value: Any, *, key: str | None = None) -> Any:
    """Normalize set-like selectors before active-request fingerprinting."""
    if isinstance(value, dict):
        return {str(item_key): _canonical_submission_value(item_value, key=str(item_key))
                for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        normalized = [_canonical_submission_value(item) for item in value]
        if key in _UNORDERED_REQUEST_LIST_KEYS:
            return sorted({str(item) for item in normalized if item not in (None, "")})
        return normalized
    return value


def _snake_case(value: str) -> str:
    """Normalize the camelCase keys used by PelagiaView preset TOML files."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


@dataclass(frozen=True, slots=True)
class PreprocessQueueRequest:
    filters: dict[str, Any]
    options: dict[str, Any]
    priority: int | None = None
    dry_run: bool = False
    submitted_by_user_id: str | None = None
    submitted_by_username: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingSeriesRequest:
    steps: tuple[dict[str, Any], ...]
    selection: dict[str, Any] | None = None
    preset_snapshot: dict[str, Any] | None = None
    failure_policy: str = "fail_fast"
    priority: int | None = None
    dry_run: bool = False
    submitted_by_user_id: str | None = None
    submitted_by_username: str | None = None


class ProcessingSubmissionConflict(ValueError):
    """A caller reused an idempotency key for a different request."""


class ProcessingSubmissionService:
    """Durably accept criteria requests before expanding frames or detections.

    Planning intentionally happens in this dispatcher, after the HTTP response.
    The stored criteria are evaluated here, so browser navigation cannot cancel
    a request and a restarted API process can reclaim abandoned planning.
    """

    def __init__(self, context: AppContext):
        if context.repository is None:
            raise RuntimeError("Processing submissions require a PostgresRepository.")
        self.context = context
        self.repository = context.repository

    def submit(
        self,
        *,
        kind: str,
        request: dict[str, Any],
        request_id: str,
        project_id: str,
        submitted_by_user_id: str | None = None,
        submitted_by_username: str | None = None,
    ) -> dict[str, Any]:
        # Reject structural errors before creating a durable accepted row. The
        # potentially large criteria expansion remains entirely asynchronous.
        persisted_request = dict(request)
        if kind == "queue":
            stage = str(persisted_request.get("stage") or "")
            if stage not in {"preprocess_frames", "segment", "roi_refinement"}:
                raise ValueError(f"Unsupported processing stage: {stage}.")
            filters = dict(persisted_request.get("filters") or {})
            if stage == "preprocess_frames":
                filters["preprocessing_status"] = filters.get("preprocessing_status") or ["unknown", "failed", "cancelled"]
            elif stage == "segment":
                filters["candidate_detection_status"] = filters.get("candidate_detection_status") or ["unknown", "failed", "cancelled"]
            else:
                filters["refinement_state"] = filters.get("refinement_state") or ["unrefined"]
            persisted_request["filters"] = filters
        elif kind == "series":
            steps = tuple(dict(step) for step in persisted_request.get("steps") or ())
            selection = dict(persisted_request.get("selection") or persisted_request.get("targets") or {})
            ProcessingQueueService._validate_series_steps(steps)
            ProcessingQueueService._validate_series_selection(selection)
            if str(persisted_request.get("failure_policy") or "fail_fast") not in {"fail_fast", "continue"}:
                raise ValueError("failure_policy must be one of: fail_fast, continue.")
            normalized_steps = []
            for step in steps:
                stage = str(step["stage"])
                filters = dict(step.get("filters") or {})
                if stage == "preprocess_frames":
                    filters["preprocessing_status"] = filters.get("preprocessing_status") or ["unknown", "failed", "cancelled"]
                elif stage == "segment":
                    filters["candidate_detection_status"] = filters.get("candidate_detection_status") or ["unknown", "failed", "cancelled"]
                else:
                    filters["refinement_state"] = filters.get("refinement_state") or ["unrefined"]
                normalized_steps.append(ProcessingQueueService._resolve_series_step_options(
                    {**step, "filters": filters},
                    dict(persisted_request.get("preset_snapshot") or {}),
                ))
            persisted_request["steps"] = normalized_steps
            persisted_request["selection"] = selection
        else:
            raise ValueError("Processing submission kind must be 'queue' or 'series'.")
        persisted_request = _canonical_submission_value(persisted_request)
        try:
            submission, _created = self.repository.create_processing_submission(
                project_id=project_id, request_id=request_id, kind=kind, request=persisted_request,
                submitted_by_user_id=submitted_by_user_id,
                submitted_by_username=submitted_by_username,
            )
        except ValueError as exc:
            if "request_id was already used" in str(exc):
                raise ProcessingSubmissionConflict(str(exc)) from exc
            raise
        return submission

    def dispatch(self, *, limit: int = 1) -> int:
        """Run a bounded number of claims; safe in every API process."""
        reconcile = getattr(self.repository, "reconcile_processing_submissions", None)
        if reconcile is not None:
            reconcile(limit=100)
        release = getattr(self.repository, "release_terminal_processing_submission_reservations", None)
        if release is not None:
            release(limit=50_000)
        ready_series = getattr(self.repository, "list_ready_processing_submission_series", None)
        if ready_series is not None:
            for ready in ready_series(limit=25):
                self.advance_series_for_job(str(ready["job_id"]))
        completed = 0
        for _ in range(min(max(1, int(limit)), 25)):
            submission = self.repository.claim_processing_submission(
                lease_seconds=PROCESSING_SUBMISSION_CLAIM_SECONDS,
            )
            if submission is None:
                break
            self._dispatch_claimed(submission)
            completed += 1
        return completed

    def _dispatch_claimed(self, submission: dict[str, Any]) -> None:
        submission_id, claim_token = str(submission["id"]), str(submission["claim_token"])
        project_id = str(submission["project_id"])
        try:
            if submission["kind"] == "queue":
                self._dispatch_queue_page(submission)
                return
            if submission["kind"] == "series":
                self._dispatch_series_page(submission)
                return
            raise ValueError("Unsupported processing submission kind.")
        except Exception as exc:
            self.repository.fail_processing_submission(submission_id, claim_token=claim_token, error=str(exc))

    def _dispatch_queue_page(self, submission: dict[str, Any]) -> None:
        """Create and commit one worker-sized page, then yield to another request."""
        submission_id, claim_token = str(submission["id"]), str(submission["claim_token"])
        project_id = str(submission["project_id"])
        body = dict(submission.get("request") or {})
        stage = str(body["stage"])
        request = PreprocessQueueRequest(
            filters=dict(body.get("filters") or {}), options=dict(body.get("options") or {}),
            priority=body.get("priority"), dry_run=False,
            submitted_by_user_id=submission.get("submitted_by_user_id"),
            submitted_by_username=submission.get("submitted_by_username"),
        )
        page_size = INCREMENTAL_REFINEMENT_PAGE_SIZE if stage == "roi_refinement" else INCREMENTAL_FRAME_PAGE_SIZE
        with self.repository.processing_submission_transaction(
            submission_id, project_id=project_id, claim_token=claim_token,
        ) as connection:
            if connection is None:
                return
            page = self.repository.plan_processing_submission_page(
                submission_id=submission_id, project_id=project_id, stage=stage,
                filters=request.filters,
                payload_kind=str(
                    request.options.get(
                        "expansion_frame_payload_kind" if stage == "roi_refinement" else "frame_payload_kind",
                        "preprocessed" if stage == "roi_refinement" else "original",
                    )
                ),
                boundary=dict(submission.get("selection_boundary") or {}),
                planning_cursor=dict(submission.get("planning_cursor") or {}),
                limit=page_size, _connection=connection,
            )
            planned_rows = list(page["rows"])
            rows = _submission_page_batch(
                planned_rows, stage=stage, options=request.options, max_units=page_size,
            )
            if rows and len(rows) < len(planned_rows):
                page["has_more"] = True
                page["waiting_for_submission_id"] = None
                page["cursor"] = {
                    "created_at": rows[-1]["unit_created_at"],
                    "detection_id" if stage == "roi_refinement" else "frame_id": str(
                        rows[-1]["detection_id" if stage == "roi_refinement" else "frame_id"]
                    ),
                }
                if stage in {"preprocess_frames", "segment"} and page["boundary"].get("frame_order") == "asset_frame_v1":
                    page["cursor"].update({"asset_id": str(rows[-1]["asset_id"]), "frame_index": rows[-1]["frame_index"]})
                if stage == "roi_refinement" and page["boundary"].get("roi_order") == "generation_v1":
                    page["cursor"].update({
                        "asset_id": str(rows[-1]["asset_id"]), "frame_index": rows[-1]["frame_index"],
                        "frame_id": str(rows[-1]["frame_id"]), "roi_index": rows[-1]["roi_index"],
                    })
            job = None
            unit_ids: list[str] = []
            if rows:
                selected_ids = [str(row["detection_id"] if stage == "roi_refinement" else row["frame_id"]) for row in rows]
                eligible_ids = set(self.repository.lock_processing_submission_page(
                    project_id=project_id, stage=stage, unit_ids=selected_ids,
                    filters=request.filters, _connection=connection,
                ))
                rows = [row for row in rows if str(
                    row["detection_id"] if stage == "roi_refinement" else row["frame_id"]
                ) in eligible_ids]
            if rows:
                if stage == "roi_refinement":
                    unit_ids = [str(row["detection_id"]) for row in rows]
                    payload = RoiRefinementCommand(
                        detection_ids=tuple(unit_ids), options=dict(request.options),
                    ).to_payload()
                    summary = f"roi refinement queued for {len(unit_ids)} detections"
                else:
                    unit_ids = [str(row["frame_id"]) for row in rows]
                    command = PreprocessFramesCommand if stage == "preprocess_frames" else SegmentFramesCommand
                    payload = command(
                        selection=FrameSelection(frame_ids=tuple(unit_ids)), options=dict(request.options),
                    ).to_payload()
                    summary = f"{stage} queued for {len(unit_ids)} frames"
                assets = {str(row["asset_id"]) for row in rows}
                runs = {str(row["run_id"]) for row in rows if row.get("run_id")}
                cursor_hash = hashlib.sha256(
                    repr(sorted(page["cursor"].items())).encode("utf-8")
                ).hexdigest()[:24]
                job = self.repository.create_job(
                    stage, project_id=project_id,
                    run_id=next(iter(runs)) if len(runs) == 1 else None,
                    asset_id=next(iter(assets)) if len(assets) == 1 else None,
                    priority=request.priority, payload=payload, summary=summary,
                    submitted_by_user_id=request.submitted_by_user_id,
                    submitted_by_username=request.submitted_by_username,
                    idempotency_key=f"submission:{submission_id}:{stage}:{cursor_hash}",
                    _connection=connection,
                )
            previous_scheduled = int((submission.get("progress") or {}).get("scheduled_count") or 0)
            total_scheduled = previous_scheduled + len(unit_ids)
            previous_jobs = int((submission.get("progress") or {}).get("job_count") or 0)
            total_jobs = previous_jobs + (1 if job is not None else 0)
            result = None if page["has_more"] else {
                "stage": stage,
                "unit": "detections" if stage == "roi_refinement" else "frames",
                "matched_count": total_scheduled,
                "job_count": total_jobs,
                "max_units_per_job": page_size,
                "max_input_bytes_per_job": (
                    ROI_REFINEMENT_INPUT_BYTES_PER_JOB if stage == "roi_refinement"
                    else PREPROCESS_INPUT_BYTES_PER_JOB if stage == "preprocess_frames"
                    else SEGMENT_INPUT_BYTES_PER_JOB
                ),
            }
            checkpointed = self.repository.checkpoint_processing_submission(
                submission_id, claim_token=claim_token, project_id=project_id,
                boundary=page["boundary"], cursor=page["cursor"],
                scheduled_count=previous_scheduled, job_id=None if job is None else str(job["id"]),
                stage=stage, unit_ids=unit_ids, has_more=bool(page["has_more"]),
                result=result, waiting_for_submission_id=page.get("waiting_for_submission_id"),
                _connection=connection,
            )
            if checkpointed is None:
                raise RuntimeError("Processing submission claim was lost while checkpointing a page.")

    def _dispatch_series_page(self, submission: dict[str, Any]) -> None:
        """Plan one independently committed page of the currently eligible series step."""
        submission_id, claim_token = str(submission["id"]), str(submission["claim_token"])
        project_id = str(submission["project_id"])
        body = dict(submission.get("request") or {})
        with self.repository.processing_submission_transaction(
            submission_id, project_id=project_id, claim_token=claim_token,
        ) as connection:
            if connection is None:
                return
            result_snapshot = dict(submission.get("result") or {})
            series_ref = result_snapshot.get("series") if isinstance(result_snapshot.get("series"), dict) else {}
            series_id = str(series_ref.get("id") or "")
            if not series_id:
                steps = tuple(dict(step) for step in body.get("steps") or ())
                series = self.repository.create_processing_series(
                    project_id=project_id, steps=steps,
                    selection=dict(body.get("selection") or body.get("targets") or {}),
                    preset_snapshot=dict(body.get("preset_snapshot") or {}),
                    failure_policy=str(body.get("failure_policy") or "fail_fast"),
                    priority=body.get("priority"),
                    submitted_by_user_id=submission.get("submitted_by_user_id"),
                    submitted_by_username=submission.get("submitted_by_username"),
                    submission_id=submission_id, _connection=connection,
                )
                series_id = str(series["id"])
            series = self.repository.get_processing_series(
                series_id, project_id=project_id, _connection=connection,
            )
            if series is not None and str(series.get("status")) == "paused":
                checkpointed = self.repository.checkpoint_processing_submission(
                    submission_id, claim_token=claim_token, project_id=project_id,
                    boundary=dict(submission.get("selection_boundary") or {}),
                    cursor=dict(submission.get("planning_cursor") or {}),
                    scheduled_count=int((submission.get("progress") or {}).get("scheduled_count") or 0),
                    job_id=None, stage="segment", unit_ids=[], has_more=False,
                    result={"series": {"id": series_id, "status": "paused"}},
                    next_status_override="processing", _connection=connection,
                )
                if checkpointed is None:
                    raise RuntimeError("Processing submission claim was lost while pausing its series.")
                return
            steps = self.repository.list_processing_series_steps(
                series_id, project_id=project_id, _connection=connection,
            )
            step = next((item for item in steps if str(item.get("status")) == "planning"), None)
            if step is None:
                step = self.repository.claim_processing_series_step(
                    series_id, project_id=project_id, _connection=connection,
                )
            if step is None:
                series = self.repository.get_processing_series(series_id, project_id=project_id, _connection=connection)
                terminal = str((series or {}).get("status")) in {"succeeded", "failed", "cancelled"}
                checkpointed = self.repository.checkpoint_processing_submission(
                    submission_id, claim_token=claim_token, project_id=project_id,
                    boundary={}, cursor={}, scheduled_count=int((submission.get("progress") or {}).get("scheduled_count") or 0),
                    job_id=None, stage="segment", unit_ids=[], has_more=not terminal,
                    result={"series": series or {"id": series_id}},
                    next_status_override=str((series or {}).get("status")) if terminal else "planning",
                    _connection=connection,
                )
                if checkpointed is None:
                    raise RuntimeError("Processing submission claim was lost while finalizing its series.")
                return
            stage = str(step["stage"])
            filters = dict(step.get("filters") or {})
            options = dict(step.get("options") or {})
            page_size = INCREMENTAL_REFINEMENT_PAGE_SIZE if stage == "roi_refinement" else INCREMENTAL_FRAME_PAGE_SIZE
            page = self.repository.plan_processing_submission_page(
                submission_id=submission_id, project_id=project_id, stage=stage, filters=filters,
                payload_kind=str(
                    options.get(
                        "expansion_frame_payload_kind" if stage == "roi_refinement" else "frame_payload_kind",
                        "preprocessed" if stage == "roi_refinement" else "original",
                    )
                ),
                boundary=dict(step.get("selection_boundary") or {}),
                planning_cursor=dict(step.get("planning_cursor") or {}),
                limit=page_size, _connection=connection,
            )
            planned_rows = list(page["rows"])
            rows = _submission_page_batch(
                planned_rows, stage=stage, options=options, max_units=page_size,
            )
            if rows and len(rows) < len(planned_rows):
                page["has_more"] = True
                page["waiting_for_submission_id"] = None
                page["cursor"] = {
                    "created_at": rows[-1]["unit_created_at"],
                    "detection_id" if stage == "roi_refinement" else "frame_id": str(
                        rows[-1]["detection_id" if stage == "roi_refinement" else "frame_id"]
                    ),
                }
                if stage in {"preprocess_frames", "segment"} and page["boundary"].get("frame_order") == "asset_frame_v1":
                    page["cursor"].update({"asset_id": str(rows[-1]["asset_id"]), "frame_index": rows[-1]["frame_index"]})
                if stage == "roi_refinement" and page["boundary"].get("roi_order") == "generation_v1":
                    page["cursor"].update({
                        "asset_id": str(rows[-1]["asset_id"]), "frame_index": rows[-1]["frame_index"],
                        "frame_id": str(rows[-1]["frame_id"]), "roi_index": rows[-1]["roi_index"],
                    })
            job = None
            unit_ids: list[str] = []
            if rows:
                selected_ids = [str(row["detection_id"] if stage == "roi_refinement" else row["frame_id"]) for row in rows]
                eligible_ids = set(self.repository.lock_processing_submission_page(
                    project_id=project_id, stage=stage, unit_ids=selected_ids,
                    filters=filters, _connection=connection,
                ))
                rows = [row for row in rows if str(
                    row["detection_id"] if stage == "roi_refinement" else row["frame_id"]
                ) in eligible_ids]
            if rows:
                if stage == "roi_refinement":
                    unit_ids = [str(row["detection_id"]) for row in rows]
                    payload = RoiRefinementCommand(detection_ids=tuple(unit_ids), options=options).to_payload()
                    summary = f"roi refinement queued for {len(unit_ids)} detections"
                else:
                    unit_ids = [str(row["frame_id"]) for row in rows]
                    command = PreprocessFramesCommand if stage == "preprocess_frames" else SegmentFramesCommand
                    payload = command(selection=FrameSelection(frame_ids=tuple(unit_ids)), options=options).to_payload()
                    summary = f"{stage} queued for {len(unit_ids)} frames"
                assets = {str(row["asset_id"]) for row in rows}
                runs = {str(row["run_id"]) for row in rows if row.get("run_id")}
                cursor_hash = hashlib.sha256(repr(sorted(page["cursor"].items())).encode("utf-8")).hexdigest()[:24]
                job = self.repository.create_job(
                    stage, project_id=project_id,
                    run_id=next(iter(runs)) if len(runs) == 1 else None,
                    asset_id=next(iter(assets)) if len(assets) == 1 else None,
                    priority=step.get("series_priority"), payload=payload, summary=summary,
                    submitted_by_user_id=step.get("submitted_by_user_id"),
                    submitted_by_username=step.get("submitted_by_username"),
                    idempotency_key=f"submission:{submission_id}:step:{step['id']}:{cursor_hash}",
                    _connection=connection,
                )
            step_result = self.repository.checkpoint_processing_series_step(
                series_id=series_id, step_id=str(step["id"]), boundary=page["boundary"],
                planning_cursor=page["cursor"], job_id=None if job is None else str(job["id"]),
                matched_count=len(unit_ids), has_more=bool(page["has_more"]), _connection=connection,
            )
            final_step_status = str(step_result["step"]["status"])
            series_status = str(step_result["series"]["status"])
            if page["has_more"]:
                next_submission_status, more = "planning", True
            elif final_step_status == "active":
                next_submission_status, more = "processing", False
            elif series_status in {"succeeded", "failed", "cancelled"}:
                next_submission_status, more = series_status, False
            else:
                next_submission_status, more = "planning", True
            previous_scheduled = int((submission.get("progress") or {}).get("scheduled_count") or 0)
            checkpointed = self.repository.checkpoint_processing_submission(
                submission_id, claim_token=claim_token, project_id=project_id,
                boundary={"step_id": str(step["id"]), **dict(page["boundary"])},
                cursor={"step_id": str(step["id"]), **dict(page["cursor"])},
                scheduled_count=previous_scheduled, job_id=None if job is None else str(job["id"]),
                stage=stage, step_id=str(step["id"]), unit_ids=unit_ids, has_more=more,
                result={
                    "series": {"id": series_id, "status": series_status},
                    "batch_limits": {
                        "stage": stage,
                        "max_units_per_job": page_size,
                        "max_input_bytes_per_job": (
                            ROI_REFINEMENT_INPUT_BYTES_PER_JOB if stage == "roi_refinement"
                            else PREPROCESS_INPUT_BYTES_PER_JOB if stage == "preprocess_frames"
                            else SEGMENT_INPUT_BYTES_PER_JOB
                        ),
                    },
                },
                next_status_override=next_submission_status,
                waiting_for_submission_id=page.get("waiting_for_submission_id"),
                _connection=connection,
            )
            if checkpointed is None:
                raise RuntimeError("Processing submission claim was lost while checkpointing a series page.")


class ProcessingQueueService:
    """Resolve work units and create efficiently ordered processing jobs."""

    def __init__(self, context: AppContext, *, _connection=None):
        if context.repository is None:
            raise RuntimeError("Processing queue operations require a PostgresRepository.")
        self.context = context
        self.repository = context.repository
        self._connection = _connection

    def _repository_call(self, method_name: str, /, *args: Any, **kwargs: Any):
        """Keep all materialization operations on the dispatch transaction."""
        if self._connection is not None:
            kwargs["_connection"] = self._connection
        return getattr(self.repository, method_name)(*args, **kwargs)

    def queue_preprocess(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        filters = {**request.filters}
        filters["preprocessing_status"] = filters.get("preprocessing_status") or ["unknown", "failed", "cancelled"]
        frames = self._plan_by_assets("plan_preprocess_frames", project_id=project_id, filters=filters)
        frames.sort(key=lambda row: (str(row["asset_id"]), int(row.get("frame_index") or 0), str(row["frame_id"])))
        self._set_frame_input_bytes(frames)
        batches = batch_processing_units_by_limit(
            frames, PREPROCESS_FRAMES_PER_JOB, PREPROCESS_INPUT_BYTES_PER_JOB,
        )
        planned_jobs = [self._job_for_batch(batch, request.options) for batch in batches]
        result = {
            "stage": "preprocess_frames",
            "unit": "frames",
            "matched_count": len(frames),
            "job_count": len(planned_jobs),
            "ordering": "asset_frame",
            "max_units_per_job": PREPROCESS_FRAMES_PER_JOB,
            "max_input_bytes_per_job": PREPROCESS_INPUT_BYTES_PER_JOB,
            "batch_sizes": [len(batch) for batch in batches],
            "sample_frame_ids": [str(row["frame_id"]) for row in frames[:20]],
            "dry_run": request.dry_run,
        }
        if request.dry_run:
            return result
        created = self._repository_call(
            "create_preprocess_jobs",
            project_id=project_id,
            jobs=planned_jobs,
            eligible_statuses=filters["preprocessing_status"],
            priority=request.priority,
            submitted_by_user_id=request.submitted_by_user_id,
            submitted_by_username=request.submitted_by_username,
        )
        return {**result, "job_ids": [str(row["id"]) for row in created]}

    def queue_segment(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        payload_kind = str(request.options.get("frame_payload_kind", "original"))
        filters = {**request.filters}
        filters["candidate_detection_status"] = filters.get("candidate_detection_status") or ["unknown", "failed", "cancelled"]
        frames = self._plan_by_assets(
            "plan_segment_frames", project_id=project_id, filters=filters, payload_kind=payload_kind
        )
        return self._queue_frame_stage(
            "segment",
            frames,
            request,
            project_id=project_id,
            command=SegmentFramesCommand,
            max_units=SEGMENT_FRAMES_PER_JOB,
            max_input_bytes=SEGMENT_INPUT_BYTES_PER_JOB,
            ordering="asset_frame",
        )

    def queue_roi_refinement(self, request: PreprocessQueueRequest, *, project_id: str) -> dict[str, Any]:
        payload_kind = str(request.options.get("expansion_frame_payload_kind", "preprocessed"))
        detections = self._plan_by_assets(
            "plan_roi_refinement_detections", project_id=project_id, filters=request.filters,
            frame_payload_kind=payload_kind,
        )
        return self._queue_detection_stage(
            detections,
            request,
            project_id=project_id,
            max_units=ROI_REFINEMENT_DETECTIONS_PER_JOB,
            max_input_bytes=ROI_REFINEMENT_INPUT_BYTES_PER_JOB,
        )

    def create_series(self, request: ProcessingSeriesRequest, *, project_id: str) -> dict[str, Any]:
        """Persist a staged plan, then let the director materialize only its first step."""
        self._validate_series_steps(request.steps)
        self._validate_series_selection(request.selection)
        if request.failure_policy not in {"fail_fast", "continue"}:
            raise ValueError("failure_policy must be one of: fail_fast, continue.")
        steps = tuple(self._resolve_series_step_options(step, request.preset_snapshot) for step in request.steps)
        if request.dry_run:
            planned_steps = [self._plan_step(step, project_id=project_id, dry_run=True) for step in steps]
            return {
                "dry_run": True,
                "failure_policy": request.failure_policy,
                "steps": planned_steps,
                "eligibility": self._eligibility(planned_steps),
            }
        series = self._repository_call(
            "create_processing_series",
            project_id=project_id, steps=steps, selection=request.selection or {},
            preset_snapshot=request.preset_snapshot or {}, failure_policy=request.failure_policy,
            priority=request.priority, submitted_by_user_id=request.submitted_by_user_id,
            submitted_by_username=request.submitted_by_username,
        )
        advanced = self.advance_series(str(series["id"]), project_id=project_id)
        if self._connection is not None and advanced is not None:
            # The full series projection normally uses several read operations.
            # During submission materialization, return the newly-created row
            # plus the status established by this transaction and let later
            # GET requests build the expanded projection after commit.
            return {**series, **advanced}
        return advanced or series

    @staticmethod
    def _eligibility(planned_steps: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        total = 0
        for step in planned_steps:
            matched = int(step.get("matched_count") or 0)
            total += matched
            rows.append({
                "stage": step.get("stage"),
                "eligible_count": matched,
                "ineligible_count": 0,
                "job_count": int(step.get("job_count") or 0),
                "reasons": {},
            })
        return {"eligible_count": total, "ineligible_count": 0, "by_step": rows}

    def advance_series_for_job(self, job_id: str) -> dict[str, Any] | None:
        """Completion hook used by workers; it is safe to call more than once."""
        advanced = self.repository.advance_processing_series_for_job(job_id)
        if advanced is None or not advanced.get("ready"):
            return advanced
        series = self.repository.get_processing_series(str(advanced["series_id"]))
        if series is None:
            return advanced
        linked_submission = getattr(self.repository, "get_processing_submission_for_series", lambda _id: None)(
            str(advanced["series_id"])
        )
        if linked_submission is not None:
            self.repository.wake_processing_submission_for_series(str(advanced["series_id"]))
            return advanced
        if series.get("status") in {"failed", "cancelled", "paused"}:
            return advanced
        self.advance_series(str(advanced["series_id"]), project_id=str(series["project_id"]))
        return advanced

    def resume_series(
        self,
        series_id: str,
        *,
        project_id: str,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Resume a series and immediately dispatch a step made ready while paused.

        Repository control methods intentionally only mutate durable state.  The
        director follow-up belongs here so a final job that completed during a
        pause cannot leave its next queued step stranded until another job
        happens to finish.
        """
        series = self.repository.resume_processing_series(
            series_id, project_id=project_id, reason=reason,
        )
        if series is None:
            return None
        linked_submission = getattr(self.repository, "get_processing_submission_for_series", lambda _id: None)(series_id)
        if linked_submission is not None:
            self.repository.wake_processing_submission_for_series(series_id)
            return self.repository.get_processing_series(series_id, project_id=project_id)
        return self.advance_series(series_id, project_id=project_id)

    def reconcile_series(
        self,
        *,
        project_id: str,
        limit: int = 100,
        planning_timeout_seconds: int = PROCESSING_SERIES_PLANNING_TIMEOUT_SECONDS,
    ) -> dict[str, int]:
        """Recover dispatch that was interrupted after durable job completion.

        This routine is safe to run repeatedly from a scheduler.  It advances
        only queued/active series and delegates stale planning reclamation to
        the repository, where the timestamp predicate and status transition can
        be atomic.  It deliberately never advances paused or terminal series.

        The repository's ``reclaim_stale_processing_series_planning_steps``
        method is optional during the rolling upgrade; without it terminal-ready
        recovery remains available, while planning rows are left untouched.
        """
        if planning_timeout_seconds <= 0:
            raise ValueError("planning_timeout_seconds must be positive.")
        bounded_limit = min(max(1, int(limit)), 1_000)
        reclaimed = 0
        advanced = 0
        examined = 0

        reclaim = getattr(self.repository, "reclaim_stale_processing_series_planning_steps", None)
        if reclaim is not None:
            reclaimed_rows = reclaim(
                project_id=project_id,
                older_than_seconds=planning_timeout_seconds,
                limit=bounded_limit,
            )
            # A reclaimed row is queued again; advance_series atomically claims
            # it, so another reconciler/director may win without duplicating it.
            seen_reclaimed_series: set[str] = set()
            for row in reclaimed_rows:
                recovered_series_id = str(row["series_id"])
                if recovered_series_id in seen_reclaimed_series:
                    continue
                seen_reclaimed_series.add(recovered_series_id)
                reclaimed += 1
                self.advance_series(recovered_series_id, project_id=project_id)

        for series in self.repository.list_processing_series(
            project_id=project_id, status=["queued", "active"], limit=bounded_limit, offset=0,
        ):
            examined += 1
            series_id = str(series["id"])
            steps = list(series.get("steps") or [])
            active_steps = [step for step in steps if step.get("status") == "active"]
            if not active_steps:
                # This includes a newly-created queued series after a process
                # crash before its initial director call.
                if not any(step.get("status") == "queued" for step in steps):
                    # In particular, do not count or disturb a non-stale
                    # planning step owned by another director.
                    continue
                before = self.repository.get_processing_series(series_id, project_id=project_id)
                self.advance_series(series_id, project_id=project_id)
                if before is not None:
                    advanced += 1
                continue

            for step in active_steps:
                step_id = str(step["id"])
                units = [
                    unit for unit in self.repository.list_processing_work_units(series_id, project_id=project_id)
                    if str(unit.get("step_id")) == step_id
                ]
                # An active step without attached units is not safe to infer as
                # complete: it may be between job creation and attachment.
                if not units or any(str(unit.get("job_status")) not in _TERMINAL_JOB_STATUSES for unit in units):
                    continue
                self.advance_series_for_job(str(units[0]["job_id"]))
                advanced += 1

        return {"examined": examined, "reclaimed_planning": reclaimed, "advanced": advanced}

    def advance_series(self, series_id: str, *, project_id: str) -> dict[str, Any] | None:
        """Plan queued steps one at a time, recursively passing empty selections."""
        while True:
            step = self._repository_call("claim_processing_series_step", series_id, project_id=project_id)
            if step is None:
                if self._connection is not None:
                    # This transactional path is used only while creating a new
                    # series. Reaching the end means every empty step was
                    # recorded as skipped and attach marked the series done.
                    return {"id": series_id, "project_id": project_id, "status": "succeeded"}
                return self._repository_call("get_processing_series", series_id, project_id=project_id)
            result = self._plan_step(step, project_id=project_id, dry_run=False)
            self._repository_call(
                "attach_processing_work_units",
                series_id=series_id, step_id=str(step["id"]), job_ids=result.get("job_ids", []),
                matched_count=int(result["matched_count"]),
            )
            # An empty selection is explicitly recorded as skipped and the next
            # stage can be evaluated immediately.  Nonempty work awaits workers.
            if result.get("job_ids"):
                if self._connection is not None:
                    return {"id": series_id, "project_id": project_id, "status": "active"}
                return self._repository_call("get_processing_series", series_id, project_id=project_id)

    def _plan_step(self, step: dict[str, Any], *, project_id: str, dry_run: bool) -> dict[str, Any]:
        request = PreprocessQueueRequest(
            filters=dict(step.get("filters") or {}), options=dict(step.get("options") or {}),
            priority=step.get("priority", step.get("series_priority")), dry_run=dry_run,
            submitted_by_user_id=step.get("submitted_by_user_id"),
            submitted_by_username=step.get("submitted_by_username"),
        )
        stage = str(step["stage"])
        if stage == "preprocess_frames":
            return self.queue_preprocess(request, project_id=project_id)
        if stage == "segment":
            return self.queue_segment(request, project_id=project_id)
        if stage == "roi_refinement":
            return self.queue_roi_refinement(request, project_id=project_id)
        raise ValueError(f"Unsupported processing series stage: {stage}.")

    @staticmethod
    def _validate_series_steps(steps: tuple[dict[str, Any], ...]) -> None:
        if not steps:
            raise ValueError("A processing series requires at least one step.")
        canonical_order = ("preprocess_frames", "segment", "roi_refinement")
        allowed = set(canonical_order)
        invalid = [str(step.get("stage")) for step in steps if str(step.get("stage")) not in allowed]
        if invalid:
            raise ValueError(f"Unsupported processing series stage(s): {', '.join(invalid)}.")
        stages = tuple(str(step["stage"]) for step in steps)
        if len(set(stages)) != len(stages) or tuple(sorted(stages, key=canonical_order.index)) != stages:
            raise ValueError("Processing series stages must use the canonical Pelagia order.")

    @staticmethod
    def _validate_series_selection(selection: dict[str, Any] | None) -> None:
        """Series runs must name a bounded source scope; an empty filter is project-wide."""
        selection = selection or {}
        target_keys = ("asset_ids", "frame_ids", "collections", "collection")
        if not any(selection.get(key) for key in target_keys):
            raise ValueError("A processing series requires at least one asset, frame, or collection target.")

    @classmethod
    def _resolve_series_step_options(
        cls, step: dict[str, Any], preset_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge recognized preset settings into a stage payload without losing explicit overrides."""
        resolved = dict(step)
        stage = str(step["stage"])
        preset_options = cls._preset_options_for_stage(stage, preset_snapshot)
        # Explicit step options are an intentional API override of the selected preset.
        resolved["options"] = {**preset_options, **dict(step.get("options") or {})}
        return resolved

    @classmethod
    def _preset_options_for_stage(cls, stage: str, preset_snapshot: dict[str, Any] | None) -> dict[str, Any]:
        settings = (preset_snapshot or {}).get("settings") or {}
        if not isinstance(settings, dict):
            return {}
        options: dict[str, Any] = {}
        for key, value in settings.items():
            normalized = _PRESET_KEY_ALIASES.get(_snake_case(str(key)), _snake_case(str(key)))
            if normalized in _PRESET_OPTION_KEYS[stage] and value is not None:
                options[normalized] = value
        return options

    def _queue_frame_stage(
        self,
        stage: str,
        frames: list[dict[str, Any]],
        request: PreprocessQueueRequest,
        *,
        project_id: str,
        command,
        max_units: int,
        max_input_bytes: int,
        ordering: str,
    ) -> dict[str, Any]:
        frames.sort(key=lambda row: (str(row["asset_id"]), int(row.get("frame_index") or 0), str(row["frame_id"])))
        self._set_frame_input_bytes(frames)
        batches = batch_processing_units_by_limit(frames, max_units, max_input_bytes)
        result = {
            "stage": stage,
            "unit": "frames",
            "matched_count": len(frames),
            "job_count": len(batches),
            "ordering": ordering,
            "max_units_per_job": max_units,
            "max_input_bytes_per_job": max_input_bytes,
            "batch_sizes": [len(batch) for batch in batches],
            "dry_run": request.dry_run,
        }
        if request.dry_run:
            return result
        jobs = []
        for batch in batches:
            frame_ids = [str(row["frame_id"]) for row in batch]
            assets, runs = {str(row["asset_id"]) for row in batch}, {str(row["run_id"]) for row in batch if row.get("run_id")}
            payload = command(selection=FrameSelection(frame_ids=tuple(frame_ids)), options=dict(request.options)).to_payload()
            jobs.append(
                self._repository_call(
                    "create_job",
                    stage,
                    project_id=project_id,
                    run_id=next(iter(runs)) if len(runs) == 1 else None,
                    asset_id=next(iter(assets)) if len(assets) == 1 else None,
                    priority=request.priority,
                    payload=payload,
                    summary=f"{stage} queued for {len(frame_ids)} frames",
                    submitted_by_user_id=request.submitted_by_user_id,
                    submitted_by_username=request.submitted_by_username,
                )
            )
        return {**result, "job_ids": [str(job["id"]) for job in jobs]}

    def _queue_detection_stage(
        self,
        detections: list[dict[str, Any]],
        request: PreprocessQueueRequest,
        *,
        project_id: str,
        max_units: int,
        max_input_bytes: int,
    ) -> dict[str, Any]:
        detections.sort(key=lambda row: (str(row.get("unit_created_at") or ""), str(row["asset_id"]), int(row.get("frame_index") or 0), str(row["frame_id"]), int(row.get("roi_index") or 0), str(row["detection_id"])))
        for detection in detections:
            detection["input_bytes"] = _roi_input_bytes(detection)
        batches = batch_processing_units_by_limit(detections, max_units, max_input_bytes)
        result = {
            "stage": "roi_refinement",
            "unit": "detections",
            "matched_count": len(detections),
            "job_count": len(batches),
            "ordering": "candidate_generation",
            "max_units_per_job": max_units,
            "max_input_bytes_per_job": max_input_bytes,
            "batch_sizes": [len(batch) for batch in batches],
            "dry_run": request.dry_run,
        }
        if request.dry_run:
            return result
        jobs = []
        for batch in batches:
            detection_ids = [str(row["detection_id"]) for row in batch]
            assets, runs = {str(row["asset_id"]) for row in batch}, {str(row["run_id"]) for row in batch if row.get("run_id")}
            payload = RoiRefinementCommand(detection_ids=tuple(detection_ids), options=dict(request.options)).to_payload()
            jobs.append(
                self._repository_call(
                    "create_job",
                    "roi_refinement",
                    project_id=project_id,
                    run_id=next(iter(runs)) if len(runs) == 1 else None,
                    asset_id=next(iter(assets)) if len(assets) == 1 else None,
                    priority=request.priority,
                    payload=payload,
                    summary=f"roi refinement queued for {len(detection_ids)} detections",
                    submitted_by_user_id=request.submitted_by_user_id,
                    submitted_by_username=request.submitted_by_username,
                )
            )
        return {**result, "job_ids": [str(job["id"]) for job in jobs]}

    def _plan_by_assets(self, method_name: str, *, project_id: str, filters: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Apply a multi-asset filter through existing project-scoped planner queries."""
        asset_ids = [str(asset_id) for asset_id in filters.get("asset_ids") or [] if asset_id]
        asset_ids = list(dict.fromkeys(asset_ids))
        base_filters = {key: value for key, value in filters.items() if key != "asset_ids"}
        if not asset_ids:
            return self._repository_call(method_name, project_id=project_id, filters=base_filters, **kwargs)
        rows: list[dict[str, Any]] = []
        for asset_id in asset_ids:
            rows.extend(self._repository_call(
                method_name,
                project_id=project_id,
                filters={**base_filters, "asset_id": asset_id},
                **kwargs,
            ))
        return rows

    @staticmethod
    def _set_frame_input_bytes(frames: list[dict[str, Any]]) -> None:
        for frame in frames:
            frame["input_bytes"] = _frame_input_bytes(frame)

    @staticmethod
    def _job_for_batch(batch: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        frame_ids = [str(row["frame_id"]) for row in batch]
        asset_ids = {str(row["asset_id"]) for row in batch}
        run_ids = {str(row["run_id"]) for row in batch if row.get("run_id") is not None}
        payload = PreprocessFramesCommand(
            selection=FrameSelection(frame_ids=tuple(frame_ids)),
            options=dict(options),
        ).to_payload()
        asset_id = next(iter(asset_ids)) if len(asset_ids) == 1 else None
        run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
        return {
            "frame_ids": frame_ids,
            "asset_id": asset_id,
            "run_id": run_id,
            "payload": payload,
            "summary": f"preprocess queued for {len(frame_ids)} frames",
        }
