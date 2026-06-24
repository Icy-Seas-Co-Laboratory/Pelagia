from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from fastapi.testclient import TestClient

from Pelagia.api import create_app
from Pelagia.config import CoreConfig
from dataclasses import asdict

from Pelagia.domain import FrameRecord, PipelineStage
from Pelagia.processing import frame_store
from Pelagia.processing.frame_model import FrameData
from Pelagia.services.context import AppContext


class FakeRepository:
    schema = "pelagia"

    def __init__(self):
        self.created_jobs = []
        self.registered_runs = []
        self.shutdown_requests = []
        self.priority_updates = []
        self.cancel_job_calls = []
        self.delete_job_calls = []
        self.logs = []
        self.preprocessed_payload_ref = None
        self.sandbox_frames = {}
        self.deleted_sandbox_frames = []
        self.users = {
            "ada": {
                "id": "user-1",
                "username": "ada",
                "display_name": "Ada",
                "is_admin": False,
                "is_active": True,
                "password": "secret",
            },
            "admin": {
                "id": "user-admin",
                "username": "admin",
                "display_name": "Admin",
                "is_admin": True,
                "is_active": True,
                "password": "secret",
            },
        }
        self.projects = {
            "project-1": {
                "id": "project-1",
                "project_key": "default",
                "project_name": "Default",
                "is_active": True,
            },
            "project-2": {
                "id": "project-2",
                "project_key": "other",
                "project_name": "Other",
                "is_active": True,
            },
        }
        self.memberships = {("user-1", "project-1"): "editor"}
        self.sessions = {}
        self.resource_projects = {
            "run-1": "project-1",
            "asset-1": "project-1",
            "frame-1": "project-1",
            "frame-2": "project-1",
            "det-1": "project-1",
            "refined-det-1": "project-1",
            "det-wide": "project-1",
            "det-no-roi": "project-1",
            "job-1": "project-1",
        }

    def _visible(self, resource_id, project_id):
        return project_id is None or self.resource_projects.get(resource_id) == project_id

    def list_runs(self, **kwargs):
        if not self._visible("run-1", kwargs.get("project_id")):
            return []
        return [{"id": "run-1", **kwargs}]

    def get_run(self, run_id, **kwargs):
        if run_id != "run-1" or not self._visible(run_id, kwargs.get("project_id")):
            return None
        return {"id": run_id, "status": "queued", "job_summary": [], **kwargs}

    def list_assets(self, **kwargs):
        if not self._visible("asset-1", kwargs.get("project_id")):
            return []
        return [
            {
                "id": "asset-1",
                "run_id": kwargs.get("run_id"),
                "kind": "video",
                "collections": ["skq202510S-T1", "test"],
                **kwargs,
            }
        ]

    def get_asset(self, asset_id, **kwargs):
        if asset_id != "asset-1" or not self._visible(asset_id, kwargs.get("project_id")):
            return None
        return {"id": asset_id, "run_id": "run-1", "kind": "video", "collections": ["test"], **kwargs}

    def count_frames(self, asset_id, **kwargs):
        if asset_id != "asset-1":
            return 0
        return 12

    def list_frames(self, asset_id, **kwargs):
        if not self._visible(asset_id, kwargs.get("project_id")):
            return []
        return [{"id": "frame-1", "asset_id": asset_id, "frame_index": 2, "preview_thumbhash": b"abc", **kwargs}]

    def get_frame(self, frame_id, **kwargs):
        if frame_id in self.sandbox_frames:
            return dict(self.sandbox_frames[frame_id])
        if frame_id not in {"frame-1", "frame-2"} or not self._visible(frame_id, kwargs.get("project_id")):
            return None
        return {
            "id": frame_id,
            "asset_id": "asset-1",
            "frame_index": 2 if frame_id == "frame-1" else 3,
            "preview_thumbhash": b"abc",
            "preprocessed_payload_ref": self.preprocessed_payload_ref,
            "preprocessed_kvstore_hash": self.preprocessed_payload_ref,
            "preprocessed_preview_thumbhash": b"def" if self.preprocessed_payload_ref else None,
            "background_payload_ref": None,
            "background_kvstore_hash": None,
            "background_payload_encoding": None,
            "background_payload_format": None,
            "background_payload_dtype": None,
            "background_payload_shape": [],
            "background_metadata": {},
            "width": 10,
            "height": 10,
            "metadata": {"frame_id": frame_id},
        }

    def get_frame_by_asset_index(self, asset_id, frame_index, **kwargs):
        if asset_id != "asset-1" or frame_index != 2 or not self._visible(asset_id, kwargs.get("project_id")):
            return None
        return {"id": "frame-1", "asset_id": asset_id, "frame_index": frame_index}

    def get_frame_record(self, frame_id, **kwargs):
        if frame_id in self.sandbox_frames:
            return FrameRecord.from_row(self.sandbox_frames[frame_id])
        if frame_id not in {"frame-1", "frame-2"} or not self._visible(frame_id, kwargs.get("project_id")):
            return None
        return FrameRecord(
            id=frame_id,
            run_id="run-1",
            asset_id="asset-1",
            frame_index=2 if frame_id == "frame-1" else 3,
            width=10,
            height=10,
            kvstore_hash="fake-kv-key",
            preview_thumbhash=b"abc",
            preprocessed_payload_ref=self.preprocessed_payload_ref,
            preprocessed_kvstore_hash=self.preprocessed_payload_ref,
            background_payload_ref=None,
            background_kvstore_hash=None,
            metadata={"run_id": "run-1", "asset_id": "asset-1", "frame_id": frame_id},
        )

    def create_live_frame_copy(self, frame_id, *, operation, project_id=None, metadata=None):
        source = self.get_frame_record(frame_id, project_id=project_id)
        if source is None:
            raise KeyError(frame_id)
        sandbox_id = f"live-frame-{len(self.sandbox_frames) + 1}"
        live_metadata = {
            **dict(source.metadata or {}),
            "frame_id": sandbox_id,
            "live_preview": {
                "is_sandbox": True,
                "source_frame_id": frame_id,
                "operation": operation,
                **dict(metadata or {}),
            },
        }
        row = {
            "id": sandbox_id,
            "run_id": source.run_id,
            "asset_id": source.asset_id,
            "frame_index": -len(self.sandbox_frames) - 1,
            "captured_at": source.captured_at,
            "width": source.width,
            "height": source.height,
            "bbox_x": source.bbox_x,
            "bbox_y": source.bbox_y,
            "parent_frame_id": source.id,
            "source_ref": source.source_ref,
            "kvstore_hash": source.kvstore_hash,
            "preview_thumbhash": source.preview_thumbhash,
            "payload_ref": source.payload_ref or source.kvstore_hash,
            "payload_encoding": source.payload_encoding,
            "payload_format": source.payload_format,
            "payload_dtype": source.payload_dtype,
            "payload_shape": source.payload_shape,
            "metadata": live_metadata,
            "preprocessed_payload_ref": None,
            "preprocessed_kvstore_hash": None,
            "preprocessed_preview_thumbhash": None,
            "background_payload_ref": source.background_payload_ref,
            "background_kvstore_hash": source.background_kvstore_hash,
            "background_payload_encoding": source.background_payload_encoding,
            "background_payload_format": source.background_payload_format,
            "background_payload_dtype": source.background_payload_dtype,
            "background_payload_shape": list(source.background_payload_shape or []),
            "background_metadata": dict(source.background_metadata or {}),
        }
        self.sandbox_frames[sandbox_id] = row
        self.resource_projects[sandbox_id] = self.resource_projects.get(frame_id, "project-1")
        return dict(row)

    def update_frame_background_payloads(
        self,
        frame_ids,
        *,
        project_id=None,
        kvstore_hash,
        payload_ref,
        payload_encoding,
        payload_format,
        payload_dtype,
        payload_shape,
        metadata=None,
    ):
        rows = []
        for frame_id in frame_ids:
            if not self._visible(frame_id, project_id):
                raise KeyError(frame_id)
            row = self.sandbox_frames.get(frame_id)
            if row is None:
                row = self.get_frame(frame_id, project_id=project_id)
                if row is None:
                    raise KeyError(frame_id)
            row.update(
                {
                    "background_kvstore_hash": kvstore_hash,
                    "background_payload_ref": payload_ref,
                    "background_payload_encoding": payload_encoding,
                    "background_payload_format": payload_format,
                    "background_payload_dtype": payload_dtype,
                    "background_payload_shape": list(payload_shape or []),
                    "background_metadata": dict(metadata or {}),
                }
            )
            if frame_id in self.sandbox_frames:
                self.sandbox_frames[frame_id] = row
            rows.append(dict(row))
        return rows

    def count_frame_payload_references(self, payload_ref, *, exclude_frame_id=None):
        return 0

    def list_live_frame_copies(self, *, source_frame_id=None, operation=None, project_id=None, limit=100, offset=0):
        rows = []
        for row in self.sandbox_frames.values():
            if not self._visible(row["id"], project_id):
                continue
            live_preview = (row.get("metadata") or {}).get("live_preview") or {}
            if source_frame_id and live_preview.get("source_frame_id") != source_frame_id:
                continue
            if operation and live_preview.get("operation") != operation:
                continue
            rows.append(dict(row))
        return rows[max(0, int(offset)):max(0, int(offset)) + int(limit)]

    def delete_live_frame_copy(self, frame_id, *, project_id=None):
        if not self._visible(frame_id, project_id):
            return None
        row = self.sandbox_frames.pop(frame_id, None)
        if row is None:
            return None
        self.deleted_sandbox_frames.append(frame_id)
        self.resource_projects.pop(frame_id, None)
        keys = sorted(
            {
                key
                for key in (
                    row.get("preprocessed_payload_ref"),
                    row.get("preprocessed_kvstore_hash"),
                    row.get("background_payload_ref"),
                    row.get("background_kvstore_hash"),
                )
                if key
            }
        )
        return {
            "frame": row,
            "generated_kvstore_keys": keys,
            "unreferenced_kvstore_keys": keys,
        }

    def _detection_row(self, **overrides):
        row = {
            "id": "det-1",
            "run_id": "run-1",
            "asset_id": "asset-1",
            "asset_filename": "sample.mkv",
            "frame_id": "frame-1",
            "frame_index": 2,
            "roi_index": 1,
            "bbox_x": 3,
            "bbox_y": 4,
            "bbox_w": 5,
            "bbox_h": 6,
            "crop_bbox_x": 1,
            "crop_bbox_y": 2,
            "crop_bbox_w": 9,
            "crop_bbox_h": 10,
            "roi_payload": np.array([[0, 128], [255, 64]], dtype=np.uint8).tobytes(order="C"),
            "mask_payload": b"mask",
            "roi_encoding": "raw",
            "roi_format": "raw_ndarray_c_order",
            "roi_dtype": "uint8",
            "roi_shape": [2, 2],
            "mask_encoding": "raw",
            "mask_format": "raw_ndarray_c_order",
            "mask_dtype": "uint8",
            "mask_shape": [2, 2],
        }
        row.update(overrides)
        return row

    def list_detections(self, asset_id=None, **kwargs):
        return [
            self._detection_row(
                asset_id=asset_id or "asset-1",
                refined_detection_id="refined-det-1",
                **kwargs,
            )
        ]

    def get_detection(self, detection_id, **kwargs):
        if not self._visible(detection_id, kwargs.get("project_id")):
            return None
        if detection_id == "det-wide":
            return self._detection_row(
                id=detection_id,
                roi_payload=np.array(
                    [[0, 10, 20], [30, 40, 50]],
                    dtype=np.uint8,
                ).tobytes(order="C"),
                roi_shape=[2, 3],
                mask_payload=np.array(
                    [[0, 255, 0], [255, 0, 255]],
                    dtype=np.uint8,
                ).tobytes(order="C"),
                mask_shape=[2, 3],
            )
        if detection_id == "det-1":
            return self._detection_row()
        if detection_id == "det-no-roi":
            return self._detection_row(
                id=detection_id,
                roi_payload=None,
                crop_bbox_x=1,
                crop_bbox_y=2,
                crop_bbox_w=2,
                crop_bbox_h=2,
            )
        return None

    def get_refined_detection_for_candidate(self, detection_id, **kwargs):
        if detection_id != "det-1":
            return None
        return self.get_refined_detection("refined-det-1", **kwargs)

    def get_refined_detection(self, refined_detection_id, **kwargs):
        if not self._visible(refined_detection_id, kwargs.get("project_id")):
            return None
        if refined_detection_id != "refined-det-1":
            return None
        return self._detection_row(
            id=refined_detection_id,
            candidate_detection_id="det-1",
            roi_payload=np.array([[5, 6], [7, 8]], dtype=np.uint8).tobytes(order="C"),
            mask_payload=np.array([[0, 255], [255, 0]], dtype=np.uint8).tobytes(order="C"),
            metadata={"detection_stage": "refined"},
        )

    def upsert_refined_detections(self, refined_detections, *, job_id=None, project_id=None):
        rows = []
        for candidate_detection_id, detection in refined_detections:
            row = asdict(detection)
            row["id"] = f"refined-{candidate_detection_id}"
            row["candidate_detection_id"] = candidate_detection_id
            row["job_id"] = job_id
            row["asset_id"] = "asset-1"
            row["frame_index"] = 2
            row["asset_filename"] = "sample.mkv"
            rows.append(row)
        return rows

    def list_asset_detection_stats(self, **kwargs):
        return {
            "summary": {
                "total_asset_count": 2,
                "identified_asset_count": 1,
                "total_detection_count": 7,
            },
            "assets": [
                {
                    "asset_id": "asset-1",
                    "filename": "sample.mkv",
                    "detection_count": 7,
                    **kwargs,
                }
            ],
        }

    def list_asset_processing_state(self, **kwargs):
        return {
            "summary": {
                "total_asset_count": 2,
                "total_frame_count": 12,
                "total_preprocessed_frame_count": 7,
                "total_detected_frame_count": 3,
                "total_detection_count": 9,
            },
            "assets": [
                {
                    "asset_id": "asset-1",
                    "filename": "sample.mkv",
                    "run_id": "run-1",
                    "kind": kwargs.get("kind") or "video",
                    "collections": ["test"],
                    "frame_count": 12,
                    "preprocessed_frame_count": 7,
                    "detected_frame_count": 3,
                    "detection_count": 9,
                    **kwargs,
                    "preprocessing_state": "partially-preprocessed",
                    "detection_state": "partially-detected",
                }
            ],
        }

    def list_frame_processing_state(self, **kwargs):
        query_fields = {
            key: value
            for key, value in kwargs.items()
            if value is not None and key not in {"asset_id", "run_id"}
        }
        return {
            "summary": {
                "total_frame_count": 2,
                "total_preprocessed_frame_count": 1,
                "total_detected_frame_count": 1,
                "total_detection_count": 7,
                "total_refined_candidate_detection_count": 3,
                "total_unrefined_detection_count": 4,
                "total_refined_detection_count": 3,
            },
            "frames": [
                {
                    "frame_id": "frame-1",
                    "run_id": "run-1",
                    "asset_id": "asset-1",
                    "frame_index": 2,
                    "asset_filename": "sample.mkv",
                    "kind": kwargs.get("kind") or "video",
                    "collections": ["test"],
                    "has_preprocessed_payload": True,
                    "detection_count": 7,
                    "refined_candidate_detection_count": 3,
                    "unrefined_detection_count": 4,
                    "refined_detection_count": 3,
                    "preprocessing_state": "fully-preprocessed",
                    "detection_state": "fully-detected",
                    "refinement_state": "partially-refined",
                    **query_fields,
                },
                {
                    "frame_id": "frame-2",
                    "run_id": "run-1",
                    "asset_id": "asset-1",
                    "frame_index": 3,
                    "asset_filename": "sample.mkv",
                    "kind": kwargs.get("kind") or "video",
                    "collections": ["test"],
                    "has_preprocessed_payload": False,
                    "detection_count": 0,
                    "refined_candidate_detection_count": 0,
                    "unrefined_detection_count": 0,
                    "refined_detection_count": 0,
                    "preprocessing_state": "needs-preprocessed",
                    "detection_state": "needs-detections",
                    "refinement_state": "no-detections",
                    **query_fields,
                },
            ],
        }

    def replace_frame_detections(self, run_id, frame_ids, detections, **kwargs):
        project_id = kwargs.get("project_id")
        if project_id is not None and (
            not self._visible(run_id, project_id)
            or any(not self._visible(frame_id, project_id) for frame_id in frame_ids)
        ):
            raise KeyError("Frame was not found in project.")
        rows = []
        for index, detection in enumerate(detections, start=1):
            frame_id = getattr(detection, "frame_id", None)
            if frame_id is None and isinstance(detection, dict):
                frame_id = detection.get("frame_id")
            rows.append(
                self._detection_row(
                    id=f"det-{index}",
                    run_id=run_id,
                    frame_id=frame_id or frame_ids[0],
                    roi_payload=b"roi",
                    mask_payload=b"mask",
                )
            )
        return rows

    def list_models(self, **kwargs):
        return [{"id": "model-1", "model_key": "demo", **kwargs}]

    def list_collections(self, **kwargs):
        return [{"collection": kwargs.get("collection") or "test", "asset_count": 1, "limit": kwargs.get("limit")}]

    def get_model(self, model_id, **kwargs):
        return {"id": model_id, "model_key": "demo", **kwargs} if model_id == "model-1" else None

    def list_jobs(self, **kwargs):
        row = {"id": "job-1", "stage": PipelineStage.EXTRACT_FRAMES.value, **kwargs}
        if kwargs.get("include_details"):
            row["payload"] = {"frame_ids": ["frame-1"]}
            row["result"] = {"detection_ids": ["det-1"]}
        else:
            row["payload_bytes"] = 1024
            row["result_bytes"] = 2048
        return [row]

    def get_job(self, job_id, **kwargs):
        return {"id": job_id, "status": "queued", **kwargs} if job_id == "job-1" and self._visible(job_id, kwargs.get("project_id")) else None

    def create_job(self, stage, **kwargs):
        project_id = kwargs.get("project_id")
        if project_id is not None:
            if kwargs.get("run_id") and not self._visible(kwargs["run_id"], project_id):
                raise KeyError("Run was not found in project.")
            if kwargs.get("asset_id") and not self._visible(kwargs["asset_id"], project_id):
                raise KeyError("Asset was not found in project.")
            payload = kwargs.get("payload") or {}
            for frame_id in payload.get("frame_ids") or []:
                if not self._visible(frame_id, project_id):
                    raise KeyError("Frame was not found in project.")
            for detection_id in payload.get("detection_ids") or []:
                if not self._visible(detection_id, project_id):
                    raise KeyError("Detection was not found in project.")
        stage_value = stage.value if hasattr(stage, "value") else stage
        job = {"id": "job-new", "stage": stage_value, **kwargs}
        self.created_jobs.append(job)
        return job

    def list_job_events(self, **kwargs):
        return [{"id": 1, "event_type": "job.created", **kwargs}]

    def list_logs(self, **kwargs):
        return [{"id": 1, "event_type": "job.created", "level": kwargs.get("level") or "info", **kwargs}]

    def append_log(self, **kwargs):
        row = {"id": len(self.logs) + 1, **kwargs}
        self.logs.append(row)
        return row

    def pause_job(self, job_id, reason=None, **kwargs):
        if not self._visible(job_id, kwargs.get("project_id")):
            return None
        return {"id": job_id, "status": "paused", "reason": reason}

    def resume_job(self, job_id, reason=None, **kwargs):
        if not self._visible(job_id, kwargs.get("project_id")):
            return None
        return {"id": job_id, "status": "queued", "reason": reason}

    def retry_job(self, job_id, **kwargs):
        if not self._visible(job_id, kwargs.get("project_id")):
            return None
        return {"id": job_id, "status": "queued"}

    def set_job_priority(self, job_id, priority, reason=None, **kwargs):
        if not self._visible(job_id, kwargs.get("project_id")):
            return None
        self.priority_updates.append((job_id, priority, reason))
        return {"id": job_id, "priority": priority, "reason": reason}

    def cancel_jobs(self, **kwargs):
        self.cancel_job_calls.append(kwargs)
        if kwargs.get("project_id") == "project-2":
            return {
                "matched_count": 0,
                "cancellable_count": 0,
                "cancelled_count": 0,
                "dry_run": bool(kwargs.get("dry_run")),
                "jobs": [],
            }
        if kwargs.get("dry_run"):
            return {
                "matched_count": 1,
                "cancellable_count": 1,
                "cancelled_count": 0,
                "dry_run": True,
                "jobs": [],
            }
        return {
            "matched_count": 1,
            "cancellable_count": 1,
            "cancelled_count": 1,
            "dry_run": False,
            "jobs": [
                {
                    "id": "job-1",
                    "status": "cancelled",
                    "stage": PipelineStage.EXTRACT_FRAMES.value,
                    "project_id": kwargs.get("project_id"),
                    "control_reason": kwargs.get("reason"),
                }
            ],
        }

    def delete_jobs(self, **kwargs):
        self.delete_job_calls.append(kwargs)
        if kwargs.get("dry_run"):
            return {
                "matched_count": 1,
                "cancellable_count": 0,
                "cancelled_count": 0,
                "deleted_count": 0,
                "dry_run": True,
                "jobs": [],
            }
        return {
            "matched_count": 1,
            "cancellable_count": 0,
            "cancelled_count": 0,
            "deleted_count": 1,
            "dry_run": False,
            "jobs": [
                {
                    "id": "job-1",
                    "status": "queued",
                    "stage": PipelineStage.EXTRACT_FRAMES.value,
                    "project_id": kwargs.get("project_id"),
                }
            ],
        }

    def list_worker_sessions(self, **kwargs):
        return [{"worker_id": "extract-1", "status": kwargs.get("status") or "idle", **kwargs}]

    def get_worker_session(self, worker_id):
        return {"worker_id": worker_id, "status": "idle"} if worker_id == "extract-1" else None

    def request_worker_shutdown(self, worker_id, reason=None):
        self.shutdown_requests.append((worker_id, reason))
        return {"worker_id": worker_id, "shutdown_requested": True, "reason": reason}

    def get_status_summary(self, **kwargs):
        if kwargs.get("project_id") == "project-2":
            return {"queue": {"queued": 5}, "workers": {"total": 1, "online": 1, "busy": 0}}
        return {"queue": {"queued": 2}, "workers": {"total": 1, "online": 1, "busy": 0}}

    def register_planned_run(self, planned_run, **kwargs):
        project_id = kwargs.get("project_id")
        if project_id is not None:
            self.resource_projects[str(planned_run.manifest.run_id)] = project_id
            for asset in planned_run.manifest.assets:
                self.resource_projects[str(asset.asset_id)] = project_id
        self.registered_runs.append(planned_run)
        return {"run": {"id": planned_run.manifest.run_id}, "asset_count": 1, "job_count": 0}

    def verify_user_password(self, username, password):
        user = self.users.get(str(username).lower())
        if user and user["password"] == password and user["is_active"]:
            return dict(user)
        return None

    def get_user(self, user_id):
        for user in self.users.values():
            if user["id"] == user_id:
                return dict(user)
        return None

    def get_user_by_username(self, username):
        user = self.users.get(str(username).lower())
        return None if user is None else dict(user)

    def list_users(self, *, project_id=None, active_only=True, limit=100, offset=0):
        rows = []
        for user in self.users.values():
            if active_only and not user.get("is_active", True):
                continue
            row = dict(user)
            if project_id is not None:
                role = self.memberships.get((user["id"], project_id))
                if role is None:
                    continue
                row["project_id"] = project_id
                row["project_role"] = role
            rows.append(row)
        rows.sort(key=lambda item: item["username"])
        return rows[max(0, int(offset)):max(0, int(offset)) + int(limit)]

    def create_user(self, username, **kwargs):
        normalized = str(username).strip().lower()
        if normalized in self.users:
            raise ValueError("duplicate user")
        user = {
            "id": f"user-{len(self.users) + 1}",
            "username": normalized,
            "display_name": kwargs.get("display_name"),
            "is_admin": bool(kwargs.get("is_admin", False)),
            "is_active": bool(kwargs.get("is_active", True)),
            "password": kwargs.get("password"),
            "metadata": dict(kwargs.get("metadata") or {}),
        }
        self.users[normalized] = user
        return dict(user)

    def deactivate_user(self, user_id, *, metadata=None):
        for user in self.users.values():
            if user["id"] == user_id:
                user["is_active"] = False
                user["metadata"] = {**dict(user.get("metadata") or {}), **dict(metadata or {})}
                for session in self.sessions.values():
                    if session["user_id"] == user_id:
                        session["revoked_at"] = "now"
                return dict(user)
        return None

    def reset_user_password(self, user_id, password, *, metadata=None):
        for user in self.users.values():
            if user["id"] == user_id:
                user["password"] = password
                user["metadata"] = {**dict(user.get("metadata") or {}), **dict(metadata or {})}
                for session in self.sessions.values():
                    if session["user_id"] == user_id:
                        session["revoked_at"] = "now"
                return dict(user)
        return None

    def delete_user(self, user_id):
        for username, user in list(self.users.items()):
            if user["id"] == user_id:
                self.memberships = {
                    key: role
                    for key, role in self.memberships.items()
                    if key[0] != user_id
                }
                for session in self.sessions.values():
                    if session["user_id"] == user_id:
                        session["revoked_at"] = "now"
                return self.users.pop(username)
        return None

    def get_project(self, project_id):
        project = self.projects.get(project_id)
        return None if project is None else dict(project)

    def get_project_by_key(self, project_key):
        for project in self.projects.values():
            if project["project_key"] == str(project_key).lower():
                return dict(project)
        return None

    def create_project(self, project_key, **kwargs):
        normalized_key = str(project_key).strip().lower()
        project = {
            "id": f"project-{len(self.projects) + 1}",
            "project_key": normalized_key,
            "project_name": kwargs.get("project_name") or normalized_key,
            "description": kwargs.get("description"),
            "kvstore_root_path": kwargs.get("kvstore_root_path"),
            "is_active": bool(kwargs.get("is_active", True)),
            "metadata": dict(kwargs.get("metadata") or {}),
        }
        self.projects[project["id"]] = project
        return dict(project)

    def add_project_member(self, user_id, project_id, *, role="viewer", metadata=None):
        self.memberships[(user_id, project_id)] = role
        return {
            "user_id": user_id,
            "project_id": project_id,
            "role": role,
            "metadata": dict(metadata or {}),
        }

    def get_project_membership(self, user_id, project_id):
        role = self.memberships.get((user_id, project_id))
        if role is None:
            return None
        project = self.get_project(project_id)
        user = self.get_user(user_id)
        return {
            "user_id": user_id,
            "project_id": project_id,
            "role": role,
            "username": None if user is None else user["username"],
            "project_key": None if project is None else project["project_key"],
            "project_name": None if project is None else project["project_name"],
        }

    def deactivate_project(self, project_id, *, metadata=None):
        project = self.projects.get(project_id)
        if project is None:
            return None
        project["is_active"] = False
        project["metadata"] = {**dict(project.get("metadata") or {}), **dict(metadata or {})}
        return dict(project)

    def list_user_projects(self, user_id, **kwargs):
        return [
            {**self.projects[project_id], "membership_role": role}
            for (member_user_id, project_id), role in self.memberships.items()
            if member_user_id == user_id
        ]

    def list_projects(self, **kwargs):
        active_only = kwargs.get("active_only", True)
        return [
            dict(project)
            for project in self.projects.values()
            if not active_only or project.get("is_active")
        ]

    def create_session(self, user_id, project_id, **kwargs):
        user = self.get_user(user_id)
        if user is None:
            raise ValueError("missing user")
        if not user.get("is_admin") and (user_id, project_id) not in self.memberships:
            raise PermissionError("User is not a member of the requested project.")
        token = f"token-{len(self.sessions) + 1}"
        session = {
            "id": f"session-{len(self.sessions) + 1}",
            "user_id": user_id,
            "project_id": project_id,
            "token_hash": f"hash-{token}",
            "user_agent": kwargs.get("user_agent"),
            "remote_addr": kwargs.get("remote_addr"),
            "ttl_seconds": kwargs.get("ttl_seconds"),
            "expires_at": "2099-01-01T00:00:00Z",
            "revoked_at": None,
        }
        self.sessions[token] = session
        return {"token": token, "session": session}

    def get_session(self, token, **kwargs):
        session = self.sessions.get(token)
        if session is None or session.get("revoked_at"):
            return None
        user = self.get_user(session["user_id"])
        project = self.get_project(session["project_id"])
        if project is None or not project.get("is_active", True):
            return None
        role = self.memberships.get((session["user_id"], session["project_id"]))
        if user and user.get("is_admin") and role is None:
            role = "admin"
        return {
            **session,
            "username": user["username"],
            "display_name": user["display_name"],
            "is_admin": user["is_admin"],
            "project_key": project["project_key"],
            "project_name": project["project_name"],
            "project_role": role,
        }

    def revoke_session(self, token):
        session = self.sessions.get(token)
        if session is None:
            return None
        session["revoked_at"] = "now"
        return dict(session)

    def connect(self):
        raise AssertionError("API unit tests should not open a database connection.")


class FakeKVStore:
    initialized = True

    def __init__(self, root_path="/tmp/pelagia-kv", total_stored_blobs=3):
        self.deleted_keys = []
        self.root_path = root_path
        self.total_stored_blobs = total_stored_blobs

    def status(self):
        return {
            "root_path": self.root_path,
            "initialized": True,
            "total_stored_blobs": self.total_stored_blobs,
        }

    def check_health(self):
        return {"healthy": True, "errors": [], "warnings": []}

    def key_delete(self, key):
        self.deleted_keys.append(key)


def make_client(*, auth_enabled=False):
    config = CoreConfig()
    config.auth.enabled = auth_enabled
    app = create_app(config)
    repository = FakeRepository()
    kvstore = FakeKVStore()
    app.state.config = config
    app.state.context = AppContext(config=config, repository=repository, kvstore=kvstore)
    return TestClient(app), repository, kvstore


def auth_headers(client, *, username="ada", project_key="default"):
    response = client.post(
        "/auth/login",
        json={"username": username, "password": "secret", "project_key": project_key},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_api_lists_system_status_without_live_database():
    client, _, _ = make_client()

    response = client.get("/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["queue"] == {"queued": 2}
    assert body["workers"]["online"] == 1
    assert body["kvstore"]["initialized"] is True


def test_api_lists_project_system_status_with_project_kvstore():
    client, repo, _ = make_client(auth_enabled=True)
    client.app.state.context._project_kvstores["project-2"] = FakeKVStore(
        root_path="/tmp/pelagia-kv/projects/project-2",
        total_stored_blobs=8,
    )
    repo.memberships[("user-1", "project-2")] = "viewer"
    headers = auth_headers(client, username="ada", project_key="default")

    unauthenticated = client.get("/system/status/other")
    response = client.get("/system/status/other", headers=headers)

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    body = response.json()
    assert body["project"]["project_key"] == "other"
    assert body["kvstore"]["root_path"] == "/tmp/pelagia-kv/projects/project-2"
    assert body["kvstore"]["total_stored_blobs"] == 8
    assert body["queue"] == {"queued": 5}
    assert body["workers"]["online"] == 1


def test_api_project_system_status_requires_project_membership_or_admin():
    client, _, _ = make_client(auth_enabled=True)
    client.app.state.context._project_kvstores["project-2"] = FakeKVStore(
        root_path="/tmp/pelagia-kv/projects/project-2",
        total_stored_blobs=8,
    )
    user_headers = auth_headers(client, username="ada", project_key="default")
    admin_headers = auth_headers(client, username="admin", project_key="default")

    denied = client.get("/system/status/other", headers=user_headers)
    admin_response = client.get("/system/status/other", headers=admin_headers)
    missing = client.get("/system/status/missing-project", headers=admin_headers)

    assert denied.status_code == 403
    assert admin_response.status_code == 200
    assert admin_response.json()["project"]["project_key"] == "other"
    assert missing.status_code == 404


def test_api_project_system_status_does_not_initialize_missing_project_kvstore():
    client, repo, _ = make_client(auth_enabled=True)
    repo.projects["project-2"]["kvstore_root_path"] = "/storage/cruise-1"
    repo.memberships[("user-1", "project-2")] = "viewer"
    headers = auth_headers(client, username="ada", project_key="default")

    response = client.get("/system/status/other", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["project"]["project_key"] == "other"
    assert body["kvstore"]["root_path"] == "/storage/cruise-1"
    assert body["kvstore"]["initialized"] is False
    assert body["queue"] == {"queued": 5}


def test_api_exposes_system_config():
    client, _, _ = make_client()

    response = client.get("/system/config")

    assert response.status_code == 200
    body = response.json()
    assert body["effective"]["database"]["schema_name"] == "pelagia"
    assert body["effective"]["processing"]["preprocessing"]["apply_mask"] is False
    assert "mask_augmentation" in body["effective"]["processing"]
    assert "roi_assembly" in body["effective"]["processing"]
    assert "roi_filter" in body["effective"]["processing"]
    assert "roi_recording" in body["effective"]["processing"]
    assert body["effective"]["processing"]["roi_recording"]["roi_encoding"] == "zstd"
    assert body["effective"]["kvstore"]["root_path"]
    assert body["defaults"]["processing"]["video_ingest"]["n_tile"] == 4
    assert body["defaults"]["processing"]["preprocessing"]["mask_path"] is None


def test_api_exposes_system_capabilities():
    client, _, _ = make_client()

    response = client.get("/system/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Pelagia"
    assert body["api"]["endpoints"]["segmentation_options"] == "/segmentation/options"
    assert body["api"]["endpoints"]["preprocessing_options"] == "/preprocessing/options"
    assert body["api"]["endpoints"]["live_threshold"] == "/live/threshold"
    assert body["api"]["endpoints"]["live_detection_candidate"] == "/live/detection-candidate"
    assert body["api"]["endpoints"]["live_sandbox"] == "/live/sandbox"
    assert body["api"]["endpoints"]["delete_live_sandbox"] == "/live/sandbox/{sandbox_frame_id}"
    assert "live_segmentation" not in body["api"]["endpoints"]
    assert "extract_frames" in body["supported"]["pipeline_stages"]
    assert "background_frames" in body["jobs"]["queueable_stages"]
    assert "preprocess_frames" in body["jobs"]["queueable_stages"]
    assert "roi_refinement" in body["jobs"]["queueable_stages"]
    assert body["api"]["endpoints"]["generate_background"] == "/frame/background"
    assert body["api"]["endpoints"]["queue_background"] == "/frame/background/jobs"
    assert "jpg" in body["supported"]["image_encodings"]
    assert "preprocessed" in body["supported"]["frame_payload_kinds"]
    assert "segmentation" in body["processing"]
    assert "preprocessing" in body["processing"]
    assert "background" in body["processing"]["groups"]
    assert "roi_refinement" in body["processing"]
    assert "mask_augmentation" in body["processing"]["groups"]
    assert "roi_refinement" in body["processing"]["groups"]
    assert "builtin:model/roi_refinement/example_model" in body["processing"]["roi_refinement"]["supported"]["model_refs"]
    assert body["storage"]["kvstore"]["hash_algorithm_options"] == ["sha256", "blake3"]


def test_api_auth_login_me_projects_and_logout():
    client, repo, _ = make_client()

    bad = client.post("/auth/login", json={"username": "ada", "password": "wrong"})
    assert bad.status_code == 401

    login = client.post(
        "/auth/login",
        json={"username": "ada", "password": "secret", "project_key": "default"},
    )

    assert login.status_code == 200
    body = login.json()
    token = body["token"]
    assert token
    assert body["user"]["username"] == "ada"
    assert body["project"]["project_key"] == "default"
    assert body["project"]["role"] == "editor"

    headers = {"Authorization": f"Bearer {token}"}
    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["auth"]["project_id"] == "project-1"

    projects = client.get("/projects", headers=headers)
    assert projects.status_code == 200
    assert [project["project_key"] for project in projects.json()["projects"]] == ["default"]

    project_names = client.get("/projects?include_all_names=true", headers=headers)
    assert project_names.status_code == 200
    assert [project["project_key"] for project in project_names.json()["projects"]] == ["default"]
    assert project_names.json()["all_project_names"] == ["Default", "Other"]

    runs = client.get("/runs", headers=headers)
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["project_id"] == "project-1"

    assert token in repo.sessions
    logout = client.post("/auth/logout", headers=headers)
    assert logout.status_code == 200
    assert logout.json()["revoked"] is True
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_api_admin_can_create_project_and_non_admin_cannot():
    client, repo, _ = make_client()
    user_headers = auth_headers(client, username="ada", project_key="default")
    admin_headers = auth_headers(client, username="admin", project_key="default")

    denied = client.post(
        "/projects",
        headers=user_headers,
        json={"project_key": "survey", "project_name": "Survey"},
    )
    created = client.post(
        "/projects",
        headers=admin_headers,
        json={
            "project_key": "survey",
            "project_name": "Survey",
            "description": "Field survey data.",
            "kvstore_root_path": "/tmp/pelagia/survey",
        },
    )
    duplicate = client.post(
        "/projects",
        headers=admin_headers,
        json={"project_key": "survey"},
    )

    assert denied.status_code == 403
    assert created.status_code == 200
    body = created.json()
    assert body["project"]["project_key"] == "survey"
    assert body["project"]["project_name"] == "Survey"
    assert body["project"]["kvstore_root_path"] == "/tmp/pelagia/survey"
    assert body["membership"]["role"] == "admin"
    assert repo.memberships[("user-admin", body["project"]["id"])] == "admin"
    assert duplicate.status_code == 409


def test_api_project_delete_is_soft_delete_and_requires_manager():
    client, repo, _ = make_client()
    repo.memberships[("user-1", "project-2")] = "editor"
    admin_headers = auth_headers(client, username="admin", project_key="default")
    editor_headers = auth_headers(client, username="ada", project_key="other")

    denied = client.delete("/projects/other", headers=editor_headers)
    assert denied.status_code == 403
    assert repo.projects["project-2"]["is_active"] is True

    repo.memberships[("user-1", "project-2")] = "manager"
    manager_headers = auth_headers(client, username="ada", project_key="other")
    deleted = client.delete("/projects/other", headers=manager_headers)
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert deleted.json()["project"]["is_active"] is False
    assert repo.projects["project-2"]["is_active"] is False
    assert repo.projects["project-2"]["metadata"]["deleted_by_user_id"] == "user-1"

    default_delete = client.delete("/projects/default", headers=admin_headers)
    assert default_delete.status_code == 422


def test_api_logged_in_users_can_list_active_project_users():
    client, repo, _ = make_client(auth_enabled=True)
    repo.users["ben"] = {
        "id": "user-ben",
        "username": "ben",
        "display_name": "Ben",
        "is_admin": False,
        "is_active": True,
        "password": "secret",
        "metadata": {"lab": "A"},
    }
    repo.users["inactive"] = {
        "id": "user-inactive",
        "username": "inactive",
        "display_name": "Inactive",
        "is_admin": False,
        "is_active": False,
        "password": "secret",
    }
    repo.users["other"] = {
        "id": "user-other",
        "username": "other",
        "display_name": "Other",
        "is_admin": False,
        "is_active": True,
        "password": "secret",
    }
    repo.memberships[("user-ben", "project-1")] = "viewer"
    repo.memberships[("user-inactive", "project-1")] = "viewer"
    repo.memberships[("user-other", "project-2")] = "editor"
    headers = auth_headers(client, username="ada", project_key="default")

    unauthenticated = client.get("/users")
    response = client.get("/users", headers=headers)
    inactive_response = client.get("/users?active_only=false", headers=headers)

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    users = response.json()["users"]
    assert [user["username"] for user in users] == ["ada", "ben"]
    assert users[0]["project_role"] == "editor"
    assert users[1]["project_role"] == "viewer"
    assert all("password" not in user and "password_hash" not in user for user in users)
    assert [user["username"] for user in inactive_response.json()["users"]] == ["ada", "ben", "inactive"]


def test_api_user_admin_can_request_global_user_list():
    client, repo, _ = make_client()
    repo.users["ben"] = {
        "id": "user-ben",
        "username": "ben",
        "display_name": "Ben",
        "is_admin": False,
        "is_active": True,
        "password": "secret",
    }
    repo.users["other"] = {
        "id": "user-other",
        "username": "other",
        "display_name": "Other",
        "is_admin": False,
        "is_active": True,
        "password": "secret",
    }
    repo.memberships[("user-ben", "project-1")] = "viewer"
    repo.memberships[("user-other", "project-2")] = "editor"
    user_headers = auth_headers(client, username="ada", project_key="default")
    admin_headers = auth_headers(client, username="admin", project_key="default")

    denied = client.get("/users?include_all_projects=true", headers=user_headers)
    global_response = client.get("/users?include_all_projects=true", headers=admin_headers)

    assert denied.status_code == 403
    assert global_response.status_code == 200
    assert [user["username"] for user in global_response.json()["users"]] == ["ada", "admin", "ben", "other"]
    assert all("project_role" not in user for user in global_response.json()["users"])


def test_api_admin_can_create_user_and_reset_password():
    client, repo, _ = make_client()
    admin_headers = auth_headers(client, username="admin", project_key="default")

    created = client.post(
        "/users",
        headers=admin_headers,
        json={
            "username": "Grace",
            "password": "initial",
            "display_name": "Grace Hopper",
            "project_key": "default",
            "role": "manager",
        },
    )
    duplicate = client.post(
        "/users",
        headers=admin_headers,
        json={"username": "Grace", "project_key": "default"},
    )

    assert created.status_code == 200
    body = created.json()
    assert body["user"]["username"] == "grace"
    assert "password" not in body["user"]
    assert body["membership"]["role"] == "manager"
    assert repo.memberships[(body["user"]["id"], "project-1")] == "manager"
    assert duplicate.status_code == 409

    reset = client.post(
        "/users/grace/reset-password",
        headers=admin_headers,
        json={"password": "changed"},
    )
    assert reset.status_code == 200
    assert repo.users["grace"]["password"] == "changed"
    relogin = client.post(
        "/auth/login",
        json={"username": "grace", "password": "changed", "project_key": "default"},
    )
    assert relogin.status_code == 200


def test_api_project_manager_can_manage_users_in_active_project_only():
    client, repo, _ = make_client()
    repo.memberships[("user-1", "project-1")] = "manager"
    manager_headers = auth_headers(client, username="ada", project_key="default")

    created = client.post(
        "/users",
        headers=manager_headers,
        json={
            "username": "Ben",
            "password": "secret",
            "project_key": "default",
            "role": "editor",
        },
    )
    assert created.status_code == 200
    user_id = created.json()["user"]["id"]
    assert repo.memberships[(user_id, "project-1")] == "editor"

    make_admin = client.post(
        "/users",
        headers=manager_headers,
        json={"username": "Root", "is_admin": True, "project_key": "default"},
    )
    other_project = client.post(
        "/users",
        headers=manager_headers,
        json={"username": "OtherUser", "project_key": "other"},
    )
    manager_role = client.post(
        "/users",
        headers=manager_headers,
        json={"username": "Lead", "project_key": "default", "role": "manager"},
    )
    assert make_admin.status_code == 403
    assert other_project.status_code == 403
    assert manager_role.status_code == 403

    user_headers = auth_headers(client, username="ben", project_key="default")
    reset = client.post(
        "/users/ben/reset-password",
        headers=manager_headers,
        json={"password": "changed"},
    )
    assert reset.status_code == 200
    assert client.get("/auth/me", headers=user_headers).status_code == 401
    relogin = client.post(
        "/auth/login",
        json={"username": "ben", "password": "changed", "project_key": "default"},
    )
    assert relogin.status_code == 200

    deactivate = client.post("/users/ben/deactivate", headers=manager_headers)
    assert deactivate.status_code == 200
    assert repo.users["ben"]["is_active"] is False
    assert repo.users["ben"]["metadata"]["deactivated_by_user_id"] == "user-1"
    failed_login = client.post(
        "/auth/login",
        json={"username": "ben", "password": "changed", "project_key": "default"},
    )
    assert failed_login.status_code == 401

    repo.users["ben"]["is_active"] = True
    delete = client.delete("/users/ben", headers=manager_headers)
    assert delete.status_code == 200
    assert "ben" not in repo.users


def test_api_user_management_requires_manager_and_protects_admin_accounts():
    client, repo, _ = make_client()
    repo.memberships[("user-1", "project-1")] = "editor"
    repo.users["ben"] = {
        "id": "user-ben",
        "username": "ben",
        "display_name": "Ben",
        "is_admin": False,
        "is_active": True,
        "password": "secret",
    }
    repo.memberships[("user-ben", "project-2")] = "editor"
    editor_headers = auth_headers(client, username="ada", project_key="default")

    denied_create = client.post(
        "/users",
        headers=editor_headers,
        json={"username": "Nope", "project_key": "default"},
    )
    denied_reset = client.post(
        "/users/ben/reset-password",
        headers=editor_headers,
        json={"password": "changed"},
    )
    assert denied_create.status_code == 403
    assert denied_reset.status_code == 403

    repo.memberships[("user-1", "project-1")] = "manager"
    manager_headers = auth_headers(client, username="ada", project_key="default")
    out_of_project = client.post(
        "/users/ben/reset-password",
        headers=manager_headers,
        json={"password": "changed"},
    )
    admin_reset = client.post(
        "/users/admin/reset-password",
        headers=manager_headers,
        json={"password": "changed"},
    )
    self_delete = client.delete("/users/ada", headers=manager_headers)

    assert out_of_project.status_code == 404
    assert admin_reset.status_code == 403
    assert self_delete.status_code == 422


def test_api_auth_login_clamps_requested_session_ttl():
    client, repo, _ = make_client(auth_enabled=True)

    response = client.post(
        "/auth/login",
        json={
            "username": "ada",
            "password": "secret",
            "project_key": "default",
            "ttl_seconds": 999999999,
        },
    )

    assert response.status_code == 200
    assert repo.sessions[response.json()["token"]]["ttl_seconds"] == client.app.state.config.auth.session_ttl_seconds


def test_api_roi_refinement_options_are_ui_ready():
    client, _, _ = make_client()

    response = client.get("/roi-refinement/options")

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_stage_order"] == [
        "source",
        "model_selection",
        "tiling",
        "prediction",
        "expansion",
        "residual_discovery",
        "reconciliation",
        "recording",
    ]
    assert "identity" in body["supported"]["model_kinds"]
    assert "keras_artifact" in body["supported"]["model_kinds"]
    assert "builtin:model/roi_refinement/example_model" in body["supported"]["model_refs"]
    assert body["defaults"]["roi_refinement"]["model_ref"] == "builtin:model/roi_refinement/example_model"
    fields = {field["key"]: field for field in body["fields"]["tiling"]}
    assert fields["batch_size"]["type"] == "nullable-integer"
    assert fields["tile_size"]["min"] == 1
    reconciliation_fields = {field["key"]: field for field in body["fields"]["reconciliation"]}
    assert reconciliation_fields["overlap_iou_threshold"]["max"] == 1
    assert body["defaults"]["roi_refinement"]["overlap_reconciliation_enabled"] is True
    residual_fields = {field["key"]: field for field in body["fields"]["residual_discovery"]}
    assert residual_fields["residual_roi_assembly_connectivity"]["options"] == [4, 8]
    assert body["defaults"]["roi_refinement"]["residual_discovery_enabled"] is False


def test_api_roi_refinement_get_describes_post_contract():
    client, _, _ = make_client()

    response = client.get("/roi-refinement")

    assert response.status_code == 200
    body = response.json()
    assert body["endpoint"] == "/roi-refinement"
    assert body["methods"]["POST"]
    assert body["options_url"] == "/roi-refinement/options"
    assert body["jobs_url"] == "/roi-refinement/jobs"
    assert "detection_ids" in body["required_payload"]


def test_api_roi_refinement_dry_run_resolves_builtin_model_ref():
    client, _, _ = make_client()

    response = client.post(
        "/roi-refinement",
        json={
            "detection_ids": ["det-1"],
            "model_ref": "builtin:model/roi_refinement/example_model",
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert body["candidate_count"] == 1
    assert body["model"]["ref"] == "builtin:model/roi_refinement/example_model"
    assert body["model"]["artifact_path"].endswith("/model.keras")


def test_api_roi_refinement_identity_stores_refined_detection():
    client, _, _ = make_client()

    response = client.post(
        "/roi-refinement",
        json={
            "detection_ids": ["det-1"],
            "model_kind": "identity",
            "allow_frame_expansion": False,
            "store": True,
            "batch_size": 1,
            "encoding": "raw",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stored"] is True
    assert body["refined_count"] == 1
    assert body["stored_count"] == 1
    refined = body["refined_detections"][0]
    assert refined["candidate_detection_id"] == "det-1"
    assert refined["primary_candidate_detection_id"] == "det-1"
    assert refined["candidate_detection_ids"] == ["det-1"]
    assert refined["refinement_relationship"] == "one_to_one"
    assert refined["refined_roi_url"] == "/refined-detections/refined-det-1/roi"
    assert refined["refined_mask_url"] == "/refined-detections/refined-det-1/mask"
    assert refined["metadata"]["detection_stage"] == "refined"
    assert refined["metadata"]["refinement_method"] == "identity"


def test_api_roi_refinement_rejects_missing_roi_payload():
    client, _, _ = make_client()

    response = client.post(
        "/roi-refinement",
        json={
            "detection_ids": ["det-no-roi"],
            "model_kind": "identity",
            "allow_frame_expansion": False,
        },
    )

    assert response.status_code == 422
    assert "do not include ROI payload data" in response.json()["detail"]


def test_api_roi_refinement_loads_frame_crop_when_roi_payload_is_missing(monkeypatch):
    from Pelagia.api.routes import roi_refinement as roi_refinement_route

    loaded = []

    def fake_retrieve_frame(frame_id, *, context, payload_kind):
        loaded.append((frame_id, payload_kind))
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=1,
            data=np.arange(100, dtype=np.uint8).reshape(10, 10),
        )

    monkeypatch.setattr(roi_refinement_route, "retrieve_frame", fake_retrieve_frame)
    client, _, _ = make_client()

    response = client.post(
        "/roi-refinement",
        json={
            "detection_ids": ["det-no-roi"],
            "model_kind": "identity",
            "store": True,
            "batch_size": 1,
            "encoding": "raw",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["refined_count"] == 1
    assert loaded == [("frame-1", "preprocessed")]
    assert body["refined_detections"][0]["metadata"]["refinement_initial_roi_source"] == "frame"


def test_api_roi_refinement_auto_encoding_reuses_candidate_encoding():
    client, _, _ = make_client()

    response = client.post(
        "/roi-refinement",
        json={
            "detection_ids": ["det-1"],
            "model_kind": "identity",
            "allow_frame_expansion": False,
            "store": True,
            "batch_size": 1,
            "encoding": "auto",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stored_count"] == 1
    assert body["resolved_options"]["encoding"] is None


def test_api_queue_roi_refinement_job():
    client, repo, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/roi-refinement/jobs",
        headers=headers,
        json={
            "detection_ids": ["det-1"],
            "model_kind": "identity",
            "allow_frame_expansion": False,
            "batch_size": 2,
            "residual_discovery_enabled": True,
            "residual_min_area": 4,
            "residual_roi_assembly_connectivity": 4,
            "priority": 7,
            "depends_on": ["job-1"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"]["stage"] == PipelineStage.ROI_REFINEMENT.value
    assert body["job"]["project_id"] == "project-1"
    assert body["job"]["run_id"] == "run-1"
    assert body["job"]["asset_id"] == "asset-1"
    assert body["job"]["priority"] == 7
    assert body["job"]["depends_on"] == ["job-1"]
    assert body["job"]["payload"]["detection_ids"] == ["det-1"]
    assert body["job"]["payload"]["model_kind"] == "identity"
    assert body["job"]["payload"]["batch_size"] == 2
    assert body["job"]["payload"]["residual_discovery_enabled"] is True
    assert body["job"]["payload"]["residual_min_area"] == 4
    assert body["job"]["payload"]["residual_roi_assembly_connectivity"] == 4
    assert body["job"]["payload"]["allow_frame_expansion"] is False
    assert repo.created_jobs[-1]["stage"] == PipelineStage.ROI_REFINEMENT.value


def test_api_queue_roi_refinement_job_dry_run():
    client, _, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/roi-refinement/jobs",
        headers=headers,
        json={
            "detection_ids": ["det-1"],
            "model_ref": "builtin:model/roi_refinement/example_model",
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert body["payload"]["detection_ids"] == ["det-1"]
    assert body["model"]["ref"] == "builtin:model/roi_refinement/example_model"


def test_api_preprocessing_options_are_ui_ready():
    client, _, _ = make_client()

    response = client.get("/preprocessing/options")

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_stage_order"] == [
        "source",
        "crop",
        "mask",
        "flatfield",
        "background_correction",
        "inversion",
        "recording",
    ]
    assert "jpg" in body["supported"]["image_encodings"]
    assert body["defaults"]["preprocessing"]["invert_intensity"] is True
    assert body["defaults"]["flatfield"]["flatfield_q"] == 0.9
    flatfield_fields = {field["key"]: field for field in body["fields"]["flatfield"]}
    assert flatfield_fields["flatfield_q"]["min"] == 0
    assert flatfield_fields["flatfield_q"]["max"] == 1
    assert flatfield_fields["flatfield_axis"]["options"] == [0, 1]
    assert flatfield_fields["flatfield_min_field_value"]["min"] == 0
    assert flatfield_fields["flatfield_max_field_value"]["type"] == "nullable-number"
    background_fields = {field["key"]: field for field in body["fields"]["background_correction"]}
    assert background_fields["background_asset_id"]["type"] == "nullable-string"
    assert background_fields["background_frame_ids"]["type"] == "string-list"
    assert background_fields["background_start_frame"]["min"] == 0
    assert background_fields["background_end_frame"]["min"] == 0
    assert background_fields["background_limit"]["min"] == 1
    assert sorted(background_fields["background_payload_kind"]["options"]) == [
        "corrected",
        "original",
        "preprocessed",
        "processed",
        "raw",
    ]
    assert background_fields["background_encoding"]["default"] == "zstd"
    assert background_fields["background_min_field_value"]["min"] == 0
    assert background_fields["background_max_field_value"]["type"] == "nullable-number"
    recording_fields = {field["key"]: field for field in body["fields"]["recording"]}
    assert recording_fields["encoding"]["options"] == ["jpg", "png", "raw", "zstd"]


def test_api_kvstore_includes_status_and_health():
    client, _, _ = make_client()

    response = client.get("/kvstore")

    assert response.status_code == 200
    body = response.json()
    assert body["status"]["initialized"] is True
    assert body["status"]["total_stored_blobs"] == 3
    assert body["health"]["healthy"] is True


def test_api_live_files_indexes_server_directory(tmp_path):
    visible_dir = tmp_path / "frames"
    visible_dir.mkdir()
    visible_file = visible_dir / "sample.mkv"
    visible_file.write_bytes(b"video")
    hidden_file = visible_dir / ".hidden"
    hidden_file.write_text("secret", encoding="utf-8")
    client, _, _ = make_client()
    client.app.state.context.config.file_browser.root_path_import_dir = tmp_path
    client.app.state.context.config.file_browser.root_path_kvstore = tmp_path / "kvstore"

    roots_response = client.get("/live/files")

    assert roots_response.status_code == 200
    roots_body = roots_response.json()
    assert roots_body["directory"] is None
    assert {root["key"] for root in roots_body["roots"]} == {"import", "kvstore"}

    response = client.get("/live/files", params={"directory": str(tmp_path)})

    assert response.status_code == 200
    body = response.json()
    assert body["directory"] == str(tmp_path.resolve())
    assert body["root"]["key"] == "import"
    assert body["count"] == 1
    assert body["entries"][0]["name"] == "frames"
    assert body["entries"][0]["is_dir"] is True

    recursive = client.get(
        "/live/files",
        params={"directory": str(tmp_path), "recursive": True, "include_hidden": True},
    )
    names = {entry["name"] for entry in recursive.json()["entries"]}
    assert {"frames", "sample.mkv", ".hidden"}.issubset(names)


def test_api_live_files_rejects_paths_outside_allowed_roots(tmp_path):
    allowed_dir = tmp_path / "allowed"
    blocked_dir = tmp_path / "blocked"
    allowed_dir.mkdir()
    blocked_dir.mkdir()
    client, _, _ = make_client()
    client.app.state.context.config.file_browser.root_path_import_dir = allowed_dir
    client.app.state.context.config.file_browser.root_path_kvstore = tmp_path / "kvstore"

    response = client.get("/live/files", params={"directory": str(blocked_dir)})

    assert response.status_code == 403


def test_api_live_files_skips_symlink_escape(tmp_path):
    allowed_dir = tmp_path / "allowed"
    blocked_dir = tmp_path / "blocked"
    allowed_dir.mkdir()
    blocked_dir.mkdir()
    (blocked_dir / "secret.mkv").write_bytes(b"video")
    escape = allowed_dir / "escape"
    try:
        escape.symlink_to(blocked_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available on this filesystem")
    client, _, _ = make_client()
    client.app.state.context.config.file_browser.root_path_import_dir = allowed_dir
    client.app.state.context.config.file_browser.root_path_kvstore = tmp_path / "kvstore"

    response = client.get("/live/files", params={"directory": str(allowed_dir)})

    assert response.status_code == 200
    assert "escape" not in {entry["name"] for entry in response.json()["entries"]}


def test_api_live_threshold_and_detection_candidate_are_separate(monkeypatch):
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={"run_id": "run-1", "asset_id": "asset-1", "frame_id": "frame-1"},
    )
    monkeypatch.setattr(frame_store, "retrieve_frame", lambda frame_id, context, payload_kind="original": frame)
    monkeypatch.setattr("Pelagia.api.routes.live.retrieve_frame", lambda frame_id, context, payload_kind="original": frame)
    client, _, _ = make_client()

    threshold_response = client.get(
        "/live/threshold",
        params={
            "frame_id": "frame-1",
            "threshold": 1,
            "apply_preprocessing": False,
            "include_mask_payload": True,
        },
    )

    assert threshold_response.status_code == 200
    threshold_body = threshold_response.json()
    assert threshold_body["saved"] is False
    assert threshold_body["sandboxed"] is True
    assert threshold_body["source_frame_id"] == "frame-1"
    assert threshold_body["sandbox_frame_id"] == "live-frame-1"
    assert threshold_body["mask"]["shape"] == [10, 10]
    assert threshold_body["mask"]["foreground_pixels"] == 12
    assert threshold_body["mask"]["mask_encoding"] == "png"
    assert threshold_body["resolved_options"]["thresholding"]["threshold_method"] == "manual"

    response = client.get(
        "/live/detection-candidate",
        params={
            "frame_id": "frame-1",
            "threshold": 1,
            "apply_preprocessing": False,
            "min_perimeter": 0,
            "padding": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["saved"] is False
    assert body["sandboxed"] is True
    assert body["source_frame_id"] == "frame-1"
    assert body["sandbox_frame_id"] == "live-frame-2"
    assert body["candidate_detection_count"] == 1
    assert body["detection_count"] == 1
    detection = body["candidate_detections"][0]
    assert detection["frame_id"] == "frame-1"
    assert detection["bbox_x"] == 3
    assert detection["bbox_y"] == 2
    assert "roi_payload" not in detection
    assert "roi_payload_bytes" not in detection
    assert "mask_payload_bytes" not in detection
    assert detection["roi_encoding"] is None
    assert body["payloads_encoded"] is False
    assert body["resolved_options"]["thresholding"]["threshold_method"] == "manual"
    assert body["resolved_options"]["mask_augmentation"]["mask_augmentation_steps"] == []
    assert body["stage_counts"]["recorded_detection_count"] == 1
    assert "roi_assembly" in body["stage_durations_ms"]

    removed = client.get("/live/segmentation", params={"frame_id": "frame-1"})
    assert removed.status_code == 404


def test_api_live_threshold_requires_preprocessed_payload_before_preprocessed_source():
    client, _, _ = make_client()

    response = client.get(
        "/live/threshold",
        params={
            "frame_id": "frame-1",
            "frame_payload_kind": "preprocessed",
            "apply_preprocessing": False,
        },
    )

    assert response.status_code == 422
    assert "Run /live/preprocess first" in response.json()["detail"]


def test_api_live_detection_candidate_requires_preprocessed_payload_before_preprocessed_source():
    client, _, _ = make_client()

    response = client.get(
        "/live/detection-candidate",
        params={
            "frame_id": "frame-1",
            "frame_payload_kind": "preprocessed",
            "apply_preprocessing": False,
        },
    )

    assert response.status_code == 422
    assert "Run /live/preprocess first" in response.json()["detail"]


def test_api_segmentation_options_are_ui_ready():
    client, _, _ = make_client()

    response = client.get("/segmentation/options")

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_stage_order"] == [
        "source",
        "preprocessing",
        "thresholding",
        "mask_augmentation",
        "roi_assembly",
        "roi_filter",
        "roi_recording",
    ]
    assert "bounded_otsu_canny" in body["supported"]["threshold_methods"]
    assert "erode" in body["supported"]["mask_augmentation_steps"]
    assert "connected_components" in body["supported"]["roi_assembly_methods"]
    assert body["defaults"]["roi_filter"]["min_perimeter"] is None
    assert body["defaults"]["roi_recording"]["roi_encoding"] == "zstd"
    assert body["defaults"]["preprocessing"]["flatfield_min_field_value"] == 1.0
    assert body["defaults"]["preprocessing"]["background_min_field_value"] == 1.0
    assert body["config_defaults"]["flatfield"]["flatfield_min_field_value"] == 1.0
    preprocessing_fields = {field["key"]: field for field in body["fields"]["preprocessing"]}
    assert preprocessing_fields["flatfield_max_field_value"]["type"] == "nullable-number"
    assert preprocessing_fields["background_max_field_value"]["type"] == "nullable-number"
    threshold_fields = {field["key"]: field for field in body["fields"]["thresholding"]}
    assert threshold_fields["canny_low_threshold"]["threshold_methods"] == [
        "canny",
        "bounded_otsu_canny",
    ]
    assert body["fields"]["mask_augmentation"][1]["type"] == "multi-enum"


def test_api_can_create_queue_job():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/jobs",
        headers=headers,
        json={
            "stage": "extract_frames",
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"source_path": "/tmp/source.avi"},
        },
    )

    assert response.status_code == 200
    assert response.json()["job"]["stage"] == "extract_frames"
    assert response.json()["job"]["project_id"] == "project-1"
    assert repository.created_jobs[0]["run_id"] == "run-1"
    assert repository.created_jobs[0]["project_id"] == "project-1"


def test_api_create_queue_job_rejects_cross_project_asset():
    client, repository, _ = make_client()
    headers = auth_headers(client, username="admin", project_key="other")

    response = client.post(
        "/jobs",
        headers=headers,
        json={
            "stage": "extract_frames",
            "run_id": "run-1",
            "asset_id": "asset-1",
            "payload": {"source_path": "/tmp/source.avi"},
        },
    )

    assert response.status_code == 404
    assert repository.created_jobs == []


def test_api_lists_jobs_without_details_by_default():
    client, _, _ = make_client()

    response = client.get("/jobs")

    assert response.status_code == 200
    job = response.json()["jobs"][0]
    assert job["include_details"] is False
    assert job["payload_bytes"] == 1024
    assert job["result_bytes"] == 2048
    assert "payload" not in job
    assert "result" not in job


def test_api_lists_jobs_with_details_when_requested():
    client, _, _ = make_client()

    response = client.get("/jobs?include_details=true")

    assert response.status_code == 200
    job = response.json()["jobs"][0]
    assert job["include_details"] is True
    assert job["payload"]["frame_ids"] == ["frame-1"]
    assert job["result"]["detection_ids"] == ["det-1"]


def test_api_can_clear_jobs():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/jobs/clear",
        headers=headers,
        json={
            "stage": ["extract_frames"],
            "status": ["queued", "leased"],
            "reason": "reset queue",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["matched_count"] == 1
    assert body["cancellable_count"] == 1
    assert body["cancelled_count"] == 1
    assert body["jobs"][0]["status"] == "cancelled"
    assert repository.cancel_job_calls[-1]["project_id"] == "project-1"
    assert repository.cancel_job_calls[-1]["stages"] == ["extract_frames"]
    assert repository.cancel_job_calls[-1]["statuses"] == ["queued", "leased"]
    assert repository.cancel_job_calls[-1]["reason"] == "reset queue"


def test_api_clear_jobs_supports_dry_run():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/jobs/clear",
        headers=headers,
        json={"dry_run": True, "worker_id": "worker-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert body["matched_count"] == 1
    assert body["cancelled_count"] == 0
    assert body["jobs"] == []
    assert repository.cancel_job_calls[-1]["worker_id"] == "worker-1"
    assert repository.cancel_job_calls[-1]["dry_run"] is True


def test_api_clear_jobs_accepts_empty_body():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post("/jobs/clear", headers=headers)

    assert response.status_code == 200
    assert response.json()["cancelled_count"] == 1
    assert repository.cancel_job_calls[-1]["project_id"] == "project-1"


def test_api_clear_jobs_delete_mode_dispatches_to_delete_jobs():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/jobs/clear",
        headers=headers,
        json={"mode": "delete", "status": ["cancelled"], "stage": ["extract_frames"]},
    )

    assert response.status_code == 200
    assert response.json()["deleted_count"] == 1
    assert repository.delete_job_calls[-1]["project_id"] == "project-1"
    assert repository.delete_job_calls[-1]["statuses"] == ["cancelled"]
    assert repository.delete_job_calls[-1]["stages"] == ["extract_frames"]
    assert repository.cancel_job_calls == []


def test_api_clear_jobs_uses_active_project_scope():
    client, repository, _ = make_client()
    headers = auth_headers(client, username="admin", project_key="other")

    response = client.post("/jobs/clear", headers=headers, json={})

    assert response.status_code == 200
    assert response.json()["matched_count"] == 0
    assert repository.cancel_job_calls[-1]["project_id"] == "project-2"


def test_api_clear_jobs_rejects_invalid_filters():
    client, _, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/jobs/clear",
        headers=headers,
        json={"status": ["not-a-status"]},
    )

    assert response.status_code == 422


def test_api_can_create_and_list_structured_logs():
    client, repository, _ = make_client()

    create_response = client.post(
        "/logs",
        json={
            "event_type": "ui.status_loaded",
            "message": "Status page loaded",
            "level": "info",
            "duration_ms": 12.5,
            "payload": {"route": "/status"},
        },
    )
    list_response = client.get("/logs?level=warning&limit=5&offset=10")

    assert create_response.status_code == 200
    assert create_response.json()["log"]["event_type"] == "ui.status_loaded"
    assert repository.logs[0]["duration_ms"] == 12.5
    assert list_response.status_code == 200
    assert list_response.json()["logs"][0]["level"] == "warning"
    assert list_response.json()["logs"][0]["limit"] == 5
    assert list_response.json()["logs"][0]["offset"] == 10


def test_api_can_queue_segmentation_job():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/segmentation/jobs",
        headers=headers,
        json={
            "run_id": "run-1",
            "asset_id": "asset-1",
            "frame_ids": ["frame-1"],
            "padding": 4,
            "roi_encoding": "raw",
            "mask_augmentation_steps": ["erode"],
            "erode_kernel_w": 5,
            "erode_kernel_h": 3,
            "erode_iterations": 2,
            "roi_assembly_method": "contours",
            "min_area": 7,
            "min_width": 2,
            "store_roi_payload_min_area": 20,
            "always_store_mask": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["job"]["stage"] == "segment"
    assert response.json()["job"]["project_id"] == "project-1"
    assert repository.created_jobs[-1]["run_id"] == "run-1"
    assert repository.created_jobs[-1]["asset_id"] == "asset-1"
    assert repository.created_jobs[-1]["payload"]["frame_ids"] == ["frame-1"]
    assert repository.created_jobs[-1]["payload"]["padding"] == 4
    assert repository.created_jobs[-1]["payload"]["roi_encoding"] == "raw"
    assert repository.created_jobs[-1]["payload"]["mask_augmentation_steps"] == ["erode"]
    assert repository.created_jobs[-1]["payload"]["erode_kernel_w"] == 5
    assert repository.created_jobs[-1]["payload"]["erode_iterations"] == 2
    assert repository.created_jobs[-1]["payload"]["roi_assembly_method"] == "contours"
    assert repository.created_jobs[-1]["payload"]["min_area"] == 7
    assert repository.created_jobs[-1]["payload"]["min_width"] == 2
    assert repository.created_jobs[-1]["payload"]["store_roi_payload_min_area"] == 20
    assert repository.created_jobs[-1]["payload"]["always_store_mask"] is False
    assert repository.created_jobs[-1]["payload"]["flatfield_correction"] is True
    assert repository.created_jobs[-1]["payload"]["flatfield_q"] == 0.9
    assert repository.created_jobs[-1]["payload"]["flatfield_axis"] == 0
    assert repository.created_jobs[-1]["payload"]["flatfield_min_field_value"] == 1.0
    assert repository.created_jobs[-1]["payload"]["background_min_field_value"] == 1.0


def test_api_rejects_preprocessed_segmentation_job_for_unprocessed_frames():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/segmentation/jobs",
        headers=headers,
        json={
            "run_id": "run-1",
            "asset_id": "asset-1",
            "frame_ids": ["frame-1"],
            "frame_payload_kind": "preprocessed",
        },
    )

    assert response.status_code == 422
    assert "lack preprocessed payloads" in response.json()["detail"]
    assert repository.created_jobs == []


def test_api_validate_segmentation_resolves_without_queueing():
    client, repository, _ = make_client()

    response = client.post(
        "/segmentation/validate",
        json={
            "asset_id": "asset-1",
            "frame_ids": ["frame-1"],
            "threshold_method": "adaptive_mean",
            "mask_augmentation_steps": ["open", "fill_holes"],
            "open_iterations": 2,
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["payload"]["threshold_method"] == "adaptive_mean"
    assert body["payload"]["mask_augmentation_steps"] == ["open", "fill_holes"]
    assert body["resolved_options"]["mask_augmentation"]["open_iterations"] == 2
    assert repository.created_jobs == []


def test_api_queue_segmentation_dry_run_does_not_create_job():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/segmentation/jobs",
        headers=headers,
        json={"asset_id": "asset-1", "frame_ids": ["frame-1"], "dry_run": True},
    )

    assert response.status_code == 200
    assert response.json()["dry_run"] is True
    assert response.json()["payload"]["frame_ids"] == ["frame-1"]
    assert repository.created_jobs == []


def test_api_segmentation_rejects_invalid_enum_values():
    client, _, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/segmentation/jobs",
        headers=headers,
        json={"asset_id": "asset-1", "threshold_method": "definitely-not-real"},
    )

    assert response.status_code == 422


def test_api_can_queue_frame_preprocess_job():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/frame/preprocess/jobs",
        headers=headers,
        json={
            "frame_ids": ["frame-1"],
            "flatfield_correction": False,
            "background_correction": True,
            "encoding": "jpg",
        },
    )

    assert response.status_code == 200
    assert response.json()["job"]["stage"] == "preprocess_frames"
    assert response.json()["job"]["project_id"] == "project-1"
    assert repository.created_jobs[-1]["run_id"] == "run-1"
    assert repository.created_jobs[-1]["asset_id"] == "asset-1"
    assert repository.created_jobs[-1]["payload"]["frame_ids"] == ["frame-1"]
    assert repository.created_jobs[-1]["payload"]["flatfield_correction"] is False
    assert repository.created_jobs[-1]["payload"]["background_correction"] is True
    assert repository.created_jobs[-1]["payload"]["encoding"] == "jpg"


def test_api_can_generate_frame_background(monkeypatch):
    from Pelagia.api.routes import frame

    calls = []

    def fake_generate_background_for_frames(frame_ids, **kwargs):
        calls.append((frame_ids, kwargs))
        return {
            "background_payload_ref": "background-key",
            "frame_ids": frame_ids,
            "frame_count": len(frame_ids),
            "updated_frame_count": len(frame_ids),
        }

    monkeypatch.setattr(frame, "generate_background_for_frames", fake_generate_background_for_frames)
    client, _, _ = make_client()

    response = client.post(
        "/frame/background",
        json={
            "frame_ids": ["frame-1"],
            "payload_kind": "original",
            "encoding": "raw",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == PipelineStage.BACKGROUND_FRAMES.value
    assert body["run_id"] == "run-1"
    assert body["asset_id"] == "asset-1"
    assert body["background_payload_ref"] == "background-key"
    assert calls[0][0] == ["frame-1"]
    assert calls[0][1]["payload_kind"] == "original"
    assert calls[0][1]["encoding"] == "raw"


def test_api_can_queue_frame_background_job():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/frame/background/jobs",
        headers=headers,
        json={
            "frame_ids": ["frame-1"],
            "payload_kind": "original",
            "encoding": "zstd",
            "priority": 7,
            "depends_on": ["job-1"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job"]["stage"] == PipelineStage.BACKGROUND_FRAMES.value
    assert body["job"]["run_id"] == "run-1"
    assert body["job"]["asset_id"] == "asset-1"
    assert body["job"]["priority"] == 7
    assert body["job"]["depends_on"] == ["job-1"]
    assert repository.created_jobs[-1]["stage"] == PipelineStage.BACKGROUND_FRAMES.value
    assert repository.created_jobs[-1]["payload"]["frame_ids"] == ["frame-1"]
    assert repository.created_jobs[-1]["payload"]["payload_kind"] == "original"
    assert repository.created_jobs[-1]["payload"]["encoding"] == "zstd"


def test_api_queue_frame_background_job_dry_run_does_not_create_job():
    client, repository, _ = make_client()
    headers = auth_headers(client)

    response = client.post(
        "/frame/background/jobs",
        headers=headers,
        json={"asset_id": "asset-1", "start_frame": 2, "end_frame": 4, "dry_run": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert body["asset_id"] == "asset-1"
    assert body["payload"]["frame_ids"] == ["frame-1"]
    assert body["payload"]["start_frame"] == 2
    assert repository.created_jobs == []


def test_api_frame_endpoints_accept_dimension_resize(monkeypatch):
    from Pelagia.api.routes import frame

    calls = []

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original"):
        calls.append((frame_id, payload_kind))
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=2,
            data=np.arange(10 * 20, dtype=np.uint8).reshape((10, 20)),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    monkeypatch.setattr(frame, "retrieve_frame", fake_retrieve_frame)
    client, _, _ = make_client()

    original = client.get("/frames/original?frame_id=frame-1&format=matrix&width=5")
    preprocessed = client.get("/frames/preprocess?frame_id=frame-1&format=png&height=4")

    assert original.status_code == 200
    assert original.json()["shape"] == [2, 5]
    assert original.json()["requested_width"] == 5
    assert preprocessed.status_code == 200
    decoded = cv2.imdecode(np.frombuffer(preprocessed.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (4, 8)
    assert preprocessed.headers["x-pelagia-height"] == "4"
    assert preprocessed.headers["x-pelagia-source-width"] == "20"
    assert preprocessed.headers["x-pelagia-source-height"] == "10"
    assert preprocessed.headers["x-pelagia-image-width"] == "8"
    assert preprocessed.headers["x-pelagia-image-height"] == "4"
    assert preprocessed.headers["x-pelagia-scale-x"] == "0.4"
    assert preprocessed.headers["x-pelagia-scale-y"] == "0.4"
    assert calls == [("frame-1", "original"), ("frame-1", "preprocessed")]


def test_api_frame_original_uuid_asset_query_requires_auth_with_cors(monkeypatch):
    import uuid

    asset_id = str(uuid.uuid4())
    client, _, _ = make_client(auth_enabled=True)

    response = client.get(
        f"/frame/original?format=matrix&asset_id={asset_id}&frame_num=1&width=1100"
        f"&token_id={asset_id}&session_id={asset_id}",
        headers={"Origin": "http://127.0.0.1:5173"},
    )

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_api_frame_image_endpoints_accept_head_for_scale_headers(monkeypatch):
    from Pelagia.api.routes import frame

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original"):
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=2,
            data=np.arange(10 * 20, dtype=np.uint8).reshape((10, 20)),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    monkeypatch.setattr(frame, "retrieve_frame", fake_retrieve_frame)
    client, _, _ = make_client()

    original = client.head("/frame/original?frame_id=frame-1&format=jpg&width=5")
    preprocessed = client.head("/frames/preprocess?frame_id=frame-1&format=jpg&height=4")

    assert original.status_code == 200
    assert original.headers["x-pelagia-source-width"] == "20"
    assert original.headers["x-pelagia-source-height"] == "10"
    assert original.headers["x-pelagia-image-width"] == "5"
    assert original.headers["x-pelagia-image-height"] == "2"
    assert original.headers["x-pelagia-scale-x"] == "0.25"
    assert original.headers["x-pelagia-scale-y"] == "0.2"
    assert original.content == b""
    assert preprocessed.status_code == 200
    assert preprocessed.headers["x-pelagia-source-width"] == "20"
    assert preprocessed.headers["x-pelagia-source-height"] == "10"
    assert preprocessed.headers["x-pelagia-image-width"] == "8"
    assert preprocessed.headers["x-pelagia-image-height"] == "4"
    assert preprocessed.headers["x-pelagia-scale-x"] == "0.4"
    assert preprocessed.headers["x-pelagia-scale-y"] == "0.4"
    assert preprocessed.content == b""


def test_api_frame_image_endpoints_encode_float_preprocessed_frames(monkeypatch):
    from Pelagia.api.routes import frame

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original"):
        if payload_kind == "preprocessed":
            data = np.linspace(-2.0, 7.0, 10 * 20, dtype=np.float32).reshape((10, 20))
        else:
            data = np.arange(10 * 20, dtype=np.uint8).reshape((10, 20))
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=2,
            data=data,
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    monkeypatch.setattr(frame, "retrieve_frame", fake_retrieve_frame)
    client, _, _ = make_client()

    response = client.get("/frame/preprocessed?frame_id=frame-1&format=jpg&width=5")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (2, 5)
    assert decoded.dtype == np.uint8


def test_api_frame_context_returns_ui_ready_contract():
    client, repository, _ = make_client()
    repository.preprocessed_payload_ref = "preprocessed-key"

    response = client.get(
        "/frames/frame-1/context?width=320&detection_limit=1&detection_offset=4"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["frame"]["id"] == "frame-1"
    assert body["frame"]["has_preprocessed_payload"] is True
    assert body["asset"]["id"] == "asset-1"
    assert body["image_urls"]["original"] == "/frame/original?frame_id=frame-1&format=jpg&width=320"
    assert body["image_urls"]["preprocessed"] == "/frame/preprocessed?frame_id=frame-1&format=jpg&width=320"
    assert body["detection_count"] == 1
    assert body["detections"][0]["bbox"] == {"x": 3, "y": 4, "w": 5, "h": 6}
    assert body["detections"][0]["crop_bbox"] == {"x": 1, "y": 2, "w": 9, "h": 10}
    assert body["page"] == {"limit": 1, "offset": 4, "count": 1, "next_offset": 5}


def test_api_frame_context_handles_missing_frame_and_missing_preprocessed_image():
    client, repository, _ = make_client()

    missing = client.get("/frames/missing/context")
    no_preprocessed = client.get("/frames/frame-1/context?include_detections=false")

    assert missing.status_code == 404
    assert no_preprocessed.status_code == 200
    body = no_preprocessed.json()
    assert body["image_urls"]["original"].startswith("/frame/original?")
    assert body["image_urls"]["preprocessed"] is None
    assert body["detections"] == []
    assert body["detection_count"] == 0
    assert body["page"] == {"limit": 500, "offset": 0, "count": 0, "next_offset": None}


def test_api_direct_preprocess_accepts_frame_ids(monkeypatch):
    from Pelagia.api.routes import frame

    stored = []

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original"):
        return FrameData(
            sourcePath="/tmp",
            filename=f"{frame_id}.png",
            frameNumber=2,
            data=np.zeros((2, 2), dtype=np.uint8),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    def fake_preprocess_frame(source_frame, **kwargs):
        array = np.asarray(source_frame.read()) + 1
        return FrameData(
            sourcePath=source_frame.sourcePath,
            filename=source_frame.filename,
            frameNumber=source_frame.frameNumber,
            data=array,
            metadata={"preprocessed": True},
        )

    def fake_store_preprocessed_frame(frame_id, processed, **kwargs):
        stored.append((frame_id, kwargs.get("encoding")))
        return {
            "id": frame_id,
            "asset_id": "asset-1",
            "frame_index": 2,
            "preprocessed_payload_ref": f"{frame_id}-preprocessed",
        }

    monkeypatch.setattr(frame, "retrieve_frame", fake_retrieve_frame)
    monkeypatch.setattr(frame, "preprocess_frame_for_segmentation", fake_preprocess_frame)
    monkeypatch.setattr(frame, "store_preprocessed_frame", fake_store_preprocessed_frame)
    client, _, _ = make_client()

    response = client.post(
        "/frame/preprocess",
        json={"frame_ids": ["frame-1", "frame-2"], "encoding": "png"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["frame_count"] == 2
    assert body["frame_ids"] == ["frame-1", "frame-2"]
    assert [frame["stored"] for frame in body["frames"]] == [True, True]
    assert stored == [("frame-1", "png"), ("frame-2", "png")]

    asset_response = client.post(
        "/frame/preprocess",
        json={"asset_id": "asset-1", "start_frame": 2, "limit": 1, "store": False},
    )
    assert asset_response.status_code == 200
    assert asset_response.json()["asset_id"] == "asset-1"
    assert asset_response.json()["frame_count"] == 1
    assert asset_response.json()["frames"][0]["stored"] is False


def test_api_direct_segmentation_accepts_frame_ids(monkeypatch):
    from Pelagia.api.routes import segmentation

    segmented = []

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original"):
        return FrameData(
            sourcePath="/tmp",
            filename=f"{frame_id}.png",
            frameNumber=2,
            data=np.zeros((2, 2), dtype=np.uint8),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    def fake_segment_frame(source_frame, *, frame_record, **kwargs):
        segmented.append((frame_record.id, kwargs.get("padding")))
        return [{"frame_id": frame_record.id, "roi_payload": b"roi", "mask_payload": b"mask"}]

    monkeypatch.setattr(segmentation, "retrieve_frame", fake_retrieve_frame)
    monkeypatch.setattr(segmentation, "segment_frame", fake_segment_frame)
    client, _, _ = make_client()

    response = client.post(
        "/segmentation/frames",
        json={"frame_ids": ["frame-1", "frame-2"], "padding": 4},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["frame_count"] == 2
    assert body["detection_count"] == 2
    assert body["frame_ids"] == ["frame-1", "frame-2"]
    assert body["frames"][0]["resolved_options"]["roi_recording"]["padding"] == 4
    assert body["frames"][0]["stage_counts"]["recorded_detection_count"] == 1
    assert segmented == [("frame-1", 4), ("frame-2", 4)]


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

    assert frame_response.json()["frames"][0]["preview_thumbhash_bytes"] == 3
    assert frame_response.json()["frames"][0]["preview_thumbhash_base64"] == "YWJj"
    assert "preview_thumbhash" not in frame_response.json()["frames"][0]
    assert detection_response.json()["detections"][0]["roi_payload_bytes"] == 4
    assert detection_response.json()["detections"][0]["mask_payload_bytes"] == 4


def test_api_asset_detail_includes_frame_count():
    client, _, _ = make_client()

    response = client.get("/assets/asset-1")

    assert response.status_code == 200
    assert response.json()["asset"]["id"] == "asset-1"
    assert response.json()["asset"]["frame_count"] == 12


def test_api_filters_asset_detections():
    client, _, _ = make_client()

    response = client.get(
        "/assets/asset-1/detections"
        "?frame_id=frame-1&start_frame=2&end_frame=5&roi_index=1"
        "&min_bbox_x=1&max_bbox_x=10&min_bbox_y=2&max_bbox_y=11"
        "&min_bbox_w=3&max_bbox_w=12&min_bbox_h=4&max_bbox_h=13"
        "&min_area=5.5&max_area=100.5&min_perimeter=6.5&max_perimeter=80.5"
        "&roi_encoding=raw&roi_format=raw_ndarray_c_order"
        "&mask_encoding=raw&mask_format=raw_ndarray_c_order&limit=7&offset=3"
    )

    assert response.status_code == 200
    detection = response.json()["detections"][0]
    assert detection["frame_id"] == "frame-1"
    assert detection["start_frame"] == 2
    assert detection["end_frame"] == 5
    assert detection["roi_index"] == 1
    assert detection["min_bbox_x"] == 1
    assert detection["max_bbox_h"] == 13
    assert detection["min_area"] == 5.5
    assert detection["max_perimeter"] == 80.5
    assert detection["roi_encoding"] == "raw"
    assert detection["mask_format"] == "raw_ndarray_c_order"
    assert detection["limit"] == 7
    assert detection["offset"] == 3
    assert detection["bbox"] == {"x": 3, "y": 4, "w": 5, "h": 6}
    assert detection["crop_bbox"] == {"x": 1, "y": 2, "w": 9, "h": 10}
    assert response.json()["page"] == {"limit": 7, "offset": 3, "count": 1, "next_offset": None}


def test_api_lists_global_detections_without_image_payloads():
    client, _, _ = make_client()

    response = client.get("/detections?asset_id=asset-1&collection=test&limit=100&offset=400")

    assert response.status_code == 200
    detection = response.json()["detections"][0]
    assert detection["asset_id"] == "asset-1"
    assert detection["collection"] == "test"
    assert detection["limit"] == 100
    assert detection["offset"] == 400
    assert "roi_payload" not in detection
    assert detection["roi_payload_bytes"] == 4
    assert detection["mask_payload_bytes"] == 4
    assert response.json()["page"] == {"limit": 100, "offset": 400, "count": 1, "next_offset": None}


def test_api_get_detection_includes_payload_data():
    client, _, _ = make_client()

    response = client.get("/detections/det-1")

    assert response.status_code == 200
    detection = response.json()["detection"]
    assert detection["id"] == "det-1"
    assert detection["roi_payload"] == "0080ff40"
    assert detection["mask_payload"] == "6d61736b"


def test_api_detection_framedata_returns_matrix_and_png():
    client, _, _ = make_client()

    matrix_response = client.get("/detections/det-1/framedata?format=matrix")
    png_response = client.get("/detections/det-1/framedata?format=png")

    assert matrix_response.status_code == 200
    assert matrix_response.json()["shape"] == [2, 2]
    assert matrix_response.json()["data"] == [[0, 128], [255, 64]]
    assert png_response.status_code == 200
    assert png_response.headers["content-type"] == "image/png"
    assert png_response.content.startswith(b"\x89PNG")


def test_api_detection_roi_endpoint_returns_matrix_and_images():
    client, _, _ = make_client()

    matrix_response = client.get("/detections/det-1/roi?format=matrix")
    png_response = client.get("/detections/det-1/roi?format=png")
    jpg_response = client.get("/detections/det-1/roi?format=jpg&width=1")

    assert matrix_response.status_code == 200
    assert matrix_response.json()["payload_kind"] == "roi"
    assert matrix_response.json()["shape"] == [2, 2]
    assert matrix_response.json()["data"] == [[0, 128], [255, 64]]
    assert png_response.status_code == 200
    assert png_response.headers["content-type"] == "image/png"
    assert png_response.headers["x-pelagia-payload-kind"] == "roi"
    assert png_response.headers["x-pelagia-source-width"] == "2"
    assert png_response.headers["x-pelagia-image-width"] == "2"
    assert png_response.content.startswith(b"\x89PNG")
    assert jpg_response.status_code == 200
    assert jpg_response.headers["content-type"] == "image/jpeg"
    assert jpg_response.headers["x-pelagia-image-width"] == "1"


def test_api_detection_refined_roi_endpoint_returns_matrix_and_png():
    client, _, _ = make_client()

    matrix_response = client.get("/detections/det-1/refined-roi?format=matrix")
    png_response = client.get("/detections/det-1/refined-roi?format=png")

    assert matrix_response.status_code == 200
    assert matrix_response.json()["payload_kind"] == "roi"
    assert matrix_response.json()["shape"] == [2, 2]
    assert matrix_response.json()["data"] == [[5, 6], [7, 8]]
    assert png_response.status_code == 200
    assert png_response.headers["content-type"] == "image/png"
    assert png_response.headers["x-pelagia-payload-kind"] == "roi"
    assert png_response.headers["x-pelagia-source-width"] == "2"
    assert png_response.content.startswith(b"\x89PNG")


def test_api_refined_detection_id_endpoints_return_contract_and_payloads():
    client, _, _ = make_client()

    detail_response = client.get("/refined-detections/refined-det-1")
    matrix_response = client.get("/refined-detections/refined-det-1/roi?format=matrix")
    mask_response = client.get("/refined-detections/refined-det-1/mask?format=matrix")

    assert detail_response.status_code == 200
    refined = detail_response.json()["detection"]
    assert refined["id"] == "refined-det-1"
    assert refined["candidate_detection_id"] == "det-1"
    assert refined["primary_candidate_detection_id"] == "det-1"
    assert refined["candidate_detection_ids"] == ["det-1"]
    assert refined["refinement_relationship"] == "one_to_one"
    assert refined["refined_roi_url"] == "/refined-detections/refined-det-1/roi"
    assert refined["refined_mask_url"] == "/refined-detections/refined-det-1/mask"
    assert matrix_response.status_code == 200
    assert matrix_response.json()["detection_id"] == "refined-det-1"
    assert matrix_response.json()["payload_kind"] == "roi"
    assert matrix_response.json()["data"] == [[5, 6], [7, 8]]
    assert mask_response.status_code == 200
    assert mask_response.json()["payload_kind"] == "mask"
    assert mask_response.json()["data"] == [[0, 255], [255, 0]]


def test_api_detection_roi_can_apply_mask():
    client, _, _ = make_client()

    response = client.get("/detections/det-wide/framedata?format=matrix&apply_mask=true")

    assert response.status_code == 200
    body = response.json()
    assert body["payload_kind"] == "roi"
    assert body["mask_applied"] is True
    assert body["data"] == [[0, 10, 0], [30, 0, 50]]


def test_api_detection_refined_roi_can_apply_mask():
    client, _, _ = make_client()

    response = client.get("/detections/det-1/refined-roi?format=matrix&apply_mask=true")

    assert response.status_code == 200
    body = response.json()
    assert body["payload_kind"] == "roi"
    assert body["mask_applied"] is True
    assert body["data"] == [[0, 6], [7, 0]]


def test_api_detection_roi_mask_and_framedata_support_head():
    client, _, _ = make_client()

    for path, payload_kind in [
        ("/detections/det-1/framedata", "roi"),
        ("/detections/det-1/roi", "roi"),
        ("/detections/det-1/mask", "mask"),
        ("/detections/det-1/refined-roi", "roi"),
        ("/detections/det-1/refined-mask", "mask"),
        ("/refined-detections/refined-det-1/roi", "roi"),
        ("/refined-detections/refined-det-1/mask", "mask"),
    ]:
        response = client.head(f"{path}?format=png")

        assert response.status_code == 200
        assert response.content == b""
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-pelagia-payload-kind"] == payload_kind
        assert response.headers["x-pelagia-source-width"] == "2"
        assert response.headers["x-pelagia-image-width"] == "2"


def test_api_detection_mask_endpoint_returns_matrix_and_png():
    client, _, _ = make_client()

    matrix_response = client.get("/detections/det-1/mask?format=matrix")
    png_response = client.get("/detections/det-1/mask?format=png&height=1")

    assert matrix_response.status_code == 200
    assert matrix_response.json()["payload_kind"] == "mask"
    assert matrix_response.json()["shape"] == [2, 2]
    assert matrix_response.json()["data"] == [[109, 97], [115, 107]]
    assert png_response.status_code == 200
    assert png_response.headers["content-type"] == "image/png"
    assert png_response.headers["x-pelagia-payload-kind"] == "mask"
    assert png_response.headers["x-pelagia-image-height"] == "1"
    assert png_response.content.startswith(b"\x89PNG")


def test_api_detection_roi_can_pad_square_then_invert():
    client, _, _ = make_client()

    response = client.get(
        "/detections/det-wide/roi?format=matrix&pad_square=true&invert=true"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["payload_kind"] == "roi"
    assert body["shape"] == [3, 3]
    assert body["pad_square"] is True
    assert body["inverted"] is True
    assert body["data"] == [
        [255, 245, 235],
        [225, 215, 205],
        [255, 255, 255],
    ]


def test_api_detection_roi_can_add_scale_bar_after_padding_and_inversion():
    client, _, _ = make_client()

    matrix_response = client.get(
        "/detections/det-wide/roi"
        "?format=matrix&square=true&invert=true"
        "&scale_bar=true&scale_bar_length_px=2&scale_bar_height_px=1"
        "&scale_bar_margin_px=0&scale_bar_color=black"
    )
    png_response = client.get(
        "/detections/det-wide/roi"
        "?format=png&square=true&invert=true&scale_bar=true"
        "&scale_bar_length_px=2&scale_bar_height_px=1"
        "&scale_bar_margin_px=0&scale_bar_color=black"
    )

    assert matrix_response.status_code == 200
    body = matrix_response.json()
    assert body["shape"] == [3, 3]
    assert body["scale_bar"] is True
    assert body["data"] == [
        [255, 245, 235],
        [225, 215, 205],
        [0, 0, 255],
    ]
    assert png_response.status_code == 200
    assert png_response.headers["x-pelagia-pad-square"] == "true"
    assert png_response.headers["x-pelagia-inverted"] == "true"
    assert png_response.headers["x-pelagia-scale-bar"] == "true"
    assert png_response.headers["x-pelagia-image-width"] == "3"
    assert png_response.headers["x-pelagia-image-height"] == "3"


def test_api_detection_framedata_accepts_scale():
    client, _, _ = make_client()

    matrix_response = client.get("/detections/det-1/framedata?format=matrix&scale=0.5")
    png_response = client.get("/detections/det-1/framedata?format=png&scale=0.5")

    assert matrix_response.status_code == 200
    assert matrix_response.json()["shape"] == [1, 1]
    assert matrix_response.json()["scale"] == 0.5
    assert png_response.status_code == 200
    assert png_response.headers["x-pelagia-scale"] == "0.5"
    decoded = cv2.imdecode(np.frombuffer(png_response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (1, 1)


def test_openapi_documents_core_response_schemas_and_head_routes():
    client, _, _ = make_client()

    response = client.get("/openapi.json")

    assert response.status_code == 200
    spec = response.json()
    assert "DetectionsListResponse" in spec["components"]["schemas"]
    assert "DetectionSummary" in spec["components"]["schemas"]
    assert "FrameContextResponse" in spec["components"]["schemas"]
    assert "SystemCapabilitiesResponse" in spec["components"]["schemas"]
    assert "head" in spec["paths"]["/detections/{detection_id}/roi"]
    assert "head" in spec["paths"]["/detections/{detection_id}/mask"]
    assert "head" in spec["paths"]["/detections/{detection_id}/framedata"]
    assert "head" in spec["paths"]["/assets/{asset_id}/framedata/{frame_num}"]


def test_api_reports_asset_detection_stats():
    client, _, _ = make_client()

    response = client.get(
        "/assets/detections?collection=test&kind=video&filename=sample&min_detection_count=1&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total_asset_count"] == 2
    assert body["summary"]["identified_asset_count"] == 1
    assert body["summary"]["total_detection_count"] == 7
    assert body["assets"][0]["asset_id"] == "asset-1"
    assert body["assets"][0]["detection_count"] == 7
    assert body["assets"][0]["collection"] == "test"
    assert body["assets"][0]["kind"] == "video"
    assert body["assets"][0]["filename"] == "sample"
    assert body["assets"][0]["min_detection_count"] == 1
    assert body["assets"][0]["limit"] == 5


def test_api_reports_asset_processing_state():
    client, _, _ = make_client()

    response = client.get(
        "/assets/processing-state?collection=test&kind=video&filename=sample&preprocessing_state=has-preprocessed&detection_state=has-detections&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total_asset_count"] == 2
    assert body["summary"]["total_frame_count"] == 12
    assert body["summary"]["total_preprocessed_frame_count"] == 7
    assert body["summary"]["total_detected_frame_count"] == 3
    assert body["summary"]["total_detection_count"] == 9
    assert body["assets"][0]["asset_id"] == "asset-1"
    assert body["assets"][0]["filename"] == "sample"
    assert body["assets"][0]["kind"] == "video"
    assert body["assets"][0]["collection"] == "test"
    assert body["assets"][0]["preprocessing_state"] == "partially-preprocessed"
    assert body["assets"][0]["detection_state"] == "partially-detected"
    assert body["assets"][0]["frame_count"] == 12
    assert body["assets"][0]["preprocessed_frame_count"] == 7
    assert body["assets"][0]["detected_frame_count"] == 3
    assert body["assets"][0]["detection_count"] == 9
    assert body["page"] == {"limit": 5, "offset": 0, "count": 1, "next_offset": None}


def test_api_reports_frame_processing_state():
    client, _, _ = make_client()

    response = client.get(
        "/frames/processing-state?collection=test&kind=video&preprocessing_state=fully-preprocessed&detection_state=fully-detected&start_frame=2&end_frame=5&limit=5"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total_frame_count"] == 2
    assert body["summary"]["total_preprocessed_frame_count"] == 1
    assert body["summary"]["total_detected_frame_count"] == 1
    assert body["summary"]["total_detection_count"] == 7
    assert body["summary"]["total_refined_detection_count"] == 3
    assert body["summary"]["total_unrefined_detection_count"] == 4
    assert body["frames"][0]["frame_id"] == "frame-1"
    assert body["frames"][0]["asset_id"] == "asset-1"
    assert body["frames"][0]["asset_filename"] == "sample.mkv"
    assert body["frames"][0]["frame_index"] == 2
    assert body["frames"][0]["preprocessing_state"] == "fully-preprocessed"
    assert body["frames"][0]["detection_state"] == "fully-detected"
    assert body["frames"][0]["refinement_state"] == "partially-refined"
    assert body["frames"][0]["refined_candidate_detection_count"] == 3
    assert body["frames"][0]["start_frame"] == 2
    assert body["frames"][0]["end_frame"] == 5
    assert body["page"] == {"limit": 5, "offset": 0, "count": 2, "next_offset": None}


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


def test_api_asset_framedata_supports_head(monkeypatch):
    from Pelagia.api.routes import assets

    class FakeFrame:
        def read(self):
            return np.array([[0, 255], [128, 64]], dtype=np.uint8)

    monkeypatch.setattr(assets, "retrieve_frame", lambda frame_id, context: FakeFrame())
    client, _, _ = make_client()

    response = client.head("/assets/asset-1/framedata/2?format=png")

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-pelagia-scale"] == "1.0"


def test_api_framedata_accepts_scale(monkeypatch):
    from Pelagia.api.routes import assets

    class FakeFrame:
        def read(self):
            return np.arange(40 * 80, dtype=np.uint8).reshape((40, 80))

    monkeypatch.setattr(assets, "retrieve_frame", lambda frame_id, context: FakeFrame())
    client, _, _ = make_client()

    matrix_response = client.get("/assets/asset-1/framedata/2?format=matrix&scale=0.25")
    png_response = client.get("/assets/asset-1/framedata/2?format=png&scale=0.25")

    assert matrix_response.status_code == 200
    assert matrix_response.json()["shape"] == [10, 20]
    assert matrix_response.json()["scale"] == 0.25
    assert png_response.status_code == 200
    assert png_response.headers["x-pelagia-scale"] == "0.25"
    decoded = cv2.imdecode(np.frombuffer(png_response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (10, 20)


def test_api_framedata_accepts_flatfield_options(monkeypatch):
    from Pelagia.api.routes import assets
    from Pelagia.processing.frame_correction import flatfield_correction

    data = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    class FakeFrame:
        bkg = None

        def read(self):
            return data

    monkeypatch.setattr(assets, "retrieve_frame", lambda frame_id, context: FakeFrame())
    client, _, _ = make_client()

    matrix_response = client.get(
        "/assets/asset-1/framedata/2"
        "?format=matrix&flatfield_correction=true&flatfield_q=0.5&flatfield_axis=0"
        "&flatfield_min_field_value=2&flatfield_max_field_value=100"
    )
    png_response = client.get(
        "/assets/asset-1/framedata/2"
        "?format=png&flatfield_correction=true&flatfield_q=0.5&flatfield_axis=0"
        "&flatfield_min_field_value=2&flatfield_max_field_value=100"
    )

    expected = flatfield_correction(
        data,
        q=0.5,
        axis=0,
        min_field_value=2,
        max_field_value=100,
    )
    assert matrix_response.status_code == 200
    assert matrix_response.json()["flatfield_correction"] is True
    assert matrix_response.json()["flatfield_q"] == 0.5
    assert matrix_response.json()["flatfield_axis"] == 0
    assert matrix_response.json()["flatfield_min_field_value"] == 2.0
    assert matrix_response.json()["flatfield_max_field_value"] == 100.0
    assert matrix_response.json()["background_correction"] is False
    assert matrix_response.json()["data"] == expected.tolist()
    assert png_response.status_code == 200
    assert png_response.headers["x-pelagia-flatfield-correction"] == "true"
    assert png_response.headers["x-pelagia-flatfield-q"] == "0.5"
    assert png_response.headers["x-pelagia-flatfield-axis"] == "0"
    assert png_response.headers["x-pelagia-flatfield-min-field-value"] == "2.0"
    assert png_response.headers["x-pelagia-flatfield-max-field-value"] == "100.0"
    assert png_response.headers["x-pelagia-background-correction"] == "false"


def test_api_framedata_accepts_background_correction_options(monkeypatch):
    from Pelagia.api.routes import assets
    from Pelagia.processing.frame_correction import divide_background

    class FakeFrame:
        bkg = np.array([[10, 20], [10, 20]], dtype=np.uint8)

        def read(self):
            return np.array([[10, 40], [30, 80]], dtype=np.uint8)

    monkeypatch.setattr(assets, "retrieve_frame", lambda frame_id, context: FakeFrame())
    client, _, _ = make_client()

    response = client.get(
        "/assets/asset-1/framedata/2"
        "?format=matrix&background_correction=true"
        "&background_min_field_value=2&background_max_field_value=100"
    )

    expected = divide_background(
        FakeFrame().read(),
        background=FakeFrame.bkg,
        min_field_value=2,
        max_field_value=100,
    )
    assert response.status_code == 200
    assert response.json()["background_correction"] is True
    assert response.json()["background_method"] == "divide"
    assert response.json()["background_min_field_value"] == 2.0
    assert response.json()["background_max_field_value"] == 100.0
    assert response.json()["data"] == expected.tolist()


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


def test_api_queues_video_ingestion(tmp_path, monkeypatch):
    from Pelagia.api.routes import ingestion as ingestion_route

    def fail_if_called(path):
        raise AssertionError("queueing should not compute a full-file checksum by default")

    monkeypatch.setattr(ingestion_route, "_sha256_file", fail_if_called)
    client, repository, _ = make_client()
    headers = auth_headers(client)
    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"not-a-real-video")

    response = client.post(
        "/ingestion/videos",
        headers=headers,
        json={
            "source_path": str(video_path),
            "n_tile": 2,
            "enqueue_segment": True,
            "roi_padding": 4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"]
    assert body["run_id"]
    assert body["checksum_status"] == "deferred"
    assert body["job"]["stage"] == "extract_frames"
    assert repository.registered_runs[0].manifest.assets[0].path == str(video_path.resolve())
    assert repository.registered_runs[0].manifest.assets[0].collections == ["none"]
    assert repository.registered_runs[0].manifest.assets[0].metadata["checksum_status"] == "deferred"
    assert repository.created_jobs[0]["payload"]["checksum_status"] == "deferred"
    assert repository.created_jobs[0]["payload"]["enqueue_segment"] is True
    assert "flatfield_correction" not in repository.created_jobs[0]["payload"]
    assert "flatfield_q" not in repository.created_jobs[0]["payload"]
    assert "flatfield_axis" not in repository.created_jobs[0]["payload"]
    assert "flatfield_maximum_value" not in repository.created_jobs[0]["payload"]
    assert repository.created_jobs[0]["payload"]["adaptive_background_subtraction"] is False
    assert repository.created_jobs[0]["payload"]["adaptive_background_period"] == 50
    assert repository.created_jobs[0]["payload"]["apply_mask"] is False
    assert repository.created_jobs[0]["payload"]["mask_path"] is None
    assert repository.created_jobs[0]["project_id"] == "project-1"


def test_api_queues_video_ingestion_can_compute_checksum(tmp_path, monkeypatch):
    from Pelagia.api.routes import ingestion as ingestion_route

    calls = []

    def fake_sha256(path):
        calls.append(path)
        return "digest"

    monkeypatch.setattr(ingestion_route, "_sha256_file", fake_sha256)
    client, repository, _ = make_client()
    headers = auth_headers(client)
    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"not-a-real-video")

    response = client.post(
        "/ingestion/videos",
        headers=headers,
        json={
            "source_path": str(video_path),
            "compute_checksum": True,
        },
    )

    assert response.status_code == 200
    assert calls == [video_path.resolve()]
    assert response.json()["checksum_status"] == "computed"
    asset = repository.registered_runs[0].manifest.assets[0]
    assert asset.checksum == "sha256:digest"
    assert asset.metadata["checksum_status"] == "computed"
    assert repository.created_jobs[0]["payload"]["checksum_status"] == "computed"


def test_api_queues_video_ingestion_with_collections(tmp_path):
    client, repository, _ = make_client()
    headers = auth_headers(client)
    video_path = tmp_path / "sample.avi"
    video_path.write_bytes(b"not-a-real-video")

    response = client.post(
        "/ingestion/videos",
        headers=headers,
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


def test_live_preprocess_writes_to_sandbox_frame(monkeypatch):
    client, repository, kvstore = make_client()
    repository.preprocessed_payload_ref = "old-preprocessed-key"
    calls = []

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original"):
        calls.append(("retrieve", frame_id, payload_kind))
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=2,
            data=np.zeros((2, 2), dtype=np.uint8),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    def fake_preprocess_frame(frame, **kwargs):
        calls.append(("preprocess", kwargs.get("context")))
        return frame

    def fake_store_preprocessed_frame(frame_id, frame, **kwargs):
        calls.append(("store", frame_id, kwargs.get("encoding")))
        repository.sandbox_frames[frame_id]["preprocessed_payload_ref"] = "new-preprocessed-key"
        repository.sandbox_frames[frame_id]["preprocessed_kvstore_hash"] = "new-preprocessed-key"
        repository.sandbox_frames[frame_id]["preprocessed_preview_thumbhash"] = b"def"
        return {
            "id": frame_id,
            "asset_id": "asset-1",
            "frame_index": -1,
            "preprocessed_payload_ref": "new-preprocessed-key",
            "preprocessed_kvstore_hash": "new-preprocessed-key",
            "preprocessed_preview_thumbhash": b"def",
        }

    monkeypatch.setattr("Pelagia.api.routes.live.retrieve_frame", fake_retrieve_frame)
    monkeypatch.setattr("Pelagia.api.routes.live.preprocess_frame_for_segmentation", fake_preprocess_frame)
    monkeypatch.setattr("Pelagia.api.routes.live.store_preprocessed_frame", fake_store_preprocessed_frame)

    response = client.post("/live/preprocess?frame_id=frame-1&encoding=png")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stored"
    assert body["saved"] is True
    assert body["sandboxed"] is True
    assert body["sandbox_created"] is True
    assert body["source_frame_id"] == "frame-1"
    assert body["sandbox_frame_id"] == "live-frame-1"
    assert body["frame_id"] == "live-frame-1"
    assert body["old_preprocessed_key"] is None
    assert body["new_preprocessed_key"] == "new-preprocessed-key"
    assert body["old_preprocessed_deleted"] is False
    assert body["old_preprocessed_missing"] is False
    assert kvstore.deleted_keys == []
    assert calls[0] == ("retrieve", "live-frame-1", "original")
    assert calls[2] == ("store", "live-frame-1", "png")

    list_response = client.get("/live/sandbox", params={"source_frame_id": "frame-1"})

    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert list_response.json()["sandbox_frames"][0]["id"] == "live-frame-1"
    assert list_response.json()["sandbox_frames"][0]["metadata"]["live_preview"]["operation"] == "preprocess"

    delete_response = client.delete("/live/sandbox/live-frame-1")

    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"
    assert delete_response.json()["deleted_kvstore_keys"][0]["key"] == "new-preprocessed-key"
    assert kvstore.deleted_keys == ["new-preprocessed-key"]
    assert repository.deleted_sandbox_frames == ["live-frame-1"]


def test_live_preprocess_can_generate_background_for_sandbox(monkeypatch):
    client, repository, _ = make_client()
    calls = []

    def fake_retrieve_frame(frame_id, context=None, payload_kind="original"):
        calls.append(("retrieve", frame_id, payload_kind))
        return FrameData(
            sourcePath="/tmp",
            filename="frame.png",
            frameNumber=2,
            data=np.full((2, 2), 10, dtype=np.uint8),
            metadata={"frame_id": frame_id, "run_id": "run-1", "asset_id": "asset-1"},
        )

    def fake_build_background_payload_for_frames(frame_ids, **kwargs):
        calls.append(("background", frame_ids, kwargs.get("payload_kind"), kwargs.get("encoding")))
        return {
            "background_payload_ref": "background-key",
            "background_payload_encoding": "raw",
            "background_payload_format": "raw_ndarray_c_order",
            "background_payload_dtype": "float32",
            "background_payload_shape": [2, 2],
            "frame_ids": frame_ids,
            "frame_count": len(frame_ids),
            "updated_frame_count": len(frame_ids),
        }

    def fake_preprocess_frame(frame, **kwargs):
        calls.append(("preprocess", kwargs.get("background_correction"), kwargs.get("background_min_field_value")))
        return frame

    def fake_store_preprocessed_frame(frame_id, frame, **kwargs):
        calls.append(("store", frame_id, kwargs.get("encoding")))
        repository.sandbox_frames[frame_id]["preprocessed_payload_ref"] = "new-preprocessed-key"
        repository.sandbox_frames[frame_id]["preprocessed_kvstore_hash"] = "new-preprocessed-key"
        return {
            "id": frame_id,
            "asset_id": "asset-1",
            "frame_index": -1,
            "preprocessed_payload_ref": "new-preprocessed-key",
            "preprocessed_kvstore_hash": "new-preprocessed-key",
        }

    monkeypatch.setattr("Pelagia.api.routes.live.retrieve_frame", fake_retrieve_frame)
    monkeypatch.setattr("Pelagia.api.routes.live.build_background_payload_for_frames", fake_build_background_payload_for_frames)
    monkeypatch.setattr("Pelagia.api.routes.live.preprocess_frame_for_segmentation", fake_preprocess_frame)
    monkeypatch.setattr("Pelagia.api.routes.live.store_preprocessed_frame", fake_store_preprocessed_frame)

    response = client.post(
        "/live/preprocess"
        "?frame_id=frame-1&asset_id=asset-1&start_frame=2&end_frame=5&limit=1"
        "&background_encoding=raw&background_min_field_value=2"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["background_generation"]["background_payload_ref"] == "background-key"
    assert body["background_generation"]["asset_id"] == "asset-1"
    assert body["background_generation"]["start_frame"] == 2
    assert body["background_generation"]["end_frame"] == 5
    assert body["background_generation"]["limit"] == 1
    assert repository.sandbox_frames["live-frame-1"]["background_payload_ref"] == "background-key"
    assert repository.sandbox_frames["live-frame-1"]["background_payload_shape"] == [2, 2]
    assert calls[0] == ("background", ["frame-1"], "original", "raw")
    assert calls[2] == ("preprocess", True, 2.0)


def test_live_preprocess_uuid_frame_requires_auth_with_cors(monkeypatch):
    import uuid

    client, _, _ = make_client(auth_enabled=True)
    frame_id = str(uuid.uuid4())

    response = client.post(
        f"/live/preprocess?frame_id={frame_id}&encoding=png&background_correction=false"
        "&flatfield_correction=true&flatfield_q=0.95&flatfield_axis=0"
        "&flatfield_min_field_value=1&flatfield_max_field_value=255"
        "&apply_mask=false&crop_enabled=false&invert_intensity=true",
        headers={"Origin": "http://127.0.0.1:5173"},
    )

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


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
        "/assets?collection=test&kind=video&filename=sample&limit=5&offset=2"
    )
    runs_response = client.get(
        "/runs?collection=test&instrument=api&source_type=video&status=registered&limit=7&offset=14"
    )
    models_response = client.get("/models?task=classification&model_key=demo&limit=3&offset=9")
    workers_response = client.get("/workers?status=idle&capability=extract_frames&limit=4&offset=8")

    asset = assets_response.json()["assets"][0]
    assert asset["collection"] == "test"
    assert asset["kind"] == "video"
    assert asset["filename"] == "sample"
    assert asset["limit"] == 5
    assert asset["offset"] == 2
    run = runs_response.json()["runs"][0]
    assert run["collection"] == "test"
    assert run["instrument"] == "api"
    assert run["source_type"] == "video"
    assert run["status"] == "registered"
    assert run["limit"] == 7
    assert run["offset"] == 14
    model = models_response.json()["models"][0]
    assert model["task"] == "classification"
    assert model["model_key"] == "demo"
    assert model["limit"] == 3
    assert model["offset"] == 9
    worker = workers_response.json()["workers"][0]
    assert worker["status"] == "idle"
    assert worker["capability"] == "extract_frames"
    assert worker["limit"] == 4
    assert worker["offset"] == 8
