"""Bounded, restartable application of a project's image storage policy."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..processing.frame_codec import decode_array_payload, encode_array_payload
from ..processing.thumbhash import compute_thumbhash
from ..services.context import AppContext
from ..workers.progress import JobProgressReporter


RESOURCES = ("frames", "detection_candidate", "detections_refined")
FRAME_VARIANTS = {
    "original": ("payload_ref", "kvstore_hash", "payload_encoding", "payload_format", "payload_dtype", "payload_shape", "metadata"),
    "preprocessed": ("preprocessed_payload_ref", "preprocessed_kvstore_hash", "preprocessed_payload_encoding", "preprocessed_payload_format", "preprocessed_payload_dtype", "preprocessed_payload_shape", "preprocessed_metadata"),
    "background": ("background_payload_ref", "background_kvstore_hash", "background_payload_encoding", "background_payload_format", "background_payload_dtype", "background_payload_shape", "background_metadata"),
}
LOSSLESS = {"raw", "zstd", "png"}


def _policy_fingerprint(policy: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _decode(payload: bytes, *, encoding: str | None, format: str | None,
            dtype: str | None, shape: list[int] | None) -> np.ndarray:
    return decode_array_payload(bytes(payload), {
        "kvstore_encoding": encoding or format or "png",
        "kvstore_format": format,
        "dtype": dtype,
        "shape": shape,
    })


def _encode_verified(array: np.ndarray, encoding: str, quality: int) -> tuple[bytes, str, str]:
    payload, stored_encoding, stored_format = encode_array_payload(array, encoding, quality=quality)
    decoded = _decode(payload, encoding=stored_encoding, format=stored_format,
                      dtype=str(array.dtype), shape=list(array.shape))
    if decoded.shape != array.shape or decoded.dtype != array.dtype:
        raise ValueError(
            f"Re-encoded {encoding} array changed shape or dtype: "
            f"{array.shape}/{array.dtype} to {decoded.shape}/{decoded.dtype}."
        )
    if encoding in LOSSLESS and decoded.tobytes(order="C") != np.ascontiguousarray(array).tobytes(order="C"):
        raise ValueError(f"Lossless {encoding} re-encoding changed pixel values.")
    return payload, stored_encoding, stored_format


def _reencode_frame(
    repository, kvstore, *, project_id: str, frame_id: str, policy: dict[str, Any],
    fingerprint: str, job_id: str, thumbhash_max_dim: int,
) -> tuple[int, int, int]:
    rewritten = before_bytes = after_bytes = 0
    for variant, columns in FRAME_VARIANTS.items():
        row = repository.get_storage_reencode_row(
            project_id=project_id, resource="frames", resource_id=frame_id,
        )
        if row is None:
            break
        ref_col, hash_col, enc_col, fmt_col, dtype_col, shape_col, meta_col = columns
        old_ref = row.get(ref_col) or row.get(hash_col)
        if not old_ref:
            continue
        metadata = dict(row.get(meta_col) or {})
        marker = metadata.get("storage_reencode") or {}
        if marker.get("fingerprint") == fingerprint and marker.get("ref") == str(old_ref):
            continue
        current_encoding = row.get(enc_col) or metadata.get("kvstore_encoding")
        if (current_encoding == policy["frame_encoding"]
                and (current_encoding in LOSSLESS
                     or metadata.get("kvstore_quality") == int(policy["frame_quality"]))):
            continue
        old_payload = kvstore.get_store(str(old_ref))
        array = _decode(old_payload, encoding=row.get(enc_col) or metadata.get("kvstore_encoding"),
                        format=row.get(fmt_col) or metadata.get("kvstore_format"),
                        dtype=row.get(dtype_col) or metadata.get("dtype"),
                        shape=row.get(shape_col) or metadata.get("shape"))
        new_payload, encoding, stored_format = _encode_verified(
            array, str(policy["frame_encoding"]), int(policy["frame_quality"]),
        )
        new_ref = kvstore.put_store(new_payload)
        history = list(metadata.get("storage_reencode_history") or [])
        history.append({
            "job_id": job_id, "at": datetime.now(timezone.utc).isoformat(),
            "previous_ref": str(old_ref), "previous_encoding": row.get(enc_col) or metadata.get("kvstore_encoding"),
            "new_ref": new_ref, "new_encoding": encoding,
        })
        metadata.update({
            "kvstore_key": new_ref, "kvstore_hash": new_ref,
            "kvstore_encoding": encoding, "kvstore_format": stored_format,
            "kvstore_quality": int(policy["frame_quality"]),
            "dtype": str(array.dtype), "shape": list(array.shape),
            "storage_reencode_history": history,
            "storage_reencode": {"fingerprint": fingerprint, "job_id": job_id, "ref": new_ref},
        })
        preview = None if variant == "background" else compute_thumbhash(array, max_dim=thumbhash_max_dim)
        if repository.update_storage_reencode_frame(
            project_id=project_id, frame_id=frame_id, row_version=str(row["row_version"]),
            variant=variant, old_ref=str(old_ref), new_ref=new_ref,
            encoding=encoding, format=stored_format, dtype=str(array.dtype),
            shape=list(array.shape), metadata=metadata, preview_thumbhash=preview,
        ):
            rewritten += 1
            before_bytes += len(old_payload)
            after_bytes += len(new_payload)
    return rewritten, before_bytes, after_bytes


def _reencode_detection(
    repository, *, project_id: str, resource: str, resource_id: str,
    policy: dict[str, Any], fingerprint: str, job_id: str,
) -> tuple[int, int, int]:
    row = repository.get_storage_reencode_row(
        project_id=project_id, resource=resource, resource_id=resource_id,
    )
    if row is None:
        return 0, 0, 0
    metadata = dict(row.get("metadata") or {})
    old_roi = row.get("roi_payload")
    old_mask = row.get("mask_payload")
    if old_roi is None and old_mask is None:
        return 0, 0, 0
    marker = metadata.get("storage_reencode") or {}
    if (marker.get("fingerprint") == fingerprint
            and marker.get("roi_sha256") == (None if old_roi is None else hashlib.sha256(bytes(old_roi)).hexdigest())
            and marker.get("mask_sha256") == (None if old_mask is None else hashlib.sha256(bytes(old_mask)).hexdigest())):
        return 0, 0, 0
    if old_roi is None and row.get("mask_encoding") == policy["mask_encoding"]:
        return 0, 0, 0
    if old_roi is not None and resource == "detection_candidate":
        shape = row.get("roi_shape") or []
        pixels = int(shape[0] * shape[1]) if len(shape) >= 2 else None
        target = None if pixels is None else (
            policy["large_roi_encoding"] if pixels >= int(policy["large_roi_min_pixels"])
            else policy["small_roi_encoding"]
        )
        recorded = metadata.get("roi_storage_policy") or {}
        if (row.get("roi_encoding") == target
                and (target in LOSSLESS or recorded.get("quality") == int(policy["roi_quality"]))
                and (old_mask is None or row.get("mask_encoding") == policy["mask_encoding"])):
            return 0, 0, 0
    roi_payload, roi_encoding, roi_format = old_roi, row.get("roi_encoding"), row.get("roi_format")
    mask_payload, mask_encoding, mask_format = old_mask, row.get("mask_encoding"), row.get("mask_format")
    if old_roi is not None:
        roi = _decode(old_roi, encoding=roi_encoding, format=roi_format,
                      dtype=row.get("roi_dtype"), shape=row.get("roi_shape"))
        pixels = int(roi.shape[0] * roi.shape[1])
        target = (policy["large_roi_encoding"] if pixels >= int(policy["large_roi_min_pixels"])
                  else policy["small_roi_encoding"])
        roi_payload, roi_encoding, roi_format = _encode_verified(
            roi, str(target), int(policy["roi_quality"]),
        )
    if old_mask is not None:
        mask = _decode(old_mask, encoding=mask_encoding, format=mask_format,
                       dtype=row.get("mask_dtype"), shape=row.get("mask_shape"))
        mask_payload, mask_encoding, mask_format = _encode_verified(
            mask, str(policy["mask_encoding"]), int(policy["roi_quality"]),
        )
    history = list(metadata.get("storage_reencode_history") or [])
    history.append({
        "job_id": job_id, "at": datetime.now(timezone.utc).isoformat(),
        "previous_roi_sha256": None if old_roi is None else hashlib.sha256(bytes(old_roi)).hexdigest(),
        "previous_mask_sha256": None if old_mask is None else hashlib.sha256(bytes(old_mask)).hexdigest(),
        "previous_roi_encoding": row.get("roi_encoding"),
        "previous_mask_encoding": row.get("mask_encoding"),
    })
    metadata["storage_reencode_history"] = history
    metadata["storage_reencode"] = {
        "fingerprint": fingerprint, "job_id": job_id,
        "roi_sha256": None if roi_payload is None else hashlib.sha256(bytes(roi_payload)).hexdigest(),
        "mask_sha256": None if mask_payload is None else hashlib.sha256(bytes(mask_payload)).hexdigest(),
    }
    metadata["roi_storage_policy"] = {
        "selected_encoding": roi_encoding,
        "pixel_count": None if old_roi is None else pixels,
        "large_min_pixels": int(policy["large_roi_min_pixels"]),
        "quality": int(policy["roi_quality"]),
        "mask_encoding": mask_encoding,
    }
    updated = repository.update_storage_reencode_detection(
        project_id=project_id, resource=resource, resource_id=resource_id,
        row_version=str(row["row_version"]), roi_payload=roi_payload,
        roi_encoding=roi_encoding, roi_format=roi_format, mask_payload=mask_payload,
        mask_encoding=mask_encoding, mask_format=mask_format, metadata=metadata,
    )
    if not updated:
        return 0, 0, 0
    before = len(old_roi or b"") + len(old_mask or b"")
    after = len(roi_payload or b"") + len(mask_payload or b"")
    return int(old_roi is not None) + int(old_mask is not None), before, after


def reencode_project_storage(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    repository = context.repository
    project_id = str(job.get("project_id") or "")
    if repository is None or not project_id:
        raise RuntimeError("Storage re-encoding requires a project-scoped repository.")
    kvstore = context.kvstore_for_project(project_id)
    if kvstore is None:
        raise RuntimeError("Storage re-encoding requires the project's KVStore.")
    payload = dict(job.get("payload") or {})
    policy = dict(payload["policy"])
    boundary = str(payload["created_before"])
    fingerprint = _policy_fingerprint(policy)
    job_id = str(job["id"])
    previous = dict((job.get("progress") or {}).get("secondary") or {})
    checkpoint = dict(previous.get("storage_checkpoint") or {})
    resource = str(checkpoint.get("resource") or RESOURCES[0])
    if resource not in RESOURCES:
        raise ValueError("Storage re-encoding checkpoint has an unknown resource.")
    completed = int(previous.get("examined") or 0)
    rewritten = int(previous.get("rewritten") or 0)
    before_bytes = int(previous.get("before_bytes") or 0)
    after_bytes = int(previous.get("after_bytes") or 0)
    reporter = JobProgressReporter(job, context, stage="storage_reencode", unit="resources",
                                   total=None, total_known=False, emit_every=1)

    if checkpoint.get("complete"):
        reporter.finish(completed=completed, secondary=previous,
                        message=f"Updated {rewritten:,} payloads across {completed:,} resources")
        return {"project_id": project_id, "policy": policy, "examined": completed,
                "rewritten": rewritten, "before_bytes": before_bytes,
                "after_bytes": after_bytes}

    def publish(message: str) -> None:
        reporter.update(
            completed,
            current={"phase": resource, "resource": resource},
            secondary={"storage_checkpoint": checkpoint, "examined": completed,
                       "rewritten": rewritten, "before_bytes": before_bytes,
                       "after_bytes": after_bytes},
            message=message, force=True,
        )

    publish("Scanning project datastores")
    for index in range(RESOURCES.index(resource), len(RESOURCES)):
        resource = RESOURCES[index]
        after_at = checkpoint.get("after_created_at") if checkpoint.get("resource") == resource else None
        after_id = checkpoint.get("after_id") if checkpoint.get("resource") == resource else None
        while True:
            reporter.checkpoint()
            rows = repository.list_storage_reencode_page(
                project_id=project_id, resource=resource, created_before=boundary,
                after_created_at=after_at, after_id=after_id, limit=100,
            )
            if not rows:
                break
            for row in rows:
                reporter.checkpoint()
                resource_id = str(row["id"])
                if resource == "frames":
                    changed, old_size, new_size = _reencode_frame(
                        repository, kvstore, project_id=project_id, frame_id=resource_id,
                        policy=policy, fingerprint=fingerprint, job_id=job_id,
                        thumbhash_max_dim=context.config.processing.thumbhash.max_dim,
                    )
                else:
                    changed, old_size, new_size = _reencode_detection(
                        repository, project_id=project_id, resource=resource,
                        resource_id=resource_id, policy=policy,
                        fingerprint=fingerprint, job_id=job_id,
                    )
                completed += 1
                rewritten += changed
                before_bytes += old_size
                after_bytes += new_size
                after_at, after_id = row["created_at"].isoformat(), resource_id
            checkpoint = {"resource": resource, "after_created_at": after_at, "after_id": after_id}
            publish(f"Examined {completed:,} project resources; updating {resource}")
        checkpoint = {"resource": RESOURCES[index + 1]} if index + 1 < len(RESOURCES) else {"resource": resource, "complete": True}
        publish(f"Finished {resource}")
    reporter.finish(completed=completed,
                    secondary={"examined": completed, "rewritten": rewritten,
                               "before_bytes": before_bytes, "after_bytes": after_bytes,
                               "storage_checkpoint": checkpoint},
                    message=f"Updated {rewritten:,} payloads across {completed:,} resources")
    return {"project_id": project_id, "policy": policy, "examined": completed,
            "rewritten": rewritten, "before_bytes": before_bytes,
            "after_bytes": after_bytes}
