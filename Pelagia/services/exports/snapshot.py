"""Resolve immutable export inputs and provide advisory size estimates.

The snapshot deliberately records immutable membership and fingerprints, rather
than allowing a queued worker to re-evaluate curation filters against changed
data.  Product writers re-check those fingerprints before publication.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from collections.abc import Callable
from typing import Any, Mapping

from ...services import registry_generation
from ...utils.serialization import json_ready
from .contracts import ExportProduct
from .telemetry import _selected_sources


def _sha256(value: bytes | None) -> str | None:
    return None if value is None else hashlib.sha256(value).hexdigest()


def prepare_export_snapshot(
    repository: Any, *, project_id: str, request: Mapping[str, Any], config: Any | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve membership, estimate work, and return a persistable snapshot.

    Counts and byte/file estimates are informational only.  An export is never
    rejected because of its size; product writers partition format-constrained
    outputs, such as XLSX worksheets, as necessary.
    """
    products = {str(value) for value in request.get("products") or ()}
    selection = dict(request.get("filters") or {})
    selection.update({
        "asset_ids": list(request.get("asset_ids") or ()),
        "run_ids": list(request.get("run_ids") or ()),
        "source_ids": list(request.get("telemetry_source_ids") or ()),
        "roi_stage": str(request.get("roi_stage") or "refined"),
    })
    roi_items: list[dict[str, Any]] = []
    if products & {ExportProduct.RAW_ROI_STATISTICS.value, ExportProduct.BINNED_ROI_STATISTICS.value, ExportProduct.ROI_EVIDENCE.value}:
        for ordinal, row in enumerate(registry_generation._selected_rows(repository, project_id, selection, 1), 1):
            payload = row.get("roi_payload")
            payload_bytes = bytes(payload) if payload is not None else None
            roi_items.append({
                "roi_id": str(row["id"]), "asset_id": str(row["source_asset_id"]),
                "frame_id": str(row["frame_id"]),
                "candidate_roi_id": None if row.get("candidate_detection_id") is None else str(row["candidate_detection_id"]),
                "roi_payload_sha256": _sha256(payload_bytes),
                "roi_payload_bytes": len(payload_bytes or b""),
                "source_asset_checksum": row.get("source_asset_checksum"),
                "evidence_id": None if row.get("evidence_id") is None else str(row["evidence_id"]),
                "inference_run_id": None if row.get("inference_run_id") is None else str(row["inference_run_id"]),
                "annotation_id": None if row.get("annotation_id") is None else str(row["annotation_id"]),
                "review_id": None if row.get("review_id") is None else str(row["review_id"]),
            })
            if progress_callback is not None and ordinal % 1_000 == 0:
                progress_callback(ordinal, f"Freezing ROI input {ordinal:,}")

    telemetry_items: list[dict[str, Any]] = []
    if ExportProduct.TELEMETRY.value in products:
        telemetry = getattr(repository, "telemetry", repository)
        sources = _selected_sources(
            telemetry, project_id=project_id,
            source_ids={str(value) for value in selection["source_ids"] if value},
            run_ids={str(value) for value in selection["run_ids"] if value},
        )
        for ordinal, source in enumerate(sources, 1):
            telemetry_items.append({
                "source_id": str(source["id"]), "run_id": str(source["run_id"]),
                "raw_asset_id": None if source.get("raw_asset_id") is None else str(source["raw_asset_id"]),
                "source_checksum": source.get("checksum"), "source_size_bytes": int(source.get("size_bytes") or 0),
                "observation_count": int(source.get("observation_count") or 0),
                "parser_name": source.get("parser_name"), "parser_version": source.get("parser_version"),
            })
            if progress_callback is not None and ordinal % 1_000 == 0:
                progress_callback(len(roi_items) + ordinal, f"Freezing telemetry input {ordinal:,}")

    if progress_callback is not None:
        progress_callback(
            len(roi_items) + len(telemetry_items),
            f"Frozen {len(roi_items):,} ROI and {len(telemetry_items):,} telemetry inputs",
        )

    roi_payload_bytes = sum(item["roi_payload_bytes"] for item in roi_items)
    telemetry_input_bytes = sum(item["source_size_bytes"] for item in telemetry_items)
    asset_ids = {item["asset_id"] for item in roi_items}
    asset_ids.update(str(value) for value in request.get("asset_ids") or () if value)
    # Images are normally close to their source payload size; sidecars and ZIP
    # metadata need a conservative fixed allowance per ROI/source.
    estimated_files = (
        len(asset_ids) * (1 if ExportProduct.RAW_ROI_STATISTICS.value in products else 0)
        + (1 if ExportProduct.BINNED_ROI_STATISTICS.value in products else 0)
        + len(roi_items) * (2 if ExportProduct.ROI_EVIDENCE.value in products else 0)
        + len(telemetry_items) * 3 + 8
    )
    estimated_bundle_bytes = roi_payload_bytes + telemetry_input_bytes + len(roi_items) * 4096 + len(telemetry_items) * 8192 + len(asset_ids) * 16384
    counts = {
        "asset_count": len(asset_ids), "roi_count": len(roi_items),
        "telemetry_source_count": len(telemetry_items),
        "input_bytes": roi_payload_bytes + telemetry_input_bytes,
        "estimated_bundle_bytes": estimated_bundle_bytes, "estimated_file_count": estimated_files,
    }
    snapshot = json_ready({
        "schema_version": "1.0", "captured_at": datetime.now(UTC).isoformat(),
        "selection": selection, "roi_items": roi_items, "telemetry_items": telemetry_items,
    })
    return snapshot, {
        "schema_version": "1.0", "counts": counts,
        "advisory": True,
        "message": "Estimates are informational; export size does not restrict submission or publication.",
    }


def assert_snapshot_matches(repository: Any, *, project_id: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Re-resolve frozen identifiers and fail closed if an input was altered."""
    selection = dict(snapshot.get("selection") or {})
    roi_items = list(snapshot.get("roi_items") or ())
    if roi_items:
        selection["roi_ids"] = [item["roi_id"] for item in roi_items]
        actual = {str(row["id"]): row for row in registry_generation._selected_rows(repository, project_id, selection, 1)}
        for item in roi_items:
            row = actual.get(str(item["roi_id"]))
            if row is None:
                raise ValueError(f"Frozen ROI input {item['roi_id']} is unavailable.")
            payload = row.get("roi_payload")
            if _sha256(None if payload is None else bytes(payload)) != item.get("roi_payload_sha256"):
                raise ValueError(f"Frozen ROI input {item['roi_id']} changed after export submission.")
            if row.get("source_asset_checksum") != item.get("source_asset_checksum"):
                raise ValueError(f"Source asset for frozen ROI {item['roi_id']} changed after export submission.")
            for key in ("evidence_id", "inference_run_id", "annotation_id", "review_id"):
                expected = item.get(key)
                current = row.get(key)
                if (None if current is None else str(current)) != expected:
                    raise ValueError(f"Frozen ROI input {item['roi_id']} has different {key} after export submission.")
    telemetry_items = list(snapshot.get("telemetry_items") or ())
    if telemetry_items:
        telemetry = getattr(repository, "telemetry", repository)
        source_ids = {str(item["source_id"]) for item in telemetry_items}
        actual = {str(row["id"]): row for row in _selected_sources(telemetry, project_id=project_id, source_ids=source_ids, run_ids=set())}
        for item in telemetry_items:
            row = actual.get(str(item["source_id"]))
            if row is None or row.get("checksum") != item.get("source_checksum"):
                raise ValueError(f"Frozen telemetry source {item['source_id']} changed or is unavailable.")
    return selection
