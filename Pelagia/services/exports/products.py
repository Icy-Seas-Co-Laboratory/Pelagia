"""Orchestrate typed export products into one verified bundle.

This module is deliberately the only bridge between worker lifecycle and the
product writers.  It contains no HTTP parsing and accepts the persisted,
normalized request from an ``export_artifacts`` row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from .bundle import ExportBundleWriter
from .contracts import ExportProduct
from .snapshot import assert_snapshot_matches
from .roi import write_binned_roi_statistics, write_raw_roi_statistics, write_roi_evidence
from .telemetry import write_telemetry_bundle


def estimate_export_work_units(artifact: Mapping[str, Any]) -> int:
    """Return stable, coarse work units from the export's frozen input snapshot.

    Product writers have different natural granularities (assets, ROIs, and
    telemetry sources).  The estimate is deliberately a progress aid, not a
    promise about elapsed time or output size.  One final unit reserves visible
    space for manifest construction and atomic ZIP publication.
    """
    request = dict(artifact.get("request") or {})
    products = {str(value) for value in request.get("products") or ()}
    snapshot = dict(artifact.get("input_snapshot") or {})
    roi_items = list(snapshot.get("roi_items") or ())
    telemetry_items = list(snapshot.get("telemetry_items") or ())
    asset_ids = {str(item.get("asset_id")) for item in roi_items if item.get("asset_id")}
    units = 1  # Finalize and publish the bundle.
    if ExportProduct.RAW_ROI_STATISTICS.value in products:
        units += max(1, len(roi_items) * 2)  # Read rows, then write analysis files.
    if ExportProduct.BINNED_ROI_STATISTICS.value in products:
        units += max(1, len(roi_items) * 2)
    if ExportProduct.ROI_EVIDENCE.value in products:
        units += max(1, len(roi_items) * 2)
    if ExportProduct.TELEMETRY.value in products:
        units += max(1, len(telemetry_items))
    return units


def export_artifact_path(context: Any, *, project_id: str, export_id: str) -> Path:
    """Return a config-owned, UUID-only durable artifact path.

    Exports are not stored in a project KVStore: they are independently
    downloadable release files and must not affect immutable source blobs.
    """
    project = str(UUID(str(project_id)))
    export = str(UUID(str(export_id)))
    root = Path(context.config.artifacts.local_root).expanduser().resolve(strict=False)
    return root / "exports" / project / f"{export}.zip"


def build_export_bundle(*, job: Mapping[str, Any], context: Any, artifact: Mapping[str, Any], reporter: Any, on_product_event: Any = None) -> dict[str, Any]:
    """Write requested products into an atomic UUID-rooted ZIP archive."""
    project_id = str(job["project_id"])
    export_id = str(artifact["id"])
    request = dict(artifact.get("request") or {})
    requested = [str(value) for value in request.get("products") or ()]
    if not requested:
        raise ValueError("Export artifact has no requested products.")
    try:
        products = [ExportProduct(value) for value in requested]
    except ValueError as exc:
        raise ValueError("Export artifact includes an unsupported product.") from exc
    destination = export_artifact_path(context, project_id=project_id, export_id=export_id)
    snapshot = dict(artifact.get("input_snapshot") or {})
    selection = assert_snapshot_matches(context.repository, project_id=project_id, snapshot=snapshot) if snapshot else _selection(request)
    formats = dict(request.get("formats") or {})
    project_store = context.kvstore_for_project(project_id)
    product_results: dict[str, Any] = {}
    log_entries: list[dict[str, Any]] = []
    total = len(products)
    product_units = {
        product: _product_work_units(product, snapshot)
        for product in products
    }
    completed_work = 0

    with ExportBundleWriter(export_id, destination) as bundle:
        for completed, product in enumerate(products, start=1):
            reporter.checkpoint()
            file_format = _format_for(product, formats)
            work_units = product_units[product]
            reporter.update(
                completed_work,
                current={"product": product.value, "phase": "writing", "estimated": True},
                secondary={"product_completed": 0, "product_total": work_units},
                message=f"Writing {product.value}",
            )

            def product_progress(*args: Any, **kwargs: Any) -> None:
                product_completed, product_total, message = _progress_values(*args, **kwargs)
                # The work denominator was frozen before queueing. Clamp writer
                # callbacks so a buggy or partial callback cannot regress or
                # overstate the job-level estimate.
                local_completed = min(work_units, max(0, product_completed))
                reporter.update(
                    completed_work + local_completed,
                    current={"product": product.value, "phase": "writing", "estimated": True},
                    secondary={"product_completed": local_completed, "product_total": max(1, product_total or work_units)},
                    message=message or f"Writing {product.value}",
                )
            if callable(on_product_event):
                on_product_event(product.value, "working", {"ordinal": completed, "total": total})
            try:
                result = _write_product(
                    product, repository=context.repository, project_id=project_id,
                    selection=selection, output_root=bundle.root, file_format=file_format,
                    kvstore=project_store, progress_callback=product_progress,
                )
            except Exception as exc:
                if callable(on_product_event):
                    on_product_event(product.value, "failed", {"error": str(exc)[:4000]})
                raise
            product_results[product.value] = result
            if callable(on_product_event):
                on_product_event(product.value, "succeeded", result)
            log_entries.append({
                "timestamp": datetime.now(UTC).isoformat(), "level": "info",
                "stage": "product_complete", "product": product.value,
                "counts": {key: value for key, value in result.items() if key.endswith("_count")},
            })
            completed_work += work_units
            reporter.update(
                completed_work,
                current={"product": product.value, "phase": "complete", "estimated": True},
                secondary={"product_completed": work_units, "product_total": work_units},
                message=f"Wrote {product.value}",
            )
        reporter.update(
            estimate_export_work_units(artifact) - 1,
            current={"phase": "finalizing", "estimated": True},
            message="Finalizing and publishing export bundle",
        )

        def finalize_progress(phase: str, completed: int, total: int) -> None:
            reporter.update(
                estimate_export_work_units(artifact) - 1,
                current={"phase": phase, "estimated": True},
                secondary={"finalization_completed": completed, "finalization_total": total},
                message=phase.replace("_", " ").capitalize(), force=True,
            )

        return bundle.finalize(
            project_id=project_id,
            request={**request, "input_snapshot": snapshot, "preflight_estimate": artifact.get("estimate")},
            products=product_results,
            versions={"product_schema_versions": {key: value.get("schema_version") for key, value in product_results.items()}},
            readme=_readme(product_results), data_dictionary=_data_dictionary(product_results),
            log_entries=log_entries, snapshot_at=datetime.now(UTC), progress_callback=finalize_progress,
        )


def _product_work_units(product: ExportProduct, snapshot: Mapping[str, Any]) -> int:
    roi_items = list(snapshot.get("roi_items") or ())
    telemetry_items = list(snapshot.get("telemetry_items") or ())
    asset_ids = {str(item.get("asset_id")) for item in roi_items if item.get("asset_id")}
    if product in {ExportProduct.RAW_ROI_STATISTICS, ExportProduct.BINNED_ROI_STATISTICS}:
        return max(1, len(roi_items) * 2)
    if product is ExportProduct.ROI_EVIDENCE:
        return max(1, len(roi_items) * 2)
    if product is ExportProduct.TELEMETRY:
        return max(1, len(telemetry_items))
    raise AssertionError(f"Unhandled export product {product!r}")


def _progress_values(*args: Any, **kwargs: Any) -> tuple[int, int, str | None]:
    """Normalize ROI's positional and telemetry's mapping progress callbacks."""
    if args and isinstance(args[0], Mapping):
        payload = args[0]
        return int(payload.get("completed") or 0), int(payload.get("total") or 0), None
    completed = int(args[0] if args else kwargs.get("completed") or 0)
    total = int(args[1] if len(args) > 1 else kwargs.get("total") or 0)
    message = args[2] if len(args) > 2 else kwargs.get("message")
    return completed, total, None if message is None else str(message)


def _selection(request: Mapping[str, Any]) -> dict[str, Any]:
    selection = dict(request.get("filters") or {})
    selection["asset_ids"] = list(request.get("asset_ids") or ())
    selection["run_ids"] = list(request.get("run_ids") or ())
    selection["source_ids"] = list(request.get("telemetry_source_ids") or ())
    selection["roi_stage"] = str(request.get("roi_stage") or "refined")
    return selection


def _format_for(product: ExportProduct, formats: Mapping[str, Any]) -> str:
    aliases = {
        ExportProduct.RAW_ROI_STATISTICS: ("raw_roi_statistics", "roi_statistics"),
        ExportProduct.BINNED_ROI_STATISTICS: ("binned_roi_statistics", "roi_statistics_binned"),
        ExportProduct.ROI_EVIDENCE: ("roi_evidence",),
        ExportProduct.TELEMETRY: ("telemetry",),
    }[product]
    for key in aliases:
        if key in formats:
            return str(formats[key]).lower()
    # JSON is the portable default.  Evidence images are always PNG plus JSON
    # sidecars; its format setting is retained only for contract consistency.
    return "json"


def _write_product(product: ExportProduct, **kwargs: Any) -> dict[str, Any]:
    if product is ExportProduct.RAW_ROI_STATISTICS:
        return write_raw_roi_statistics(**kwargs)
    if product is ExportProduct.BINNED_ROI_STATISTICS:
        return write_binned_roi_statistics(**kwargs)
    if product is ExportProduct.ROI_EVIDENCE:
        return write_roi_evidence(**kwargs)
    if product is ExportProduct.TELEMETRY:
        return write_telemetry_bundle(**kwargs)
    raise AssertionError(f"Unhandled export product {product!r}")


def _readme(products: Mapping[str, Any]) -> str:
    names = ", ".join(sorted(products))
    return (
        "# Pelagia export bundle\n\n"
        f"This bundle contains: {names}. `manifest.json` is the authoritative index; "
        "`checksums.sha256` verifies every indexed file. UUID directory names are "
        "stable Pelagia object identifiers and may be used to link products.\n"
    )


def _data_dictionary(products: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "description": "Product-specific field definitions are stored with their analysis files; this index identifies included products.",
        "products": {name: {"schema_version": result.get("schema_version")} for name, result in products.items()},
    }
