"""ROI export products.

The tables written here intentionally use an analyst-facing vocabulary.  Database
metadata is retained in a separate long-form table instead of leaking storage
keys into the principal statistics table.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from ...processing.frame_codec import decode_array_payload, encode_array_payload
from .. import registry_generation
from .xlsx import write_xlsx


class RoiExportError(ValueError):
    """An ROI export request cannot be represented faithfully."""


_CORE_COLUMNS = (
    "roi_id", "candidate_roi_id", "image_id", "frame_id", "run_id", "image_name",
    "frame_number", "capture_time", "collection", "image_width_px", "image_height_px",
    "bounding_box_x_px", "bounding_box_y_px", "bounding_box_width_px",
    "bounding_box_height_px", "bounding_box_area_px2", "object_area_px2", "perimeter_px",
    "major_axis_length_px", "minor_axis_length_px", "minimum_intensity", "mean_intensity",
    "detection_method", "refinement_method", "analysis_time", "instrument", "deployment",
    "cruise", "station", "sample_id", "depth_m",
)

# These names are deliberately scientific/user-facing.  Sources may supply one
# of several conventional spellings, but that detail is never exposed as a
# table header.
_CONTEXT_ALIASES = {
    "instrument": ("instrument", "instrument_name", "camera", "camera_name"),
    "deployment": ("deployment", "deployment_id"),
    "cruise": ("cruise", "cruise_id"),
    "station": ("station", "station_id"),
    "sample_id": ("sample_id", "sample", "sample_name"),
    "depth_m": ("depth_m", "depth", "depth_metres", "depth_meters"),
}
_EXPORTED_METADATA_KEYS = {"detection_method", *(
    key for aliases in _CONTEXT_ALIASES.values() for key in aliases
)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date, UUID)):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _first_metadata_value(metadatas: Iterable[Mapping[str, Any]], aliases: tuple[str, ...]) -> Any:
    for metadata in metadatas:
        for alias in aliases:
            if metadata.get(alias) is not None:
                return metadata[alias]
    return None


def statistics_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a refined-ROI row into the stable public statistics schema."""
    asset_metadata = _mapping(row.get("source_asset_metadata", row.get("asset_metadata")))
    frame_metadata = _mapping(row.get("frame_metadata"))
    roi_metadata = _mapping(row.get("metadata"))
    contexts = (roi_metadata, frame_metadata, asset_metadata)
    width = row.get("bbox_w")
    height = row.get("bbox_h")
    result = {
        "roi_id": row.get("id"), "candidate_roi_id": row.get("candidate_detection_id"),
        "image_id": row.get("source_asset_id", row.get("asset_id")), "frame_id": row.get("frame_id"),
        "run_id": row.get("run_id"), "image_name": row.get("source_asset_filename", row.get("asset_filename")),
        "frame_number": row.get("frame_index"), "capture_time": row.get("frame_captured_at", row.get("captured_at")),
        "collection": "; ".join(map(str, row.get("source_asset_collections", row.get("collections", ())) or ())),
        "image_width_px": row.get("frame_width"), "image_height_px": row.get("frame_height"),
        "bounding_box_x_px": row.get("bbox_x"), "bounding_box_y_px": row.get("bbox_y"),
        "bounding_box_width_px": width, "bounding_box_height_px": height,
        "bounding_box_area_px2": (width * height) if width is not None and height is not None else None,
        "object_area_px2": row.get("area"), "perimeter_px": row.get("perimeter"),
        "major_axis_length_px": row.get("major_axis_length"), "minor_axis_length_px": row.get("minor_axis_length"),
        "minimum_intensity": row.get("min_gray_value"), "mean_intensity": row.get("mean_gray_value"),
        "detection_method": roi_metadata.get("detection_method"),
        "refinement_method": row.get("refinement_method"), "analysis_time": row.get("created_at"),
    }
    result.update({name: _first_metadata_value(contexts, aliases) for name, aliases in _CONTEXT_ALIASES.items()})
    return {key: _jsonable(value) for key, value in result.items()}


def additional_metadata_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Preserve available metadata in a simple long-form analyst table."""
    records = (("image", row.get("source_asset_id", row.get("asset_id")), _mapping(row.get("source_asset_metadata", row.get("asset_metadata")))),
               ("frame", row.get("frame_id"), _mapping(row.get("frame_metadata"))),
               ("roi", row.get("id"), _mapping(row.get("metadata"))))
    output = []
    for record_type, record_id, metadata in records:
        for field_name in sorted(metadata):
            value = metadata[field_name]
            output.append({"record_type": record_type, "record_id": str(record_id) if record_id else None,
                           "field_name": str(field_name),
                           "value": json.dumps(_jsonable(value), sort_keys=True) if isinstance(value, (dict, list)) else _jsonable(value),
                           "unit": metadata.get(f"{field_name}_unit")})
    return output


def export_metadata_tables(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return non-core metadata once per asset, frame, or ROI.

    Asset and frame values are intentionally not repeated for every ROI.  The
    UUID in each table is the join key back to ``roi_statistics``.
    """
    tables: dict[str, list[dict[str, Any]]] = {
        "asset_metadata": [], "frame_metadata": [], "roi_metadata": [],
    }
    seen: set[tuple[str, str]] = set()
    records = (
        ("asset_metadata", "image_id", "source_asset_id", "source_asset_metadata"),
        ("frame_metadata", "frame_id", "frame_id", "frame_metadata"),
        ("roi_metadata", "roi_id", "id", "metadata"),
    )
    for row in rows:
        for table_name, identifier_name, source_key, metadata_key in records:
            identifier = row.get(source_key)
            if identifier is None or (table_name, str(identifier)) in seen:
                continue
            seen.add((table_name, str(identifier)))
            metadata = _mapping(row.get(metadata_key))
            for field_name in sorted(metadata):
                if field_name in _EXPORTED_METADATA_KEYS or field_name.endswith("_unit"):
                    continue
                value = metadata[field_name]
                tables[table_name].append({
                    identifier_name: str(identifier), "field_name": str(field_name),
                    "value": json.dumps(_jsonable(value), sort_keys=True) if isinstance(value, (dict, list)) else _jsonable(value),
                    "unit": metadata.get(f"{field_name}_unit"),
                })
    return tables


def bbox_area_bins(maximum: float) -> list[tuple[int, int | None]]:
    """Return [lower, upper) bins with 10px² then 100px² then 1000px² steps."""
    bins = [(lower, lower + 10) for lower in range(0, 100, 10)]
    bins.extend((lower, lower + 100) for lower in range(100, 1000, 100))
    lower = 1000
    while lower <= maximum:
        bins.append((lower, lower + 1000))
        lower += 1000
    return bins


def binned_statistics(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rows by requested bounding-box area intervals."""
    normalized = [statistics_row(row) for row in rows]
    numeric = [float(row["bounding_box_area_px2"]) for row in normalized if row["bounding_box_area_px2"] is not None]
    if not numeric:
        return []
    bins = bbox_area_bins(max(numeric))
    output = []
    for lower, upper in bins:
        selected = [row for row in normalized if row["bounding_box_area_px2"] is not None and lower <= float(row["bounding_box_area_px2"]) < upper]
        if not selected:
            continue
        def mean(field: str) -> float | None:
            values = [float(value) for item in selected if (value := item.get(field)) is not None]
            return sum(values) / len(values) if values else None
        output.append({
            "area_bin_lower_px2": lower, "area_bin_upper_px2": upper,
            "roi_count": len(selected), "mean_bounding_box_area_px2": mean("bounding_box_area_px2"),
            "mean_object_area_px2": mean("object_area_px2"),
            "mean_bounding_box_width_px": mean("bounding_box_width_px"),
            "mean_bounding_box_height_px": mean("bounding_box_height_px"),
        })
    return output


def portable_roi_image(row: Mapping[str, Any]) -> tuple[bytes, str, dict[str, Any]]:
    """Return a PNG/JPEG payload suitable for extraction outside Pelagia."""
    payload = bytes(row["roi_payload"])
    encoding = str(row.get("roi_encoding") or row.get("roi_format") or "bin").lower()
    shape = list(row.get("roi_shape") or [])
    dtype = row.get("roi_dtype")
    if encoding in {"png", "image/png"}:
        return payload, "png", {"source_encoding": encoding, "output_encoding": "png", "shape": shape, "dtype": dtype}
    if encoding in {"jpg", "jpeg", "image/jpeg"}:
        return payload, "jpg", {"source_encoding": encoding, "output_encoding": "jpg", "shape": shape, "dtype": dtype}
    array = decode_array_payload(payload, {"kvstore_encoding": row.get("roi_encoding"), "kvstore_format": row.get("roi_format"), "dtype": dtype, "shape": shape})
    encoded, _, _ = encode_array_payload(array, "png")
    return encoded, "png", {"source_encoding": encoding, "output_encoding": "png", "shape": list(array.shape), "dtype": str(array.dtype)}


def _write_tables(path: Path, tables: Mapping[str, list[Mapping[str, Any]]], file_format: str) -> None:
    if file_format == "json":
        path.write_text(json.dumps({key: [_jsonable(row) for row in value] for key, value in tables.items()}, indent=2, sort_keys=True), encoding="utf-8")
    elif file_format == "sqlite":
        with sqlite3.connect(path) as db:
            for name, rows in tables.items():
                columns = list(rows[0]) if rows else _empty_table_columns(name)
                if not columns: continue
                db.execute(f'CREATE TABLE "{name}" ({", ".join(f"{column} TEXT" for column in columns)})')
                db.executemany(f'INSERT INTO "{name}" VALUES ({", ".join("?" for _ in columns)})', [tuple(_jsonable(row.get(column)) for column in columns) for row in rows])
    elif file_format == "xlsx": path.write_bytes(write_xlsx(tables))
    else: raise RoiExportError("ROI export format must be json, sqlite, or xlsx")


def _empty_table_columns(name: str) -> list[str]:
    identifiers = {
        "asset_metadata": "image_id", "frame_metadata": "frame_id", "roi_metadata": "roi_id",
    }
    identifier = identifiers.get(name)
    return [] if identifier is None else [identifier, "field_name", "value", "unit"]


_PROGRESS_BATCH_SIZE = 1_000


def _selected_rows(
    repository: Any, project_id: str, selection: Mapping[str, Any],
    progress_callback: Callable[[int, str], None] | None = None,
) -> list[dict[str, Any]]:
    stage = str(selection.get("roi_stage") or "refined")
    if stage != "refined":
        raise RoiExportError("ROI export currently supports refined ROIs only")
    rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(registry_generation._selected_rows(repository, project_id, selection, 1), 1):
        rows.append(row)
        if progress_callback is not None and ordinal % _PROGRESS_BATCH_SIZE == 0:
            progress_callback(ordinal, f"Read {ordinal:,} ROI records")
    if progress_callback is not None:
        progress_callback(len(rows), f"Read {len(rows):,} ROI records")
    return rows


def write_raw_roi_statistics(repository: Any, *, project_id: str, selection: Mapping[str, Any], output_root: Path, file_format: str = "json", kvstore: Any = None, progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    rows_by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = _selected_rows(
        repository, project_id, selection,
        None if progress_callback is None else lambda completed, message: progress_callback(completed, 0, message),
    )
    for row in rows: rows_by_asset[str(row["source_asset_id"])].append(row)
    paths = []
    suffix = {"json": "json", "sqlite": "sqlite", "xlsx": "xlsx"}.get(file_format)
    if suffix is None: raise RoiExportError("ROI export format must be json, sqlite, or xlsx")
    written = 0
    for _ordinal, (asset_id, asset_rows) in enumerate(sorted(rows_by_asset.items()), 1):
        target = output_root / "products" / "roi-statistics" / "assets" / asset_id
        target.mkdir(parents=True, exist_ok=True)
        output = target / f"statistics.{suffix}"
        _write_tables(output, {
            "roi_statistics": [statistics_row(row) for row in asset_rows],
            **export_metadata_tables(asset_rows),
        }, file_format)
        paths.append(str(output.relative_to(output_root)))
        written += len(asset_rows)
        if progress_callback: progress_callback(len(rows) + written, len(rows) * 2, f"Wrote statistics for asset {asset_id}")
    return {"product": "roi-statistics", "schema_version": "1.1", "asset_count": len(rows_by_asset), "roi_count": sum(map(len, rows_by_asset.values())), "paths": paths}


def write_binned_roi_statistics(repository: Any, *, project_id: str, selection: Mapping[str, Any], output_root: Path, file_format: str = "json", kvstore: Any = None, progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    rows_by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = _selected_rows(
        repository, project_id, selection,
        None if progress_callback is None else lambda completed, message: progress_callback(completed, 0, message),
    )
    for row in rows: rows_by_asset[str(row["source_asset_id"])].append(row)
    suffix = {"json": "json", "sqlite": "sqlite", "xlsx": "xlsx"}.get(file_format)
    if suffix is None: raise RoiExportError("ROI export format must be json, sqlite, or xlsx")
    paths = []
    written = 0
    for _ordinal, (asset_id, asset_rows) in enumerate(sorted(rows_by_asset.items()), 1):
        target = output_root / "products" / "roi-statistics-binned" / "assets" / asset_id
        target.mkdir(parents=True, exist_ok=True); output = target / f"summary.{suffix}"
        _write_tables(output, {"roi_area_bins": binned_statistics(asset_rows)}, file_format); paths.append(str(output.relative_to(output_root)))
        written += len(asset_rows)
        if progress_callback: progress_callback(len(rows) + written, len(rows) * 2, f"Wrote bins for asset {asset_id}")
    return {"product": "roi-statistics-binned", "schema_version": "1.0", "asset_count": len(rows_by_asset), "roi_count": sum(map(len, rows_by_asset.values())), "paths": paths}


def write_roi_evidence(repository: Any, *, project_id: str, selection: Mapping[str, Any], output_root: Path, file_format: str = "json", kvstore: Any = None, progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    rows = _selected_rows(
        repository, project_id, selection,
        None if progress_callback is None else lambda completed, message: progress_callback(completed, 0, message),
    ); paths = []
    for ordinal, row in enumerate(rows, 1):
        if row.get("roi_payload") is None: continue
        asset_id, frame_id, roi_id = map(str, (row["source_asset_id"], row["frame_id"], row["id"]))
        target = output_root / "products" / "roi-evidence" / "assets" / asset_id / "frames" / frame_id / "rois" / roi_id
        target.mkdir(parents=True, exist_ok=True)
        payload, extension, representation = portable_roi_image(row); image = target / f"image.{extension}"; image.write_bytes(payload)
        detail = repository.get_curation_roi(roi_id, project_id=project_id) if hasattr(repository, "get_curation_roi") else {}
        sidecar = {"roi": _jsonable(dict(row)), "statistics": statistics_row(row), "additional_metadata": additional_metadata_rows(row), "evidence": _jsonable(detail), "image": {"path": str(image.relative_to(output_root)), "sha256": hashlib.sha256(payload).hexdigest(), **representation}}
        metadata = target / "metadata.json"; metadata.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        paths.extend((str(image.relative_to(output_root)), str(metadata.relative_to(output_root))))
        if progress_callback and (ordinal % _PROGRESS_BATCH_SIZE == 0 or ordinal == len(rows)):
            progress_callback(len(rows) + ordinal, len(rows) * 2, f"Wrote evidence through ROI {ordinal:,}")
    return {"product": "roi-evidence", "schema_version": "1.0", "roi_count": len(rows), "paths": paths}
