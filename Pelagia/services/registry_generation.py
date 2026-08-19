from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, Mapping

from oracle_data_contracts.datasets import initialize_database, read_dataset_info, validate_database

from ..processing.frame_codec import decode_array_payload, encode_array_payload
from .registry_transfer import load_sqlite_workspace


class RegistryGenerationError(ValueError):
    pass


def _json(value: Any) -> dict[str, Any] | list[Any]:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _portable_image(row: Mapping[str, Any]) -> tuple[bytes, str, str, list[Any], str | None]:
    payload = bytes(row["roi_payload"])
    encoding = str(row.get("roi_encoding") or row.get("roi_format") or "bin").lower()
    shape = list(row.get("roi_shape") or [])
    dtype = row.get("roi_dtype")
    if encoding in {"png", "image/png"}:
        return payload, "png", "image/png", shape, dtype
    if encoding in {"jpg", "jpeg", "image/jpeg"}:
        return payload, "jpg", "image/jpeg", shape, dtype
    array = decode_array_payload(payload, {
        "kvstore_encoding": row.get("roi_encoding"), "kvstore_format": row.get("roi_format"),
        "dtype": dtype, "shape": shape,
    })
    encoded, portable_encoding, _ = encode_array_payload(array, "png")
    return encoded, portable_encoding, "image/png", list(array.shape), str(array.dtype)


def _selection_sql(repository, project_id: str, selection: Mapping[str, Any]) -> tuple[str, list[Any]]:
    schema = repository.schema
    clauses = ["assets.project_id = %s", "refined.roi_payload IS NOT NULL"]
    params: list[Any] = [project_id]

    asset_ids = [str(value) for value in selection.get("asset_ids") or () if value]
    if asset_ids:
        clauses.append("assets.id = ANY(%s::uuid[])")
        params.append(list(dict.fromkeys(asset_ids)))

    annotation_state = str(selection.get("annotation_state") or "all")
    if annotation_state == "labeled":
        clauses.append("annotation.id IS NOT NULL AND annotation.status <> 'deprecated'")
    elif annotation_state == "unlabeled":
        clauses.append("(annotation.id IS NULL OR annotation.status = 'deprecated')")

    review_state = str(selection.get("review_state") or "all")
    if review_state == "unreviewed":
        clauses.append("annotation.id IS NOT NULL AND review.id IS NULL")
    elif review_state in {"verified", "rejected", "needs_review"}:
        clauses.append("review.decision = %s")
        params.append(review_state)

    evidence_state = str(selection.get("evidence_state") or "all")
    if evidence_state == "available":
        clauses.append("evidence.id IS NOT NULL")
    elif evidence_state == "missing":
        clauses.append("evidence.id IS NULL")
    elif evidence_state == "disagreement":
        clauses.append(
            "evidence.id IS NOT NULL AND ((evidence.prototype_class_index IS NOT NULL AND "
            "evidence.prototype_class_index <> evidence.predicted_class_index) OR "
            "(evidence.knn_class_index IS NOT NULL AND evidence.knn_class_index <> evidence.predicted_class_index))"
        )

    if selection.get("min_area") is not None:
        clauses.append("refined.area >= %s")
        params.append(float(selection["min_area"]))
    if selection.get("max_area") is not None:
        clauses.append("refined.area <= %s")
        params.append(float(selection["max_area"]))

    joins = f"""
        LEFT JOIN LATERAL (
            SELECT * FROM {schema}.roi_label_annotations value
            WHERE value.refined_detection_id = refined.id AND value.is_current
            ORDER BY value.created_at DESC, value.id DESC LIMIT 1
        ) annotation ON true
        LEFT JOIN LATERAL (
            SELECT * FROM {schema}.roi_annotation_reviews value
            WHERE value.annotation_id = annotation.id
            ORDER BY value.created_at DESC, value.id DESC LIMIT 1
        ) review ON true
        LEFT JOIN LATERAL (
            SELECT value.*, runs.model_selector, runs.status AS inference_status,
                   runs.parameters AS inference_parameters, runs.metadata AS inference_metadata,
                   runs.created_at AS inference_created_at, runs.completed_at AS inference_completed_at,
                   artifacts.artifact_id, artifacts.run_id AS model_run_id,
                   artifacts.artifact_fingerprint
            FROM {schema}.classification_evidence value
            JOIN {schema}.classification_inference_runs runs ON runs.id = value.inference_run_id
            LEFT JOIN {schema}.model_artifacts artifacts ON artifacts.id = runs.model_artifact_id
            WHERE value.refined_detection_id = refined.id
            ORDER BY value.created_at DESC, value.id DESC LIMIT 1
        ) evidence ON true
    """
    return f"{joins} WHERE {' AND '.join(clauses)}", params


def preview_registry_dataset(
    repository,
    *,
    project_id: str,
    selection: Mapping[str, Any],
    subsample_ratio: int,
) -> dict[str, int]:
    ratio = max(1, min(1000, int(subsample_ratio)))
    selection_sql, params = _selection_sql(repository, project_id, selection)
    schema = repository.schema
    with repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                WITH matched AS (
                    SELECT octet_length(refined.roi_payload) AS payload_bytes,
                           row_number() OVER (
                               ORDER BY assets.id, frames.frame_index, refined.roi_index, refined.id
                           ) AS selection_ordinal
                    FROM {schema}.detections_refined refined
                    JOIN {schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {schema}.raw_assets assets ON assets.id = frames.asset_id
                    {selection_sql}
                )
                SELECT count(*) AS matching_count,
                       count(*) FILTER (WHERE selection_ordinal %% %s = 0) AS selected_count,
                       coalesce(sum(payload_bytes) FILTER (WHERE selection_ordinal %% %s = 0), 0) AS payload_bytes
                FROM matched
                """,
                (*params, ratio, ratio),
            )
            row = cursor.fetchone() or {}
    selected_count = int(row.get("selected_count") or 0)
    payload_bytes = int(row.get("payload_bytes") or 0)
    return {
        "matching_count": int(row.get("matching_count") or 0),
        "selected_count": selected_count,
        "payload_bytes": payload_bytes,
        "estimated_sqlite_bytes": payload_bytes + selected_count * 2048 + 1024 * 1024,
        "subsample_ratio": ratio,
    }


def _selected_rows(repository, project_id: str, selection: Mapping[str, Any], ratio: int) -> Iterator[dict[str, Any]]:
    selection_sql, params = _selection_sql(repository, project_id, selection)
    schema = repository.schema
    with repository.connect() as connection:
        with connection.cursor(name=f"registry_export_{uuid.uuid4().hex}") as cursor:
            cursor.execute(
                f"""
                WITH matched AS (
                    SELECT refined.*, frames.frame_index, assets.id AS source_asset_id,
                           assets.filename AS source_asset_filename,
                           annotation.id AS annotation_id, annotation.label_id,
                           annotation.actor_username, annotation.method AS annotation_method,
                           annotation.status AS annotation_status,
                           annotation.parent_annotation_id, annotation.notes AS annotation_notes,
                           annotation.metadata AS annotation_metadata,
                           annotation.created_at AS annotation_created_at,
                           review.id AS review_id, review.reviewer_username, review.decision AS review_decision,
                           review.notes AS review_notes, review.metadata AS review_metadata,
                           review.created_at AS review_created_at,
                           evidence.id AS evidence_id, evidence.inference_run_id,
                           evidence.predicted_label_id, evidence.predicted_label_name,
                           evidence.confidence, evidence.prototype_similarity,
                           evidence.knn_agreement, evidence.knn_weighted_support,
                           evidence.probability_margin, evidence.evidence_packet,
                           evidence.probabilities, evidence.created_at AS evidence_created_at,
                           evidence.model_selector, evidence.inference_status,
                           evidence.inference_parameters, evidence.inference_metadata,
                           evidence.inference_created_at, evidence.inference_completed_at,
                           evidence.artifact_id, evidence.model_run_id, evidence.artifact_fingerprint,
                           row_number() OVER (
                               ORDER BY assets.id, frames.frame_index, refined.roi_index, refined.id
                           ) AS selection_ordinal
                    FROM {schema}.detections_refined refined
                    JOIN {schema}.frames frames ON frames.id = refined.frame_id
                    JOIN {schema}.raw_assets assets ON assets.id = frames.asset_id
                    {selection_sql}
                )
                SELECT * FROM matched
                WHERE selection_ordinal %% %s = 0
                ORDER BY selection_ordinal
                """,
                (*params, ratio),
            )
            while batch := cursor.fetchmany(128):
                yield from batch


def _labels(repository, project_id: str) -> list[dict[str, Any]]:
    with repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM {repository.schema}.classification_labels WHERE project_id=%s ORDER BY name,id",
                (project_id,),
            )
            return list(cursor.fetchall())


def generate_and_load_registry_dataset(
    repository,
    destination_path: str | Path,
    *,
    project_id: str,
    owner_username: str,
    name: str,
    selection: Mapping[str, Any],
    subsample_ratio: int,
    dataset_id: str,
    revision_id: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    ratio = max(1, min(1000, int(subsample_ratio)))
    if selection.get("min_area") is not None and selection.get("max_area") is not None:
        if float(selection["min_area"]) > float(selection["max_area"]):
            raise RegistryGenerationError("Minimum ROI area cannot exceed maximum ROI area")
    destination = Path(destination_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            with sqlite3.connect(destination) as existing:
                existing_info = read_dataset_info(existing)
                report = validate_database(existing)
                selected_count = int(existing.execute("SELECT count(*) FROM dataset_items").fetchone()[0])
        except Exception as exc:
            raise RegistryGenerationError(
                f"Destination already exists and is not this export revision: {destination}"
            ) from exc
        if (
            existing_info["dataset_id"] != dataset_id
            or existing_info["revision_id"] != revision_id
            or not report["valid"]
        ):
            raise RegistryGenerationError(
                f"Destination already exists and is not this export revision: {destination}"
            )
        loaded = load_sqlite_workspace(
            repository, destination, project_id=project_id, owner_username=owner_username
        )
        return {
            "path": str(destination), "dataset_id": dataset_id, "revision_id": revision_id,
            "selected_count": selected_count, "subsample_ratio": ratio, "resumed": True, **loaded,
        }
    preview = preview_registry_dataset(
        repository, project_id=project_id, selection=selection, subsample_ratio=ratio
    )
    total = preview["selected_count"]
    if total < 1:
        raise RegistryGenerationError("No refined ROIs match the dataset selection")
    labels = _labels(repository, project_id)
    required_bytes = preview["payload_bytes"] + total * 4096 + 16 * 1024 * 1024
    if shutil.disk_usage(destination.parent).free < required_bytes:
        raise RegistryGenerationError(f"Dataset export requires at least {required_bytes} bytes of free space")

    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    created_at = _now()
    selection_fingerprint = hashlib.sha256(_dump({"selection": selection, "subsample_ratio": ratio}).encode()).hexdigest()
    try:
        with sqlite3.connect(temporary) as output:
            output.row_factory = sqlite3.Row
            info = initialize_database(
                output, "classification", dataset_id=dataset_id, revision_id=revision_id,
                name=name, title=name,
                description="Registry dataset generated from Pelagia Curation",
                metadata={
                    "pelagia": {
                        "project_id": project_id,
                        "generated_by": owner_username,
                        "selection": dict(selection),
                        "subsample_ratio": ratio,
                        "ordering": "asset_id, frame_index, roi_index, refined_detection_id",
                        "selection_fingerprint_sha256": selection_fingerprint,
                    }
                },
            )
            label_ids = {str(row["id"]) for row in labels}
            for label in labels:
                output.execute(
                    """INSERT INTO annotation_labels
                    (label_id,dataset_id,name,display_name,parent_label_id,rank,description,metadata_json,created_at,deprecated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (str(label["id"]), info["dataset_id"], label["name"], label["display_name"], None,
                     label["rank"], label["description"], _dump(label["metadata"] or {}),
                     str(label["created_at"]), str(label["deprecated_at"]) if label["deprecated_at"] else None),
                )
            for label in labels:
                if label["parent_label_id"]:
                    output.execute(
                        "UPDATE annotation_labels SET parent_label_id=? WHERE label_id=?",
                        (str(label["parent_label_id"]), str(label["id"])),
                    )

            inference_runs: set[str] = set()
            exported_count = 0
            for index, row in enumerate(_selected_rows(repository, project_id, selection, ratio), 1):
                exported_count = index
                item_id = str(row["id"])
                payload, encoding, media_type, portable_shape, portable_dtype = _portable_image(row)
                asset_id = str(uuid.uuid5(uuid.UUID(dataset_id), f"roi-asset:{item_id}"))
                output.execute(
                    "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (asset_id, info["dataset_id"], hashlib.sha256(payload).hexdigest(), payload, None,
                     encoding, media_type, _dump(portable_shape), portable_dtype,
                     f"{item_id}.{encoding}", _dump({"pelagia": {
                         "refined_detection_id": item_id,
                         "source_encoding": row.get("roi_encoding"),
                         "source_format": row.get("roi_format"),
                     }}), created_at),
                )
                output.execute(
                    "INSERT INTO dataset_items VALUES(?,?,?,?,?,?,?)",
                    (item_id, info["dataset_id"], None, f"pelagia:roi:{item_id}",
                     _dump({"pelagia": {"source_asset_id": str(row["source_asset_id"]),
                                           "source_asset_filename": row["source_asset_filename"],
                                           "frame_id": str(row["frame_id"]), "frame_index": row["frame_index"],
                                           "roi_index": row["roi_index"], "area": row["area"],
                                           "selection_ordinal": row["selection_ordinal"]}}),
                     str(row["created_at"]), created_at),
                )
                output.execute("INSERT INTO classification_items VALUES(?,?)", (item_id, asset_id))

                annotation_id = row.get("annotation_id")
                if annotation_id and str(row.get("label_id")) in label_ids:
                    output.execute(
                        "INSERT INTO item_label_annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (str(annotation_id), item_id, str(row["label_id"]), str(row["annotation_created_at"]),
                         row["actor_username"], row["annotation_method"] or "human", None,
                         row["annotation_status"], 1, None, "{}", row["annotation_notes"],
                         _dump({**(_json(row["annotation_metadata"]) if isinstance(_json(row["annotation_metadata"]), dict) else {}),
                                "pelagia_parent_annotation_id": str(row["parent_annotation_id"]) if row["parent_annotation_id"] else None})),
                    )
                    if row.get("review_id"):
                        output.execute(
                            "INSERT INTO annotation_reviews VALUES(?,?,?,?,?,?,?)",
                            (str(row["review_id"]), str(annotation_id), row["reviewer_username"], row["review_decision"],
                             str(row["review_created_at"]), row["review_notes"], _dump(row["review_metadata"] or {})),
                        )

                if row.get("evidence_id"):
                    run_id = str(row["inference_run_id"])
                    if run_id not in inference_runs:
                        output.execute(
                            "INSERT INTO inference_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (run_id, info["dataset_id"], selection_fingerprint,
                             str(row["artifact_id"]) if row["artifact_id"] else None,
                             str(row["model_run_id"]) if row["model_run_id"] else None,
                             row["artifact_fingerprint"], row["model_selector"], row["inference_status"],
                             str(row["inference_created_at"]), str(row["inference_completed_at"]) if row["inference_completed_at"] else None,
                             "{}", _dump(row["inference_parameters"] or {}), "{}", _dump(row["inference_metadata"] or {})),
                        )
                        inference_runs.add(run_id)
                    raw_packet = _json(row["evidence_packet"])
                    packet = dict(raw_packet) if isinstance(raw_packet, dict) else {"value": raw_packet}
                    packet.setdefault("probabilities", _json(row["probabilities"]))
                    output.execute(
                        "INSERT INTO model_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (str(row["evidence_id"]), run_id, item_id,
                         str(row["predicted_label_id"]) if row["predicted_label_id"] and str(row["predicted_label_id"]) in label_ids else None,
                         row["confidence"], row["prototype_similarity"], row["knn_agreement"],
                         row["knn_weighted_support"], row["probability_margin"], None, None, None,
                         _dump(packet), _dump({"predicted_label_name": row["predicted_label_name"]}), str(row["evidence_created_at"])),
                    )
                if progress_callback and (index == total or index == 1 or index % 100 == 0):
                    progress_callback(index, total, f"Wrote {index:,} of {total:,} ROIs")

            if exported_count < 1:
                raise RegistryGenerationError("No refined ROIs remained when the dataset export began")
            output.execute(
                "INSERT INTO dataset_events VALUES(?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), info["dataset_id"], info["revision_id"], "pelagia_export", created_at,
                 owner_username, _dump({"selection": dict(selection), "subsample_ratio": ratio,
                                        "selected_count": exported_count, "selection_fingerprint_sha256": selection_fingerprint})),
            )
            output.commit()
            report = validate_database(output)
            if not report["valid"]:
                raise RegistryGenerationError("Generated dataset failed validation: " + "; ".join(report["errors"]))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    if progress_callback:
        progress_callback(exported_count, exported_count, "Loading generated dataset into Registry")
    loaded = load_sqlite_workspace(
        repository, destination, project_id=project_id, owner_username=owner_username
    )
    return {
        "path": str(destination), "dataset_id": dataset_id, "revision_id": revision_id,
        "selected_count": exported_count, "subsample_ratio": ratio, **loaded,
    }
