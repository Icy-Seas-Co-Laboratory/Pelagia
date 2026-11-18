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
from typing import Any, Callable, Iterable

from oracle_data_contracts.datasets import (
    initialize_database,
    read_dataset_info,
    validate_database,
)

try:
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - PostgreSQL is an optional Pelagia extra
    Jsonb = None  # type: ignore


class RegistryTransferError(ValueError):
    pass


_REGISTRY_ASSET_BATCH_ITEMS = 250
_REGISTRY_ASSET_BATCH_BYTES = 64 * 1024 * 1024
_REGISTRY_ASSET_BATCH_ROWS = 250


def _json(value: Any, default: Any = None) -> Any:
    if value is None:
        return {} if default is None else default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {} if default is None else default


def _jsonb(value: Any):
    resolved = _json(value)
    return Jsonb(resolved) if Jsonb is not None else json.dumps(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_checkpointed_sqlite(path: Path) -> None:
    wal_path = path.with_name(path.name + "-wal")
    if wal_path.is_file() and wal_path.stat().st_size > 0:
        raise RegistryTransferError(
            "SQLite source has an active WAL; close or checkpoint the database before importing it"
        )


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _migrated_copy(source: Path) -> tuple[Path, sqlite3.Connection]:
    temporary = tempfile.NamedTemporaryFile(prefix="pelagia-registry-", suffix=".sqlite", delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with sqlite3.connect(source) as input_connection, sqlite3.connect(temporary_path) as output_connection:
            input_connection.backup(output_connection)
        connection = sqlite3.connect(temporary_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        identity = connection.execute(
            "SELECT dataset_type FROM dataset WHERE singleton=1"
        ).fetchone()
        if identity is None:
            raise RegistryTransferError("Dataset identity row is missing")
        initialize_database(connection, str(identity[0]))
        connection.commit()
        report = validate_database(connection)
        if not report["valid"]:
            raise RegistryTransferError("Dataset validation failed: " + "; ".join(report["errors"]))
        return temporary_path, connection
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _rows(connection: sqlite3.Connection, table: str) -> Iterable[sqlite3.Row]:
    if table not in _tables(connection):
        return []
    return connection.execute(f'SELECT * FROM "{table}"')


def load_sqlite_workspace(
    repository,
    source_path: str | Path,
    *,
    project_id: str,
    owner_username: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Load a validated Oracle Dataset revision into project-scoped PostgreSQL tables."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise RegistryTransferError(f"SQLite dataset does not exist: {source}")
    source_hash = _sha256(source)
    temporary_path, source_connection = _migrated_copy(source)
    try:
        info = read_dataset_info(source_connection)
        schema = repository.schema
        with repository.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"registry:{project_id}:{owner_username}",),
                )
                cursor.execute(
                    f"""SELECT * FROM {schema}.registry_workspaces
                    WHERE project_id=%s AND owner_username=%s AND dataset_id=%s AND revision_id=%s
                    FOR UPDATE""",
                    (project_id, owner_username, info["dataset_id"], info["revision_id"]),
                )
                existing = cursor.fetchone()
                if existing is not None and existing["status"] == "loaded":
                    if existing["source_sha256"] != source_hash:
                        raise RegistryTransferError(
                            "This dataset revision is already loaded from different bytes; create a new revision before loading it again."
                        )
                    cursor.execute(
                        f"UPDATE {schema}.registry_workspaces SET is_active=false WHERE project_id=%s AND owner_username=%s",
                        (project_id, owner_username),
                    )
                    cursor.execute(
                        f"UPDATE {schema}.registry_workspaces SET is_active=true,status='loaded' WHERE id=%s",
                        (existing["id"],),
                    )
                    connection.commit()
                    return {"workspace_id": str(existing["id"]), "reused": True}
                if existing is not None:
                    cursor.execute(
                        f"DELETE FROM {schema}.registry_workspaces WHERE id=%s",
                        (existing["id"],),
                    )

                cursor.execute(
                    f"UPDATE {schema}.registry_workspaces SET is_active=false WHERE project_id=%s AND owner_username=%s",
                    (project_id, owner_username),
                )
                cursor.execute(
                    f"""INSERT INTO {schema}.registry_workspaces (
                      project_id,owner_username,dataset_id,revision_id,parent_revision_id,
                      dataset_type,name,title,description,dataset_version,dataset_lifecycle,
                      contract_schema_name,contract_schema_version,source_path,source_sha256,
                      source_size_bytes,status,is_active,metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'loading',true,%s)
                    RETURNING id""",
                    (
                        project_id, owner_username, info["dataset_id"], info["revision_id"],
                        info.get("parent_revision_id"), info["dataset_type"], info["name"],
                        info.get("title"), info.get("description"), info.get("version"),
                        info["lifecycle"], info["schema_name"], info["schema_version"],
                        str(source), source_hash, source.stat().st_size, _jsonb(info.get("metadata")),
                    ),
                )
                workspace_id = cursor.fetchone()["id"]
                _load_assets(repository, cursor, schema, workspace_id, source_connection)
                if progress_callback: progress_callback(1, "Loaded assets")
                _load_items(cursor, schema, workspace_id, source_connection, info["dataset_type"])
                if progress_callback: progress_callback(2, "Loaded dataset items")
                _load_labels_and_annotations(cursor, schema, workspace_id, source_connection)
                if progress_callback: progress_callback(3, "Loaded labels and annotations")
                _load_descriptors(cursor, schema, workspace_id, source_connection)
                if progress_callback: progress_callback(4, "Loaded descriptors")
                _load_masks(cursor, schema, workspace_id, source_connection)
                if progress_callback: progress_callback(5, "Loaded mask annotations")
                _load_evidence(cursor, schema, workspace_id, source_connection)
                if progress_callback: progress_callback(6, "Loaded model evidence")
                _load_events(cursor, schema, workspace_id, source_connection)
                if progress_callback: progress_callback(7, "Loaded dataset events")
                _load_auxiliary_records(cursor, schema, workspace_id, source_connection)
                if progress_callback: progress_callback(8, "Loaded auxiliary contract records")
                cursor.execute(
                    f"UPDATE {schema}.registry_workspaces SET status='loaded' WHERE id=%s",
                    (workspace_id,),
                )
            connection.commit()
        return {"workspace_id": str(workspace_id), "reused": False}
    except Exception:
        raise
    finally:
        source_connection.close()
        temporary_path.unlink(missing_ok=True)


def _load_assets(repository, cursor, schema: str, workspace_id: Any, source: sqlite3.Connection) -> None:
    hot_store = getattr(repository, "hot_store", None)
    if hot_store is None:
        raise RegistryTransferError("Hot object storage is not configured for Registry assets")

    pending: list[tuple[sqlite3.Row, bytes | None]] = []
    pending_payload_count = 0
    pending_payload_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_payload_count, pending_payload_bytes
        if not pending:
            return

        payloads = [payload for _, payload in pending if payload is not None]
        put_many = getattr(hot_store, "put_many", None)
        if payloads and callable(put_many):
            payload_refs = list(put_many(payloads))
        else:
            payload_refs = [hot_store.put_store(payload) for payload in payloads]
        if len(payload_refs) != len(payloads):
            raise RegistryTransferError("Hot storage returned an incorrect number of Registry asset references")

        ref_iter = iter(payload_refs)
        for row, payload in pending:
            payload_ref = next(ref_iter) if payload is not None else None
            cursor.execute(
                f"""INSERT INTO {schema}.registry_assets (
                  workspace_id,asset_id,content_sha256,payload,payload_ref,payload_bytes,external_uri,encoding,media_type,
                  shape,dtype,original_filename,metadata,created_at
                ) VALUES (%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    workspace_id,row["asset_id"],row["content_sha256"],payload_ref,None if payload is None else len(payload),row["external_uri"],
                    row["encoding"],row["media_type"],_jsonb(row["shape_json"]),row["dtype"],
                    row["original_filename"],_jsonb(row["metadata_json"]),row["created_at"],
                ),
            )
        pending = []
        pending_payload_count = 0
        pending_payload_bytes = 0

    for row in _rows(source, "assets"):
        payload = None if row["payload"] is None else bytes(row["payload"])
        payload_bytes = 0 if payload is None else len(payload)
        if pending and (
            len(pending) >= _REGISTRY_ASSET_BATCH_ROWS
            or (payload is not None and pending_payload_count >= _REGISTRY_ASSET_BATCH_ITEMS)
            or (payload is not None and pending_payload_bytes + payload_bytes > _REGISTRY_ASSET_BATCH_BYTES)
        ):
            flush()
        pending.append((row, payload))
        pending_payload_count += 1 if payload is not None else 0
        pending_payload_bytes += payload_bytes
        if (
            pending_payload_count >= _REGISTRY_ASSET_BATCH_ITEMS
            or pending_payload_bytes >= _REGISTRY_ASSET_BATCH_BYTES
        ):
            flush()
    flush()


def _load_items(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection, dataset_type: str) -> None:
    relation = "classification_items" if dataset_type == "classification" else "mask_refinement_items"
    task_rows = {str(row["item_id"]): row for row in _rows(source, relation)}
    geometry_rows = {str(row["item_id"]): row for row in _rows(source, "item_geometry")}
    for ordinal, row in enumerate(_rows(source, "dataset_items")):
        task = task_rows[str(row["item_id"])]
        geometry = geometry_rows.get(str(row["item_id"]))
        cursor.execute(
            f"""INSERT INTO {schema}.registry_items (
              workspace_id,item_id,ordinal,sample_weight,source_key,image_asset_id,
              candidate_mask_asset_id,metadata,created_at,updated_at,
              coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
              crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h,spatial_metadata
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                workspace_id,row["item_id"],ordinal,row["sample_weight"],row["source_key"],
                task["image_asset_id"],task["candidate_mask_asset_id"] if dataset_type == "mask_refinement" else None,
                _jsonb(row["metadata_json"]),row["created_at"],row["updated_at"],
                geometry["coordinate_space"] if geometry else None,
                geometry["bbox_x"] if geometry else None,geometry["bbox_y"] if geometry else None,
                geometry["bbox_w"] if geometry else None,geometry["bbox_h"] if geometry else None,
                geometry["crop_bbox_x"] if geometry else None,geometry["crop_bbox_y"] if geometry else None,
                geometry["crop_bbox_w"] if geometry else None,geometry["crop_bbox_h"] if geometry else None,
                _jsonb(geometry["metadata_json"] if geometry else {}),
            ),
        )


def _load_labels_and_annotations(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection) -> None:
    tables = _tables(source)
    label_specs = [
        ("classification_labels", "classification", "classification_annotations"),
        ("annotation_labels", "workspace", "item_label_annotations"),
    ]
    for label_table, origin, annotation_table in label_specs:
        if label_table not in tables:
            continue
        for row in _rows(source, label_table):
            metadata = _json(row["metadata_json"])
            registry_metadata = metadata.get("registry", {}) if isinstance(metadata, dict) else {}
            cursor.execute(
                f"""INSERT INTO {schema}.registry_labels (
                  workspace_id,label_id,origin,class_index,name,display_name,parent_label_id,
                  rank,description,metadata,created_at,deprecated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    workspace_id,row["label_id"],origin,
                    row["class_index"] if origin == "classification" else None,row["name"],
                    registry_metadata.get("display_name") if origin == "classification" else row["display_name"],
                    row["parent_label_id"],registry_metadata.get("rank") if origin == "classification" else row["rank"],
                    registry_metadata.get("description") if origin == "classification" else row["description"],
                    _jsonb(metadata),None if origin == "classification" else row["created_at"],
                    registry_metadata.get("deprecated_at") if origin == "classification" else row["deprecated_at"],
                ),
            )
        if annotation_table in tables:
            for row in _rows(source, annotation_table):
                cursor.execute(
                    f"""INSERT INTO {schema}.registry_annotations (
                      workspace_id,annotation_id,item_id,label_id,origin,created_at,annotator,
                      method,confidence,status,is_current,parent_annotation_id,parameters,notes,metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        workspace_id,row["annotation_id"],row["item_id"],row["label_id"],origin,
                        row["created_at"],row["annotator"],row["source"] if origin == "classification" else row["method"],
                        row["confidence"],row["status"],bool(row["is_current"]),row["parent_annotation_id"],
                        _jsonb({} if origin == "classification" else row["parameters_json"]),row["notes"],
                        _jsonb(row["metadata_json"]),
                    ),
                )
        review_table = "classification_annotation_reviews" if origin == "classification" else "annotation_reviews"
        if review_table in tables:
            for row in _rows(source, review_table):
                cursor.execute(
                    f"""INSERT INTO {schema}.registry_reviews (
                      workspace_id,review_id,annotation_id,origin,reviewer,decision,created_at,notes,metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (workspace_id,row["review_id"],row["annotation_id"],origin,row["reviewer"],row["decision"],
                     row["created_at"],row["notes"],_jsonb(row["metadata_json"])),
                )


def _load_descriptors(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection) -> None:
    for row in _rows(source, "descriptor_definitions"):
        cursor.execute(
            f"""INSERT INTO {schema}.registry_descriptors (
              workspace_id,descriptor_id,scope,name,parent_descriptor_id,concept_id,concept_type,
              selectable,exclusive_within_parent,preferred,metadata,created_at,deprecated_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id,row["descriptor_id"],row["scope"],row["name"],row["parent_descriptor_id"],
             row["concept_id"],row["concept_type"],bool(row["selectable"]),bool(row["exclusive_within_parent"]),
             bool(row["preferred"]),_jsonb(row["metadata_json"]),row["created_at"],row["deprecated_at"]),
        )
    for row in _rows(source, "item_descriptor_annotations"):
        cursor.execute(
            f"""INSERT INTO {schema}.registry_descriptor_annotations (
              workspace_id,annotation_id,item_id,descriptor_id,created_at,annotator,status,
              is_current,parent_annotation_id,notes,metadata
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id,row["annotation_id"],row["item_id"],row["descriptor_id"],row["created_at"],
             row["annotator"],row["status"],bool(row["is_current"]),row["parent_annotation_id"],
             row["notes"],_jsonb(row["metadata_json"])),
        )


def _load_masks(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection) -> None:
    for row in _rows(source, "mask_annotations"):
        cursor.execute(
            f"""INSERT INTO {schema}.registry_mask_annotations (
              workspace_id,annotation_id,item_id,mask_asset_id,created_at,annotator,method,
              parameters,validation,status,is_current,parent_annotation_id,notes
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id,row["annotation_id"],row["item_id"],row["mask_asset_id"],row["created_at"],
             row["annotator"],row["method"],_jsonb(row["parameters_json"]),_jsonb(row["validation_json"]),
             row["status"],bool(row["is_current"]),row["parent_annotation_id"],row["notes"]),
        )


def _load_evidence(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection) -> None:
    for row in _rows(source, "inference_runs"):
        cursor.execute(
            f"""INSERT INTO {schema}.registry_inference_runs (
              workspace_id,inference_run_id,dataset_fingerprint_sha256,model_artifact_id,model_run_id,
              model_artifact_fingerprint_sha256,name,status,created_at,completed_at,input_contract,
              parameters,software_environment,metadata
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id,row["inference_run_id"],row["dataset_fingerprint_sha256"],row["model_artifact_id"],
             row["model_run_id"],row["model_artifact_fingerprint_sha256"],row["name"],row["status"],
             row["created_at"],row["completed_at"],_jsonb(row["input_contract_json"]),_jsonb(row["parameters_json"]),
             _jsonb(row["software_environment_json"]),_jsonb(row["metadata_json"])),
        )
    for row in _rows(source, "evidence_arrays"):
        cursor.execute(
            f"""INSERT INTO {schema}.registry_evidence_arrays (
              workspace_id,array_id,content_sha256,payload,encoding,media_type,shape,dtype,created_at,metadata
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id,row["array_id"],row["content_sha256"],row["payload"],row["encoding"],row["media_type"],
             _jsonb(row["shape_json"]),row["dtype"],row["created_at"],_jsonb(row["metadata_json"])),
        )
    for row in _rows(source, "model_evidence"):
        cursor.execute(
            f"""INSERT INTO {schema}.registry_model_evidence (
              workspace_id,evidence_id,inference_run_id,item_id,predicted_label_id,prediction_confidence,
              nearest_neighbor_similarity,top_k_label_agreement,weighted_label_support,label_margin,
              logits_array_id,embedding_array_id,output_array_id,packet,metadata,created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id,row["evidence_id"],row["inference_run_id"],row["item_id"],row["predicted_label_id"],
             row["prediction_confidence"],row["nearest_neighbor_similarity"],row["top_k_label_agreement"],
             row["weighted_label_support"],row["label_margin"],row["logits_array_id"],row["embedding_array_id"],
             row["output_array_id"],_jsonb(row["packet_json"]),_jsonb(row["metadata_json"]),row["created_at"]),
        )


def _load_events(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection) -> None:
    for row in _rows(source, "dataset_events"):
        cursor.execute(
            f"""INSERT INTO {schema}.registry_dataset_events (
              workspace_id,event_id,revision_id,event_type,created_at,actor,details
            ) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id,row["event_id"],row["revision_id"],row["event_type"],row["created_at"],
             row["actor"],_jsonb(row["details_json"])),
        )


def _load_auxiliary_records(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection) -> None:
    for relation in (
        "metadata_documents",
        "import_events",
        "taxonomy_concepts",
        "taxonomy_concept_mappings",
        "classification_label_concepts",
    ):
        for ordinal, row in enumerate(_rows(source, relation)):
            record = {key: row[key] for key in row.keys()}
            cursor.execute(
                f"INSERT INTO {schema}.registry_contract_records "
                "(workspace_id,relation,ordinal,record) VALUES (%s,%s,%s,%s)",
                (workspace_id, relation, ordinal, _jsonb(record)),
            )


def _import_id(source_sha256: str, kind: str, value: Any) -> str:
    """Return a stable, contract-compatible identifier for an imported record."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pelagia-registry-import:{source_sha256}:{kind}:{value}"))


def _source_items(source: sqlite3.Connection, dataset_type: str) -> list[dict[str, Any]]:
    relation = "classification_items" if dataset_type == "classification" else "mask_refinement_items"
    tasks = {str(row["item_id"]): row for row in _rows(source, relation)}
    assets = {str(row["asset_id"]): row for row in _rows(source, "assets")}
    result = []
    for row in _rows(source, "dataset_items"):
        task = tasks[str(row["item_id"])]
        image = assets[str(task["image_asset_id"])]
        result.append({
            "item": row,
            "task": task,
            "image_sha256": str(image["content_sha256"]),
        })
    return result


def _import_workspace(cursor, schema: str, *, project_id: str, owner_username: str,
                      workspace_id: str | None, for_update: bool) -> dict[str, Any]:
    lock = " FOR UPDATE" if for_update else ""
    if workspace_id is None:
        cursor.execute(
            f"""SELECT * FROM {schema}.registry_workspaces
            WHERE project_id=%s AND owner_username=%s AND is_active=true AND status='loaded'{lock}""",
            (project_id, owner_username),
        )
    else:
        cursor.execute(
            f"""SELECT * FROM {schema}.registry_workspaces
            WHERE id=%s AND project_id=%s AND owner_username=%s AND status='loaded'{lock}""",
            (workspace_id, project_id, owner_username),
        )
    workspace = cursor.fetchone()
    if workspace is None:
        raise RegistryTransferError("Loaded Registry workspace was not found")
    return workspace


def _import_duplicate_inventory(cursor, schema: str, workspace_id: Any,
                                source_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cursor.execute(
        f"""SELECT item.item_id,item.source_key,asset.content_sha256 image_sha256,item.metadata
        FROM {schema}.registry_items item
        JOIN {schema}.registry_assets asset
          ON asset.workspace_id=item.workspace_id AND asset.asset_id=item.image_asset_id
        WHERE item.workspace_id=%s""",
        (workspace_id,),
    )
    target_rows = list(cursor.fetchall())
    by_id = {str(row["item_id"]): row for row in target_rows}
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in target_rows:
        by_hash.setdefault(str(row["image_sha256"]), []).append(row)
    duplicates = []
    for entry in source_items:
        row = entry["item"]
        source_id = str(row["item_id"])
        matches: dict[str, dict[str, Any]] = {}
        reasons: dict[str, set[str]] = {}
        if source_id in by_id:
            matches[source_id] = by_id[source_id]
            reasons.setdefault(source_id, set()).add("uuid")
        for target in by_hash.get(entry["image_sha256"], []):
            target_id = str(target["item_id"])
            matches[target_id] = target
            reasons.setdefault(target_id, set()).add("roi_hash")
        if not matches:
            continue
        # UUID is authoritative; otherwise choose a stable first hash match and
        # report every candidate so callers can see an ambiguous content match.
        target_id = source_id if source_id in matches else sorted(matches)[0]
        source_metadata = _json(row["metadata_json"])
        target_metadata = _json(matches[target_id].get("metadata"))
        conflicts = sorted(
            key for key in source_metadata.keys() & target_metadata.keys()
            if source_metadata[key] != target_metadata[key]
        )
        all_reasons = sorted({reason for value in reasons.values() for reason in value})
        duplicates.append({
            "source_item_id": source_id,
            "source_key": row["source_key"],
            "target_item_id": target_id,
            "matching_item_ids": sorted(matches),
            "reasons": all_reasons,
            "roi_hash": entry["image_sha256"],
            "target_roi_hash": str(matches[target_id]["image_sha256"]),
            "metadata_conflicts": conflicts,
        })
    return duplicates


def preview_sqlite_import(repository, source_path: str | Path, *, project_id: str,
                          owner_username: str, workspace_id: str | None = None) -> dict[str, Any]:
    """Validate and compare an SQLite revision with a scoped Registry workspace."""
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise RegistryTransferError(f"SQLite dataset does not exist: {source_path}")
    _require_checkpointed_sqlite(source_path)
    source_sha256 = _sha256(source_path)
    temporary_path, source = _migrated_copy(source_path)
    try:
        _require_checkpointed_sqlite(source_path)
        if _sha256(source_path) != source_sha256:
            raise RegistryTransferError("Source SQLite file changed while it was being inspected")
        info = read_dataset_info(source)
        source_items = _source_items(source, info["dataset_type"])
        with repository.connect() as connection, connection.cursor() as cursor:
            workspace = _import_workspace(
                cursor, repository.schema, project_id=project_id, owner_username=owner_username,
                workspace_id=workspace_id, for_update=False,
            )
            if workspace["dataset_type"] != info["dataset_type"]:
                raise RegistryTransferError("Source and destination Registry dataset types must match")
            duplicates = _import_duplicate_inventory(cursor, repository.schema, workspace["id"], source_items)
        tables = _tables(source)
        label_annotations = sum(
            int(source.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            for table in ("classification_annotations", "item_label_annotations") if table in tables
        )
        descriptor_annotations = (
            int(source.execute("SELECT count(*) FROM item_descriptor_annotations").fetchone()[0])
            if "item_descriptor_annotations" in tables else 0
        )
        evidence = int(source.execute("SELECT count(*) FROM model_evidence").fetchone()[0]) if "model_evidence" in tables else 0
        schema_row = source.execute("SELECT schema_version FROM ob_schema WHERE singleton=1").fetchone()
        return {
            "source": {
                "path": str(source_path), "sha256": source_sha256,
                "dataset_id": info["dataset_id"], "revision_id": info["revision_id"],
                "name": info["name"], "dataset_type": info["dataset_type"],
                "lifecycle": info["lifecycle"], "schema_version": str(schema_row[0]),
                "size_bytes": source_path.stat().st_size,
            },
            "workspace_id": str(workspace["id"]),
            "total_items": len(source_items), "new_items": len(source_items) - len(duplicates),
            "duplicate_count": len(duplicates), "duplicates": duplicates,
            "labels": {"available": label_annotations > 0, "annotation_count": label_annotations},
            "descriptors": {"available": descriptor_annotations > 0, "annotation_count": descriptor_annotations},
            "evidence": {"available": evidence > 0, "registry_count": evidence, "legacy_count": 0},
        }
    finally:
        source.close()
        temporary_path.unlink(missing_ok=True)


def _put_import_assets(repository, cursor, schema: str, workspace_id: Any,
                       source: sqlite3.Connection, source_sha256: str) -> dict[str, str]:
    cursor.execute(
        f"SELECT asset_id,content_sha256,encoding,shape,dtype,media_type FROM {schema}.registry_assets WHERE workspace_id=%s",
        (workspace_id,),
    )
    existing_rows = list(cursor.fetchall())
    by_content = {(str(row["content_sha256"]), str(row["encoding"])): row for row in existing_rows}
    existing_ids = {str(row["asset_id"]) for row in existing_rows}
    result: dict[str, str] = {}
    hot_store = getattr(repository, "hot_store", None)
    if hot_store is None:
        raise RegistryTransferError("Hot object storage is not configured for Registry assets")
    for row in _rows(source, "assets"):
        source_id = str(row["asset_id"])
        content_key = (str(row["content_sha256"]), str(row["encoding"]))
        if content_key in by_content:
            existing = by_content[content_key]
            if (_json(existing.get("shape"), None) != _json(row["shape_json"], None)
                    or existing.get("dtype") != row["dtype"]
                    or existing.get("media_type") != row["media_type"]):
                raise RegistryTransferError(
                    f"Asset {source_id} has the same content hash and encoding as a destination asset but incompatible shape, dtype, or media type"
                )
            result[source_id] = str(existing["asset_id"])
            continue
        target_id = source_id if source_id not in existing_ids else _import_id(source_sha256, "asset", source_id)
        payload = None if row["payload"] is None else bytes(row["payload"])
        payload_ref = hot_store.put_store(payload) if payload is not None else None
        cursor.execute(
            f"""INSERT INTO {schema}.registry_assets
            (workspace_id,asset_id,content_sha256,payload,payload_ref,payload_bytes,external_uri,
             encoding,media_type,shape,dtype,original_filename,metadata,created_at)
            VALUES (%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id, target_id, row["content_sha256"], payload_ref,
             len(payload) if payload is not None else None, row["external_uri"], row["encoding"],
             row["media_type"], _jsonb(row["shape_json"]), row["dtype"], row["original_filename"],
             _jsonb(row["metadata_json"]), row["created_at"]),
        )
        existing_ids.add(target_id)
        by_content[content_key] = {
            "asset_id": target_id, "shape": _json(row["shape_json"], None),
            "dtype": row["dtype"], "media_type": row["media_type"],
        }
        result[source_id] = target_id
    return result


def _import_labels(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection,
                   source_sha256: str) -> tuple[dict[tuple[str, str], str], int, int]:
    cursor.execute(f"SELECT * FROM {schema}.registry_labels WHERE workspace_id=%s", (workspace_id,))
    existing = list(cursor.fetchall())
    by_name = {(row["origin"], str(row["name"])): row for row in existing}
    ids = {str(row["label_id"]) for row in existing}
    next_class = max((int(row["class_index"]) for row in existing if row["class_index"] is not None), default=-1) + 1
    mapping: dict[tuple[str, str], str] = {}
    created = conflicts = 0
    specs = (("classification_labels", "classification"), ("annotation_labels", "workspace"))
    pending = [(table, origin, row) for table, origin in specs for row in _rows(source, table)]
    # Parent references are filled after all definitions exist.
    for table, origin, row in pending:
        source_id = str(row["label_id"])
        match = by_name.get((origin, str(row["name"])))
        if match is not None:
            mapping[(origin, source_id)] = str(match["label_id"])
            incoming_metadata = _json(row["metadata_json"])
            if incoming_metadata and incoming_metadata != _json(match.get("metadata")):
                conflicts += 1
            continue
        target_id = source_id if source_id not in ids else _import_id(source_sha256, "label", source_id)
        metadata = _json(row["metadata_json"])
        registry_metadata = metadata.get("registry", {}) if isinstance(metadata, dict) else {}
        cursor.execute(
            f"""INSERT INTO {schema}.registry_labels
            (workspace_id,label_id,origin,class_index,name,display_name,parent_label_id,rank,
             description,metadata,created_at,deprecated_at)
            VALUES (%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s)""",
            (workspace_id, target_id, origin, next_class if origin == "classification" else None,
             row["name"], registry_metadata.get("display_name") if origin == "classification" else row["display_name"],
             registry_metadata.get("rank") if origin == "classification" else row["rank"],
             registry_metadata.get("description") if origin == "classification" else row["description"],
             _jsonb(metadata), None if origin == "classification" else row["created_at"],
             registry_metadata.get("deprecated_at") if origin == "classification" else row["deprecated_at"]),
        )
        if origin == "classification":
            next_class += 1
        mapping[(origin, source_id)] = target_id
        ids.add(target_id)
        created += 1
    for table, origin, row in pending:
        parent = row["parent_label_id"]
        child_key = (origin, str(row["label_id"]))
        parent_key = (origin, str(parent)) if parent is not None else None
        if parent_key is not None and child_key in mapping and parent_key in mapping:
            cursor.execute(
                f"UPDATE {schema}.registry_labels SET parent_label_id=%s WHERE workspace_id=%s AND label_id=%s AND parent_label_id IS NULL",
                (mapping[parent_key], workspace_id, mapping[child_key]),
            )
    return mapping, created, conflicts


def _import_descriptors(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection,
                        source_sha256: str) -> tuple[dict[str, str], int]:
    cursor.execute(f"SELECT * FROM {schema}.registry_descriptors WHERE workspace_id=%s", (workspace_id,))
    existing = list(cursor.fetchall())
    by_name = {(str(row["scope"]), str(row["name"])): str(row["descriptor_id"]) for row in existing}
    ids = {str(row["descriptor_id"]) for row in existing}
    rows = list(_rows(source, "descriptor_definitions"))
    mapping: dict[str, str] = {}
    created = 0
    for row in rows:
        source_id = str(row["descriptor_id"])
        key = (str(row["scope"]), str(row["name"]))
        if key in by_name:
            mapping[source_id] = by_name[key]
            continue
        target_id = source_id if source_id not in ids else _import_id(source_sha256, "descriptor", source_id)
        cursor.execute(
            f"""INSERT INTO {schema}.registry_descriptors
            (workspace_id,descriptor_id,scope,name,parent_descriptor_id,concept_id,concept_type,
             selectable,exclusive_within_parent,preferred,metadata,created_at,deprecated_at)
            VALUES (%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (workspace_id,target_id,row["scope"],row["name"],row["concept_id"],row["concept_type"],
             bool(row["selectable"]),bool(row["exclusive_within_parent"]),bool(row["preferred"]),
             _jsonb(row["metadata_json"]),row["created_at"],row["deprecated_at"]),
        )
        mapping[source_id] = target_id
        by_name[key] = target_id
        ids.add(target_id)
        created += 1
    for row in rows:
        parent = row["parent_descriptor_id"]
        if parent is not None:
            cursor.execute(
                f"UPDATE {schema}.registry_descriptors SET parent_descriptor_id=%s WHERE workspace_id=%s AND descriptor_id=%s AND parent_descriptor_id IS NULL",
                (mapping.get(str(parent)), workspace_id, mapping[str(row["descriptor_id"])]),
            )
    return mapping, created


def _import_auxiliary_records(cursor, schema: str, workspace: dict[str, Any], source: sqlite3.Connection,
                              source_sha256: str, label_map: dict[tuple[str, str], str]) -> dict[str, str]:
    """Keep portable metadata/import/taxonomy records and translate their references."""
    workspace_id = workspace["id"]
    cursor.execute(
        f"SELECT relation,ordinal,record FROM {schema}.registry_contract_records WHERE workspace_id=%s ORDER BY relation,ordinal",
        (workspace_id,),
    )
    existing = list(cursor.fetchall())
    next_ordinal: dict[str, int] = {}
    existing_concepts: dict[tuple[str, str], str] = {}
    existing_primary_keys: dict[str, set[str]] = {
        "metadata_documents": set(), "import_events": set(), "taxonomy_concepts": set(),
        "taxonomy_concept_mappings": set(), "classification_label_concepts": set(),
    }
    primary_columns = {
        "metadata_documents": "document_id", "import_events": "import_id",
        "taxonomy_concepts": "concept_id", "taxonomy_concept_mappings": "mapping_id",
        "classification_label_concepts": "label_id",
    }
    for row in existing:
        relation = str(row["relation"])
        next_ordinal[relation] = max(next_ordinal.get(relation, 0), int(row["ordinal"]) + 1)
        record = _json(row["record"])
        primary_column = primary_columns.get(relation)
        if primary_column is not None and record.get(primary_column) is not None:
            existing_primary_keys[relation].add(str(record[primary_column]))
        if relation == "taxonomy_concepts":
            existing_concepts[(str(record.get("vocabulary_id")), str(record.get("vocabulary_node_id")))] = str(record.get("concept_id"))

    concept_rows = list(_rows(source, "taxonomy_concepts"))
    concept_map: dict[str, str] = {}
    for row in concept_rows:
        source_id = str(row["concept_id"])
        stable_key = (str(row["vocabulary_id"]), str(row["vocabulary_node_id"]))
        concept_map[source_id] = existing_concepts.get(stable_key) or _import_record_id(source_sha256, "taxonomy-concept", source_id)

    relation_rows: list[tuple[str, dict[str, Any]]] = []
    for row in _rows(source, "metadata_documents"):
        record = {key: row[key] for key in row.keys()}
        record["document_id"] = _import_record_id(source_sha256, "metadata-document", row["document_id"])
        record["dataset_id"] = workspace["dataset_id"]
        # Dataset document names are unique; retain both by making imported
        # provenance explicit in the portable name.
        record["name"] = f"import:{str(source_sha256)[:12]}:{row['name']}"
        relation_rows.append(("metadata_documents", record))
    for row in _rows(source, "import_events"):
        record = {key: row[key] for key in row.keys()}
        record["import_id"] = _import_record_id(source_sha256, "source-import-event", row["import_id"])
        record["dataset_id"] = workspace["dataset_id"]
        record["revision_id"] = workspace["revision_id"]
        relation_rows.append(("import_events", record))
    for row in concept_rows:
        record = {key: row[key] for key in row.keys()}
        record["concept_id"] = concept_map[str(row["concept_id"])]
        if row["parent_concept_id"] is not None:
            record["parent_concept_id"] = concept_map[str(row["parent_concept_id"])]
        # A stable vocabulary node already present in the destination should
        # remain a single row on export.
        if (str(row["vocabulary_id"]), str(row["vocabulary_node_id"])) not in existing_concepts:
            relation_rows.append(("taxonomy_concepts", record))
    for row in _rows(source, "taxonomy_concept_mappings"):
        record = {key: row[key] for key in row.keys()}
        record["mapping_id"] = _import_record_id(source_sha256, "taxonomy-mapping", row["mapping_id"])
        record["concept_id"] = concept_map[str(row["concept_id"])]
        relation_rows.append(("taxonomy_concept_mappings", record))
    for row in _rows(source, "classification_label_concepts"):
        target_label = label_map.get(("classification", str(row["label_id"])))
        if target_label is None:
            continue
        record = {key: row[key] for key in row.keys()}
        record["label_id"] = target_label
        record["concept_id"] = concept_map[str(row["concept_id"])]
        relation_rows.append(("classification_label_concepts", record))

    for relation, record in relation_rows:
        primary_column = primary_columns[relation]
        primary_value = str(record[primary_column])
        if primary_value in existing_primary_keys[relation]:
            continue
        ordinal = next_ordinal.get(relation, 0)
        cursor.execute(
            f"INSERT INTO {schema}.registry_contract_records (workspace_id,relation,ordinal,record) VALUES (%s,%s,%s,%s)",
            (workspace_id, relation, ordinal, _jsonb(record)),
        )
        next_ordinal[relation] = ordinal + 1
        existing_primary_keys[relation].add(primary_value)
    return concept_map


def _existing_ids(cursor, schema: str, table: str, column: str, workspace_id: Any) -> set[str]:
    cursor.execute(f"SELECT {column} value FROM {schema}.{table} WHERE workspace_id=%s", (workspace_id,))
    return {str(row["value"]) for row in cursor.fetchall()}


def _import_record_id(source_sha256: str, kind: str, source_id: Any) -> str:
    # Always namespace history/evidence IDs. This avoids confusing an unrelated
    # same-text ID with a retry, and stays stable for idempotent retries.
    return _import_id(source_sha256, kind, source_id)


def _remap_json_strings(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _remap_json_strings(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_json_strings(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def import_sqlite_into_workspace(
    repository,
    source_path: str | Path,
    *,
    project_id: str,
    owner_username: str,
    include_labels: bool = True,
    include_evidence: bool = True,
    duplicate_policy: str = "skip",
    duplicate_actions: dict[str, str] | None = None,
    expected_source_sha256: str | None = None,
    workspace_id: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Merge an Oracle Dataset SQLite revision into one scoped Registry workspace.

    The PostgreSQL changes are atomic. Blob publication is content-addressed and
    may leave unreachable objects after a rollback; normal object GC can reclaim
    them without exposing partial Registry state.
    """
    if duplicate_policy not in {"skip", "merge"}:
        raise RegistryTransferError("Duplicate policy must be 'skip' or 'merge'")
    duplicate_actions = dict(duplicate_actions or {})
    if any(value not in {"skip", "merge"} for value in duplicate_actions.values()):
        raise RegistryTransferError("Duplicate actions must be 'skip' or 'merge'")
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise RegistryTransferError(f"SQLite dataset does not exist: {source_path}")
    _require_checkpointed_sqlite(source_path)
    source_sha256 = _sha256(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise RegistryTransferError("Source SQLite file changed after it was inspected")
    temporary_path, source = _migrated_copy(source_path)
    try:
        _require_checkpointed_sqlite(source_path)
        if _sha256(source_path) != source_sha256:
            raise RegistryTransferError("Source SQLite file changed while it was being prepared for import")
        info = read_dataset_info(source)
        source_items = _source_items(source, info["dataset_type"])
        options_fingerprint = hashlib.sha256(json.dumps({
            "include_labels": include_labels, "include_evidence": include_evidence,
            "duplicate_policy": duplicate_policy, "duplicate_actions": duplicate_actions,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        import_event_id = _import_id(source_sha256, "import-operation", options_fingerprint)
        schema = repository.schema
        with repository.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"registry:{project_id}:{owner_username}",),
                )
                workspace = _import_workspace(
                    cursor, schema, project_id=project_id, owner_username=owner_username,
                    workspace_id=workspace_id, for_update=True,
                )
                if workspace["dataset_type"] != info["dataset_type"]:
                    raise RegistryTransferError("Source and destination Registry dataset types must match")
                cursor.execute(
                    f"SELECT details FROM {schema}.registry_dataset_events WHERE workspace_id=%s AND event_id=%s",
                    (workspace["id"], import_event_id),
                )
                prior = cursor.fetchone()
                if prior is not None:
                    result = _json(prior["details"]).get("result")
                    if isinstance(result, dict):
                        return {**result, "reused": True}

                duplicates = _import_duplicate_inventory(cursor, schema, workspace["id"], source_items)
                duplicate_by_source = {row["source_item_id"]: row for row in duplicates}
                item_map: dict[str, str] = {}
                merged_source_items: set[str] = set()
                skipped = 0
                for entry in source_items:
                    source_id = str(entry["item"]["item_id"])
                    duplicate = duplicate_by_source.get(source_id)
                    if duplicate is None:
                        item_map[source_id] = source_id
                        continue
                    action = duplicate_actions.get(source_id, duplicate_policy)
                    item_map[source_id] = str(duplicate["target_item_id"])
                    if action == "merge":
                        if len(duplicate["matching_item_ids"]) != 1:
                            raise RegistryTransferError(
                                f"ROI {source_id} matches multiple destination items; choose skip because merge is ambiguous"
                            )
                        if duplicate["roi_hash"] != duplicate["target_roi_hash"]:
                            raise RegistryTransferError(
                                f"ROI {source_id} has the same UUID as a destination item but different image content; choose skip"
                            )
                        merged_source_items.add(source_id)
                    else:
                        skipped += 1

                asset_map = _put_import_assets(repository, cursor, schema, workspace["id"], source, source_sha256)
                cursor.execute(f"SELECT coalesce(max(ordinal),-1) value FROM {schema}.registry_items WHERE workspace_id=%s", (workspace["id"],))
                ordinal = int(cursor.fetchone()["value"]) + 1
                cursor.execute(f"SELECT source_key FROM {schema}.registry_items WHERE workspace_id=%s AND source_key IS NOT NULL", (workspace["id"],))
                source_keys = {str(row["source_key"]) for row in cursor.fetchall()}
                geometry = {str(row["item_id"]): row for row in _rows(source, "item_geometry")}
                imported_item_ids: set[str] = set()
                for entry in source_items:
                    row, task = entry["item"], entry["task"]
                    source_id = str(row["item_id"])
                    if source_id in duplicate_by_source:
                        continue
                    target_id = item_map[source_id]
                    source_key = row["source_key"]
                    if source_key is not None and str(source_key) in source_keys:
                        source_key = f"import:{info['dataset_id']}:{source_key}"
                    if source_key is not None:
                        source_keys.add(str(source_key))
                    geo = geometry.get(source_id)
                    cursor.execute(
                        f"""INSERT INTO {schema}.registry_items
                        (workspace_id,item_id,ordinal,sample_weight,source_key,image_asset_id,candidate_mask_asset_id,
                         metadata,created_at,updated_at,coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
                         crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h,spatial_metadata)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (workspace["id"],target_id,ordinal,row["sample_weight"],source_key,
                         asset_map[str(task["image_asset_id"])],
                         (asset_map.get(str(task["candidate_mask_asset_id"]))
                          if info["dataset_type"] == "mask_refinement" and task["candidate_mask_asset_id"] is not None
                          else None),
                         _jsonb(row["metadata_json"]),row["created_at"],row["updated_at"],
                         geo["coordinate_space"] if geo else None,geo["bbox_x"] if geo else None,
                         geo["bbox_y"] if geo else None,geo["bbox_w"] if geo else None,geo["bbox_h"] if geo else None,
                         geo["crop_bbox_x"] if geo else None,geo["crop_bbox_y"] if geo else None,
                         geo["crop_bbox_w"] if geo else None,geo["crop_bbox_h"] if geo else None,
                         _jsonb(geo["metadata_json"] if geo else {})),
                    )
                    imported_item_ids.add(source_id)
                    ordinal += 1

                eligible_items = imported_item_ids | merged_source_items
                for entry in source_items:
                    source_id = str(entry["item"]["item_id"])
                    if source_id not in merged_source_items:
                        continue
                    cursor.execute(
                        f"SELECT metadata FROM {schema}.registry_items WHERE workspace_id=%s AND item_id=%s",
                        (workspace["id"], item_map[source_id]),
                    )
                    target_metadata = _json(cursor.fetchone()["metadata"])
                    source_metadata = _json(entry["item"]["metadata_json"])
                    merged_metadata = {**source_metadata, **target_metadata}
                    cursor.execute(
                        f"UPDATE {schema}.registry_items SET metadata=%s WHERE workspace_id=%s AND item_id=%s",
                        (_jsonb(merged_metadata), workspace["id"], item_map[source_id]),
                    )
                label_map: dict[tuple[str, str], str] = {}
                descriptor_map: dict[str, str] = {}
                created_labels = label_conflicts = created_descriptors = 0
                # Evidence label references are part of ML provenance even when
                # human annotations/descriptors were excluded by the caller.
                if include_labels or include_evidence:
                    label_map, created_labels, label_conflicts = _import_labels(
                        cursor, schema, workspace["id"], source, source_sha256,
                    )
                concept_map = _import_auxiliary_records(
                    cursor, schema, workspace, source, source_sha256, label_map,
                )
                if include_labels:
                    descriptor_map, created_descriptors = _import_descriptors(
                        cursor, schema, workspace["id"], source, source_sha256,
                    )
                    for row in _rows(source, "descriptor_definitions"):
                        if row["concept_id"] is not None and str(row["concept_id"]) in concept_map:
                            cursor.execute(
                                f"UPDATE {schema}.registry_descriptors SET concept_id=%s WHERE workspace_id=%s AND descriptor_id=%s AND concept_id IS NULL",
                                (concept_map[str(row["concept_id"])], workspace["id"], descriptor_map[str(row["descriptor_id"])]),
                            )

                imported_annotations = 0
                annotation_map: dict[str, str] = {}
                annotation_specs = (
                    ("classification_annotations", "classification"),
                    ("item_label_annotations", "workspace"),
                )
                if include_labels:
                    for table, origin in annotation_specs:
                        rows = [row for row in _rows(source, table) if str(row["item_id"]) in eligible_items]
                        for row in rows:
                            annotation_map[str(row["annotation_id"])] = _import_record_id(source_sha256, f"{origin}-annotation", row["annotation_id"])
                        for row in rows:
                            source_item_id = str(row["item_id"])
                            target_item_id = item_map[source_item_id]
                            is_current = bool(row["is_current"])
                            if source_item_id in merged_source_items and is_current:
                                cursor.execute(
                                    f"SELECT 1 FROM {schema}.registry_annotations WHERE workspace_id=%s AND item_id=%s AND origin=%s AND is_current LIMIT 1",
                                    (workspace["id"], target_item_id, origin),
                                )
                                is_current = cursor.fetchone() is None
                            cursor.execute(
                                f"""INSERT INTO {schema}.registry_annotations
                                (workspace_id,annotation_id,item_id,label_id,origin,created_at,annotator,method,
                                 confidence,status,is_current,parent_annotation_id,parameters,notes,metadata)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                ON CONFLICT (workspace_id,annotation_id) DO NOTHING""",
                                (workspace["id"],annotation_map[str(row["annotation_id"])],target_item_id,
                                 label_map[(origin, str(row["label_id"]))],origin,row["created_at"],row["annotator"],
                                 row["source"] if origin == "classification" else row["method"],row["confidence"],
                                 row["status"],is_current,annotation_map.get(str(row["parent_annotation_id"])),
                                 _jsonb({} if origin == "classification" else row["parameters_json"]),row["notes"],
                                 _jsonb(row["metadata_json"])),
                            )
                            imported_annotations += cursor.rowcount
                        review_table = "classification_annotation_reviews" if origin == "classification" else "annotation_reviews"
                        for row in _rows(source, review_table):
                            source_annotation = str(row["annotation_id"])
                            if source_annotation not in annotation_map:
                                continue
                            cursor.execute(
                                f"""INSERT INTO {schema}.registry_reviews
                                (workspace_id,review_id,annotation_id,origin,reviewer,decision,created_at,notes,metadata)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                ON CONFLICT (workspace_id,review_id) DO NOTHING""",
                                (workspace["id"],_import_record_id(source_sha256, f"{origin}-review", row["review_id"]),
                                 annotation_map[source_annotation],origin,row["reviewer"],row["decision"],row["created_at"],
                                 row["notes"],_jsonb(row["metadata_json"])),
                            )

                descriptor_annotation_map: dict[str, str] = {}
                imported_descriptor_annotations = 0
                if include_labels:
                    descriptor_rows = [row for row in _rows(source, "item_descriptor_annotations") if str(row["item_id"]) in eligible_items]
                    for row in descriptor_rows:
                        descriptor_annotation_map[str(row["annotation_id"])] = _import_record_id(source_sha256, "descriptor-annotation", row["annotation_id"])
                    for row in descriptor_rows:
                        source_item_id = str(row["item_id"])
                        target_item_id = item_map[source_item_id]
                        target_descriptor = descriptor_map[str(row["descriptor_id"])]
                        is_current = bool(row["is_current"])
                        if source_item_id in merged_source_items and is_current:
                            cursor.execute(
                                f"SELECT 1 FROM {schema}.registry_descriptor_annotations WHERE workspace_id=%s AND item_id=%s AND descriptor_id=%s AND is_current LIMIT 1",
                                (workspace["id"],target_item_id,target_descriptor),
                            )
                            is_current = cursor.fetchone() is None
                        cursor.execute(
                            f"""INSERT INTO {schema}.registry_descriptor_annotations
                            (workspace_id,annotation_id,item_id,descriptor_id,created_at,annotator,status,is_current,
                             parent_annotation_id,notes,metadata)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (workspace_id,annotation_id) DO NOTHING""",
                            (workspace["id"],descriptor_annotation_map[str(row["annotation_id"])],target_item_id,
                             target_descriptor,row["created_at"],row["annotator"],row["status"],is_current,
                             descriptor_annotation_map.get(str(row["parent_annotation_id"])),row["notes"],
                             _jsonb(row["metadata_json"])),
                        )
                        imported_descriptor_annotations += cursor.rowcount

                mask_map: dict[str, str] = {}
                mask_rows = [row for row in _rows(source, "mask_annotations") if str(row["item_id"]) in eligible_items]
                for row in mask_rows:
                    mask_map[str(row["annotation_id"])] = _import_record_id(source_sha256, "mask-annotation", row["annotation_id"])
                for row in mask_rows:
                    source_item_id = str(row["item_id"])
                    is_current = bool(row["is_current"])
                    if source_item_id in merged_source_items and is_current:
                        cursor.execute(
                            f"SELECT 1 FROM {schema}.registry_mask_annotations WHERE workspace_id=%s AND item_id=%s AND is_current LIMIT 1",
                            (workspace["id"],item_map[source_item_id]),
                        )
                        is_current = cursor.fetchone() is None
                    cursor.execute(
                        f"""INSERT INTO {schema}.registry_mask_annotations
                        (workspace_id,annotation_id,item_id,mask_asset_id,created_at,annotator,method,parameters,
                         validation,status,is_current,parent_annotation_id,notes)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (workspace_id,annotation_id) DO NOTHING""",
                        (workspace["id"],mask_map[str(row["annotation_id"])],item_map[source_item_id],
                         asset_map[str(row["mask_asset_id"])],row["created_at"],row["annotator"],row["method"],
                         _jsonb(row["parameters_json"]),_jsonb(row["validation_json"]),row["status"],is_current,
                         mask_map.get(str(row["parent_annotation_id"])),row["notes"]),
                    )

                imported_evidence = 0
                run_map: dict[str, str] = {}
                array_map: dict[str, str] = {}
                if include_evidence:
                    for row in _rows(source, "inference_runs"):
                        target_id = _import_record_id(source_sha256, "inference-run", row["inference_run_id"])
                        run_map[str(row["inference_run_id"])] = target_id
                        cursor.execute(
                            f"""INSERT INTO {schema}.registry_inference_runs
                            (workspace_id,inference_run_id,dataset_fingerprint_sha256,model_artifact_id,model_run_id,
                             model_artifact_fingerprint_sha256,name,status,created_at,completed_at,input_contract,
                             parameters,software_environment,metadata)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (workspace_id,inference_run_id) DO NOTHING""",
                            (workspace["id"],target_id,row["dataset_fingerprint_sha256"],row["model_artifact_id"],
                             row["model_run_id"],row["model_artifact_fingerprint_sha256"],row["name"],row["status"],
                             row["created_at"],row["completed_at"],_jsonb(row["input_contract_json"]),
                             _jsonb(row["parameters_json"]),_jsonb(row["software_environment_json"]),
                             _jsonb(row["metadata_json"])),
                        )
                    cursor.execute(f"SELECT array_id,content_sha256 FROM {schema}.registry_evidence_arrays WHERE workspace_id=%s", (workspace["id"],))
                    arrays_by_hash = {str(row["content_sha256"]): str(row["array_id"]) for row in cursor.fetchall()}
                    for row in _rows(source, "evidence_arrays"):
                        source_id = str(row["array_id"])
                        target_id = arrays_by_hash.get(str(row["content_sha256"])) or _import_record_id(source_sha256, "evidence-array", source_id)
                        array_map[source_id] = target_id
                        cursor.execute(
                            f"""INSERT INTO {schema}.registry_evidence_arrays
                            (workspace_id,array_id,content_sha256,payload,encoding,media_type,shape,dtype,created_at,metadata)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING""",
                            (workspace["id"],target_id,row["content_sha256"],row["payload"],row["encoding"],
                             row["media_type"],_jsonb(row["shape_json"]),row["dtype"],row["created_at"],
                             _jsonb(row["metadata_json"])),
                        )
                    for row in _rows(source, "model_evidence"):
                        source_item_id = str(row["item_id"])
                        if source_item_id not in eligible_items:
                            continue
                        predicted_label = row["predicted_label_id"]
                        cursor.execute(
                            f"""INSERT INTO {schema}.registry_model_evidence
                            (workspace_id,evidence_id,inference_run_id,item_id,predicted_label_id,prediction_confidence,
                             nearest_neighbor_similarity,top_k_label_agreement,weighted_label_support,label_margin,
                             logits_array_id,embedding_array_id,output_array_id,packet,metadata,created_at)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING""",
                            (workspace["id"],_import_record_id(source_sha256, "model-evidence", row["evidence_id"]),
                             run_map[str(row["inference_run_id"])],item_map[source_item_id],
                             label_map.get(("workspace", str(predicted_label))) if predicted_label is not None else None,
                             row["prediction_confidence"],row["nearest_neighbor_similarity"],row["top_k_label_agreement"],
                             row["weighted_label_support"],row["label_margin"],array_map.get(str(row["logits_array_id"])),
                             array_map.get(str(row["embedding_array_id"])),array_map.get(str(row["output_array_id"])),
                             _jsonb(_remap_json_strings(_json(row["packet_json"]), item_map)),
                             _jsonb(row["metadata_json"]),row["created_at"]),
                        )
                        imported_evidence += cursor.rowcount

                for row in _rows(source, "dataset_events"):
                    cursor.execute(
                        f"""INSERT INTO {schema}.registry_dataset_events
                        (workspace_id,event_id,revision_id,event_type,created_at,actor,details)
                        VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (workspace["id"],_import_record_id(source_sha256,"source-event",row["event_id"]),
                         row["revision_id"],row["event_type"],row["created_at"],row["actor"],_jsonb(row["details_json"])),
                    )

                result = {
                    "workspace_id": str(workspace["id"]), "imported_items": len(imported_item_ids),
                    "merged_items": len(merged_source_items), "skipped_items": skipped,
                    "metadata_conflicts": sum(len(row["metadata_conflicts"]) for row in duplicates),
                    "imported_labels": imported_annotations, "label_conflicts": label_conflicts,
                    "imported_label_definitions": created_labels,
                    "imported_descriptors": imported_descriptor_annotations,
                    "imported_descriptor_definitions": created_descriptors,
                    "imported_evidence": imported_evidence, "source_path": str(source_path),
                    "source_sha256": source_sha256, "backup_path": None,
                }
                now = datetime.now(timezone.utc).isoformat()
                details = {
                    "source_dataset_id": info["dataset_id"], "source_revision_id": info["revision_id"],
                    "source_sha256": source_sha256, "options_sha256": options_fingerprint, "result": result,
                }
                cursor.execute(
                    f"""INSERT INTO {schema}.registry_dataset_events
                    (workspace_id,event_id,revision_id,event_type,created_at,actor,details)
                    VALUES (%s,%s,%s,'registry.sqlite_imported',%s,%s,%s)""",
                    (workspace["id"],import_event_id,workspace["revision_id"],now,actor or owner_username,_jsonb(details)),
                )
                cursor.execute(
                    f"UPDATE {schema}.registry_workspaces SET dirty_at=NOW() WHERE id=%s",
                    (workspace["id"],),
                )
            connection.commit()
        return result
    finally:
        source.close()
        temporary_path.unlink(missing_ok=True)


def export_sqlite_workspace(
    repository,
    workspace_id: str,
    destination_path: str | Path,
    *,
    project_id: str,
    owner_username: str,
    replace_source: bool = False,
    progress_callback: Callable[[int, str], None] | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Export one loaded workspace as a validated child SQLite revision."""
    destination = Path(destination_path).expanduser().resolve()
    schema = repository.schema
    with repository.connect() as pg_connection:
        with pg_connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"registry:{project_id}:{owner_username}",),
            )
            cursor.execute(
                f"""SELECT * FROM {schema}.registry_workspaces
                WHERE id=%s AND project_id=%s AND owner_username=%s AND status='loaded'
                FOR UPDATE""",
                (workspace_id, project_id, owner_username),
            )
            workspace = cursor.fetchone()
            if workspace is None:
                raise RegistryTransferError("Loaded Registry workspace was not found")
            if operation_id and workspace.get("last_export_operation_id") == operation_id:
                return {
                    "path": workspace["last_export_path"], "sha256": workspace["last_export_sha256"],
                    "revision_id": workspace["revision_id"], "parent_revision_id": workspace["parent_revision_id"],
                    "backup_path": None, "reused": True,
                }
            cursor.execute(
                f"UPDATE {schema}.registry_workspaces SET status='exporting' WHERE id=%s",
                (workspace_id,),
            )
            if replace_source and not workspace["source_path"]:
                raise RegistryTransferError("Blank Registry workspaces must be exported to a new SQLite path first")
            if workspace["source_path"] and destination == Path(workspace["source_path"]).resolve():
                if not replace_source:
                    raise RegistryTransferError("Replacing the source requires replace_source=true")
                if not destination.exists() or _sha256(destination) != workspace["source_sha256"]:
                    raise RegistryTransferError("The source SQLite file changed after it was loaded")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False)
            temporary_path = Path(temporary.name)
            temporary.close()
            new_revision_id = str(uuid.uuid4())
            replaced_destination = False
            backup_path = None
            try:
                with sqlite3.connect(temporary_path) as output:
                    initialize_database(
                        output, workspace["dataset_type"], dataset_id=workspace["dataset_id"],
                        revision_id=new_revision_id, parent_revision_id=workspace["revision_id"],
                        name=workspace["name"], title=workspace["title"], description=workspace["description"],
                        version=workspace["dataset_version"], metadata=workspace["metadata"],
                    )
                    _write_workspace(repository, cursor, schema, workspace_id, output, workspace, progress_callback=progress_callback)
                    output.commit()
                    report = validate_database(output)
                    if not report["valid"]:
                        raise RegistryTransferError("Export validation failed: " + "; ".join(report["errors"]))
                if destination.exists():
                    backup_path = destination.with_suffix(f".pre-export-{uuid.uuid4().hex[:8]}{destination.suffix}")
                    shutil.copy2(destination, backup_path)
                os.replace(temporary_path, destination)
                replaced_destination = True
                digest = _sha256(destination)
                cursor.execute(
                    f"""UPDATE {schema}.registry_workspaces SET revision_id=%s,parent_revision_id=%s,
                    source_path=%s,source_sha256=%s,source_size_bytes=%s,
                    status='loaded',dirty_at=NULL,exported_at=NOW(),last_export_path=%s,last_export_sha256=%s,
                    last_export_operation_id=%s
                    WHERE id=%s""",
                    (new_revision_id, workspace["revision_id"], str(destination), digest, destination.stat().st_size,
                     str(destination), digest, operation_id, workspace_id),
                )
                pg_connection.commit()
                return {"path": str(destination), "sha256": digest, "revision_id": new_revision_id,
                        "parent_revision_id": workspace["revision_id"],
                        "backup_path": str(backup_path) if backup_path else None}
            except Exception:
                temporary_path.unlink(missing_ok=True)
                if replaced_destination:
                    if backup_path is not None and backup_path.exists():
                        os.replace(backup_path, destination)
                    else:
                        destination.unlink(missing_ok=True)
                raise


def _write_workspace(repository, cursor, schema: str, workspace_id: str, output: sqlite3.Connection, workspace: dict[str, Any],
                     *, progress_callback: Callable[[int, str], None] | None = None) -> None:
    def pg_rows(table: str, order: str) -> list[dict[str, Any]]:
        cursor.execute(f"SELECT * FROM {schema}.{table} WHERE workspace_id=%s ORDER BY {order}", (workspace_id,))
        return list(cursor.fetchall())

    contract_rows = pg_rows("registry_contract_records", "relation,ordinal")

    def write_contract_row(row: dict[str, Any]) -> None:
        relation = row["relation"]
        columns = [entry[1] for entry in output.execute(f'PRAGMA table_info("{relation}")')]
        if not columns:
            return
        record = row["record"]
        values = [record.get(column) for column in columns]
        marks = ",".join("?" for _ in columns)
        names = ",".join(f'"{column}"' for column in columns)
        output.execute(f'INSERT OR IGNORE INTO "{relation}" ({names}) VALUES ({marks})', values)

    # Descriptor definitions can reference taxonomy concepts. Materialize these
    # dependency tables before the editable Registry relations.
    for relation in ("taxonomy_concepts", "taxonomy_concept_mappings"):
        for row in contract_rows:
            if row["relation"] == relation:
                write_contract_row(row)

    hot_store = getattr(repository, "hot_store", None)
    if hot_store is None:
        raise RegistryTransferError("Hot object storage is not configured for Registry assets")
    for row in pg_rows("registry_assets", "asset_id"):
        payload = row["payload"]
        if payload is None and row.get("payload_ref"):
            payload = hot_store.get_store(str(row["payload_ref"]))
        output.execute("""INSERT INTO assets(asset_id,dataset_id,content_sha256,payload,external_uri,encoding,
          media_type,shape_json,dtype,original_filename,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
          (row["asset_id"],workspace["dataset_id"],row["content_sha256"],bytes(payload) if payload is not None else None,
           row["external_uri"],row["encoding"],row["media_type"],json.dumps(row["shape"]),row["dtype"],
           row["original_filename"],json.dumps(row["metadata"],sort_keys=True),row["created_at"]))
    if progress_callback: progress_callback(1, "Exported assets")
    for row in pg_rows("registry_items", "ordinal"):
        output.execute("""INSERT INTO dataset_items
          (item_id,dataset_id,sample_weight,source_key,metadata_json,created_at,updated_at)
          VALUES(?,?,?,?,?,?,?)""",
          (row["item_id"],workspace["dataset_id"],row["sample_weight"],row["source_key"],json.dumps(row["metadata"],sort_keys=True),row["created_at"],row["updated_at"]))
        output.execute("""INSERT INTO item_geometry (
          item_id,coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
          crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h,metadata_json
          ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
          (row["item_id"],row["coordinate_space"],row["bbox_x"],row["bbox_y"],row["bbox_w"],row["bbox_h"],
           row["crop_bbox_x"],row["crop_bbox_y"],row["crop_bbox_w"],row["crop_bbox_h"],
           json.dumps(row["spatial_metadata"] or {},sort_keys=True)))
        if workspace["dataset_type"] == "classification":
            output.execute("INSERT INTO classification_items VALUES(?,?)",(row["item_id"],row["image_asset_id"]))
        else:
            output.execute("INSERT INTO mask_refinement_items VALUES(?,?,?)",(row["item_id"],row["image_asset_id"],row["candidate_mask_asset_id"]))
    if progress_callback: progress_callback(2, "Exported dataset items")
    for row in pg_rows("registry_labels", "origin,class_index NULLS LAST,label_id"):
        if row["origin"] == "classification":
            metadata = dict(row["metadata"] or {})
            registry = metadata.setdefault("registry", {})
            registry.update({key: value for key, value in {
                "display_name": row["display_name"], "rank": row["rank"],
                "description": row["description"], "created_at": row["created_at"],
                "deprecated_at": row["deprecated_at"],
            }.items() if value is not None})
            output.execute("INSERT INTO classification_labels VALUES(?,?,?,?,?,?)",
              (row["label_id"],workspace["dataset_id"],row["class_index"],row["name"],row["parent_label_id"],json.dumps(metadata,sort_keys=True)))
        else:
            output.execute("""INSERT INTO annotation_labels VALUES(?,?,?,?,?,?,?,?,?,?)""",
              (row["label_id"],workspace["dataset_id"],row["name"],row["display_name"],row["parent_label_id"],row["rank"],row["description"],
               json.dumps(row["metadata"],sort_keys=True),row["created_at"],row["deprecated_at"]))
    for row in pg_rows("registry_annotations", "created_at,annotation_id"):
        if row["origin"] == "classification":
            output.execute("""INSERT INTO classification_annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
              (row["annotation_id"],row["item_id"],row["label_id"],row["created_at"],row["annotator"],row["method"],row["confidence"],
               row["status"],int(row["is_current"]),row["parent_annotation_id"],row["notes"],json.dumps(row["metadata"],sort_keys=True)))
        else:
            output.execute("""INSERT INTO item_label_annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (row["annotation_id"],row["item_id"],row["label_id"],row["created_at"],row["annotator"],row["method"],row["confidence"],
               row["status"],int(row["is_current"]),row["parent_annotation_id"],json.dumps(row["parameters"],sort_keys=True),row["notes"],json.dumps(row["metadata"],sort_keys=True)))
    for row in pg_rows("registry_reviews", "created_at,review_id"):
        table = "classification_annotation_reviews" if row["origin"] == "classification" else "annotation_reviews"
        output.execute(f"INSERT INTO {table} VALUES(?,?,?,?,?,?,?)",
          (row["review_id"],row["annotation_id"],row["reviewer"],row["decision"],row["created_at"],row["notes"],json.dumps(row["metadata"],sort_keys=True)))
    if progress_callback: progress_callback(3, "Exported labels and annotations")
    for row in pg_rows("registry_descriptors", "descriptor_id"):
        output.execute("INSERT INTO descriptor_definitions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
          (row["descriptor_id"],workspace["dataset_id"],row["scope"],row["name"],row["parent_descriptor_id"],row["concept_id"],row["concept_type"],
           int(row["selectable"]),int(row["exclusive_within_parent"]),int(row["preferred"]),json.dumps(row["metadata"],sort_keys=True),row["created_at"],row["deprecated_at"]))
    for row in pg_rows("registry_descriptor_annotations", "created_at,annotation_id"):
        output.execute("INSERT INTO item_descriptor_annotations VALUES(?,?,?,?,?,?,?,?,?,?)",
          (row["annotation_id"],row["item_id"],row["descriptor_id"],row["created_at"],row["annotator"],row["status"],int(row["is_current"]),
           row["parent_annotation_id"],row["notes"],json.dumps(row["metadata"],sort_keys=True)))
    if progress_callback: progress_callback(4, "Exported descriptors")
    for row in pg_rows("registry_mask_annotations", "created_at,annotation_id"):
        output.execute("INSERT INTO mask_annotations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
          (row["annotation_id"],row["item_id"],row["mask_asset_id"],row["created_at"],row["annotator"],row["method"],
           json.dumps(row["parameters"],sort_keys=True),json.dumps(row["validation"],sort_keys=True),row["status"],int(row["is_current"]),
           row["parent_annotation_id"],row["notes"]))
    if progress_callback: progress_callback(5, "Exported mask annotations")
    for row in pg_rows("registry_inference_runs", "created_at,inference_run_id"):
        output.execute("INSERT INTO inference_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
          (row["inference_run_id"],workspace["dataset_id"],row["dataset_fingerprint_sha256"],row["model_artifact_id"],row["model_run_id"],
           row["model_artifact_fingerprint_sha256"],row["name"],row["status"],row["created_at"],row["completed_at"],
           json.dumps(row["input_contract"],sort_keys=True),json.dumps(row["parameters"],sort_keys=True),
           json.dumps(row["software_environment"],sort_keys=True),json.dumps(row["metadata"],sort_keys=True)))
    for row in pg_rows("registry_evidence_arrays", "array_id"):
        output.execute("INSERT INTO evidence_arrays VALUES(?,?,?,?,?,?,?,?,?)",
          (row["array_id"],row["content_sha256"],bytes(row["payload"]),row["encoding"],row["media_type"],json.dumps(row["shape"]),
           row["dtype"],row["created_at"],json.dumps(row["metadata"],sort_keys=True)))
    for row in pg_rows("registry_model_evidence", "created_at,evidence_id"):
        output.execute("INSERT INTO model_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
          (row["evidence_id"],row["inference_run_id"],row["item_id"],row["predicted_label_id"],row["prediction_confidence"],
           row["nearest_neighbor_similarity"],row["top_k_label_agreement"],row["weighted_label_support"],row["label_margin"],
           row["logits_array_id"],row["embedding_array_id"],row["output_array_id"],json.dumps(row["packet"],sort_keys=True),
           json.dumps(row["metadata"],sort_keys=True),row["created_at"]))
    if progress_callback: progress_callback(6, "Exported model evidence")
    for row in pg_rows("registry_dataset_events", "created_at,event_id"):
        output.execute("INSERT OR IGNORE INTO dataset_events VALUES(?,?,?,?,?,?,?)",
          (row["event_id"],workspace["dataset_id"],row["revision_id"],row["event_type"],row["created_at"],row["actor"],json.dumps(row["details"],sort_keys=True)))
    if progress_callback: progress_callback(7, "Exported dataset events")
    for row in contract_rows:
        if row["relation"] not in {"taxonomy_concepts", "taxonomy_concept_mappings"}:
            write_contract_row(row)
    if progress_callback: progress_callback(8, "Exported auxiliary contract records")
