from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
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
                _load_assets(cursor, schema, workspace_id, source_connection)
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


def _load_assets(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection) -> None:
    for row in _rows(source, "assets"):
        cursor.execute(
            f"""INSERT INTO {schema}.registry_assets (
              workspace_id,asset_id,content_sha256,payload,external_uri,encoding,media_type,
              shape,dtype,original_filename,metadata,created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                workspace_id,row["asset_id"],row["content_sha256"],row["payload"],row["external_uri"],
                row["encoding"],row["media_type"],_jsonb(row["shape_json"]),row["dtype"],
                row["original_filename"],_jsonb(row["metadata_json"]),row["created_at"],
            ),
        )


def _load_items(cursor, schema: str, workspace_id: Any, source: sqlite3.Connection, dataset_type: str) -> None:
    relation = "classification_items" if dataset_type == "classification" else "mask_refinement_items"
    task_rows = {str(row["item_id"]): row for row in _rows(source, relation)}
    for ordinal, row in enumerate(_rows(source, "dataset_items")):
        task = task_rows[str(row["item_id"])]
        cursor.execute(
            f"""INSERT INTO {schema}.registry_items (
              workspace_id,item_id,ordinal,sample_weight,source_key,image_asset_id,
              candidate_mask_asset_id,metadata,created_at,updated_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                workspace_id,row["item_id"],ordinal,row["sample_weight"],row["source_key"],
                task["image_asset_id"],task["candidate_mask_asset_id"] if dataset_type == "mask_refinement" else None,
                _jsonb(row["metadata_json"]),row["created_at"],row["updated_at"],
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
            if destination == Path(workspace["source_path"]).resolve():
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
                    _write_workspace(cursor, schema, workspace_id, output, workspace, progress_callback=progress_callback)
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


def _write_workspace(cursor, schema: str, workspace_id: str, output: sqlite3.Connection, workspace: dict[str, Any],
                     *, progress_callback: Callable[[int, str], None] | None = None) -> None:
    def pg_rows(table: str, order: str) -> list[dict[str, Any]]:
        cursor.execute(f"SELECT * FROM {schema}.{table} WHERE workspace_id=%s ORDER BY {order}", (workspace_id,))
        return list(cursor.fetchall())

    for row in pg_rows("registry_assets", "asset_id"):
        output.execute("""INSERT INTO assets(asset_id,dataset_id,content_sha256,payload,external_uri,encoding,
          media_type,shape_json,dtype,original_filename,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
          (row["asset_id"],workspace["dataset_id"],row["content_sha256"],bytes(row["payload"]) if row["payload"] is not None else None,
           row["external_uri"],row["encoding"],row["media_type"],json.dumps(row["shape"]),row["dtype"],
           row["original_filename"],json.dumps(row["metadata"],sort_keys=True),row["created_at"]))
    if progress_callback: progress_callback(1, "Exported assets")
    for row in pg_rows("registry_items", "ordinal"):
        output.execute("INSERT INTO dataset_items VALUES(?,?,?,?,?,?,?)",
          (row["item_id"],workspace["dataset_id"],row["sample_weight"],row["source_key"],json.dumps(row["metadata"],sort_keys=True),row["created_at"],row["updated_at"]))
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
    for row in pg_rows("registry_contract_records", "relation,ordinal"):
        relation = row["relation"]
        columns = [entry[1] for entry in output.execute(f'PRAGMA table_info("{relation}")')]
        if not columns:
            continue
        record = row["record"]
        values = [record.get(column) for column in columns]
        marks = ",".join("?" for _ in columns)
        names = ",".join(f'"{column}"' for column in columns)
        output.execute(f'INSERT OR IGNORE INTO "{relation}" ({names}) VALUES ({marks})', values)
    if progress_callback: progress_callback(8, "Exported auxiliary contract records")
