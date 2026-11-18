"""Transactional ROI transfers between project-scoped Registry workspaces."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from oracle_data_contracts.datasets import initialize_database, read_dataset_info

from .registry_transfer import RegistryTransferError, _json, _jsonb

TransferMode = Literal["copy", "move"]
DuplicateAction = Literal["skip", "overwrite", "keep_destination_remove_source"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(source_workspace_id: Any, destination_workspace_id: Any, kind: str, value: Any) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL,
        f"pelagia-registry-workspace-transfer:{source_workspace_id}:{destination_workspace_id}:{kind}:{value}"))


def _bind(value: Any) -> Any:
    return _jsonb(value) if isinstance(value, (dict, list)) else value


def _semantic_metadata(value: Any) -> dict[str, Any]:
    resolved = dict(_json(value))
    resolved.pop("pelagia_transfer", None)
    return resolved


def _insert_row(cursor, schema: str, table: str, row: dict[str, Any], *, ignore_conflict: bool = False) -> None:
    columns = list(row)
    conflict = " ON CONFLICT DO NOTHING" if ignore_conflict else ""
    cursor.execute(
        f"INSERT INTO {schema}.{table} ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))}){conflict}",
        tuple(_bind(row[column]) for column in columns),
    )


def _workspace(cursor, schema: str, workspace_id: str, *, project_id: str,
               owner_username: str, for_update: bool = False) -> dict[str, Any]:
    cursor.execute(
        f"""SELECT * FROM {schema}.registry_workspaces
        WHERE id=%s AND project_id=%s AND owner_username=%s AND status='loaded'
        {"FOR UPDATE" if for_update else ""}""",
        (workspace_id, project_id, owner_username),
    )
    row = cursor.fetchone()
    if row is None:
        raise RegistryTransferError("Loaded Registry workspace was not found")
    return dict(row)


def _workspace_response(cursor, schema: str, workspace_id: Any) -> dict[str, Any]:
    cursor.execute(
        f"""SELECT w.id workspace_id,w.dataset_id,w.revision_id,w.parent_revision_id,
        w.dataset_type,w.name,w.title,w.description,w.dataset_version version,
        w.dataset_lifecycle lifecycle,w.source_path,w.source_size_bytes,w.status,w.is_active,
        w.dirty_at,w.loaded_at,w.exported_at,w.last_export_path,w.contract_schema_version,
        (SELECT count(*) FROM {schema}.registry_items i WHERE i.workspace_id=w.id) item_count,
        (SELECT count(DISTINCT a.item_id) FROM {schema}.registry_annotations a
          WHERE a.workspace_id=w.id AND a.is_current AND a.status IN ('accepted','candidate')) labeled_count
        FROM {schema}.registry_workspaces w WHERE w.id=%s""", (workspace_id,))
    result = dict(cursor.fetchone())
    result["workspace_id"] = str(result["workspace_id"])
    return result


def create_blank_workspace(repository, *, project_id: str, owner_username: str, name: str,
                           dataset_type: str, title: str | None = None,
                           description: str | None = None, activate: bool = True) -> dict[str, Any]:
    """Create an empty, exportable Registry workspace in the caller's scope."""
    resolved_name = name.strip()
    if not resolved_name:
        raise RegistryTransferError("Workspace name is required")
    if dataset_type not in {"classification", "mask_refinement"}:
        raise RegistryTransferError("Registry dataset type must be classification or mask_refinement")
    dataset_id, revision_id = str(uuid.uuid4()), str(uuid.uuid4())
    with sqlite3.connect(":memory:") as portable:
        initialize_database(portable, dataset_type, dataset_id=dataset_id, revision_id=revision_id,
            name=resolved_name, title=title, description=description,
            metadata={"pelagia": {"blank_workspace": True}})
        info = read_dataset_info(portable)
    schema = repository.schema
    with repository.connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                       (f"registry:{project_id}:{owner_username}",))
        if activate:
            cursor.execute(f"UPDATE {schema}.registry_workspaces SET is_active=false "
                           "WHERE project_id=%s AND owner_username=%s AND is_active",
                           (project_id, owner_username))
        cursor.execute(
            f"""INSERT INTO {schema}.registry_workspaces
            (project_id,owner_username,dataset_id,revision_id,parent_revision_id,dataset_type,
             name,title,description,dataset_version,dataset_lifecycle,contract_schema_name,
            contract_schema_version,source_path,source_sha256,source_size_bytes,status,is_active,dirty_at,metadata)
            VALUES (%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,'','',0,'loaded',%s,NOW(),%s)
            RETURNING id""",
            (project_id, owner_username, dataset_id, revision_id, dataset_type, resolved_name,
             title, description, info.get("version"), info["lifecycle"], info["schema_name"],
             info["schema_version"], activate, _jsonb({"pelagia": {"blank_workspace": True}})))
        workspace_id = cursor.fetchone()["id"]
        cursor.execute(
            f"""INSERT INTO {schema}.registry_dataset_events
            (workspace_id,event_id,revision_id,event_type,created_at,actor,details)
            VALUES (%s,%s,%s,'registry.workspace_created',%s,%s,%s)""",
            (workspace_id, str(uuid.uuid4()), revision_id, _now(), owner_username,
             _jsonb({"blank": True, "dataset_type": dataset_type})))
        result = _workspace_response(cursor, schema, workspace_id)
        connection.commit()
    return result


def _selected_source_items(cursor, schema: str, workspace_id: Any, item_ids: list[str]) -> list[dict[str, Any]]:
    selected = list(dict.fromkeys(str(value) for value in item_ids if str(value)))
    if not selected:
        raise RegistryTransferError("Select at least one ROI to transfer")
    cursor.execute(
        f"""SELECT item.*,asset.content_sha256 image_sha256,asset.shape image_shape
        FROM {schema}.registry_items item JOIN {schema}.registry_assets asset
          ON asset.workspace_id=item.workspace_id AND asset.asset_id=item.image_asset_id
        WHERE item.workspace_id=%s AND item.item_id=ANY(%s) ORDER BY item.ordinal,item.item_id""",
        (workspace_id, selected))
    rows = [dict(row) for row in cursor.fetchall()]
    missing = sorted(set(selected) - {str(row["item_id"]) for row in rows})
    if missing:
        raise RegistryTransferError(f"Selected ROI was not found in the source workspace: {missing[0]}")
    return rows


def _duplicate_inventory(cursor, schema: str, destination_workspace_id: Any,
                         source_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cursor.execute(
        f"""SELECT item.*,asset.content_sha256 image_sha256,asset.shape image_shape
        FROM {schema}.registry_items item JOIN {schema}.registry_assets asset
          ON asset.workspace_id=item.workspace_id AND asset.asset_id=item.image_asset_id
        WHERE item.workspace_id=%s""", (destination_workspace_id,))
    targets = [dict(row) for row in cursor.fetchall()]
    by_id = {str(row["item_id"]): row for row in targets}
    by_key = {str(row["source_key"]): row for row in targets if row.get("source_key")}
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in targets:
        by_hash.setdefault(str(row["image_sha256"]), []).append(row)

    def states(workspace_id: Any, item_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not item_ids:
            return {}
        cursor.execute(f"""SELECT i.item_id,l.origin label_origin,l.name label_name,r.decision
            FROM {schema}.registry_items i
            LEFT JOIN {schema}.registry_annotations a ON a.workspace_id=i.workspace_id
              AND a.item_id=i.item_id AND a.is_current AND a.status IN ('accepted','candidate')
            LEFT JOIN {schema}.registry_labels l ON l.workspace_id=a.workspace_id AND l.label_id=a.label_id
            LEFT JOIN LATERAL (SELECT decision FROM {schema}.registry_reviews review
              WHERE review.workspace_id=a.workspace_id AND review.annotation_id=a.annotation_id
              ORDER BY created_at DESC LIMIT 1) r ON true
            WHERE i.workspace_id=%s AND i.item_id=ANY(%s)""", (workspace_id, item_ids))
        result = {str(row["item_id"]): {"label": (row["label_origin"], row["label_name"])
                  if row["label_name"] is not None else None, "review": row["decision"], "descriptors": []}
                  for row in cursor.fetchall()}
        cursor.execute(f"""SELECT a.item_id,d.scope,d.name FROM {schema}.registry_descriptor_annotations a
            JOIN {schema}.registry_descriptors d ON d.workspace_id=a.workspace_id AND d.descriptor_id=a.descriptor_id
            WHERE a.workspace_id=%s AND a.item_id=ANY(%s) AND a.is_current AND a.status='accepted'
            ORDER BY d.scope,d.name""", (workspace_id, item_ids))
        for row in cursor.fetchall():
            result.setdefault(str(row["item_id"]), {"label": None, "review": None, "descriptors": []})[
                "descriptors"].append((row["scope"], row["name"]))
        return result

    source_states = states(source_items[0]["workspace_id"], [str(row["item_id"]) for row in source_items]) if source_items else {}
    target_states = states(destination_workspace_id, [str(row["item_id"]) for row in targets])
    duplicates = []
    for source in source_items:
        source_id = str(source["item_id"]); matches = {}; reasons: dict[str, set[str]] = {}
        if source_id in by_id:
            matches[source_id] = by_id[source_id]; reasons.setdefault(source_id, set()).add("uuid")
        source_key = source.get("source_key")
        if source_key and str(source_key) in by_key:
            row = by_key[str(source_key)]; target_id = str(row["item_id"])
            matches[target_id] = row; reasons.setdefault(target_id, set()).add("source_key")
        for row in by_hash.get(str(source["image_sha256"]), []):
            target_id = str(row["item_id"]); matches[target_id] = row
            reasons.setdefault(target_id, set()).add("roi_hash")
        if not matches:
            continue
        authoritative = source_id if source_id in matches else None
        if authoritative is None and source_key and str(source_key) in by_key:
            authoritative = str(by_key[str(source_key)]["item_id"])
        target_id = authoritative or sorted(matches)[0]
        source_metadata, target_metadata = _json(source.get("metadata")), _json(matches[target_id].get("metadata"))
        source_state = source_states.get(source_id, {}); target_state = target_states.get(target_id, {})
        conflicts = []
        metadata_conflicts = sorted(key for key in source_metadata.keys() & target_metadata.keys()
                                    if source_metadata[key] != target_metadata[key])
        if metadata_conflicts:
            conflicts.append("metadata")
        if str(source["image_sha256"]) != str(matches[target_id]["image_sha256"]):
            conflicts.append("image_content")
        def effective_geometry(row: dict[str, Any]) -> tuple[Any, ...]:
            crop = tuple(row.get(f"crop_bbox_{axis}") for axis in ("x", "y", "w", "h"))
            if any(value is None for value in crop):
                shape = row.get("image_shape") or []
                crop = (0, 0, int(shape[1]), int(shape[0])) if len(shape) > 1 else crop
            bbox = tuple(row.get(f"bbox_{axis}") for axis in ("x", "y", "w", "h"))
            if any(value is None for value in bbox):
                bbox = crop
            return (row.get("coordinate_space") or "image_pixels", *bbox, *crop)
        if effective_geometry(source) != effective_geometry(matches[target_id]):
            conflicts.append("geometry")
        for field in ("label", "review", "descriptors"):
            if source_state.get(field) != target_state.get(field):
                conflicts.append(field)
        duplicates.append({"source_item_id": source_id, "source_key": source_key,
            "target_item_id": target_id, "matching_item_ids": sorted(matches),
            "reasons": sorted({reason for values in reasons.values() for reason in values}),
            "roi_hash": str(source["image_sha256"]),
            "target_roi_hash": str(matches[target_id]["image_sha256"]),
            "content_conflict": str(source["image_sha256"]) != str(matches[target_id]["image_sha256"]),
            "metadata_conflicts": metadata_conflicts, "conflicts": conflicts,
            "source_state": source_state, "target_state": target_state})
    return duplicates


def _preview_token(source: dict[str, Any], destination: dict[str, Any],
                   source_items: list[dict[str, Any]], duplicates: list[dict[str, Any]], mode: str) -> str:
    material = {"source": str(source["id"]), "source_revision": source["revision_id"],
        "source_dirty": str(source.get("dirty_at")), "destination": str(destination["id"]),
        "destination_revision": destination["revision_id"], "destination_dirty": str(destination.get("dirty_at")),
        "mode": mode, "items": [(str(row["item_id"]), str(row.get("updated_at")), str(row["image_sha256"]))
                                  for row in source_items], "duplicates": duplicates}
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def _preview(cursor, schema: str, *, source: dict[str, Any], destination: dict[str, Any],
             item_ids: list[str], mode: TransferMode) -> dict[str, Any]:
    if source["id"] == destination["id"]:
        raise RegistryTransferError("Source and destination workspaces must be different")
    if source["dataset_type"] != destination["dataset_type"]:
        raise RegistryTransferError("Source and destination Registry dataset types must match")
    if mode not in {"copy", "move"}:
        raise RegistryTransferError("Transfer mode must be copy or move")
    items = _selected_source_items(cursor, schema, source["id"], item_ids)
    duplicates = _duplicate_inventory(cursor, schema, destination["id"], items)
    return {"preview_token": _preview_token(source, destination, items, duplicates, mode),
        "mode": mode, "source_workspace_id": str(source["id"]),
        "destination_workspace_id": str(destination["id"]), "total_items": len(items),
        "new_items": len(items) - len(duplicates), "duplicate_count": len(duplicates),
        "duplicates": duplicates, "_source_items": items}


def preview_workspace_transfer(repository, *, project_id: str, owner_username: str,
                               source_workspace_id: str, destination_workspace_id: str,
                               item_ids: list[str], mode: TransferMode) -> dict[str, Any]:
    schema = repository.schema
    try:
        source_uuid, destination_uuid = str(uuid.UUID(source_workspace_id)), str(uuid.UUID(destination_workspace_id))
    except (ValueError, AttributeError) as exc:
        raise RegistryTransferError("Registry workspace ID must be a valid UUID") from exc
    with repository.connect() as connection, connection.cursor() as cursor:
        source = _workspace(cursor, schema, source_uuid, project_id=project_id, owner_username=owner_username)
        destination = _workspace(cursor, schema, destination_uuid, project_id=project_id, owner_username=owner_username)
        result = _preview(cursor, schema, source=source, destination=destination, item_ids=item_ids, mode=mode)
    result.pop("_source_items")
    return result


def _rows_for_items(cursor, schema: str, table: str, workspace_id: Any,
                    item_ids: list[str]) -> list[dict[str, Any]]:
    cursor.execute(f"SELECT * FROM {schema}.{table} WHERE workspace_id=%s AND item_id=ANY(%s)",
                   (workspace_id, item_ids))
    return [dict(row) for row in cursor.fetchall()]


def _definition_maps(cursor, schema: str, source_id: Any, destination_id: Any,
                     label_ids: set[str], descriptor_ids: set[str]) -> tuple[dict[str, str], dict[str, str], set[str]]:
    maps: list[dict[str, str]] = []
    created_descriptors: set[str] = set()
    for table, id_column, key_columns, kind in (
        ("registry_labels", "label_id", ("origin", "name"), "label"),
        ("registry_descriptors", "descriptor_id", ("scope", "name"), "descriptor"),
    ):
        cursor.execute(f"SELECT * FROM {schema}.{table} WHERE workspace_id=%s", (source_id,))
        sources = [dict(row) for row in cursor.fetchall()]
        wanted = label_ids if kind == "label" else descriptor_ids
        source_lookup = {str(row[id_column]): row for row in sources}
        parent_column = "parent_label_id" if kind == "label" else "parent_descriptor_id"
        pending_ids = list(wanted)
        while pending_ids:
            current = source_lookup.get(pending_ids.pop())
            if current and current.get(parent_column) is not None:
                parent_id = str(current[parent_column])
                if parent_id not in wanted:
                    wanted.add(parent_id); pending_ids.append(parent_id)
        sources = [row for row in sources if str(row[id_column]) in wanted]
        cursor.execute(f"SELECT * FROM {schema}.{table} WHERE workspace_id=%s", (destination_id,))
        destinations = [dict(row) for row in cursor.fetchall()]
        by_key = {tuple(row[column] for column in key_columns): row for row in destinations}
        source_by_id = {str(row[id_column]): row for row in sources}
        destination_by_id = {str(row[id_column]): row for row in destinations}
        used_ids = {str(row[id_column]) for row in destinations}; mapping: dict[str, str] = {}
        pending: list[tuple[dict[str, Any], str]] = []
        used_class_indexes = {int(row["class_index"]) for row in destinations
                              if kind == "label" and row.get("class_index") is not None}
        next_class_index = max(used_class_indexes, default=-1) + 1
        for row in sources:
            source_value = str(row[id_column]); existing = by_key.get(tuple(row[column] for column in key_columns))
            if existing:
                semantic_columns = (("display_name", "rank", "description", "metadata") if kind == "label" else
                                    ("concept_type", "selectable", "exclusive_within_parent",
                                     "preferred", "metadata"))
                if any((_semantic_metadata(existing.get(column)) != _semantic_metadata(row.get(column)))
                       if column == "metadata" else (existing.get(column) != row.get(column))
                       for column in semantic_columns):
                    raise RegistryTransferError(
                        f"Destination {kind} {row['name']!r} has different semantic metadata"
                    )
                source_parent = row.get("parent_label_id" if kind == "label" else "parent_descriptor_id")
                target_parent = existing.get("parent_label_id" if kind == "label" else "parent_descriptor_id")
                source_parent_name = source_by_id.get(str(source_parent), {}).get("name") if source_parent else None
                target_parent_name = destination_by_id.get(str(target_parent), {}).get("name") if target_parent else None
                if source_parent_name != target_parent_name:
                    raise RegistryTransferError(
                        f"Destination {kind} {row['name']!r} has a different parent definition"
                    )
                mapping[source_value] = str(existing[id_column]); continue
            target_value = source_value
            if target_value in used_ids:
                target_value = _stable_id(source_id, destination_id, kind, source_value)
            used_ids.add(target_value); mapping[source_value] = target_value; pending.append((row, target_value))
        for row, target_value in pending:
            copied = dict(row); copied["workspace_id"] = destination_id; copied[id_column] = target_value
            if kind == "label" and copied.get("origin") == "classification" and copied.get("class_index") in used_class_indexes:
                metadata = _json(copied.get("metadata"))
                transfer = metadata.setdefault("pelagia_transfer", {})
                transfer["source_class_index"] = copied["class_index"]
                copied["metadata"] = metadata; copied["class_index"] = next_class_index
                used_class_indexes.add(next_class_index); next_class_index += 1
            elif kind == "label" and copied.get("class_index") is not None:
                used_class_indexes.add(int(copied["class_index"]))
            if copied.get(parent_column) is not None:
                copied[parent_column] = mapping.get(str(copied[parent_column]))
            _insert_row(cursor, schema, table, copied)
            if kind == "descriptor":
                created_descriptors.add(target_value)
        maps.append(mapping)
    return maps[0], maps[1], created_descriptors


def _copy_taxonomy_links(cursor, schema: str, source_id: Any, destination_id: Any,
                         label_map: dict[str, str], descriptor_map: dict[str, str],
                         created_descriptors: set[str]) -> None:
    relations = ["taxonomy_concepts", "taxonomy_concept_mappings", "classification_label_concepts"]
    cursor.execute(f"SELECT relation,ordinal,record FROM {schema}.registry_contract_records "
                   "WHERE workspace_id=%s AND relation=ANY(%s) ORDER BY relation,ordinal", (source_id, relations))
    source_rows = [dict(row) for row in cursor.fetchall()]
    cursor.execute(f"SELECT relation,ordinal,record FROM {schema}.registry_contract_records "
                   "WHERE workspace_id=%s AND relation=ANY(%s) ORDER BY relation,ordinal", (destination_id, relations))
    destination_rows = [dict(row) for row in cursor.fetchall()]
    cursor.execute(f"SELECT descriptor_id,concept_id FROM {schema}.registry_descriptors WHERE workspace_id=%s",
                   (source_id,))
    source_descriptors = [dict(row) for row in cursor.fetchall()]
    label_links = [_json(row["record"]) for row in source_rows
                   if row["relation"] == "classification_label_concepts"
                   and str(_json(row["record"]).get("label_id")) in label_map]
    needed_concepts = {str(record.get("concept_id")) for record in label_links if record.get("concept_id") is not None}
    needed_concepts.update(str(row["concept_id"]) for row in source_descriptors
                           if str(row["descriptor_id"]) in descriptor_map and row.get("concept_id") is not None)
    all_concepts = {str(record.get("concept_id")): record for record in
                    (_json(row["record"]) for row in source_rows if row["relation"] == "taxonomy_concepts")}
    pending_concepts = list(needed_concepts)
    while pending_concepts:
        record = all_concepts.get(pending_concepts.pop())
        if record and record.get("parent_concept_id") is not None:
            parent = str(record["parent_concept_id"])
            if parent not in needed_concepts:
                needed_concepts.add(parent); pending_concepts.append(parent)
    primary = {"taxonomy_concepts": "concept_id", "taxonomy_concept_mappings": "mapping_id",
               "classification_label_concepts": "label_id"}
    next_ordinal: dict[str, int] = {}; existing_keys: dict[str, set[str]] = {}
    existing_records: dict[str, dict[str, dict[str, Any]]] = {}; nodes = {}
    for row in destination_rows:
        relation, record = str(row["relation"]), _json(row["record"])
        next_ordinal[relation] = max(next_ordinal.get(relation, 0), int(row["ordinal"]) + 1)
        record_key = str(record.get(primary[relation]))
        existing_keys.setdefault(relation, set()).add(record_key)
        existing_records.setdefault(relation, {})[record_key] = record
        if relation == "taxonomy_concepts":
            nodes[(str(record.get("vocabulary_id")), str(record.get("vocabulary_node_id")))] = record
    concepts = [record for concept_id, record in all_concepts.items() if concept_id in needed_concepts]
    concept_map = {}
    for record in concepts:
        key = (str(record.get("vocabulary_id")), str(record.get("vocabulary_node_id")))
        existing = nodes.get(key)
        if existing is not None:
            semantic_keys = set(record) | set(existing)
            semantic_keys -= {"concept_id", "parent_concept_id", "dataset_id"}
            if any(record.get(field) != existing.get(field) for field in semantic_keys):
                raise RegistryTransferError(
                    f"Taxonomy concept {key[1]!r} has incompatible destination metadata"
                )
        concept_map[str(record.get("concept_id"))] = str(existing.get("concept_id")) if existing else _stable_id(
            source_id, destination_id, "taxonomy-concept", record.get("concept_id"))
    for record in concepts:
        key = (str(record.get("vocabulary_id")), str(record.get("vocabulary_node_id")))
        existing = nodes.get(key)
        if existing is None:
            continue
        source_parent = record.get("parent_concept_id")
        expected_parent = concept_map.get(str(source_parent), source_parent) if source_parent is not None else None
        if existing.get("parent_concept_id") != expected_parent:
            raise RegistryTransferError(
                f"Taxonomy concept {key[1]!r} has a different destination parent"
            )
    pending: list[tuple[str, dict[str, Any]]] = []
    for record in concepts:
        key = (str(record.get("vocabulary_id")), str(record.get("vocabulary_node_id")))
        if key in nodes:
            continue
        copied = dict(record); copied["concept_id"] = concept_map[str(record.get("concept_id"))]
        copied.setdefault("selectable", True)
        copied.setdefault("metadata_json", json.dumps(copied.pop("metadata", {}), sort_keys=True))
        if record.get("parent_concept_id") is not None:
            copied["parent_concept_id"] = concept_map.get(str(record["parent_concept_id"]), record["parent_concept_id"])
        pending.append(("taxonomy_concepts", copied))
    for row in source_rows:
        relation, record = str(row["relation"]), _json(row["record"])
        if relation == "taxonomy_concept_mappings" and str(record.get("concept_id")) in needed_concepts:
            copied = dict(record)
            copied["mapping_id"] = _stable_id(source_id, destination_id, "taxonomy-mapping", record.get("mapping_id"))
            copied["concept_id"] = concept_map.get(str(record.get("concept_id")), record.get("concept_id"))
            pending.append((relation, copied))
        elif relation == "classification_label_concepts" and str(record.get("label_id")) in label_map:
            copied = dict(record); copied["label_id"] = label_map[str(record["label_id"])]
            copied["concept_id"] = concept_map.get(str(record.get("concept_id")), record.get("concept_id"))
            pending.append((relation, copied))
    for relation, record in pending:
        key = str(record.get(primary[relation]))
        if key in existing_keys.setdefault(relation, set()):
            if relation == "classification_label_concepts":
                existing = existing_records[relation][key]
                if any(existing.get(field) != record.get(field) for field in ("concept_id", "relationship")):
                    raise RegistryTransferError(
                        f"Classification label {key!r} maps to a different destination taxonomy concept"
                    )
            continue
        ordinal = next_ordinal.get(relation, 0)
        cursor.execute(f"INSERT INTO {schema}.registry_contract_records "
                       "(workspace_id,relation,ordinal,record) VALUES (%s,%s,%s,%s)",
                       (destination_id, relation, ordinal, _jsonb(record)))
        next_ordinal[relation] = ordinal + 1; existing_keys[relation].add(key)
        existing_records.setdefault(relation, {})[key] = record
    cursor.execute(f"SELECT descriptor_id,concept_id FROM {schema}.registry_descriptors WHERE workspace_id=%s",
                   (destination_id,))
    destination_concepts = {str(row["descriptor_id"]): row["concept_id"] for row in cursor.fetchall()}
    for source_descriptor in source_descriptors:
        source_descriptor_id = str(source_descriptor["descriptor_id"])
        if source_descriptor_id not in descriptor_map:
            continue
        target_descriptor = descriptor_map[source_descriptor_id]
        source_concept = source_descriptor.get("concept_id")
        expected_concept = concept_map.get(str(source_concept), source_concept) if source_concept is not None else None
        if target_descriptor in created_descriptors:
            cursor.execute(f"UPDATE {schema}.registry_descriptors SET concept_id=%s "
                           "WHERE workspace_id=%s AND descriptor_id=%s",
                           (expected_concept, destination_id, target_descriptor))
        elif destination_concepts.get(target_descriptor) != expected_concept:
            raise RegistryTransferError(
                f"Destination descriptor {target_descriptor!r} maps to a different taxonomy concept"
            )


def _asset_map(cursor, schema: str, source_id: Any, destination_id: Any,
               asset_ids: set[str]) -> dict[str, str]:
    if not asset_ids:
        return {}
    cursor.execute(f"SELECT * FROM {schema}.registry_assets WHERE workspace_id=%s AND asset_id=ANY(%s)",
                   (source_id, sorted(asset_ids)))
    sources = [dict(row) for row in cursor.fetchall()]
    if len(sources) != len(asset_ids):
        raise RegistryTransferError("A selected ROI references an unavailable Registry asset")
    cursor.execute(f"SELECT * FROM {schema}.registry_assets WHERE workspace_id=%s", (destination_id,))
    destinations = [dict(row) for row in cursor.fetchall()]
    by_content: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in destinations:
        by_content.setdefault((str(row["content_sha256"]), str(row["encoding"])), []).append(row)
    used_ids = {str(row["asset_id"]) for row in destinations}; mapping: dict[str, str] = {}
    for row in sources:
        source_asset_id = str(row["asset_id"])
        candidates = by_content.get((str(row["content_sha256"]), str(row["encoding"])), [])
        identity_columns = ("shape", "dtype", "media_type", "external_uri", "original_filename", "metadata")
        existing = next((candidate for candidate in candidates
                         if all(candidate.get(column) == row.get(column) for column in identity_columns)), None)
        if existing is not None:
            mapping[source_asset_id] = str(existing["asset_id"]); continue
        target_id = source_asset_id
        if target_id in used_ids:
            target_id = _stable_id(source_id, destination_id, "asset", source_asset_id)
        copied = dict(row); copied["workspace_id"] = destination_id; copied["asset_id"] = target_id
        _insert_row(cursor, schema, "registry_assets", copied)
        used_ids.add(target_id); by_content.setdefault(
            (str(row["content_sha256"]), str(row["encoding"])), []).append(copied)
        mapping[source_asset_id] = target_id
    return mapping


def _copy_item_records(cursor, schema: str, *, source: dict[str, Any], destination: dict[str, Any],
                       copied_items: list[dict[str, Any]], item_map: dict[str, str],
                       overwrite_source_ids: set[str], operation_id: str,
                       overwrite_archive: list[dict[str, Any]]) -> dict[str, int]:
    source_id, destination_id = source["id"], destination["id"]
    source_item_ids = [str(row["item_id"]) for row in copied_items]
    annotation_rows = _rows_for_items(cursor, schema, "registry_annotations", source_id, source_item_ids)
    descriptor_rows = _rows_for_items(cursor, schema, "registry_descriptor_annotations", source_id, source_item_ids)
    mask_rows = _rows_for_items(cursor, schema, "registry_mask_annotations", source_id, source_item_ids)
    evidence_rows = _rows_for_items(cursor, schema, "registry_model_evidence", source_id, source_item_ids)
    label_ids = {str(row["label_id"]) for row in annotation_rows}
    label_ids.update(str(row["predicted_label_id"]) for row in evidence_rows if row.get("predicted_label_id") is not None)
    descriptor_ids = {str(row["descriptor_id"]) for row in descriptor_rows}
    label_map, descriptor_map, created_descriptors = _definition_maps(
        cursor, schema, source_id, destination_id, label_ids, descriptor_ids)
    _copy_taxonomy_links(cursor, schema, source_id, destination_id, label_map,
                         descriptor_map, created_descriptors)
    asset_ids = {str(value) for row in copied_items
                 for value in (row.get("image_asset_id"), row.get("candidate_mask_asset_id")) if value is not None}
    asset_ids.update(str(row["mask_asset_id"]) for row in mask_rows)
    asset_map = _asset_map(cursor, schema, source_id, destination_id, asset_ids)
    cursor.execute(f"SELECT coalesce(max(ordinal),-1) value FROM {schema}.registry_items WHERE workspace_id=%s",
                   (destination_id,))
    next_ordinal = int(cursor.fetchone()["value"]) + 1
    for source_item in copied_items:
        source_item_id = str(source_item["item_id"]); target_item_id = item_map[source_item_id]
        row = dict(source_item); row.pop("image_sha256", None); row.pop("image_shape", None)
        row["workspace_id"] = destination_id
        row["item_id"] = target_item_id; row["image_asset_id"] = asset_map[str(row["image_asset_id"])]
        if row.get("candidate_mask_asset_id") is not None:
            row["candidate_mask_asset_id"] = asset_map[str(row["candidate_mask_asset_id"])]
        if source_item_id in overwrite_source_ids:
            cursor.execute(f"SELECT * FROM {schema}.registry_items WHERE workspace_id=%s AND item_id=%s",
                           (destination_id, target_item_id))
            prior_item = dict(cursor.fetchone())
            prior_item.pop("workspace_id", None)
            cursor.execute(f"SELECT * FROM {schema}.registry_model_evidence WHERE workspace_id=%s AND item_id=%s",
                           (destination_id, target_item_id))
            prior_evidence = []
            for prior in cursor.fetchall():
                value = dict(prior); value.pop("workspace_id", None); prior_evidence.append(value)
            prior_current_revisions: dict[str, list[str]] = {}
            for table in ("registry_annotations", "registry_descriptor_annotations", "registry_mask_annotations"):
                cursor.execute(f"SELECT annotation_id FROM {schema}.{table} "
                               "WHERE workspace_id=%s AND item_id=%s AND is_current ORDER BY annotation_id",
                               (destination_id, target_item_id))
                prior_current_revisions[table] = [str(value["annotation_id"]) for value in cursor.fetchall()]
            overwrite_archive.append({"source_item_id": source_item_id,
                                      "destination_item_id": target_item_id,
                                      "prior_item": prior_item, "prior_evidence": prior_evidence,
                                      "prior_current_revisions": prior_current_revisions})
            cursor.execute(f"DELETE FROM {schema}.registry_model_evidence WHERE workspace_id=%s AND item_id=%s",
                           (destination_id, target_item_id))
            assignments = [column for column in row if column not in {"workspace_id", "item_id", "ordinal"}]
            cursor.execute(f"UPDATE {schema}.registry_items SET "
                           f"{','.join(f'{column}=%s' for column in assignments)} WHERE workspace_id=%s AND item_id=%s",
                           tuple(_bind(row[column]) for column in assignments) + (destination_id, target_item_id))
            for table in ("registry_annotations", "registry_descriptor_annotations", "registry_mask_annotations"):
                cursor.execute(f"UPDATE {schema}.{table} SET is_current=false "
                               "WHERE workspace_id=%s AND item_id=%s AND is_current",
                               (destination_id, target_item_id))
        else:
            row["ordinal"] = next_ordinal; next_ordinal += 1
            _insert_row(cursor, schema, "registry_items", row)

    annotation_map = {str(row["annotation_id"]): _stable_id(
        source_id, destination_id, f"annotation:{operation_id}", row["annotation_id"])
                      for row in annotation_rows}
    for row in annotation_rows:
        copied = dict(row); copied["workspace_id"] = destination_id
        copied["annotation_id"] = annotation_map[str(row["annotation_id"])]
        copied["item_id"] = item_map[str(row["item_id"])]
        copied["label_id"] = label_map[str(row["label_id"])]
        copied["parent_annotation_id"] = annotation_map.get(str(row.get("parent_annotation_id")))
        _insert_row(cursor, schema, "registry_annotations", copied, ignore_conflict=True)
    if annotation_map:
        cursor.execute(f"SELECT * FROM {schema}.registry_reviews WHERE workspace_id=%s AND annotation_id=ANY(%s)",
                       (source_id, list(annotation_map)))
        for row in cursor.fetchall():
            copied = dict(row); copied["workspace_id"] = destination_id
            copied["review_id"] = _stable_id(source_id, destination_id, f"review:{operation_id}", row["review_id"])
            copied["annotation_id"] = annotation_map[str(row["annotation_id"])]
            _insert_row(cursor, schema, "registry_reviews", copied, ignore_conflict=True)

    descriptor_annotation_map = {
        str(row["annotation_id"]): _stable_id(
            source_id, destination_id, f"descriptor-annotation:{operation_id}", row["annotation_id"])
        for row in descriptor_rows}
    for row in descriptor_rows:
        copied = dict(row); copied["workspace_id"] = destination_id
        copied["annotation_id"] = descriptor_annotation_map[str(row["annotation_id"])]
        copied["item_id"] = item_map[str(row["item_id"])]
        copied["descriptor_id"] = descriptor_map[str(row["descriptor_id"])]
        copied["parent_annotation_id"] = descriptor_annotation_map.get(str(row.get("parent_annotation_id")))
        _insert_row(cursor, schema, "registry_descriptor_annotations", copied, ignore_conflict=True)

    mask_map = {str(row["annotation_id"]): _stable_id(
        source_id, destination_id, f"mask-annotation:{operation_id}", row["annotation_id"])
                for row in mask_rows}
    for row in mask_rows:
        copied = dict(row); copied["workspace_id"] = destination_id
        copied["annotation_id"] = mask_map[str(row["annotation_id"])]
        copied["item_id"] = item_map[str(row["item_id"])]
        copied["mask_asset_id"] = asset_map[str(row["mask_asset_id"])]
        copied["parent_annotation_id"] = mask_map.get(str(row.get("parent_annotation_id")))
        _insert_row(cursor, schema, "registry_mask_annotations", copied, ignore_conflict=True)

    run_map: dict[str, str] = {}; array_map: dict[str, str] = {}
    if evidence_rows:
        source_run_ids = sorted({str(row["inference_run_id"]) for row in evidence_rows})
        cursor.execute(f"SELECT * FROM {schema}.registry_inference_runs WHERE workspace_id=%s", (destination_id,))
        existing_runs = {str(row["inference_run_id"]): dict(row) for row in cursor.fetchall()}
        cursor.execute(f"SELECT * FROM {schema}.registry_inference_runs WHERE workspace_id=%s AND inference_run_id=ANY(%s)",
                       (source_id, source_run_ids))
        for raw in cursor.fetchall():
            row = dict(raw); source_run_id = str(row["inference_run_id"])
            target_run_id = _stable_id(source_id, destination_id, f"inference-run:{operation_id}", source_run_id)
            run_map[source_run_id] = target_run_id
            if target_run_id not in existing_runs:
                row["workspace_id"] = destination_id; row["inference_run_id"] = target_run_id
                metadata = _json(row.get("metadata")); metadata["transfer_source_workspace_id"] = str(source_id)
                row["metadata"] = metadata
                _insert_row(cursor, schema, "registry_inference_runs", row); existing_runs[target_run_id] = row
        source_array_ids = sorted({str(value) for row in evidence_rows for value in
            (row.get("logits_array_id"), row.get("embedding_array_id"), row.get("output_array_id")) if value is not None})
        cursor.execute(f"SELECT * FROM {schema}.registry_evidence_arrays WHERE workspace_id=%s", (destination_id,))
        destination_arrays = [dict(row) for row in cursor.fetchall()]
        arrays_by_hash = {str(row["content_sha256"]): row for row in destination_arrays}
        used_array_ids = {str(row["array_id"]) for row in destination_arrays}
        if source_array_ids:
            cursor.execute(f"SELECT * FROM {schema}.registry_evidence_arrays "
                           "WHERE workspace_id=%s AND array_id=ANY(%s)", (source_id, source_array_ids))
            for raw in cursor.fetchall():
                row = dict(raw); source_array_id = str(row["array_id"])
                existing = arrays_by_hash.get(str(row["content_sha256"]))
                if existing:
                    array_map[source_array_id] = str(existing["array_id"]); continue
                target_array_id = source_array_id
                if target_array_id in used_array_ids:
                    target_array_id = _stable_id(source_id, destination_id, "evidence-array", source_array_id)
                row["workspace_id"] = destination_id; row["array_id"] = target_array_id
                _insert_row(cursor, schema, "registry_evidence_arrays", row)
                arrays_by_hash[str(row["content_sha256"])] = row; used_array_ids.add(target_array_id)
                array_map[source_array_id] = target_array_id
        for row in evidence_rows:
            copied = dict(row); target_item_id = item_map[str(row["item_id"])]
            copied["workspace_id"] = destination_id
            copied["evidence_id"] = _stable_id(
                source_id, destination_id, f"model-evidence:{operation_id}", row["evidence_id"])
            copied["inference_run_id"] = run_map[str(row["inference_run_id"])]
            copied["item_id"] = target_item_id
            if row.get("predicted_label_id") is not None:
                copied["predicted_label_id"] = label_map.get(str(row["predicted_label_id"]))
            for column in ("logits_array_id", "embedding_array_id", "output_array_id"):
                if row.get(column) is not None:
                    copied[column] = array_map[str(row[column])]
            metadata = _json(copied.get("metadata")); metadata["transfer_source_workspace_id"] = str(source_id)
            copied["metadata"] = metadata
            _insert_row(cursor, schema, "registry_model_evidence", copied, ignore_conflict=True)
    return {"annotations": len(annotation_rows), "descriptor_annotations": len(descriptor_rows),
            "mask_annotations": len(mask_rows), "evidence_records": len(evidence_rows)}


def _remove_source_items(cursor, schema: str, workspace_id: Any, item_ids: list[str]) -> None:
    if not item_ids:
        return
    cursor.execute(f"SELECT annotation_id FROM {schema}.registry_annotations "
                   "WHERE workspace_id=%s AND item_id=ANY(%s)", (workspace_id, item_ids))
    annotation_ids = [str(row["annotation_id"]) for row in cursor.fetchall()]
    if annotation_ids:
        cursor.execute(f"DELETE FROM {schema}.registry_reviews WHERE workspace_id=%s AND annotation_id=ANY(%s)",
                       (workspace_id, annotation_ids))
    for table in ("registry_annotations", "registry_descriptor_annotations", "registry_mask_annotations",
                  "registry_model_evidence"):
        cursor.execute(f"DELETE FROM {schema}.{table} WHERE workspace_id=%s AND item_id=ANY(%s)",
                       (workspace_id, item_ids))
    cursor.execute(f"DELETE FROM {schema}.registry_items WHERE workspace_id=%s AND item_id=ANY(%s)",
                   (workspace_id, item_ids))
    # Retain now-unreferenced asset and evidence-array rows. Transfer events can
    # refer to their IDs when an overwritten state is inspected; historical-aware
    # object cleanup is deliberately separate from a move transaction.


def validate_transfer_actions(preview: dict[str, Any], mode: TransferMode,
                              duplicate_policy: DuplicateAction,
                              duplicate_actions: dict[str, DuplicateAction]) -> None:
    if duplicate_policy not in {"skip", "overwrite", "keep_destination_remove_source"}:
        raise RegistryTransferError("Unknown duplicate policy")
    selected = {str(row["source_item_id"]) for row in preview["duplicates"]}
    unexpected = sorted(set(duplicate_actions) - selected)
    if unexpected:
        raise RegistryTransferError(f"Duplicate action refers to a non-duplicate ROI: {unexpected[0]}")
    claimed_targets: dict[str, str] = {}
    for duplicate in preview["duplicates"]:
        source_item_id = str(duplicate["source_item_id"])
        action = duplicate_actions.get(source_item_id, duplicate_policy)
        if action not in {"skip", "overwrite", "keep_destination_remove_source"}:
            raise RegistryTransferError(f"Unknown duplicate action for ROI {source_item_id}")
        if mode == "copy" and action == "keep_destination_remove_source":
            raise RegistryTransferError("Copy transfers cannot remove a source duplicate")
        if action == "skip":
            continue
        if len(duplicate["matching_item_ids"]) != 1:
            raise RegistryTransferError(f"ROI {source_item_id} matches multiple destination items; resolution is ambiguous")
        if duplicate.get("content_conflict") and source_item_id not in duplicate_actions:
            raise RegistryTransferError(
                f"ROI {source_item_id} has different image content; choose its duplicate action explicitly"
            )
        target = str(duplicate["target_item_id"])
        prior = claimed_targets.get(target)
        if prior is not None:
            raise RegistryTransferError(
                f"ROIs {prior} and {source_item_id} resolve to the same destination ROI; resolve one as skip"
            )
        claimed_targets[target] = source_item_id


def _source_provenance_snapshot(cursor, schema: str, source: dict[str, Any],
                                item_ids: list[str]) -> dict[str, Any]:
    cursor.execute(f"SELECT relation,ordinal,record FROM {schema}.registry_contract_records "
                   "WHERE workspace_id=%s AND relation=ANY(%s) ORDER BY relation,ordinal",
                   (source["id"], ["metadata_documents", "import_events"]))
    contract = [{"relation": row["relation"], "ordinal": row["ordinal"], "record": _json(row["record"])}
                for row in cursor.fetchall()]
    cursor.execute(f"SELECT event_id,revision_id,event_type,created_at,actor,details "
                   f"FROM {schema}.registry_dataset_events WHERE workspace_id=%s ORDER BY created_at,event_id",
                   (source["id"],))
    source_events = []
    for row in cursor.fetchall():
        details = _json(row["details"])
        serialized = json.dumps(details, sort_keys=True, default=str)
        event = {key: row[key] for key in ("event_id", "revision_id", "event_type", "created_at", "actor")}
        if any(item_id in serialized for item_id in item_ids):
            event["details"] = details
        source_events.append(event)
    return {"dataset": {key: source.get(key) for key in (
                "dataset_id", "revision_id", "parent_revision_id", "dataset_type", "name", "title",
                "description", "dataset_version", "dataset_lifecycle", "contract_schema_name",
                "contract_schema_version", "metadata")},
            "contract_provenance": contract, "events": source_events}


def execute_workspace_transfer(repository, *, project_id: str, owner_username: str,
                               source_workspace_id: str, destination_workspace_id: str,
                               item_ids: list[str], mode: TransferMode, expected_preview_token: str,
                               duplicate_policy: DuplicateAction,
                               duplicate_actions: dict[str, DuplicateAction] | None,
                               operation_id: str, actor: str,
                               progress_callback: Callable[[int, str], None] | None = None) -> dict[str, Any]:
    """Copy or move selected ROIs and their complete item-scoped provenance."""
    if duplicate_policy not in {"skip", "overwrite", "keep_destination_remove_source"}:
        raise RegistryTransferError("Unknown duplicate policy")
    if not operation_id.strip():
        raise RegistryTransferError("A transfer operation ID is required")
    try:
        source_uuid, destination_uuid = str(uuid.UUID(source_workspace_id)), str(uuid.UUID(destination_workspace_id))
    except (ValueError, AttributeError) as exc:
        raise RegistryTransferError("Registry workspace ID must be a valid UUID") from exc
    duplicate_actions = duplicate_actions or {}
    schema = repository.schema
    event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pelagia-registry-transfer-operation:{operation_id}"))
    request_fingerprint = hashlib.sha256(json.dumps({"source_workspace_id": source_uuid,
        "destination_workspace_id": destination_uuid, "item_ids": sorted(set(item_ids)), "mode": mode,
        "expected_preview_token": expected_preview_token, "duplicate_policy": duplicate_policy,
        "duplicate_actions": duplicate_actions}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    total_steps = max(len(set(item_ids)), 1)
    with repository.connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                       (f"registry:{project_id}:{owner_username}",))
        locked = {}
        for workspace_id in sorted({source_uuid, destination_uuid}):
            locked[workspace_id] = _workspace(cursor, schema, workspace_id, project_id=project_id,
                                               owner_username=owner_username, for_update=True)
        source, destination = locked[source_uuid], locked[destination_uuid]
        cursor.execute(f"SELECT details FROM {schema}.registry_dataset_events WHERE workspace_id=%s AND event_id=%s",
                       (destination["id"], event_id))
        prior = cursor.fetchone()
        if prior is not None:
            prior_details = _json(prior["details"])
            if prior_details.get("request_fingerprint") != request_fingerprint:
                raise RegistryTransferError("Transfer operation ID was already used for a different request")
            result = prior_details.get("result")
            if isinstance(result, dict):
                return {**result, "reused": True}
        preview = _preview(cursor, schema, source=source, destination=destination, item_ids=item_ids, mode=mode)
        if preview["preview_token"] != expected_preview_token:
            raise RegistryTransferError("Registry workspaces changed after the transfer preview; preview the transfer again")
        validate_transfer_actions(preview, mode, duplicate_policy, duplicate_actions)
        source_provenance = _source_provenance_snapshot(cursor, schema, source, item_ids)
        if progress_callback:
            progress_callback(0, "Validated Registry transfer preview")
        duplicates = {row["source_item_id"]: row for row in preview["duplicates"]}
        copy_items: list[dict[str, Any]] = []; item_map: dict[str, str] = {}
        overwrite_source_ids: set[str] = set(); remove_source_ids: list[str] = []; skipped = 0
        for source_item in preview["_source_items"]:
            source_item_id = str(source_item["item_id"]); duplicate = duplicates.get(source_item_id)
            if duplicate is None:
                copy_items.append(source_item); item_map[source_item_id] = source_item_id
                if mode == "move":
                    remove_source_ids.append(source_item_id)
                continue
            action = duplicate_actions.get(source_item_id, duplicate_policy)
            if action == "skip":
                skipped += 1; continue
            if action == "keep_destination_remove_source":
                remove_source_ids.append(source_item_id); continue
            target_item_id = str(duplicate["target_item_id"])
            copy_items.append(source_item); item_map[source_item_id] = target_item_id
            overwrite_source_ids.add(source_item_id)
            if mode == "move":
                remove_source_ids.append(source_item_id)
        overwrite_archive: list[dict[str, Any]] = []
        dependency_counts = _copy_item_records(cursor, schema, source=source, destination=destination,
            copied_items=copy_items, item_map=item_map, overwrite_source_ids=overwrite_source_ids,
            operation_id=operation_id, overwrite_archive=overwrite_archive) if copy_items else {
                "annotations": 0, "descriptor_annotations": 0, "mask_annotations": 0, "evidence_records": 0}
        if progress_callback:
            progress_callback(max(len(copy_items) - 1, 0), "Copied ROI records and provenance")
        if mode == "move":
            _remove_source_items(cursor, schema, source["id"], remove_source_ids)
        copied_source_ids = {str(row["item_id"]) for row in copy_items}
        result = {"operation_id": operation_id, "mode": mode,
            "source_workspace_id": str(source["id"]), "destination_workspace_id": str(destination["id"]),
            "copied_items": len(copy_items) - len(overwrite_source_ids),
            "moved_items": len(remove_source_ids) if mode == "move" else 0,
            "overwritten_items": len(overwrite_source_ids), "skipped_items": skipped,
            "removed_source_duplicates": len(set(remove_source_ids) - copied_source_ids), **dependency_counts}
        details = {"operation_id": operation_id, "source_workspace_id": str(source["id"]),
            "destination_workspace_id": str(destination["id"]), "selection_item_ids": item_ids,
            "preview_token": expected_preview_token, "overwritten_prior_states": overwrite_archive,
            "source_provenance": source_provenance,
            "request_fingerprint": request_fingerprint, "result": result}
        created_at = _now()
        for workspace, event_type in ((destination, "registry.rois_received"),
            (source, "registry.rois_moved" if mode == "move" else "registry.rois_copied")):
            cursor.execute(f"""INSERT INTO {schema}.registry_dataset_events
                (workspace_id,event_id,revision_id,event_type,created_at,actor,details)
                VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (workspace["id"], event_id, workspace["revision_id"], event_type,
                 created_at, actor, _jsonb(details)))
            cursor.execute(f"UPDATE {schema}.registry_workspaces SET dirty_at=NOW() WHERE id=%s", (workspace["id"],))
        connection.commit()
    if progress_callback:
        progress_callback(total_steps, "Registry transfer complete")
    return result


__all__ = ["create_blank_workspace", "preview_workspace_transfer", "validate_transfer_actions",
           "execute_workspace_transfer"]
