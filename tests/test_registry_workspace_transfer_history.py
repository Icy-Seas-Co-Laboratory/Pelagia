from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

from oracle_data_contracts.datasets import initialize_database, validate_database

from Pelagia.services.registry_transfer import export_sqlite_workspace, load_sqlite_workspace
from Pelagia.services.registry_workspace import RegistryWorkspaceService
from Pelagia.services.registry_workspace_transfer import (
    create_blank_workspace,
    execute_workspace_transfer,
    preview_workspace_transfer,
)
from test_postgres import postgres_repo


def _scope(repository):
    project = repository.create_project(f"transfer-history-{uuid.uuid4().hex}")
    return {"project_id": str(project["id"]), "owner_username": "curator"}


def _source_dataset(path):
    item_id, asset_id, candidate_asset_id = (str(uuid.uuid4()) for _ in range(3))
    with sqlite3.connect(path) as connection:
        info = initialize_database(connection, "mask_refinement", name="source history")
        connection.execute(
            """INSERT INTO assets(asset_id,dataset_id,content_sha256,payload,encoding,shape_json,dtype,created_at)
            VALUES (?,?,?,?,'raw','[1,1]','uint8','2026-11-01T00:00:00Z')""",
            (asset_id, info["dataset_id"], hashlib.sha256(b"image").hexdigest(), b"image"),
        )
        connection.execute(
            """INSERT INTO assets(asset_id,dataset_id,content_sha256,payload,encoding,shape_json,dtype,created_at)
            VALUES (?,?,?,?,'raw','[1,1]','uint8','2026-11-01T00:00:00Z')""",
            (candidate_asset_id, info["dataset_id"], hashlib.sha256(b"mask").hexdigest(), b"mask"),
        )
        connection.execute(
            """INSERT INTO dataset_items(item_id,dataset_id,metadata_json,created_at,updated_at)
            VALUES (?,?,?,'2026-11-01T00:00:00Z','2026-11-01T00:00:00Z')""",
            (item_id, info["dataset_id"], "{}"),
        )
        connection.execute("INSERT INTO mask_refinement_items VALUES (?,?,?)", (item_id, asset_id, candidate_asset_id))
        connection.execute("INSERT INTO item_geometry VALUES (?,'image_pixels',0,0,1,1,0,0,1,1,'{}')", (item_id,))
        connection.commit()
    return item_id


def _transfer(repository, scope, source_id, destination_id, item_id, operation_id, policy="skip"):
    preview = preview_workspace_transfer(
        repository, **scope, source_workspace_id=source_id, destination_workspace_id=destination_id,
        item_ids=[item_id], mode="copy",
    )
    return execute_workspace_transfer(
        repository, **scope, source_workspace_id=source_id, destination_workspace_id=destination_id,
        item_ids=[item_id], mode="copy", expected_preview_token=preview["preview_token"],
        duplicate_policy=policy, duplicate_actions={}, operation_id=operation_id, actor="curator",
    )


def test_transfer_preserves_full_roi_history_and_export_contract(postgres_repo, tmp_path):
    scope = _scope(postgres_repo)
    source_path = tmp_path / "history-source.sqlite"
    item_id = _source_dataset(source_path)
    source = load_sqlite_workspace(postgres_repo, source_path, **scope)
    destination = create_blank_workspace(postgres_repo, **scope, name="History destination",
                                         dataset_type="mask_refinement", activate=False)
    schema, source_id, destination_id = postgres_repo.schema, source["workspace_id"], destination["workspace_id"]
    concept_id, descriptor_id, array_id, run_id = (str(uuid.uuid4()) for _ in range(4))
    workspace = RegistryWorkspaceService(postgres_repo, **scope)
    label = workspace.add_label({"name": "copepod"})
    created = workspace.assign([item_id], label["label_id"], "curator")
    workspace.review([item_id], "verified", "reviewer")
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""INSERT INTO {schema}.registry_descriptors
            (workspace_id,descriptor_id,scope,name,concept_id,concept_type,selectable,exclusive_within_parent,preferred,metadata,created_at)
            VALUES (%s,%s,'target','marine',%s,'taxon',true,false,true,'{{}}','2026-11-01T00:00:00Z')""",
            (source_id, descriptor_id, concept_id),
        )
        cursor.execute(
            f"""INSERT INTO {schema}.registry_descriptor_annotations
            (workspace_id,annotation_id,item_id,descriptor_id,created_at,annotator,status,is_current,metadata)
            VALUES (%s,%s,%s,%s,'2026-11-01T01:00:00Z','curator','accepted',true,'{{}}')""",
            (source_id, str(uuid.uuid4()), item_id, descriptor_id),
        )
        cursor.execute(f"SELECT candidate_mask_asset_id FROM {schema}.registry_items WHERE workspace_id=%s AND item_id=%s",
                       (source_id, item_id))
        mask_asset_id = cursor.fetchone()["candidate_mask_asset_id"]
        cursor.execute(
            f"""INSERT INTO {schema}.registry_mask_annotations
            (workspace_id,annotation_id,item_id,mask_asset_id,created_at,method,parameters,validation,status,is_current)
            VALUES (%s,%s,%s,%s,'2026-11-01T01:00:00Z','manual','{{}}','{{}}','accepted',true)""",
            (source_id, str(uuid.uuid4()), item_id, mask_asset_id),
        )
        cursor.execute(
            f"""INSERT INTO {schema}.registry_inference_runs
            (workspace_id,inference_run_id,dataset_fingerprint_sha256,status,created_at)
            VALUES (%s,%s,'fingerprint','complete','2026-11-01T00:00:00Z')""", (source_id, run_id))
        cursor.execute(
            f"""INSERT INTO {schema}.registry_evidence_arrays
            (workspace_id,array_id,content_sha256,payload,encoding,media_type,shape,dtype,created_at,metadata)
            VALUES (%s,%s,%s,%s,'raw','application/octet-stream','[1]','uint8','2026-11-01T00:00:00Z','{{}}')""",
            (source_id, array_id, hashlib.sha256(b"evidence").hexdigest(), b"evidence"),
        )
        cursor.execute(
            f"""INSERT INTO {schema}.registry_model_evidence
            (workspace_id,evidence_id,inference_run_id,item_id,prediction_confidence,embedding_array_id,created_at)
            VALUES (%s,%s,%s,%s,0.9,%s,'2026-11-01T01:00:00Z')""",
            (source_id, str(uuid.uuid4()), run_id, item_id, array_id),
        )
        cursor.execute(
            f"""INSERT INTO {schema}.registry_contract_records(workspace_id,relation,ordinal,record)
            VALUES (%s,'taxonomy_concepts',0,%s)""",
            (source_id, json.dumps({"concept_id": concept_id, "vocabulary_id": "marine-v1",
                                    "vocabulary_version": "1", "vocabulary_node_id": "copepoda",
                                    "name": "Copepoda", "display_name": None, "scientific_name": None,
                                    "concept_type": "taxon", "rank": None, "parent_concept_id": None,
                                    "selectable": 1, "metadata_json": "{}", "created_at": "2026-11-01T00:00:00Z"})),
        )
        connection.commit()

    assert _transfer(postgres_repo, scope, source_id, destination_id, item_id, "first") == {
        "operation_id": "first", "mode": "copy", "source_workspace_id": source_id,
        "destination_workspace_id": destination_id, "copied_items": 1, "moved_items": 0,
        "overwritten_items": 0, "skipped_items": 0, "removed_source_duplicates": 0,
        "annotations": 1, "descriptor_annotations": 1, "mask_annotations": 1, "evidence_records": 1,
    }
    overwritten = _transfer(postgres_repo, scope, source_id, destination_id, item_id, "second", "overwrite")
    assert overwritten["overwritten_items"] == 1

    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        for table in ("registry_annotations", "registry_descriptor_annotations", "registry_mask_annotations"):
            cursor.execute(f"SELECT count(*) FILTER (WHERE is_current) current,count(*) total FROM {schema}.{table} "
                           "WHERE workspace_id=%s AND item_id=%s", (destination_id, item_id))
            row = cursor.fetchone()
            assert (row["current"], row["total"]) == (1, 2)
        cursor.execute(f"SELECT count(*) count FROM {schema}.registry_model_evidence WHERE workspace_id=%s AND item_id=%s",
                       (destination_id, item_id))
        assert cursor.fetchone()["count"] == 1
        cursor.execute(f"SELECT details FROM {schema}.registry_dataset_events WHERE workspace_id=%s "
                       "AND event_type='registry.rois_received' ORDER BY created_at DESC LIMIT 1", (destination_id,))
        archived = cursor.fetchone()["details"]["overwritten_prior_states"][0]["prior_evidence"]
        assert len(archived) == 1
        cursor.execute(f"SELECT concept_id FROM {schema}.registry_descriptors WHERE workspace_id=%s AND name='marine'", (destination_id,))
        descriptor_concept = str(cursor.fetchone()["concept_id"])
        cursor.execute(f"SELECT record FROM {schema}.registry_contract_records WHERE workspace_id=%s AND relation='taxonomy_concepts'", (destination_id,))
        assert descriptor_concept == str(cursor.fetchone()["record"]["concept_id"])

    exported = tmp_path / "history-export.sqlite"
    export_sqlite_workspace(postgres_repo, destination_id, exported, **scope, operation_id="history-export")
    with sqlite3.connect(exported) as connection:
        assert validate_database(connection)["valid"] is True
        assert connection.execute("SELECT count(*) FROM annotation_reviews").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM item_descriptor_annotations WHERE is_current=1").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM mask_annotations WHERE is_current=1").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM model_evidence").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM taxonomy_concepts").fetchone()[0] == 1
