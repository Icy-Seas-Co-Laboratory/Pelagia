from __future__ import annotations

import json
import sqlite3
import uuid

import pytest
from oracle_data_contracts.datasets import validate_database

from Pelagia.services import registry_workspace_transfer
from Pelagia.services.registry_transfer import RegistryTransferError, export_sqlite_workspace, load_sqlite_workspace
from Pelagia.services.registry_workspace_transfer import (
    create_blank_workspace,
    execute_workspace_transfer,
    preview_workspace_transfer,
)
from test_postgres import postgres_repo
from test_registry_import import _dataset


def _scope(repository):
    project = repository.create_project(f"workspace-transfer-{uuid.uuid4().hex}")
    return {"project_id": str(project["id"]), "owner_username": "curator"}


def _add_classification(path, item_id: str, name: str) -> None:
    with sqlite3.connect(path) as connection:
        dataset_id = connection.execute("SELECT dataset_id FROM dataset WHERE singleton=1").fetchone()[0]
        label_id = str(uuid.uuid4())
        connection.execute("INSERT INTO classification_labels VALUES (?,?,0,?,NULL,'{}')",
                           (label_id, dataset_id, name))
        connection.execute(
            "INSERT INTO classification_annotations VALUES (?,?,?,'2026-11-01T00:00:00Z','curator','manual',NULL,'accepted',1,NULL,NULL,'{}')",
            (str(uuid.uuid4()), item_id, label_id),
        )
        connection.commit()


def test_blank_workspace_copy_is_exportable_and_retry_is_idempotent(postgres_repo, tmp_path):
    scope = _scope(postgres_repo)
    source_path = tmp_path / "source.sqlite"
    item_id = _dataset(source_path, metadata={"cruise": "north"})
    source = load_sqlite_workspace(postgres_repo, source_path, **scope)
    destination = create_blank_workspace(
        postgres_repo, **scope, name="Curated subset", dataset_type="classification", activate=False,
    )

    preview = preview_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[item_id], mode="copy",
    )
    result = execute_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[item_id], mode="copy",
        expected_preview_token=preview["preview_token"], duplicate_policy="skip", duplicate_actions={},
        operation_id="copy-one", actor="curator",
    )
    assert result["copied_items"] == 1
    assert execute_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[item_id], mode="copy",
        expected_preview_token=preview["preview_token"], duplicate_policy="skip", duplicate_actions={},
        operation_id="copy-one", actor="curator",
    )["reused"] is True

    exported = tmp_path / "blank-export.sqlite"
    export_sqlite_workspace(
        postgres_repo, destination["workspace_id"], exported, **scope, operation_id="export-blank",
    )
    with sqlite3.connect(exported) as connection:
        assert validate_database(connection)["valid"] is True
        assert connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0] == 1


def test_overwrite_archives_prior_item_and_preserves_source_history(postgres_repo, tmp_path):
    scope = _scope(postgres_repo)
    source_path, destination_path = tmp_path / "source.sqlite", tmp_path / "destination.sqlite"
    source_item = _dataset(source_path, payload=b"same", metadata={"quality": "incoming"})
    destination_item = _dataset(destination_path, payload=b"same", metadata={"quality": "existing"})
    source = load_sqlite_workspace(postgres_repo, source_path, **scope)
    destination = load_sqlite_workspace(postgres_repo, destination_path, **scope)

    preview = preview_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[source_item], mode="copy",
    )
    assert preview["duplicates"][0]["target_item_id"] == destination_item
    execute_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[source_item], mode="copy",
        expected_preview_token=preview["preview_token"], duplicate_policy="overwrite", duplicate_actions={},
        operation_id="overwrite-one", actor="curator",
    )
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT metadata FROM {postgres_repo.schema}.registry_items WHERE workspace_id=%s AND item_id=%s",
                       (destination["workspace_id"], destination_item))
        assert cursor.fetchone()["metadata"] == {"quality": "incoming"}
        cursor.execute(f"SELECT details FROM {postgres_repo.schema}.registry_dataset_events "
                       "WHERE workspace_id=%s AND event_type='registry.rois_received'",
                       (destination["workspace_id"],))
        details = cursor.fetchone()["details"]
    assert details["overwritten_prior_states"][0]["prior_item"]["metadata"] == {"quality": "existing"}

    exported = tmp_path / "overwritten.sqlite"
    export_sqlite_workspace(
        postgres_repo, destination["workspace_id"], exported, **scope, operation_id="export-overwrite",
    )
    with sqlite3.connect(exported) as connection:
        assert validate_database(connection)["valid"] is True
        transfer = connection.execute(
            "SELECT details_json FROM dataset_events WHERE event_type='registry.rois_received'"
        ).fetchone()
        assert json.loads(transfer[0])["overwritten_prior_states"][0]["prior_item"]["metadata"] == {
            "quality": "existing"
        }


def test_repeated_overwrite_keeps_one_current_annotation_and_reindexes_labels(postgres_repo, tmp_path):
    scope = _scope(postgres_repo)
    source_path, destination_path = tmp_path / "source-label.sqlite", tmp_path / "destination-label.sqlite"
    source_item = _dataset(source_path, payload=b"source-label")
    destination_item = _dataset(destination_path, payload=b"destination-label")
    _add_classification(source_path, source_item, "incoming")
    _add_classification(destination_path, destination_item, "existing")
    source = load_sqlite_workspace(postgres_repo, source_path, **scope)
    destination = load_sqlite_workspace(postgres_repo, destination_path, **scope)
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT annotation_id FROM {postgres_repo.schema}.registry_annotations "
                       "WHERE workspace_id=%s AND item_id=%s AND is_current",
                       (source["workspace_id"], source_item))
        annotation_id = cursor.fetchone()["annotation_id"]
        cursor.execute(f"INSERT INTO {postgres_repo.schema}.registry_reviews "
                       "(workspace_id,review_id,annotation_id,origin,reviewer,decision,created_at,metadata) "
                       "VALUES (%s,%s,%s,'classification','reviewer','verified','2026-11-01T01:00:00Z','{}')",
                       (source["workspace_id"], str(uuid.uuid4()), annotation_id))
        connection.commit()

    first = preview_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[source_item], mode="copy",
    )
    execute_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[source_item], mode="copy",
        expected_preview_token=first["preview_token"], duplicate_policy="skip", duplicate_actions={},
        operation_id="first-copy", actor="curator",
    )
    second = preview_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[source_item], mode="copy",
    )
    execute_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[source_item], mode="copy",
        expected_preview_token=second["preview_token"], duplicate_policy="overwrite", duplicate_actions={},
        operation_id="second-copy", actor="curator",
    )
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT class_index,name FROM {postgres_repo.schema}.registry_labels "
                       "WHERE workspace_id=%s AND origin='classification' ORDER BY class_index",
                       (destination["workspace_id"],))
        assert [(row["class_index"], row["name"]) for row in cursor.fetchall()] == [(0, "existing"), (1, "incoming")]
        cursor.execute(f"SELECT is_current FROM {postgres_repo.schema}.registry_annotations "
                       "WHERE workspace_id=%s AND item_id=%s ORDER BY created_at,annotation_id",
                       (destination["workspace_id"], source_item))
        states = [row["is_current"] for row in cursor.fetchall()]
        assert states.count(True) == 1
        assert len(states) == 2
        cursor.execute(f"SELECT count(*) count FROM {postgres_repo.schema}.registry_reviews WHERE workspace_id=%s",
                       (destination["workspace_id"],))
        assert cursor.fetchone()["count"] == 2

    exported = tmp_path / "labels-export.sqlite"
    export_sqlite_workspace(postgres_repo, destination["workspace_id"], exported, **scope,
                            operation_id="export-labels")
    with sqlite3.connect(exported) as connection:
        assert validate_database(connection)["valid"] is True


def test_move_can_keep_destination_and_remove_source_duplicate(postgres_repo, tmp_path):
    scope = _scope(postgres_repo)
    source_path, destination_path = tmp_path / "move-source.sqlite", tmp_path / "move-destination.sqlite"
    source_item = _dataset(source_path, payload=b"duplicate", metadata={"side": "source"})
    destination_item = _dataset(destination_path, payload=b"duplicate", metadata={"side": "destination"})
    source = load_sqlite_workspace(postgres_repo, source_path, **scope)
    destination = load_sqlite_workspace(postgres_repo, destination_path, **scope)
    preview = preview_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[source_item], mode="move",
    )
    result = execute_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[source_item], mode="move",
        expected_preview_token=preview["preview_token"], duplicate_policy="skip",
        duplicate_actions={source_item: "keep_destination_remove_source"},
        operation_id="remove-source-duplicate", actor="curator",
    )
    assert result["removed_source_duplicates"] == 1
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT count(*) count FROM {postgres_repo.schema}.registry_items WHERE workspace_id=%s",
                       (source["workspace_id"],))
        assert cursor.fetchone()["count"] == 0
        cursor.execute(f"SELECT item_id,metadata FROM {postgres_repo.schema}.registry_items WHERE workspace_id=%s",
                       (destination["workspace_id"],))
        row = cursor.fetchone()
        assert row["item_id"] == destination_item
        assert row["metadata"] == {"side": "destination"}


def test_transfer_rejects_stale_preview_and_wrong_owner(postgres_repo, tmp_path):
    scope = _scope(postgres_repo)
    source_path = tmp_path / "stale.sqlite"
    item_id = _dataset(source_path)
    source = load_sqlite_workspace(postgres_repo, source_path, **scope)
    destination = create_blank_workspace(
        postgres_repo, **scope, name="Destination", dataset_type="classification", activate=False,
    )
    preview = preview_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[item_id], mode="copy",
    )
    with pytest.raises(RegistryTransferError, match="not found"):
        preview_workspace_transfer(
            postgres_repo, project_id=scope["project_id"], owner_username="other",
            source_workspace_id=source["workspace_id"], destination_workspace_id=destination["workspace_id"],
            item_ids=[item_id], mode="copy",
        )
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"UPDATE {postgres_repo.schema}.registry_items SET updated_at=%s "
                       "WHERE workspace_id=%s AND item_id=%s",
                       ("2026-11-18T12:00:00Z", source["workspace_id"], item_id))
        connection.commit()
    with pytest.raises(RegistryTransferError, match="preview the transfer again"):
        execute_workspace_transfer(
            postgres_repo, **scope, source_workspace_id=source["workspace_id"],
            destination_workspace_id=destination["workspace_id"], item_ids=[item_id], mode="copy",
            expected_preview_token=preview["preview_token"], duplicate_policy="skip", duplicate_actions={},
            operation_id="stale", actor="curator",
        )


def test_move_rolls_back_destination_when_source_removal_fails(postgres_repo, tmp_path, monkeypatch):
    scope = _scope(postgres_repo)
    source_path = tmp_path / "rollback.sqlite"
    item_id = _dataset(source_path)
    source = load_sqlite_workspace(postgres_repo, source_path, **scope)
    destination = create_blank_workspace(
        postgres_repo, **scope, name="Rollback destination", dataset_type="classification", activate=False,
    )
    preview = preview_workspace_transfer(
        postgres_repo, **scope, source_workspace_id=source["workspace_id"],
        destination_workspace_id=destination["workspace_id"], item_ids=[item_id], mode="move",
    )
    monkeypatch.setattr(registry_workspace_transfer, "_remove_source_items",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected removal failure")))
    with pytest.raises(RuntimeError, match="injected removal failure"):
        execute_workspace_transfer(
            postgres_repo, **scope, source_workspace_id=source["workspace_id"],
            destination_workspace_id=destination["workspace_id"], item_ids=[item_id], mode="move",
            expected_preview_token=preview["preview_token"], duplicate_policy="skip", duplicate_actions={},
            operation_id="rollback", actor="curator",
        )
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT count(*) count FROM {postgres_repo.schema}.registry_items WHERE workspace_id=%s",
                       (destination["workspace_id"],))
        assert cursor.fetchone()["count"] == 0
