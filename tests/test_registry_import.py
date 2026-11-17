from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

import pytest
from oracle_data_contracts.datasets import initialize_database

from Pelagia.services import registry_transfer
from Pelagia.services.registry_transfer import RegistryTransferError, load_sqlite_workspace
from Pelagia.services.registry_import import import_sqlite_into_workspace, preview_sqlite_import
from Pelagia.services.registry_workspace import RegistryWorkspaceService
from test_postgres import postgres_repo  # Reuse the isolated PostgreSQL schema/blob fixture.


def _dataset(path, *, item_id=None, payload=b"a", metadata=None):
    item_id = item_id or str(uuid.uuid4())
    asset_id = str(uuid.uuid4())
    with sqlite3.connect(path) as connection:
        info = initialize_database(connection, "classification", name=path.stem)
        connection.execute(
            """INSERT INTO assets
            (asset_id,dataset_id,content_sha256,payload,encoding,shape_json,dtype,created_at)
            VALUES (?,?,?,?,'raw','[1,1]','uint8','2026-11-01T00:00:00Z')""",
            (asset_id, info["dataset_id"], hashlib.sha256(payload).hexdigest(), payload),
        )
        connection.execute(
            """INSERT INTO dataset_items (item_id,dataset_id,metadata_json,created_at,updated_at)
            VALUES (?,?,?,'2026-11-01T00:00:00Z','2026-11-01T00:00:00Z')""",
            (item_id, info["dataset_id"], json.dumps(metadata or {})),
        )
        connection.execute("INSERT INTO classification_items VALUES (?,?)", (item_id, asset_id))
        connection.execute("INSERT INTO item_geometry VALUES (?,'image_pixels',0,0,1,1,0,0,1,1,'{}')", (item_id,))
        connection.commit()
    return item_id


def _workspace(repository, tmp_path, *, metadata=None):
    destination = tmp_path / "destination.sqlite"
    target = _dataset(destination, metadata=metadata)
    project = repository.create_project(f"import-guards-{uuid.uuid4().hex}")
    scope = {"project_id": str(project["id"]), "owner_username": "curator"}
    loaded = load_sqlite_workspace(repository, destination, **scope)
    return scope, loaded["workspace_id"], target


def test_registry_duplicate_skip_and_merge_preserve_destination_metadata(postgres_repo, tmp_path):
    scope, workspace_id, target = _workspace(postgres_repo, tmp_path, metadata={"quality": "reviewed"})
    source = tmp_path / "additional.sqlite"
    incoming = _dataset(source, metadata={"quality": "unreviewed", "cruise": "field-1"})
    preview = preview_sqlite_import(postgres_repo, source, **scope)
    assert preview["new_items"] == 0
    assert preview["duplicates"][0]["source_item_id"] == incoming
    assert preview["duplicates"][0]["target_item_id"] == target
    assert "roi_hash" in preview["duplicates"][0]["reasons"]
    assert preview["duplicates"][0]["metadata_conflicts"] == ["quality"]
    skipped = import_sqlite_into_workspace(postgres_repo, source, **scope)
    assert (skipped["imported_items"], skipped["skipped_items"]) == (0, 1)
    merged = import_sqlite_into_workspace(postgres_repo, source, duplicate_policy="merge", **scope)
    assert merged["merged_items"] == 1
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT item_id,metadata FROM {postgres_repo.schema}.registry_items WHERE workspace_id=%s", (workspace_id,))
        rows = list(cursor.fetchall())
    assert len(rows) == 1
    assert rows[0]["item_id"] == target
    assert rows[0]["metadata"] == {"quality": "reviewed", "cruise": "field-1"}


def test_registry_import_rolls_back_partial_rows(postgres_repo, tmp_path, monkeypatch):
    scope, workspace_id, target = _workspace(postgres_repo, tmp_path)
    service = RegistryWorkspaceService(postgres_repo, **scope)
    before = service.active_workspace()
    source = tmp_path / "additional.sqlite"
    _dataset(source, payload=b"b")

    def fail(*args, **kwargs):
        raise RegistryTransferError("injected failure after item insertion")

    monkeypatch.setattr(registry_transfer, "_import_auxiliary_records", fail)
    with pytest.raises(RegistryTransferError, match="injected failure"):
        import_sqlite_into_workspace(postgres_repo, source, **scope)
    assert service.active_workspace() == before
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT item_id FROM {postgres_repo.schema}.registry_items WHERE workspace_id=%s", (workspace_id,))
        assert [row["item_id"] for row in cursor.fetchall()] == [target]
        cursor.execute(f"SELECT count(*) count FROM {postgres_repo.schema}.registry_assets WHERE workspace_id=%s", (workspace_id,))
        assert cursor.fetchone()["count"] == 1


def test_registry_import_rejects_changed_source_and_wrong_owner(postgres_repo, tmp_path):
    scope, workspace_id, _ = _workspace(postgres_repo, tmp_path)
    source = tmp_path / "additional.sqlite"
    _dataset(source, payload=b"b")
    preview = preview_sqlite_import(postgres_repo, source, **scope)
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE dataset_items SET metadata_json='{}' || ' '")
        connection.commit()
    with pytest.raises(RegistryTransferError, match="changed"):
        import_sqlite_into_workspace(postgres_repo, source, expected_source_sha256=preview["source"]["sha256"], **scope)
    for operation in (preview_sqlite_import, import_sqlite_into_workspace):
        with pytest.raises(RegistryTransferError, match="not found"):
            operation(postgres_repo, source, project_id=scope["project_id"], owner_username="other", workspace_id=workspace_id)


def test_registry_import_rejects_merging_same_uuid_with_different_image(postgres_repo, tmp_path):
    scope, _, target = _workspace(postgres_repo, tmp_path)
    source = tmp_path / "conflicting.sqlite"
    _dataset(source, item_id=target, payload=b"b")
    with pytest.raises(RegistryTransferError, match="image|content"):
        import_sqlite_into_workspace(postgres_repo, source, duplicate_policy="merge", **scope)


def test_registry_import_preserves_existing_descriptor_concept(postgres_repo, tmp_path):
    paths = [tmp_path / "destination.sqlite", tmp_path / "source.sqlite"]
    concepts = [str(uuid.uuid4()), str(uuid.uuid4())]
    for index, path in enumerate(paths):
        _dataset(path, payload=bytes([index]))
        with sqlite3.connect(path) as connection:
            dataset_id = connection.execute("SELECT dataset_id FROM dataset").fetchone()[0]
            connection.execute(
                """INSERT INTO taxonomy_concepts
                (concept_id,vocabulary_id,vocabulary_version,vocabulary_node_id,name,concept_type,created_at)
                VALUES (?,?, '1','node','Reviewed','quality','2026-11-01T00:00:00Z')""",
                (concepts[index], f"vocabulary-{index}"),
            )
            connection.execute(
                """INSERT INTO descriptor_definitions
                (descriptor_id,dataset_id,scope,name,concept_id,created_at)
                VALUES (?,?,'target','Reviewed',?,'2026-11-01T00:00:00Z')""",
                (str(uuid.uuid4()), dataset_id, concepts[index]),
            )
            connection.commit()
    project = postgres_repo.create_project(f"import-concepts-{uuid.uuid4().hex}")
    scope = {"project_id": str(project["id"]), "owner_username": "curator"}
    loaded = load_sqlite_workspace(postgres_repo, paths[0], **scope)
    import_sqlite_into_workspace(postgres_repo, paths[1], **scope)
    with postgres_repo.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT concept_id FROM {postgres_repo.schema}.registry_descriptors WHERE workspace_id=%s", (loaded["workspace_id"],))
        assert [row["concept_id"] for row in cursor.fetchall()] == [concepts[0]]
