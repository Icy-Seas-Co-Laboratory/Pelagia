"""One-time, offline copy of a project's legacy objects into SkerryKV.

Stop all Pelagia writers before running this command. The source store must remain
available until the copy has completed and the database has been backed up.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Any


def _copy(source: Any, destination: Any, ref: str, cache: dict[str, str]) -> str:
    if ref not in cache:
        payload = source.get_store(ref)
        new_ref = destination.put_store(payload)
        if destination.get_store(new_ref) != payload:
            raise RuntimeError(f"SkerryKV verification failed for {ref}")
        cache[ref] = new_ref
    return cache[ref]


def migrate_project(connection: Any, schema: str, project_id: str, source: Any, destination: Any, destination_id: str) -> dict[str, int]:
    """Copy frame, embedding, and telemetry source objects and replace their DB references.

    Run with writers stopped. A checkpoint is committed with each updated row so
    that a failed or interrupted run can be restarted with the same destination.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError("Invalid PostgreSQL schema name")
    from psycopg.types.json import Jsonb

    cache: dict[str, str] = {}
    counts = {"frames": 0, "embeddings": 0, "telemetry_sources": 0}
    with connection.cursor() as cursor:
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {schema}.skerry_cold_migration (
            project_id uuid NOT NULL, kind text NOT NULL, row_id uuid NOT NULL, destination text NOT NULL,
            PRIMARY KEY (project_id, kind, row_id))""")
    connection.commit()
    specs = (
        ("frames", f"SELECT f.id FROM {schema}.frames f JOIN {schema}.runs r ON r.id = f.run_id WHERE r.project_id = %s"),
        *((name, f"SELECT id FROM {schema}.{name} WHERE project_id = %s AND embedding_payload_ref IS NOT NULL")
          for name in ("classification_evidence", "clustering_evidence", "embedding_evidence")),
        ("telemetry_sources", f"SELECT id FROM {schema}.telemetry_sources WHERE project_id = %s AND source_payload_key IS NOT NULL"),
    )
    for kind, query in specs:
        with connection.cursor() as cursor:
            cursor.execute(query, (project_id,))
            ids = [row[0] for row in cursor.fetchall()]
        connection.commit()
        if kind != "frames":
            from concurrent.futures import ThreadPoolExecutor

            column = "source_payload_key" if kind == "telemetry_sources" else "embedding_payload_ref"
            with ThreadPoolExecutor(max_workers=16) as executor:
                for offset in range(0, len(ids), 128):
                    batch = ids[offset:offset + 128]
                    with connection.cursor() as cursor:
                        cursor.execute(f"""SELECT row_id, destination FROM {schema}.skerry_cold_migration
                            WHERE project_id = %s AND kind = %s AND row_id = ANY(%s::uuid[])""", (project_id, kind, batch))
                        checkpoints = dict(cursor.fetchall())
                        if any(value != destination_id for value in checkpoints.values()):
                            raise RuntimeError("Migration destination differs from the existing checkpoint")
                        pending = [row_id for row_id in batch if row_id not in checkpoints]
                        if not pending:
                            connection.rollback()
                            continue
                        cursor.execute(f"SELECT id, {column} FROM {schema}.{kind} WHERE id = ANY(%s::uuid[]) AND project_id = %s FOR UPDATE", (pending, project_id))
                        rows = cursor.fetchall()
                        if len(rows) != len(pending) or any(ref is None for _, ref in rows):
                            raise RuntimeError(f"{kind} rows changed during migration")
                        refs = list(executor.map(lambda row: _copy(source, destination, str(row[1]), cache), rows))
                        cursor.executemany(f"UPDATE {schema}.{kind} SET {column} = %s WHERE id = %s", [(ref, row[0]) for row, ref in zip(rows, refs)])
                        if kind == "telemetry_sources":
                            cursor.executemany(f"""UPDATE {schema}.raw_assets SET metadata = jsonb_set(metadata, '{{source_payload_key}}', to_jsonb(%s::text))
                                WHERE id = (SELECT raw_asset_id FROM {schema}.telemetry_sources WHERE id = %s)
                                  AND metadata->>'source_payload_key' = %s""", [(ref, row[0], str(row[1])) for row, ref in zip(rows, refs)])
                        cursor.executemany(f"""INSERT INTO {schema}.skerry_cold_migration
                            (project_id, kind, row_id, destination) VALUES (%s, %s, %s, %s)""", [(project_id, kind, row[0], destination_id) for row in rows])
                    connection.commit()
                    counts["telemetry_sources" if kind == "telemetry_sources" else "embeddings"] += len(rows)
            continue
        from concurrent.futures import ThreadPoolExecutor

        def copy_frame(row):
            values = []
            for i in (1, 4, 7):
                ref = row[i] or row[i + 1]
                meta = dict(row[i + 2] or {})
                if i == 1:
                    ref = ref or meta.get("kvstore_key") or meta.get("kvstore_hash")
                new_ref = _copy(source, destination, str(ref), cache) if ref else None
                if i == 1 and new_ref is None:
                    raise RuntimeError(f"Frame {row[0]} has no object reference")
                if new_ref:
                    for key in ("kvstore_key", "kvstore_hash"):
                        if key in meta:
                            meta[key] = new_ref
                values.extend((new_ref, new_ref, Jsonb(meta)))
            return (*values, row[0])

        with ThreadPoolExecutor(max_workers=16) as executor:
            for offset in range(0, len(ids), 128):
                batch = ids[offset:offset + 128]
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT row_id, destination FROM {schema}.skerry_cold_migration WHERE project_id = %s AND kind = %s AND row_id = ANY(%s::uuid[])", (project_id, kind, batch))
                    checkpoints = dict(cursor.fetchall())
                    if any(value != destination_id for value in checkpoints.values()):
                        raise RuntimeError("Migration destination differs from the existing checkpoint")
                    pending = [row_id for row_id in batch if row_id not in checkpoints]
                    if not pending:
                        connection.rollback()
                        continue
                    cursor.execute(f"""SELECT id, payload_ref, kvstore_hash, metadata,
                        preprocessed_payload_ref, preprocessed_kvstore_hash, preprocessed_metadata,
                        background_payload_ref, background_kvstore_hash, background_metadata
                        FROM {schema}.frames WHERE id = ANY(%s::uuid[]) FOR UPDATE""", (pending,))
                    rows = cursor.fetchall()
                    if len(rows) != len(pending):
                        raise RuntimeError("Frames changed during migration")
                    values = list(executor.map(copy_frame, rows))
                    cursor.executemany(f"""UPDATE {schema}.frames SET payload_ref = %s, kvstore_hash = %s, metadata = %s,
                        preprocessed_payload_ref = %s, preprocessed_kvstore_hash = %s, preprocessed_metadata = %s,
                        background_payload_ref = %s, background_kvstore_hash = %s, background_metadata = %s
                        WHERE id = %s""", values)
                    cursor.executemany(f"""INSERT INTO {schema}.skerry_cold_migration (project_id, kind, row_id, destination)
                        VALUES (%s, %s, %s, %s)""", [(project_id, kind, row[0], destination_id) for row in rows])
                connection.commit()
                counts["frames"] += len(rows)
        continue
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="PostgreSQL connection string")
    parser.add_argument("--schema", default="pelagia")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--legacy-backend", choices=("kvstore", "kvstore2"), default="kvstore2")
    parser.add_argument("--skerry-url", required=True)
    parser.add_argument("--token-env", default="SKERRY_TOKEN", help="Environment variable containing the SkerryKV token")
    args = parser.parse_args()
    import psycopg
    from ..storage.kvstore import KVStore
    from ..storage.kvstore2 import KVStore2
    from ..storage.skerrykv import SkerryKVStore

    source = (KVStore2 if args.legacy_backend == "kvstore2" else KVStore)(args.legacy_root)
    if not source.initialized:
        raise RuntimeError("Legacy store is not initialized")
    destination = SkerryKVStore(args.skerry_url, token=os.environ.get(args.token_env))
    try:
        with psycopg.connect(args.dsn) as connection:
            print(migrate_project(connection, args.schema, args.project_id, source, destination, args.skerry_url.rstrip("/")))
    finally:
        close_source = getattr(source, "close", None)
        if close_source:
            close_source()
        close = getattr(destination, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
