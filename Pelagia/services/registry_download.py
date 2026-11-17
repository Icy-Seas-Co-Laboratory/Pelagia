from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from oracle_data_contracts.datasets import initialize_database, validate_database

from .registry_transfer import RegistryTransferError, _write_workspace


def snapshot_sqlite_workspace(repository, destination: Path, *, project_id: str, owner_username: str) -> None:
    """Write a consistent portable snapshot without changing the workspace's source or revision."""
    with repository.connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cursor.execute(
            f"""SELECT * FROM {repository.schema}.registry_workspaces
            WHERE project_id=%s AND owner_username=%s AND is_active AND status='loaded'""",
            (project_id, owner_username),
        )
        workspace = cursor.fetchone()
        if workspace is None:
            raise RegistryTransferError("No Registry dataset is loaded")
        with sqlite3.connect(destination) as output:
            initialize_database(
                output, workspace["dataset_type"], dataset_id=workspace["dataset_id"],
                revision_id=str(uuid.uuid4()), parent_revision_id=workspace["revision_id"],
                name=workspace["name"], title=workspace["title"], description=workspace["description"],
                version=workspace["dataset_version"], metadata=workspace["metadata"],
            )
            _write_workspace(repository, cursor, repository.schema, workspace["id"], output, workspace)
            output.commit()
            report = validate_database(output)
            if not report["valid"]:
                raise RegistryTransferError("Download validation failed: " + "; ".join(report["errors"]))
