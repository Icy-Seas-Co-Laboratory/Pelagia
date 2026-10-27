import pytest

from Pelagia.services.migrate_cold_storage import _copy


class Store:
    def __init__(self, data=None, *, corrupt=False):
        self.data = data or {}
        self.corrupt = corrupt
        self.writes = 0

    def get_store(self, key):
        value = self.data[key]
        return b"corrupt" if self.corrupt else value

    def put_store(self, payload):
        self.writes += 1
        self.data["new-hash"] = payload
        return "new-hash"


def test_cold_migration_verifies_and_deduplicates_copies():
    source = Store({"old-hash": b"frame"})
    destination = Store()
    cache = {}
    assert _copy(source, destination, "old-hash", cache) == "new-hash"
    assert _copy(source, destination, "old-hash", cache) == "new-hash"
    assert destination.writes == 1


def test_cold_migration_rejects_bad_readback():
    with pytest.raises(RuntimeError, match="verification failed"):
        _copy(Store({"old-hash": b"frame"}), Store(corrupt=True), "old-hash", {})


def test_cold_migration_updates_project_frame_and_resumes(postgres_repo):
    import psycopg
    from test_incremental_processing_submissions import _project_with_frames
    from test_postgres import POSTGRES_TEST_DSN
    from Pelagia.services.migrate_cold_storage import migrate_project

    project_id, asset_id = _project_with_frames(postgres_repo, count=1)
    frame = postgres_repo.list_frames(asset_id)[0]
    source = Store({str(frame["payload_ref"]): b"frame bytes"})
    destination = Store()
    with psycopg.connect(POSTGRES_TEST_DSN) as connection:
        result = migrate_project(connection, postgres_repo.schema, project_id, source, destination, "test-store")
        assert result == {"frames": 1, "embeddings": 0, "telemetry_sources": 0}
        assert migrate_project(connection, postgres_repo.schema, project_id, source, destination, "test-store") == {"frames": 0, "embeddings": 0, "telemetry_sources": 0}
        with pytest.raises(RuntimeError, match="destination differs"):
            migrate_project(connection, postgres_repo.schema, project_id, source, destination, "other-store")
    updated = postgres_repo.list_frames(asset_id)[0]
    assert updated["payload_ref"] == "new-hash"
    assert destination.data["new-hash"] == b"frame bytes"


from test_postgres import postgres_repo  # noqa: E402, F401 -- shared PostgreSQL fixture
