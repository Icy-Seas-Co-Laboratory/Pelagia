from Pelagia.config import CoreConfig
from Pelagia.services.context import AppContext
from Pelagia.services.system_usage import SystemUsageService


class UsageCursor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query):
        assert "pg_database_size" in query

    def fetchone(self):
        return {
            "data_directory": "/tmp",
            "database_name": "pelagia",
            "database_bytes": 4096,
        }


class UsageConnection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return UsageCursor()


class UsageRepository:
    def connect(self):
        return UsageConnection()


def test_system_usage_includes_postgres_storage_location_and_size(tmp_path):
    config = CoreConfig()
    config.kvstore.root_path = tmp_path / "kvstore"
    config.file_browser.root_path_import_dir = tmp_path / "import"

    usage = SystemUsageService(AppContext(config=config, repository=UsageRepository())).snapshot()

    assert usage["storage"]["kvstore_default"]["resolved_path"] == str(tmp_path / "kvstore")
    assert usage["storage"]["raw_assets_default"]["resolved_path"] == str(tmp_path / "import")
    database_storage = usage["storage"]["database"]["storage"]
    assert database_storage["available"] is True
    assert database_storage["database_name"] == "pelagia"
    assert database_storage["database_bytes"] == 4096
    assert database_storage["data_directory"] == "/tmp"
    assert database_storage["filesystem"]["available"] is True
