from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from Pelagia.config import CoreConfig
from Pelagia.services.context import AppContext
import Pelagia.cli.app as cli_module


class _FakeRepository:
    def __init__(self):
        self.users_by_username = {}
        self.users_by_id = {}
        self.projects_by_key = {}
        self.projects_by_id = {}
        self.memberships = {}
        self.sessions = {}
        self.initialized = False
        self.purged = False

    def initialize_schema(self):
        self.initialized = True

    def schema_status(self):
        return {"schema": "pelagia", "ready": True, "missing_tables": []}

    def purge_all(self):
        self.purged = True
        return {"schema": "pelagia", "total_rows_deleted": 0}

    def create_user(self, username, **kwargs):
        user = {
            "id": f"user-{len(self.users_by_username) + 1}",
            "username": username,
            "display_name": kwargs.get("display_name"),
            "is_admin": bool(kwargs.get("is_admin", False)),
            "is_active": bool(kwargs.get("is_active", True)),
        }
        self.users_by_username[username] = user
        self.users_by_id[user["id"]] = user
        return dict(user)

    def get_user_by_username(self, username):
        user = self.users_by_username.get(username)
        return None if user is None else dict(user)

    def create_project(self, project_key, **kwargs):
        project = {
            "id": f"project-{len(self.projects_by_key) + 1}",
            "project_key": project_key,
            "project_name": kwargs.get("project_name") or project_key,
            "description": kwargs.get("description"),
            "kvstore_root_path": kwargs.get("kvstore_root_path"),
            "is_active": bool(kwargs.get("is_active", True)),
        }
        self.projects_by_key[project_key] = project
        self.projects_by_id[project["id"]] = project
        return dict(project)

    def get_project_by_key(self, project_key):
        project = self.projects_by_key.get(project_key)
        return None if project is None else dict(project)

    def get_project(self, project_id):
        project = self.projects_by_id.get(project_id)
        return None if project is None else dict(project)

    def add_project_member(self, user_id, project_id, *, role):
        membership = {"user_id": user_id, "project_id": project_id, "role": role}
        self.memberships[(user_id, project_id)] = membership
        return dict(membership)

    def create_session(self, user_id, project_id, **kwargs):
        token = f"token-{len(self.sessions) + 1}"
        session = {
            "id": f"session-{len(self.sessions) + 1}",
            "token": token,
            "user_id": user_id,
            "project_id": project_id,
            "expires_at": "later",
            "ttl_seconds": kwargs.get("ttl_seconds"),
        }
        self.sessions[token] = session
        return dict(session)

    def list_projects(self, **kwargs):
        return [dict(project) for project in self.projects_by_key.values()]

    def list_user_projects(self, user_id):
        rows = []
        for (member_user_id, project_id), membership in self.memberships.items():
            if member_user_id != user_id:
                continue
            project = dict(self.projects_by_id[project_id])
            project["role"] = membership["role"]
            rows.append(project)
        return rows


class _FakeKVStore:
    initialized = True

    def status(self):
        return {"initialized": True}


def _install_fake_context(monkeypatch):
    config = CoreConfig()
    repo = _FakeRepository()
    context = AppContext(config=config, repository=repo, kvstore=_FakeKVStore())
    monkeypatch.setattr(cli_module, "_context_from_options", lambda *args, **kwargs: context)
    monkeypatch.setattr(
        cli_module,
        "initialize_project_kvstore",
        lambda context, project: {
            "initialized": True,
            "root_path": f"/tmp/pelagia-kv/projects/{project['id']}",
        },
    )
    return repo


def test_cli_create_dev_login_bootstraps_user_project_and_session(monkeypatch):
    repo = _install_fake_context(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "create-dev-login",
            "--username",
            "ada",
            "--password",
            "secret",
            "--project-key",
            "reef",
            "--ttl-seconds",
            "120",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["username"] == "ada"
    assert body["password"] == "secret"
    assert body["password_applied"] is True
    assert body["project"]["project_key"] == "reef"
    assert body["kvstore"]["initialized"] is True
    assert body["kvstore"]["root_path"] == "/tmp/pelagia-kv/projects/project-1"
    assert body["token"] == "token-1"
    assert repo.memberships[("user-1", "project-1")]["role"] == "admin"
    assert repo.sessions["token-1"]["ttl_seconds"] == 120


def test_cli_create_dev_login_bootstraps_projectless_admin(monkeypatch):
    repo = _install_fake_context(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "create-dev-login",
            "--username",
            "ada",
            "--password",
            "secret",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["username"] == "ada"
    assert body["password"] == "secret"
    assert body["password_applied"] is True
    assert body["project_creation_required"] is True
    assert body["project"] is None
    assert body["kvstore"] is None
    assert body["membership"] is None
    assert body["token"] is None
    assert body["session"] is None
    assert repo.projects_by_key == {}
    assert repo.memberships == {}
    assert repo.sessions == {}


def test_cli_create_dev_login_does_not_print_password_for_existing_user(monkeypatch):
    repo = _install_fake_context(monkeypatch)
    repo.create_user("ada", password="old-secret", is_admin=True)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "create-dev-login",
            "--username",
            "ada",
            "--password",
            "new-secret",
            "--project-key",
            "reef",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["username"] == "ada"
    assert body["password"] is None
    assert body["password_applied"] is False
    assert body["token"] == "token-1"


def test_cli_create_project_user_membership_and_list(monkeypatch):
    _install_fake_context(monkeypatch)
    runner = CliRunner()

    assert runner.invoke(cli_module.app, ["create-user", "ben", "--password", "secret"]).exit_code == 0
    create_project = runner.invoke(cli_module.app, ["create-project", "survey", "--project-name", "Survey"])
    assert create_project.exit_code == 0
    created_project = json.loads(create_project.output)
    assert created_project["kvstore"]["initialized"] is True
    assert runner.invoke(cli_module.app, ["add-project-user", "ben", "survey", "--role", "viewer"]).exit_code == 0
    result = runner.invoke(cli_module.app, ["list-projects", "--username", "ben"])

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["projects"] == [
        {
            "description": None,
            "id": "project-1",
            "is_active": True,
            "kvstore_root_path": str((Path("data/kvstores") / "survey").resolve()),
            "project_key": "survey",
            "project_name": "Survey",
            "role": "viewer",
        }
    ]


def test_cli_reset_allows_projectless_system_without_kvstore(monkeypatch):
    repo = _install_fake_context(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(cli_module.app, ["reset", "--delete"])

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["deleted"] is True
    assert body["database"]["total_rows_deleted"] == 0
    assert body["kvstores"] == {
        "project_count": 0,
        "reset_count": 0,
        "results": [],
    }
    assert repo.initialized is True
    assert repo.purged is True


def test_cli_check_system_treats_projectless_storage_as_ready(monkeypatch):
    _install_fake_context(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(cli_module.app, ["check-system"])

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["ready"] is True
    assert body["kvstore"] == {
        "required": False,
        "initialized": True,
        "project_count": 0,
        "stores": [],
    }


def test_cli_reset_resets_configured_project_kvstores(monkeypatch, tmp_path):
    repo = _install_fake_context(monkeypatch)
    root_path = tmp_path / "stores" / "reef"
    repo.create_project("reef", kvstore_root_path=str(root_path))
    reset_calls = []

    class _ProjectStore:
        initialized = True

    monkeypatch.setattr(cli_module, "create_kvstore", lambda root, config: _ProjectStore())
    monkeypatch.setattr(
        cli_module,
        "reset_kvstore",
        lambda store, config: reset_calls.append(store) or {"root_path": str(root_path)},
    )
    runner = CliRunner()

    result = runner.invoke(cli_module.app, ["reset", "--delete"])

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["kvstores"]["project_count"] == 1
    assert body["kvstores"]["reset_count"] == 1
    assert body["kvstores"]["results"][0]["status"] == "reset"
    assert len(reset_calls) == 1
    assert repo.purged is True


def test_cli_environment_sync_dry_run_reports_profile_commands(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "env",
            "sync",
            "cpu",
            "--root",
            str(tmp_path),
            "--python",
            sys.executable,
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "Requirements file was not found" in result.output

    (tmp_path / "requirements-worker-cpu.txt").write_text("", encoding="utf-8")
    result = runner.invoke(
        cli_module.app,
        [
            "env",
            "sync",
            "cpu",
            "--root",
            str(tmp_path),
            "--python",
            sys.executable,
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["profile"] == "cpu"
    assert body["venv"] == str(tmp_path / ".venv")
    assert body["dry_run"] is True
