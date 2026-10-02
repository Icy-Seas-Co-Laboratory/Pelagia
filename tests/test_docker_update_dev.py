from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy/docker/update-dev.py"
SPEC = importlib.util.spec_from_file_location("pelagia_update_dev", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
update_dev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update_dev)


def test_update_refuses_active_oracle_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        update_dev,
        "oracle_request",
        lambda *_args: {"leases": [{"lease_id": "one", "status": "active"}]},
    )
    with pytest.raises(RuntimeError, match="active worker lease"):
        update_dev.require_idle_oracle()


def test_update_refuses_unmanaged_running_oracle_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        update_dev,
        "oracle_request",
        lambda *_args: {"deployments": [
            {"deployment_id": "one", "provider": "local", "desired_state": "running"}
        ]},
    )
    with pytest.raises(RuntimeError, match="non-Docker"):
        update_dev.running_oracle_deployments()


def test_stopped_oracle_container_is_removed_before_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def request(method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path))
        assert body == {"timeout_seconds": 120}
        return {}

    def run(*args: str, **_kwargs: object) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(update_dev, "oracle_request", request)
    monkeypatch.setattr(update_dev, "run", run)
    update_dev.stop_oracle_deployments(["deploy-one"])
    assert calls == [
        ("POST", "/v1/worker-deployments/deploy-one:stop"),
        ("docker", "rm", "oracle-worker-deploy-one"),
    ]


def test_compose_uses_the_stack_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []

    def run(*args: str, **_kwargs: object) -> str:
        commands.append(args)
        return ""

    monkeypatch.setattr(update_dev, "run", run)
    update_dev.compose(update_dev.ORACLE_COMPOSE, "config", "--quiet")
    assert commands[0][2:6] == (
        "--env-file", str(update_dev.ORACLE_COMPOSE.parent / ".env"),
        "-f", str(update_dev.ORACLE_COMPOSE),
    )
