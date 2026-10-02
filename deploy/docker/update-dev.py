#!/usr/bin/env python3
"""Update the three main branches and the single-host development deployment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPOS = {
    "pelagia": ROOT / "Pelagia",
    "pelagiaview": ROOT / "PelagiaView",
    "oracle-builder": ROOT / "oracle-builder",
}
PELAGIA_COMPOSE = HERE / "compose.yaml"
ORACLE_COMPOSE = REPOS["oracle-builder"] / "deploy/docker/compose.yaml"
STATE_FILE = HERE / "runtime/dev-update.json"
MANIFEST_FILE = HERE / "runtime/dev-deployment.json"

ORACLE_REQUEST = r"""
const [method, path, body] = process.argv.slice(1);
const token = process.env.ORCHESTRATOR_OPERATOR_TOKEN;
const base = process.env.ORCHESTRATOR_URL;
if (!token || !base) throw new Error('Oracle operator URL/token is unavailable');
const response = await fetch(new URL(path, base), {
  method,
  headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
  body: body || undefined
});
const value = await response.json();
if (!response.ok) throw new Error(`Oracle HTTP ${response.status}: ${JSON.stringify(value)}`);
console.log(JSON.stringify(value));
"""


def run(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def git(repo: Path, *args: str) -> str:
    return run("git", *args, cwd=repo, capture=True)


def compose(file: Path, *args: str, capture: bool = False) -> str:
    return run(
        "docker", "compose", "--env-file", str(file.parent / ".env"),
        "-f", str(file), *args, capture=capture,
    )


def oracle_request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = compose(
        ORACLE_COMPOSE,
        "exec", "-T", "webgui", "node", "-e", ORACLE_REQUEST,
        method, path, json.dumps(body) if body is not None else "",
        capture=True,
    )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected Oracle response from {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check_repositories() -> None:
    for name, repo in REPOS.items():
        if not (repo / ".git").exists():
            raise RuntimeError(f"Missing sibling checkout: {repo}")
        if git(repo, "branch", "--show-current") != "main":
            raise RuntimeError(f"{name} must be on main")
        if git(repo, "status", "--porcelain"):
            raise RuntimeError(f"{name} has uncommitted changes; commit or stash them first")


def fetch_and_fast_forward() -> None:
    # Verify every remote can advance before changing any local branch.
    for name, repo in REPOS.items():
        print(f"Fetching {name}/main", flush=True)
        run("git", "fetch", "origin", "main", cwd=repo)
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"),
            cwd=repo,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{name}/main diverged from origin/main; update it manually")
    for name, repo in REPOS.items():
        print(f"Fast-forwarding {name}/main", flush=True)
        run("git", "merge", "--ff-only", "FETCH_HEAD", cwd=repo)


def running_oracle_deployments() -> list[str]:
    deployments = oracle_request("GET", "/v1/worker-deployments").get("deployments")
    if not isinstance(deployments, list):
        raise RuntimeError("Oracle did not return a deployment list")
    running = []
    for deployment in deployments:
        if not isinstance(deployment, dict):
            raise RuntimeError("Oracle returned an invalid deployment")
        if deployment.get("desired_state") == "running":
            if deployment.get("provider") != "docker":
                raise RuntimeError("A non-Docker Oracle deployment is running; update it separately")
            running.append(str(deployment["deployment_id"]))
    return running


def require_idle_oracle() -> None:
    leases = oracle_request("GET", "/v1/worker-leases").get("leases")
    if not isinstance(leases, list):
        raise RuntimeError("Oracle did not return a lease list")
    active = [lease for lease in leases if isinstance(lease, dict) and lease.get("status") == "active"]
    if active:
        raise RuntimeError(f"Oracle has {len(active)} active worker lease(s); wait for jobs to finish and retry")


def stop_oracle_deployments(ids: list[str]) -> None:
    for deployment_id in ids:
        oracle_request(
            "POST", f"/v1/worker-deployments/{deployment_id}:stop", {"timeout_seconds": 120}
        )
        # The provider stops but retains the named container. Remove that stopped
        # instance so the same deployment ID can start from the newly built image.
        run("docker", "rm", f"oracle-worker-{deployment_id}")


def start_oracle_deployments(ids: list[str]) -> None:
    for deployment_id in ids:
        oracle_request("POST", f"/v1/worker-deployments/{deployment_id}:start")


def verify_health() -> None:
    checks = (
        (ORACLE_COMPOSE, ("exec", "-T", "orchestrator", "python", "-c",
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8110/health/ready', timeout=10)")),
        (PELAGIA_COMPOSE, ("exec", "-T", "api", "python", "-c",
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/postgres', timeout=10)")),
        (PELAGIA_COMPOSE, ("exec", "-T", "view", "node", "-e",
            "const r = await fetch('http://127.0.0.1:3000/'); if (!r.ok) process.exit(1)")),
    )
    for file, args in checks:
        for attempt in range(12):
            try:
                compose(file, *args)
                break
            except subprocess.CalledProcessError:
                if attempt == 11:
                    raise
                time.sleep(5)
    compose(PELAGIA_COMPOSE, "exec", "-T", "api", "pelagia", "check-system")


def record_manifest() -> None:
    images = {}
    for name, image in {
        "pelagia": "pelagia/backend:local",
        "pelagiaview": "pelagia/view:local",
        "oracle-orchestrator": "oracle-builder/orchestrator:demo",
        "oracle-worker": "oracle-builder/worker:demo",
        "oracle-webgui": "oracle-builder/webgui:demo",
    }.items():
        images[name] = run("docker", "image", "inspect", "--format", "{{.Id}}", image, capture=True)
    write_json(MANIFEST_FILE, {
        "channel": "main",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "commits": {name: git(repo, "rev-parse", "HEAD") for name, repo in REPOS.items()},
        "images": images,
    })


def verify_oracle_containers(ids: list[str]) -> None:
    for deployment_id in ids:
        status = run(
            "docker", "inspect", "--format", "{{.State.Status}}",
            f"oracle-worker-{deployment_id}", capture=True,
        )
        if status != "running":
            raise RuntimeError(f"Oracle deployment {deployment_id} is {status}, not running")
    if ids:
        for attempt in range(12):
            try:
                compose(
                    ORACLE_COMPOSE, "exec", "-T", "orchestrator", "python", "-c",
                    "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8110/health/ready', timeout=10)); assert data['status'] == 'ready'",
                )
                break
            except subprocess.CalledProcessError:
                if attempt == 11:
                    raise
                time.sleep(5)


def update() -> None:
    if STATE_FILE.exists():
        raise RuntimeError(f"An earlier update is unfinished: {STATE_FILE}")
    check_repositories()
    run("docker", "compose", "version")
    run("docker", "info", "--format", "{{.ServerVersion}}")
    for path in (HERE / ".env", ORACLE_COMPOSE.parent / ".env"):
        if not path.is_file():
            raise RuntimeError(f"Missing deployment configuration: {path}")
    compose(PELAGIA_COMPOSE, "config", "--quiet")
    compose(ORACLE_COMPOSE, "config", "--quiet")
    worker_count = len(compose(PELAGIA_COMPOSE, "ps", "-q", "worker", capture=True).splitlines())
    if worker_count < 1:
        raise RuntimeError("No running Pelagia worker was found; start the stack before updating")
    running_oracle_deployments()  # Require the Oracle control plane before pulling code.
    fetch_and_fast_forward()

    print("Building development images", flush=True)
    compose(ORACLE_COMPOSE, "--profile", "image", "build", "orchestrator", "webgui", "worker-image")
    compose(PELAGIA_COMPOSE, "build", "api", "worker", "view")
    require_idle_oracle()
    ids = running_oracle_deployments()
    write_json(STATE_FILE, {"oracle_deployments": ids, "pelagia_worker_count": worker_count, "step": "stopping"})
    print("Stopping Pelagia services and idle Oracle deployments", flush=True)
    compose(PELAGIA_COMPOSE, "stop", "-t", "86400", "worker")
    compose(PELAGIA_COMPOSE, "stop", "api", "proxy")
    try:
        require_idle_oracle()
    except RuntimeError:
        # No migration has run. Restore the old containers if another client
        # acquired an Oracle lease during the Pelagia worker shutdown.
        compose(PELAGIA_COMPOSE, "start", "api", "worker", "proxy")
        STATE_FILE.unlink()
        raise
    stop_oracle_deployments(ids)

    print("Applying migrations and starting new services", flush=True)
    write_json(STATE_FILE, {"oracle_deployments": ids, "pelagia_worker_count": worker_count, "step": "migrating"})
    compose(PELAGIA_COMPOSE, "run", "--rm", "api", "pelagia", "migrate-db")
    compose(ORACLE_COMPOSE, "up", "-d", "orchestrator", "webgui", "caddy")
    compose(PELAGIA_COMPOSE, "up", "-d", "--scale", f"worker={worker_count}", "api", "worker", "view", "proxy")
    verify_health()
    start_oracle_deployments(ids)
    verify_oracle_containers(ids)
    record_manifest()
    STATE_FILE.unlink()
    print(f"Update complete. Deployed commits and image IDs: {MANIFEST_FILE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check local branches and checkout state only")
    args = parser.parse_args()
    try:
        if args.check:
            check_repositories()
            print("All three repositories are clean and on main")
        else:
            update()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Update stopped: {exc}", file=sys.stderr)
        if STATE_FILE.exists():
            print(f"Services may need attention. See {STATE_FILE}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
