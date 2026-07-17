from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ENVIRONMENT_SCRIPT = ROOT_DIR / "scripts" / "pelagia_env.py"


def test_environment_bootstrap_sync_dry_run(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='pelagia-test'\nversion='0.0.0'\nrequires-python='>=3.12,<3.13'\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ENVIRONMENT_SCRIPT),
            "sync",
            "cpu",
            "--root",
            str(tmp_path),
            "--python",
            sys.executable,
            "--dry-run",
        ],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["profile"] == "cpu"
    assert body["dry_run"] is True
    assert body["venv"] == str(tmp_path / ".venv")
    assert body["manager"] == "uv"
    assert body["python_request"] == sys.executable
    assert "worker-cpu" in body["extras"]
    assert body["commands"][0][1] == "sync"
    assert "--locked" in body["commands"][0]
