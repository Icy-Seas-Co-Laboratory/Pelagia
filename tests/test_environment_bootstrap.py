from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ENVIRONMENT_SCRIPT = ROOT_DIR / "scripts" / "pelagia_env.py"


def test_environment_bootstrap_sync_dry_run(tmp_path):
    (tmp_path / "requirements-worker-cpu.txt").write_text("", encoding="utf-8")

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
