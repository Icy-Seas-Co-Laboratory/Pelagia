from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

__version__ = "0.0.1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_repo_root()), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


@lru_cache(maxsize=1)
def build_info() -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    short_commit = _git_output("rev-parse", "--short", "HEAD")
    dirty_status = _git_output("status", "--porcelain")
    return {
        "version": __version__,
        "git_commit": commit,
        "git_commit_short": short_commit,
        "git_dirty": None if dirty_status is None else bool(dirty_status),
    }
