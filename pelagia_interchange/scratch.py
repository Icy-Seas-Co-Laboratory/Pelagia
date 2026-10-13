"""Recoverable local scratch paths for video ingestion."""
from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path

from .exceptions import FormatError


def scratch_root(output: str | Path, base: str | Path | None = None, *, create: bool = False) -> Path:
    """Return a stable, private scratch directory for one output dataset."""
    parent = (Path(base).expanduser().resolve() if base is not None
              else Path(tempfile.gettempdir()).resolve())
    identity = hashlib.sha256(str(Path(output).expanduser().resolve()).encode()).hexdigest()[:24]
    root = parent / "pelagia-interchange" / identity
    if create:
        if not parent.is_dir():
            raise ValueError(f"staging directory does not exist: {parent}")
        for directory in (root.parent, root, root / "partials"):
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                pass
            details = directory.lstat()
            if (not stat.S_ISDIR(details.st_mode) or
                    (os.name == "posix" and (details.st_mode & 0o077 or details.st_uid != os.getuid()))):
                raise FormatError(f"scratch directory is not private: {directory}")
    return root
