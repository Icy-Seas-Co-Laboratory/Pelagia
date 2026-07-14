"""Live file-browser preview endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


def _file_entry(path: Path, current_root: Path, browser_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(current_root)),
        "root_relative_path": str(path.relative_to(browser_root)),
        "is_dir": path.is_dir(),
        "size_bytes": None if path.is_dir() else stat.st_size,
        "modified_at": stat.st_mtime,
    }


def _root_entry(path: Path, *, key: str, label: str) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "key": key,
        "name": label,
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir() if exists else False,
        "size_bytes": None,
        "modified_at": None if stat is None else stat.st_mtime,
    }


if APIRouter is not None:
    from ._common import as_response, get_context

    router = APIRouter(prefix="/live", tags=["live"])

    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _configured_file_roots(request: Request) -> list[dict[str, Any]]:
        context = get_context(request)
        browser = context.config.file_browser
        candidates: list[tuple[str, str, Path]] = [
            ("import", "Raw Asset Import Directory", browser.root_path_import_dir),
            ("kvstore", "KVStore Root", browser.root_path_kvstore or context.config.kvstore.root_path),
        ]
        candidates.extend(
            (f"allowed_{index}", path.name or str(path), path)
            for index, path in enumerate(browser.allowed_root_paths, start=1)
        )
        roots: list[dict[str, Any]] = []
        seen: set[str] = set()
        for key, label, path in candidates:
            resolved = Path(path).expanduser().resolve()
            if str(resolved) not in seen:
                seen.add(str(resolved))
                roots.append({"key": key, "label": label, "path": resolved})
        return roots

    def _resolve_directory(request: Request, directory: str) -> tuple[Path, dict[str, Any]]:
        resolved = Path(directory).expanduser().resolve()
        for root in _configured_file_roots(request):
            if resolved == root["path"] or _is_relative_to(resolved, root["path"]):
                return resolved, root
        raise HTTPException(status_code=403, detail="Directory is outside the configured file browser roots.")

    @router.get("/files")
    def list_server_files(
        request: Request,
        directory: str | None = None,
        recursive: bool = False,
        include_hidden: bool = False,
        limit: int = 500,
    ) -> dict:
        if limit < 1:
            raise HTTPException(status_code=422, detail="limit must be >= 1.")
        roots = _configured_file_roots(request)
        root_entries = [_root_entry(root["path"], key=str(root["key"]), label=str(root["label"])) for root in roots]
        if directory is None:
            entries = root_entries[:limit]
            return as_response({"directory": None, "root": None, "roots": root_entries, "recursive": recursive, "include_hidden": include_hidden, "limit": limit, "count": len(entries), "entries": entries})
        root, browser_root = _resolve_directory(request, directory)
        browser_root_path = browser_root["path"]
        if not root.exists():
            raise HTTPException(status_code=404, detail=f"Directory {str(root)!r} was not found.")
        if not root.is_dir():
            raise HTTPException(status_code=422, detail=f"{str(root)!r} is not a directory.")
        entries = []
        for path in root.rglob("*") if recursive else root.iterdir():
            try:
                if not include_hidden and any(part.startswith(".") for part in path.relative_to(root).parts):
                    continue
                if not _is_relative_to(path.resolve(), browser_root_path):
                    continue
                entries.append(_file_entry(path, root, browser_root_path))
            except OSError:
                continue
            if len(entries) >= limit:
                break
        entries.sort(key=lambda item: (not item["is_dir"], item["relative_path"].lower()))
        parent = root.parent.resolve()
        return as_response({
            "directory": str(root),
            "root": _root_entry(browser_root_path, key=str(browser_root["key"]), label=str(browser_root["label"])),
            "roots": root_entries,
            "parent_directory": str(parent) if root != browser_root_path and _is_relative_to(parent, browser_root_path) else None,
            "recursive": recursive,
            "include_hidden": include_hidden,
            "limit": limit,
            "count": len(entries),
            "entries": entries,
        })
else:
    router = None
