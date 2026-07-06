from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_auth, scoped_project_id
    from ._common import as_response, get_context, get_repository

    def _bounded_limit(limit: int | None) -> int:
        return min(max(1, 100 if limit is None else limit), 1000)

    def _bounded_offset(offset: int | None) -> int:
        return max(0, 0 if offset is None else offset)

    def _bounded_tail_lines(tail_lines: int | None) -> int:
        return min(max(1, 200 if tail_lines is None else tail_lines), 5000)

    def _bounded_max_bytes(max_bytes: int | None) -> int:
        return min(max(1024, 256 * 1024 if max_bytes is None else max_bytes), 5 * 1024 * 1024)

    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _log_roots(request: Request) -> dict[str, Path]:
        context = get_context(request)
        roots: dict[str, Path] = {}

        def add_root(key: str, path: str | os.PathLike[str] | None) -> None:
            if path is None:
                return
            try:
                resolved = Path(path).expanduser().resolve(strict=False)
            except OSError:
                return
            if resolved.is_dir() and resolved not in roots.values():
                roots[key] = resolved

        add_root("configured", context.config.logging.log_path)
        run_dir = os.environ.get("PELAGIA_RUN_DIR")
        if run_dir:
            add_root("runtime:env", Path(run_dir) / "logs")

        base_run_dir = _repo_root() / ".pelagia" / "run"
        add_root("runtime", base_run_dir / "logs")
        if base_run_dir.is_dir():
            for candidate in sorted(base_run_dir.iterdir()):
                if candidate.is_dir():
                    add_root(f"runtime:{candidate.name}", candidate / "logs")
        return roots

    def _file_summary(root_key: str, root_path: Path, path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "root": root_key,
            "root_path": str(root_path),
            "name": path.name,
            "relative_path": path.relative_to(root_path).as_posix(),
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
        }

    def _safe_relative_path(value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise HTTPException(status_code=422, detail="Log file path must be a relative file name below a known log root.")
        return relative

    def _resolve_log_file(request: Request, file_name: str, root: str | None) -> tuple[str, Path, Path]:
        roots = _log_roots(request)
        if not roots:
            raise HTTPException(status_code=404, detail="No physical log roots were found.")
        relative = _safe_relative_path(file_name)
        if root is not None:
            root_path = roots.get(root)
            if root_path is None:
                raise HTTPException(status_code=404, detail=f"Log root {root!r} was not found.")
            candidates = [(root, root_path)]
        else:
            candidates = list(roots.items())

        matches: list[tuple[str, Path, Path]] = []
        for root_key, root_path in candidates:
            path = (root_path / relative).resolve(strict=False)
            try:
                path.relative_to(root_path)
            except ValueError:
                continue
            if path.is_file():
                matches.append((root_key, root_path, path))

        if not matches:
            raise HTTPException(status_code=404, detail=f"Log file {file_name!r} was not found.")
        if root is None and len(matches) > 1:
            roots_with_file = [match[0] for match in matches]
            raise HTTPException(
                status_code=409,
                detail=f"Log file {file_name!r} exists in multiple roots; specify root. Matches: {', '.join(roots_with_file)}.",
            )
        return matches[0]

    def _read_tail(path: Path, *, tail_lines: int, max_bytes: int) -> tuple[str, list[str], bool]:
        size = path.stat().st_size
        truncated = size > max_bytes
        with path.open("rb") as handle:
            if truncated:
                handle.seek(max(0, size - max_bytes))
            payload = handle.read(max_bytes)
        text = payload.decode("utf-8", errors="replace")
        if truncated:
            _, _, text = text.partition("\n")
        lines = text.splitlines()
        if len(lines) > tail_lines:
            lines = lines[-tail_lines:]
            text = "\n".join(lines)
        return text, lines, truncated

    class CreateLogRequest(BaseModel):
        event_type: str
        message: str | None = None
        level: str = "info"
        logger: str = "pelagia.api"
        run_id: str | None = None
        asset_id: str | None = None
        job_id: str | None = None
        worker_id: str | None = None
        request_id: str | None = None
        duration_ms: float | None = None
        payload: dict[str, Any] = Field(default_factory=dict)

    router = APIRouter(prefix="/logs", tags=["logs"])

    @router.get("")
    def list_logs(
        request: Request,
        after_id: int | None = None,
        before_id: int | None = None,
        level: str | None = None,
        event_type: str | None = None,
        logger: str | None = None,
        run_id: str | None = None,
        asset_id: str | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        request_id: str | None = None,
        limit: int | None = 100,
        offset: int = 0,
    ) -> dict[str, list]:
        repository = get_repository(request)
        logs = repository.list_logs(
            project_id=scoped_project_id(request),
            after_id=after_id,
            before_id=before_id,
            level=level,
            event_type=event_type,
            logger=logger,
            run_id=run_id,
            asset_id=asset_id,
            job_id=job_id,
            worker_id=worker_id,
            request_id=request_id,
            limit=_bounded_limit(limit),
            offset=_bounded_offset(offset),
        )
        return {"logs": as_response(logs)}

    @router.get("/files")
    def list_log_files(request: Request, root: str | None = None, limit: int | None = 200) -> dict[str, Any]:
        require_auth(request)
        roots = _log_roots(request)
        selected_roots = {root: roots[root]} if root is not None and root in roots else roots
        if root is not None and root not in roots:
            raise HTTPException(status_code=404, detail=f"Log root {root!r} was not found.")
        files = []
        for root_key, root_path in selected_roots.items():
            for path in sorted(root_path.rglob("*.log")):
                if path.is_file():
                    files.append(_file_summary(root_key, root_path, path))
        files.sort(key=lambda item: item["modified_at"], reverse=True)
        resolved_limit = min(max(1, 200 if limit is None else limit), 1000)
        return {
            "roots": [{"key": key, "path": str(path)} for key, path in roots.items()],
            "files": as_response(files[:resolved_limit]),
        }

    @router.get("/files/{file_name:path}")
    def read_log_file(
        request: Request,
        file_name: str,
        root: str | None = None,
        tail_lines: int | None = 200,
        max_bytes: int | None = 256 * 1024,
    ) -> dict[str, Any]:
        require_auth(request)
        root_key, root_path, path = _resolve_log_file(request, file_name, root)
        resolved_tail_lines = _bounded_tail_lines(tail_lines)
        resolved_max_bytes = _bounded_max_bytes(max_bytes)
        text, lines, truncated = _read_tail(
            path,
            tail_lines=resolved_tail_lines,
            max_bytes=resolved_max_bytes,
        )
        return {
            "file": as_response(_file_summary(root_key, root_path, path)),
            "tail_lines": resolved_tail_lines,
            "max_bytes": resolved_max_bytes,
            "truncated": truncated,
            "line_count": len(lines),
            "text": text,
            "lines": lines,
        }

    @router.post("")
    def create_log(request: Request, body: CreateLogRequest) -> dict:
        repository = get_repository(request)
        row = repository.append_log(
            event_type=body.event_type,
            project_id=scoped_project_id(request),
            message=body.message,
            level=body.level,
            logger=body.logger,
            run_id=body.run_id,
            asset_id=body.asset_id,
            job_id=body.job_id,
            worker_id=body.worker_id,
            request_id=body.request_id,
            duration_ms=body.duration_ms,
            payload=body.payload,
        )
        return {"log": as_response(row)}
else:
    router = None
