"""Append-only local operational output journal.

The journal deliberately records diagnostic output outside PostgreSQL.  Database
rows remain the authority for jobs, leases, controls, and their latest progress;
these files provide searchable operator history on a shared local disk.
"""

from __future__ import annotations

import fcntl
import heapq
import io
import json
import os
import re
import sys
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

import zstandard

from ..utils.serialization import json_ready


_MAX_EVENT_ID = 9_007_199_254_740_991  # JavaScript's largest exact integer.
_MAX_LINE_BYTES = 8 * 1024 * 1024
_STREAMS = frozenset(("logs", "events"))
_BUCKETS = frozenset(("high_frequency", "job_event", "routine_log", "important_log"))


class OutputStore:
    """A flock-coordinated JSONL journal for one shared local filesystem.

    Every record gets a global integer sequence so existing SSE clients can use
    their numeric cursor unchanged.  A record is written to exactly one stream:
    normal output to ``logs`` and job history to ``events``.  Log queries read
    both streams, except the high-volume progress event that is intentionally
    only present in the event stream.
    """

    def __init__(self, logging_config: Any) -> None:
        self.config = logging_config
        self.root = Path(logging_config.log_path)
        name = Path(str(logging_config.file_name)).stem or "pelagia"
        self.prefix = name
        self._lock_path = self.root / f".{name}.output.lock"
        self._maintenance_lock_path = self.root / f".{name}.output.maintenance.lock"
        self._sequence_path = self.root / f".{name}.output.sequence"
        self._highwater: int | None = None
        self._checkpoint_lock = Lock()
        self._active_checkpoints: OrderedDict[tuple[int, int, int], deque[tuple[int, int]]] = OrderedDict()
        streams = "|".join(sorted(_STREAMS))
        buckets = "|".join(sorted(_BUCKETS))
        self._sealed_pattern = re.compile(
            rf"^{re.escape(name)}\.({streams})\.({buckets})\.(\d{{16}})(?:-(\d{{16}}))?\.jsonl(?:\.zst)?$"
        )

    def append(self, record: dict[str, Any], *, stream: str = "logs") -> dict[str, Any]:
        """Append one JSON record under the journal lock."""
        if stream not in _STREAMS:
            raise ValueError(f"Unknown output stream {stream!r}.")
        payload = dict(json_ready(record))
        with self._locked():
            payload["stream"] = stream
            bucket = self._retention_bucket(payload)
            path = self._active_path(stream, bucket)
            self._repair_active_locked(path)
            sequence = self._next_sequence()
            payload["id"] = sequence
            payload.setdefault("event_id", str(uuid4()))
            payload.setdefault("schema_version", 1)
            payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            if len(encoded) > _MAX_LINE_BYTES:
                raise ValueError(f"Output record exceeds {_MAX_LINE_BYTES} bytes.")
            path.parent.mkdir(parents=True, exist_ok=True)
            rotation_due = path.exists() and path.stat().st_size and path.stat().st_size + len(encoded) > self.config.max_bytes
            interval = int(getattr(self.config, "rotation_interval_seconds", 0) or 0)
            if interval and path.exists() and path.stat().st_size and time.time() - self._active_started_at(path) >= interval:
                rotation_due = True
            if rotation_due:
                self._rotate_locked(stream, bucket)
                path = self._active_path(stream, bucket)
            with path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._write_sequence(sequence)
            self._highwater = sequence
        return payload

    def list_logs(
        self,
        *,
        project_id: str | None = None,
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
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if after_id is not None and after_id >= self._current_highwater():
            return []
        filters = {
            "project_id": project_id, "level": None if level is None else str(level).lower(),
            "event_type": event_type, "logger": logger, "run_id": run_id,
            "asset_id": asset_id, "job_id": job_id, "worker_id": worker_id,
            "request_id": request_id,
        }
        records = (
            row for row in self._records(("logs", "events"), after_id=after_id, before_id=before_id)
            if not (row.get("stream") == "events" and row.get("event_type") == "job.progress_updated")
            and not (row.get("stream") == "events" and row.get("project_id") is None)
            and self._matches(row, filters, after_id=after_id, before_id=before_id)
        )
        return self._page(records, limit=limit, offset=offset, ascending=False)

    def list_job_events(
        self,
        *,
        project_id: str | None = None,
        after_id: int | None = None,
        run_id: str | None = None,
        job_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        if after_id is not None and after_id >= self._current_highwater():
            return []
        records = (
            row for row in self._records(("events",), after_id=after_id)
            if self._matches(row, {"project_id": project_id, "run_id": run_id, "job_id": job_id}, after_id=after_id)
        )
        return self._page(records, limit=limit, offset=offset, ascending=ascending)

    def prune(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Compress sealed segments and remove segments whose every row expired.

        Segment deletion is conservative: a mixed-retention segment stays until
        its longest-lived record expires.  Active files are never removed.
        """
        result: dict[str, Any] = {"dry_run": dry_run, "acquired": False, "deleted": {}, "compressed": 0, "errors": []}
        with self._maintenance_locked() as acquired:
            if not acquired:
                return result
            result["acquired"] = True
            # Sealed segments are immutable.  Compression therefore happens
            # outside the append lock and cannot block running workers.
            with self._locked():
                compressible = [path for path in self._sealed_paths() if path.suffix == ".jsonl"]
            if not dry_run:
                for path in compressible:
                    compressed, error = self._compress_sealed(path)
                    if compressed:
                        result["compressed"] += 1
                    elif error:
                        result["errors"].append(error)
                        print(error, file=sys.stderr)
            victims, errors = self._prune_candidates()
            result["errors"].extend(errors)
            for error in errors:
                print(error, file=sys.stderr)
            for path, bucket in victims:
                result["deleted"][bucket] = result["deleted"].get(bucket, 0) + 1
            if not dry_run:
                with self._locked():
                    for path, _bucket in victims:
                        path.unlink(missing_ok=True)
        return result

    def _prune_candidates(self) -> tuple[list[tuple[Path, str]], list[str]]:
        """Inspect immutable segments without delaying appends."""
        segments = self._sealed_paths()
        entries: list[tuple[Path, str, bool, int]] = []
        errors: list[str] = []
        for path in segments:
            first: dict[str, Any] | None = None
            expired = True
            read_errors: list[str] = []
            for row in self._read_file(path, errors=read_errors):
                first = row if first is None else first
                expired = expired and self._expired(row)
            if read_errors:
                errors.extend(read_errors)
                expired = False
            bucket = self._retention_bucket(first or {})
            entries.append((path, bucket, not read_errors and (first is None or expired), path.stat().st_size))
        victims = [(path, bucket) for path, bucket, expired, _size in entries if expired]
        retained = [(path, bucket, size) for path, bucket, expired, size in entries if not expired]
        backup_count = max(1, int(getattr(self.config, "backup_count", 10)))
        for path, bucket, _size in retained[:-backup_count]:
            victims.append((path, bucket))
        maximum = int(getattr(self.config, "max_total_bytes", 0) or 0)
        if maximum > 0:
            active_bytes = self._active_size_snapshot()
            total = active_bytes + sum(size for _path, _bucket, size in retained)
            already = {path for path, _bucket in victims}
            for path, bucket, size in retained:
                if total <= maximum:
                    break
                if path not in already:
                    victims.append((path, bucket))
                    already.add(path)
                total -= size
        return victims, errors

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _maintenance_locked(self) -> Iterator[bool]:
        """Attempt one maintenance pass without queueing behind another one."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self._maintenance_lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _active_path(self, stream: str, bucket: str) -> Path:
        return self.root / f"{self.prefix}.{stream}.{bucket}.jsonl"

    def _active_paths(self) -> list[Path]:
        return [self._active_path(stream, bucket) for stream in _STREAMS for bucket in _BUCKETS]

    def _sealed_paths(self) -> list[Path]:
        return self._canonical_paths(_STREAMS, include_active=False)

    def _canonical_paths(self, streams: tuple[str, ...] | frozenset[str], *, include_active: bool = True) -> list[Path]:
        """Return one source/archive path per sealed segment, never a temp."""
        selected: dict[str, Path] = {}
        try:
            candidates = self.root.iterdir()
        except OSError:
            candidates = ()
        for path in candidates:
            match = self._sealed_pattern.match(path.name)
            if match is None or match.group(1) not in streams:
                continue
            base = path.name.removesuffix(".zst")
            current = selected.get(base)
            if current is None or path.suffix == ".zst":
                selected[base] = path
        paths = list(selected.values())
        if include_active:
            paths.extend(
                path for path in self._active_paths()
                if path.exists() and any(path.name.startswith(f"{self.prefix}.{stream}.") for stream in streams)
            )
        return sorted(paths, key=self._segment_sort_key)

    @staticmethod
    def _segment_sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"\.(\d+)(?:-\d+)?\.jsonl(?:\.zst)?$", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    def _path_range(self, path: Path) -> tuple[int | None, int | None]:
        match = self._sealed_pattern.match(path.name)
        if match is None:
            return None, None
        return int(match.group(3)), None if match.group(4) is None else int(match.group(4))

    def _outside_cursor_range(self, path: Path, *, after_id: int | None, before_id: int | None) -> bool:
        first, last = self._path_range(path)
        if first is None:
            return False
        if before_id is not None and first >= before_id:
            return True
        return after_id is not None and last is not None and last <= after_id

    def _rotate_locked(self, stream: str, bucket: str) -> None:
        active = self._active_path(stream, bucket)
        if not active.exists() or not active.stat().st_size:
            return
        # The first ID in the segment makes names sortable without relying on
        # wall-clock resolution shared by several worker processes.
        first = next(self._read_file(active), {}).get("id", int(time.time() * 1_000_000))
        last = self._tail_record_id(active)
        sealed = self.root / f"{self.prefix}.{stream}.{bucket}.{int(first):016d}-{int(last):016d}.jsonl"
        os.replace(active, sealed)

    def _next_sequence(self) -> int:
        try:
            value = int(self._sequence_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            # Counter recovery is the only append-time archive scan.  It runs
            # under the caller's append lock and is needed only after damage.
            value = max(
                (int(row.get("id", 0)) for path in self._canonical_paths(_STREAMS) for row in self._read_file(path)),
                default=0,
            )
        # A power loss can occur after the record fsync and before the sidecar
        # replacement.  Active tails are cheap to inspect and prevent reuse.
        value = max(value, *(self._tail_record_id(path) for path in self._active_paths()))
        # Timestamp-based initialization avoids reusing legacy PostgreSQL
        # bigserial cursors after cutover; subsequent records remain a sequence.
        value = max(value + 1, time.time_ns() // 1_000)
        if value > _MAX_EVENT_ID:
            raise RuntimeError("Output event sequence exceeds JavaScript-safe integer range.")
        return value

    def _current_highwater(self) -> int:
        """Read the durable counter plus active tails without scanning archives."""
        try:
            counter = int(self._sequence_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            counter = 0
        tails = (self._tail_record_id(path) for path in self._active_paths())
        return max(counter, self._highwater or 0, *tails)

    def _repair_active_locked(self, path: Path) -> None:
        """Remove only an unterminated crash tail before the next append."""
        if not path.exists() or not path.stat().st_size:
            return
        with path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return
            end = handle.tell()
            start = max(0, end - _MAX_LINE_BYTES)
            handle.seek(start)
            chunk = handle.read(end - start)
            boundary = chunk.rfind(b"\n")
            handle.truncate(start + boundary + 1 if boundary >= 0 else 0)
            handle.flush()
            os.fsync(handle.fileno())

    def _tail_record_id(self, path: Path) -> int:
        try:
            with path.open("rb") as handle:
                end = handle.seek(0, os.SEEK_END)
                chunk_size = 4096
                buffered = b""
                position = end
                while position > 0 and len(buffered) <= _MAX_LINE_BYTES:
                    position = max(0, position - chunk_size)
                    handle.seek(position)
                    buffered = handle.read(min(chunk_size, end - position)) + buffered
                    lines = buffered.split(b"\n")
                    # The final entry is empty for a complete file and a
                    # crash tail otherwise; neither is a complete record.
                    for line in reversed(lines[:-1]):
                        try:
                            return int(json.loads(line).get("id", 0))
                        except (TypeError, ValueError, json.JSONDecodeError):
                            continue
                    chunk_size = min(_MAX_LINE_BYTES, chunk_size * 2)
        except OSError:
            pass
        return 0

    def _active_size_snapshot(self) -> int:
        """Read active sizes while rotation cannot rename a path beneath us."""
        with self._locked():
            total = 0
            for path in self._active_paths():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    continue
            return total

    def _active_started_at(self, path: Path) -> float:
        first = next(self._read_file(path), None)
        if first is not None:
            try:
                return datetime.fromisoformat(str(first["created_at"]).replace("Z", "+00:00")).timestamp()
            except (KeyError, ValueError):
                pass
        return path.stat().st_mtime

    def _write_sequence(self, value: int) -> None:
        temporary = self._sequence_path.with_suffix(".tmp")
        with temporary.open("w", encoding="ascii") as handle:
            handle.write(f"{value}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._sequence_path)

    def _checkpoint_offset(self, key: tuple[int, int, int], *, after_id: int | None, byte_limit: int) -> int:
        if after_id is None:
            return 0
        with self._checkpoint_lock:
            points = self._active_checkpoints.get(key)
            if not points:
                return 0
            self._active_checkpoints.move_to_end(key)
            return max((offset for identifier, offset in points if identifier <= after_id and offset <= byte_limit), default=0)

    def _remember_checkpoint(self, key: tuple[int, int, int], identifier: int, offset: int) -> None:
        with self._checkpoint_lock:
            points = self._active_checkpoints.get(key)
            if points is None:
                points = deque(maxlen=256)
                self._active_checkpoints[key] = points
            self._active_checkpoints.move_to_end(key)
            points.append((identifier, offset))
            while len(self._active_checkpoints) > 64:
                self._active_checkpoints.popitem(last=False)

    @contextmanager
    def _snapshot(
        self, streams: tuple[str, ...] | frozenset[str], *, after_id: int | None = None, before_id: int | None = None,
    ) -> Iterator[list[tuple[Path, Any, int, tuple[int, int, int] | None, int]]]:
        """Open a canonical inventory before concurrent rotation can remove it."""
        handles: list[tuple[Path, Any, int, tuple[int, int, int] | None, int]] = []
        with self._locked():
            for path in self._canonical_paths(streams):
                if self._outside_cursor_range(path, after_id=after_id, before_id=before_id):
                    continue
                try:
                    handle = path.open("rb")
                    stat = os.fstat(handle.fileno())
                    key = None
                    if path.suffix != ".zst":
                        first_id, _last_id = self._path_range(path)
                        if first_id is None:
                            first = next(self._read_handle(path, handle, byte_limit=stat.st_size), {})
                            first_id = int(first.get("id", 0))
                            handle.seek(0)
                        # Filesystems can reuse an expired segment's inode for
                        # a new active file. Its first ID distinguishes those
                        # generations so an old checkpoint cannot skip output.
                        key = (stat.st_dev, stat.st_ino, first_id)
                    start = 0 if key is None else self._checkpoint_offset(key, after_id=after_id, byte_limit=stat.st_size)
                    handles.append((path, handle, stat.st_size, key, start))
                except OSError:
                    continue
        try:
            yield handles
        finally:
            for _path, handle, _size, _key, _start in handles:
                handle.close()

    def _records(
        self, streams: tuple[str, ...] | frozenset[str], *, after_id: int | None = None, before_id: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        with self._snapshot(streams, after_id=after_id, before_id=before_id) as snapshot:
            for path, handle, size, checkpoint_key, start in snapshot:
                # Freeze active EOFs at the same point as the inventory. Seeing
                # a later append in one category but missing an earlier append
                # to an already-read category would advance SSE past that event.
                if start:
                    handle.seek(start)
                yield from self._read_handle(
                    path, handle, byte_limit=None if path.suffix == ".zst" else size,
                    checkpoint_key=checkpoint_key,
                )

    def _read_file(self, path: Path, *, errors: list[str] | None = None) -> Iterator[dict[str, Any]]:
        try:
            with path.open("rb") as handle:
                yield from self._read_handle(path, handle, errors=errors)
        except (OSError, zstandard.ZstdError):
            if errors is not None:
                errors.append(f"Unable to read output segment {path}; retaining it.")
            return

    def _read_handle(
        self,
        path: Path,
        input_handle: Any,
        *,
        errors: list[str] | None = None,
        byte_limit: int | None = None,
        checkpoint_key: tuple[int, int, int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        reader: Any = input_handle
        if path.suffix == ".zst":
            reader = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(input_handle))

        def read_line() -> bytes:
            budget = _MAX_LINE_BYTES + 1
            if byte_limit is not None:
                budget = min(budget, max(0, byte_limit - reader.tell()))
            return reader.readline(budget) if budget else b""

        try:
            while line := read_line():
                if len(line) > _MAX_LINE_BYTES:
                    while line and not line.endswith(b"\n"):
                        line = read_line()
                    continue
                if not line.endswith(b"\n"):
                    continue
                try:
                    item = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if errors is not None:
                        errors.append(f"Invalid JSONL record in output segment {path}; retaining it.")
                    continue
                if isinstance(item, dict):
                    if checkpoint_key is not None:
                        try:
                            self._remember_checkpoint(checkpoint_key, int(item.get("id", 0)), reader.tell())
                        except (OSError, ValueError):
                            pass
                    yield item
        except (OSError, zstandard.ZstdError):
            if errors is not None:
                errors.append(f"Unable to read output segment {path}; retaining it.")
            return
        finally:
            if reader is not input_handle:
                reader.close()

    def _compress_sealed(self, source: Path) -> tuple[bool, str | None]:
        """Compress an immutable segment outside the global append lock."""
        destination = source.with_suffix(source.suffix + ".zst")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
                zstandard.ZstdCompressor(
                    level=int(getattr(self.config, "compression_level", 3)), write_checksum=True,
                ).copy_stream(input_handle, output_handle)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            with self._locked():
                # A prior crash may have written the archive before removing
                # its source; preserve that destination rather than overwrite.
                if destination.exists():
                    source.unlink(missing_ok=True)
                elif source.exists():
                    os.replace(temporary, destination)
                    source.unlink()
                else:
                    return False, f"Output segment {source} disappeared during compression."
            return True, None
        except (OSError, zstandard.ZstdError) as exc:
            return False, f"Unable to compress output segment {source}: {exc}"
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _matches(row: dict[str, Any], fields: dict[str, Any], *, after_id: int | None = None, before_id: int | None = None) -> bool:
        identifier = int(row.get("id", 0))
        if after_id is not None and identifier <= after_id:
            return False
        if before_id is not None and identifier >= before_id:
            return False
        return all(value is None or str(row.get(name)) == str(value) for name, value in fields.items())

    @staticmethod
    def _page(records: Iterator[dict[str, Any]], *, limit: int, offset: int, ascending: bool) -> list[dict[str, Any]]:
        start = max(0, int(offset))
        count = start + max(1, int(limit))
        key = lambda row: int(row.get("id", 0))
        selected = heapq.nsmallest(count, records, key=key) if ascending else heapq.nlargest(count, records, key=key)
        return selected[start:]

    def _expired(self, row: dict[str, Any]) -> bool:
        try:
            created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            return False
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days = getattr(self.config, f"{self._retention_bucket(row)}_retention_days")
        return created < datetime.now(timezone.utc) - timedelta(days=int(days))

    @staticmethod
    def _retention_bucket(row: dict[str, Any]) -> str:
        event_type = str(row.get("event_type") or "")
        if str(row.get("level") or "").lower() in {"warning", "error", "critical"} or event_type == "jobs.deleted":
            return "important_log"
        if event_type in {"job.progress_updated", "job.heartbeat", "worker.heartbeat", "worker.touched"}:
            return "high_frequency"
        if row.get("stream") == "events":
            return "job_event"
        return "routine_log"
