from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .exceptions import UnsafePathError

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?I?B)?\s*$", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_size(value: int | str) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("size must be positive")
        return value
    match = _SIZE_RE.match(value)
    if not match:
        raise ValueError(f"invalid byte size: {value!r}")
    number, suffix = match.groups()
    suffix = (suffix or "B").upper()
    decimal = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
    binary = {"KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
    multiplier = (decimal | binary).get(suffix)
    if multiplier is None:
        raise ValueError(f"invalid byte-size suffix: {suffix}")
    result = int(float(number) * multiplier)
    if result <= 0:
        raise ValueError("size must be positive")
    return result


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _new_hasher(algorithm: str) -> Any:
    normalized = algorithm.lower()
    if normalized == "blake3":
        try:
            return importlib.import_module("blake3").blake3()
        except ImportError as exc:
            raise ValueError("hash algorithm 'blake3' requires the optional blake3 package") from exc
    if normalized == "xxh3":
        try:
            return importlib.import_module("xxhash").xxh3_128()
        except ImportError as exc:
            raise ValueError("hash algorithm 'xxh3' requires the optional xxhash package") from exc
    try:
        return hashlib.new(normalized)
    except ValueError as exc:
        raise ValueError(f"hash algorithm {algorithm!r} is unavailable") from exc


def hash_stream(stream: BinaryIO, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    digest = _new_hasher(algorithm)
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hash_stream(stream, algorithm)


def hash_bytes(data: bytes, algorithm: str = "sha256") -> str:
    digest = _new_hasher(algorithm)
    digest.update(data)
    return str(digest.hexdigest())


def available_hash(name: str) -> bool:
    normalized = name.lower()
    if normalized in hashlib.algorithms_available:
        return True
    module = {"blake3": "blake3", "xxh3": "xxhash"}.get(normalized)
    if module is None:
        return False
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def safe_relative_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafePathError(f"unsafe relative path: {value!s}")
    return path


def confined_path(root: Path, relative: str | Path) -> Path:
    relative_path = safe_relative_path(relative)
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise UnsafePathError(f"path escapes root: {relative!s}")
    return candidate


def fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def iter_chunks(iterable: Iterator[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
