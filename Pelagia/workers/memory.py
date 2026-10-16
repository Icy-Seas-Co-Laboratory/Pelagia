"""Available-memory check for admitting new worker jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True, slots=True)
class MemoryAvailability:
    source: str
    available_bytes: int
    total_bytes: int

    @property
    def fraction(self) -> float:
        return self.available_bytes / self.total_bytes


def _inactive_file(path: Path, key: str) -> int:
    try:
        for line in path.read_text().splitlines():
            name, value = line.split(maxsplit=1)
            if name == key:
                return int(value)
    except FileNotFoundError:
        return 0
    return 0


def _cgroup_availability(root: Path, *, host_total: int) -> MemoryAvailability | None:
    # Docker mounts the process's cgroup at this root. An unlimited cgroup
    # reports "max" (v2) or an enormous sentinel limit (v1).
    for directory, limit_name, usage_name, stat_key, source in (
        (root, "memory.max", "memory.current", "inactive_file", "cgroup-v2"),
        (root / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file", "cgroup-v1"),
    ):
        limit_path = directory / limit_name
        if not limit_path.exists():
            continue
        raw_limit = limit_path.read_text().strip()
        if raw_limit == "max":
            return None
        limit = int(raw_limit)
        if limit <= 0 or limit >= host_total:
            return None
        usage = int((directory / usage_name).read_text().strip())
        reclaimable_cache = _inactive_file(directory / "memory.stat", stat_key)
        available = max(0, min(limit, limit - usage + reclaimable_cache))
        return MemoryAvailability(source, available, limit)
    return None


def available_memory(*, cgroup_root: Path = Path("/sys/fs/cgroup")) -> MemoryAvailability:
    """Return the tighter host or container available-memory measurement.

    Host MemAvailable includes reclaimable page cache. Cgroup usage includes
    cache too, so add back its inactive file pages to avoid blocking work merely
    because the kernel is caching disk reads.
    """
    host = psutil.virtual_memory()
    if host.total <= 0:
        raise ValueError("Host memory total is unavailable")
    host_available = MemoryAvailability("host", int(host.available), int(host.total))
    cgroup = _cgroup_availability(cgroup_root, host_total=host_available.total_bytes)
    return min((host_available, cgroup), key=lambda item: item.fraction) if cgroup else host_available
