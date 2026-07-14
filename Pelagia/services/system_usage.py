"""Host and storage metrics for the administrative system-usage API."""

from __future__ import annotations

import os
import platform
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .context import AppContext

try:
    import psutil
except ImportError:  # pragma: no cover - supported fallback for minimal installs
    psutil = None  # type: ignore[assignment]


class SystemUsageService:
    """Collect lightweight host, filesystem, and PostgreSQL storage metrics."""

    def __init__(self, context: AppContext):
        self.context = context

    def snapshot(self) -> dict[str, Any]:
        storage = {
            "kvstore_directory": _filesystem_usage(self.context.config.kvstore.directory),
            "raw_assets_default": _filesystem_usage(
                self.context.config.file_browser.root_path_import_dir
            ),
            "database": self._database_usage(),
        }
        cpu = _cpu_usage()
        memory = _memory_usage()
        return {
            "reported_at": datetime.now(timezone.utc).isoformat(),
            "host": _host_usage(),
            "cpu": cpu,
            "memory": memory,
            "process": _process_usage(),
            "storage": storage,
            "stress": _stress_summary(cpu=cpu, memory=memory, storage=storage),
        }

    def _database_usage(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "connection": _database_connection_summary(self.context.config.database.dsn),
            "storage": {
                "available": False,
                "reason": "PostgreSQL storage metrics are unavailable.",
            },
        }
        repository = self.context.repository
        if repository is None:
            result["storage"]["reason"] = "PostgreSQL is not configured."
            return result
        try:
            with repository.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT current_setting('data_directory', true) AS data_directory,
                               current_database() AS database_name,
                               pg_database_size(current_database())::bigint AS database_bytes
                        """
                    )
                    row = cursor.fetchone()
        except Exception:
            return result

        data_directory = _row_value(row, "data_directory")
        database_name = _row_value(row, "database_name")
        database_bytes = _row_value(row, "database_bytes")
        database_path = None if data_directory is None else Path(str(data_directory))
        result["storage"] = {
            "available": True,
            "database_name": None if database_name is None else str(database_name),
            "database_bytes": None if database_bytes is None else int(database_bytes),
            "data_directory": None if data_directory is None else str(data_directory),
            "filesystem": (
                _filesystem_usage(database_path)
                if database_path is not None and database_path.exists()
                else {
                    "available": False,
                    "reason": "PostgreSQL data directory is not visible to the API process.",
                }
            ),
        }
        return result


def _filesystem_usage(path: Path) -> dict[str, Any]:
    configured_path = Path(path).expanduser()
    resolved_path = configured_path.resolve(strict=False)
    probe_path = _existing_parent(resolved_path)
    result: dict[str, Any] = {
        "configured_path": str(configured_path),
        "resolved_path": str(resolved_path),
        "probe_path": str(probe_path),
        "path_exists": resolved_path.exists(),
    }
    try:
        usage = shutil.disk_usage(probe_path)
    except OSError:
        return {
            **result,
            "available": False,
            "reason": "Filesystem usage could not be read.",
        }
    total_bytes = int(usage.total)
    used_bytes = int(usage.used)
    free_bytes = int(usage.free)
    return {
        **result,
        "available": True,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "free_bytes": free_bytes,
        "used_percent": _percent(used_bytes, total_bytes),
        "free_percent": _percent(free_bytes, total_bytes),
    }


def _existing_parent(path: Path) -> Path:
    candidate = path
    while candidate != candidate.parent and not candidate.exists():
        candidate = candidate.parent
    return candidate


def _host_usage() -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if psutil is not None:
        result["booted_at"] = datetime.fromtimestamp(psutil.boot_time(), timezone.utc).isoformat()
    return result


def _cpu_usage() -> dict[str, Any]:
    logical_cpus = os.cpu_count()
    load_average = None
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        load_average = {
            "one_minute": load_1m,
            "five_minutes": load_5m,
            "fifteen_minutes": load_15m,
        }
    except (AttributeError, OSError):
        pass
    result: dict[str, Any] = {
        "logical_cpus": logical_cpus,
        "load_average": load_average,
        "utilization_percent": None,
    }
    if psutil is not None:
        result["physical_cpus"] = psutil.cpu_count(logical=False)
        result["utilization_percent"] = psutil.cpu_percent(interval=0.05)
    return result


def _memory_usage() -> dict[str, Any]:
    if psutil is not None:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "source": "psutil",
            "total_bytes": int(memory.total),
            "used_bytes": int(memory.used),
            "available_bytes": int(memory.available),
            "used_percent": float(memory.percent),
            "available_percent": _percent(int(memory.available), int(memory.total)),
            "swap_total_bytes": int(swap.total),
            "swap_used_bytes": int(swap.used),
            "swap_used_percent": float(swap.percent),
        }

    page_size = _sysconf_value("SC_PAGE_SIZE")
    total_pages = _sysconf_value("SC_PHYS_PAGES")
    available_pages = _sysconf_value("SC_AVPHYS_PAGES")
    total_bytes = page_size * total_pages if page_size and total_pages else None
    available_bytes = page_size * available_pages if page_size and available_pages else None
    used_bytes = total_bytes - available_bytes if total_bytes is not None and available_bytes is not None else None
    return {
        "source": "sysconf",
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "available_bytes": available_bytes,
        "used_percent": _percent(used_bytes, total_bytes),
        "available_percent": _percent(available_bytes, total_bytes),
        "swap_total_bytes": None,
        "swap_used_bytes": None,
        "swap_used_percent": None,
    }


def _process_usage() -> dict[str, Any]:
    result: dict[str, Any] = {"pid": os.getpid()}
    if psutil is None:
        return result
    process = psutil.Process()
    memory = process.memory_info()
    result.update(
        {
            "rss_bytes": int(memory.rss),
            "vms_bytes": int(memory.vms),
            "thread_count": process.num_threads(),
            "cpu_percent": process.cpu_percent(interval=None),
        }
    )
    if hasattr(process, "num_fds"):
        try:
            result["open_file_descriptors"] = process.num_fds()
        except OSError:
            pass
    return result


def _database_connection_summary(dsn: str) -> dict[str, Any]:
    parsed = urlparse(str(dsn))
    if parsed.scheme.startswith("postgres"):
        return {
            "host": parsed.hostname,
            "port": parsed.port,
            "database_name": parsed.path.lstrip("/") or None,
        }
    return {"host": None, "port": None, "database_name": None}


def _stress_summary(
    *,
    cpu: dict[str, Any],
    memory: dict[str, Any],
    storage: dict[str, Any],
) -> dict[str, Any]:
    alerts: list[dict[str, str]] = []
    for name, entry in storage.items():
        filesystem = entry.get("filesystem") if name == "database" else entry
        if not isinstance(filesystem, dict) or not filesystem.get("available"):
            continue
        free_percent = filesystem.get("free_percent")
        if isinstance(free_percent, (int, float)):
            _add_threshold_alert(alerts, free_percent, 5, 15, f"storage.{name}.free_percent")

    available_percent = memory.get("available_percent")
    if isinstance(available_percent, (int, float)):
        _add_threshold_alert(alerts, available_percent, 5, 10, "memory.available_percent")

    utilization_percent = cpu.get("utilization_percent")
    if isinstance(utilization_percent, (int, float)):
        _add_threshold_alert(alerts, 100 - utilization_percent, 5, 15, "cpu.utilization_percent")

    load_average = cpu.get("load_average")
    logical_cpus = cpu.get("logical_cpus")
    if isinstance(load_average, dict) and isinstance(logical_cpus, int) and logical_cpus > 0:
        one_minute = load_average.get("one_minute")
        if isinstance(one_minute, (int, float)):
            load_percent = one_minute * 100 / logical_cpus
            if load_percent >= 100:
                alerts.append(
                    {
                        "level": "critical",
                        "metric": "cpu.load_average.one_minute",
                        "message": "CPU load exceeds logical CPU capacity.",
                    }
                )
            elif load_percent >= 85:
                alerts.append(
                    {
                        "level": "warning",
                        "metric": "cpu.load_average.one_minute",
                        "message": "CPU load is high relative to logical CPU capacity.",
                    }
                )

    status = (
        "critical"
        if any(alert["level"] == "critical" for alert in alerts)
        else "warning"
        if alerts
        else "ok"
    )
    return {"status": status, "alerts": alerts}


def _add_threshold_alert(
    alerts: list[dict[str, str]],
    value: float,
    critical_threshold: float,
    warning_threshold: float,
    metric: str,
) -> None:
    if value <= critical_threshold:
        alerts.append(
            {"level": "critical", "metric": metric, "message": f"{metric} is critically low."}
        )
    elif value <= warning_threshold:
        alerts.append({"level": "warning", "metric": metric, "message": f"{metric} is low."})


def _sysconf_value(name: str) -> int | None:
    try:
        return int(os.sysconf(name))
    except (AttributeError, OSError, ValueError):
        return None


def _row_value(row: Any, key: str) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return getattr(row, key, None)


def _percent(value: int | float | None, total: int | float | None) -> float | None:
    if value is None or total in {None, 0}:
        return None
    return round(float(value) * 100 / float(total), 2)
