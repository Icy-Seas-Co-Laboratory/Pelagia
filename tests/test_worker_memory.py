from pathlib import Path
from types import SimpleNamespace

import pytest

from Pelagia.workers.memory import available_memory


def test_host_available_memory_is_used_without_a_container_limit(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "Pelagia.workers.memory.psutil.virtual_memory",
        lambda: SimpleNamespace(total=1_000, available=190),
    )

    memory = available_memory(cgroup_root=tmp_path)

    assert memory.source == "host"
    assert memory.fraction == pytest.approx(0.19)


def test_container_limit_uses_reclaimable_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "Pelagia.workers.memory.psutil.virtual_memory",
        lambda: SimpleNamespace(total=1_000, available=800),
    )
    (tmp_path / "memory.max").write_text("500\n")
    (tmp_path / "memory.current").write_text("450\n")
    (tmp_path / "memory.stat").write_text("inactive_file 25\nactive_file 10\n")

    memory = available_memory(cgroup_root=tmp_path)

    assert memory.source == "cgroup-v2"
    assert memory.available_bytes == 75
    assert memory.fraction == pytest.approx(0.15)


def test_unlimited_container_uses_host_memory(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "Pelagia.workers.memory.psutil.virtual_memory",
        lambda: SimpleNamespace(total=1_000, available=250),
    )
    (tmp_path / "memory.max").write_text("max\n")

    assert available_memory(cgroup_root=tmp_path).source == "host"


def test_legacy_container_limit_is_checked(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "Pelagia.workers.memory.psutil.virtual_memory",
        lambda: SimpleNamespace(total=1_000, available=900),
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "memory.limit_in_bytes").write_text("500\n")
    (memory_dir / "memory.usage_in_bytes").write_text("450\n")
    (memory_dir / "memory.stat").write_text("total_inactive_file 20\n")

    assert available_memory(cgroup_root=tmp_path).fraction == pytest.approx(0.14)
