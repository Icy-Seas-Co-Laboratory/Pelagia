from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

from Pelagia.config import LoggingConfig
from Pelagia.storage.output import OutputStore


def _store(tmp_path: Path, *, max_bytes: int = 250) -> OutputStore:
    return OutputStore(LoggingConfig(log_path=tmp_path, file_name="pelagia.log", max_bytes=max_bytes, backup_count=10))


def _append_in_process(root: str, count: int) -> None:
    store = _store(Path(root), max_bytes=10_000)
    for index in range(count):
        store.append({"project_id": "p1", "event_type": "concurrent", "producer_index": index})


def test_output_store_assigns_global_ids_and_filters_event_history(tmp_path):
    store = _store(tmp_path)
    log = store.append({"project_id": "p1", "event_type": "pipeline.started", "level": "info", "logger": "test"})
    progress = store.append({"project_id": "p1", "job_id": "j1", "event_type": "job.progress_updated", "level": "debug"}, stream="events")
    completed = store.append({"project_id": "p1", "job_id": "j1", "event_type": "job.completed", "level": "info"}, stream="events")

    assert log["id"] < progress["id"] < completed["id"]
    assert [row["event_type"] for row in store.list_logs(project_id="p1")] == ["job.completed", "pipeline.started"]
    assert [row["event_type"] for row in store.list_job_events(project_id="p1", after_id=1, ascending=True)] == ["job.progress_updated", "job.completed"]


def test_output_store_rotates_compresses_and_skips_partial_tail(tmp_path):
    store = _store(tmp_path, max_bytes=120)
    for index in range(4):
        store.append({"project_id": "p1", "event_type": "message", "message": "x" * 40, "index": index})
    active = tmp_path / "pelagia.logs.routine_log.jsonl"
    with active.open("ab") as handle:
        handle.write(b'{"id":999')

    result = store.prune()
    assert result["compressed"] >= 1
    assert list(tmp_path.glob("*.zst"))
    assert [row["index"] for row in store.list_logs(project_id="p1")] == [3, 2, 1, 0]


def test_output_store_ignores_temporary_archives_and_snapshot_survives_removal(tmp_path):
    store = _store(tmp_path, max_bytes=120)
    store.append({"project_id": "p1", "event_type": "first", "message": "x" * 50})
    store.append({"project_id": "p1", "event_type": "second", "message": "x" * 50})
    store.prune()
    archive = next(tmp_path.glob("*.zst"))
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.write_bytes(archive.read_bytes())
    assert [row["event_type"] for row in store.list_logs(project_id="p1")] == ["second", "first"]

    with store._snapshot(("logs",)) as snapshot:
        path, handle, _size, _key, _start = next(item for item in snapshot if item[0].suffix == ".zst")
        path.unlink()
        assert [row["event_type"] for row in store._read_handle(path, handle)] == ["first"]


def test_event_replay_freezes_all_active_file_lengths_before_reading(tmp_path, monkeypatch):
    store = _store(tmp_path, max_bytes=100000)
    first = store.append({"project_id": "p1", "event_type": "job.progress_updated"}, stream="events")
    second = store.append({"project_id": "p1", "event_type": "job.created"}, stream="events")
    read_handle = store._read_handle
    injected = []

    def append_between_categories(path, handle, **kwargs):
        yield from read_handle(path, handle, **kwargs)
        if path.name.endswith("events.high_frequency.jsonl") and not injected:
            injected.append(True)
            store.append({"project_id": "p1", "event_type": "job.progress_updated"}, stream="events")
            store.append({"project_id": "p1", "event_type": "job.completed"}, stream="events")

    monkeypatch.setattr(store, "_read_handle", append_between_categories)
    rows = store.list_job_events(project_id="p1", ascending=True)
    assert [row["id"] for row in rows] == [first["id"], second["id"]]
    replay = store.list_job_events(project_id="p1", after_id=second["id"], ascending=True)
    assert [row["event_type"] for row in replay] == ["job.progress_updated", "job.completed"]


def test_event_replay_skips_archives_ending_at_the_cursor(tmp_path, monkeypatch):
    store = _store(tmp_path, max_bytes=180)
    first = store.append({"project_id": "p1", "event_type": "job.created", "message": "x" * 60}, stream="events")
    old = store.append({"project_id": "p1", "event_type": "job.leased", "message": "x" * 60}, stream="events")
    third = store.append({"project_id": "p1", "event_type": "job.started", "message": "x" * 60}, stream="events")
    store.prune()
    fresh = store.append({"project_id": "p1", "event_type": "job.completed", "message": "x" * 60}, stream="events")
    calls: list[Path] = []
    original = store._read_handle

    def count_archives(path, handle, **kwargs):
        if path.suffix == ".zst":
            calls.append(path)
        yield from original(path, handle, **kwargs)

    monkeypatch.setattr(store, "_read_handle", count_archives)
    rows = store.list_job_events(project_id="p1", after_id=third["id"], ascending=True)
    assert [row["id"] for row in rows] == [fresh["id"]]
    assert not calls
    assert first["id"] < old["id"] < third["id"] < fresh["id"]


def test_recent_active_cursor_reuses_bounded_read_checkpoint(tmp_path):
    store = _store(tmp_path, max_bytes=100000)
    first = store.append({"project_id": "p1", "event_type": "job.created"}, stream="events")
    second = store.append({"project_id": "p1", "event_type": "job.leased"}, stream="events")
    store.append({"project_id": "p1", "event_type": "job.completed"}, stream="events")
    store.list_job_events(project_id="p1", after_id=first["id"], ascending=True)
    with store._snapshot(("events",), after_id=second["id"]) as snapshot:
        active = next(item for item in snapshot if item[0].suffix == ".jsonl")
        assert active[-1] > 0


def test_output_store_retains_and_reports_corrupt_archive(tmp_path):
    store = _store(tmp_path, max_bytes=120)
    store.append({"project_id": "p1", "event_type": "first", "message": "x" * 50})
    store.append({"project_id": "p1", "event_type": "second", "message": "x" * 50})
    store.prune()
    archive = next(tmp_path.glob("*.zst"))
    archive.write_bytes(b"not a zstd frame")
    result = store.prune()
    assert result["errors"]
    assert archive.exists()


def test_output_store_recovers_sequence_after_missing_counter(tmp_path):
    store = _store(tmp_path)
    store.append({"project_id": "p1", "event_type": "first"})
    (tmp_path / ".pelagia.output.sequence").unlink()
    assert store.append({"project_id": "p1", "event_type": "second"})["id"] > 1
    (tmp_path / ".pelagia.output.sequence").write_text("not-a-number", encoding="ascii")
    assert store.append({"project_id": "p1", "event_type": "third"})["id"] > 1


def test_output_store_repairs_crash_tail_before_next_append(tmp_path):
    store = _store(tmp_path)
    first = store.append({"project_id": "p1", "event_type": "first"})
    active = tmp_path / "pelagia.logs.routine_log.jsonl"
    with active.open("ab") as handle:
        handle.write(b'{"id":999')
    second = store.append({"project_id": "p1", "event_type": "second"})
    rows = store.list_logs(project_id="p1")
    assert [row["event_type"] for row in rows] == ["second", "first"]
    assert second["id"] > first["id"]


def test_output_store_serializes_independent_process_writers(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_append_in_process, args=(str(tmp_path), 4)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    rows = _store(tmp_path).list_logs(project_id="p1", limit=20)
    assert len(rows) == 8
    assert len({row["id"] for row in rows}) == 8


def test_output_store_prunes_expired_sealed_segment(tmp_path):
    store = _store(tmp_path, max_bytes=100)
    store.config.routine_log_retention_days = 1
    store.append({"project_id": "p1", "event_type": "old", "created_at": "2000-01-01T00:00:00+00:00", "message": "x" * 50})
    store.append({"project_id": "p1", "event_type": "new", "message": "x" * 50})
    assert store.prune()["deleted"]["routine_log"] == 1


def test_warning_progress_uses_important_retention(tmp_path):
    store = _store(tmp_path)
    store.append({"event_type": "job.progress_updated", "level": "warning"}, stream="events")
    assert (tmp_path / "pelagia.events.important_log.jsonl").exists()
