from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pelagia_interchange.cli import main, parser
from pelagia_interchange.tui import browse_entries, run_creator, validate_creation


def options(tmp_path):
    source = tmp_path / "videos"
    source.mkdir()
    (source / "capture.mp4").write_bytes(b"video")
    return parser().parse_args(["create", "--from-videos", str(source)])


def test_validation_defaults_output_without_writing(tmp_path):
    args = options(tmp_path)
    assert validate_creation(args) == [args.from_videos / "capture.mp4"]
    assert args.output == tmp_path / "videos_interchange"
    assert not args.output.exists()


@pytest.mark.parametrize(("field", "value", "error"), [
    ("jpeg_quality", 0, "quality"),
    ("jpeg_quality", None, "quality"),
    ("jpeg_subsampling", "bad", "subsampling"),
    ("jpeg_workers", 0, "workers"),
    ("queue_depth", -1, "depth"),
    ("preview_width", 1, "width"),
    ("preview_count", None, "count"),
    ("stream", " ", "stream"),
    ("shard_target_size", "nonsense", "size"),
])
def test_validation_rejects_invalid_settings(tmp_path, field, value, error):
    args = options(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=error):
        validate_creation(args)
    assert not (tmp_path / "videos_interchange").exists()


@pytest.mark.parametrize("destination", ["same", "child", "parent", "populated"])
def test_validation_protects_existing_data(tmp_path, destination):
    args = options(tmp_path)
    args.output = {"same": args.from_videos, "child": args.from_videos / "archive",
                   "parent": tmp_path, "populated": tmp_path / "existing"}[destination]
    if destination == "populated":
        args.output.mkdir()
        (args.output / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="overlap|new or empty"):
        validate_creation(args)
    assert (args.from_videos / "capture.mp4").read_bytes() == b"video"


def test_validation_suggests_recursive_discovery(tmp_path):
    args = options(tmp_path)
    nested = args.from_videos / "nested"
    nested.mkdir()
    (args.from_videos / "capture.mp4").rename(nested / "capture.mp4")
    with pytest.raises(ValueError, match="Include subdirectories"):
        validate_creation(args)
    args.recursive = True
    assert validate_creation(args) == [nested / "capture.mp4"]


def test_browser_lists_directories_first_and_can_filter_files(tmp_path):
    (tmp_path / "z-folder").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "a.toml").write_text("")
    assert [p.name for p in browse_entries(tmp_path, files=True)] == [".hidden", "z-folder", "a.toml"]
    assert [p.name for p in browse_entries(tmp_path, files=False)] == [".hidden", "z-folder"]


class Screen:
    def __init__(self, keys):
        self.keys = iter(keys)
        self.lines = []

    def getmaxyx(self):
        return 24, 100

    def erase(self):
        pass

    def refresh(self):
        pass

    def addnstr(self, row, column, text, width, *attrs):
        self.lines.append(text[:width])

    def get_wch(self):
        return next(self.keys)


def terminal(monkeypatch, keys):
    screen = Screen(keys)
    fake = SimpleNamespace(KEY_UP=1001, KEY_DOWN=1002, KEY_LEFT=1003,
                           KEY_BACKSPACE=1004, KEY_ENTER=1005, KEY_F5=1006,
                           A_REVERSE=1, error=RuntimeError,
                           wrapper=lambda callback: callback(screen))
    monkeypatch.setitem(sys.modules, "curses", fake)
    monkeypatch.setattr("pelagia_interchange.tui.terminal_available", lambda: True)
    return screen


def test_full_screen_create_browses_source_and_names_output(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.from_videos = None
    monkeypatch.chdir(tmp_path)
    # Browse source without typing a path, select output parent, type only its name.
    screen = terminal(monkeypatch, ["\n", 1002, "\n", "s", 1002, "\n", "s",
                                    "\x15", *"archive", "\n", 1006, "\n"])
    executed = []

    def execute(settings, progress):
        assert not settings.output.exists()
        executed.append((settings.from_videos, settings.output))
        progress("Transcoding capture.mp4")
        return "Created archive"

    assert run_creator(args, execute) == "Created archive"
    assert executed == [(tmp_path / "videos", tmp_path / "archive")]
    assert "Transcoding capture.mp4" in screen.lines


def test_browser_cancel_keeps_output_and_form_cancel_never_ingests(tmp_path, monkeypatch, capsys):
    args = options(tmp_path)
    terminal(monkeypatch, [1002, "\n", "\x1b", "\x1b"])
    from pelagia_interchange.tui import CreationCancelled
    with pytest.raises(CreationCancelled):
        run_creator(args, lambda *args: pytest.fail("must not ingest"))
    assert args.output is None
    terminal(monkeypatch, ["\x1b"])
    assert main(["create", "--tui"]) == 130
    assert "cancelled" in capsys.readouterr().err


def test_review_back_allows_editing_and_ingestion_errors_exit(tmp_path, monkeypatch):
    args = options(tmp_path)
    terminal(monkeypatch, [1006, "\x1b", 1006, "\n"])

    def execute(*args):
        raise ValueError("encoder failed")

    with pytest.raises(ValueError, match="encoder failed"):
        run_creator(args, execute)


def test_bad_settings_stay_in_form(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.jpeg_quality = 0
    screen = terminal(monkeypatch, [1006, "\x1b"])
    from pelagia_interchange.tui import CreationCancelled
    with pytest.raises(CreationCancelled):
        run_creator(args, lambda *args: pytest.fail("must not ingest"))
    assert any("quality must be" in line for line in screen.lines)


def test_explicit_tui_requires_terminal(monkeypatch, capsys):
    monkeypatch.setattr("pelagia_interchange.tui.terminal_available", lambda: False)
    assert main(["create", "--tui"]) == 2
    assert "needs a terminal" in capsys.readouterr().err


def test_metadata_file_can_be_chosen_without_typing_path(tmp_path, monkeypatch):
    args = options(tmp_path)
    metadata = tmp_path / "metadata.toml"
    metadata.write_text('[dataset]\ntitle = "Selected metadata"\n')
    monkeypatch.chdir(tmp_path)
    # Metadata is field 6. Browser sorts parent, videos directory, metadata file.
    terminal(monkeypatch, [*[1002] * 6, "\n", 1002, 1002, "\n", 1006, "\n"])
    assert run_creator(args, lambda settings, progress: str(settings.metadata)) == str(metadata)


def test_resume_validation_accepts_only_incomplete_dataset(tmp_path):
    from pelagia_interchange import DatasetBuilder
    args = options(tmp_path)
    args.output = tmp_path / "archive"
    builder = DatasetBuilder(args.output, title="Incomplete")
    args.resume = True
    assert validate_creation(args)
    builder.finalize()
    with pytest.raises(ValueError, match="incomplete"):
        validate_creation(args)


def test_unreadable_metadata_returns_to_form(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.metadata = tmp_path / "missing.toml"
    screen = terminal(monkeypatch, [1006, "\x1b"])
    from pelagia_interchange.tui import CreationCancelled
    with pytest.raises(CreationCancelled):
        run_creator(args, lambda *args: pytest.fail("must not ingest"))
    assert any("cannot read metadata" in line for line in screen.lines)
