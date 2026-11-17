"""Standard-library terminal creator; importing the package never requires curses."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

from .exceptions import InterchangeError
from .ingestion import discover_videos
from .manifest import Manifest
from .metadata import Metadata
from .util import parse_size


class CreationCancelled(Exception):
    """The user left the creator before starting ingestion."""


# name, label, help, editor (directory, output, file, bool, text, integer)
FIELDS = (
    ("from_videos", "Source videos", "Browse to a directory containing acquisition videos.", "directory"),
    ("output", "Output dataset", "Choose a parent directory, then name a new dataset.", "output"),
    ("title", "Title", "Scientific dataset title; blank uses the output directory name.", "text"),
    ("description", "Description", "Initial dataset description.", "text"),
    ("stream", "Camera / stream", "Name recorded with every frame.", "text"),
    ("recursive", "Include subdirectories", "Discover supported videos recursively.", "bool"),
    ("metadata", "Metadata TOML", "Optional existing metadata file. Backspace clears the selection.", "file"),
    ("shard_target_size", "Target shard size", "For example 10GB or 2GiB; rollover occurs at frame boundaries.", "text"),
    ("jpeg_quality", "JPEG quality", "1-100. Video decoding transcodes frames to JPEG.", "integer"),
    ("jpeg_subsampling", "JPEG subsampling", "444, 422, or 420; ignored for grayscale.", "text"),
    ("grayscale", "Grayscale", "Convert decoded frames to grayscale.", "bool"),
    ("jpeg_workers", "JPEG workers", "Positive integer; blank uses the importer default.", "integer"),
    ("queue_depth", "Queue depth", "Positive integer; blank uses the importer default.", "integer"),
    ("staging_dir", "Scratch directory", "Optional existing local directory for source copies and active shards.", "directory"),
    ("source_file_boundary", "Shard per source", "Start a new shard for every source video.", "bool"),
    ("no_source_hash", "Skip source hashing", "Default OFF: SHA-256 records source identity during staging.", "bool"),
    ("no_previews", "Disable previews", "Skip representative thumbnails and contact sheets.", "bool"),
    ("preview_count", "Previews per video", "Positive number of representative thumbnails.", "integer"),
    ("preview_width", "Preview width", "Maximum thumbnail width, 32-8192 pixels.", "integer"),
    ("require_previews", "Require previews", "Fail creation if preview generation fails.", "bool"),
    ("resume", "Resume incomplete dataset", "Select the existing output dataset using Browse instead of naming a new one.", "bool"),
    ("ffmpeg", "FFmpeg executable", "Executable name on PATH or a full path.", "text"),
    ("ffprobe", "FFprobe executable", "Executable name on PATH or a full path.", "text"),
    ("turbojpeg_library", "TurboJPEG library", "Optional library file; blank uses automatic discovery.", "file"),
)


def terminal_available() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb"


def validate_creation(args: argparse.Namespace) -> list[Path]:
    """Read-only checks before the confirmation screen; ingestion stays authoritative."""
    if args.from_videos is None:
        raise ValueError("Select a source video directory.")
    source = args.from_videos.expanduser().resolve()
    if args.output is None:
        args.output = source.with_name(source.name + "_interchange")
    output = args.output.expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must not overlap.")
    if args.resume:
        manifest = Manifest.read(output / "manifest.json")
        if manifest.state not in {"building", "finalizing"}:
            raise ValueError("Resume requires an incomplete dataset.")
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory; use Resume for an incomplete dataset.")
    parse_size(args.shard_target_size)
    if not args.stream.strip():
        raise ValueError("Camera / stream must not be empty.")
    if args.jpeg_quality is None or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("JPEG quality must be 1-100.")
    if args.jpeg_subsampling not in {"444", "422", "420"}:
        raise ValueError("JPEG subsampling must be 444, 422, or 420.")
    for name in ("jpeg_workers", "queue_depth", "preflight_workers"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive.")
    if args.require_previews and args.no_previews:
        raise ValueError("Enable previews or disable Require previews.")
    if not args.no_previews and (args.preview_count is None or args.preview_count < 1 or args.preview_width is None or not 32 <= args.preview_width <= 8192):
        raise ValueError("Preview count must be positive and width must be 32-8192.")
    if args.staging_dir and not args.staging_dir.expanduser().is_dir():
        raise ValueError("Scratch directory must exist.")
    if args.metadata:
        Metadata.read(args.metadata.expanduser())
    videos = discover_videos(source, recursive=args.recursive)
    if not videos:
        raise ValueError("No supported videos found. Try Include subdirectories or another folder.")
    for name in ("from_videos", "output", "metadata", "staging_dir", "turbojpeg_library"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    return videos


def browse_entries(directory: Path, *, files: bool) -> list[Path]:
    """List navigable directories first, including hidden and mounted directories."""
    return sorted((p for p in directory.iterdir() if files or p.is_dir()),
                  key=lambda p: (not p.is_dir(), p.name.casefold(), p.name))


def run_creator(args: argparse.Namespace, execute: Callable[[argparse.Namespace, Callable[[str], None]], str]) -> str:
    if not terminal_available():
        raise ValueError("The full-screen creator needs a terminal. Use --interactive for plain prompts.")
    try:
        import curses
    except ImportError as exc:
        raise ValueError("curses is unavailable; use --interactive for plain prompts.") from exc

    def session(screen, progress):
        def put(y, text, selected=False):
            height, width = screen.getmaxyx()
            if 0 <= y < height:
                # Keep filenames and progress messages from emitting terminal controls.
                clean = "".join(c if c.isprintable() else "?" for c in str(text))
                try:
                    screen.addnstr(y, 0, clean, max(0, width - 1), curses.A_REVERSE if selected else 0)
                except curses.error:
                    pass  # A concurrent resize can invalidate the previous dimensions.

        def draw(title, rows, index=0, footer="", message=""):
            screen.erase()
            height, width = screen.getmaxyx()
            put(0, title, True)
            if height < 10 or width < 50:
                put(2, "Enlarge terminal to at least 50 columns / 10 rows.")
            else:
                visible = height - 6
                offset = max(0, index - visible + 1)
                for row, text in enumerate(rows[offset:offset + visible], 2):
                    put(row, text, offset + row - 2 == index)
                put(height - 3, message)
                put(height - 2, footer)
            screen.refresh()

        def edit(label, value):
            original = str(value) if value is not None else ""
            text = original
            while True:
                draw(label, [text[-max(1, screen.getmaxyx()[1] - 2):]], footer="Type to edit | Ctrl+U clear | Enter save | Esc cancel")
                key = screen.get_wch()
                if key in ("\n", "\r", curses.KEY_ENTER):
                    return text.strip()
                if key == "\x1b":
                    return original
                if key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                    text = text[:-1]
                elif key == "\x15":
                    text = ""
                elif isinstance(key, str) and key.isprintable():
                    text += key

        def browse(value, files=False):
            directory = Path(value).expanduser().absolute() if value else Path.cwd()
            if not directory.is_dir():
                directory = directory.parent
            while not directory.is_dir() and directory != directory.parent:
                directory = directory.parent
            index, message = 0, ""
            while True:
                try:
                    entries = [directory.parent] + browse_entries(directory, files=files)
                except OSError as exc:
                    entries, message = [directory.parent], str(exc)
                rows = ["../"] + [("[dir] " if p.is_dir() else "      ") + p.name for p in entries[1:]]
                draw(f"Browse: {directory}", rows, index,
                     ("Up/Down | Enter open/select file | Left parent | / path | Esc back" if files else
                      "Up/Down | Enter open | Left parent | s select folder | / path | Esc back"), message)
                key = screen.get_wch()
                if key == "\x1b":
                    return None
                if key in (curses.KEY_LEFT, curses.KEY_BACKSPACE, "\x7f", "\b"):
                    directory, index, message = directory.parent, 0, ""
                elif key == "/":
                    candidate = Path(edit("Go to directory", directory)).expanduser().absolute()
                    if candidate.is_dir():
                        directory, index, message = candidate, 0, ""
                    else:
                        message = "Directory does not exist."
                elif key == "s" and not files:
                    return directory
                elif key in (curses.KEY_UP, "k"):
                    index = max(0, index - 1)
                elif key in (curses.KEY_DOWN, "j"):
                    index = min(len(entries) - 1, index + 1)
                elif key in ("\n", "\r", curses.KEY_ENTER):
                    chosen = entries[index]
                    if chosen.is_dir():
                        directory, index, message = chosen, 0, ""
                    elif files:
                        return chosen

        index, message = 0, ""
        while True:
            rows = [f"{label:29} {('ON' if value else 'OFF') if kind == 'bool' else (str(value) if value is not None else '(choose / default)')}"
                    for name, label, _, kind in FIELDS for value in [getattr(args, name)]]
            draw("Pelagia interchange | Create from videos", rows, index,
                 "Up/Down move | Enter edit/browse | Backspace clear | F5 review | Esc cancel",
                 message or FIELDS[index][2])
            key = screen.get_wch()
            name, label, _, kind = FIELDS[index]
            value = getattr(args, name)
            if key == "\x1b":
                raise CreationCancelled()
            if key in (curses.KEY_UP, "k"):
                index = max(0, index - 1)
            elif key in (curses.KEY_DOWN, "j", "\t"):
                index = (index + 1) % len(FIELDS)
            elif key in (curses.KEY_BACKSPACE, "\x7f", "\b") and name in {"metadata", "staging_dir", "turbojpeg_library", "jpeg_workers", "queue_depth"}:
                setattr(args, name, None)
            elif key in ("\n", "\r", curses.KEY_ENTER, " "):
                message = ""
                try:
                    if kind == "bool":
                        setattr(args, name, not value)
                    elif kind in {"directory", "file", "output"}:
                        chosen = browse(value, files=kind == "file")
                        if chosen is None:
                            continue
                        if kind == "output" and not args.resume:
                            folder_name = edit("New dataset name inside " + str(chosen),
                                               args.output.name if args.output else
                                               (args.from_videos.name + "_interchange" if args.from_videos else "dataset"))
                            if not folder_name or folder_name in {".", ".."} or Path(folder_name).name != folder_name:
                                raise ValueError("Use a single directory name.")
                            chosen = chosen / folder_name
                        setattr(args, name, chosen)
                    else:
                        text = edit(label, value)
                        setattr(args, name, (int(text) if text else None) if kind == "integer" else text)
                except ValueError as exc:
                    message = str(exc)
            elif key == curses.KEY_F5:
                try:
                    draw("Checking source files...", [], footer="No dataset has been written.")
                    videos = validate_creation(args)
                    review = [f"{label}: {getattr(args, name)}" for name, label, _, _ in FIELDS]
                    review += [f"{len(videos)} supported video(s); {sum(p.stat().st_size for p in videos):,} source bytes",
                               "Video frames will be transcoded to JPEG."]
                    review += [str(p.relative_to(args.from_videos)) for p in videos]
                except (InterchangeError, OSError, ValueError) as exc:
                    message = str(exc)
                    continue
                position = 0
                while True:
                    draw("Review | Enter starts creation", review, position,
                         "Up/Down scroll | Enter create | Esc return to settings")
                    answer = screen.get_wch()
                    if answer == "\x1b":
                        break
                    if answer in (curses.KEY_DOWN, "j"):
                        position = min(len(review) - 1, position + 1)
                    elif answer in (curses.KEY_UP, "k"):
                        position = max(0, position - 1)
                    elif answer in ("\n", "\r", curses.KEY_ENTER):
                        return execute(args, progress)


    messages: list[str] = []
    # Ingestion is synchronous and retains its existing interrupt/cleanup behavior.
    def wrapped(screen):
        def update(message):
            messages.append(message)
            del messages[:-100]
            screen.erase()
            height, width = screen.getmaxyx()
            lines = ["Pelagia interchange | Creating dataset", str(args.output),
                     "Ctrl+C interrupts; incomplete data and scratch can be resumed."] + messages[-max(1, height - 5):]
            for row, line in enumerate(lines[:height - 1]):
                try:
                    screen.addnstr(row, 0, "".join(c if c.isprintable() else "?" for c in line), max(0, width - 1))
                except curses.error:
                    pass
            screen.refresh()
        return session(screen, update)

    try:
        return curses.wrapper(wrapped)
    except curses.error as exc:
        raise ValueError("Cannot initialize the terminal creator; check TERM or use plain prompts.") from exc
