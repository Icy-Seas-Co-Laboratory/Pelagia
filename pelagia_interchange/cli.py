from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .builder import DatasetBuilder
from .dataset import Dataset
from .exceptions import InterchangeError
from .extraction import extract_frames
from .ingestion import DEFAULT_FFMPEG_QSCALE, discover_videos, ingest_video_directory
from .metadata import Metadata
from .validation import Validator


def _range(value: str) -> tuple[int | None, int | None]:
    start, separator, end = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("range must be START:END")
    return (int(start) if start else None, int(end) if end else None)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pii", description="Scientific Image Interchange tools")
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a dataset, optionally ingesting a directory of videos")
    create.add_argument("output", type=Path, nargs="?"); create.add_argument("--title"); create.add_argument("--description", default="")
    create.add_argument("--shard-target-size", default="10GB")
    create.add_argument("--from-videos", type=Path, metavar="DIRECTORY")
    create.add_argument("--recursive", action="store_true"); create.add_argument("--stream", default="camera")
    create.add_argument("--metadata", type=Path, metavar="TOML"); create.add_argument("--interactive", action="store_true")
    create.add_argument("--ffmpeg", default="ffmpeg"); create.add_argument("--ffprobe", default="ffprobe")
    create.add_argument("--ffmpeg-qscale", type=int, default=DEFAULT_FFMPEG_QSCALE,
                        help="deprecated compatibility option; use --jpeg-quality")
    create.add_argument("--jpeg-quality", type=int, default=90)
    create.add_argument("--jpeg-subsampling", choices=("444", "422", "420"), default="444")
    create.add_argument("--jpeg-workers", type=int, default=None)
    create.add_argument("--preflight-workers", type=int, default=None)
    create.add_argument("--queue-depth", type=int, default=None)
    create.add_argument("--turbojpeg-library", type=Path, metavar="PATH")
    create.add_argument("--grayscale", action="store_true")
    create.add_argument("--no-source-hash", action="store_true"); create.add_argument("--source-file-boundary", action="store_true")
    create.add_argument("--no-previews", action="store_true"); create.add_argument("--preview-count", type=int, default=12)
    create.add_argument("--preview-width", type=int, default=512); create.add_argument("--require-previews", action="store_true")
    create.add_argument("--resume", action="store_true", help="resume an incomplete video ingestion")
    inspect = commands.add_parser("inspect", help="summarize a dataset")
    inspect.add_argument("path", type=Path); inspect.add_argument("--json", action="store_true")
    verify = commands.add_parser("verify", help="verify package integrity")
    verify.add_argument("path", type=Path); verify.add_argument("--level", choices=("quick", "structural", "full", "archival"), default="quick")
    verify.add_argument("--image-signatures", action="store_true"); verify.add_argument("--json", action="store_true")
    extract = commands.add_parser("extract", help="stream retained images to ordinary files")
    extract.add_argument("path", type=Path); extract.add_argument("--output", type=Path, default=Path("frames"))
    extract.add_argument("--camera"); extract.add_argument("--frame", type=int); extract.add_argument("--frames", type=_range)
    extract.add_argument("--source-file", type=int); extract.add_argument("--source-uuid"); extract.add_argument("--shard")
    extract.add_argument("--timestamps", type=_range); extract.add_argument("--overwrite", choices=("error", "skip", "replace"), default="error")
    extract.add_argument("--source-frame-names", action="store_true"); extract.add_argument("--dry-run", action="store_true")
    metadata = commands.add_parser("metadata", help="print parsed metadata")
    metadata.add_argument("path", type=Path); metadata.add_argument("--json", action="store_true")
    history = commands.add_parser("history", help="print provenance events")
    history.add_argument("path", type=Path); history.add_argument("--json", action="store_true")
    shards = commands.add_parser("shards", help="list finalized or abandoned partial shards")
    shards.add_argument("path", type=Path); shards.add_argument("--json", action="store_true")
    shards.add_argument("--partials", action="store_true"); shards.add_argument("--quarantine-partials", type=Path)
    sources = commands.add_parser("sources", help="list manifest sources")
    sources.add_argument("path", type=Path); sources.add_argument("--json", action="store_true")
    return root


def _pretty(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _prompt(label: str, default: str | None = None) -> str:
    value = input(f"{label}{f' [{default}]' if default is not None else ''}: ").strip()
    if value:
        return value
    if default is not None:
        return default
    raise ValueError(f"{label} is required")


def _interactive_create(args: argparse.Namespace) -> None:
    print("Interactive Scientific Image Interchange creation")
    args.from_videos = Path(_prompt("Directory containing video files", str(args.from_videos) if args.from_videos else None))
    args.output = Path(_prompt("Output dataset directory", str(args.output) if args.output else args.from_videos.name + "_interchange"))
    args.title = _prompt("Dataset title", args.title or args.output.name)
    args.description = _prompt("Dataset description", args.description or "")
    args.stream = _prompt("Camera/stream name", args.stream)
    args.shard_target_size = _prompt("Target shard size", args.shard_target_size)
    args.recursive = _prompt("Search subdirectories? (y/n)", "y" if args.recursive else "n").lower().startswith("y")
    args.grayscale = _prompt("Transcode to grayscale? (y/n)", "y" if args.grayscale else "n").lower().startswith("y")
    args.source_file_boundary = _prompt("Start a new shard for every source video? (y/n)", "y" if args.source_file_boundary else "n").lower().startswith("y")
    args.no_previews = not _prompt("Generate representative previews? (y/n)", "n" if args.no_previews else "y").lower().startswith("y")
    if not args.no_previews:
        args.preview_count = int(_prompt("Representative thumbnail count", str(args.preview_count)))
        args.preview_width = int(_prompt("Maximum thumbnail width", str(args.preview_width)))
        args.require_previews = _prompt("Fail creation if previews fail? (y/n)", "y" if args.require_previews else "n").lower().startswith("y")
    else:
        args.require_previews = False
    metadata_value = _prompt("Existing metadata TOML (blank to generate minimal metadata)", "")
    args.metadata = Path(metadata_value) if metadata_value else None
    videos = discover_videos(args.from_videos, recursive=args.recursive)
    print(f"\nFound {len(videos)} supported video file(s):")
    for path in videos[:20]: print(f"  {path.relative_to(args.from_videos)}")
    if len(videos) > 20: print(f"  ... and {len(videos) - 20} more")
    if _prompt("Create this dataset? (y/n)", "y").lower() not in {"y", "yes"}:
        raise ValueError("creation cancelled")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "create":
            if args.interactive: _interactive_create(args)
            if args.output is None and args.from_videos is not None:
                source_directory = args.from_videos.resolve()
                args.output = source_directory.parent / f"{source_directory.name}_interchange"
            if args.output is None: raise ValueError("output path is required (or use --interactive)")
            if args.from_videos is not None:
                metadata = Metadata.read(args.metadata) if args.metadata else None
                result = ingest_video_directory(args.from_videos, args.output, title=args.title,
                                                description=args.description, stream=args.stream,
                                                recursive=args.recursive, shard_target_size=args.shard_target_size,
                                                metadata=metadata, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                                                ffmpeg_qscale=args.ffmpeg_qscale, grayscale=args.grayscale,
                                                jpeg_quality=args.jpeg_quality, jpeg_subsampling=args.jpeg_subsampling,
                                                jpeg_workers=args.jpeg_workers,
                                                preflight_workers=args.preflight_workers,
                                                queue_depth=args.queue_depth,
                                                turbojpeg_library=args.turbojpeg_library,
                                                hash_sources=not args.no_source_hash,
                                                source_file_boundary=args.source_file_boundary,
                                                generate_previews=not args.no_previews,
                                                preview_count=args.preview_count, preview_width=args.preview_width,
                                                require_previews=args.require_previews,
                                                resume=args.resume,
                                                progress=lambda message: print(message, file=sys.stderr))
                print(f"Created {args.output}: {result.source_files} source file(s), {result.frames} frame(s), {result.previews} representative preview(s)")
                return 0
            if args.resume:
                raise ValueError("--resume requires --from-videos")
            DatasetBuilder(args.output, title=args.title, description=args.description, shard_target_size=args.shard_target_size).finalize()
            print(f"Created {args.output}"); return 0
        if args.command == "verify":
            result = Validator(args.path).verify(args.level, image_signatures=args.image_signatures)
            if args.json:
                _pretty(result.to_dict())
            else:
                label = "ARCHIVAL READY" if result.archival_ready else "VALID" if result.valid else "INVALID"
                print(f"{label}: {args.level}; {len(result.errors)} error(s), {len(result.warnings)} warning(s)")
                for issue in result.errors: print(f"ERROR {issue.code}: {issue.message}")
                for issue in result.warnings: print(f"WARNING {issue.code}: {issue.message}")
            return 0 if result.valid else 2
        if args.command == "shards" and (args.partials or args.quarantine_partials):
            records = ([str(path) for path in DatasetBuilder.quarantine_partials(args.path, args.quarantine_partials)]
                       if args.quarantine_partials else [str(path) for path in DatasetBuilder.partials(args.path)])
            _pretty(records) if args.json else [print(item) for item in records]
            return 0
        dataset = Dataset.open(args.path)
        if args.command == "inspect":
            summary = dataset.summary(); _pretty(summary) if args.json else print("\n".join(f"{key}: {value}" for key, value in summary.items())); return 0
        if args.command == "extract":
            start, end = args.frames or (args.frame, args.frame)
            time_start, time_end = args.timestamps or (None, None)
            result = extract_frames(dataset, args.output, camera=args.camera, frame_start=start, frame_end=end,
                                    source_file_id=args.source_file, source_uuid=args.source_uuid, shard=args.shard,
                                    timestamp_start=time_start, timestamp_end=time_end, overwrite=args.overwrite,
                                    source_frame_names=args.source_frame_names, dry_run=args.dry_run,
                                    progress=lambda count: print(f"{count} frames", file=sys.stderr))
            print(f"selected={result.selected} written={result.written} skipped={result.skipped} bytes={result.bytes_written}"); return 0
        if args.command == "metadata":
            _pretty(dataset.metadata.data); return 0
        if args.command == "history":
            events = list(dataset.history); _pretty(events) if args.json else [print(json.dumps(item, sort_keys=True)) for item in events]; return 0
        records = dataset.manifest.shards if args.command == "shards" else dataset.manifest.source_files
        _pretty(records) if args.json else [print(json.dumps(item, sort_keys=True)) for item in records]
        return 0
    except (InterchangeError, OSError, ValueError) as exc:
        print(f"pii: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
