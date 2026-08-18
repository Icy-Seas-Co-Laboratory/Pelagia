#!/usr/bin/env python3
"""Dependency-free streaming extractor for Scientific Image Interchange datasets."""
import argparse, json, sqlite3, sys
from pathlib import Path

def frame_range(value):
    try:
        a, b = value.split(":", 1); return (int(a) if a else None, int(b) if b else None)
    except ValueError as exc: raise argparse.ArgumentTypeError("expected START:END") from exc

def safe(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts: raise ValueError("unsafe path in manifest")
    path = (root / rel).resolve()
    if root.resolve() not in path.parents: raise ValueError("path escapes dataset")
    return path

def main():
    p = argparse.ArgumentParser(); p.add_argument("path", type=Path); p.add_argument("--output", type=Path, default=Path("frames"))
    p.add_argument("--all", action="store_true"); p.add_argument("--camera"); p.add_argument("--frame", type=int); p.add_argument("--frames", type=frame_range)
    p.add_argument("--source-file", type=int); p.add_argument("--source-uuid"); p.add_argument("--shard"); p.add_argument("--timestamps", type=frame_range)
    p.add_argument("--source-frame-names", action="store_true"); p.add_argument("--overwrite", choices=("error","skip","replace"), default="error")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--progress", type=int, default=1000); args = p.parse_args()
    if args.path.is_file(): root, shards, sources = args.path.parent, [{"relative_path": args.path.name}], []
    else:
        root = args.path
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")); shards = manifest.get("shards", []); sources = manifest.get("source_files", [])
    source_id = args.source_file
    if args.source_uuid:
        source = next((x for x in sources if x.get("source_uuid") == args.source_uuid), None)
        if source is None: p.error("source UUID not found")
        source_id = source["source_file_id"]
    start, end = args.frames or (args.frame, args.frame); ts0, ts1 = args.timestamps or (None, None)
    out = args.output.resolve()
    if not args.dry_run: out.mkdir(parents=True, exist_ok=True)
    written = skipped = byte_count = 0
    for shard in shards:
        if args.camera and args.camera not in (shard.get("stream_name"), shard.get("stream_uuid")): continue
        if args.shard and args.shard not in (shard.get("shard_uuid"), shard.get("relative_path"), Path(shard.get("relative_path","")).name): continue
        path = safe(root, shard["relative_path"])
        clauses, values = [], []
        for sql, value in (("f.frame_id>=?",start),("f.frame_id<=?",end),("f.source_file_id=?",source_id),("f.timestamp_ns>=?",ts0),("f.timestamp_ns<=?",ts1)):
            if value is not None: clauses.append(sql); values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        uri = "file:" + str(path.resolve()) + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            sql = "SELECT f.frame_id,f.source_file_id,f.source_frame_number,f.blob,s.codec FROM frames f LEFT JOIN storage_formats s USING(storage_id)" + where + " ORDER BY f.frame_id"
            for frame_id, sid, source_frame, blob, codec in db.execute(sql, values):
                if blob is None: skipped += 1; continue
                ext = {"jpeg":".jpg","jpg":".jpg","png":".png"}.get((codec or "").lower(), ".bin")
                stem = f"source_{sid:06d}_{source_frame:012d}" if args.source_frame_names else f"{frame_id:012d}"
                destination = (out / (stem + ext)).resolve()
                if destination.parent != out: raise ValueError("unsafe output path")
                if destination.exists() and args.overwrite != "replace":
                    if args.overwrite == "skip": skipped += 1; continue
                    raise FileExistsError(destination)
                if not args.dry_run: destination.write_bytes(blob)
                written += 1; byte_count += len(blob)
                if args.progress and written % args.progress == 0: print(f"{written} frames", file=sys.stderr)
    print(f"written={written} skipped={skipped} bytes={byte_count}")
if __name__ == "__main__": main()

