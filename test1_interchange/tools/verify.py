#!/usr/bin/env python3
"""Dependency-free quick, structural, or full verifier."""
import argparse, hashlib, json, sqlite3, sys
from pathlib import Path
REQUIRED=("manifest.json","metadata.toml","history.jsonl","README.md","checksums.sha256")
TABLES={"frames","source_files","storage_formats","shard_metadata"}
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()
def safe(root,value):
    rel=Path(value)
    if rel.is_absolute() or ".." in rel.parts: raise ValueError("unsafe package path")
    path=(root/rel).resolve()
    if root.resolve() not in path.parents: raise ValueError("package path escapes root")
    return path
def main():
    p=argparse.ArgumentParser(); p.add_argument("path",type=Path); p.add_argument("--level",choices=("quick","structural","full","archival"),default="quick"); p.add_argument("--json",action="store_true"); a=p.parse_args(); root=a.path; errors=[]; warnings=[]; frames=0
    for name in REQUIRED:
        if not (root/name).is_file(): errors.append(f"missing required file: {name}")
    try: manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    except Exception as exc: manifest={}; errors.append(f"manifest: {exc}")
    source_expected={x.get("source_file_id"):x.get("frame_count") for x in manifest.get("source_files",[])}; source_observed={x:0 for x in source_expected}; previous={}; previous_source={}
    for item in sorted(manifest.get("shards",[]),key=lambda x:(str(x.get("stream_uuid")),x.get("first_frame") if x.get("first_frame") is not None else -1)):
        try:
            path=safe(root,item["relative_path"])
            if not path.is_file(): errors.append(f"missing shard: {path}"); continue
            if path.stat().st_size!=item.get("byte_size"): errors.append(f"size mismatch: {path}")
            record=item.get("file_hash",{})
            if record.get("algorithm")=="sha256" and digest(path)!=record.get("value"): errors.append(f"hash mismatch: {path}")
            if a.level in ("structural","full","archival"):
                with sqlite3.connect("file:"+str(path.resolve())+"?mode=ro",uri=True) as db:
                    if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok": errors.append(f"integrity failure: {path}")
                    tables={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if TABLES-tables: errors.append(f"missing tables in {path}: {sorted(TABLES-tables)}"); continue
                    count=db.execute("SELECT count(*) FROM frames").fetchone()[0]
                    if count!=item.get("frame_count"): errors.append(f"frame count mismatch: {path}")
                    local={key:json.loads(value) for key,value in db.execute("SELECT key,value FROM shard_metadata")}
                    aggregates=db.execute("SELECT count(*),min(frame_id),max(frame_id),min(timestamp_ns),max(timestamp_ns),coalesce(sum(byte_size),0) FROM frames").fetchone()
                    for key,value in zip(("frame_count","first_frame","last_frame","first_timestamp","last_timestamp","encoded_bytes"),aggregates):
                        if local.get(key)!=value or item.get(key)!=value: errors.append(f"shard metadata mismatch for {key}: {path}")
                    for source_id,count in db.execute("SELECT source_file_id,count(*) FROM frames GROUP BY source_file_id"):
                        source_observed[source_id]=source_observed.get(source_id,0)+count
                    if a.level in ("full","archival"):
                        stream=str(item.get("stream_uuid"))
                        for fid,source_id,source_number,blob,size,algorithm,expected,status in db.execute("SELECT frame_id,source_file_id,source_frame_number,blob,byte_size,hash_algorithm,hash,status FROM frames ORDER BY frame_id"):
                            frames+=1
                            if stream in previous and fid!=previous[stream]+1: errors.append(f"duplicate or unrepresented frame ID near {stream}:{fid}")
                            previous[stream]=fid; source_key=(stream,source_id)
                            if source_key in previous_source and source_number!=previous_source[source_key]+1: errors.append(f"duplicate or unrepresented source frame near {source_id}:{source_number}")
                            previous_source[source_key]=source_number
                            if len(blob or b"")!=size: errors.append(f"BLOB size mismatch: {path}:{fid}")
                            if blob is not None and algorithm=="sha256" and hashlib.sha256(blob).hexdigest()!=expected: errors.append(f"BLOB hash mismatch: {path}:{fid}")
                            if blob is not None and not expected: (errors if a.level=="archival" else warnings).append(f"missing BLOB hash: {path}:{fid}")
                            if blob is not None and algorithm not in (None,"sha256"): (errors if a.level=="archival" else warnings).append(f"unavailable BLOB hash algorithm {algorithm}: {path}:{fid}")
                            if blob is None and status=="valid": errors.append(f"valid frame lacks BLOB: {path}:{fid}")
        except Exception as exc: errors.append(f"{item.get('relative_path')}: {exc}")
    checks=root/"checksums.sha256"
    if checks.is_file():
        seen=set()
        for line in checks.read_text(encoding="utf-8").splitlines():
            expected,sep,rel=line.partition("  ")
            try:
                path=safe(root,rel)
                if not sep or not path.is_file() or digest(path)!=expected: errors.append(f"package checksum mismatch: {rel}")
                if rel in seen: errors.append(f"duplicate checksum path: {rel}")
                seen.add(rel)
            except Exception as exc: errors.append(str(exc))
        expected_paths={str(x.relative_to(root)).replace("\\","/") for x in root.rglob("*") if x.is_file() and x.name!="checksums.sha256" and not x.name.endswith((".partial",".tmp"))}
        for rel in sorted(expected_paths-seen): errors.append(f"package file absent from checksums: {rel}")
    if a.level=="archival":
        if list((root/"data").glob("*.partial")): errors.append("partial shards remain")
        operations=[]
        try:
            operations=[event.get("operation") for event in (json.loads(line) for line in (root/"history.jsonl").read_text().splitlines() if line.strip()) if event.get("status")=="success"]
        except Exception as exc: errors.append(f"history: {exc}")
        for required in ("dataset_created","dataset_finalized"):
            if required not in operations: errors.append(f"missing history event: {required}")
        if manifest.get("shards") and "shard_finalized" not in operations: errors.append("missing history event: shard_finalized")
        for source_id,expected in source_expected.items():
            if expected is None: errors.append(f"source {source_id} has no expected frame count")
            elif source_observed.get(source_id,0)!=expected: errors.append(f"source {source_id} expected {expected} frames, represented {source_observed.get(source_id,0)}")
    result={"valid":not errors,"archival_ready":a.level=="archival" and not errors,"level":a.level,"errors":errors,"warnings":warnings,"checked_frames":frames}
    print(json.dumps(result,indent=2) if a.json else (("ARCHIVAL READY" if result["archival_ready"] else "VALID" if result["valid"] else "INVALID")+f": {len(errors)} error(s), {len(warnings)} warning(s)"))
    if not a.json:
        for error in errors: print("ERROR:",error)
    return 0 if not errors else 2
if __name__=="__main__": raise SystemExit(main())
