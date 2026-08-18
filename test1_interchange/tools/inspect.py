#!/usr/bin/env python3
"""Dependency-free package summary."""
import argparse, json, sqlite3, tomllib
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("path",type=Path); p.add_argument("--json",action="store_true"); a=p.parse_args()
    root=a.path; manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8")); metadata=tomllib.loads((root/"metadata.toml").read_text(encoding="utf-8"))
    status={}; codecs={}; encoded=0
    for shard in manifest.get("shards",[]):
        path=root/shard["relative_path"]
        with sqlite3.connect("file:"+str(path.resolve())+"?mode=ro",uri=True) as db:
            for name,count in db.execute("SELECT status,count(*) FROM frames GROUP BY status"): status[name]=status.get(name,0)+count
            for name,count,total in db.execute("SELECT coalesce(s.codec,'none'),count(*),coalesce(sum(f.byte_size),0) FROM frames f LEFT JOIN storage_formats s USING(storage_id) GROUP BY s.codec"):
                codecs[name]=codecs.get(name,0)+count; encoded+=total
    result={"format":manifest.get("format"),"format_version":manifest.get("format_version"),"schema_version":manifest.get("schema_version"),"dataset_uuid":manifest.get("dataset_uuid"),"state":manifest.get("state"),"title":metadata.get("dataset",{}).get("title"),"collection":metadata.get("collection",{}),"instruments":metadata.get("instruments",[]),"streams":metadata.get("streams",[]),"source_files":len(manifest.get("source_files",[])),"shards":len(manifest.get("shards",[])),"total_frames":sum(x.get("frame_count",0) for x in manifest.get("shards",[])),"encoded_image_bytes":encoded,"package_bytes":sum(x.stat().st_size for x in root.rglob("*") if x.is_file()),"storage_distribution":codecs,"status_distribution":status,"validation":manifest.get("validation",{})}
    print(json.dumps(result,indent=2,sort_keys=True) if a.json else "\n".join(f"{k}: {v}" for k,v in result.items()))
if __name__=="__main__": main()

