#!/usr/bin/env python3
"""Small, dependency-free throughput benchmark; scale --frames for host testing."""
from __future__ import annotations
import argparse, random, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pelagia_interchange import Dataset, DatasetBuilder, StorageFormat, Validator
from pelagia_interchange.extraction import extract_frames

def rate(label, count, seconds): print(f"{label}: {count / max(seconds, 1e-9):,.1f}/s ({seconds:.3f}s)")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--frames",type=int,default=10000); p.add_argument("--bytes",type=int,default=10000); p.add_argument("--keep",type=Path); a=p.parse_args()
    temp=None; root=a.keep
    if root is None: temp=tempfile.TemporaryDirectory(); root=Path(temp.name)/"benchmark"
    payload=b"\xff\xd8"+b"x"*max(0,a.bytes-4)+b"\xff\xd9"; fmt=StorageFormat("jpeg",quality=90,pixel_format="gray8")
    builder=DatasetBuilder(root,title="Benchmark",shard_target_bytes=max(len(payload)*a.frames//4,1))
    source=builder.register_source_file(original_filename="synthetic.avi",frame_count=a.frames)
    start=time.perf_counter()
    for i in range(a.frames): builder.add_frame(stream="camera",source_file=source,frame_id=i,source_frame_number=i,encoded_bytes=payload,storage_format=fmt)
    rate("sequential inserts",a.frames,time.perf_counter()-start)
    start=time.perf_counter(); ds=builder.finalize(); print(f"shard/package finalization: {time.perf_counter()-start:.3f}s")
    start=time.perf_counter(); count=sum(1 for _ in ds.iter_frames()); rate("sequential reads",count,time.perf_counter()-start)
    ids=random.sample(range(a.frames),min(1000,a.frames)); start=time.perf_counter()
    for i in ids: ds.get_frame(camera="camera",frame_number=i)
    rate("random reads",len(ids),time.perf_counter()-start)
    extraction_root=root.parent/"extracted"; start=time.perf_counter(); extracted=extract_frames(ds,extraction_root)
    rate("sequential extraction",extracted.written,time.perf_counter()-start)
    start=time.perf_counter(); result=Validator(root).verify("full"); rate("full verification frames",result.checked_frames,time.perf_counter()-start)
    print("valid:",result.valid,"shards:",len(ds.manifest.shards),"path:",root)
if __name__=="__main__": main()
