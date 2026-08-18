from __future__ import annotations

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
import sqlite3
from pathlib import Path

import pytest

from pelagia_interchange import (
    CompatibilityError, Dataset, DatasetBuilder, DatasetStateError, FrameStatus,
    Manifest, Metadata, StorageFormat, Validator,
)
from pelagia_interchange.cli import main
from pelagia_interchange.extraction import extract_frames
from pelagia_interchange.ingestion import VideoIngestionError, discover_videos, ingest_video_directory
from pelagia_interchange.util import hash_file

JPEG = b"\xff\xd8synthetic-jpeg\xff\xd9"
PNG = b"\x89PNG\r\n\x1a\nsynthetic"


def build(path: Path, *, count: int = 5, target: int = 10_000, cameras: tuple[str, ...] = ("port",)) -> Dataset:
    jpeg = StorageFormat("jpeg", quality=90, pixel_format="gray8", encoder="test")
    with DatasetBuilder(path, title="Synthetic", shard_target_bytes=target) as builder:
        sources = [builder.register_source_file(original_filename=f"source-{camera}.avi", frame_count=count) for camera in cameras]
        for camera, source in zip(cameras, sources):
            for number in range(count):
                builder.add_frame(stream=camera, source_file=source, frame_id=number,
                                  source_frame_number=number, encoded_bytes=JPEG,
                                  storage_format=jpeg, timestamp_ns=1_000_000_000 + number,
                                  timestamp_source="video PTS", timestamp_precision_ns=1_000_000)
    return Dataset.open(path)


def refresh_integrity(path: Path) -> None:
    manifest_path = path / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    for shard in raw["shards"]:
        shard_path = path / shard["relative_path"]
        shard["byte_size"] = shard_path.stat().st_size
        shard["file_hash"]["value"] = hash_file(shard_path)
    manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    ds = Dataset.open(path)
    ds.regenerate_checksums()


def test_create_open_get_and_exact_extract(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset")
    assert ds.frame_count == 5
    frame = ds.get_frame(camera="port", frame_number=3)
    assert frame.encoded_bytes == JPEG
    destination = tmp_path / "one.jpg"
    frame.save(destination)
    assert destination.read_bytes() == JPEG


def test_video_discovery_is_deterministic_and_optional_recursive(tmp_path: Path) -> None:
    (tmp_path / "B.MP4").write_bytes(b"video")
    (tmp_path / "a.avi").write_bytes(b"video")
    (tmp_path / "ignore.txt").write_text("no")
    nested = tmp_path / "nested"; nested.mkdir(); (nested / "c.mov").write_bytes(b"video")
    assert [path.name for path in discover_videos(tmp_path)] == ["a.avi", "B.MP4"]
    assert [path.name for path in discover_videos(tmp_path, recursive=True)] == ["a.avi", "B.MP4", "c.mov"]


def test_interactive_video_create_collects_and_confirms_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    videos = tmp_path / "videos"; videos.mkdir(); (videos / "a.mp4").write_bytes(b"video")
    output = tmp_path / "dataset"; answers = iter([str(videos), str(output), "Interactive title", "", "port", "1GB", "n", "y", "n", "y", "8", "320", "n", "", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    captured: dict = {}
    def fake_ingest(input_directory, destination, **kwargs):
        captured.update(input=input_directory, output=destination, **kwargs)
        return SimpleNamespace(source_files=1, frames=10, previews=8)
    monkeypatch.setattr("pelagia_interchange.cli.ingest_video_directory", fake_ingest)
    assert main(["create", "--interactive"]) == 0
    assert captured["input"] == videos and captured["output"] == output
    assert captured["title"] == "Interactive title" and captured["stream"] == "port"
    assert captured["grayscale"] is True and captured["shard_target_size"] == "1GB"
    assert captured["source_file_boundary"] is False
    assert captured["generate_previews"] is True and captured["preview_count"] == 8
    assert captured["preview_width"] == 320 and captured["require_previews"] is False


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable")
def test_ingest_video_directory_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    videos = tmp_path / "videos"; videos.mkdir(); video = videos / "sample_1.mp4"
    completed = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x24:rate=5",
                                "-frames:v", "3", "-pix_fmt", "yuv420p", str(video)],
                               check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    shutil.copyfile(video, videos / "sample_2.mp4")
    result = ingest_video_directory(videos, tmp_path / "dataset", title="Automatic import")
    assert result.source_files == 2 and result.frames == 6 and result.dataset.frame_count == 6
    assert result.previews == 6
    assert len(result.dataset.manifest.shards) == 1
    assert len(result.dataset.manifest.previews) == 8
    preview_index = json.loads((result.dataset.root / "preview" / "index.json").read_text())
    assert [item["frame_id"] for item in preview_index["frames"]] == list(range(6))
    assert all(item["source_uuid"] for item in preview_index["frames"])
    assert result.dataset.manifest.source_files[0]["frame_count"] == 3
    assert Validator(result.dataset.root).verify("archival", image_signatures=True).valid
    aligned = ingest_video_directory(videos, tmp_path / "source_aligned", title="Source aligned",
                                     source_file_boundary=True, generate_previews=False)
    assert len(aligned.dataset.manifest.shards) == 2
    assert aligned.previews == 0 and aligned.dataset.manifest.previews == []
    monkeypatch.setattr("pelagia_interchange.ingestion._thumbnail",
                        lambda *args, **kwargs: (_ for _ in ()).throw(VideoIngestionError("preview failure")))
    optional = ingest_video_directory(videos, tmp_path / "optional_preview_failure", title="Optional previews")
    assert optional.dataset.frame_count == 6 and optional.previews == 0
    assert Validator(optional.dataset.root).verify("archival").valid
    with pytest.raises(VideoIngestionError, match="preview failure"):
        ingest_video_directory(videos, tmp_path / "required_preview_failure", title="Required previews",
                               require_previews=True)


def test_multiple_sources_cameras_and_ranges(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=4, cameras=("port", "starboard"))
    assert len(ds.manifest.source_files) == 2
    assert [f.record.frame_id for f in ds.iter_frames(camera="starboard", frame_start=1, frame_end=2)] == [1, 2]
    source_uuid = ds.manifest.source_files[0]["source_uuid"]
    assert len(list(ds.iter_frames(source_uuid=source_uuid))) == 4


def test_shard_rollover_and_deterministic_names(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=4, target=len(JPEG) + 1)
    assert [Path(x["relative_path"]).name for x in ds.manifest.shards] == [
        "port_000001.sqlite", "port_000002.sqlite", "port_000003.sqlite", "port_000004.sqlite"
    ]


def test_manual_source_and_maximum_count_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "source-boundary"; fmt = StorageFormat("jpeg")
    with DatasetBuilder(path, title="Boundaries", source_file_boundary=True) as builder:
        first = builder.register_source_file(original_filename="a", frame_count=1)
        second = builder.register_source_file(original_filename="b", frame_count=1)
        builder.add_frame(stream="x", source_file=first, frame_id=0, source_frame_number=0, encoded_bytes=JPEG, storage_format=fmt)
        builder.add_frame(stream="x", source_file=second, frame_id=1, source_frame_number=0, encoded_bytes=JPEG, storage_format=fmt)
    assert len(Dataset.open(path).manifest.shards) == 2
    path = tmp_path / "count-boundary"
    with DatasetBuilder(path, title="Boundaries", maximum_frame_count=2) as builder:
        source = builder.register_source_file(original_filename="a", frame_count=4)
        for i in range(4):
            builder.add_frame(stream="x", source_file=source, frame_id=i, source_frame_number=i, encoded_bytes=JPEG, storage_format=fmt)
    assert [x["frame_count"] for x in Dataset.open(path).manifest.shards] == [2, 2]


def test_explicit_missing_and_corrupt_records(tmp_path: Path) -> None:
    path = tmp_path / "dataset"; fmt = StorageFormat("jpeg")
    with DatasetBuilder(path, title="Gaps") as builder:
        source = builder.register_source_file(original_filename="x.avi", frame_count=3)
        builder.add_frame(stream="camera", source_file=source, frame_id=10, source_frame_number=10, encoded_bytes=JPEG, storage_format=fmt)
        builder.add_frame(stream="camera", source_file=source, frame_id=11, source_frame_number=11, encoded_bytes=None, status=FrameStatus.MISSING)
        builder.add_frame(stream="camera", source_file=source, frame_id=12, source_frame_number=12, encoded_bytes=b"bad", storage_format=fmt, status=FrameStatus.CORRUPT)
    records = [f.record for f in Dataset.open(path).iter_frames()]
    assert [r.status for r in records] == [FrameStatus.VALID, FrameStatus.MISSING, FrameStatus.CORRUPT]
    assert records[1].encoded_bytes is None


def test_very_large_frame_id_and_empty_dataset(tmp_path: Path) -> None:
    empty = DatasetBuilder(tmp_path / "empty", title="Empty").finalize()
    assert empty.frame_count == 0
    path = tmp_path / "large"
    with DatasetBuilder(path, title="Large") as builder:
        source = builder.register_source_file(original_filename="x", frame_count=1)
        builder.add_frame(stream="x", source_file=source, frame_id=2**63 - 1,
                          source_frame_number=2**63 - 1, encoded_bytes=PNG, storage_format=StorageFormat("png"))
    assert Dataset.open(path).get_frame(camera="x", frame_number=2**63 - 1).encoded_bytes == PNG


def test_metadata_unknown_extension_and_manifest_field_roundtrip(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=1)
    ds.metadata.extensions["lab"] = {"nested": {"value": "preserved"}}
    ds.save_metadata(message="test edit")
    assert Dataset.open(ds.root).metadata.extensions["lab"]["nested"]["value"] == "preserved"
    raw = json.loads((ds.root / "manifest.json").read_text()); raw["future_field"] = {"x": 1}
    (ds.root / "manifest.json").write_text(json.dumps(raw))
    manifest = Manifest.read(ds.root / "manifest.json"); manifest.write(ds.root / "manifest.json")
    assert json.loads((ds.root / "manifest.json").read_text())["future_field"] == {"x": 1}


def test_annotated_metadata_nested_tables_roundtrip(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "docs" / "metadata.example.toml"
    metadata = Metadata.read(source); destination = tmp_path / "metadata.toml"; metadata.write(destination)
    reloaded = Metadata.read(destination)
    assert reloaded.data == metadata.data
    assert reloaded.data["instruments"][0]["cameras"][0]["name"] == "port"


@pytest.mark.parametrize("level", ["quick", "structural", "full", "archival"])
def test_verification_levels(tmp_path: Path, level: str) -> None:
    result = Validator(build(tmp_path / "dataset", count=3).root).verify(level)  # type: ignore[arg-type]
    assert result.valid, result.to_dict()
    assert result.archival_ready is (level == "archival")


def test_missing_and_tampered_shard_detection(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=2)
    shard = ds.root / ds.manifest.shards[0]["relative_path"]
    shard.write_bytes(shard.read_bytes() + b"tamper")
    assert "shard_size_mismatch" in {x.code for x in Validator(ds.root).verify("quick").errors}
    shard.unlink()
    assert "missing_shard" in {x.code for x in Validator(ds.root).verify("quick").errors}


def test_resources_and_checksum_inventory(tmp_path: Path) -> None:
    source = tmp_path / "calibration.json"; source.write_text('{"scale": 1.0}')
    path = tmp_path / "dataset"
    builder = DatasetBuilder(path, title="Resources")
    builder.add_resource(source, kind="calibration", description="pixel scale")
    ds = builder.finalize()
    assert Validator(path).verify("quick").valid
    (path / "extra.txt").write_text("not inventoried")
    assert "unchecksummed_package_file" in {x.code for x in Validator(path).verify("quick").errors}
    (path / "extra.txt").unlink()
    calibration = path / ds.manifest.calibration[0]["relative_path"]
    calibration.write_text("tampered")
    codes = {x.code for x in Validator(path).verify("quick").errors}
    assert {"resource_size_mismatch", "resource_hash_mismatch", "package_checksum_mismatch"} & codes


def test_tampered_frame_detected_after_container_hashes_refreshed(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=2)
    shard = ds.root / ds.manifest.shards[0]["relative_path"]
    with sqlite3.connect(shard) as db:
        db.execute("UPDATE frames SET blob=? WHERE frame_id=1", (b"different",))
    refresh_integrity(ds.root)
    assert "blob_hash_mismatch" in {x.code for x in Validator(ds.root).verify("full").errors}


def test_corrupt_sqlite_and_path_traversal_are_handled(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=1)
    manifest_path = ds.root / "manifest.json"; raw = json.loads(manifest_path.read_text())
    raw["shards"][0]["relative_path"] = "../escape.sqlite"; manifest_path.write_text(json.dumps(raw))
    assert "unsafe_shard_path" in {x.code for x in Validator(ds.root).verify("quick").errors}
    raw["shards"][0]["relative_path"] = "data/broken.sqlite"; manifest_path.write_text(json.dumps(raw))
    (ds.root / "data" / "broken.sqlite").write_bytes(b"not sqlite")
    errors = Validator(ds.root).verify("structural").errors
    assert any(x.code in {"shard_size_mismatch", "shard_hash_mismatch", "shard_structure"} for x in errors)


def test_extraction_filters_naming_overwrite_and_dry_run(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=3)
    out = tmp_path / "out"
    result = extract_frames(ds, out, camera="port", frame_start=1, frame_end=2, source_frame_names=True)
    assert result.written == 2
    assert sorted(p.name for p in out.iterdir()) == ["source_000001_000000000001.jpg", "source_000001_000000000002.jpg"]
    assert extract_frames(ds, out, frame_start=1, frame_end=2, source_frame_names=True, overwrite="skip").skipped == 2
    dry = extract_frames(ds, tmp_path / "dry", dry_run=True)
    assert dry.written == 3 and not (tmp_path / "dry").exists()


def test_interrupted_builder_preserves_partial_and_is_not_openable(tmp_path: Path) -> None:
    path = tmp_path / "dataset"; builder = DatasetBuilder(path, title="Interrupted")
    source = builder.register_source_file(original_filename="x", frame_count=1)
    builder.add_frame(stream="x", source_file=source, frame_id=0, source_frame_number=0,
                      encoded_bytes=JPEG, storage_format=StorageFormat("jpeg"))
    builder.abort(message="simulated")
    assert DatasetBuilder.partials(path)
    with pytest.raises(DatasetStateError): Dataset.open(path)
    assert Dataset.open(path, allow_incomplete=True).manifest.state == "building"
    quarantine = tmp_path / "recovery"
    moved = DatasetBuilder.quarantine_partials(path, quarantine)
    assert len(moved) == 1 and moved[0].is_file()
    assert not DatasetBuilder.partials(path)


def test_compatibility_checks(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=1)
    path = ds.root / "manifest.json"; raw = json.loads(path.read_text()); raw["format_version"] = "2.0"; path.write_text(json.dumps(raw))
    with pytest.raises(CompatibilityError): Dataset.open(ds.root)


def test_standalone_tools_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = build(tmp_path / "dataset", count=2)
    assert main(["verify", str(ds.root), "--level", "full", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    completed = subprocess.run([sys.executable, str(ds.root / "tools" / "verify.py"), str(ds.root), "--level", "full", "--json"],
                               check=False, capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["valid"]


def test_new_minor_and_schema_compatibility(tmp_path: Path) -> None:
    ds = build(tmp_path / "dataset", count=1); path = ds.root / "manifest.json"
    raw = json.loads(path.read_text()); raw["format_version"] = "1.99"; path.write_text(json.dumps(raw))
    assert Dataset.open(ds.root).manifest.format_version == "1.99"
    raw["schema_version"] = "2"; path.write_text(json.dumps(raw))
    with pytest.raises(CompatibilityError): Dataset.open(ds.root)
