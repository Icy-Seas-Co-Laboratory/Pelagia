# Using `pelagia-interchange`

This guide shows how to create, inspect, extract, and verify Scientific Image Interchange datasets. The format is generic: Pelagia can produce or consume it, but a dataset does not depend on Pelagia and remains readable with Python and SQLite.

For normative requirements, see the [Scientific Image Interchange Format 1.0 specification](interchange-specification.md). For a complete scientific metadata template, see [metadata.example.toml](metadata.example.toml).

## Install

The interchange package targets Python 3.11 or newer and has no required third-party runtime dependencies.

From this repository:

```bash
python -m pip install ./pelagia_interchange
```

For development:

```bash
python -m pip install -e './pelagia_interchange[test]'
pytest -q tests/test_interchange.py
```

Installation provides the `pii` command. Confirm it is available with:

```bash
pii --help
```

Optional hash implementations can be installed when a dataset uses them:

```bash
python -m pip install './pelagia_interchange[blake3]'
python -m pip install './pelagia_interchange[xxh3]'
```

SHA-256 requires no optional package and is the archival default.

## Create your first dataset

The package does not decode video. The caller supplies already encoded image bytes and the provenance that connects each retained frame to its acquisition source. This keeps FFmpeg, PyAV, camera SDKs, and project-specific decoding outside the preservation format.

### Create automatically from a directory of videos

For the common case, point `pii create` at a directory of acquisition videos:

```bash
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --title "Deployment 042" \
  --description "Port-camera imagery from Station 12" \
  --stream port \
  --shard-target-size 10GB

pii inspect deployment_042
pii verify deployment_042 --level structural
```

The output positional argument may be omitted. In that case, `/acquisition/deployment_042` produces a sibling directory named `/acquisition/deployment_042_interchange`:

```bash
pii create \
  --from-videos /acquisition/deployment_042 \
  --title "Deployment 042" \
  --stream port
```

This workflow automatically:

- discovers common video containers in deterministic filename order;
- probes the first video stream in each file with FFprobe;
- records container, codec, pixel format, dimensions, rational frame rate, frame count, creation time when available, file size, relative source path, and source UUID;
- calculates a streaming SHA-256 hash of every source video;
- transcodes frames to standalone JPEG payloads with FFmpeg;
- numbers source frames from zero within each acquisition file and assigns continuous retained frame IDs across the stream;
- fills each SQLite shard across consecutive source videos until the configured target size is reached, then rolls over deterministically;
- creates persistent shard and stream UUIDs;
- records the FFmpeg operation and parameters in `history.jsonl`;
- generates a bounded representative preview set, contact sheet, and provenance index;
- finalizes the manifest, generated README, standalone tools, and package checksums.

FFmpeg and FFprobe must be installed and available on `PATH` for video ingestion. They are external creation tools only; recipients do not need them to inspect, extract, or verify the resulting dataset. The importer probes FFmpeg's supported options and uses either modern `-fps_mode passthrough` or the equivalent legacy `-vsync 0`; the selected frame-synchronization mode is recorded in dataset provenance.

Supported discovery extensions are `.avi`, `.mov`, `.mp4`, `.m4v`, `.mkv`, `.mpg`, `.mpeg`, `.mts`, and `.m2ts`. Add `--recursive` to search subdirectories:

```bash
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --recursive \
  --stream port \
  --title "Deployment 042"
```

Source files are hashed by default. `--no-source-hash` skips that potentially long pass, but source SHA-256 hashes are strongly recommended before an archival or raw-retirement decision.

By default, a source-video boundary does not force a shard boundary. A shard may contain frames from several source videos, while every frame retains its `source_file_id` and `source_frame_number`. This avoids producing many undersized shards. To deliberately start a new shard for every source video, use `--source-file-boundary`:

```bash
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --source-file-boundary
```

Use a prepared metadata document when the scientific description is ready:

```bash
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --metadata deployment_042.metadata.toml \
  --stream port
```

The importer preserves the supplied metadata and uses a matching stream UUID when the TOML already contains a stream with the selected name. Command-line `--title` and `--description` values override those two fields when explicitly supplied. Without `--metadata`, creation writes a valid minimal document that can be reviewed later.

The default output is color JPEG encoded by FFmpeg's MJPEG encoder at qscale 3. Use `--grayscale` for a documented grayscale transform or choose another FFmpeg qscale from 2 (highest quality/largest) through 31:

```bash
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --stream port \
  --grayscale \
  --ffmpeg-qscale 3
```

The importer uses FFmpeg's error-exit behavior and reconciles FFprobe's expected decoded-frame count with the produced frames. A decode error or mismatch stops finalization and preserves the incomplete package and `.partial` shard for review; it does not silently close a sequence gap.

### Automatic previews

Automatic creation generates previews by default. For each stream it selects up to 12 evenly spaced retained frame IDs and creates:

```text
preview/
├── index.json
└── streams/
    └── port/
        ├── contact_sheet.jpg
        ├── frame_000000000000.jpg
        ├── frame_000000120034.jpg
        └── ...
```

Thumbnails are at most 512 pixels wide by default. `preview/index.json` records the stream UUID, retained frame ID, source-file UUID, source frame number, timestamp when available, selection method, and preview settings. Every preview file is inventoried in the manifest and covered by package checksums.

Previews are explicitly non-authoritative derivatives. The SQLite frame BLOB remains authoritative, and preview failure does not invalidate otherwise successful archival data unless previews are required.

Controls:

```bash
# Change the bounded sample and thumbnail size
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --preview-count 16 \
  --preview-width 640

# Disable preview generation
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --no-previews

# Treat any preview-generation failure as a creation failure
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --require-previews
```

Selection is deterministic for a fixed retained frame count: the first and last frames are included, with the remaining samples evenly spaced. This is a browsing aid, not a claim that the selected frames are scientifically representative.

The same automatic workflow is available as a Python API:

```python
from pelagia_interchange import ingest_video_directory

result = ingest_video_directory(
    "/acquisition/deployment_042",
    "/archive/deployment_042",
    title="Deployment 042",
    description="Port-camera imagery from Station 12",
    stream="port",
    recursive=True,
    shard_target_size="10GB",
    grayscale=True,
    source_file_boundary=False,
    generate_previews=True,
    preview_count=12,
    preview_width=512,
    require_previews=False,
    progress=print,
)

print(result.source_files, result.frames, result.dataset.frame_count)
```

### Use the interactive creator

Run the guided workflow without remembering the flags:

```bash
pii create --interactive
```

The prompts collect:

- input video directory;
- output dataset directory;
- title and initial description;
- camera/stream name;
- target shard size;
- recursive discovery choice;
- color or grayscale output;
- whether each source video should force a new shard (default: no);
- whether to generate previews, plus count and maximum width (default: yes, 12, 512 px);
- whether preview failure should fail creation (default: no);
- optional existing `metadata.toml` path;
- confirmation after displaying the discovered videos.

Options can be supplied before entering interactive mode to become prompt defaults:

```bash
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --stream port \
  --shard-target-size 10GB \
  --interactive
```

### Resulting package

Automatic and interactive creation finalize the complete package structure:

```text
example_dataset/
├── manifest.json
├── metadata.toml
├── history.jsonl
├── README.md
├── checksums.sha256
├── data/
├── calibration/
├── preview/
└── tools/
```

To create an intentionally empty package for a format demonstration or metadata-only handoff, omit `--from-videos`:

```bash
pii create empty_dataset --title "Empty example"
```

An empty package is finalized immediately; it is not an initialization step for later frame ingestion.

### Create a dataset from numbered JPEG files

This complete example creates an interchange dataset from a directory containing `000000.jpg`, `000001.jpg`, and similarly numbered JPEGs. Save it as `create_dataset.py`, update the three paths, and run `python create_dataset.py`.

The numbered files should represent every source position. If a position is absent because decoding failed or a frame was removed, add an explicit status row as shown in [Represent missing, failed, and removed frames](#represent-missing-failed-and-removed-frames).

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from pelagia_interchange import DatasetBuilder, StorageFormat

OUTPUT = Path("deployment_042")
SOURCE_VIDEO = Path("/acquisition/CAM1_0042.avi")
ENCODED_FRAMES = Path("encoded_frames")

# Filenames are the source-frame numbers: 000000.jpg, 000001.jpg, ...
frame_paths = sorted(ENCODED_FRAMES.glob("*.jpg"), key=lambda path: int(path.stem))
if not frame_paths:
    raise SystemExit(f"No numbered JPEG files found in {ENCODED_FRAMES}")

# Hash the original acquisition as a stream; it is never loaded into memory.
with SOURCE_VIDEO.open("rb") as source_stream:
    source_sha256 = hashlib.file_digest(source_stream, "sha256").hexdigest()

jpeg90 = StorageFormat(
    codec="jpeg",
    quality=90,
    pixel_format="gray8",
    bit_depth=8,
    encoder="libjpeg-turbo",
    description="JPEG quality 90, 8-bit grayscale",
)

with DatasetBuilder(
    OUTPUT,
    title="Deployment 042",
    description="Retained port-camera imagery.",
    shard_target_size="10GB",
) as builder:
    builder.register_stream("port")

    source = builder.register_source_file(
        path=SOURCE_VIDEO,
        original_relative_path="camera/CAM1_0042.avi",
        sha256=source_sha256,
        container="avi",
        codec="mpeg2video",
        frame_rate=(60, 1),
        frame_count=len(frame_paths),
    )

    for frame_path in frame_paths:
        frame_number = int(frame_path.stem)
        builder.add_frame(
            stream="port",
            source_file=source,
            frame_id=frame_number,
            source_frame_number=frame_number,
            encoded_bytes=frame_path.read_bytes(),
            storage_format=jpeg90,
        )

print(f"Created {OUTPUT} with {len(frame_paths):,} frames")
```

Then inspect and verify the result:

```bash
python create_dataset.py
pii inspect deployment_042
pii verify deployment_042 --level structural
pii extract deployment_042 --frame 0 --output extracted
```

For a very large image directory, replace the in-memory `sorted(...)` list with the ordered streaming iterator produced by the acquisition/transcoding pipeline. `DatasetBuilder` itself writes in batches and does not retain all frames or frame hashes in memory.

### Add detailed timestamp and acquisition provenance

The following production-oriented example shows the additional fields available when a decoder supplies timestamp evidence and acquisition metadata:

```python
from pathlib import Path

from pelagia_interchange import DatasetBuilder, StorageFormat

output = Path("deployment_042")

jpeg90 = StorageFormat(
    codec="jpeg",
    quality=90,
    pixel_format="gray8",
    bit_depth=8,
    encoder="libjpeg-turbo",
    encoder_version="3.1.0",
    parameters={"optimize": True},
    description="JPEG quality 90, 8-bit grayscale",
)

with DatasetBuilder(
    output,
    title="Deployment 042",
    description="Retained imagery from the port camera.",
    shard_target_size="10GB",
) as builder:
    stream_uuid = builder.register_stream(
        "port",
        stream_uuid="14317b6f-4a23-47c8-ac9a-9af570033a69",
    )

    source = builder.register_source_file(
        path=Path("/acquisition/CAM1_0042.avi"),
        original_relative_path="camera/CAM1_0042.avi",
        sha256="SOURCE_FILE_SHA256_HEX",
        container="avi",
        codec="mpeg2video",
        pixel_format="gray8",
        width=2048,
        height=2048,
        frame_rate=(60, 1),
        frame_count=1_800_000,
        start_timestamp="2026-07-14T20:10:00Z",
        end_timestamp="2026-07-14T20:40:00Z",
    )

    for source_frame_number, encoded_jpeg in encoded_frame_generator():
        builder.add_frame(
            stream="port",
            source_file=source,
            frame_id=source_frame_number,
            source_frame_number=source_frame_number,
            encoded_bytes=encoded_jpeg,
            storage_format=jpeg90,
            width=2048,
            height=2048,
            timestamp_ns=1_752_524_200_000_000_000 + source_frame_number * 16_666_667,
            source_timestamp_ns=source_frame_number * 16_666_667,
            timestamp_source="video PTS",
            clock_source="acquisition-computer clock",
            timezone="UTC",
            utc_conversion="host clock recorded as UTC at acquisition",
            timestamp_precision_ns=1_000_000,
            synchronization_method="pre-deployment comparison to GNSS clock",
            known_offset_ns=12_000_000,
            known_drift_ppb=20.0,
            interpolated=False,
        )
```

The context manager finalizes all open shards, writes the manifest and checksums, records provenance, and generates the dataset README and standalone tools. While building, each shard uses a `.sqlite.partial` name. Only successfully finalized shards appear in `manifest.json`.

Do not use a placeholder such as `SOURCE_FILE_SHA256_HEX` in a real dataset. Supply the verified acquisition-file hash or omit `sha256` until it is known. Unknown values should remain unknown rather than be guessed.

PNG and custom representations use the same API:

```python
png = StorageFormat(codec="png", pixel_format="gray8", bit_depth=8)
custom = StorageFormat(
    codec="c_array_lz4",
    parameters={"array_order": "C", "compression_level": 9},
    description="C-order array compressed with LZ4",
)
```

### Feeding frames from another decoder

A decoder integration only needs to yield encoded image bytes and reliable provenance. A minimal adapter can look like this:

```python
def encoded_frame_generator():
    for image_path in sorted(Path("encoded_frames").glob("*.jpg")):
        frame_number = int(image_path.stem)
        yield frame_number, image_path.read_bytes()
```

For very large datasets, the generator should read one encoded image at a time. `DatasetBuilder` batches SQLite writes and does not retain all frames or hashes in memory.

## Represent missing, failed, and removed frames

Sequence gaps must be explicit. A failed decode is a row with the original source position and no invented image payload:

```python
from pelagia_interchange import FrameStatus

builder.add_frame(
    stream="port",
    source_file=source,
    frame_id=120_034,
    source_frame_number=120_034,
    encoded_bytes=None,
    storage_format=None,
    status=FrameStatus.DECODE_FAILED,
    timestamp_ns=1_752_526_200_566_666_667,
    timestamp_source="video PTS",
)
```

Use the status that describes the evidence:

- `missing`: the source position is expected but no frame bytes are available.
- `decode_failed`: a decode was attempted and failed.
- `duplicate`: the source position is represented but identified as a duplicate.
- `intentionally_removed`: a documented policy or operator action removed the payload.
- `timestamp_invalid`: image bytes may be retained, but the timestamp is not valid.
- `corrupt`: bytes may be retained for diagnosis, but they are known to be corrupt.

Do not renumber later frames to close a gap. Full and archival verification report unrepresented frame and source-frame gaps.

## Choose shard boundaries

The default target is 10 GB. A shard never splits an individual encoded frame.

```python
builder = DatasetBuilder(
    "deployment_042",
    shard_target_size="10GB",
    maximum_frame_count=500_000,
    source_file_boundary=True,
)
```

To close the current shard at a known acquisition or operational boundary:

```python
builder.boundary("port")
```

Shard filenames are deterministic, such as `port_000001.sqlite`, but the shard UUID in the manifest is its persistent identity.

Automatic video-directory ingestion uses `source_file_boundary=False` by default, so consecutive source videos continue filling the current shard when space remains. `source_file_boundary=True` and `--source-file-boundary` are explicit source-aligned modes.

## Add scientific metadata

Pass a `Metadata` object when creating the dataset:

```python
from pelagia_interchange import DatasetBuilder, Metadata

metadata = Metadata(
    {
        "schema": {
            "name": "scientific-image-interchange-metadata",
            "version": "1",
        },
        "dataset": {
            "title": "Deployment 042",
            "description": "Port-camera imagery from Station 12.",
            "license": "CC-BY-4.0",
        },
        "collection": {
            "start": "2026-07-14T20:10:00Z",
            "end": "2026-07-14T20:40:00Z",
            "geographic_region": "Gulf of Alaska",
        },
        "investigators": [
            {
                "name": "Example Investigator",
                "role": "principal investigator",
                "institution": "Example Ocean Institute",
            }
        ],
        "instruments": [],
        "deployments": [],
        "streams": [],
        "funding": [],
        "extensions": {
            "example_org": {"campaign_code": "GOA26"},
        },
    }
)

with DatasetBuilder("deployment_042", metadata=metadata) as builder:
    # Register source files and stream encoded frames as shown above.
    ...
```

Use stable UUID fields for deployments, instruments, and streams. Human-readable names are labels, not persistent identifiers. Organization-specific fields belong below `extensions`; the reader preserves unknown extension content.

To edit metadata after finalization:

```python
from pelagia_interchange import Dataset

dataset = Dataset.open("deployment_042")
dataset.metadata.data["dataset"]["description"] = "Updated after cruise review."
dataset.save_metadata(
    operator="analyst@example.org",
    message="Clarified the collection description after cruise review.",
)
```

`save_metadata()` writes the TOML, appends a `metadata_modified` history event, marks the package `modified`, and regenerates package checksums. Editing `metadata.toml` manually is allowed, but checksums and processing history then need to be updated deliberately.

## Add calibration and preview resources

Resources are copied into the package and inventoried with hashes:

```python
with DatasetBuilder("deployment_042", title="Deployment 042") as builder:
    builder.add_resource(
        "working/port_lens_calibration.json",
        kind="calibration",
        relative_name="port/lens_distortion.json",
        description="Brown-Conrady lens calibration derived 2026-06-30.",
    )
    builder.add_resource(
        "working/contact_sheet.jpg",
        kind="preview",
        relative_name="contact_sheet.jpg",
        description="Representative non-authoritative frames.",
    )
    # Register sources and add frames here.
```

Calibration files remain ordinary resources; the interchange format does not impose a specialized calibration encoding. Preview files are informational derivatives and are never authoritative frame data.

## Open and inspect a dataset in Python

```python
from pelagia_interchange import Dataset

dataset = Dataset.open("/archive/deployment_042")

print(dataset.manifest.dataset_uuid)
print(dataset.metadata.title)
print(dataset.frame_count)
print(dataset.encoded_bytes)
print(dataset.summary())
```

`summary()` uses database aggregates and does not read all image BLOBs.

Iterate without loading the dataset into memory:

```python
for frame in dataset.iter_frames(
    camera="port",
    frame_start=120_000,
    frame_end=121_000,
):
    record = frame.record
    print(record.frame_id, record.source_frame_number, record.status)
```

Ranges are inclusive. Filters are available for stream/camera, retained frame ID, source file ID or UUID, shard, and nanosecond timestamp.

## Extract ordinary image files

Extract one frame without decoding or re-encoding:

```python
frame = dataset.get_frame(camera="port", frame_number=120_034)
frame.save("frame_120034.jpg")
```

Stream a range with the library:

```python
from pelagia_interchange.extraction import extract_frames

result = extract_frames(
    dataset,
    "extracted/port",
    camera="port",
    frame_start=120_000,
    frame_end=121_000,
    overwrite="skip",
)

print(result.written, result.skipped, result.bytes_written)
```

Equivalent CLI examples:

```bash
# All retained payloads
pii extract deployment_042 --output extracted/all

# One frame
pii extract deployment_042 --camera port --frame 120034 --output extracted/one

# Inclusive frame range
pii extract deployment_042 --camera port --frames 120000:121000 --output extracted/range

# Open-ended ranges
pii extract deployment_042 --frames 120000: --output extracted/from_120000
pii extract deployment_042 --frames :121000 --output extracted/through_121000

# Source-file and timestamp filters
pii extract deployment_042 --source-file 3 --output extracted/source_3
pii extract deployment_042 --source-uuid 93ef622a-75cf-44f4-a0db-d6f17fadbb18 --output extracted/source_uuid
pii extract deployment_042 --shard port_000042.sqlite --output extracted/shard_42
pii extract deployment_042 --timestamps 1752524200000000000:1752524260000000000 --output extracted/minute

# Preview selection without writing files
pii extract deployment_042 --camera port --frames 120000:121000 --dry-run

# Avoid collisions by including source identity in filenames
pii extract deployment_042 --source-frame-names --overwrite skip --output extracted/by_source
```

JPEG payloads use `.jpg`, PNG payloads use `.png`, and unknown representations use `.bin`. Null-payload status rows are skipped and counted; the extractor does not manufacture files for missing frames.

## Inspect package contents

```bash
pii inspect deployment_042
pii inspect deployment_042 --json
pii sources deployment_042 --json
pii shards deployment_042 --json
pii metadata deployment_042 --json
pii history deployment_042 --json
```

JSON output is suitable for scripts. For example:

```bash
pii inspect deployment_042 --json > deployment_042-summary.json
```

You can also inspect a shard with ordinary SQLite:

```bash
sqlite3 deployment_042/data/port_000001.sqlite '.tables'
sqlite3 deployment_042/data/port_000001.sqlite \
  'SELECT status, count(*) FROM frames GROUP BY status;'
sqlite3 deployment_042/data/port_000001.sqlite \
  'SELECT codec, quality, pixel_format, description FROM storage_formats;'
```

Do not edit a finalized shard with SQLite. Copying and querying it read-only is safe; authoritative changes should create a replacement shard with repair provenance.

## Verify integrity

Choose a level appropriate to the decision:

```bash
pii verify deployment_042 --level quick
pii verify deployment_042 --level structural
pii verify deployment_042 --level full
pii verify deployment_042 --level archival --image-signatures
```

- `quick` verifies required files, manifest compatibility, package checksums, and shard sizes/hashes.
- `structural` also opens every shard, runs SQLite integrity checks, validates tables, and reconciles counts, ranges, sources, and required metadata.
- `full` also streams all frame records and validates BLOB sizes and available hashes while detecting unrepresented gaps.
- `archival` adds the strict pre-retirement conditions: expected source counts, complete verifiable BLOB hashes, no partial shards, and successful creation/finalization provenance.

Machine-readable output and exit status are suitable for automation:

```bash
if pii verify deployment_042 --level archival --json > archival-verification.json; then
    echo "Mechanical archival verification passed"
else
    echo "Verification failed; retain source acquisitions and review the report" >&2
    exit 1
fi
```

In Python:

```python
from pelagia_interchange import Validator

result = Validator("deployment_042").verify(
    "archival",
    image_signatures=True,
)

if not result.archival_ready:
    for issue in result.errors:
        print(issue.code, issue.message, issue.path)
```

An `ARCHIVAL READY` result is a high-confidence mechanical validation result, not permission to delete raw data. Confirm the project's retention policy, legal or funder requirements, backup state, independent copy, scientific acceptance criteria, and responsible-person approval separately. Neither the library nor the CLI deletes source acquisitions.

## Record additional provenance

History is append-only through the public API:

```python
dataset.history.append(
    operation="dataset_exported",
    operator="analyst@example.org",
    parameters={"destination": "institutional archive"},
    inputs=[
        {
            "identifier": dataset.manifest.dataset_uuid,
            "identifier_type": "dataset_uuid",
        }
    ],
    outputs=[
        {
            "identifier": "ACCESSION-2026-042",
            "identifier_type": "accession",
        }
    ],
    status="success",
    message="Transferred to the institutional archive.",
)
dataset.regenerate_checksums()
```

Appending history changes a checksummed file. Call `regenerate_checksums()` after a direct history append. `save_metadata()` performs this step automatically.

## Recover from an interrupted build

An interrupted build may leave `.sqlite.partial` files. They are not listed as authoritative shards and are never deleted automatically.

```bash
pii shards deployment_042 --partials
pii shards deployment_042 --quarantine-partials ./partial-recovery
```

The quarantine command moves each partial file intact to the requested directory. It fails rather than overwriting a file with the same name. Review partial databases independently before deciding whether to resume, salvage, or remove them.

In Python:

```python
from pelagia_interchange import DatasetBuilder

partials = DatasetBuilder.partials("deployment_042")
moved = DatasetBuilder.quarantine_partials(
    "deployment_042",
    "partial-recovery",
)
```

To inspect an incomplete package without treating it as complete:

```python
dataset = Dataset.open("deployment_042", allow_incomplete=True)
print(dataset.manifest.state)
```

## Use a dataset without installing the package

Every completed dataset contains standard-library scripts under `tools/`:

```bash
python deployment_042/tools/inspect.py deployment_042
python deployment_042/tools/inspect.py deployment_042 --json
python deployment_042/tools/extract.py deployment_042 --camera port --frames 1000:2000 --output extracted
python deployment_042/tools/verify.py deployment_042 --level full
```

The standalone extractor can also operate directly on one shard:

```bash
python deployment_042/tools/extract.py \
  deployment_042/data/port_000042.sqlite \
  --all \
  --output extracted/shard_42
```

These scripts require only Python 3.11 or newer. A recipient can also use any ordinary SQLite installation to query shard metadata and extract BLOBs.

## Relocate or copy a dataset

Package resources use relative paths, so the whole dataset directory can be moved or copied as a unit:

```bash
rsync -a --info=progress2 deployment_042/ /archive/deployment_042/
pii verify /archive/deployment_042 --level quick
```

Original absolute acquisition paths may remain in provenance, but readers never rely on them to locate package resources. After transport, run at least quick verification; use full or archival verification when the transfer supports a preservation or source-retirement decision.

## Common mistakes

- Do not decode and re-encode during ordinary extraction; copy the stored BLOB exactly.
- Do not omit rows for failed or removed source positions.
- Do not treat a shard filename or camera name as persistent identity; use UUIDs.
- Do not claim nanosecond accuracy merely because timestamps are stored as integer nanoseconds.
- Do not edit finalized SQLite shards in place.
- Do not append history or manually edit metadata without regenerating package checksums.
- Do not treat preview resources as authoritative images.
- Do not delete source acquisitions solely because a command returned zero.

## Benchmark a host

The repository includes a synthetic benchmark that measures sequential insertion, shard/package finalization, sequential reads and extraction, random reads, and full verification:

```bash
python benchmarks/interchange_benchmark.py --frames 100000 --bytes 100000
```

Use representative payload sizes and storage hardware before selecting production shard targets. Benchmark results are host- and filesystem-specific; they are not format guarantees.
