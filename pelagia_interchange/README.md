# pelagia-interchange

`pelagia-interchange` is a Python 3.11+ standard-library package for creating, reading, extracting, and validating portable archival scientific-image datasets. Retained encoded images are stored byte-for-byte in immutable, standalone SQLite shards; JSON, TOML, JSON Lines, and SHA-256 files provide inventory, scientific metadata, provenance, and integrity.

## Install

From the Pelagia repository:

```bash
python -m pip install ./pelagia_interchange
pii --help
```

The package has no required third-party runtime dependencies. Optional BLAKE3 and XXH3 implementations are available through the `blake3` and `xxh3` extras; SHA-256 works with the standard library and remains the archival default.

## Create your first dataset

Create a complete dataset from a directory of videos:

```bash
pii create deployment_042 \
  --from-videos /acquisition/deployment_042 \
  --title "Deployment 042" \
  --stream port \
  --shard-target-size 10GB

pii inspect deployment_042
pii verify deployment_042 --level structural
```

FFprobe discovers source properties and frame counts; source files are SHA-256 hashed; FFmpeg streams JPEG frames into immutable SQLite shards; and the builder records provenance, generates standalone tools, previews, and package checksums. FFmpeg and FFprobe are only needed for creation. Consecutive videos fill the current shard by default; use `--source-file-boundary` only when every video should start a new shard. Other controls include `--recursive`, `--grayscale`, `--ffmpeg-qscale`, and `--no-source-hash`.

Creation also produces up to 12 evenly spaced 512-pixel thumbnails, a contact sheet, and `preview/index.json` mapping derivatives to authoritative frame/source identities. Control this with `--preview-count`, `--preview-width`, `--no-previews`, and `--require-previews`. Previews are checksummed but explicitly non-authoritative.

For a guided workflow:

```bash
pii create --interactive
```

Interactive mode prompts for the video directory, output, title, description, stream name, shard size, recursive discovery, color/grayscale choice, optional metadata TOML, and final confirmation.

The automatic workflow is also available in Python:

```python
from pelagia_interchange import ingest_video_directory

result = ingest_video_directory(
    "/acquisition/deployment_042",
    "/archive/deployment_042",
    title="Deployment 042",
    stream="port",
    shard_target_size="10GB",
    source_file_boundary=False,
    generate_previews=True,
    preview_count=12,
    preview_width=512,
    progress=print,
)
print(result.frames)
```

To create an intentionally empty finalized package, omit `--from-videos`:

```bash
pii create empty_dataset --title "Empty example"
```

The following complete example packages numbered JPEG files such as `000000.jpg` and `000001.jpg`. The caller supplies encoded image bytes because video decoding intentionally remains outside this package.

```python
import hashlib
from pathlib import Path

from pelagia_interchange import DatasetBuilder, StorageFormat

output = Path("deployment_042")
source_video = Path("/acquisition/CAM1_0042.avi")
frame_paths = sorted(
    Path("encoded_frames").glob("*.jpg"),
    key=lambda path: int(path.stem),
)
if not frame_paths:
    raise SystemExit("No numbered JPEG files found")

with source_video.open("rb") as stream:
    source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

jpeg90 = StorageFormat(
    codec="jpeg",
    quality=90,
    pixel_format="gray8",
    encoder="libjpeg-turbo",
)

with DatasetBuilder(
    output,
    title="Deployment 042",
    shard_target_size="10GB",
) as builder:
    source = builder.register_source_file(
        path=source_video,
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
```

The context manager finalizes shards, generates package documentation and standalone tools, writes checksums, and only then marks the dataset complete.

Use an explicit null-payload record for a missing or failed frame:

```python
from pelagia_interchange import FrameStatus

builder.add_frame(
    stream="port",
    source_file=source,
    frame_id=120034,
    source_frame_number=120034,
    encoded_bytes=None,
    storage_format=None,
    status=FrameStatus.DECODE_FAILED,
)
```

Never renumber later frames to hide a source-sequence gap.

## Read and extract

```python
from pelagia_interchange import Dataset

dataset = Dataset.open("/data/cruise_2026")
print(dataset.metadata.title)
print(dataset.frame_count)

frame = dataset.get_frame(camera="port", frame_number=120034)
frame.save("frame.jpg")
```

Extraction copies the stored encoded bytes directly; it does not decode or re-encode the image.

## Command-line examples

The `pii` command offers `create`, `inspect`, `extract`, `verify`, `metadata`, `history`, `shards`, and `sources`:

```bash
pii create deployment_042 --from-videos /acquisition/deployment_042 --stream port
pii create deployment_042 --from-videos /acquisition/deployment_042 --stream port --resume
pii create --interactive
pii inspect deployment_042
pii inspect deployment_042 --json
pii extract deployment_042 --camera port --frames 1000:2000 --output extracted
pii extract deployment_042 --frame 12345 --output extracted/one
pii verify deployment_042 --level quick
pii verify deployment_042 --level full --image-signatures
pii sources deployment_042 --json
pii shards deployment_042 --json
```

Frame and timestamp ranges are inclusive. Use `--dry-run` to preview extraction and `--overwrite error|skip|replace` to choose collision behavior.

## Archival verification

```bash
pii verify deployment_042 --level archival --json > archival-verification.json
```

Archival verification checks package and shard integrity, all available frame hashes, source/frame reconciliation, explicit gaps, partial shards, and successful finalization provenance. An `ARCHIVAL READY` result is a high-confidence mechanical check, not authorization to delete raw acquisitions. Confirm retention requirements, backups, scientific acceptance, and responsible-person approval separately. The package never deletes source files.

## Standalone access and recovery

Every finalized dataset contains standard-library scripts under `tools/`:

```bash
python deployment_042/tools/inspect.py deployment_042
python deployment_042/tools/extract.py deployment_042 --all --output extracted
python deployment_042/tools/verify.py deployment_042 --level full
```

Interrupted `.sqlite.partial` shards remain non-authoritative and are preserved. List or quarantine them with:

```bash
pii shards deployment_042 --partials
pii shards deployment_042 --quarantine-partials ./partial-recovery
```

An interrupted video ingestion can be resumed in place with `pii create --from-videos ... --resume`. The source files are checked against the incomplete package, finalized shards remain immutable, and only the durable source-frame prefix is skipped/replayed.

The detailed [usage guide](../docs/interchange-usage.md), [normative specification](../docs/interchange-specification.md), and [annotated metadata example](../docs/metadata.example.toml) are in the repository. Every generated dataset also includes a durable README and dependency-free scripts under `tools/`.
