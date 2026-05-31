<p align="center">
  <img src="docs/assets/pelagia_logo.jpeg" alt="Pelagia" width="520">
</p>

# Pelagia

Pelagia is an image-processing, storage, and worker system for video-frame extraction, ROI segmentation, and downstream analysis. Large full-frame image payloads are stored in a cold `KVStore`, while smaller ROI payloads and masks are stored directly with Postgres metadata for easier analysis.

<p align="center">
  <img src="docs/assets/pelagia_icon.png" alt="Pelagia icon" width="96">
</p>

## Project Layout

Pelagia is organized so the API, CLI, and workers share the same core services instead of each growing separate business logic.

```text
Pelagia/
  config.py              shared typed configuration
  domain.py              stable domain records and enums
  storage/               persistence adapters for Postgres and KVStore
  processing/            data-in/data-out processing routines
  services/              application operations shared by all interfaces
  workers/               job claiming, dispatch, heartbeat, and execution
  api/                   FastAPI app and route modules
  cli/                   command-line entrypoint and commands
  utils/                 small shared helpers
docs/assets/             documentation images and branding assets
```

The intended dependency direction is:

```text
API / CLI / Workers -> Services -> Storage + Processing
```

Keep FastAPI routes, CLI commands, and worker loops thin. When logic is useful in more than one interface, put it in `services/`. When logic transforms data without caring who called it, put it in `processing/`.

## Configuration

Pelagia loads configuration in this order:

```text
Pelagia/default.config.toml < ./config.toml < environment variables < explicit CLI options
```

The packaged [default.config.toml](Pelagia/default.config.toml) contains development-friendly defaults. Create a local `config.toml` in the repository root for machine-specific overrides; it is ignored by git.

## Basic Usage

### KVStore

`KVStore` is a content-addressed cold-storage pool for large frame payloads. Payloads are stored as bytes in SQLite databases, and each payload's hash digest is its canonical key. The key prefix determines the shard directory, and each shard can rotate through multiple SQLite files as size or row-count limits are reached.

```python
from Pelagia.storage.kvstore import KVStore

store = KVStore("./my_store")
store.initialize(
    hash_algorithm="sha256",
    prefix_length=2,
    max_db_bytes=4 * 1024 * 1024 * 1024,
    max_rows=1_000_000,
)

payload = b"hello world"
key = store.put_store(payload)

assert store.key_exists(key)
assert store.get_store(key) == payload

print(store.status())
print(store.check_health())
```

## Layout

With `prefix_length=2`, a key beginning with `aa` is routed to `root/aa/`:

```text
root/
  config.json
  manifest.json
  00/
    store_000001.sqlite
  aa/
    store_000001.sqlite
    store_000002.sqlite
```

Prefix length controls shard fanout:

- `1` creates 16 directories.
- `2` creates 256 directories.
- `3` creates 4096 directories.
- `4` creates 65536 directories and may be excessive for many filesystems.

## Rotation

Each prefix directory starts with `store_000001.sqlite`. Before a new blob is inserted, the store checks the highest-numbered SQLite file for that prefix. If the file is at or above `max_db_bytes` or `max_rows`, the store creates the next file, such as `store_000002.sqlite`, and writes the new blob there.

Rotation is per prefix, never global. Existing blobs are not moved during rotation, so reads search all SQLite files for the relevant prefix, newest first.

## Hash Algorithms

`sha256` uses Python's standard `hashlib.sha256`. `blake3` is supported when the third-party `blake3` package is installed. If `blake3` is requested without that dependency, `KVStoreConfigError` is raised with a clear message.

## Concurrency

The current implementation uses one SQLite connection per operation and enables WAL mode with `synchronous=NORMAL`. Writes and rotation are guarded by a small standard-library file lock per prefix directory, so unrelated prefixes can still progress independently. Initialization also takes a root-level lock.

This lock is intentionally simple. It is suitable for local multi-process usage on a normal filesystem, but it is not a distributed lock manager. For network filesystems or clustered workers, add an external coordinator or move locking into the catalog database.

Root paths are normalized with `pathlib` after expanding `~` and environment variables, so paths such as `~/pelagia-store`, `$PELAGIA_STORE_ROOT/data`, and relative paths work consistently across supported operating systems.
