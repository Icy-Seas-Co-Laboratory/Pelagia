# Standalone tools

These Python 3.11+ scripts use only the standard library and do not require Pelagia or `pelagia_interchange` to be installed.

- `inspect.py DATASET [--json]` summarizes the package.
- `extract.py DATASET [filters] --output DIR` copies encoded image BLOBs without re-encoding. It also accepts one SQLite shard directly.
- `verify.py DATASET --level quick|structural|full|archival [--json]` checks integrity and exits zero only when valid.

Run each script with `--help` for all options. Dataset contents are treated as untrusted; relative paths are confined to the package and generated extraction names cannot escape the output directory.

Examples:

```bash
python tools/inspect.py . --json
python tools/extract.py . --frame 12345 --output extracted/one
python tools/extract.py . --camera port --frames 1000:2000 --output extracted/range
python tools/extract.py data/port_000042.sqlite --all --output extracted/shard_42
python tools/verify.py . --level structural
python tools/verify.py . --level archival --json
```

Frame and timestamp ranges are inclusive. Extraction copies stored BLOB bytes without image decoding or re-encoding; null-payload status records are skipped. An archival verification success is a mechanical integrity result, not authorization to delete original acquisition files. Confirm retention requirements, backups, scientific acceptance criteria, and responsible-person approval separately.
