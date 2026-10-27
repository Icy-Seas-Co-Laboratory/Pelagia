# SkerryKV Storage

Pelagia uses two logical SkerryKV roles. Cold storage contains source,
preprocessed, and background frame payloads. Hot storage contains candidate and
refined ROI images and masks. Configure `[cold_storage]` and `[hot_storage]`
independently; both normally use `http://127.0.0.1:7788` in development.

Tokens belong in environment variables, not checked-in TOML files:

```bash
export PELAGIA_COLD_STORAGE_TOKEN="$SKERRY_TOKEN"
export PELAGIA_HOT_STORAGE_TOKEN="$SKERRY_TOKEN"
```

`scripts/pelagia_dev_stack.sh` starts a shared loopback SkerryKV service by
default. It creates data in `data/skerry-cold`; point hot storage at a different
endpoint and set `SKERRY_HOT_ROOT` to start a separate local instance. The TOML
worker stack reads the same endpoint and token fields and starts its shared
instance from `[skerry]`; set `managed = true` for externally operated stores.

SkerryKV objects are immutable and content addressed. PostgreSQL is the source
of project ownership and object references, so do not use a hash obtained from
another project to bypass Pelagia authorization. SkerryKV does not currently
provide garbage collection, deletion, or replication. Plan capacity for
unreferenced objects produced by failed application transactions and back up
each store alongside PostgreSQL.

For a maintenance-window backup, stop Pelagia, snapshot PostgreSQL and each
SkerryKV root, then run `skerry verify --path <copied-root>` before accepting
the copy. Restore both the database and matching object-store snapshots before
restarting Pelagia.
