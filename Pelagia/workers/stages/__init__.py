"""Stage-local worker entry points.

Each module is intentionally small at first: it is the stable boundary for a
stage while the legacy implementations are migrated without changing leases,
progress events, or result payloads.
"""

