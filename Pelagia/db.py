"""Compatibility wrapper for PostgreSQL storage.

New code should import from ``Pelagia.storage.postgres``.
"""

from .storage.postgres import PostgresRepository, render_schema

__all__ = ["PostgresRepository", "render_schema"]
