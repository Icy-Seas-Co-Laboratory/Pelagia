from __future__ import annotations

import re


SCHEMA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_schema_name(schema: str) -> str:
    """Validate a PostgreSQL schema identifier used in generated SQL."""
    if not isinstance(schema, str) or not SCHEMA_NAME_RE.match(schema):
        raise ValueError(
            "schema name must start with a letter or underscore and contain only letters, numbers, and underscores"
        )
    return schema
