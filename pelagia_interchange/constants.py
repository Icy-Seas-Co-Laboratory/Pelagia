"""Version and format constants."""

LIBRARY_VERSION = "0.2.0"
FORMAT_NAME = "Scientific Image Interchange"
FORMAT_ID = "scientific-image-interchange"
# 0.x intentionally permits breaking revisions.  Format 0.2 changes the
# authoritative lineage from imported source files to acquisition segments.
FORMAT_VERSION = "0.2"
SCHEMA_VERSION = "2"

REQUIRED_PACKAGE_FILES = (
    "manifest.json",
    "metadata.toml",
    "history.jsonl",
    "README.md",
    "checksums.sha256",
)
