"""Safe UUID-rooted bundle writing and verification primitives."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from collections.abc import Callable
from typing import Any
from uuid import UUID

from .contracts import EXPORT_BUNDLE_FORMAT_VERSION


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n"


def _relative_path(value: str | Path) -> PurePosixPath:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Bundle paths must be non-empty relative paths.")
    return path


class ExportBundleWriter:
    """Construct an export in a temporary UUID root and publish an atomic ZIP."""

    def __init__(self, export_id: str, destination: Path):
        self.export_id = str(UUID(str(export_id)))
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = Path(tempfile.mkdtemp(prefix=f"pelagia-export-{self.export_id}-", dir=self.destination.parent))
        self.root = self._tmp / self.export_id
        self.root.mkdir()
        self._entries: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "ExportBundleWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self.cleanup()

    def write_bytes(self, relative_path: str | Path, data: bytes) -> Path:
        relative = _relative_path(relative_path)
        target = self.root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self._entries[str(relative)] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        return target

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        return self.write_bytes(relative_path, _json_bytes(value))

    def register_file(self, relative_path: str | Path) -> dict[str, Any]:
        """Register a product-writer file created directly beneath ``root``.

        Product writers may stream large media files themselves.  Registration
        preserves the same manifest/checksum guarantee as ``write_bytes``.
        """
        relative = _relative_path(relative_path)
        target = self.root.joinpath(*relative.parts)
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"Export product did not create a regular file: {relative}")
        entry = {"sha256": _sha256_file(target), "size_bytes": target.stat().st_size}
        self._entries[str(relative)] = entry
        return entry

    def register_tree(
        self, relative_path: str | Path = "products", *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Register every regular file under a product subtree deterministically."""
        relative = _relative_path(relative_path)
        subtree = self.root.joinpath(*relative.parts)
        if not subtree.exists():
            return {}
        if subtree.is_symlink() or not subtree.is_dir():
            raise ValueError(f"Export subtree is not a directory: {relative}")
        registered: dict[str, dict[str, Any]] = {}
        paths = sorted(subtree.rglob("*"))
        for target in paths:
            if target.is_symlink():
                raise ValueError("Export bundles must not contain symbolic links.")
        files = [target for target in paths if target.is_file()]
        for ordinal, target in enumerate(files, 1):
            item = self.register_file(target.relative_to(self.root))
            registered[target.relative_to(self.root).as_posix()] = item
            if progress_callback is not None and (ordinal % 100 == 0 or ordinal == len(files)):
                progress_callback(ordinal, len(files))
        return registered

    def finalize(
        self, *, project_id: str, request: dict[str, Any], products: dict[str, Any],
        versions: dict[str, Any], readme: str, data_dictionary: dict[str, Any] | None = None,
        log_entries: list[dict[str, Any]] | None = None, snapshot_at: datetime | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Write common records, verify, and atomically publish the ZIP."""
        timestamp = snapshot_at or datetime.now(UTC)
        # Product writers can write large files directly.  Scan once here so a
        # forgotten explicit registration cannot publish an unverified file.
        if progress_callback is not None:
            progress_callback("checksumming_product_files", 0, 0)
        self.register_tree(
            progress_callback=(
                None if progress_callback is None
                else lambda completed, total: progress_callback("checksumming_product_files", completed, total)
            ),
        )
        manifest = {
            "bundle_format_version": EXPORT_BUNDLE_FORMAT_VERSION,
            "export_id": self.export_id,
            "project_id": str(UUID(str(project_id))),
            "snapshot_at": timestamp.isoformat(),
            "request": request,
            "products": products,
            "files": self._entries,
        }
        self.write_json("versions.json", {"export_bundle_format": EXPORT_BUNDLE_FORMAT_VERSION, **versions})
        self.write_bytes("README.md", readme.encode("utf-8"))
        if data_dictionary is not None:
            self.write_json("data-dictionary.json", data_dictionary)
        lines = "".join(json.dumps(entry, sort_keys=True, default=str) + "\n" for entry in (log_entries or []))
        self.write_bytes("export.log", lines.encode("utf-8"))
        # Manifest deliberately lists common files too, except itself/checksum file.
        manifest["files"] = dict(self._entries)
        # A manifest cannot checksum itself without a recursive contract.  It
        # and checksums.sha256 are structural records; every payload/common
        # content file is represented in ``manifest.files``.
        (self.root / "manifest.json").write_bytes(_json_bytes(manifest))
        checksums = "".join(f"{entry['sha256']}  {path}\n" for path, entry in sorted(self._entries.items()))
        self.write_bytes("checksums.sha256", checksums.encode("utf-8"))
        archive_tmp = self._tmp / "bundle.zip"
        archive_paths = [path for path in sorted(self.root.rglob("*")) if path.is_file()]
        if progress_callback is not None:
            progress_callback("packaging_archive", 0, len(archive_paths))
        with zipfile.ZipFile(archive_tmp, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
            for ordinal, path in enumerate(archive_paths, 1):
                archive.write(path, path.relative_to(self._tmp).as_posix())
                if progress_callback is not None and (ordinal % 100 == 0 or ordinal == len(archive_paths)):
                    progress_callback("packaging_archive", ordinal, len(archive_paths))
        if progress_callback is not None:
            progress_callback("verifying_archive", 0, 0)
        verification = verify_bundle(archive_tmp)
        if progress_callback is not None:
            progress_callback("publishing_archive", 0, 0)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        archive_tmp.replace(self.destination)
        size = self.destination.stat().st_size
        result = {"path": str(self.destination), "sha256": _sha256_file(self.destination), "size_bytes": size,
                  "manifest": manifest, "snapshot_at": timestamp, "verification": verification}
        shutil.rmtree(self._tmp, ignore_errors=True)
        return result

    def cleanup(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(path: Path) -> dict[str, Any]:
    """Reject unsafe ZIP paths and mismatched files/checksums before publication/import."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names:
            raise ValueError("Export archive is empty.")
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            raise ValueError("Export archive must have exactly one root directory.")
        root = next(iter(roots))
        UUID(root)
        for name in names:
            normalized = PurePosixPath(name)
            if normalized.is_absolute() or ".." in normalized.parts:
                raise ValueError(f"Unsafe ZIP path {name!r}.")
        manifest_name = f"{root}/manifest.json"
        if manifest_name not in names:
            raise ValueError("Export archive is missing manifest.json.")
        manifest = json.loads(archive.read(manifest_name))
        if str(manifest.get("export_id")) != root:
            raise ValueError("Manifest export_id does not match archive root.")
        files = manifest.get("files") or {}
        for relative, expected in files.items():
            relative_path = _relative_path(relative)
            member = f"{root}/{relative_path.as_posix()}"
            if member not in names:
                raise ValueError(f"Manifest file is missing from archive: {relative}")
            data = archive.read(member)
            if hashlib.sha256(data).hexdigest() != expected.get("sha256"):
                raise ValueError(f"Checksum mismatch for {relative}")
        checksum_member = f"{root}/checksums.sha256"
        if checksum_member not in names:
            raise ValueError("Export archive is missing checksums.sha256.")
        expected_checksums = "".join(
            f"{entry['sha256']}  {relative}\n" for relative, entry in sorted(files.items())
        ).encode("utf-8")
        if archive.read(checksum_member) != expected_checksums:
            raise ValueError("checksums.sha256 does not match the manifest.")
        return {"export_id": root, "file_count": len(files), "valid": True}
