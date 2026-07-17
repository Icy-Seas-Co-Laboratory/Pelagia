from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_PYTHON = "3.12"
RUNTIME_EXTRAS = ("api", "cli", "postgres", "kvstore-blake3")


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    name: str
    venv_name: str
    extras: tuple[str, ...]
    requires_tensorflow: bool = False


SYNC_PROFILES = {
    "cpu": EnvironmentProfile("cpu", ".venv", (*RUNTIME_EXTRAS, "worker-cpu")),
    "dev": EnvironmentProfile("dev", ".venv", (*RUNTIME_EXTRAS, "worker-cpu", "test")),
    "ml-metal": EnvironmentProfile("ml-metal", ".venv-ml", (*RUNTIME_EXTRAS, "ml-apple-metal"), True),
    "ml-cuda": EnvironmentProfile("ml-cuda", ".venv-ml", (*RUNTIME_EXTRAS, "ml"), True),
}
DOCTOR_PROFILES = {
    "cpu": (".venv", False),
    "gpu-ml": (".venv-ml", True),
}


def resolve_root(root: Path | None = None) -> Path:
    return (root or Path.cwd()).expanduser().resolve()


def profile_venv_path(profile: str, *, root: Path | None = None) -> Path:
    resolved_root = resolve_root(root)
    normalized = profile.strip().lower().replace("_", "-")
    if normalized in {"cpu", "default", "dev"}:
        return resolved_root / ".venv"
    if normalized in {"gpu-ml", "ml-metal", "ml-cuda"}:
        return resolved_root / ".venv-ml"
    valid = ", ".join(sorted({*SYNC_PROFILES, *DOCTOR_PROFILES}))
    raise ValueError(f"Unknown environment profile {profile!r}. Valid profiles: {valid}.")


def profile_python(profile: str, *, root: Path | None = None) -> Path:
    return profile_venv_path(profile, root=root) / "bin" / "python"


def sync_profile(
    profile_name: str,
    *,
    root: Path | None = None,
    python: str | Path | None = None,
    uv: Path | None = None,
    imagecodecs_wheel: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized = profile_name.strip().lower().replace("_", "-")
    profile = SYNC_PROFILES.get(normalized)
    if profile is None:
        valid = ", ".join(sorted(SYNC_PROFILES))
        raise ValueError(f"sync profile must be one of: {valid}.")

    resolved_root = resolve_root(root)
    if not (resolved_root / "pyproject.toml").is_file():
        raise ValueError(f"Pelagia pyproject.toml was not found under: {resolved_root}")
    uv_executable = _resolve_uv(uv)
    python_request = str(python or PROFILE_PYTHON)
    if isinstance(python, Path):
        python_request = str(python.expanduser().resolve())

    venv_path = resolved_root / profile.venv_name
    if imagecodecs_wheel is not None:
        imagecodecs_wheel = imagecodecs_wheel.expanduser().resolve()
        if not imagecodecs_wheel.is_file():
            raise ValueError(f"imagecodecs wheel was not found: {imagecodecs_wheel}")

    commands = [[
        str(uv_executable),
        "sync",
        "--locked",
        "--python",
        python_request,
        "--no-dev",
        *(item for extra in profile.extras for item in ("--extra", extra)),
    ]]
    if imagecodecs_wheel is not None:
        commands.append(
            [
                str(uv_executable),
                "pip",
                "install",
                "--python",
                str(venv_path / "bin" / "python"),
                "--force-reinstall",
                "--no-deps",
                str(imagecodecs_wheel),
            ]
        )

    if not dry_run:
        command_environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv_path)}
        for command in commands:
            subprocess.run(command, check=True, cwd=resolved_root, env=command_environment)
        manifest_path = venv_path / ".pelagia-environment.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "profile": profile.name,
                    "manager": "uv",
                    "python_request": python_request,
                    "extras": list(profile.extras),
                    "lockfile": str(resolved_root / "uv.lock"),
                    "imagecodecs_wheel": None if imagecodecs_wheel is None else str(imagecodecs_wheel),
                    "python": str(venv_path / "bin" / "python"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "profile": profile.name,
        "venv": str(venv_path),
        "python": str(venv_path / "bin" / "python"),
        "manager": "uv",
        "python_request": python_request,
        "extras": list(profile.extras),
        "lockfile": str(resolved_root / "uv.lock"),
        "imagecodecs_wheel": None if imagecodecs_wheel is None else str(imagecodecs_wheel),
        "dry_run": dry_run,
        "commands": commands,
    }


def _resolve_uv(executable: Path | None) -> Path:
    if executable is not None:
        resolved = executable.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"uv executable was not found: {resolved}")
        return resolved
    discovered = shutil.which("uv")
    if discovered:
        return Path(discovered).resolve()
    raise ValueError(
        "uv is required to synchronize Pelagia environments. Install it from "
        "https://docs.astral.sh/uv/getting-started/installation/."
    )


def doctor_profiles(
    profile_name: str = "all",
    *,
    root: Path | None = None,
    require_gpu: bool = False,
    require_jpegxs: bool = False,
) -> dict[str, Any]:
    normalized = profile_name.strip().lower().replace("_", "-")
    names = list(DOCTOR_PROFILES) if normalized == "all" else [normalized]
    unknown = [name for name in names if name not in DOCTOR_PROFILES]
    if unknown:
        valid = ", ".join(["all", *sorted(DOCTOR_PROFILES)])
        raise ValueError(f"doctor profile must be one of: {valid}.")

    resolved_root = resolve_root(root)
    profiles = []
    healthy = True
    for name in names:
        venv_name, requires_tensorflow = DOCTOR_PROFILES[name]
        venv_path = resolved_root / venv_name
        executable = venv_path / "bin" / "python"
        if not executable.is_file():
            profiles.append(
                {
                    "profile": name,
                    "venv": str(venv_path),
                    "healthy": False,
                    "error": "virtual environment is missing",
                }
            )
            healthy = False
            continue

        probe = _probe_environment(executable, requires_tensorflow=requires_tensorflow)
        errors = list(probe.pop("errors"))
        warnings = list(probe.pop("warnings"))
        if require_jpegxs and not probe["jpegxs_available"]:
            errors.append("JPEG XS support is required but unavailable")
        if require_gpu and requires_tensorflow and not probe["gpu_devices"]:
            errors.append("GPU support is required but no TensorFlow GPU device was detected")
        profile_healthy = not errors
        healthy = healthy and profile_healthy
        profiles.append(
            {
                "profile": name,
                "venv": str(venv_path),
                "healthy": profile_healthy,
                "errors": errors,
                "warnings": warnings,
                **probe,
            }
        )
    return {"healthy": healthy, "profiles": profiles}


def _probe_environment(executable: Path, *, requires_tensorflow: bool) -> dict[str, Any]:
    probe_script = """
import importlib.metadata
import json
import sys

result = {
    'python': sys.version,
    'packages': {},
    'jpegxs_available': False,
    'gpu_devices': [],
    'errors': [],
    'warnings': [],
}
for package in ('numpy', 'imagecodecs', 'tensorflow', 'tensorflow-metal'):
    try:
        result['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        result['packages'][package] = None
try:
    import imagecodecs
    result['jpegxs_available'] = bool(getattr(imagecodecs.JPEGXS, 'available', False))
    if not result['jpegxs_available']:
        result['warnings'].append('imagecodecs JPEG XS support is unavailable')
except Exception as exc:
    result['errors'].append(f'imagecodecs import failed: {exc}')
if __REQUIRES_TENSORFLOW__:
    try:
        import tensorflow as tf
        result['gpu_devices'] = [device.name for device in tf.config.list_physical_devices('GPU')]
        if not result['gpu_devices']:
            result['warnings'].append('TensorFlow did not detect a GPU device')
    except Exception as exc:
        result['errors'].append(f'TensorFlow import failed: {exc}')
print(json.dumps(result))
""".replace("__REQUIRES_TENSORFLOW__", repr(requires_tensorflow))
    result = subprocess.run(
        [str(executable), "-c", probe_script],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "python": None,
            "packages": {},
            "jpegxs_available": False,
            "gpu_devices": [],
            "errors": [result.stderr.strip() or "environment probe failed"],
            "warnings": [],
        }
    return json.loads(result.stdout)
