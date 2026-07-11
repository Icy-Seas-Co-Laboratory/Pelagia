from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MINIMUM_PROFILE_PYTHON = (3, 12)


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    name: str
    venv_name: str
    requirements_file: str | None
    requires_tensorflow: bool = False


SYNC_PROFILES = {
    "cpu": EnvironmentProfile("cpu", ".venv", "requirements-worker-cpu.txt"),
    "ml-metal": EnvironmentProfile("ml-metal", ".venv-ml", "requirements-ml-apple-metal.txt", True),
    "ml-cuda": EnvironmentProfile("ml-cuda", ".venv-ml", "requirements-ml.txt", True),
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
    if normalized in {"cpu", "default"}:
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
    python: Path | None = None,
    imagecodecs_wheel: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized = profile_name.strip().lower().replace("_", "-")
    profile = SYNC_PROFILES.get(normalized)
    if profile is None:
        valid = ", ".join(sorted(SYNC_PROFILES))
        raise ValueError(f"sync profile must be one of: {valid}.")

    resolved_root = resolve_root(root)
    source_python = (python or Path(sys.executable)).expanduser().resolve()
    if not source_python.is_file():
        raise ValueError(f"Python executable was not found: {source_python}")
    source_version = _python_version(source_python)
    if source_version < MINIMUM_PROFILE_PYTHON:
        required = ".".join(map(str, MINIMUM_PROFILE_PYTHON))
        raise ValueError(f"The {profile.name} profile requires Python {required}+.")

    venv_path = resolved_root / profile.venv_name
    requirement_path = resolved_root / str(profile.requirements_file)
    if not requirement_path.is_file():
        raise ValueError(f"Requirements file was not found: {requirement_path}")
    if imagecodecs_wheel is not None:
        imagecodecs_wheel = imagecodecs_wheel.expanduser().resolve()
        if not imagecodecs_wheel.is_file():
            raise ValueError(f"imagecodecs wheel was not found: {imagecodecs_wheel}")

    commands = [
        [str(source_python), "-m", "venv", str(venv_path)],
        [str(venv_path / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip"],
        [str(venv_path / "bin" / "python"), "-m", "pip", "install", "-r", str(requirement_path)],
    ]
    if imagecodecs_wheel is not None:
        commands.append(
            [
                str(venv_path / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(imagecodecs_wheel),
            ]
        )

    if not dry_run:
        for command in commands:
            subprocess.run(command, check=True)
        manifest_path = venv_path / ".pelagia-environment.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "profile": profile.name,
                    "requirements_file": str(requirement_path),
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
        "requirements_file": str(requirement_path),
        "imagecodecs_wheel": None if imagecodecs_wheel is None else str(imagecodecs_wheel),
        "dry_run": dry_run,
        "commands": commands,
    }


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


def _python_version(executable: Path) -> tuple[int, int]:
    result = subprocess.run(
        [str(executable), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"Could not inspect Python executable: {executable}")
    try:
        major, minor = result.stdout.strip().split(".", maxsplit=1)
        return int(major), int(minor)
    except ValueError as exc:
        raise ValueError(f"Could not parse Python version from {executable}: {result.stdout!r}") from exc
