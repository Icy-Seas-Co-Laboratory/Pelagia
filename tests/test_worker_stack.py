from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
STACK_SCRIPT = ROOT_DIR / "scripts" / "pelagia_stack_from_toml.sh"


def _venv_path(tmp_path: Path, name: str) -> Path:
    venv_path = tmp_path / name
    python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(Path(sys.executable))
    return venv_path


def _write_stack_config(tmp_path: Path, contents: str) -> Path:
    config_path = tmp_path / "workers.toml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def test_stack_validate_selects_venv_by_worker_capability(tmp_path):
    cpu_venv = _venv_path(tmp_path, "cpu")
    gpu_ml_venv = _venv_path(tmp_path, "gpu-ml")
    config_path = _write_stack_config(
        tmp_path,
        f'''
[stack]
name = "pytest-worker-stack"
run_dir = "{tmp_path / "run"}"

[api]
cors_allow_origin_regex = "https?://(localhost|loopback)"

[worker_profiles]
default = "cpu"

[[worker]]
name = "ingest"
capabilities = ["ingest", "preprocess"]

[[worker]]
name = "refine"
capabilities = ["roi_refinement"]
''',
    )

    result = subprocess.run(
        [str(STACK_SCRIPT), "validate", str(config_path)],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PELAGIA_CPU_VENV": str(cpu_venv),
            "PELAGIA_GPU_ML_VENV": str(gpu_ml_venv),
        },
    )

    assert result.returncode == 0, result.stderr
    assert f"worker-ingest profile=cpu stages=extract_frames,preprocess_frames python={cpu_venv}/bin/python" in result.stdout
    assert f"worker-refine profile=gpu-ml stages=roi_refinement python={gpu_ml_venv}/bin/python" in result.stdout


def test_stack_validate_rejects_mixed_gpu_ml_and_cpu_capabilities(tmp_path):
    config_path = _write_stack_config(
        tmp_path,
        f'''
[stack]
name = "pytest-worker-stack"
run_dir = "{tmp_path / "run"}"

[[worker]]
name = "invalid"
capabilities = ["ingest", "roi_refinement"]
''',
    )

    result = subprocess.run(
        [str(STACK_SCRIPT), "validate", str(config_path)],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "mixes GPU/ML capabilities with CPU capabilities" in result.stderr
