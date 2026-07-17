# Python Environment Setup

Pelagia uses uv to install Python, resolve dependencies from `pyproject.toml`,
and synchronize environments from the committed `uv.lock`. The requirements
files remain only for compatibility with older deployments.

## 1. Install uv

Install uv using its standalone installer or your platform package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No system Python 3.12 package is required. Pelagia pins Python 3.12 in
`.python-version`, and uv downloads a managed interpreter when it is absent.
This avoids modifying Debian's externally managed system Python. Debian 13's
default Python 3.13 can remain installed unchanged.

## 2. Install Pelagia

For a normal local backend environment with API, CLI, PostgreSQL, codecs, and
cold frame storage support:

```bash
./scripts/pelagia_env sync cpu
```

The shell entrypoint first ensures a uv-managed Python 3.12 exists, so it also
works on hosts where `python3.12` is unavailable. It then synchronizes `.venv`
from the lockfile. For development and tests:

```bash
./scripts/pelagia_env sync dev
```

For machines that will run learned ROI refinement models such as the
oracle-builder U-Net adapter, synchronize one of the managed ML profiles:

```bash
./scripts/pelagia_env sync ml-metal  # Apple Metal
# or: ./scripts/pelagia_env sync ml-cuda
```

The ML install includes TensorFlow/Keras and is intentionally separate because
it is much heavier than the normal backend runtime. On Linux x86_64,
`requirements-ml.txt` uses TensorFlow's `and-cuda` pip extra so NVIDIA GPU
support is preferred when the driver/runtime can use it.

The `ml-metal` profile installs the TensorFlow 2.18 / `tensorflow-metal` 1.2
combination; `ml-cuda` installs the current TensorFlow CUDA target. Use
`./scripts/pelagia_env doctor gpu-ml --require-gpu` after synchronization to
verify hardware visibility.

## 2.1 Separate CPU And GPU/ML Workers

Keep the API plus ingest, background, preprocess, and segment workers in the
managed `cpu` profile (`.venv`). Use the managed ML profile (`.venv-ml`) for
`roi_refinement` workers. The bootstrap script synchronizes each environment
from `uv.lock` and records its profile:

```bash
./scripts/pelagia_env sync cpu
./scripts/pelagia_env sync ml-metal  # Apple Metal
# or: ./scripts/pelagia_env sync ml-cuda
```

Configure the TOML worker stack to select managed environments by capability:

```toml
[worker_profiles]
default = "cpu"
roi_refinement = "gpu-ml"
```

`roi_refinement` is the GPU/ML capability and must have a dedicated worker;
the stack rejects a worker that mixes it with CPU pipeline capabilities. Confirm
the environment and resolved interpreters with:

```bash
./scripts/pelagia_env doctor all
./scripts/pelagia_stack_from_toml.sh validate scripts/pelagia_workers.toml
```

The public `imagecodecs` wheel does not include JPEG XS. Install the internally
built wheel while synchronizing the CPU profile, then require it in the doctor:

```bash
./scripts/pelagia_env sync cpu --imagecodecs-wheel /path/to/imagecodecs-*.whl
./scripts/pelagia_env doctor cpu --require-jpegxs
```

On a GPU-only worker host, disable the API and set `control = "gpu-ml"` in
`[worker_profiles]`; this uses the ML environment for stack lifecycle commands
without starting CPU workers.

## 3. Configure Local Settings

Pelagia loads settings in this order:

```text
Pelagia/default.config.toml < ./config.toml < environment variables < CLI flags
```

For local overrides, create a repository-root `config.toml`. This file is
ignored by git.

Common overrides:

```toml
[database]
dsn = "postgresql://postgres:postgres@localhost:5432/pelagia"
schema_name = "pelagia"

[kvstore]
root_path = "./data/kvstore"

[api]
host = "127.0.0.1"
port = 8000
```

For learned ROI refinement with the current oracle-builder test run:

```toml
[processing.roi_refinement]
enabled = true
model_kind = "oracle_builder_unet"
model_run_dir = "../oracle-builder/runs/unet-test"
model_artifact = "auto"
```

## 4. Initialize And Run

Initialize storage:

```bash
python -m Pelagia.cli.app init-system
```

Start the local development stack:

```bash
./scripts/pelagia_dev_stack.sh start
./scripts/pelagia_dev_stack.sh status
```

Stop it when finished:

```bash
./scripts/pelagia_dev_stack.sh stop
```

## 5. Run Tests

```bash
python -m pytest
```

You can also run these without activation:

```bash
.venv/bin/pelagia init-system
.venv/bin/python -m pytest
```

The oracle-builder U-Net artifact tests validate artifact metadata without
TensorFlow. Actual SavedModel inference tests require the ML environment and are
skipped when TensorFlow is not installed.
