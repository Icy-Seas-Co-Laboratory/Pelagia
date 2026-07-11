# Python Environment Setup

Pelagia is packaged with `pyproject.toml`, and the requirements files are thin
install targets for common environments.

## 1. Create A Virtual Environment

Use Python 3.10 or newer for the base backend. The current CPU codec and ML
profiles require Python 3.12 because current `imagecodecs` does.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install Pelagia

For a normal local backend environment with API, CLI, PostgreSQL, and cold frame
storage support:

```bash
python -m pip install -r requirements.txt
```

For the API and CPU workers, synchronize the managed CPU profile instead:

```bash
./scripts/pelagia_env.py sync cpu
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

For machines that will run learned ROI refinement models such as the
oracle-builder U-Net adapter, synchronize one of the managed ML profiles:

```bash
./scripts/pelagia_env.py sync ml-metal  # Apple Metal
# or: ./scripts/pelagia_env.py sync ml-cuda
```

The ML install includes TensorFlow/Keras and is intentionally separate because
it is much heavier than the normal backend runtime. On Linux x86_64,
`requirements-ml.txt` uses TensorFlow's `and-cuda` pip extra so NVIDIA GPU
support is preferred when the driver/runtime can use it.

The `ml-metal` profile installs the TensorFlow 2.18 / `tensorflow-metal` 1.2
combination; `ml-cuda` installs the current TensorFlow CUDA target. Use
`pelagia_env.py doctor gpu-ml --require-gpu` after synchronization to verify
hardware visibility.

## 2.1 Separate CPU And GPU/ML Workers

Keep the API plus ingest, background, preprocess, and segment workers in the
managed `cpu` profile (`.venv`). Use the managed ML profile (`.venv-ml`) for
`roi_refinement` workers. The bootstrap script creates each environment,
installs the appropriate requirements, and records its profile:

```bash
./scripts/pelagia_env.py sync cpu
./scripts/pelagia_env.py sync ml-metal  # Apple Metal
# or: ./scripts/pelagia_env.py sync ml-cuda
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
./scripts/pelagia_env.py doctor all
./scripts/pelagia_stack_from_toml.sh validate scripts/pelagia_workers.toml
```

The public `imagecodecs` wheel does not include JPEG XS. Install the internally
built wheel while synchronizing the CPU profile, then require it in the doctor:

```bash
./scripts/pelagia_env.py sync cpu --imagecodecs-wheel /path/to/imagecodecs-*.whl
./scripts/pelagia_env.py doctor cpu --require-jpegxs
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

The oracle-builder U-Net artifact tests validate artifact metadata without
TensorFlow. Actual SavedModel inference tests require the ML environment and are
skipped when TensorFlow is not installed.
