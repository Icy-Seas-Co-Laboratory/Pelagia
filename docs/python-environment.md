# Python Environment Setup

Pelagia is packaged with `pyproject.toml`, and the requirements files are thin
install targets for common environments.

## 1. Create A Virtual Environment

Use Python 3.10 or newer. Python 3.11+ is preferred because TOML support is
built into the standard library.

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

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

For machines that will run learned ROI refinement models such as the
oracle-builder U-Net adapter:

```bash
python -m pip install -r requirements-ml.txt
```

The ML install includes TensorFlow/Keras and is intentionally separate because
it is much heavier than the normal backend runtime.

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
python -m Pelagia.cli.app init_system
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
