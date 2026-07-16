#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-start}"
CONFIG_FILE="${2:-${PELAGIA_STACK_CONFIG:-$ROOT_DIR/scripts/pelagia_workers.example.toml}}"

cd "$ROOT_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "worker stack config not found: $CONFIG_FILE" >&2
    echo "usage: $0 [start|stop|restart|status] [workers.toml]" >&2
    exit 2
fi

CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

STACK_NAME=""
RUN_DIR=""
PID_DIR=""
LOG_DIR=""
PELAGIA_DATABASE_DSN=""
PELAGIA_DATABASE_SCHEMA=""
PELAGIA_KVSTORE_BACKEND=""
PELAGIA_KVSTORE_ROOT=""
PELAGIA_KVSTORE_MAX_BLOB_BYTES=""
PELAGIA_API_ENABLED=true
PELAGIA_API_HOST="0.0.0.0"
PELAGIA_API_PORT="8000"
PELAGIA_API_CORS_ALLOW_ORIGIN_REGEX=""
PELAGIA_FILE_BROWSER_ROOT_PATH_KVSTORE=""
PELAGIA_FILE_BROWSER_ROOT_PATH_IMPORT_DIR=""
PELAGIA_FILE_BROWSER_ALLOWED_ROOT_PATHS=""
PELAGIA_VIDEO_INGEST_N_TILE=""
PELAGIA_VIDEO_INGEST_PREFER_SOFTWARE_DECODE=""
PELAGIA_INIT_ON_START=""
PELAGIA_INIT_STATEMENT_TIMEOUT_MS=""
PELAGIA_CONTROL_PYTHON="python"
WORKER_ROWS=()

load_stack_config() {
    local parser_output row kind key value
    WORKER_ROWS=()
    parser_output="$(python - "$CONFIG_FILE" "$ROOT_DIR" <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


config_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
data = tomllib.loads(config_path.read_text(encoding="utf-8"))


def section(name: str) -> dict:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"[{name}] must be a TOML table")
    return value


def scalar(value, default):
    return default if value is None else value


def clean(value: object) -> str:
    text = str(value)
    if "\t" in text or "\n" in text:
        raise SystemExit(f"Config values may not contain tabs or newlines: {text!r}")
    return text


def worker_field(value: object) -> str:
    text = clean(value)
    if "|" in text:
        raise SystemExit(f"Worker values may not contain pipes: {text!r}")
    return text


def bool_text(value: object) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return "true"
        if lowered in {"0", "false", "no", "off"}:
            return "false"
        raise SystemExit(f"Invalid boolean value: {value!r}")
    return "true" if bool(value) else "false"


def path_value(value: object, default: Path) -> str:
    if value is None:
        return str(default)
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = root_dir / path
    return str(path)


def stack_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "toml"


stage_aliases = {
    "ingest": "extract_frames",
    "extract": "extract_frames",
    "extract_frame": "extract_frames",
    "extract_frames": "extract_frames",
    "background": "background_frames",
    "background_frame": "background_frames",
    "background_frames": "background_frames",
    "calculate_background": "background_frames",
    "preprocess": "preprocess_frames",
    "preprocess_frame": "preprocess_frames",
    "preprocess_frames": "preprocess_frames",
    "segment": "segment",
    "segmentation": "segment",
    "roi_detection": "segment",
    "refine": "roi_refinement",
    "refinement": "roi_refinement",
    "roi_refinement": "roi_refinement",
    "refine_rois": "roi_refinement",
}

stack = section("stack")
database = section("database")
kvstore = section("kvstore")
file_browser = section("file_browser")
api = section("api")
worker_defaults = section("worker_defaults")
worker_profiles = section("worker_profiles")
worker_venvs = section("worker_venvs")
processing = section("processing")
video_ingest = processing.get("video_ingest", {})
if not isinstance(video_ingest, dict):
    raise SystemExit("[processing.video_ingest] must be a TOML table")

stack_name = stack_slug(str(stack.get("name") or config_path.stem))
run_dir = path_value(
    stack.get("run_dir", os.environ.get("PELAGIA_RUN_DIR")),
    root_dir / ".pelagia" / "run" / stack_name,
)
kvstore_root = path_value(
    kvstore.get("root_path", kvstore.get("root")),
    Path(os.environ.get("PELAGIA_KVSTORE_ROOT", root_dir / "data" / "kvstore")),
)


def venv_path(value: object, *, label: str) -> tuple[str, str]:
    if value is None or not str(value).strip():
        return sys.executable, ""
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = root_dir / path
    executable = path / "bin" / "python"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SystemExit(f"{label} must contain an executable bin/python: {path}")
    return str(executable), str(path)


default_venv = worker_venvs.get("default", os.environ.get("PELAGIA_WORKER_VENV"))
if worker_profiles and worker_venvs:
    raise SystemExit("Use either [worker_profiles] or the legacy [worker_venvs] table, not both.")


def managed_venv(profile: object, *, label: str) -> str:
    normalized = str(profile).strip().lower().replace("_", "-")
    if normalized in {"cpu", "default"}:
        return os.environ.get("PELAGIA_CPU_VENV", str(root_dir / ".venv"))
    if normalized in {"gpu-ml", "ml-metal", "ml-cuda"}:
        return os.environ.get("PELAGIA_GPU_ML_VENV", str(root_dir / ".venv-ml"))
    raise SystemExit(f"{label} must be one of: cpu, gpu-ml, ml-metal, ml-cuda")


capability_venvs: dict[str, object] = {}
if worker_venvs:
    default_venv = worker_venvs.get("default", os.environ.get("PELAGIA_WORKER_VENV"))
    control_venv = worker_venvs.get("control", default_venv)
    for raw_capability, value in worker_venvs.items():
        if raw_capability in {"default", "control"}:
            continue
        stage = stage_aliases.get(str(raw_capability).strip())
        if stage is None:
            valid = ", ".join(sorted(stage_aliases))
            raise SystemExit(f"Unknown [worker_venvs] capability {raw_capability!r}. Valid aliases: {valid}")
        if not isinstance(value, str):
            raise SystemExit(f"[worker_venvs].{raw_capability} must be a virtual-environment path")
        capability_venvs[stage] = value
else:
    default_venv = managed_venv(worker_profiles.get("default", "cpu"), label="[worker_profiles].default")
    control_venv = managed_venv(
        worker_profiles.get("control", worker_profiles.get("default", "cpu")),
        label="[worker_profiles].control",
    )
    for raw_capability, value in worker_profiles.items():
        if raw_capability in {"default", "control"}:
            continue
        stage = stage_aliases.get(str(raw_capability).strip())
        if stage is None:
            valid = ", ".join(sorted(stage_aliases))
            raise SystemExit(f"Unknown [worker_profiles] capability {raw_capability!r}. Valid aliases: {valid}")
        capability_venvs[stage] = managed_venv(
            value,
            label=f"[worker_profiles].{raw_capability}",
        )

capability_venvs.setdefault(
    "roi_refinement",
    managed_venv("gpu-ml", label="ROI refinement environment"),
)

control_python, _ = venv_path(control_venv, label="worker control environment")


def path_list_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        items = value
    else:
        raise SystemExit("[file_browser].allowed_root_paths must be a list or comma-separated string")
    return ",".join(path_value(item, root_dir) for item in items)

config_rows = {
    "stack_name": stack_name,
    "run_dir": run_dir,
    "database_dsn": scalar(
        database.get("dsn"),
        os.environ.get("PELAGIA_DATABASE_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/pelagia"),
    ),
    "database_schema": scalar(database.get("schema"), os.environ.get("PELAGIA_DATABASE_SCHEMA", "pelagia")),
    "kvstore_backend": scalar(kvstore.get("backend"), os.environ.get("PELAGIA_KVSTORE_BACKEND", "kvstore")),
    "kvstore_root": kvstore_root,
    "kvstore_max_blob_bytes": scalar(
        kvstore.get("max_blob_bytes"),
        os.environ.get("PELAGIA_KVSTORE_MAX_BLOB_BYTES", "67108864"),
    ),
    "file_browser_root_path_kvstore": path_value(
        file_browser.get("root_path_kvstore"),
        Path(os.environ.get("PELAGIA_FILE_BROWSER_ROOT_PATH_KVSTORE", kvstore_root)),
    ),
    "file_browser_root_path_import_dir": path_value(
        file_browser.get("root_path_import_dir"),
        Path(os.environ.get("PELAGIA_FILE_BROWSER_ROOT_PATH_IMPORT_DIR", root_dir / "data" / "import")),
    ),
    "file_browser_allowed_root_paths": path_list_value(
        file_browser.get("allowed_root_paths"),
        os.environ.get("PELAGIA_FILE_BROWSER_ALLOWED_ROOT_PATHS", ""),
    ),
    "video_ingest_n_tile": scalar(
        video_ingest.get("n_tile"),
        os.environ.get("PELAGIA_VIDEO_INGEST_N_TILE", "4"),
    ),
    "video_ingest_prefer_software_decode": bool_text(
        video_ingest.get(
            "prefer_software_decode",
            os.environ.get("PELAGIA_VIDEO_INGEST_PREFER_SOFTWARE_DECODE", "true"),
        )
    ),
    "api_enabled": bool_text(api.get("enabled", True)),
    "api_host": scalar(api.get("host"), os.environ.get("PELAGIA_API_HOST", "127.0.0.1")),
    "api_port": scalar(api.get("port"), os.environ.get("PELAGIA_API_PORT", "8000")),
    "api_cors_allow_origin_regex": scalar(
        api.get("cors_allow_origin_regex"),
        os.environ.get("PELAGIA_API_CORS_ALLOW_ORIGIN_REGEX", ""),
    ),
    "init_on_start": scalar(stack.get("init_on_start"), os.environ.get("PELAGIA_INIT_ON_START", "auto")),
    "init_statement_timeout_ms": scalar(
        stack.get("init_statement_timeout_ms"),
        os.environ.get("PELAGIA_INIT_STATEMENT_TIMEOUT_MS", "0"),
    ),
    "control_python": control_python,
}

for key, value in config_rows.items():
    print(f"config\t{key}\t{clean(value)}")


def worker_entries() -> list[dict]:
    workers_table = data.get("workers")
    entries = []
    if isinstance(workers_table, dict):
        value = workers_table.get("worker")
        if isinstance(value, list):
            entries.extend(value)
        elif isinstance(value, dict):
            entries.append(value)

    value = data.get("worker")
    if isinstance(value, list):
        entries.extend(value)
    elif isinstance(value, dict):
        entries.append(value)

    return entries


workers = worker_entries()
if not workers:
    raise SystemExit("No workers configured. Add [[worker]] or [[workers.worker]] entries.")

default_idle = worker_defaults.get("idle_sleep_seconds", os.environ.get("PELAGIA_IDLE_SLEEP_SECONDS", "2.0"))
default_requeue = worker_defaults.get("requeue_interval_seconds", os.environ.get("PELAGIA_REQUEUE_INTERVAL_SECONDS", "30.0"))
gpu_ml_stages = {"roi_refinement"}

for index, worker in enumerate(workers, start=1):
    if not isinstance(worker, dict):
        raise SystemExit("Each worker entry must be a TOML table")
    if not worker.get("enabled", True):
        continue

    name = str(worker.get("name") or worker.get("worker_id") or f"worker-{index}")
    process_name = stack_slug(str(worker.get("process_name") or f"worker-{name}"))
    worker_id = str(worker.get("worker_id") or name)
    raw_capabilities = worker.get("capabilities", worker.get("stages"))
    if raw_capabilities is None:
        raise SystemExit(f"Worker {name!r} requires capabilities or stages")
    if isinstance(raw_capabilities, str):
        raw_capabilities = [item.strip() for item in raw_capabilities.split(",") if item.strip()]
    stages = []
    for raw_stage in raw_capabilities:
        stage = stage_aliases.get(str(raw_stage).strip())
        if stage is None:
            valid = ", ".join(sorted(stage_aliases))
            raise SystemExit(f"Unknown worker capability {raw_stage!r}. Valid aliases: {valid}")
        if stage not in stages:
            stages.append(stage)
    selected_gpu_ml_stages = set(stages) & gpu_ml_stages
    if selected_gpu_ml_stages and set(stages) - gpu_ml_stages:
        raise SystemExit(
            f"Worker {name!r} mixes GPU/ML capabilities with CPU capabilities. "
            "Configure GPU/ML stages in a dedicated worker."
        )
    resolved_venvs = {capability_venvs.get(stage, default_venv) for stage in stages}
    if len(resolved_venvs) != 1:
        raise SystemExit(
            f"Worker {name!r} resolves to multiple virtual environments. "
            "Split its capabilities into dedicated workers."
        )
    worker_python, worker_venv = venv_path(
        resolved_venvs.pop(),
        label=f"Worker {name!r} virtual environment",
    )
    runtime_profile = "gpu-ml" if selected_gpu_ml_stages else "cpu"
    count = int(worker.get("count", 1))
    if count < 1:
        raise SystemExit(f"Worker {name!r} count must be >= 1")
    idle = worker.get("idle_sleep_seconds", default_idle)
    requeue = worker.get("requeue_interval_seconds", default_requeue)
    for copy_index in range(1, count + 1):
        suffix = "" if count == 1 else f"-{copy_index}"
        print(
            "worker\t"
            f"{worker_field(process_name + suffix)}\t"
            f"{worker_field(worker_id + suffix)}|{worker_field(','.join(stages))}|"
            f"{worker_field(idle)}|{worker_field(requeue)}|{worker_field(worker_python)}|"
            f"{worker_field(worker_venv)}|{worker_field(runtime_profile)}"
        )
PY
)"
    while IFS=$'\t' read -r kind key value; do
        case "$kind" in
            config)
                case "$key" in
                    stack_name) STACK_NAME="$value" ;;
                    run_dir) RUN_DIR="$value" ;;
                    database_dsn) PELAGIA_DATABASE_DSN="$value" ;;
                    database_schema) PELAGIA_DATABASE_SCHEMA="$value" ;;
                    kvstore_backend) PELAGIA_KVSTORE_BACKEND="$value" ;;
                    kvstore_root) PELAGIA_KVSTORE_ROOT="$value" ;;
                    kvstore_max_blob_bytes) PELAGIA_KVSTORE_MAX_BLOB_BYTES="$value" ;;
                    file_browser_root_path_kvstore) PELAGIA_FILE_BROWSER_ROOT_PATH_KVSTORE="$value" ;;
                    file_browser_root_path_import_dir) PELAGIA_FILE_BROWSER_ROOT_PATH_IMPORT_DIR="$value" ;;
                    file_browser_allowed_root_paths) PELAGIA_FILE_BROWSER_ALLOWED_ROOT_PATHS="$value" ;;
                    video_ingest_n_tile) PELAGIA_VIDEO_INGEST_N_TILE="$value" ;;
                    video_ingest_prefer_software_decode) PELAGIA_VIDEO_INGEST_PREFER_SOFTWARE_DECODE="$value" ;;
                    api_enabled) PELAGIA_API_ENABLED="$value" ;;
                    api_host) PELAGIA_API_HOST="$value" ;;
                    api_port) PELAGIA_API_PORT="$value" ;;
                    api_cors_allow_origin_regex) PELAGIA_API_CORS_ALLOW_ORIGIN_REGEX="$value" ;;
                    init_on_start) PELAGIA_INIT_ON_START="$value" ;;
                    init_statement_timeout_ms) PELAGIA_INIT_STATEMENT_TIMEOUT_MS="$value" ;;
                    control_python) PELAGIA_CONTROL_PYTHON="$value" ;;
                esac
                ;;
            worker)
                WORKER_ROWS+=("$key"$'\t'"$value")
                ;;
        esac
    done <<<"$parser_output"

    if [[ -z "$RUN_DIR" || "$RUN_DIR" == "/" ]]; then
        echo "resolved run_dir is unsafe: '$RUN_DIR'" >&2
        echo "Set [stack].run_dir in $CONFIG_FILE or export PELAGIA_RUN_DIR to a writable directory." >&2
        exit 2
    fi
    case "$RUN_DIR" in
        *'$'*)
            echo "resolved run_dir still contains an unresolved environment variable: $RUN_DIR" >&2
            echo "Check [stack].run_dir in $CONFIG_FILE." >&2
            exit 2
            ;;
    esac

    PID_DIR="$RUN_DIR/pids"
    LOG_DIR="$RUN_DIR/logs"
    mkdir -p "$RUN_DIR" "$PID_DIR" "$LOG_DIR"
}

is_running() {
    local pid_file="$1"
    if [[ ! -f "$pid_file" ]]; then
        return 1
    fi
    local pid kill_output
    pid="$(cat "$pid_file")"
    if [[ -z "$pid" ]]; then
        return 1
    fi
    if kill_output="$(kill -0 "$pid" 2>&1)"; then
        return 0
    fi
    [[ "$kill_output" == *"Operation not permitted"* ]]
}

start_process() {
    local name="$1"
    shift
    local pid_file="$PID_DIR/$name.pid"
    local log_file="$LOG_DIR/$name.log"
    if is_running "$pid_file"; then
        echo "$name already running with pid $(cat "$pid_file")"
        return
    fi
    nohup "$@" >"$log_file" 2>&1 &
    echo "$!" >"$pid_file"
    sleep 0.3
    if ! is_running "$pid_file"; then
        rm -f "$pid_file"
        echo "failed to start $name; log follows:"
        tail -80 "$log_file" || true
        return 1
    fi
    echo "started $name pid=$(cat "$pid_file") log=$log_file"
}

stop_pid_file() {
    local name="$1"
    local pid_file="$PID_DIR/$name.pid"
    if ! is_running "$pid_file"; then
        rm -f "$pid_file"
        echo "$name is not running"
        return
    fi
    local pid
    pid="$(cat "$pid_file")"
    kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$pid_file"
            echo "stopped $name"
            return
        fi
        sleep 0.25
    done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$pid_file"
    echo "force-stopped $name"
}

cleanup_stale_pid_files() {
    local pid_file name
    for pid_file in "$PID_DIR"/*.pid; do
        [[ -e "$pid_file" ]] || return 0
        if ! is_running "$pid_file"; then
            name="$(basename "$pid_file" .pid)"
            rm -f "$pid_file"
            echo "removed stale pid file for $name"
        fi
    done
}

storage_is_ready() {
    PELAGIA_KVSTORE_BACKEND="$PELAGIA_KVSTORE_BACKEND" \
    PELAGIA_KVSTORE_MAX_BLOB_BYTES="$PELAGIA_KVSTORE_MAX_BLOB_BYTES" \
    "$PELAGIA_CONTROL_PYTHON" -m Pelagia.cli.app check-system \
        --database-dsn "$PELAGIA_DATABASE_DSN" \
        --schema "$PELAGIA_DATABASE_SCHEMA" \
        --kvstore-root "$PELAGIA_KVSTORE_ROOT" \
        >"$LOG_DIR/storage-check.log" 2>&1
}

initialize_system() {
    local log_file="$LOG_DIR/init-system.log"
    case "$PELAGIA_INIT_ON_START" in
        never)
            echo "skipping storage initialization because init_on_start=never"
            return
            ;;
        auto)
            if storage_is_ready; then
                echo "storage already initialized"
                return
            fi
            ;;
        always)
            ;;
        *)
            echo "init_on_start must be one of: auto, always, never"
            return 2
            ;;
    esac

    echo "initializing storage..."
    if ! PELAGIA_DB_STATEMENT_TIMEOUT_MS="$PELAGIA_INIT_STATEMENT_TIMEOUT_MS" \
        PELAGIA_KVSTORE_BACKEND="$PELAGIA_KVSTORE_BACKEND" \
        PELAGIA_KVSTORE_MAX_BLOB_BYTES="$PELAGIA_KVSTORE_MAX_BLOB_BYTES" \
        "$PELAGIA_CONTROL_PYTHON" -m Pelagia.cli.app init-system \
        --database-dsn "$PELAGIA_DATABASE_DSN" \
        --schema "$PELAGIA_DATABASE_SCHEMA" \
        --kvstore-root "$PELAGIA_KVSTORE_ROOT" >"$log_file" 2>&1; then
        echo "failed to initialize storage; log follows:"
        tail -120 "$log_file" || true
        return 1
    fi
    echo "initialized storage log=$log_file"
}

start_stack() {
    cleanup_stale_pid_files
    initialize_system

    export PELAGIA_DATABASE_DSN
    export PELAGIA_DATABASE_SCHEMA
    export PELAGIA_KVSTORE_BACKEND
    export PELAGIA_KVSTORE_ROOT
    export PELAGIA_KVSTORE_MAX_BLOB_BYTES
    export PELAGIA_FILE_BROWSER_ROOT_PATH_KVSTORE
    export PELAGIA_FILE_BROWSER_ROOT_PATH_IMPORT_DIR
    export PELAGIA_FILE_BROWSER_ALLOWED_ROOT_PATHS
    export PELAGIA_VIDEO_INGEST_N_TILE
    export PELAGIA_VIDEO_INGEST_PREFER_SOFTWARE_DECODE
    if [[ -n "$PELAGIA_API_CORS_ALLOW_ORIGIN_REGEX" ]]; then
        export PELAGIA_API_CORS_ALLOW_ORIGIN_REGEX
    fi

    if [[ "$PELAGIA_API_ENABLED" == "true" ]]; then
        start_process api \
            "$PELAGIA_CONTROL_PYTHON" -m uvicorn Pelagia.api.app:create_app \
            --factory \
            --host "$PELAGIA_API_HOST" \
            --port "$PELAGIA_API_PORT" \
            --workers 1
    else
        echo "api disabled by config"
    fi

    local row process_name rest worker_id stages idle requeue worker_python worker_venv runtime_profile
    for row in "${WORKER_ROWS[@]}"; do
        process_name="${row%%$'\t'*}"
        rest="${row#*$'\t'}"
        IFS='|' read -r worker_id stages idle requeue worker_python worker_venv runtime_profile <<<"$rest"
        if [[ -n "$worker_venv" ]]; then
            start_process "$process_name" \
                env "VIRTUAL_ENV=$worker_venv" "PATH=$worker_venv/bin:$PATH" "PELAGIA_WORKER_PROFILE=$runtime_profile" \
                "$worker_python" -m Pelagia.cli.app worker_run \
                --database-dsn "$PELAGIA_DATABASE_DSN" \
                --schema "$PELAGIA_DATABASE_SCHEMA" \
                --kvstore-root "$PELAGIA_KVSTORE_ROOT" \
                --worker-id "$worker_id" \
                --stages "$stages" \
                --idle-sleep-seconds "$idle" \
                --requeue-interval-seconds "$requeue"
        else
            start_process "$process_name" \
                env "PELAGIA_WORKER_PROFILE=$runtime_profile" \
                "$worker_python" -m Pelagia.cli.app worker_run \
                --database-dsn "$PELAGIA_DATABASE_DSN" \
                --schema "$PELAGIA_DATABASE_SCHEMA" \
                --kvstore-root "$PELAGIA_KVSTORE_ROOT" \
                --worker-id "$worker_id" \
                --stages "$stages" \
                --idle-sleep-seconds "$idle" \
                --requeue-interval-seconds "$requeue"
        fi
    done

    echo "stack=$STACK_NAME"
    [[ "$PELAGIA_API_ENABLED" == "true" ]] && echo "api url=http://$PELAGIA_API_HOST:$PELAGIA_API_PORT"
    echo "config=$CONFIG_FILE"
    echo "logs=$LOG_DIR"
    echo "pids=$PID_DIR"
}

stop_stack() {
    local row process_name rest worker_id stages idle requeue worker_python worker_venv runtime_profile
    for row in "${WORKER_ROWS[@]}"; do
        process_name="${row%%$'\t'*}"
        rest="${row#*$'\t'}"
        IFS='|' read -r worker_id stages idle requeue worker_python worker_venv runtime_profile <<<"$rest"
        "$PELAGIA_CONTROL_PYTHON" -m Pelagia.cli.app worker_shutdown "$worker_id" \
            --database-dsn "$PELAGIA_DATABASE_DSN" \
            --schema "$PELAGIA_DATABASE_SCHEMA" \
            --reason "toml stack stop" >"$LOG_DIR/$process_name.shutdown.log" 2>&1 || true
        stop_pid_file "$process_name"
    done
    stop_pid_file api
}

status_stack() {
    cleanup_stale_pid_files
    for pid_file in "$PID_DIR"/*.pid; do
        [[ -e "$pid_file" ]] || {
            echo "no pid files in $PID_DIR"
            return
        }
        name="$(basename "$pid_file" .pid)"
        if is_running "$pid_file"; then
            echo "$name running pid=$(cat "$pid_file")"
        else
            echo "$name stopped stale_pid=$(cat "$pid_file")"
        fi
    done
}

validate_stack() {
    local row process_name rest worker_id stages idle requeue worker_python worker_venv runtime_profile
    for row in "${WORKER_ROWS[@]}"; do
        process_name="${row%%$'\t'*}"
        rest="${row#*$'\t'}"
        IFS='|' read -r worker_id stages idle requeue worker_python worker_venv runtime_profile <<<"$rest"
        echo "$process_name profile=$runtime_profile stages=$stages python=$worker_python"
    done
}

load_stack_config

case "$ACTION" in
    start)
        start_stack
        ;;
    stop)
        stop_stack
        ;;
    restart)
        stop_stack
        start_stack
        ;;
    status)
        status_stack
        ;;
    validate)
        validate_stack
        ;;
    *)
        echo "usage: $0 [start|stop|restart|status|validate] [workers.toml]" >&2
        exit 2
        ;;
esac
