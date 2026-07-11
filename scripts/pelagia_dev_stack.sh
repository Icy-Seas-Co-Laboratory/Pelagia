#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PELAGIA_RUN_DIR:-$ROOT_DIR/.pelagia/run}"
PID_DIR="$RUN_DIR/pids"
LOG_DIR="$RUN_DIR/logs"

PELAGIA_DATABASE_DSN="${PELAGIA_DATABASE_DSN:-postgresql://postgres:postgres@127.0.0.1:5432/pelagia}"
PELAGIA_DATABASE_SCHEMA="${PELAGIA_DATABASE_SCHEMA:-pelagia}"
PELAGIA_KVSTORE_ROOT="${PELAGIA_KVSTORE_ROOT:-$ROOT_DIR/data/kvstore}"
PELAGIA_API_HOST="${PELAGIA_API_HOST:-127.0.0.1}"
PELAGIA_API_PORT="${PELAGIA_API_PORT:-8000}"
PELAGIA_WORKER_STAGES="${PELAGIA_WORKER_STAGES:-extract_frames,background_frames,preprocess_frames,segment}"
PELAGIA_WORKER_COUNT="${PELAGIA_WORKER_COUNT:-1}"
PELAGIA_WORKER_COUNTS="${PELAGIA_WORKER_COUNTS:-}"
PELAGIA_IDLE_SLEEP_SECONDS="${PELAGIA_IDLE_SLEEP_SECONDS:-2.0}"
PELAGIA_REQUEUE_INTERVAL_SECONDS="${PELAGIA_REQUEUE_INTERVAL_SECONDS:-30.0}"
PELAGIA_INIT_ON_START="${PELAGIA_INIT_ON_START:-auto}"
PELAGIA_INIT_STATEMENT_TIMEOUT_MS="${PELAGIA_INIT_STATEMENT_TIMEOUT_MS:-0}"
PELAGIA_CPU_VENV="${PELAGIA_CPU_VENV:-$ROOT_DIR/.venv}"
PELAGIA_GPU_ML_VENV="${PELAGIA_GPU_ML_VENV:-$ROOT_DIR/.venv-ml}"

mkdir -p "$PID_DIR" "$LOG_DIR"

cd "$ROOT_DIR"

venv_python() {
    local venv_path="$1"
    local label="$2"
    if [[ ! -x "$venv_path/bin/python" ]]; then
        echo "$label virtual environment must contain an executable bin/python: $venv_path" >&2
        return 2
    fi
    echo "$venv_path/bin/python"
}

CPU_PYTHON="$(venv_python "$PELAGIA_CPU_VENV" "CPU")"

worker_runtime_for_stage() {
    local stage="$1"
    if [[ "$stage" == "roi_refinement" ]]; then
        printf 'gpu-ml|%s|%s\n' "$(venv_python "$PELAGIA_GPU_ML_VENV" "GPU/ML")" "$PELAGIA_GPU_ML_VENV"
        return
    fi
    printf 'cpu|%s|%s\n' "$CPU_PYTHON" "$PELAGIA_CPU_VENV"
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

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

worker_count_for_stage() {
    local stage="$1"
    local count="$PELAGIA_WORKER_COUNT"
    if [[ -n "$PELAGIA_WORKER_COUNTS" ]]; then
        local entries raw_entry entry entry_stage entry_count
        IFS=',' read -r -a entries <<<"$PELAGIA_WORKER_COUNTS"
        for raw_entry in "${entries[@]}"; do
            entry="$(echo "$raw_entry" | xargs)"
            [[ -z "$entry" ]] && continue
            if [[ "$entry" != *=* ]]; then
                echo "invalid PELAGIA_WORKER_COUNTS entry '$entry'; expected stage=count" >&2
                return 2
            fi
            entry_stage="$(echo "${entry%%=*}" | xargs)"
            entry_count="$(echo "${entry#*=}" | xargs)"
            if [[ "$entry_stage" == "$stage" ]]; then
                count="$entry_count"
            fi
        done
    fi
    if ! is_positive_integer "$count"; then
        echo "worker count for stage '$stage' must be a positive integer, got '$count'" >&2
        return 2
    fi
    echo "$count"
}

worker_name_for_stage() {
    local stage="$1"
    local index="$2"
    echo "pelagia-${stage//_/-}-$index"
}

storage_is_ready() {
    "$CPU_PYTHON" -m Pelagia.cli.app check-system \
        --database-dsn "$PELAGIA_DATABASE_DSN" \
        --schema "$PELAGIA_DATABASE_SCHEMA" \
        --kvstore-root "$PELAGIA_KVSTORE_ROOT" \
        >"$LOG_DIR/storage-check.log" 2>&1
}

initialize_system() {
    local log_file="$LOG_DIR/init-system.log"
    case "$PELAGIA_INIT_ON_START" in
        never)
            echo "skipping storage initialization because PELAGIA_INIT_ON_START=never"
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
            echo "PELAGIA_INIT_ON_START must be one of: auto, always, never"
            return 2
            ;;
    esac

    echo "initializing storage..."
    if ! PELAGIA_DB_STATEMENT_TIMEOUT_MS="$PELAGIA_INIT_STATEMENT_TIMEOUT_MS" "$CPU_PYTHON" -m Pelagia.cli.app init-system \
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
    export PELAGIA_KVSTORE_ROOT

    start_process api \
        "$CPU_PYTHON" -m uvicorn Pelagia.api.app:create_app \
        --factory \
        --host "$PELAGIA_API_HOST" \
        --port "$PELAGIA_API_PORT"

    IFS=',' read -r -a stages <<<"$PELAGIA_WORKER_STAGES"
    for raw_stage in "${stages[@]}"; do
        stage="$(echo "$raw_stage" | xargs)"
        if [[ -z "$stage" ]]; then
            continue
        fi
        local runtime
        if ! runtime="$(worker_runtime_for_stage "$stage")"; then
            return 2
        fi
        IFS='|' read -r runtime_profile worker_python worker_venv <<<"$runtime"
        worker_count="$(worker_count_for_stage "$stage")"
        for ((worker_index = 1; worker_index <= worker_count; worker_index++)); do
            worker_id="$(worker_name_for_stage "$stage" "$worker_index")"
            if [[ -n "$worker_venv" ]]; then
                start_process "$worker_id" \
                    env "VIRTUAL_ENV=$worker_venv" "PATH=$worker_venv/bin:$PATH" "PELAGIA_WORKER_PROFILE=$runtime_profile" \
                    "$worker_python" -m Pelagia.cli.app worker_run \
                    --database-dsn "$PELAGIA_DATABASE_DSN" \
                    --schema "$PELAGIA_DATABASE_SCHEMA" \
                    --kvstore-root "$PELAGIA_KVSTORE_ROOT" \
                    --worker-id "$worker_id" \
                    --stages "$stage" \
                    --idle-sleep-seconds "$PELAGIA_IDLE_SLEEP_SECONDS" \
                    --requeue-interval-seconds "$PELAGIA_REQUEUE_INTERVAL_SECONDS"
            else
                start_process "$worker_id" \
                    env "PELAGIA_WORKER_PROFILE=$runtime_profile" \
                    "$worker_python" -m Pelagia.cli.app worker_run \
                    --database-dsn "$PELAGIA_DATABASE_DSN" \
                    --schema "$PELAGIA_DATABASE_SCHEMA" \
                    --kvstore-root "$PELAGIA_KVSTORE_ROOT" \
                    --worker-id "$worker_id" \
                    --stages "$stage" \
                    --idle-sleep-seconds "$PELAGIA_IDLE_SLEEP_SECONDS" \
                    --requeue-interval-seconds "$PELAGIA_REQUEUE_INTERVAL_SECONDS"
            fi
        done
    done

    echo "api url=http://$PELAGIA_API_HOST:$PELAGIA_API_PORT"
    echo "logs=$LOG_DIR"
    echo "pids=$PID_DIR"
}

stop_stack() {
    IFS=',' read -r -a stages <<<"$PELAGIA_WORKER_STAGES"
    for raw_stage in "${stages[@]}"; do
        stage="$(echo "$raw_stage" | xargs)"
        if [[ -z "$stage" ]]; then
            continue
        fi
        worker_count="$(worker_count_for_stage "$stage")"
        for ((worker_index = 1; worker_index <= worker_count; worker_index++)); do
            worker_id="$(worker_name_for_stage "$stage" "$worker_index")"
            "$CPU_PYTHON" -m Pelagia.cli.app worker_shutdown "$worker_id" \
                --database-dsn "$PELAGIA_DATABASE_DSN" \
                --schema "$PELAGIA_DATABASE_SCHEMA" \
                --reason "dev stack stop" >"$LOG_DIR/$worker_id.shutdown.log" 2>&1 || true
            stop_pid_file "$worker_id"
        done
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

case "${1:-start}" in
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
    *)
        echo "usage: $0 [start|stop|restart|status]"
        exit 2
        ;;
esac
