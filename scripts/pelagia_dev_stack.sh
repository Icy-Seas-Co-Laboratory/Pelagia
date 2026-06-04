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
PELAGIA_WORKER_STAGES="${PELAGIA_WORKER_STAGES:-extract_frames,segment}"
PELAGIA_IDLE_SLEEP_SECONDS="${PELAGIA_IDLE_SLEEP_SECONDS:-2.0}"
PELAGIA_REQUEUE_INTERVAL_SECONDS="${PELAGIA_REQUEUE_INTERVAL_SECONDS:-30.0}"
PELAGIA_INIT_ON_START="${PELAGIA_INIT_ON_START:-auto}"
PELAGIA_INIT_STATEMENT_TIMEOUT_MS="${PELAGIA_INIT_STATEMENT_TIMEOUT_MS:-0}"

mkdir -p "$PID_DIR" "$LOG_DIR"

cd "$ROOT_DIR"

is_running() {
    local pid_file="$1"
    if [[ ! -f "$pid_file" ]]; then
        return 1
    fi
    local pid
    pid="$(cat "$pid_file")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
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

worker_name_for_stage() {
    local stage="$1"
    echo "pelagia-${stage//_/-}-1"
}

storage_is_ready() {
    python -m Pelagia.cli.app list_asset_ids \
        --database-dsn "$PELAGIA_DATABASE_DSN" \
        --schema "$PELAGIA_DATABASE_SCHEMA" \
        --kvstore-root "$PELAGIA_KVSTORE_ROOT" \
        --limit 1 >"$LOG_DIR/storage-check.log" 2>&1
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
    if ! PELAGIA_DB_STATEMENT_TIMEOUT_MS="$PELAGIA_INIT_STATEMENT_TIMEOUT_MS" python -m Pelagia.cli.app init-system \
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
    initialize_system

    export PELAGIA_DATABASE_DSN
    export PELAGIA_DATABASE_SCHEMA
    export PELAGIA_KVSTORE_ROOT

    start_process api \
        python -m uvicorn Pelagia.api.app:create_app \
        --factory \
        --host "$PELAGIA_API_HOST" \
        --port "$PELAGIA_API_PORT"

    IFS=',' read -r -a stages <<<"$PELAGIA_WORKER_STAGES"
    for raw_stage in "${stages[@]}"; do
        stage="$(echo "$raw_stage" | xargs)"
        if [[ -z "$stage" ]]; then
            continue
        fi
        worker_id="$(worker_name_for_stage "$stage")"
        start_process "worker-$stage" \
            python -m Pelagia.cli.app worker_run \
            --database-dsn "$PELAGIA_DATABASE_DSN" \
            --schema "$PELAGIA_DATABASE_SCHEMA" \
            --kvstore-root "$PELAGIA_KVSTORE_ROOT" \
            --worker-id "$worker_id" \
            --stages "$stage" \
            --idle-sleep-seconds "$PELAGIA_IDLE_SLEEP_SECONDS" \
            --requeue-interval-seconds "$PELAGIA_REQUEUE_INTERVAL_SECONDS"
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
        worker_id="$(worker_name_for_stage "$stage")"
        python -m Pelagia.cli.app worker_shutdown "$worker_id" \
            --database-dsn "$PELAGIA_DATABASE_DSN" \
            --schema "$PELAGIA_DATABASE_SCHEMA" \
            --reason "dev stack stop" >"$LOG_DIR/worker-$stage.shutdown.log" 2>&1 || true
        stop_pid_file "worker-$stage"
    done
    stop_pid_file api
}

status_stack() {
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
