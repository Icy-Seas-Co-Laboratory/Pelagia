#!/usr/bin/env bash

# Shared runtime checks for Pelagia's local stack launchers.

pelagia_require_available_tcp_port() {
    local host="$1"
    local port="$2"
    local service_name="$3"
    local python_executable="$4"
    local listeners=""

    if command -v lsof >/dev/null 2>&1; then
        listeners="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
        if [[ -n "$listeners" ]]; then
            echo "cannot start $service_name: TCP port $port is already in use" >&2
            echo "$listeners" >&2
            echo "stop the conflicting service or configure a different $service_name port" >&2
            return 1
        fi
        return 0
    fi

    if ! "$python_executable" - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
family = socket.AF_INET6 if ":" in host else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as sock:
    sock.bind((host, port))
PY
    then
        echo "cannot start $service_name: TCP port $port is already in use" >&2
        echo "stop the conflicting service or configure a different $service_name port" >&2
        return 1
    fi
}
