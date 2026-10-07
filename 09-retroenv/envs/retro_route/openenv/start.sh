#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Fetch or check the task bundle, then point the server at the directory prepare.py printed.
prepared="$(python prepare.py | tail -n 1)"
export RETROENV_BENCHMARK_DIR="${RETROENV_BENCHMARK_DIR:-${prepared:-${RETROENV_PREPARED_DIR:-prepared}}}"
exec uvicorn retroenv_openenv.server:app --host 0.0.0.0 --port "${PORT:-8000}" --ws-ping-timeout 600
