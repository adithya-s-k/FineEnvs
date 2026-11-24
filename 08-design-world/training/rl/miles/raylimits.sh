#!/usr/bin/env bash
# raylimits.sh <command...>: run a command with Ray capped to ${RAY_CPU_CAP:-32} CPUs (raybin/ray shim) and
# capped OpenMP threads. On some HF Jobs nodes (AWS, kernel *-amzn2023, 192 CPUs) thread creation fails with
# EAGAIN ("Resource temporarily unavailable") at ~4k threads in the container although every visible limit is
# far higher; Ray's ~192 prestarted workers alone exceed that.
export PATH="$(cd "$(dirname "$0")" && pwd)/raybin:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export RAY_num_server_call_thread=${RAY_num_server_call_thread:-4}
exec "$@"
