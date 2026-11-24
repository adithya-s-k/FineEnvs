#!/usr/bin/env bash
# What an HF Job runs: copy the source, install, prepare the serving snapshot, then one action.
#   entrypoint.sh {check|smoke|train|eval} [run.py options]
set -euo pipefail
mkdir -p /workspace/retroenv
cp -a /source/. /workspace/retroenv/
cd /workspace/retroenv
bash train/jobs/install.sh
# vLLM's kernel builds call ninja and friends by name, as in 05-multi-harness-rl.
export PATH="$PWD/.venv-train/bin:$PATH"

# The serving snapshot comes from the RetroEnv bucket, mounted read-only at /data, and every
# file is checked against the pinned manifest's SHA-256 (envs/retro_route/openenv/prepare.py).
export RETROENV_BUCKET_ROOT=/data RETROENV_PREPARED_DIR=/workspace/prepared
RETROENV_BENCHMARK_DIR="$(.venv-env/bin/python envs/retro_route/openenv/prepare.py | tail -n 1)"
export RETROENV_BENCHMARK_DIR

action="$1"
shift
case "$action" in
    check) exit 0 ;;  # install.sh already ran the setup checks
    smoke) exec .venv-train/bin/python train/jobs/smoke.py "$@" ;;
    *) exec .venv-train/bin/python train/jobs/run.py "$action" "$@" ;;
esac
