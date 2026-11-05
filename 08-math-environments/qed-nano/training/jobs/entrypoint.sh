#!/usr/bin/env bash
# The HF Job's command: copy the source, install, prepare tasks, then run.
set -euo pipefail
mkdir -p /workspace/qed-nano
cp -a /qed-nano-source/. /workspace/qed-nano/
cd /workspace/qed-nano/training
uv venv --python 3.12 --seed .venv
export PATH="$PWD/.venv/bin:$PATH"
bash jobs/install.sh
python prepare.py ${QED_PREPARE_ARGS:-}
if [[ "$1" == "smoke" ]]; then
    shift
    exec python jobs/smoke.py "$@"
fi
exec python jobs/run.py "$@"
