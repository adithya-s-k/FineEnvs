#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/tutorial
cp -a /tutorial-source/. /workspace/tutorial/
cd /workspace/tutorial
uv venv --python 3.12 --seed .venv
export PATH="$PWD/.venv/bin:$PATH"
bash jobs/install.sh
python prepare.py
if [[ "$1" == "smoke" ]]; then
    shift
    exec python jobs/smoke.py "$@"
fi
exec python jobs/run.py "$@"
