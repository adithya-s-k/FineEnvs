#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/recipe
cp -a /recipe-source/. /workspace/recipe/
cd /workspace/recipe
uv venv --python 3.12 .venv
export PATH="$PWD/.venv/bin:$PATH"
if [[ "$1" == "watch" ]]; then
    shift
    uv pip install --python .venv/bin/python huggingface_hub==1.24.0
    exec python3 eval/watch.py "$@"
fi
python3 runtime/bootstrap.py
uv pip install --python .venv/bin/python -r requirements.lock
python3 prepare.py
exec python3 run.py "$@"
