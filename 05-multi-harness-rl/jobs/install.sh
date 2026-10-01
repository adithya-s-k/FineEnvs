#!/usr/bin/env bash
set -euo pipefail
# Run from 05-multi-harness-rl in a Python 3.12 virtual environment.
python -m pip install -r requirements.txt
python -m pip install 'trl @ git+https://github.com/huggingface/trl.git@main'
mkdir -p .deps
if [[ ! -d .deps/OpenEnv/.git ]]; then
    git clone --depth 1 https://github.com/huggingface/OpenEnv.git .deps/OpenEnv
fi
python -m pip install -e .deps/OpenEnv
# The environment examples are source packages, outside OpenEnv's core wheel.
python - <<'PY'
from pathlib import Path
import site
Path(site.getsitepackages()[0], 'openenv_examples.pth').write_text(str(Path('.deps/OpenEnv/envs').resolve()) + '\n')
PY
python check_setup.py
