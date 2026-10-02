#!/usr/bin/env bash
set -euo pipefail
# Run from 05-multi-harness-rl in a Python 3.12 virtual environment.
python -m pip install -r requirements.txt
python -m pip install --upgrade "trl @ git+https://github.com/huggingface/trl.git@main"
mkdir -p .deps
if [[ ! -d .deps/OpenEnv/.git ]]; then
    git clone --depth 1 --branch main https://github.com/huggingface/OpenEnv.git .deps/OpenEnv
else
    git -C .deps/OpenEnv pull --ff-only origin main
fi
python -m pip install -e .deps/OpenEnv
# The environment examples are source packages, outside OpenEnv's core wheel.
python - <<'PY'
from pathlib import Path
from importlib.metadata import distribution
import json
import site
import subprocess
Path(site.getsitepackages()[0], 'openenv_examples.pth').write_text(str(Path('.deps/OpenEnv/envs').resolve()) + '\n')
versions = {}
for name in ('openenv', 'trl', 'transformers', 'vllm', 'harbor'):
    installed = distribution(name)
    source = json.loads(installed.read_text('direct_url.json') or '{}')
    versions[name] = {'version': installed.version, 'commit': source.get('vcs_info', {}).get('commit_id')}
versions['openenv']['commit'] = subprocess.check_output(['git', '-C', '.deps/OpenEnv', 'rev-parse', 'HEAD'], text=True).strip()
Path('.deps/revisions.json').write_text(json.dumps(versions, indent=2) + '\n')
PY
python -m pip install --no-deps -e envs/whitebox -e envs/opencode -e envs/harbor
python check_setup.py --mode "${TUTORIAL_MODE:-all}"
