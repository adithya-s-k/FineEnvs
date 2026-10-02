#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .deps
if [[ ! -d .deps/OpenEnv/.git ]]; then
    git clone --depth 1 --branch main https://github.com/huggingface/OpenEnv.git .deps/OpenEnv
else
    git -C .deps/OpenEnv pull --ff-only origin main
fi
python -m pip install -e .deps/OpenEnv -r requirements.txt
python -m pip install --no-deps -e .
python - <<'SETUP'
from pathlib import Path
import site
Path(site.getsitepackages()[0], "openenv_examples.pth").write_text(
    str(Path(".deps/OpenEnv/envs").resolve()) + "\n"
)
SETUP
