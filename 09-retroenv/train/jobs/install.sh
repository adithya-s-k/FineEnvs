#!/usr/bin/env bash
# Two environments, so the server's pins never meet vLLM's:
#   .venv-env    the project's locked environment (uv.lock): RetroEnv server and evaluator
#   .venv-train  vLLM, TRL and PEFT, plus the RetroEnv client without its server stack
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v uv >/dev/null || python -m pip install --quiet uv

UV_PROJECT_ENVIRONMENT=.venv-env uv sync --frozen --extra eval

TRL=4d25933121b197f835dce25e9306166d363584e6           # huggingface/trl main, 2026-10-09
TRANSFORMERS=9c59c0c6733990bbf8b1be7cb93b97e57dc73c39  # huggingface/transformers main, 2026-10-09
uv venv --python 3.12 --seed .venv-train
uv pip install --python .venv-train -r train/jobs/requirements.txt
uv pip install --python .venv-train \
    "trl @ git+https://github.com/huggingface/trl.git@${TRL}" \
    "transformers @ git+https://github.com/huggingface/transformers.git@${TRANSFORMERS}"
uv pip install --python .venv-train --no-deps -e envs/retro_route/openenv
.venv-train/bin/python train/jobs/check_setup.py --model "${RETROENV_MODEL:-Qwen/Qwen3.8-27B}"
