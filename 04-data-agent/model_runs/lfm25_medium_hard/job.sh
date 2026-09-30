#!/usr/bin/env bash
set -euo pipefail
cd /fsx/adithyaskolavi/projects/trl_prod
export PYTHONUNBUFFERED=1
exec .venv312/bin/python -u HuggingEnvs/04-data-agent/model_runs/lfm25_medium_hard/run.py "$@"
