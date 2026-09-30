#!/bin/bash
set -euo pipefail
exec /fsx/adithyaskolavi/projects/trl_prod/.venv312/bin/python -u /fsx/adithyaskolavi/projects/trl_prod/HuggingEnvs/04-data-agent/reports/async-comparison-20260916/collector-source-v2/consolidate_async_runs.py --out /fsx/adithyaskolavi/projects/trl_prod/HuggingEnvs/04-data-agent/reports/async-comparison-20260916 --online --watch
