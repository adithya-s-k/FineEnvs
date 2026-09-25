#!/usr/bin/env bash
# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the environment locally with the same configuration a deployment uses, so a client cannot tell
# the two apart.
#
# Usage:
#   ./serve.sh
#   PORT=8200 SPLITS=train:medium SANDBOX=e2b ./serve.sh
#   EXPOSE=gradio ./serve.sh        # publish the capture port for a remote sandbox
#
# The engine is OPTIONAL. With none named the server still comes up serving its splits, and each
# rollout names the engine it wants. That is the useful way round: the dataset and its prebuilt
# sandbox templates are the expensive things to host, while an engine restarts every training run and
# a train-tier engine and an eval-tier one are usually both wanted at once.

set -euo pipefail

# This script sits at the environment root, so its own directory IS the environment. The directory is
# named `blackbox-opencode`, which is not a legal Python identifier, so the package cannot be
# imported by adding this path to PYTHONPATH -- `uv sync` here installs it as `data_agent_env` via the
# package-dir mapping in pyproject.toml, and that is what makes the import work at all.
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")"
ENV_DIR="$PWD"
PYTHON="${PYTHON:-$ENV_DIR/.venv/bin/python}"

PORT="${PORT:-8200}"
HOST="${HOST:-0.0.0.0}"

if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON" >&2
  echo "run 'uv sync' in $ENV_DIR, or set PYTHON= to one that has openenv-data-agent-env installed" >&2
  exit 1
fi

if ! "$PYTHON" -c "import data_agent_env" 2>/dev/null; then
  echo "data_agent_env is not importable by $PYTHON" >&2
  echo "run 'uv sync' in $ENV_DIR" >&2
  exit 1
fi

# Which splits this deployment serves. A difficulty is a NAMED SPLIT (`train:medium`), never a filter
# argument: a filter shifts every index after it, and the index is task identity everywhere
# downstream -- in the eval matrix, in the trace store, in every result already recorded.
export DATA_AGENT_SPLITS="${SPLITS:-train}"
export DATA_AGENT_SANDBOX="${SANDBOX:-e2b}"

# Rollouts EXECUTING at once. The ceiling is the capture proxy, which is a single uvicorn process: it
# starved /health at ~200 concurrent and crashed outright at 320 (3,525 fds, 542 threads, 6.7 GB).
# E2B allows 500 sandboxes per account, so capture gives out first. A semaphore, not a rejection -- an
# over-limit rollout waits, because an eval run must not be able to starve a training run out.
export DATA_AGENT_MAX_CONCURRENT="${MAX_CONCURRENT:-64}"

# WebSocket SESSIONS, which is a different number: a trainer holds one per in-flight rollout, so this
# has to exceed `num_generations` or rollouts queue at the door. Core defaults to 4, a demo setting.
export MAX_CONCURRENT_ENVS="${MAX_CONCURRENT_ENVS:-128}"

# The port the capture proxy binds, and -- separately -- how the SANDBOX reaches it. The sandbox runs
# on another machine, so a deployment behind a tunnel must advertise its outside address here.
# Getting this wrong is quiet: opencode starts, cannot reach the engine, makes zero model calls, and
# the rollout returns a flat zero that reads exactly like a policy that cannot do the task.
export DATA_AGENT_CAPTURE_PORT="${CAPTURE_PORT:-8300}"

# How the sandbox reaches that port. `direct` is right only when this host is already routable from
# the sandbox -- on a cluster node it is not, and the symptom is the quiet one above. `gradio` mints
# a public URL; prefer it over `cloudflare`, which wedged for 32 minutes on this cluster, and a
# forwarder that hangs is worse than one that fails because rollouts queue behind it looking healthy.
export DATA_AGENT_CAPTURE_EXPOSE="${EXPOSE:-direct}"

# An already-published deployment (a Space, a reverse proxy) sets this and no tunnel is started.
[ -n "${CAPTURE_PUBLIC_URL:-}" ] && export CAPTURE_PUBLIC_URL

export OPENENV_LLM_URL="${LLM_URL:-}"
export OPENENV_MODEL="${MODEL:-}"
export ENABLE_WEB_INTERFACE=true

echo "data_agent_env  ->  http://$HOST:$PORT/web/"
echo "  splits      $DATA_AGENT_SPLITS"
echo "  sandbox     $DATA_AGENT_SANDBOX"
echo "  concurrency $DATA_AGENT_MAX_CONCURRENT executing / $MAX_CONCURRENT_ENVS sessions"
echo "  engine      ${OPENENV_LLM_URL:-none (each rollout names its own)}"
echo "  capture     :$DATA_AGENT_CAPTURE_PORT expose=$DATA_AGENT_CAPTURE_EXPOSE${CAPTURE_PUBLIC_URL:+ public=$CAPTURE_PUBLIC_URL}"
echo

exec "$PYTHON" -m uvicorn data_agent_env.server.app:app --host "$HOST" --port "$PORT"
