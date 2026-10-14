#!/usr/bin/env bash
# HF Job (CPU node): host the DesignGym env and drive Tinker RL from next to it.
#
# Sampling and training run on Tinker; this node only runs the env server (GUI editors in Chromium, one per
# episode) and the rollout driver, so many concurrent GUI episodes don't depend on a laptop. The run directory
# (metrics.jsonl, checkpoints.jsonl, rollout summaries, logs) is uploaded to RESULTS_REPO under rl/RUN_NAME/ every
# SYNC_MINUTES and at the end; checkpoints themselves live on Tinker.
#
# Env: BUNDLE_REPO, BUNDLE_PATH, RESULTS_REPO, RUN_NAME, ENV_WORKERS, SYNC_MINUTES, TRAIN_ARGS (key=value ...).
# Secrets: TINKER_API_KEY, HF_TOKEN.
set -euo pipefail
W=/work
mkdir -p $W/run
log() { echo "[$(date +%H:%M:%S)] $*"; }
sync_run() {
  python - <<'PY' || true
import os
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(os.environ["RESULTS_REPO"], repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=os.environ["RESULTS_REPO"], repo_type="dataset", folder_path="/work/run",
                  path_in_repo=f"rl/{os.environ['RUN_NAME']}", commit_message=f"rl {os.environ['RUN_NAME']}",
                  ignore_patterns=["**/*.html"])
PY
}
trap 'rc=$?; log "exit ($rc)"; tail -n 80 $W/run/env.log 2>/dev/null || true; sync_run' EXIT

log "system libraries (skia: EGL/GL/fontconfig)"
apt-get update -qq > /dev/null && apt-get install -y -qq --no-install-recommends libegl1 libgl1 libfontconfig1 > /dev/null
uv venv /opt/hub --python 3.12 -q && uv pip install --python /opt/hub/bin/python -q huggingface_hub
export PATH=/opt/hub/bin:$PATH

log "fetching the bundle (env + training code)"
python - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
tarfile.open(hf_hub_download(os.environ["BUNDLE_REPO"], os.environ["BUNDLE_PATH"], repo_type="dataset")).extractall("/work")
PY

log "starting the env server (${ENV_WORKERS:-8} workers, reward v3)"
cd $W/designgym
uv sync --frozen --no-dev -q
.venv/bin/python -m playwright install --with-deps chromium > /dev/null
.venv/bin/designgym pull --splits "${PACK_SPLITS:-rl,test}" > /dev/null  # the task pack (DESIGNGYM_PACK_REPO)
DESIGN_CANVAS_REWARD=3 MAX_CONCURRENT_ENVS=64 \
  .venv/bin/designgym serve --port 8027 --workers "${ENV_WORKERS:-8}" > $W/run/env.log 2>&1 &
ENV_PID=$!
for _ in $(seq 1 900); do
  kill -0 $ENV_PID 2>/dev/null || { log "env server exited"; exit 1; }
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8027/health', timeout=3)" 2>/dev/null && break
  sleep 1
done
log "env server is up"

log "installing the training project"
cd $W/tinker
uv venv .venv --python 3.12 -q
uv pip install --python .venv/bin/python -q -r requirements.lock  # the exact versions of the local training env
(while true; do sleep $(( ${SYNC_MINUTES:-10} * 60 )); sync_run > /dev/null 2>&1; done) &

log "training: $TRAIN_ARGS"
# shellcheck disable=SC2086
.venv/bin/python train_designgym.py env_url=http://127.0.0.1:8027 log_path=$W/run behavior_if_log_dir_exists=delete $TRAIN_ARGS \
  2>&1 | tee $W/run/train.log | grep --line-buffered -E "Training completed|Traceback|Error|stale|Saved checkpoints" || true
log "training finished"
