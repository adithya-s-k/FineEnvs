# Shared steps of the Miles and SkyRL HF Jobs (sourced by <framework>/job.sh).
#
# Env: BUNDLE_REPO, BUNDLE_PATH, RESULTS_REPO, FRAMEWORK, RUN_NAME, MODEL, N_TASKS, ENV_WORKERS, ENV_URL.
# Secret: HF_TOKEN. Results (logs, metrics, checkpoints) -> RESULTS_REPO/<FRAMEWORK>/<RUN_NAME>/.
export W=${W:-/work}  # job.sh picks it (root images: /work; SkyRL's runs as the `ray` user: $HOME/work)
OUT=$W/results
SUDO=$([ "$(id -u)" = 0 ] || echo "sudo -n -E")
mkdir -p $OUT $W/data
log() { echo "[$(date +%H:%M:%S)] $*"; }

pyrun() { "${PY:-python3}" "$@"; }

fetch_bundle() {
  log "fetching the bundle $BUNDLE_PATH"
  pyrun - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["BUNDLE_REPO"], os.environ["BUNDLE_PATH"], repo_type="dataset")
tarfile.open(p).extractall(os.environ.get("W", "/work"))
PY
}

scrub() {  # no token ever leaves the node in a log
  grep -rlI "hf_" $OUT 2>/dev/null | while read -r f; do sed -i -E 's/hf_[A-Za-z0-9]{20,}/hf_***/g' "$f"; done
}

upload_results() {  # upload everything in $OUT (on success or failure), tokens scrubbed
  scrub
  pyrun - <<'PY' || true
import os
from huggingface_hub import HfApi
api = HfApi()
dest = f"{os.environ['FRAMEWORK']}/{os.environ['RUN_NAME']}"
api.upload_folder(repo_id=os.environ["RESULTS_REPO"], repo_type="dataset", folder_path=os.environ["W"] + "/results",
                  path_in_repo=dest, commit_message=f"{dest} results")
print(f"results: https://huggingface.co/datasets/{os.environ['RESULTS_REPO']}/tree/main/{dest}")
PY
}

make_tasks() {  # first N_TASKS train tasks -> $W/data/tasks.jsonl (one reset kwargs dict per line)
  head -n "${N_TASKS:-8}" $W/tasks/train.jsonl > $W/data/tasks.jsonl
  log "$(wc -l < $W/data/tasks.jsonl) tasks"
}

start_env() {  # DesignGym in this job (GUI editors need Chromium); sets DESIGNGYM_URL
  if [ -n "${ENV_URL:-}" ]; then DESIGNGYM_URL=$ENV_URL; log "using env $ENV_URL"; return; fi
  log "installing the env's system libraries and uv"
  ($SUDO apt-get update -qq && $SUDO apt-get install -y -qq --no-install-recommends libegl1 libgl1 libfontconfig1) \
    > $OUT/env-install.log 2>&1
  command -v uv > /dev/null || pip install -q uv >> $OUT/env-install.log 2>&1
  (cd $W/designgym && env -u VIRTUAL_ENV -u PYTHONPATH uv sync --frozen --no-dev --python 3.12 -q \
     && $SUDO env -u PYTHONPATH PATH="$PATH" .venv/bin/python -m playwright install-deps chromium \
     && env -u PYTHONPATH .venv/bin/python -m playwright install chromium) >> $OUT/env-install.log 2>&1 \
     || { tail -50 $OUT/env-install.log; return 1; }
  log "starting DesignGym (${ENV_WORKERS:-8} workers)"
  # the task pack (DESIGNGYM_PACK_REPO): the splits the job plays
  (cd $W/designgym && env -u PYTHONPATH .venv/bin/designgym pull --splits "${PACK_SPLITS:-rl,test}") >> $OUT/env-install.log 2>&1
  (cd $W/designgym && env -u PYTHONPATH DESIGN_CANVAS_REWARD=3 \
     .venv/bin/designgym serve --port 8027 --workers "${ENV_WORKERS:-8}" > $OUT/env.log 2>&1 &)
  DESIGNGYM_URL=http://127.0.0.1:8027
  for _ in $(seq 1 600); do
    curl -sf $DESIGNGYM_URL/health > /dev/null 2>&1 && { log "DesignGym is up"; return 0; }
    sleep 1
  done
  log "DesignGym did not come up"; tail -50 $OUT/env.log; return 1
}

download_model() {  # MODEL -> /work/models/<name>; sets HF_DIR
  HF_DIR=$W/models/$(basename "$MODEL")
  log "downloading $MODEL"
  pyrun - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", local_dir="$HF_DIR", max_workers=16)
PY
  log "model at $HF_DIR ($(du -sh $HF_DIR | cut -f1))"
}

gpu_watch() {  # GPU memory / utilisation every 60 s, for the report
  (while true; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
     | tr '\n' ' ' | sed "s/^/[$(date +%H:%M:%S)] gpu /"; echo; sleep 60; done) >> $OUT/gpu.log 2>&1 &
}
