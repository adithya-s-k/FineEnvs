# Shared steps of the Miles and SkyRL HF Jobs (sourced by <framework>/job.sh).
#
# Env: BUNDLE_REPO, BUNDLE_PATH, RESULTS_REPO, FRAMEWORK, RUN_NAME, MODEL, N_TASKS, ENV_WORKERS, ENV_URL;
#      optional PACK_REPO (a dataset holding a DesignGym pack, served with --pack instead of `designgym pull`, e.g. a
#      DesignWorldBench v2 pack for its frozen episodes) and SFT_ADAPTER[/SFT_SUBFOLDER] (a PEFT LoRA merged into
#      MODEL before training: RL starts from that SFT checkpoint with a fresh LoRA).
# Secret: HF_TOKEN. Results (logs, metrics, checkpoints) -> RESULTS_REPO/<FRAMEWORK>/<RUN_NAME>/.
export W=${W:-/work}  # job.sh picks it (root images: /work; SkyRL's runs as the `ray` user: $HOME/work)
OUT=$W/results
SUDO=$([ "$(id -u)" = 0 ] || echo "sudo -n -E")
mkdir -p $OUT $W/data
export DESIGNGYM_EPISODE_LOG=$OUT/episodes.jsonl  # one JSON line per episode (common/episode.py)
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
kw = dict(repo_id=os.environ["RESULTS_REPO"], repo_type="dataset", folder_path=os.environ["W"] + "/results",
          path_in_repo=dest)
# logs, episodes and metrics first (small), then the checkpoints, so a failed adapter upload loses nothing else
for label, pats in (("logs", dict(ignore_patterns=["ckpt/**"])), ("checkpoints", dict(allow_patterns=["ckpt/**"]))):
    try:
        api.upload_folder(**kw, **pats, commit_message=f"{dest} {label}")
        print(f"uploaded {label}")
    except Exception as e:  # keep going: the other part may still fit
        print(f"upload of {label} failed: {type(e).__name__}: {str(e)[:300]}")
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
  local pack=()
  if [ -n "${PACK_REPO:-}" ]; then  # a fixed pack (e.g. DesignWorldBench v2: its tasks, decoys and workspaces)
    (cd $W/designgym && env -u PYTHONPATH .venv/bin/python -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], repo_type='dataset', local_dir=sys.argv[2], max_workers=16)" "$PACK_REPO" $W/pack) \
      >> $OUT/env-install.log 2>&1 || { tail -20 $OUT/env-install.log; return 1; }
    pack=(--pack $W/pack)
  else  # the task pack (DESIGNGYM_PACK_REPO): the splits the job plays
    (cd $W/designgym && env -u PYTHONPATH .venv/bin/designgym pull --splits "${PACK_SPLITS:-rl,test}") >> $OUT/env-install.log 2>&1
  fi
  # one BLAS/OpenMP thread per env worker: numpy/scipy otherwise start one thread per CPU in every worker process,
  # which alone exceeds the ~5k-thread container cap of some HF Jobs nodes (AWS) with 64 workers on 192 CPUs
  (cd $W/designgym && env -u PYTHONPATH DESIGN_CANVAS_REWARD=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     .venv/bin/designgym serve "${pack[@]}" --port 8027 --workers "${ENV_WORKERS:-8}" > $OUT/env.log 2>&1 &)
  DESIGNGYM_URL=http://127.0.0.1:8027
  for _ in $(seq 1 600); do
    curl -sf $DESIGNGYM_URL/health > /dev/null 2>&1 && { log "DesignGym is up"; return 0; }
    sleep 1
  done
  log "DesignGym did not come up"; tail -50 $OUT/env.log; return 1
}

thread_watch() {  # container thread count every minute (some nodes cap it near 5k) -> $OUT/threads.log
  (while true; do
     echo "thread count $(date +%H:%M:%S) threads=$(cat /proc/[0-9]*/status 2>/dev/null | awk '/^Threads:/ {s += $2} END {print s}') top: $(
       for d in /proc/[0-9]*; do printf '%s %s\n' "$(tr ' ' _ < $d/comm 2>/dev/null)" "$(awk '/^Threads:/ {print $2}' $d/status 2>/dev/null)"; done \
       | awk 'NF == 2 {s[$1] += $2; n[$1]++} END {for (k in s) print s[k], k "x" n[k]}' | sort -rn | head -8 | tr '\n' ' ')" \
     | tee -a $OUT/threads.log; sleep 60; done) 2> /dev/null &
}

download_model() {  # MODEL (with SFT_ADAPTER merged in, when set) -> /work/models/<name>; sets HF_DIR
  if [ -n "${SFT_ADAPTER:-}" ]; then
    HF_DIR=$W/models/$(basename "$MODEL")-sft
    log "merging $SFT_ADAPTER ${SFT_SUBFOLDER:-} into $MODEL"
    pyrun $W/rl/merge_lora.py --base "$MODEL" --adapter "$SFT_ADAPTER" --subfolder "${SFT_SUBFOLDER:-}" \
      --out "$HF_DIR" > $OUT/merge.log 2>&1 || { tail -30 $OUT/merge.log; return 1; }
    tail -2 $OUT/merge.log
    log "merged model at $HF_DIR ($(du -sh $HF_DIR | cut -f1))"
    return 0
  fi
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
