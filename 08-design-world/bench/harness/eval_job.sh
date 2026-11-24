#!/usr/bin/env bash
# HF Job (GPU node): run DesignWorldBench v1 (Recreation-CUA) for one or more Hub models, then publish the run.
#
# 1. Checks out REPO_URL at REPO_REF (the public FineEnvs repo, branch design-world, or a pinned commit) and
#    installs the DesignGym client from 08-design-world/envs/designgym.
# 2. vLLM serves each model in MODELS under its repo id, TP GPUs each: all at once when the node has enough
#    GPUs, one after the other otherwise. With PATCH_TOOL_LISTING the chat template lists tools as bare function
#    objects, as the Tinker renderer does in training.
# 3. The environment is the hosted DesignGym Space (ENV_URL) or, without ENV_URL, a server started here from the
#    checked-out code (Chromium for the built-in editors; DESIGN_CANVAS_APPS=daytona for the real apps).
# 4. rollout.py plays the frozen set (bench/evalsets/recreation-cua-v1.json) per model into
#    results/rollouts-<model>/. With RESUME_FROM, a previous run's episodes are seeded first (bench/harness/resume.py)
#    and only the missing or failed ones are played.
# 5. to_cua.py publishes the run as cua-v1 rows (data/RUN_NAME/ in RESULTS_REPO); everything in /work/results
#    (summaries, traces, frames, renders, job.json, vLLM / env / rollout logs) goes to runs/RUN_NAME/, on success
#    or failure.
#
# Env: MODELS, THINKING, TP, SURFACES, MODES, SUBSET, LIMIT, SKINS, CONCURRENCY, MAX_TOKENS, IMAGE_WINDOW, ENV_URL,
# RESULTS_REPO, RUN_NAME, REPO_URL, REPO_REF, VLLM_VERSION, MAX_MODEL_LEN, TOOL_PARSER, PATCH_TOOL_LISTING,
# RESUME_FROM, PUBLISH_CUA, ENV_ONLY. Secrets: HF_TOKEN (DAYTONA_API_KEY for the real apps).
set -euo pipefail
W=/work
mkdir -p $W/results
log() { echo "[$(date +%H:%M:%S)] $*"; }
dump_logs() { for f in $W/results/*.log; do [ -f "$f" ] && { echo "===== $f"; tail -n 120 "$f"; }; done; }
upload() {
  python - <<'PY' || true
import os
from huggingface_hub import HfApi
api = HfApi()
repo, run = os.environ["RESULTS_REPO"], os.environ["RUN_NAME"]
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path="/work/results", path_in_repo=f"runs/{run}",
                  commit_message=f"bench run {run}")
print(f"results: https://huggingface.co/datasets/{repo}/tree/main/runs/{run}")
PY
}
trap 'rc=$?; if [ $rc -ne 0 ]; then log "failed ($rc)"; dump_logs; upload; fi' EXIT

log "installing system libraries (git; EGL/GL/fontconfig for the env's renderer)"
apt-get update -qq > /dev/null && apt-get install -y -qq --no-install-recommends git libegl1 libgl1 libfontconfig1 > /dev/null

log "installing vLLM ${VLLM_VERSION:-0.30.0}"
uv venv /opt/vllm --python 3.12 -q
if [ -n "${ENV_ONLY:-}" ]; then uv pip install --python /opt/vllm/bin/python -q huggingface_hub
else uv pip install --python /opt/vllm/bin/python -q "vllm==${VLLM_VERSION:-0.30.0}" huggingface_hub; fi
export PATH=/opt/vllm/bin:$PATH

log "checking out ${REPO_URL} @ ${REPO_REF}"
git init -q $W/repo && cd $W/repo
git remote add origin "$REPO_URL"
git fetch -q --depth 1 origin "$REPO_REF"
git checkout -q FETCH_HEAD
export COMMIT=$(git rev-parse HEAD)
P=$W/repo/08-design-world
EVALSET=08-design-world/bench/evalsets/recreation-cua-v1.json
python - <<'PY'
import json, os
keys = ["MODELS", "THINKING", "TP", "SURFACES", "MODES", "SUBSET", "LIMIT", "SKINS", "CONCURRENCY", "MAX_TOKENS",
        "IMAGE_WINDOW", "ENV_URL", "RESULTS_REPO", "RUN_NAME", "REPO_URL", "REPO_REF", "COMMIT", "VLLM_VERSION",
        "MAX_MODEL_LEN", "TOOL_PARSER", "PATCH_TOOL_LISTING", "RESUME_FROM"]
json.dump({k.lower(): os.environ.get(k, "") for k in keys}, open("/work/results/job.json", "w"), indent=1)
PY
log "commit $COMMIT"

log "installing the DesignGym client"
cd $P/envs/designgym
uv sync --no-dev --extra rollout ${DESIGN_CANVAS_APPS:+--extra daytona} -q  # real apps: the env runs a Daytona pool
PY=$P/envs/designgym/.venv/bin/python
if [ -z "${ENV_URL:-}" ]; then  # no hosted env: run one here (needs Chromium for the built-in editors)
  log "starting a local env server"
  $PY -m playwright install --with-deps chromium > /dev/null
  .venv/bin/designgym pull --splits test > /dev/null  # the task pack's test split (DESIGNGYM_PACK_REPO)
  DESIGN_CANVAS_REWARD=3 .venv/bin/designgym serve --port 8027 --workers 4 > $W/results/env.log 2>&1 &
fi
ENV="${ENV_URL:-http://127.0.0.1:8027}"
cd $W/repo

wait_url() {  # url, name, seconds, [pid]
  for _ in $(seq 1 "$3"); do
    if [ -n "${4:-}" ] && ! kill -0 "$4" 2>/dev/null; then log "$2 exited"; exit 1; fi
    python -c "import urllib.request, sys; urllib.request.urlopen(sys.argv[1], timeout=3)" "$1" 2>/dev/null \
      && { log "$2 is up"; return 0; }
    sleep 1
  done
  log "$2 did not come up"; exit 1
}
wait_url "$ENV/health" "env server ($ENV)" 900
if [ -n "${ENV_ONLY:-}" ]; then
  ENV="$ENV" SKIN="${SKINS%%,*}" python - "$EVALSET" <<'PY'
import json, os, sys, urllib.request
task = json.load(open(sys.argv[1]))["tasks"][0]["task_id"]
req = urllib.request.Request(os.environ["ENV"] + "/api/episodes", headers={"content-type": "application/json"},
                             data=json.dumps({"task_id": task, "skin": os.environ["SKIN"] or "figma"}).encode())
print("episode ok:", urllib.request.urlopen(req, timeout=120).read()[:200])
PY
  dump_logs; upload; exit 0
fi

slug() { echo "${1//\//__}"; }
spec() { if [ "${THINKING:-off}" = default ]; then echo "vllm:$1"; else echo "vllm:$1@${THINKING:-off}"; fi; }
template_args() {  # model -> vLLM --chat-template arguments, with PATCH_TOOL_LISTING
  [ -n "${PATCH_TOOL_LISTING:-}" ] || return 0
  local path=$W/chat_template-$(slug "$1").jinja
  python - "$1" "$path" <<'PY'
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1]).chat_template
old = "{{- tool | tojson }}"
assert old in t, "this chat template does not list tools as `tool | tojson`: run without --patch-tool-listing"
open(sys.argv[2], "w").write(t.replace(old, "{{- (tool.function if tool.function is defined else tool) | tojson }}"))
PY
  echo "--chat-template $path"
}

# the slim image has no CUDA toolkit: FlashInfer's sampler would JIT-compile with nvcc at warmup
export VLLM_USE_FLASHINFER_SAMPLER=0
serve() {  # model, gpus (comma list), port -> sets SERVE_PID
  local name tmpl; name=$(slug "$1"); tmpl=$(template_args "$1")
  CUDA_VISIBLE_DEVICES=$2 vllm serve "$1" --served-model-name "$1" --port "$3" \
    --tensor-parallel-size "${TP:-1}" --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" $tmpl \
    --limit-mm-per-prompt '{"image": 12}' --gpu-memory-utilization 0.92 > $W/results/vllm-$name.log 2>&1 &
  SERVE_PID=$!
  tail -n +1 -F $W/results/vllm-$name.log 2>/dev/null | grep --line-buffered -E "Loading|loaded|startup complete|Error|Traceback" \
    | sed -u "s/^/[vllm-$name] /" &
}
evaluate() {  # model, port: rollout.py for one served model, into results/rollouts-<model>
  local name out extra=()
  name=$(slug "$1"); out=$W/results/rollouts-$name
  [ -n "${LIMIT:-}" ] && extra+=(--limit "$LIMIT")
  [ -n "${SKINS:-}" ] && extra+=(--skins "$SKINS")
  if [ -n "${RESUME_FROM:-}" ]; then  # carry the previous run's episodes; rollout.py plays only the rest
    $PY $P/bench/harness/resume.py "$RESUME_FROM" "$out" --repo "$RESULTS_REPO" --model "$(spec "$1")" \
      --surfaces "${SURFACES:-gui}" --modes "${MODES:-reference}" ${SKINS:+--skins "$SKINS"} \
      --evalset "$EVALSET" --subset "${SUBSET:-core30}" || log "nothing to resume for $1"
  fi
  VLLM_BASE_URL=http://127.0.0.1:$2/v1 VLLM_API_KEY=none $PY 08-design-world/envs/designgym/rollout.py \
    --base-url "$ENV" --models "$(spec "$1")" --surfaces "${SURFACES:-gui}" --modes "${MODES:-reference}" \
    --evalset "$EVALSET" --subset "${SUBSET:-core30}" --concurrency "${CONCURRENCY:-12}" \
    --max-tokens "${MAX_TOKENS:-4096}" --image-window "${IMAGE_WINDOW:-8,3}" --out "$out" "${extra[@]}" \
    > $W/results/rollout-$name.log 2>&1
  log "$1: $(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['tiers'])[:600])" $out/summary.json 2>/dev/null || echo 'no summary')"
}

IFS=, read -ra M <<< "$MODELS"
TP=${TP:-1}
GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
[ "$GPUS" -ge "$TP" ] || { log "$GPUS GPU(s) visible, $TP needed per model (nvidia-smi: $(nvidia-smi -L 2>&1 | head -2))"; exit 1; }
log "$GPUS GPU(s); ${#M[@]} model(s) on ${SURFACES:-gui} / ${MODES:-reference} / ${SUBSET:-core30}"
if [ "$GPUS" -ge $(( ${#M[@]} * TP )) ]; then  # every model at once, TP GPUs each
  PIDS=()
  for i in "${!M[@]}"; do
    serve "${M[$i]}" "$(seq $(( i * TP )) $(( i * TP + TP - 1 )) | paste -sd, -)" $(( 8000 + i )); PIDS+=($SERVE_PID)
  done
  for i in "${!M[@]}"; do wait_url http://127.0.0.1:$(( 8000 + i ))/v1/models "vLLM ${M[$i]}" 3600 "${PIDS[$i]}"; done
  JOBS=()
  for i in "${!M[@]}"; do evaluate "${M[$i]}" $(( 8000 + i )) & JOBS+=($!); done
  wait "${JOBS[@]}" || true
else  # one after the other
  for m in "${M[@]}"; do
    serve "$m" "$(seq 0 $(( TP - 1 )) | paste -sd, -)" 8000; wait_url http://127.0.0.1:8000/v1/models "vLLM $m" 3600 $SERVE_PID
    evaluate "$m" 8000 || true
    kill $SERVE_PID; sleep 15
  done
fi

if [ -n "${PUBLISH_CUA:-}" ]; then
  log "publishing the run as cua-v1 rows"
  $PY $P/bench/harness/to_cua.py $W/results/rollouts-* --out $W/cua --config "$RUN_NAME" \
    --evalset "$EVALSET" --push "$RESULTS_REPO" > $W/results/to_cua.log 2>&1 && tail -n 3 $W/results/to_cua.log \
    || log "cua export failed (results/to_cua.log)"
fi
log "uploading results"
upload
for m in "${M[@]}"; do [ -f "$W/results/rollouts-$(slug "$m")/summary.json" ] || { log "$m produced no summary"; exit 1; }; done
log "done"
