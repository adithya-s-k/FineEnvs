#!/usr/bin/env bash
# HF Job (GPU node): evaluate the base model and the RL model on DesignGym, mainly GUI.
#
# 1. vLLM serves BASE_MODEL as `base` and, with RL_MODEL (merged weights), the RL model as `rl`: one server per
#    GPU, in parallel when the node has two, one after the other otherwise. The chat template is patched to list
#    tools exactly as the Tinker renderer did in training (bare function objects).
# 2. The environment is the hosted DesignGym Space (ENV_URL, reward v3) or, without ENV_URL, a server started here
#    from the bundled code (with Chromium for the GUI editors).
# 3. rollout.py plays the evalset subset per model (thinking off, 0-1000 click grid); everything in /work/results
#    (summaries, traces, frames, vLLM / env / rollout logs) goes to RESULTS_REPO under runs/RUN_NAME/, on success
#    or failure.
#
# Env: BASE_MODEL, RL_MODEL (optional), SURFACES, SUBSET, LIMIT, SKINS, CONCURRENCY, ENV_URL, BUNDLE_REPO,
# BUNDLE_PATH, RESULTS_REPO, RUN_NAME, VLLM_VERSION, MAX_MODEL_LEN, ENV_ONLY, EVALSET (path in the bundle).
# Secret: HF_TOKEN.
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
api.create_repo(os.environ["RESULTS_REPO"], repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=os.environ["RESULTS_REPO"], repo_type="dataset", folder_path="/work/results",
                  path_in_repo=f"runs/{os.environ['RUN_NAME']}", commit_message=f"eval {os.environ['RUN_NAME']}")
print(f"results: https://huggingface.co/datasets/{os.environ['RESULTS_REPO']}/tree/main/runs/{os.environ['RUN_NAME']}")
PY
}
trap 'rc=$?; if [ $rc -ne 0 ]; then log "failed ($rc)"; dump_logs; upload; fi' EXIT

log "installing system libraries for the env's renderer (skia needs EGL/GL/fontconfig)"
apt-get update -qq > /dev/null && apt-get install -y -qq --no-install-recommends libegl1 libgl1 libfontconfig1 > /dev/null

log "installing vLLM ${VLLM_VERSION:-0.30.0}"
uv venv /opt/vllm --python 3.12 -q
if [ -n "${ENV_ONLY:-}" ]; then uv pip install --python /opt/vllm/bin/python -q huggingface_hub
else uv pip install --python /opt/vllm/bin/python -q "vllm==${VLLM_VERSION:-0.30.0}" huggingface_hub; fi
export PATH=/opt/vllm/bin:$PATH

log "fetching the bundle; patching the chat template"
python - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
tarfile.open(hf_hub_download(os.environ["BUNDLE_REPO"], os.environ["BUNDLE_PATH"], repo_type="dataset")).extractall("/work")
if os.environ.get("ENV_ONLY"):
    raise SystemExit(0)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(os.environ["BASE_MODEL"]).chat_template
old = "{{- tool | tojson }}"
assert old in t, "chat template changed upstream: re-check the tool listing against the Tinker renderer"
open("/work/chat_template.jinja", "w").write(t.replace(old, "{{- (tool.function if tool.function is defined else tool) | tojson }}"))
PY

log "installing the DesignGym client (bundled code)"
cd $W/designgym
uv sync --frozen --no-dev --extra rollout ${DESIGN_CANVAS_APPS:+--extra daytona} -q  # real apps: the env runs a Daytona pool
if [ -z "${ENV_URL:-}" ]; then  # no hosted env: run one here (needs Chromium for the GUI editors)
  log "starting a local env server"
  .venv/bin/python -m playwright install --with-deps chromium > /dev/null
  .venv/bin/designgym pull --splits "${PACK_SPLITS:-test}" > /dev/null  # the task pack (DESIGNGYM_PACK_REPO)
  DESIGN_CANVAS_REWARD=3 .venv/bin/designgym serve --port 8027 --workers 4 > $W/results/env.log 2>&1 &
fi
ENV="${ENV_URL:-http://127.0.0.1:8027}"

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
  python - <<'PY'
import json, os, urllib.request
url = (os.environ.get("ENV_URL") or "http://127.0.0.1:8027") + "/api/episodes"
req = urllib.request.Request(url, data=json.dumps({"task_id": "test-00001", "skin": "figma"}).encode(),
                             headers={"content-type": "application/json"})
print("episode ok:", urllib.request.urlopen(req, timeout=120).read()[:200])
PY
  dump_logs; exit 0
fi

# the slim image has no CUDA toolkit: FlashInfer's sampler would JIT-compile with nvcc at warmup
export VLLM_USE_FLASHINFER_SAMPLER=0
serve() {  # name, model, gpu, port -> sets SERVE_PID
  CUDA_VISIBLE_DEVICES=$3 vllm serve "$2" --served-model-name "$1" --port "$4" --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --chat-template $W/chat_template.jinja \
    --limit-mm-per-prompt '{"image": 12}' --gpu-memory-utilization 0.92 > $W/results/vllm-$1.log 2>&1 &
  SERVE_PID=$!
  tail -n +1 -F $W/results/vllm-$1.log 2>/dev/null | grep --line-buffered -E "Loading|loaded|startup complete|Error|Traceback" \
    | sed -u "s/^/[vllm-$1] /" &
}
evaluate() {  # name, port: rollout.py for one served model, into results/rollouts-<name>
  local extra=()
  [ -n "${LIMIT:-}" ] && extra+=(--limit "$LIMIT")
  [ -n "${SKINS:-}" ] && extra+=(--skins "$SKINS")
  if [ -n "${RESUME_FROM:-}" ]; then  # keep that run's clean episodes; rollout.py reruns only the ones that errored
    mkdir -p $W/results/rollouts-$1
    python - "$1" <<'PY' || log "nothing to resume from runs/$RESUME_FROM"
import os, shutil, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["RESULTS_REPO"], f"runs/{os.environ['RESUME_FROM']}/rollouts-{sys.argv[1]}/trace.jsonl",
                    repo_type="dataset")
shutil.copy(p, f"/work/results/rollouts-{sys.argv[1]}/trace.jsonl")
PY
  fi
  VLLM_BASE_URL=http://127.0.0.1:$2/v1 VLLM_API_KEY=none .venv/bin/python rollout.py \
    --base-url "$ENV" --models "vllm:$1@off" --surfaces "${SURFACES:-gui,tools}" \
    --evalset $W/${EVALSET:-bench/evalsets/recreation-cua-v1.json} --subset "${SUBSET:-core30}" --concurrency "${CONCURRENCY:-12}" \
    --max-tokens 4096 --image-window 8,3 --out $W/results/rollouts-$1 "${extra[@]}" > $W/results/rollout-$1.log 2>&1
  log "$1: $(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d)[:600])" $W/results/rollouts-$1/summary.json 2>/dev/null || echo 'no summary')"
}

GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
[ "$GPUS" -ge 1 ] || { log "no GPU visible on this node (nvidia-smi: $(nvidia-smi -L 2>&1 | head -2))"; exit 1; }
log "$GPUS GPU(s); evaluating base${RL_MODEL:+ and $RL_MODEL} on ${SURFACES:-gui,tools} / ${SUBSET:-core30}"
if [ "${ONLY:-}" = "base" ]; then  # one model per job (the launcher's --split)
  serve base "$BASE_MODEL" 0 8000; wait_url http://127.0.0.1:8000/v1/models "vLLM base" 3600 $SERVE_PID
  evaluate base 8000 || true
elif [ "${ONLY:-}" = "rl" ]; then
  serve rl "$RL_MODEL" 0 8000; wait_url http://127.0.0.1:8000/v1/models "vLLM rl" 3600 $SERVE_PID
  evaluate rl 8000 || true
elif [ -n "${RL_MODEL:-}" ] && [ "$GPUS" -ge 2 ]; then
  serve base "$BASE_MODEL" 0 8000; BASE_PID=$SERVE_PID
  serve rl "$RL_MODEL" 1 8001; RL_PID=$SERVE_PID
  wait_url http://127.0.0.1:8000/v1/models "vLLM base" 3600 $BASE_PID
  wait_url http://127.0.0.1:8001/v1/models "vLLM rl" 3600 $RL_PID
  evaluate base 8000 & A=$!
  evaluate rl 8001 & B=$!
  wait $A $B || true
else
  serve base "$BASE_MODEL" 0 8000; wait_url http://127.0.0.1:8000/v1/models "vLLM base" 3600 $SERVE_PID
  evaluate base 8000 || true
  if [ -n "${RL_MODEL:-}" ]; then
    kill $SERVE_PID; sleep 15
    serve rl "$RL_MODEL" 0 8000; wait_url http://127.0.0.1:8000/v1/models "vLLM rl" 3600 $SERVE_PID
    evaluate rl 8000 || true
  fi
fi

log "uploading results"
upload
for f in $W/results/rollouts-*/summary.json; do [ -f "$f" ] || { log "an evaluation produced no summary"; exit 1; }; done
ls $W/results/rollouts-*/summary.json > /dev/null 2>&1 || { log "no evaluation produced a summary"; exit 1; }
log "done"
