#!/usr/bin/env bash
# HF Job (radixark/miles:latest, one node): Miles GRPO on DesignGym GUI episodes.
# DesignGym runs here (common/job_common.sh); the model is served by SGLang and trained by Megatron-Bridge.
# Env: see common/job_common.sh, plus TRAIN_ARGS (train_designgym.py options). Secret: HF_TOKEN.
set -euo pipefail
python3 - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
tarfile.open(hf_hub_download(os.environ["BUNDLE_REPO"], os.environ["BUNDLE_PATH"], repo_type="dataset")).extractall("/work")
PY
source /work/rl/common/job_common.sh
trap 'rc=$?; log "exit $rc"; grep -E "step [0-9]+:|perf [0-9]+:" $OUT/train.log > $OUT/steps.log 2>/dev/null || true; nvidia-smi > $OUT/nvidia-smi.txt 2>&1 || true; upload_results' EXIT
gpu_watch
thread_watch
nvidia-smi -L
# Some HF Jobs nodes (AWS, kernel *-amzn2023) cap a container at ~5k threads while reporting 192 CPUs; numerical and
# tokenizer pools otherwise start one thread per CPU in every process (SGLang, Megatron, env workers).
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 RAYON_NUM_THREADS=4 \
       TOKENIZERS_PARALLELISM=false
# JIT kernel builds (SGLang's sgl_kernel JIT, torch extensions) run ninja with one job per CPU: at ~4.5k threads that
# hit the cap ("Failed to build JIT module ... can't start new thread"); Ray needs ~20 CPUs for Miles' actors
export MAX_JOBS=${MAX_JOBS:-4} TORCHINDUCTOR_COMPILE_THREADS=1 RAY_CPU_CAP=${RAY_CPU_CAP:-16}

# Ray preflight (~20 s per try). On some HF Jobs nodes (AWS, kernel *-amzn2023) Ray's Python worker processes crash
# while starting ("worker process ... is dead, probably it crashed during start"), so nothing that needs a worker
# runs (the job agent's supervisor, Miles' actors). Ray prestarts one worker per CPU when a driver connects; try
# the default first, then without prestart (RAY_enable_worker_prestart=0, inherited by Miles' `ray start`).
# On failure: dump node limits and the workers' own logs, and stop before the model merge.
ray_try() {  # ray_try <label> [env prefix]: start Ray on 127.0.0.1 and run one task
  ray stop --force > /dev/null 2>&1
  ${2:-} ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats > $OUT/ray-preflight-$1.log 2>&1 || return 1
  timeout 90 ${2:-} python3 -c "import ray; ray.init(address='auto'); print('ray preflight task ->', ray.get(ray.remote(lambda: 1).remote()))" \
    >> $OUT/ray-preflight-$1.log 2>&1
}
ray_diag() {
  echo "--- node: nproc=$(nproc) ulimit -u=$(ulimit -u) pids.max=$(cat /sys/fs/cgroup/pids.max 2>/dev/null) pids.current=$(cat /sys/fs/cgroup/pids.current 2>/dev/null) mem.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null) shm=$(df -h /dev/shm | tail -1)"
  free -g | head -2
  python3 -c "import ray; print('ray', ray.__version__, 'cpus', __import__('os').cpu_count())" 2>&1 | tail -1
  echo "--- cpu: $(lscpu | grep -m1 'Model name' | sed 's/  */ /g') | avx512f=$(grep -c -m1 avx512f /proc/cpuinfo)"
  local d=/tmp/ray/session_latest/logs
  grep -h "crashed during start\|hanging during start" $d/raylet.err 2>/dev/null | head -2
  echo "--- import the worker module directly (faulthandler)"
  timeout 60 python3 -X faulthandler -c "import ray._private.workers.default_worker; print('default_worker import ok')" 2>&1 | tail -15
  echo "rc=${PIPESTATUS[0]}"
  # the abort reason is at the START of a worker's .err (the stack trace follows it)
  for f in $(ls -S $d/worker-*.err 2>/dev/null | head -1); do echo "--- $f (head)"; grep -v "logging.cc:.*@ " $f | head -12; done
  echo "--- gcs_server / raylet errors"
  grep -h -i -E "error|fail|check" $d/gcs_server.out $d/gcs_server.err $d/raylet.out 2>/dev/null | grep -v -i "event stats" | head -12
  echo "--- thread limits: $(ulimit -a | grep -E 'stack|processes|virtual|max locked' | tr -s ' ' | paste -sd ';') | threads-max=$(cat /proc/sys/kernel/threads-max) max_map_count=$(cat /proc/sys/vm/max_map_count) overcommit=$(cat /proc/sys/vm/overcommit_memory) cgroup threads=$(cat /sys/fs/cgroup/pids.current 2>/dev/null)"
  echo "--- proxy variables (credentials masked)"
  env | grep -i -E "^(https?|grpc|all|no)_proxy=" | sed -E 's#//[^/@]*@#//***@#' || echo "none"
  echo "--- ray address files"; cat /tmp/ray/ray_current_cluster 2>/dev/null; hostname -I 2>/dev/null
}
LIMITS=/work/rl/miles/raylimits.sh  # Ray capped at ${RAY_CPU_CAP:-32} CPUs (raybin/ray shim), fewer threads
thread_probe() {  # how many threads this container can really create (EAGAIN cap), with small stacks
  timeout 60 python3 -c '
import threading, os
threading.stack_size(256 * 1024)
ev, ts = threading.Event(), []
try:
    while len(ts) < 20000:
        t = threading.Thread(target=ev.wait); t.start(); ts.append(t)
    msg = "no cap hit"
except RuntimeError as e:
    msg = "EAGAIN: %s" % e
n = len(ts); ev.set()
for t in ts: t.join()
pc = "/sys/fs/cgroup/pids.current"
print("thread probe: created %d extra threads (%s); cgroup pids.current after: %s"
      % (n, msg, open(pc).read().strip() if os.path.exists(pc) else "?"))
' 2>&1 | tail -1
}
ray_preflight() {
  local probe; probe=$(thread_probe); log "$probe"
  # REJECT_CAPPED_NODES=1: leave a node with a thread cap (AWS, ~5k) at once; the launcher retries elsewhere
  if [ "${REJECT_CAPPED_NODES:-0}" = 1 ] && echo "$probe" | grep -q "EAGAIN"; then
    log "BAD NODE: thread cap (${probe#thread probe: }); REJECT_CAPPED_NODES=1, stopping before any download"
    return 1
  fi
  if ray_try limits "$LIMITS"; then
    ray stop --force > /dev/null 2>&1; RAY_ENV="$LIMITS"; log "ray preflight ok with Ray capped at ${RAY_CPU_CAP:-32} CPUs"; return 0
  fi
  log "ray preflight FAILED (Ray capped)"; tail -3 $OUT/ray-preflight-limits.log
  if ray_try default; then ray stop --force > /dev/null 2>&1; log "ray preflight ok"; return 0; fi
  log "ray preflight FAILED (default)"; tail -3 $OUT/ray-preflight-default.log; ray_diag
  export RAY_enable_worker_prestart=0
  if ray_try noprestart; then ray stop --force > /dev/null 2>&1; log "ray preflight ok with RAY_enable_worker_prestart=0"; return 0; fi
  log "ray preflight FAILED (no prestart)"; tail -3 $OUT/ray-preflight-noprestart.log; ray_diag
  unset RAY_enable_worker_prestart
  # the workers abort creating their gRPC threads (EAGAIN) at ~4k threads: cap Ray's CPUs (fewer prestarted workers)
  if ray_try limits "$LIMITS"; then
    ray stop --force > /dev/null 2>&1; RAY_ENV="$LIMITS"; log "ray preflight ok with Ray capped at ${RAY_CPU_CAP:-32} CPUs"; return 0
  fi
  log "ray preflight FAILED (Ray capped at ${RAY_CPU_CAP:-32} CPUs)"; tail -3 $OUT/ray-preflight-limits.log; ray_diag
  ray stop --force > /dev/null 2>&1
  return 1
}
RAY_ENV=""  # set by the preflight when Ray only works with $LIMITS; applied to the trainer (and so to Ray) only
ray_preflight || { log "BAD NODE: Ray workers crash at start on $(uname -r); stopping early"; exit 3; }
# CUDA must initialise on all GPUs (some nodes return Error 802 "system not yet initialized": fabric manager not up)
# right after a node boots the fabric manager can need a few minutes: retry for ~5 min before giving up
cuda_ok=0
for try in $(seq 1 10); do
  if timeout 120 python3 -c "import torch; n = torch.cuda.device_count(); assert n >= 1, n; [torch.zeros(1, device=f'cuda:{i}') for i in range(n)]; print('cuda ok:', n, 'GPUs')" \
       > $OUT/cuda-check.log 2>&1; then cuda_ok=1; break; fi
  log "CUDA not ready (try $try/10: $(grep -o 'Error [0-9]*[^.]*' $OUT/cuda-check.log | head -1)); retrying in 30 s"
  sleep 30
done
[ "$cuda_ok" = 1 ] && log "$(tail -1 $OUT/cuda-check.log) (try $try)" \
  || { log "BAD NODE: CUDA does not initialise after ~5 min ($(grep -o 'Error [0-9]*[^.]*' $OUT/cuda-check.log | head -1)); stopping early"; exit 3; }

make_tasks
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("/work/data/tasks.jsonl")]
with open("/work/data/miles.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"prompt": "designgym", "metadata": r}) + "\n")
PY

start_env > $OUT/start-env.log 2>&1 &
ENV_PID=$!
download_model
wait $ENV_PID || { cat $OUT/start-env.log; exit 1; }
cat $OUT/start-env.log
DESIGNGYM_URL=${ENV_URL:-http://127.0.0.1:8027}

# our package for every Python process of the run (Ray workers included)
echo /work > "$(python3 -c 'import site; print(site.getsitepackages()[0])')/designgym_rl.pth"
python3 -c "import websockets" 2>/dev/null || pip install -q websockets
python3 -c "import rl.common.episode, rl.miles.designgym_generate; print('designgym adapter imports ok')"

log "smoke-checking the env from this Python"
python3 - <<PY
import asyncio, json
from rl.common.env_client import EnvSession
from rl.common.episode import EpisodeConfig, _task_reset_kwargs
task = json.loads(open("/work/data/tasks.jsonl").readline())
async def main():
    async with EnvSession("$DESIGNGYM_URL".split(",")[0]) as s:
        m = await s.reset(**_task_reset_kwargs(task, EpisodeConfig(max_turns=5)))
        shot = await s.call("screenshot", {})
        print("env ok:", m["task_id"], len(shot.images), "image(s)")
asyncio.run(main())
PY

# long runs: every 15 min copy new LoRA checkpoints into $OUT and upload logs + checkpoints, so a crash or timeout
# loses at most the last 15 minutes (the EXIT trap still does a final upload)
(set +e  # job.sh runs with set -e: one failing command (e.g. a grep with no match yet) would end this loop
 while sleep 900; do
   for d in $(ls -d /work/ckpt/iter_*/adapter 2>/dev/null); do
     it=$(basename "$(dirname "$d")")
     [ -d "$OUT/ckpt/$it" ] || { mkdir -p "$OUT/ckpt/$it" && cp -r "$d" "$OUT/ckpt/$it/"; }
   done
   grep -E "step [0-9]+:|perf [0-9]+:" $OUT/train.log > $OUT/steps.log 2>/dev/null || true
   upload_results > $OUT/periodic-upload.log 2>&1 || true
   echo "periodic upload done $(date +%H:%M:%S)"
 done) &

log "training"
cd /root/miles
set +e
for attempt in 1 2 3; do  # a Ray job submit can race the job agent at startup (HTTP 500, nothing trained): retry
  $RAY_ENV python3 /work/rl/miles/train_designgym.py --model "$MODEL" --hf-dir "$HF_DIR" --data /work/data/miles.jsonl \
    --save /work/ckpt --env-url "$DESIGNGYM_URL" $TRAIN_ARGS 2>&1 | tee -a $OUT/train.log
  RC=${PIPESTATUS[0]}
  [ "$RC" = 0 ] && break
  grep -q "step 0:" $OUT/train.log && break
  grep -q "Request failed with status code" $OUT/train.log || break
  log "Ray job submit failed before training (attempt $attempt); restarting Ray"
  for f in dashboard_agent.log dashboard.log; do  # why the job agent refused (crash vs not up yet)
    [ -f /tmp/ray/session_latest/logs/$f ] && { echo "--- $f"; tail -40 /tmp/ray/session_latest/logs/$f; }
  done
  ray stop --force > /dev/null 2>&1; sleep 20
done
set -e
log "trainer exited $RC"
# LoRA adapters (small) and the checkpoint layout; full Megatron states stay on the node
if [ -d /work/ckpt ]; then
  find /work/ckpt -maxdepth 3 | head -100 > $OUT/ckpt-tree.txt
  du -sh /work/ckpt/* >> $OUT/ckpt-tree.txt 2>/dev/null || true
  d=$(ls -d /work/ckpt/iter_*/adapter 2>/dev/null | sort | tail -1)  # the newest adapter only (upload size)
  [ -n "$d" ] && mkdir -p $OUT/ckpt/$(basename $(dirname $d)) && cp -r $d $OUT/ckpt/$(basename $(dirname $d))/
fi
grep -q -E "step [0-9]+:" $OUT/train.log || { log "no training step logged"; exit 1; }  # step 0 is the first
exit $RC
