"""One status line for an RL job launched with `launch.py rl`: job state, per-step metrics from the synced run dir,
trend, and the stop rules. Run it on a timer (every 10 minutes) while the run goes.

    .venv/bin/python hf_jobs/watch_rl.py <job_id> <run_name> [<model>]

The run dir is read from $DESIGNWORLD_RESULTS_REPO (the `--results-repo` the job was launched with) and the job from
$HF_JOBS_NAMESPACE (default: the token's user).

Stop rules (any one, printed as STOP):
- by step 15: the train-reward mean of the last 5 steps is not above that of the first 5, AND validation is not
  above its step-0 value;
- from step 20: two validation checks in a row at or below step 0;
- sampler/learner KL (kl_sample_train_v2) above 0.05 for 2 steps (off-policy instability);
- mixed-reward groups below 25% for 3 steps (no signal left to learn from).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from launch import _secret

REPO = os.environ.get("DESIGNWORLD_RESULTS_REPO")
PRICE = {"Qwen/Qwen3.6-35B-A3B": (0.54, 0.108, 1.335, 1.177), "Qwen/Qwen3.8-27B": (1.86, 0.372, 5.595, 4.103)}


def main(job_id: str, run: str, model: str = "Qwen/Qwen3.6-35B-A3B") -> None:
    if not REPO:
        raise SystemExit("set DESIGNWORLD_RESULTS_REPO to the results repo the RL job uploads to")
    token = _secret("HF_TOKEN")
    api = HfApi(token=token)
    info = api.inspect_job(job_id=job_id, namespace=os.environ.get("HF_JOBS_NAMESPACE"))
    state = str(info.status.stage).split(".")[-1]
    try:
        p = hf_hub_download(REPO, f"rl/{run}/metrics.jsonl", repo_type="dataset", token=token, force_download=True)
        rows = [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    except Exception:  # noqa: BLE001 - not synced yet
        print(f"job {state}; no metrics synced yet")
        return
    def step_of(r):  # sync runs log progress/batch; async runs log step / training_client/step
        return r.get("progress/batch", r.get("training_client/step", r.get("step", 0)))
    train = [(step_of(r), r) for r in rows if "env/all/reward/total" in r]
    evals = [(step_of(r), r["test/env/all/reward/total"]) for r in rows if "test/env/all/reward/total" in r]
    if not train:
        print(f"job {state}; eval only so far: {evals}")
        return
    step, last = train[-1]
    rew = [r["env/all/reward/total"] for _, r in train]
    mixed = [r.get("env/all/by_group/frac_mixed", 0) for _, r in train]
    kl = [r.get("optim/kl_sample_train_v2", 0) for _, r in train]
    p = PRICE[model]
    cost = sum((r.get("env/all/total_ob_tokens", 0) * p[1] + r.get("env/all/total_ac_tokens", 0) * (p[2] + p[3])) / 1e6
               for _, r in train)  # cached-prefill floor + sampled/trained actions; the full bill is ~1.3-1.6x this
    first5, last5 = statistics.fmean(rew[:5]), statistics.fmean(rew[-5:])
    print(f"job {state} | step {step} | train {rew[-1]:.3f} (first5 {first5:.3f} -> last5 {last5:.3f}) | "
          f"mixed {mixed[-1]:.0%} | turns {last.get('env/all/turns_per_episode', 0):.0f} | "
          f"tool err {last.get('env/all/openenv/tool_errors', 0):.2f} | KL {kl[-1]:.4f} | "
          f"eval {' '.join(f's{s}:{v:.3f}' for s, v in evals)} | staleness {last.get('sampling_client/step_mean', step):.1f}->{step} | "
          f"cost floor ${cost:.0f}")
    stop = []
    e0 = evals[0][1] if evals else None
    if step >= 15 and last5 <= first5 and (not evals or evals[-1][1] <= e0):
        stop.append("no upward trend in train (last5 <= first5) or eval by step 15")
    if step >= 20 and len(evals) >= 3 and all(v <= e0 for _, v in evals[-2:]):
        stop.append("two eval checks in a row at or below step 0")
    if len(kl) >= 2 and all(k > 0.05 for k in kl[-2:]):
        stop.append("sampler/learner KL > 0.05 two steps running")
    if len(mixed) >= 3 and all(m < 0.25 for m in mixed[-3:]):
        stop.append("under 25% mixed-reward groups three steps running")
    if stop:
        print("STOP: " + "; ".join(stop))


if __name__ == "__main__":
    main(*sys.argv[1:])
