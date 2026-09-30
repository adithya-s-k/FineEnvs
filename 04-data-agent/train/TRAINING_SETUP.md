# Four-harness training setup — 2026-09-15

The base-model evaluation is complete: **1,000 graded pass@1 evaluations**, with
**14.6% overall pass@1**. The [baseline report](../eval/BASELINE_PASS_AT_1.md) and
[fixed protocol](../eval/baseline_protocol.json) are the checkpoint comparison reference.

The current continuation is **78956**, resuming **checkpoint 53 of job 78831**.
Qwen3.5-2B uses **G8 / GAS4 / staleness4 / LR3e-6**, a **40,960-token packing
target**, exact rows up to **131,072 tokens**, and compiled vLLM.
The worker concurrency ceiling is **32**, with **16 total outstanding rollouts**
across generation, scoring and the ready queue. Thus actual active rollouts are
also bounded by 16. Complete eight-generation groups reserve capacity together;
credits are released after optimizer consumption or whole-rollout rejection.
Each packing unit admits at most two rollouts so accumulation leaves room for a
complete new group. Every admitted rollout still finishes all its forked rows
before the optimizer updates. Loss is one supervised-token mean over the update.

The reference blackbox OpenCode job **76585** completed 400 steps in 3h09m30s;
its first 100 measured update intervals total 46.2 minutes, excluding cold
startup. Four-harness phase **78831** matched its G8/I32/GAS4/40960 settings
and reached checkpoint 53 from 30 in 37m24s, with 23 updates (19 nonzero gradients),
mean measured step 79.28s and zero admission-receipt mismatches. However, an
unbounded ready queue caused **1,230 stale rows** to be rejected. The current
phase adds producer backpressure to address that waste. Matching reference
batch parameters alone did not establish efficient multi-harness training.

Validation: 14 recipe tests including native spawn IPC, full-group reservation,
actual-dispatch version tagging, exact stale counters, short-row deadlock
regression and native loss/gradient invariance; a further four targeted checks
include the real CPU HF optimizer loop with credits enabled and disabled.
**Checkpoint 100 is complete**, with model/optimizer/scheduler/RNG files and a
validated native resume cursor (114 / model version 105). Training continued
beyond it. Across the 47 updates since checkpoint 53, **246/246 admitted rollouts**
match independent capture row/token/group counts: Codex 56, Claude Code 62,
Mini-SWE-Agent 64 and OpenCode 64. There are zero stale drops and no nonfinite
loss, gradient or importance-ratio values; maximum staleness was 3 and
outstanding work 16. Thirty-four updates had nonzero gradients. The mean
measured update interval was 99.83 seconds, with substantial context-size variation.

One Mini-SWE-Agent rollout failed after the model deleted `/workdir`. Its capture
passed TiTO and its missing reward received zero advantage, excluded from the
group reward baseline. The 1,025 zero-advantage tokens remain in that update's 21,032-token normalizer under the native loss semantics. This is an agent-action
failure, not a capture failure; it was not converted into a negative reward.
The native scorer was replayed with the observed group rewards to verify this.

Full evaluation **78987** is running on `ip-10-53-85-89`, excluding training node
`ip-10-53-83-253`: two H100s, **TP1/DP2**, BF16, context 131072, and **100 slots**
for client/server/sandbox concurrency. The live vLLM model path points to staged
checkpoint 100. All eight initial rollouts passed TiTO; the full fixed 1,000-cell
pass@1 evaluation has started, and its score is pending. The checkpoint weights
also differ from checkpoint 89 in an inspected text-layer slice, verifying actual
optimizer writes. Monitoring continues at the established ten-minute cadence.

The E2B template preinstalls Node 22.23.2/ripgrep. Bounded SDK streaming gzip
uploads reproduced identical remote checksums for the two previously stalled
272–291 MB logs in about ten seconds. These operational improvements apply to
future evals as recorded; tasks, harness pins, sampling and scoring remain fixed.
Inference GPU memory utilization is **0.85** to leave headroom for NCCL transfers;
eval does not share this inference GPU or use training weight transfers.

Whole stale rollouts can still be rejected. Equal per-rollout weighting and
complete replay of partially consumed groups across resume are not claimed.
The restored native cursor is 85 / model version 57; queues are regenerated, so
later groups consumed before a cursor gap can repeat. The parent exited cleanly,
its cleanup verified zero owned sandboxes, and no handoff evaluation was submitted.

## Training task mix

| Difficulty | Tasks | Share |
| --- | ---: | ---: |
| Easy | 150 | 15% |
| Medium | 600 | 60% |
| Hard | 250 | 25% |
| Total | 1,000 | 100% |

Use **one harness per task per pass**, with exactly 250 task groups per harness.
Eight generations form each GRPO group, all using the same task and harness. One
pass requests **1,000 groups / 8,000 rollouts**; rollouts are not optimizer steps. Preserve
the first 32 easy tasks in pass one, then shuffle within difficulty-balanced harness
buckets (seed 0) and interleave the harnesses. Each later pass rotates a task to
the next harness and shuffles all tasks. The frozen four-pass cycle covers every
task/harness pair once (4,000 groups / 32,000 rollouts); it repeats thereafter.
The 1,000-task pool is a coverage goal, not a promise of full coverage within 24 hours.

Selection screens actual cached contents against the fixed test set by source ID,
notebook, normalized question and normalized instruction, then excludes duplicate
training instructions. All four overlap counts are zero. Original source cache
revision is unknown; original and effective file hashes are recorded rather than
claiming an unverified upstream revision.

Frozen dataset, dispatch order, explicit pairs and executable run configuration:

- [Run configuration](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-bounded-20260915/run_config.json)
- [Task manifest and file hashes](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-bounded-20260915/manifest.json)
- [Executable rotation schedule](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-bounded-20260915/harness_schedule.json)
- [Ordered task/harness pairs](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-bounded-20260915/pairs.jsonl)
- [Reproducible selection tool](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/tools/select_multi4_training.py)

The fixed test mix remains **33 easy / 118 medium / 99 hard**. Training rebalancing
does not change the baseline or checkpoint evaluation denominator.

## Active training configuration

Eight generations form one task/harness group. The worker ceiling is 32, with 16 total generating or queued rollouts.
GAS4 counts four packing units of complete rollouts, not necessarily four physical
GPU forwards. The actual forward count and token throughput remain logged.

| Setting | Value |
| --- | --- |
| Initialization | Checkpoint 53 of run 78831; original Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Learning rate | `3e-6` |
| Generations / worker ceiling / total outstanding | 8 / 32 / 16 |
| Gradient accumulation / per-device batch setting | 4 / 4; whole-rollout packing determines actual forward count |
| Optimizer / dtype | Paged AdamW 8-bit / BF16 |
| Context limit / packing target | 131,072 / 40,960; exact long rows get dedicated forwards |
| Maximum completion | 16,384 tokens per call (training); fixed eval remains 4,096 |
| Sampling | Temperature 0.8, top-p 1.0, top-k disabled, thinking disabled |
| Agent budget | 17 model calls, 600 seconds |
| Sandbox | Shared E2B template, 1 CPU, 4 GiB |
| Training resources | Two H100 80 GB GPUs: one trainer, one inference engine |
| Allocation / soft stop | Continuation: 16-hour allocation; 55,000 training seconds (about 15.3 hours) |
| Optimizer-step ceiling / checkpoint interval | 1,000 / 50, plus first optimizer boundary after 3,600 seconds since the last save |
| TiTO | Exact engine IDs, behavior logprobs and masks; lossless history forks, threshold 0 |
| Whole-rollout staleness limit | 4 optimizer versions |

This training path remains text-only. Permanent input errors (including unsupported image requests) return 4xx promptly. Explicitly synthetic zero-token Claude API error notices are excluded from model-call reconciliation; unknown mismatches and captures with no model calls remain rejected.

The H100 memory probe passed a real 50,420-token row and a 131,072-token stress
row with 8,192 supervised positions, four accumulation microbatches and optimizer
steps. Peak reserved memory was **46.66 GiB / 79.18 GiB**. This establishes capacity
for the tested kernels and shape, not all possible training shapes or queue behavior.

Harness versions match the baseline: OpenCode **1.18.31**, Claude Code **2.1.270**,
Codex **0.154.0**, mini-swe-agent **2.4.6**, passed through Harbor's existing installer
version option. The initial phase started from original base weights; the current phase restores checkpoint 53 and its optimizer. The factory applies the saved dataset cursor to local group IDs so task/harness routing remains aligned after resume.

## Diagnostic and verifier correction

Diagnostic 78282 was stopped before an optimizer update. The cached training grader
emitted `tool_efficiency: null` without tool-count metadata; Harbor rejected otherwise
valid results. Its cleanup job 78283 completed with zero remaining owned sandboxes.
The prepared copies now reuse the existing test-set grader and test script. Shared
grading function ASTs match all 1,000 originals; correctness scoring, gold answers
and tolerances are unchanged. The unavailable tool-efficiency field is omitted.
Correct, wrong and empty submissions passed all 15 checks on the five diagnostic tasks.

Replacement diagnostic 78291 uses five clean easy tasks × four harnesses, four
generations per group, in-flight four, 20 minimum / 40 maximum optimizer steps and
checkpoints every ten steps. Coverage means at least one row per pair, stable over
two optimizer boundaries. It does **not** prove every forked row was consumed.
Logs and memory evidence are under:
[diagnostic artifacts](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-diagnostic-v2-20260914/job-78291).

## Independent checkpoint evaluations

[checkpoint_evals.py](../eval/checkpoint_evals.py) queues completed checkpoints at
**100, 200, 300, ... optimizer steps**. Saves occur at **50, 100, 150, ...**, with all
checkpoints retained for later evaluation. Hourly recovery saves add off-interval checkpoints without stopping training or triggering an intermediate evaluation. The final completed checkpoint is also evaluated, including an off-interval save at the wall-time limit. A directory alone is insufficient: the callback
checks full training-state presence and safetensor headers, then publishes a small
`checkpoint.saved.json` handoff. The CPU watcher performs weight hashing and staging;
these reads do not block the training process. The evaluator stages the trained weights
and config, filling only missing tokenizer/processor assets from the pinned base revision.
Checkpoint retention is unlimited while evaluations may still reference their weights.

Each eligible checkpoint queues a separate **two-H100, TP=1 / DP=2** vLLM job, one endpoint,
**concurrency 100**, 250 fixed test tasks × four harnesses × pass@1. Only one eval job
is active at a time; later checkpoints wait without pausing training. Eval GPU jobs
explicitly exclude the training node and start their own OpenEnv server and capture
proxy. FSx and the E2B account remain shared, so absolute zero resource contention
is not guaranteed. The user confirmed a **600-sandbox account quota**: the planned
100 eval + 16 training rollouts use 116 slots, leaving 484 before other workloads. While both jobs
run, total allocation is four GPUs and up to 116 rollouts. The watcher exits after
training finishes and its queue drains; the CPU allocation has a 48-hour ceiling.

Evaluation sources and protocol are frozen once per series. Every GPU job has a
dependent owned-sandbox cleanup job. Results retain the first graded attempt, including
zeros; infrastructure retries do not turn pass@1 into best-of-k. Reports include per-harness
and per-difficulty scores, deltas from the baseline, full TiTO audit status, and observed
harness version agreement. Incomplete coverage has no full-set headline score. A changed
harness version prevents the report from being marked ready for comparison.

Validation covers the actual TRL repeated-row iterator against all four scheduled
passes, checkpoint save/handoff integrity, 100-step eligibility, retry denominators,
training-node exclusion and monitor alerts. A bounded checkpoint-serving smoke
(job **78432**, cleanup **78433**) loads clean diagnostic checkpoint 10 on a separate
node with TP=1/DP=2 and two test tasks per harness. It completed successfully:
**8/8 graded, 8/8 TiTO**, 28,212/28,212 eligible tokens retained; native harness
versions match the baseline. Both GPU and cleanup jobs completed; zero owned
sandboxes remain. This eight-cell check is not a full pass@1 measurement and does
not replace the fixed baseline or validate the combined 116-rollout load.

Inspect the prepared submission without starting jobs:

```bash
.venv312/bin/python experiments/async_grpo_harbor_data_agent/tools/submit_multi4_long.py \
  --run /fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-bounded-20260915
```

Adding `--submit` starts the training job, its dependent cleanup, a CPU watcher
that submits checkpoint evals, a CPU monitoring job, and a separate CPU Trackio sync job.
The durable `submission.json` is the authority for whether the run has been submitted.
Submission IDs are persisted. If submission is interrupted, reconcile the saved IDs
with Slurm before retrying; the launcher refuses an automatic second training job.

To inspect checkpoint eligibility without submission:

```bash
.venv312/bin/python HuggingEnvs/04-data-agent/eval/checkpoint_evals.py \
  --checkpoints /path/to/training/job/run --output /path/to/checkpoint-evals --interval 100
```

## Monitoring

The prepared CPU watchdog checks every **two minutes during startup**, then every
**10 minutes** at step 63 or later and two healthy, progressing checks. It audits newly
captured rollouts and records Slurm liveness, finite gradients, stale/oversize drops,
TiTO retention, task/harness coverage, checkpoints, eval results and backlog.
Status, history and changing alerts are written under the run’s `monitor/` directory.
It never kills or modifies training automatically. The watcher and monitor have
48-hour CPU allocations so evaluations can drain after training stops.

Automatic Codex chat wakeups are **not enabled**: this session exposes no scheduling
tool. `MONITORING_PROMPT.md` supplies a prepared review prompt; the CPU watchdog is
the implemented recurring monitor, with local artifacts rather than chat delivery.

## Offline and online logging

See [LOGGING.md](LOGGING.md). Every optimizer update remains in `audit/metrics.jsonl`
and native Trackio JSONL fragments under `job-<id>/trackio`. A separate CPU job
replays scalar metrics into a node-local Trackio database, backs it up to FSx, and
syncs to the private `AdithyaSK/multi4-qwen35-2b-trackio` Space every 60 seconds
when data changes. Network failures retry independently of training and evals.
The evaluation curve includes the baseline at step 0 and complete, comparable
1,000-cell pass@1 evaluations at their actual checkpoint step.

## Remaining training checks

[TRL issue #7206](https://github.com/huggingface/trl/issues/7206) describes the
upstream normalization/admission problem. This single-GPU recipe supplies local
whole-rollout admission and update-token normalization using the native scorer,
collator and loss. It does not modify upstream TRL to support every configuration.
`audit/optimizer_rollouts.jsonl` records completed row counts at each actual
optimizer boundary; capture TiTO audit remains a separate check. Confirm live
nonzero gradients, importance ratios, whole-rollout stale drops, all-four-harness
admission and sustained step timing before declaring the continuation stable.
