# Hard-task continuation on HF Jobs

The [copyable Python configuration](configs/hard500_trl.py) contains the TRL, harness, serving, evaluation and HF resource settings. This is a continuation from multi-harness checkpoint 500: 500 screened hard tasks, two task passes, eight rollouts per group. The original checkpoint and experiment directories stay unchanged.

Training uses `h200x2` (one optimizer GPU, one vLLM GPU). Independent checkpoint evaluations use `a100-large`, TP1/DP1, concurrency 50, the fixed 250 tasks across all four harnesses, pass@1. Save every 50 updates; queue evaluations every 100 updates and at the end. A separate CPU Job coordinates evaluations. Each GPU Job runs its own Harbor server and E2B capture tunnel; existing Spaces are unchanged.

The published HF rates checked September 17 are $10/hour for `h200x2`, $2.50/hour for `a100-large`, and $0.03/hour for `cpu-upgrade`. Hardware listings do not establish immediate allocation availability. [HF pricing](https://huggingface.co/docs/hub/jobs-pricing).

## Preparation and launch

Run from the workspace root. Preparation and preview do not upload anything or allocate Jobs:

```bash
python HuggingEnvs/04-data-agent/hf/hard_curriculum.py prepare --out /path/to/new/hf-bundle
python HuggingEnvs/04-data-agent/hf/hard_curriculum.py preview --out /path/to/new/hf-bundle --phase smoke
python HuggingEnvs/04-data-agent/hf/hard_curriculum.py preview --out /path/to/new/hf-bundle --phase long
```

The bundle freezes code, dependency locks, both task catalogs and schedule hashes. The separate 11.3 GB parent checkpoint includes optimizer, scheduler and RNG state. Runtime restoration verifies every file and resets the task cursor only in the downloaded copy. Credentials are passed through HF secrets, not included in the archive or request previews.

After launch approval:

```bash
python HuggingEnvs/04-data-agent/hf/hard_curriculum.py upload --out /path/to/hf-bundle
python HuggingEnvs/04-data-agent/hf/hard_curriculum.py submit --out /path/to/hf-bundle --phase preflight
python HuggingEnvs/04-data-agent/hf/hard_curriculum.py submit --out /path/to/hf-bundle --phase smoke
```

After the smoke completes, record its Job ID in `qualification.json` inside the bundle directory, as `{"job_id":"COMPLETED_SMOKE_JOB_ID"}`. Long submission downloads and verifies that Job's qualification artifact and exact runtime hash:

```bash
python HuggingEnvs/04-data-agent/hf/hard_curriculum.py submit --out /path/to/hf-bundle --phase long
```

Submission records are written before API calls. An ambiguous result requires reconciliation; repeating the command does not allocate another trainer. If the coordinator submission fails after the trainer starts, the saved trainer ID remains authoritative; reconcile that record rather than launching another trainer.

## Epoch and resume semantics

The finite worker dispatches 1,000 groups, drains their scored rollouts and saves a full final checkpoint. `max_steps=2501` is only a safety ceiling including parent step500. It is not the requested duration. The standard infinite async worker must not be substituted for this worker.

On a later resume, checkpointed admission records skip groups with committed optimizer work, including groups after an earlier gap. Partially consumed groups are not replayed; their unconsumed tails are abandoned and recorded. Groups with no committed work can be regenerated. Consequently, 8,000 is the requested count; failed/stale/discarded work and admitted rollouts must be reported separately. This is not exact restoration of the in-memory asynchronous queue.

GPU qualification checks two optimizer updates, full remote checkpoint restoration, continuation to the finite boundary, captured tokens against optimizer receipts, a nonzero fresh gradient, changed weights and eight evaluation cells. Qualification is separate from production; production starts again from the untouched parent checkpoint 500.

Local tests cover finite dispatch, partial packing and accumulation, resume gaps and atomic rollout preservation. A live HF GPU smoke is still required before describing this runtime as qualified. See [qualification status](QUALIFICATION_20260917.md) for smoke jobs and results.

## Logs and evidence

Every optimizer update is retained in `audit/metrics.jsonl`: rewards, loss, gradient norm, staleness, tool calls, rollout forks, supervised/context tokens and timings. `audit/optimizer_rollouts.jsonl` records which rollouts actually reached an optimizer update. Checkpoints retain curriculum admission state so dispatched work is distinguishable from consumed work.

For resumed-run speed, use `perf/step_s` and the forward/backward timings. The trainer's final aggregate `train_steps_per_second` can include inherited global steps and is not a valid continuation throughput estimate.

Trackio keeps an offline database and backup. A separate process replays scalar metrics to [the comparison dashboard](https://huggingface.co/spaces/HuggingEnvs/data-agent-training-comparison-trackio); `trackio_verified.json` records successful remote readback. Smoke data uses project `qwen35-2b-harbor-vs-opencode-20260916-smoke`, with a distinct run name for each bundle attempt. Production uses the same project name without `-smoke`. Dashboard sync does not block optimizer updates.

The artifact publisher runs every 60 seconds. Each attempt has its own owner prefix under `<run-id>/jobs/` in the configured HF bucket. Retained evidence includes service/trainer logs, exit codes, immutable bundle hashes, checkpoint publication records, eval cell results, qualification checks and sandbox cleanup. Raw captures stay in the experiment artifacts; the dashboard receives scalar summaries.

During qualification, `qualification_events.jsonl` and `smoke-evidence/<job-id>/monitor.jsonl` in the local experiment directory preserve timestamped operations and observations. The [qualification report](QUALIFICATION_20260917.md) summarizes failures, fixes, measured performance and outstanding gates; earlier attempts are preserved.

## Planned stop at checkpoint 1,000 — September 18

The user requested stopping this continuation after checkpoint 1,000 is saved and
finishing the checkpoint evaluations. HF control job `6aad02a7b1dc2b62dc59115e`
polls the uploaded completion manifest every 30 seconds, downloads the full checkpoint,
verifies every recorded file hash and trainer step, then cancels training job
`6aac5eedb1dc2b62dc58f6b0`. It will not cancel based on a local save announcement.
Training may advance a few steps while upload and verification finish; checkpoint
1,000 remains the intended endpoint. The control job is bounded to 26 hours and its
watch loop to 24 hours.

Control receipts are published under the run's `control-stop-1000/` bucket prefix.
Coordinator `6aad02a7b1dc2b62dc591160` permits two concurrent eval jobs, each on four
A100s with TP1/DP4 and concurrency 100. Checkpoint 1,000 has first priority once ready;
otherwise newer pending checkpoints are selected first. It continues after a training
cancellation only when the matching verified-stop receipt is present. Checkpoints
above 1,000 are excluded. The running checkpoint-600 eval was left in place.

Submission receipts and exact overlay hashes are recorded in the run's
`stop1000-parallel-evals-20260918/` directory. Throughput at this configuration is
still being measured; there is no new accepted benchmark score yet.
