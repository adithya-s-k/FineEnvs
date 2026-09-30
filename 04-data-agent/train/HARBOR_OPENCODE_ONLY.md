# Harbor OpenCode-only training ablation

Trainer **81075** started on **hopper-prod at 09:20:27 UTC, 2026-09-16**. Startup verification at 09:27 UTC passed: optimizer steps 1–4 had finite scalars and nonzero gradients, with maximum observed staleness 3. All 20 rollouts consumed by those updates passed independent TiTO replay; every admitted row and all 10,656 supervised tokens were retained. The first 24 completed rollouts passed TiTO overall. This verifies startup, not the full training run; checkpoint 50 and evaluation 100 are still ahead.

This is a fresh Qwen3.5-2B run through **OpenEnv → Harbor → OpenCode**, using E2B. It uses the same trainer, atomic rollout batching and numerical recipe as the final stable multi-harness allocation 80608, with OpenCode as the only training harness. It does not use the standalone blackbox-opencode implementation. It starts from the pinned base weights, not from an already-trained checkpoint.

| Setting | Value |
| --- | --- |
| Base model | Qwen/Qwen3.5-2B, revision 15852e8c16360a2fea060d615a32b45270f8a8fc |
| Training harness | OpenCode 1.18.31 through Harbor |
| Sandbox | E2B; 1 vCPU, 4 GiB per sandbox |
| Training data | Same frozen 1,000 tasks: 150 easy / 600 medium / 250 hard |
| Task order | Exact first pass from the reference; first 32 tasks easy; every assignment becomes OpenCode |
| Optimizer-step target | 1000 |
| LR / optimizer / precision | 3e-6 / paged_adamw_8bit / bfloat16 |
| Generations / staleness | 8 per task / maximum 4 |
| Worker concurrency / outstanding budget | 32 workers / 16 total queued+generating+unconsumed rollouts |
| Batch / accumulation | Per-device 4; GAS 4; whole-rollout atomic updates |
| Packing budget / max row | 40,960 tokens / 131,072 tokens |
| Completion budget | 16,384; agent 17 steps; 600-second timeout |
| Sampling | temperature 0.8; top-p 1.0; top-k disabled; thinking disabled |
| Reward | Binary correctness, same verifier and reward selection |
| TiTO | Exact captured token IDs and logprobs; lossless retention; fork threshold 0; all turns retained |
| Training allocation | 2 H100s on hopper-prod: 1 trainer + 1 weight-synced vLLM; 16 CPUs; 256 GiB requested host memory |
| Runtime budget | 24-hour allocation; 82,200 seconds soft training stop with checkpoint |
| Checkpoint cadence | Every 50 optimizer steps; all retained; additional hourly recovery checkpoints |
| Eval cadence | Every 100 optimizer steps plus final complete checkpoint |
| Eval protocol | Fixed 250 tests × OpenCode, Claude Code, Codex, Mini-SWE-Agent × pass@1 = 1,000 grades |
| Eval resources | Separate 2 H100s; TP1/DP2; concurrency 100; different node/endpoints; one active eval per run |
| Monitoring | Every 120 seconds during startup; every 600 seconds after 10 steps and 2 healthy progressing checks |
| Trackio | Same unified public dashboard; new series **Harbor OpenCode-only**; offline scalar ledger + 60-second online publisher |

Baseline is the existing audited 14.6% four-harness Harbor/E2B result from the same pinned base and protocol. It is reused as step 0; training checkpoints will be admitted only after complete coverage, TiTO and measured harness-version checks.

The single-harness schedule preserves all 1,000 reference first-pass task identities and order without reshuffling. It repeats that pass if exhausted. The old four-harness schedule instead rotated harnesses on later passes. A 1,000-step target does not guarantee complete task coverage: real groups/update depend on rollout sizes and admitted work. The new run uses the final stable recipe throughout; historical earlier changes in the resumed multi-harness run are not retroactively made identical.

## Validation and operational isolation

Training configuration equality, model/revision, dataset file hashes, zero test overlap inherited from the frozen manifest, exact task order, source hashes, save/eval cadence and generated launch commands passed preflight. 22 focused tests passed for routing all 8 generations across schedule boundaries, checkpoint completion/eligibility and comparison gates. Shared runtime versions remain torch 2.11.0, vLLM 0.25.1, transformers 5.14.1, Trackio 0.33.0. The trainer, session factory and atomic-rollout implementation are byte-identical to 80608.

Two capture transport files carry the already-qualified streaming keepalive repair from reference checkpoint 800 recovery 80912. Tokens, sampling, masking and loss are unchanged. Training has its own inference service, Harbor server, capture instance, ports, logs and checkpoint directory. Cleanup matches only this run's sandbox IDs. No existing trainer or evaluator was stopped or reconfigured. FSx and E2B quota remain shared resources.

## Jobs and artifacts

| Role | Job |
| --- | --- |
| Trainer | 81075 |
| Owned-sandbox cleanup | 81076, after training exits |
| Checkpoint evaluator controller | 81077 |
| Training/TiTO watchdog | 81078 |
| Offline Trackio replay | 81079 |
| CPU support-job supervisor | 81080 |
| Shared comparison publisher | 81083 |

- [Run configuration](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916/run_config.json)
- [Preparation validation](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916/validation.json)
- [Startup verification](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916/startup_verified.json)
- [Launch receipt](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916/submission.json)
- [Live monitor status](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916/monitor/status.json)
- [Training log](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916/job-81075/train.log)
- [Dashboard](https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?project=qwen35-2b-harbor-vs-opencode-20260916&smoothing=0)

Reproducible preparation script: `train/prepare_harbor_opencode_run.py`. The prepared run is already submitted; do not submit another copy. Frozen source and evaluated transport hashes are in the run directory. Checkpoint evals and CPU recovery use this run's own durable receipts, preventing duplicate submissions.
