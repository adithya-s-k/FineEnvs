# 500 new hard tasks, two epochs, four harnesses

September 17, 2026. **Configuration for approval; no jobs submitted.** This replaces the earlier [hard-majority proposal](HARD_CURRICULUM_FROM_500.md).

| Setting | Configuration |
| --- | --- |
| Model | Qwen3.5-2B, initialized from Harbor multi-harness checkpoint 500 |
| Initialization | Restore weights, optimizer and constant-LR scheduler; start a new task schedule at zero |
| Training data | 500 distinct hard tasks; no easy or medium introduction |
| Duration | Two shuffled task passes: 1,000 task groups, eight rollouts/group, 8,000 requested rollouts |
| Harness assignment | One harness per task per pass; rotate to a different harness in pass two |
| Harness balance | OpenCode, Claude Code, Codex, Mini-SWE-Agent: 125 groups each per pass |
| Versions | OpenCode 1.18.31; Claude Code 2.1.270; Codex 0.154.0; Mini-SWE-Agent 2.4.6 |
| Optimization | Async GRPO; LR 3e-6; paged AdamW 8-bit; bfloat16; non-reentrant gradient checkpointing |
| Batching | Four atomic packing units/update; at most two complete rollouts/unit; 40,960-token packing target |
| Flow control | Eight generations/group; staleness at most four; 16 total outstanding rollouts; worker ceiling 32 |
| TiTO | Lossless token capture; fork threshold zero; all retained agent turns; whole-rollout admission |
| Loss | Supervised-token mean across the update, retaining the parent recipe |
| Sampling | Temperature 0.8; top-p 1; top-k disabled; thinking disabled |
| Limits | 4,096 output tokens per response; 131,072 context; agent step limit 17; 600 seconds |
| Sandboxes | E2B, one CPU and 4 GiB per sandbox |
| Saving | Every 50 additional optimizer updates, plus final; keep all checkpoints |
| Evaluation | Every 100 additional updates, plus final; the same fixed 250 tasks × four harnesses × pass@1 |
| Evaluation budgets | 4,096 output tokens, 131,072 context, temperature 0.8, 17-step limit, 600 seconds |
| Logging | Separate run in the public comparison Trackio dashboard, with local JSONL and rollout receipts |
| Monitoring | Every two minutes initially; every ten minutes after stable progress |

The task passes replace the previous proposal of **500 additional optimizer updates**. Each task uses two of the four harnesses across this run. This is 1,000 scheduled groups, not 4,000 groups. Failed generation and stale rejection are reported separately from successful optimizer admission; 8,000 is the requested rollout count.

Checkpoint labels retain the parent step: save at 550, 600, 650, …; evaluate at 600, 700, … and the final checkpoint. The existing checkpoint-500 result is the starting reference: 37.0% overall and 16.4% on hard test cells. Final global step depends on actual admitted rollouts and packing, rather than being fixed at 1,000.

## Data separation

The cached catalog has 722 hard tasks. Screening leaves **521 eligible tasks**, from which 500 are shuffled and selected with seed 1. The exclusions cover:

- The 362 distinct tasks found in optimizer receipts through step 500 and early-job coverage records.
- The first 500 entries of the old training manifest, as an additional conservative interpretation of “first 500.” Together these exclusions cover 678 distinct old training tasks.
- All 250 fixed test tasks, including matching source IDs, notebooks, normalized questions and instructions.

Selected instructions are unique. Training graders are checked against the existing verifier and staged using the existing preparation helper. This establishes separation from the specified prior exposure and test set; it does not claim separation from every task in the original 1,000-task pool or foundation-model pretraining.

Local artifacts live under `experiments/async_grpo_harbor_data_agent/logs/multi4-hard500-2epochs-from500-20260917/`: selection, overlap evidence, frozen task files, per-file hashes and the 1,000-group execution schedule. The existing schedule helper emits a four-pass rotation cycle; only its first two passes belong to this run. The Cartesian pairs emitted by the task-staging helper are preserved as staging artifacts and must not be used as this run's schedule.

## Allocation proposal

The September 17, 11:42 UTC scheduler check found a two-H100 training allocation on `hopper-prod`, node `ip-10-53-81-209`, with 12 CPUs and 256 GiB. The test-only request no longer reported preemption after resources became free. Request 72 hours, with checkpointed continuation if required. No resources are reserved.

Use one separate H100 on `hopper-dev` for evaluation, TP1/DP1 and concurrency 50. That partition has six unallocated H100s at the check, but allows only one GPU per job. This changes serving capacity from the original TP1/DP2, concurrency-100 baseline, while preserving tasks, harness versions, generation budgets and scoring. Single-GPU evaluation has already been used for checkpoint recovery. Verify throughput before treating this as an ETA guarantee.

Keep evaluation on a different node with its own vLLM, OpenEnv and capture processes. Queue every due checkpoint if one evaluator is busy; never block the trainer on evaluation. FSx and the E2B account remain shared. No `hopper-extra` or `hopper-atl` allocation is proposed.

Historical throughput suggests roughly 1,600 additional updates for 8,000 admitted rollouts, around 40–56 training hours at 87–125 seconds/update. Harder tasks, the lower output cap, rejection rates and the last partial update can change this substantially. A full 72-hour request provides headroom; it is not a completion estimate.

## Before the approved launch

Data preparation is CPU-only. The new execution boundary and resume behavior are **not yet GPU-smoke validated**:

1. Separate optimizer resume from curriculum resume in both launcher and trainer. The old saved task cursor is 230; the new curriculum must begin at zero. Preserve the original checkpoint. Persist consumed/pending work so later restarts do not replay completed tasks.
2. Implement and test the finite 1,000-group boundary, drain queued admissible work and save the final partial update correctly. The current infinite async iterator does not provide two task epochs merely by setting `num_train_epochs=2`.
3. Validate the effective 4,096-token cap in requests and captures, TiTO alignment, an optimizer update, checkpoint reload and one eval trigger. Smoke output uses a separate directory; production still starts from the original checkpoint 500.
4. Track reward contrast, zero-gradient updates, per-harness admitted groups/tokens, task coverage, truncation and failures alongside reward and pass@1. Keep the all-hard curriculum fixed; do not silently replace failed hard tasks with easier ones.

Historically, 54/75 fully observed hard groups before step 500 had eight failures. All-hard training is viable, but improvement is not guaranteed. This run changes both curriculum and training output budget, so it cannot isolate the effect of harder data alone. Evidence: [training analysis](../reports/three-run-analysis-20260917/TRAINING.md).
