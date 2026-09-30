# Hard-majority continuation from multi-harness checkpoint 500

**Superseded:** the requested configuration is now [500 new hard tasks for two epochs](HARD500_TWO_EPOCHS.md). This earlier proposal is retained for reference.

September 17, 2026. **Viable as a new curriculum branch; not launched.**

## Verified inputs

- The local checkpoint at `experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915/job-79083/run/checkpoint-500` passes the existing checkpoint validator and every hash in `checkpoint.ready.json`. It contains full Qwen3.5-2B weights, tokenizer, optimizer, scheduler and RNG state. This is a file-integrity check, not a new GPU resume smoke.
- Its fixed-test pass@1 is 37.0% overall: easy 72.7%, medium 44.3%, hard 16.4%.
- The existing screened 1,000-task training pool contains 250 hard tasks. A 400-task subset with 250 hard, 110 medium and 40 easy tasks passes the existing schedule validator: 100 task groups per harness per pass, one harness per task, no duplicate task IDs. The subset inherits the existing train/test separation checks.
- The broader cached catalog contains 722 hard tasks before exclusions. Expanding beyond the existing pool requires repeating notebook/source/question/instruction overlap screening; those 722 are not all established as eligible here.

## Proposed configuration

| Setting | Proposal |
| --- | --- |
| Initialization | Parent multi-harness checkpoint 500 |
| Duration | 500 additional optimizer steps; global steps 501–1,000 |
| Optimizer | Preserve checkpoint optimizer/scheduler for continuity; recorded LR is 3e-6 |
| Curriculum | 250 hard / 110 medium / 40 easy: 62.5% / 27.5% / 10% |
| Scheduling | New shuffled schedule from group zero; one harness per task per pass; balanced harness counts |
| Harnesses | OpenCode, Claude Code, Codex, Mini-SWE-Agent; retain frozen versions |
| Rollout recipe | Eight generations per task; max staleness four; existing atomic admission and token-normalized loss |
| Output budget | Enforce 4,096 output tokens per response in serving/capture and trainer configuration to match canonical eval; validate in smoke |
| Checkpoints | Every 50 updates: 550, 600, …, 1,000 |
| Evaluation | Same 250 fixed tasks × four harnesses × pass@1, at 600, 700, …, 1,000, on separate allocations |
| Logging | Separate run identity and output directory, with parent checkpoint recorded |

This changes both curriculum and training output budget. It is a proposed improvement run, not an isolated estimate of the effect of harder data. An attribution study would need a matched control. A weights-only warm start with a fresh optimizer is also possible, but would be a separate initialization choice and should not be silently called an exact resume.

## Required setup before launch

1. **Separate optimizer resume from dataset resume.** The saved checkpoint-500 task cursor is 230. The current launcher and TRL both restore that old cursor automatically. A new curriculum must explicitly start at group zero while retaining model/optimizer state. Preserve the original checkpoint files and record the new schedule hash. Also persist completed/pending group accounting for later restarts so the replay bug cannot recur.
2. **Verify the effective generation limit.** For these loop-owning harnesses, changing the trainer flag alone is insufficient evidence: verify the serving/capture request limits and actual captured responses. The earlier training used up to 16,384 tokens per response while canonical evaluation uses 4,096.
3. **Check learning signal on hard training tasks.** Before committing the full allocation, probe a small balanced set of hard training tasks with checkpoint 500 and eight rollouts per task. During the first 25–50 updates, inspect mixed-outcome group fraction, submission, truncation, coverage, TiTO and admitted tokens, alongside rewards. Do not use held-out eval outcomes to select training tasks.
4. **Test the branch bookkeeping and save/eval path.** Verify that the first new group is zero, the parent weights/optimizer load, the run performs 500 additional updates, and checkpoint/eval jobs use the new run identity. A full Trainer resume uses `max_steps=1000`; setting it to 500 would already satisfy the stop condition at the parent checkpoint.

## Why mostly hard, rather than all hard

Historical complete eight-rollout groups admitted during steps 31–500 show **54/75 hard groups (72%) with all failures**, versus 28% with mixed outcomes. In steps 401–500 alone, 15/18 observed complete hard groups all fail; this is a small sample on changing tasks, not a measurement of checkpoint 500 on the proposed pool. These counts exclude partially admitted or unscored groups.

All-equal rewards give no group-relative advantage. Hard-only training could therefore increase cost without increasing useful learning. Retaining some medium/easy tasks supplies additional opportunities for reward contrast and can help check retention. The proposed mixture is a reasonable starting hypothesis, not a tuned optimum. [DAPO's dynamic sampling](https://dapo-sia.github.io/) explicitly addresses groups with all-correct or all-incorrect outcomes; its math results do not establish the best curriculum for these agent tasks.

At the old peak-window rate of 87 seconds/update, 500 updates take about **12 hours** of training. Late-run rates would imply around 17 hours. These are historical arithmetic estimates, not a measured ETA for the harder curriculum; queue time and independent eval completion are additional. Likewise, 500 updates do not guarantee that every task receives all eight admitted rollouts: a full 400-task pass requests 3,200 rollouts.

Evidence: [training analysis](../reports/three-run-analysis-20260917/TRAINING.md), [evaluation analysis](../reports/three-run-analysis-20260917/REPORT.md), and the local optimizer receipts in `experiments/analysis-three-runs-20260917/`. No jobs were submitted or checkpoints modified during this assessment.
