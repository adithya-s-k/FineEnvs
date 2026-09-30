# Data Agent: completed training and evaluation

Updated September 17, 2026. Three async Qwen3.5-2B runs reached 1,000 optimizer steps.
Every scheduled 100-step checkpoint evaluation is complete: 250 fixed test tasks ×
four harnesses, pass@1. The test set contains 33 easy, 118 medium and 99 hard tasks.

| Run | Baseline | Best measured checkpoint | Final step 1,000 |
| --- | ---: | ---: | ---: |
| Harbor multi-harness | 14.6% | **37.0% at 500** | 26.3% |
| Native OpenCode | 15.9% | **29.8% at 1,000** | 29.8% |
| Harbor OpenCode-only | 14.6% | **39.5% at 700** | 26.4% |

[Every checkpoint, harness and difficulty](reports/async-comparison-20260916/REPORT.md) ·
[Public Trackio comparison](https://huggingface.co/spaces/HuggingEnvs/data-agent-training-comparison-trackio)

## What training and evaluation show

**Multi-harness produces longer responses but takes fewer actions.** Between training
steps 401–500 and 901–1,000, completion tokens per admitted rollout grow **3,521 → 9,480**,
while emitted tool calls fall **15.86 → 11.24**. In the final window, 37.5% of admitted
rollouts contain a response longer than the evaluation's 4,096-token output cap.
At evaluation, output-truncated rollouts rise **9/1,000 → 556/1,000** from peak to final.
OpenCode accounts for about 64% of the net lost successful evaluations. Budget mismatch
is a plausible contributor, not a proven explanation for the whole decline.

**Harbor OpenCode-only continues working but finishes less reliably.** Peak-to-final
eval tool calls rise **16.62 → 20.97**, while submission falls **68.9% → 40.7%** on the
86-task subset with explicit submission instrumentation. Output truncation remains rare.
Training also shows longer outputs (**2,122 → 3,631 tokens**) and more tool use
(**14.63 → 17.43 calls**) per admitted rollout, comparing steps 601–700 with 901–1,000.

**Native OpenCode finishes at its best aggregate score with shorter training outputs.**
Its final 100 steps average 1,075 completion tokens and 6.05 emitted tool calls per
admitted rollout. However, late prompt forking increases context overhead: forwarded
tokens per supervised token rise **22.3× → 99.4×** between steps 401–500 and 901–1,000.

**One-harness training transfers to other harnesses.** Harbor OpenCode-only's best
checkpoint scores 46.4% under Claude Code and 40.0% under OpenCode. Native OpenCode
improves all four evaluation harnesses; its largest gains are outside OpenCode.

## Training accounting changes the interpretation

| Run | Distinct tasks covered | Supervised tokens | Forwarded tokens | Zero-fresh-gradient steps |
| --- | ---: | ---: | ---: | ---: |
| Harbor multi-harness | 482 | 22.15M | 1,001.32M | 349/1,000 |
| Native OpenCode | 566 | 5.33M | 228.41M | 583/1,000 |
| Harbor OpenCode-only | 523 | 14.63M | 419.43M | 380/1,000 |

The 1,000-step cap did not cover the whole 1,000-task training pool. Equal optimizer
steps also did not provide equal token exposure. Zero-gradient steps coincide with
zero within-group reward variance and supply no fresh GRPO contrast; optimizer
momentum may still update weights.

The multi-harness resume after step 684 revisits previously seen tasks in **1,575 of
1,579 admitted rollouts**. The saved schedule cursor does not preserve later completed
groups. This changes late data exposure but cannot explain the initial decline after
step 500, which happened before that restart.

## Next experiments

1. Fix and test resume accounting for out-of-order completed groups and unfinished work.
2. Diagnose the train/eval output-budget mismatch on a small, separately labeled cohort;
   preserve the canonical scores. Check submission and executed actions, not just reward.
3. Track per-harness tokens, calls, submission, truncation, context duplication, task
   coverage and zero-advantage groups. Compare future runs at matched exposure and budgets.

This is an observational comparison. Backend, rollout filtering, training histories and
historical eval retries differ. The 39.5% versus 37.0% peak difference is not clearly
separated by paired task-level uncertainty. All three trainers use binary correctness;
native raw efficiency bonuses are removed before training.

Evidence: [evaluation analysis and limitations](reports/three-run-analysis-20260917/REPORT.md),
[training token/tool analysis and reproduction](reports/three-run-analysis-20260917/TRAINING.md).
Raw captures and accepted scores are unchanged.

[Short message for sharing](reports/three-run-analysis-20260917/TLDR.md)
