# Article handoff: SFT and RL across agent harnesses

Prepared 29 September 2026 from accepted results through 28 September. This is an evidence packet for updating the existing article, not a claim that all methods were compared under identical training conditions. No images are embedded. The HTML version contains the same text and semantic tables.

## Start here

Extend the article from multi-harness RL to **learning from and through agent harnesses**: supervised imitation of successful teacher traces, then a comparison with the existing online RL experiments. Preserve the OpenEnv/Harbor/TiTO explanation and the earlier RL findings. Add the SFT evidence without replacing or rewriting the recorded RL history.

The central result is model-dependent. Qwen3.5-2B benefits strongly from multi-harness SFT: 14.6% baseline → 30.9% after one epoch → 32.7% after two. LFM2.5-2.6B improves with OpenCode-only SFT to 47.5%, but multi-harness SFT reaches only 43.1%, close to its approximately 42.2% baseline. LFM multi-harness RL retains the strongest final correctness at 54.2% and reduces native tool calls by 31.1% on baseline-matched successes. Multi-harness SFT also reduces tool use across all four harnesses in both models, despite having no explicit efficiency reward.

**Training objectives must be stated prominently:** Qwen RL used correctness reward only. LFM RL used correctness plus a tool-efficiency bonus. SFT used demonstration imitation with token-level cross-entropy. Tool efficiency is an evaluation measurement for every method, not a shared training objective.

The latest OpenCode-only and multi-harness SFT runs are complete: two epochs for both models, with all eight checkpoint evaluations graded at 1,000/1,000. The older 907-task bash SFT evaluations remain incomplete. The full 4,677-demonstration bash SFT evaluations are complete but have disclosed test overlap.

## 1. Evaluation protocol and definitions

- Fixed test cohort: 250 tasks, evaluated independently with OpenCode, Claude Code, Codex and Mini-SWE-Agent. One selected graded rollout per task/harness pair, pass@1; 1,000 cells per checkpoint.
- Difficulty per harness: 33 easy, 118 medium, 99 hard. Across four harnesses: 132, 472 and 396 cells respectively. Do not describe those as 1,000 unique tasks.
- Completed overall scores are averages over those 1,000 binary correctness outcomes. Infrastructure failures are retried, not counted as wrong answers. These operational retries are not pass@k sampling, but retry history remains a comparability limitation.
- Student SFT evaluations use temperature 0.8, at most 4,096 output tokens per model call, a 17-call/agent-step budget, and a 600-second agent timeout. Qwen evaluates with thinking disabled; LFM retains its qualified native serving configuration. Both reload the actual saved student checkpoint.
- SFT evaluation jobs initially use concurrency 100 on separate GPUs. LFM epoch-1 recovery used a fresh allocation at concurrency 10. Concurrency is a throughput setting, not the sample count.
- TiTO checks passed for all completed OpenCode and multi-harness SFT evaluations. This establishes the recorded capture checks, not statistical significance or a guarantee of task success.
- Qwen Harbor baseline: 146/1,000 = 14.6%. Native OpenCode has its own historical baseline, 15.9%, and a different serving path. LFM baseline: 421/998 = 42.184%, with two Mini-SWE-Agent grades missing. Preserve that qualification.

## 2. All SFT and selected RL outcomes

E1/E2 mean completed SFT epochs. RL best means the highest **complete** aggregate checkpoint observed on this evaluation set; final means step 1,000. Best is retrospective test-set selection, not an independently chosen validation checkpoint. Every per-harness value in an RL row comes from that same checkpoint.

`*` = incomplete evaluation, graded subset only. `†` = test overlap. Blank values would mean unavailable, never zero.

### Qwen3.5-2B

| Run | Graded | Overall | OpenCode | Claude Code | Codex | Mini-SWE |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained baseline | 1000/1,000 | 14.6% | 10.8% | 16.8% | 16.4% | 14.4% |
| OpenCode SFT · 801 rollouts / E1 | 1000/1,000 | 28.4% | 30.0% | 27.6% | 31.2% | 24.8% |
| OpenCode SFT · 801 rollouts / E2 | 1000/1,000 | 26.5% | 26.0% | 30.4% | 27.2% | 22.4% |
| Multi-harness SFT · 3,189 rollouts / E1 | 1000/1,000 | 30.9% | 29.6% | 33.6% | 32.8% | 27.6% |
| Multi-harness SFT · 3,189 rollouts / E2 | 1000/1,000 | 32.7% | 29.6% | 36.4% | 33.6% | 31.2% |
| Harbor OpenCode RL / best, 700 | 1000/1,000 | 39.5% | 40.0% | 46.4% | 37.2% | 34.4% |
| Harbor OpenCode RL / final, 1000 | 1000/1,000 | 26.4% | 22.0% | 30.8% | 32.4% | 20.4% |
| Harbor multi-harness RL / best, 500 | 1000/1,000 | 37.0% | 32.8% | 44.8% | 39.2% | 31.2% |
| Harbor multi-harness RL / final, 1000 | 1000/1,000 | 26.3% | 5.6% | 38.8% | 24.8% | 36.0% |
| Native OpenCode RL / best = final, 1000 | 1000/1,000 | 29.8% | 20.4% | 33.2% | 29.6% | 36.0% |
| Hard-task RL continuation / 1000 * | 924/1,000 | 26.0% | 17.3% | 40.0% | 38.9% | 9.6% |

### LFM2.5-2.6B

| Run | Graded | Overall | OpenCode | Claude Code | Codex | Mini-SWE |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained baseline * | 998/1,000 | 42.2% | 33.6% | 33.2% | 40.0% | 62.1% |
| Bash SFT · 907 tasks / E1 * | 999/1,000 | 41.7% | 35.2% | 26.1% | 44.4% | 61.2% |
| Bash SFT · 907 tasks / E2 * | 850/1,000 | 46.6% | 39.4% | 30.9% | 47.4% | 67.6% |
| Bash SFT · full 4,677 / E1 † | 1000/1,000 | 41.3% | 39.2% | 32.0% | 34.4% | 59.6% |
| Bash SFT · full 4,677 / E2 † | 1000/1,000 | 40.8% | 38.0% | 24.0% | 35.6% | 65.6% |
| OpenCode SFT · 801 rollouts / E1 | 1000/1,000 | 45.1% | 51.2% | 28.8% | 42.4% | 58.0% |
| OpenCode SFT · 801 rollouts / E2 | 1000/1,000 | 47.5% | 51.6% | 33.2% | 46.4% | 58.8% |
| Multi-harness SFT · 3,189 rollouts / E1 | 1000/1,000 | 38.3% | 22.4% | 40.8% | 42.4% | 47.6% |
| Multi-harness SFT · 3,189 rollouts / E2 | 1000/1,000 | 43.1% | 38.0% | 42.4% | 46.8% | 45.2% |
| Harbor OpenCode RL / best, 900 | 1000/1,000 | 52.4% | 56.0% | 43.6% | 48.4% | 61.6% |
| Harbor OpenCode RL / final, 1000 | 1000/1,000 | 52.3% | 58.0% | 42.0% | 43.2% | 66.0% |
| Harbor multi-harness RL / best, 700 | 1000/1,000 | 54.6% | 51.2% | 48.8% | 53.6% | 64.8% |
| Harbor multi-harness RL / final, 1000 | 1000/1,000 | 54.2% | 49.6% | 48.8% | 53.6% | 64.8% |

### What can be said from these scores

- Qwen multi-harness SFT E2 is +18.1 percentage points above baseline and +6.2 points above OpenCode-only SFT E2. Compared with the best OpenCode SFT epoch, the gap is +4.3 points. Both SFT variants improve every evaluation harness over the Qwen base model.
- Qwen multi-harness SFT E2 (32.7%) exceeds final Harbor OpenCode RL (26.4%) and final Harbor multi-harness RL (26.3%). It remains below best Harbor OpenCode RL (39.5%) and best Harbor multi-harness RL (37.0%). Do not hide the peak-to-final RL decline or use these observations to claim SFT is categorically better.
- LFM OpenCode-only SFT E2 is 47.5%, about +5.3 points above its provisional baseline. LFM multi-harness SFT E2 is 43.1%, about +0.9 points above baseline and 4.4 points below OpenCode SFT E2. The approximately 0.9-point change is not established as statistically reliable.
- LFM multi-harness SFT recovers from 38.3% at E1 to 43.1% at E2. Improvements over baseline occur in OpenCode, Claude Code and Codex; Mini-SWE falls from approximately 62.1% to 45.2%. The overall number hides that imbalance.
- LFM multi-harness RL finishes at 54.2%, versus 52.3% for OpenCode RL, 47.5% for OpenCode SFT and 43.1% for multi-harness SFT. Relative to multi-harness SFT E2, the final RL gap is 11.1 points. These methods differ in objective, data exposure and budget.
- Full bash SFT reaches 41.3% at E1 and 40.8% at E2 despite using more demonstrations. Its overlap and different task distribution prevent a clean held-out or data-scaling claim. The older 907-task E2 result (46.6% on 850 cells) must not be ranked as a complete result.

## 3. Tool efficiency, measured alongside correctness

For each checkpoint, match task/harness pairs solved correctly by both the baseline and checkpoint. Count native ATIF agent tool invocations with valid unique tool-call IDs. A tool invocation is not a model turn or the number of shell commands inside a tool call. All graded rows used in this packet had native count coverage; missing/unverifiable trajectories would be excluded rather than assigned zero.

`tool-call savings (%) = 100 × (1 − sum(checkpoint calls) / sum(baseline calls))`

Positive means fewer calls; negative means more. This is a ratio of summed calls on matched successes, not an average of per-task percentage changes. Matched subsets vary by checkpoint, so this is conditional efficiency, not the total cost of solving the full test set. SFT did not optimize this metric directly.

### Qwen3.5-2B tool use

| Run | Matched pairs | Base mean calls | Run mean calls | Savings | OC savings / n | Claude savings / n | Codex savings / n | Mini-SWE savings / n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained baseline | 146 | 9.98 | 9.98 | +0.0% | +0.0% / 27 | +0.0% / 42 | +0.0% / 41 | +0.0% / 36 |
| OpenCode SFT · 801 rollouts / E1 | 94 | 9.72 | 6.40 | +34.1% | +38.9% / 16 | +25.2% / 22 | +36.6% / 35 | +33.3% / 21 |
| OpenCode SFT · 801 rollouts / E2 | 93 | 9.95 | 6.97 | +29.9% | +49.6% / 14 | +16.4% / 29 | +44.1% / 28 | +15.9% / 22 |
| Multi-harness SFT · 3,189 rollouts / E1 | 107 | 9.83 | 4.62 | +53.0% | +42.5% / 19 | +35.4% / 32 | +60.8% / 34 | +63.6% / 22 |
| Multi-harness SFT · 3,189 rollouts / E2 | 108 | 9.75 | 4.50 | +53.8% | +43.0% / 15 | +41.9% / 34 | +62.3% / 33 | +58.0% / 26 |
| Harbor OpenCode RL / best, 700 | 123 | 9.61 | 12.41 | -29.2% | -133.8% / 21 | -33.6% / 38 | -14.4% / 36 | +2.3% / 28 |
| Harbor OpenCode RL / final, 1000 | 91 | 9.64 | 14.56 | -51.1% | -203.6% / 13 | -50.7% / 29 | -21.3% / 27 | -37.1% / 22 |
| Harbor multi-harness RL / best, 500 | 119 | 9.58 | 14.90 | -55.5% | -112.5% / 17 | -126.9% / 38 | -52.1% / 37 | +22.3% / 27 |
| Harbor multi-harness RL / final, 1000 | 80 | 10.15 | 10.95 | -7.9% | -166.7% / 1 | -61.4% / 34 | -10.1% / 22 | +43.5% / 23 |
| Native OpenCode RL / best = final, 1000 | 112 | 10.11 | 8.46 | +16.3% | +29.7% / 22 | +31.2% / 30 | -8.9% / 24 | +16.9% / 36 |
| Hard-task RL continuation / 1000 * | 78 | 9.45 | 15.71 | -66.2% | -113.6% / 9 | -123.8% / 27 | -49.0% / 35 | +16.3% / 7 |

### LFM2.5-2.6B tool use

| Run | Matched pairs | Base mean calls | Run mean calls | Savings | OC savings / n | Claude savings / n | Codex savings / n | Mini-SWE savings / n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrained baseline * | 421 | 7.68 | 7.68 | +0.0% | +0.0% / 84 | +0.0% / 83 | +0.0% / 100 | +0.0% / 154 |
| Bash SFT · 907 tasks / E1 * | 289 | 7.64 | 7.83 | -2.4% | +1.8% / 45 | -29.5% / 41 | +11.3% / 70 | -3.7% / 133 |
| Bash SFT · 907 tasks / E2 * | 282 | 7.49 | 7.59 | -1.3% | -10.5% / 46 | -17.4% / 43 | +12.0% / 66 | -1.5% / 127 |
| Bash SFT · full 4,677 / E1 † | 283 | 7.37 | 7.23 | +1.9% | -2.6% / 51 | -14.6% / 47 | +24.1% / 59 | -2.0% / 126 |
| Bash SFT · full 4,677 / E2 † | 284 | 7.30 | 7.41 | -1.5% | -12.0% / 55 | -28.1% / 34 | +19.6% / 61 | -3.5% / 134 |
| OpenCode SFT · 801 rollouts / E1 | 310 | 7.50 | 6.45 | +14.0% | +26.2% / 67 | -2.0% / 43 | +36.9% / 77 | -1.5% / 123 |
| OpenCode SFT · 801 rollouts / E2 | 309 | 7.34 | 6.72 | +8.5% | +17.1% / 61 | -10.5% / 46 | +31.7% / 80 | -4.8% / 122 |
| Multi-harness SFT · 3,189 rollouts / E1 | 255 | 7.20 | 5.50 | +23.5% | +22.6% / 28 | +20.4% / 57 | +44.3% / 69 | +11.0% / 101 |
| Multi-harness SFT · 3,189 rollouts / E2 | 276 | 7.18 | 5.44 | +24.2% | +18.0% / 46 | +27.1% / 57 | +41.8% / 74 | +10.8% / 99 |
| Harbor OpenCode RL / best, 900 | 342 | 7.47 | 6.49 | +13.2% | +34.9% / 73 | +7.3% / 62 | +18.2% / 79 | +3.4% / 128 |
| Harbor OpenCode RL / final, 1000 | 337 | 7.32 | 6.49 | +11.4% | +36.4% / 74 | -9.6% / 55 | +23.6% / 71 | +2.1% / 137 |
| Harbor multi-harness RL / best, 700 | 361 | 7.57 | 5.55 | +26.7% | +31.4% / 68 | +19.2% / 71 | +46.8% / 88 | +14.8% / 134 |
| Harbor multi-harness RL / final, 1000 | 356 | 7.43 | 5.12 | +31.1% | +32.8% / 66 | +28.3% / 69 | +53.0% / 89 | +16.2% / 132 |

### Efficiency observations worth highlighting

- Qwen multi-harness SFT E2: mean native calls decrease 9.75 → 4.50 on 108 matched successful pairs, a 53.8% saving. Savings are positive for every harness. This accompanies a large correctness gain over baseline.
- LFM multi-harness SFT E2: 7.18 → 5.44 calls on 276 pairs, a 24.2% saving. All four harnesses show positive savings. It is more efficient conditionally, but correctness remains substantially below LFM multi-harness RL.
- LFM OpenCode SFT E2 saves 8.5% overall but increases calls under Claude Code (10.5% more) and Mini-SWE (4.8% more). Its higher correctness than mixed SFT therefore comes with weaker and less uniform tool savings.
- LFM final multi-harness RL: 7.43 → 5.12 calls on 356 pairs, a 31.1% saving, with positive savings in all four harnesses. OpenCode-only RL saves 11.4% overall but uses 9.6% more calls under Claude Code.
- Qwen Harbor RL often uses more calls on matched successes. It was trained on correctness only, so do not describe this as a failed tool-efficiency reward. Qwen native OpenCode RL has its own baseline and saves 16.3% overall.
- Some Qwen RL per-harness cells have very small samples. Final Harbor multi-harness RL under OpenCode has only one matched successful pair; its percentage is not a robust population claim. Show n, and avoid highlighting that cell as a general trend.
- Efficiency differences between Qwen and LFM do not isolate an architectural effect: baseline behavior, training objectives and cohorts differ. Native action counts are verified, but the semantic size of one tool call also differs by harness.

## 4. SFT data, preparation and recipe

Two independently initialized base models were used: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc` and `LiquidAI/LFM2.5-2.6B` at `654f9463ce32b05d0429d76fe1f580b27d4c1ac0`. The full bash run started fresh, not from the 907-task checkpoint. Multi-harness SFT also started fresh, not from OpenCode SFT.

| Variant | Source and selection | Examples and weighting | Epoch checkpoints |
|---|---|---|---|
| LFM bash, matched | FineEnvs/SmolDataEnvs-sft; 907 available tasks matched to the original LFM RL pool, 376 medium / 531 hard | 907 conversations; assistant-only loss | 114, 228 |
| LFM bash, full | All 4,677 demonstrations: 1,402 easy / 2,640 medium / 635 hard | Full corpus, including disclosed test overlap | 585, 1,170 |
| OpenCode teacher SFT, both models | 801 successful OpenCode rollouts from 801 tasks: 356 medium / 445 hard | 4,825 assistant-turn examples | 604, 1,208 |
| Multi-harness teacher SFT, both models | 3,189 successful rollouts from 888 tasks: 377 medium / 511 hard | 17,929 assistant-turn examples | 2,242, 4,484 |

The teacher source is `AdithyaSK/qwen38-27b-harbor-rollouts`, pinned at `3d826b6854acdb4e4918e5047cdf9f70eb38602e`, collected with Qwen3.8-27B. Mixed data: OpenCode 801 rollouts / 4,825 turns; Claude Code 781 / 4,581; Codex 797 / 4,078; Mini-SWE-Agent 810 / 4,445. There are 692 tasks with successful published traces in all four harnesses, a possible matched subset for a future controlled study. Four successful credential-containing source rollouts were excluded from publication; do not reintroduce them.

Teacher token IDs and behavior logprobs are not student SFT targets. The actual runs used student-specific rendered messages, input IDs and completion masks. Exact model-specific labels were checked against native TRL preparation over the full mixed corpus. Tool responses and earlier history are context, with only the current assistant completion supervised. Qwen training preserves reasoning when present in the teacher response; evaluation still disables thinking as in its baseline. LFM's training-only prefix adjustment is removed when saving its original inference template.

Common recipe: TRL SFT, full parameter fine-tuning, two epochs, LR 3e-6 constant, per-device batch 1, gradient accumulation 8, BF16, paged AdamW 8-bit, gradient checkpointing, seed 42, no packing and no truncation. Save every 50 updates plus epoch boundaries; evaluate E1 and E2. The teacher-trace runs use completion-only cross-entropy; bash runs use assistant-only cross-entropy.

Multi-harness SFT shuffles turn records with standard completion-token loss. It does not balance tasks, rollouts or harnesses. Long answers and multi-turn rollouts contribute more supervised tokens. The 1,000-task RL pool is not the same as the 888 successful-demonstration tasks; these are not identical-data experiments.

### Measured SFT training, not queue-time estimates

| Run | Tasks / examples | Epoch steps | GPU job | Trainer runtime | First 50 loss | Last 50 loss |
| --- | --- | --- | --- | --- | --- | --- |
| LFM bash SFT, matched 907 | 907 / 907 | 114/228 | 86340 | 9.5 min | 0.544 | 0.397 |
| LFM bash SFT, full 4,677 | 4677 / 4677 | 585/1170 | 86367 | 45.5 min | 0.519 | 0.337 |
| LFM OpenCode SFT | 801 / 4825 | 604/1208 | 86427 | 88.3 min | 0.400 | 0.279 |
| Qwen OpenCode SFT | 801 / 4825 | 604/1208 | 86429 | 71.4 min | 0.282 | 0.215 |
| LFM multi-harness SFT | 888 / 17929 | 2242/4484 | 86472 | 386.8 min | 0.908 | 0.585 |
| Qwen multi-harness SFT | 888 / 17929 | 2242/4484 | 86473 | 318.8 min | 0.513 | 0.380 |

All six SFT trainers reached epoch 2. Logged training losses and gradient norms were finite. These losses are from different target distributions and must not be ranked across datasets. The runtimes are the trainer-reported train-loop times; they exclude queueing, most preparation and evaluation, and are not an end-to-end cost comparison.

Teacher-trace prepared token volumes per corpus pass:

| Student | OpenCode input tokens | OpenCode supervised tokens | Mixed input tokens | Mixed supervised tokens |
| --- | --- | --- | --- | --- |
| LFM | 39,730,621 | 588,577 | 168,836,025 | 4,098,011 |
| Qwen | 42,179,296 | 610,720 | 177,276,678 | 4,361,348 |

These are serialized per-example input lengths, including repeated history, rather than distinct text tokens. Mixed SFT has roughly seven times the supervised target tokens of OpenCode SFT, not simply a different harness label. No claim of equal compute or exposure is justified.

The mixed-data pipeline initially failed before production training because a JSONL reader split embedded Unicode line separators. Reading physical lines fixed six affected records in each student file without removing or changing data. Qwen's reasoning-prefix mismatch was also caught before production and resolved with per-example template settings. Full-corpus label audits and save/resume/evaluation smokes passed before the successful runs. Final model checks confirmed changed weights. Evaluation retries preserved previous grades; transient failures should not be described as model errors.

## 5. RL context to preserve

Qwen's original three async runs used a 1,000-task pool (150 easy / 600 medium / 250 hard), but 1,000 optimizer steps did not guarantee full coverage. Observed distinct tasks were 482 for Harbor multi-harness, 523 for Harbor OpenCode-only and 566 for native OpenCode. They used correctness only. The source reports document different resume histories and rollout filtering.

LFM OpenCode-only and multi-harness RL used a fresh base model and a pool of 1,000 tasks (400 medium / 600 hard), with an explicit two-pass schedule and a 1,000-update cap. A multi-harness group uses one harness for that scheduled task, not all four harnesses per update. Configured recipe: AsyncGRPO, LR 3e-6, eight generations, max in-flight 32, staleness 4, batch 4 / accumulation 4, BF16, paged AdamW 8-bit, a 40,960-token packing budget and atomic rollout admission. A trainer GPU and separate inference GPU were allocated per run. The configured two-pass schedule is not itself evidence of complete realized task coverage.

LFM reward: `correctness × (1 + 0.1 × 15 / (15 + native_tool_calls))`, with no bonus for an unverified count. The tested implementation also withholds the efficiency bonus when the count is zero. Incorrect answers get no bonus. The primary evaluation score remains binary correctness; do not substitute the shaped training reward for pass@1.

Both RL families saved every 50 updates and evaluated scheduled 100-step checkpoints; extra resume/diagnostic checkpoints exist. Best and final comparisons use only completed eligible evaluations. Historical deployment, allocation and transport configurations changed, so do not infer exact executed settings solely from an old planned config's `status` field.

Earlier Qwen findings remain relevant: the multi-harness decline starts after step 500; the documented task-repetition resume bug occurs after step 684 and cannot explain the initial decline. The earlier report records increasing output truncation and 35–58% of updates with no fresh reward-contrast gradient. These are observed associations and accounting issues, not proven explanations of the SFT results or of LFM behavior.

A Qwen hard-task continuation from checkpoint 500 remains a separate, incomplete local evaluation cohort: its latest local step-1,000 score is 26.0% on 924/1,000 grades. Do not merge it into the original Qwen RL curve or present it as a successful hard-task curriculum. SETA native-bash-only evaluation uses a different protocol and is not included in the four-harness aggregates here.

## 6. Difficulty breakdown for completed SFT and selected RL

Values are averages over available graded cells within each difficulty. The full-dataset bash SFT overlap flag still applies. Per-checkpoint raw counts and definitions remain in the linked source JSONs.

### Qwen3.5-2B

| Run | Easy | Medium | Hard |
| --- | --- | --- | --- |
| OpenCode SFT · 801 rollouts / E1 | 59.8% | 30.9% | 14.9% |
| OpenCode SFT · 801 rollouts / E2 | 63.6% | 29.7% | 10.4% |
| Multi-harness SFT · 3,189 rollouts / E1 | 71.2% | 33.7% | 14.1% |
| Multi-harness SFT · 3,189 rollouts / E2 | 73.5% | 34.1% | 17.4% |
| Harbor OpenCode RL / best, 700 | 71.2% | 44.5% | 23.0% |
| Harbor OpenCode RL / final, 1000 | 59.1% | 30.3% | 10.9% |
| Harbor multi-harness RL / best, 500 | 72.7% | 44.3% | 16.4% |
| Harbor multi-harness RL / final, 1000 | 42.4% | 32.6% | 13.4% |
| Native OpenCode RL / best = final, 1000 | 59.8% | 35.8% | 12.6% |

### LFM2.5-2.6B

| Run | Easy | Medium | Hard |
| --- | --- | --- | --- |
| Bash SFT · full 4,677 / E1 † | 66.7% | 41.7% | 32.3% |
| Bash SFT · full 4,677 / E2 † | 65.2% | 45.8% | 26.8% |
| OpenCode SFT · 801 rollouts / E1 | 74.2% | 50.0% | 29.5% |
| OpenCode SFT · 801 rollouts / E2 | 77.3% | 52.8% | 31.3% |
| Multi-harness SFT · 3,189 rollouts / E1 | 69.7% | 42.6% | 22.7% |
| Multi-harness SFT · 3,189 rollouts / E2 | 78.0% | 50.4% | 22.7% |
| Harbor OpenCode RL / best, 900 | 78.8% | 56.6% | 38.6% |
| Harbor OpenCode RL / final, 1000 | 76.5% | 56.1% | 39.6% |
| Harbor multi-harness RL / best, 700 | 79.5% | 58.5% | 41.7% |
| Harbor multi-harness RL / final, 1000 | 78.8% | 58.7% | 40.7% |

## 7. Article update instructions

The local article checkout inspected for this handoff is `/tmp/data-agent-article-merge/content/articles/multi-harness-rl/`. Its checked chapters still describe the September Qwen RL experiments. This is a local reference, not a claim that it matches the latest article PR or deployed Space. Before editing, inspect the actual target branch and preserve newer content.

1. **Introduction / framing:** keep harnesses and OpenEnv/Harbor as the organizing idea. Add that the same trajectory infrastructure supports supervised imitation as well as online RL. A suitable subtitle is “Learning across agent harnesses with SFT and RL.”
2. **Data-agent / data preparation:** add the two teacher-trace SFT datasets, success filtering, task counts, current-response masks and student-specific tokenization. Distinguish raw rollouts from assistant-turn examples. Mention the ready-to-use public SFT export.
3. **Training-ablation chapter:** organize by model first, then SFT versus RL. Show both SFT epochs and best/final RL. Preserve the original Qwen RL resume and exposure findings. Add LFM RL as a separate later experiment, not a continuation of Qwen.
4. **Evaluation chapter:** keep pass@1 primary. Add native tool-call savings on matched successes, formula and n. State Qwen RL correctness-only, LFM RL correctness plus efficiency, and SFT imitation beside the figures. Do not call all plotted savings an optimized reward.
5. **Conclusions:** replace “multi-harness training always wins” language with the actual model-dependent picture. Qwen mixed SFT improves broadly. LFM mixed SFT reduces calls but does not match OpenCode SFT correctness. LFM mixed RL combines higher correctness with broad call savings.
6. **Reproduction:** link the public portable and student-tokenized dataset configurations, exact recipe and local evidence packet. Update historical HuggingEnvs display names/links to FineEnvs where the destination exists; verify remote links before publishing.
7. **Visuals:** retain the distinct baseline / SFT / RL sections. Show incomplete and overlap flags. Avoid ranking methods only by their best result, or removing the Qwen reward-objective annotation. Use the accompanying tables/JSON to build HTML visualizations; this handoff does not require image interpretation.

Useful next experiments can be proposed as future work: compare harness mixtures on the same 692 common tasks, equalize supervised target-token exposure, balance task/rollout contributions, and compare SFT→RL against RL from base. None of those controlled comparisons has been run in this packet.

### Claims to avoid

- “SFT fails on LFM”: OpenCode SFT improves to 47.5%; mixed SFT is different.
- “Mixed SFT proves harness diversity causes the gain”: data volume, targets and compute differ.
- “RL always beats SFT”: Qwen final RL scores are below mixed SFT, while best RL checkpoints are above it.
- “Qwen's efficiency reward failed”: Qwen did not have one.
- “More efficient means cheaper to solve the whole benchmark”: savings are conditional on pairs solved by both, and tokens/latency are different metrics.
- “TiTO passing proves the run is statistically or operationally perfect”: it validates specific capture checks.
- “All evaluations are complete”: the current teacher-trace SFT evaluations are complete, but historical 907-task SFT, LFM baseline and hard-task continuation have missing cells.
- “Full bash SFT is held out”: it includes test-notebook and question overlap by explicit user choice.
- SFT metrics originally logged locally were backfilled to the public Trackio project on 29 September 2026. See TRACKIO_PUBLICATION.md for the verified dashboard and commit-pinned citation.

## 8. Artifacts and exact provenance

### Public entry points

- Existing article reference: https://huggingface.co/spaces/FineEnvs/multi-harness-rl . Historical user-facing URL: https://huggingface.co/spaces/AdithyaSK/multi-harness-rl . Verify the current destination before publishing.
- Article review PR: https://github.com/adithya-s-k/FineEnvs/pull/13 . Inspect its current head before applying this handoff.
- Historical RL dashboard: https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio/?project=data-agent-rl-comparison . The exact score artifacts in this packet take precedence over a stale dashboard view.
- Portable SFT dataset: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft . Configurations: `all`, four individual harnesses, `lfm25_2_6b`, `qwen35_2b`.
- Published training script: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft/blob/main/train_sft.py . Requirements and verification receipt live alongside it. Data/code revision tested during publication: `a57fbba49e892199d2b0c31f68034310bf2e281c`.
- Raw teacher trajectories: https://huggingface.co/datasets/AdithyaSK/qwen38-27b-harbor-rollouts . Bash demonstrations: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft (revision `a9fa95ab5dff4f4522081f7eb003109e56b7a36d`).

The Hub SFT example provides a convenient simplified epoch-save recipe. The exact historical cluster runs additionally saved every 50 updates and used the local gated pipeline. Do not claim the public script automatically reproduces the cluster scheduler, full evaluation orchestration or all save policies.

### Local run roots and checkpoints

- **LFM bash SFT, matched 907**: [run root](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-sft-20260927). Checkpoints: `production/run/checkpoint-114` and `checkpoint-228`. Metrics: [metrics.jsonl](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-sft-20260927/production/metrics.jsonl). Local Trackio: `production/trackio/`. Evaluations: `production/evaluations/checkpoint-<step>/`.

- **LFM bash SFT, full 4,677**: [run root](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-sft-full-20260927). Checkpoints: `production/run/checkpoint-585` and `checkpoint-1170`. Metrics: [metrics.jsonl](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-sft-full-20260927/production/metrics.jsonl). Local Trackio: `production/trackio/`. Evaluations: `production/evaluations/checkpoint-<step>/`.

- **LFM OpenCode SFT**: [run root](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/lfm). Checkpoints: `production/run/checkpoint-604` and `checkpoint-1208`. Metrics: [metrics.jsonl](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/lfm/production/metrics.jsonl). Local Trackio: `production/trackio/`. Evaluations: `production/evaluations/checkpoint-<step>/`.

- **Qwen OpenCode SFT**: [run root](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/qwen). Checkpoints: `production/run/checkpoint-604` and `checkpoint-1208`. Metrics: [metrics.jsonl](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/qwen/production/metrics.jsonl). Local Trackio: `production/trackio/`. Evaluations: `production/evaluations/checkpoint-<step>/`.

- **LFM multi-harness SFT**: [run root](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/lfm). Checkpoints: `production/run/checkpoint-2242` and `checkpoint-4484`. Metrics: [metrics.jsonl](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/lfm/production/metrics.jsonl). Local Trackio: `production/trackio/`. Evaluations: `production/evaluations/checkpoint-<step>/`.

- **Qwen multi-harness SFT**: [run root](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/qwen). Checkpoints: `production/run/checkpoint-2242` and `checkpoint-4484`. Metrics: [metrics.jsonl](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/qwen/production/metrics.jsonl). Local Trackio: `production/trackio/`. Evaluations: `production/evaluations/checkpoint-<step>/`.

The four teacher-trace student runs have `evaluation_verified.json` at both epoch checkpoints. For the two old incomplete bash-907 evaluations use `eval_progress.json`, not an invented complete receipt. Completion/optimizer evidence is in each production directory; mixed SFT has `training_verified_step_4484.json` and `eval_controller_summary.json`.

- Exact OpenCode SFT code: [code directory](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/code) and [final_config.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/final_config.json).
- Exact mixed SFT code: [code directory](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/code) and per-model `production/config.json`; [CONFIG.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/CONFIG.json) is a historical plan whose status field is stale.
- LFM RL roots: [OpenCode RL](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-medium-hard1000-20260921/opencode) and [multi-harness RL](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-medium-hard1000-20260921/multi-harness); configs, per-allocation train audits and canonical checkpoint scores are inside.
- Earlier Qwen explanation: [results.md](/fsx/adithyaskolavi/projects/trl_prod/HuggingEnvs/04-data-agent/results.md) and [REPORT.md](/fsx/adithyaskolavi/projects/trl_prod/HuggingEnvs/04-data-agent/reports/three-run-analysis-20260917/REPORT.md) / [TRAINING.md](/fsx/adithyaskolavi/projects/trl_prod/HuggingEnvs/04-data-agent/reports/three-run-analysis-20260917/TRAINING.md).
- Dataset publication evidence: [RESULTS.md](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-sft-hub-20260928/RESULTS.md); full source export and Hub verification receipts live alongside it.
- All-run plotting code: [plot_all.py](/fsx/adithyaskolavi/projects/trl_prod/experiments/sft-rl-complete-comparison-20260928/plot_all.py). Native-tool extraction and plots: [tool_efficiency.py](/fsx/adithyaskolavi/projects/trl_prod/experiments/sft-rl-complete-comparison-20260928/tool_efficiency.py).
- Latest correctness figures and HTML: [artifact directory](/fsx/adithyaskolavi/projects/trl_prod/experiments/sft-rl-complete-comparison-20260928/20260928T193636Z). Latest efficiency figures and method report: [artifact directory](/fsx/adithyaskolavi/projects/trl_prod/experiments/sft-rl-complete-comparison-20260928/tool-efficiency-20260928T193640Z). They are local artifacts, not published image URLs.

### Exact score source for each displayed row

- **Qwen3.5-2B, Pretrained baseline:** [canonical_results.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-baseline-20260914/job-78215/canonical_results.json)

- **Qwen3.5-2B, OpenCode SFT · 801 rollouts / E1:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/qwen/production/evaluations/checkpoint-604/evaluation_verified.json)

- **Qwen3.5-2B, OpenCode SFT · 801 rollouts / E2:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/qwen/production/evaluations/checkpoint-1208/evaluation_verified.json)

- **Qwen3.5-2B, Multi-harness SFT · 3,189 rollouts / E1:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/qwen/production/evaluations/checkpoint-2242/evaluation_verified.json)

- **Qwen3.5-2B, Multi-harness SFT · 3,189 rollouts / E2:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/qwen/production/evaluations/checkpoint-4484/evaluation_verified.json)

- **Qwen3.5-2B, Harbor OpenCode RL / best, 700:** [scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916/checkpoint-evals/step-000700/scores.json)

- **Qwen3.5-2B, Harbor OpenCode RL / final, 1000:** [scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916/checkpoint-evals/step-001000/scores.json)

- **Qwen3.5-2B, Harbor multi-harness RL / best, 500:** [scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915/checkpoint-evals/step-000500/scores.json)

- **Qwen3.5-2B, Harbor multi-harness RL / final, 1000:** [scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-cont-20260915/checkpoint-evals/step-001000/scores.json)

- **Qwen3.5-2B, Native OpenCode RL / best = final, 1000:** [canonical_scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/daytona_harness_comparison/logs/hf-20260915/local-opencode-smoke-v4/repro/outputs/local-eval-opencode-81098/canonical_scores.json)

- **Qwen3.5-2B, Hard-task RL continuation / 1000 *:** [eval_progress.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/hard500-local-evals-20260921/checkpoint-1000/eval_progress.json)

- **LFM2.5-2.6B, Pretrained baseline *:** [final_tito.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-multi4-20260921/eval-83716/final_tito.json)

- **LFM2.5-2.6B, Bash SFT · 907 tasks / E1 *:** [eval_progress.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-sft-20260927/production/evaluations/checkpoint-114/eval_progress.json)

- **LFM2.5-2.6B, Bash SFT · 907 tasks / E2 *:** [eval_progress.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-sft-20260927/production/evaluations/checkpoint-228/eval_progress.json)

- **LFM2.5-2.6B, Bash SFT · full 4,677 / E1 †:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-sft-full-20260927/production/evaluations/checkpoint-585/evaluation_verified.json)

- **LFM2.5-2.6B, Bash SFT · full 4,677 / E2 †:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-sft-full-20260927/production/evaluations/checkpoint-1170/evaluation_verified.json)

- **LFM2.5-2.6B, OpenCode SFT · 801 rollouts / E1:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/lfm/production/evaluations/checkpoint-604/evaluation_verified.json)

- **LFM2.5-2.6B, OpenCode SFT · 801 rollouts / E2:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/opencode-student-sft-20260927/lfm/production/evaluations/checkpoint-1208/evaluation_verified.json)

- **LFM2.5-2.6B, Multi-harness SFT · 3,189 rollouts / E1:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/lfm/production/evaluations/checkpoint-2242/evaluation_verified.json)

- **LFM2.5-2.6B, Multi-harness SFT · 3,189 rollouts / E2:** [evaluation_verified.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/multiharness-student-sft-20260928/lfm/production/evaluations/checkpoint-4484/evaluation_verified.json)

- **LFM2.5-2.6B, Harbor OpenCode RL / best, 900:** [canonical_scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-medium-hard1000-20260921/opencode/eval-84180/canonical_scores.json)

- **LFM2.5-2.6B, Harbor OpenCode RL / final, 1000:** [canonical_scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-medium-hard1000-20260921/opencode/eval-84206/canonical_scores.json)

- **LFM2.5-2.6B, Harbor multi-harness RL / best, 700:** [canonical_scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-medium-hard1000-20260921/multi-harness/eval-84201/canonical_scores.json)

- **LFM2.5-2.6B, Harbor multi-harness RL / final, 1000:** [canonical_scores.json](/fsx/adithyaskolavi/projects/trl_prod/experiments/lfm25-medium-hard1000-20260921/multi-harness/eval-84475/canonical_scores.json)

### Machine-readable packet

- `results.json`: all displayed rows, all recorded comparison checkpoints, source hashes and caveats.
- `all_checkpoints.csv`: full available checkpoint history, with completeness and overlap flags.
- `tool_efficiency.json` / `tool_efficiency.csv`: matched-pair counts, before/after native call means, per-harness savings, and references to audited cohort caches with trajectory hashes.
- `training_summary.json`: run roots, optimizer steps, measured training runtime, training loss windows and metrics hashes.
- `manifest.json`: checksums for the packet and the exact input snapshots.

Local absolute paths identify the original evidence. If handing this to an agent without this filesystem, send the entire packet directory; its summary and tables are self-contained. Raw traces, full checkpoints and tokenized training datasets remain at the linked paths and are not copied into this packet.

## Suggested short message to the article-writing agent

Please update the existing multi-harness article to cover SFT as well as RL using this evidence packet. Keep the architecture and earlier RL history, then add a model-by-model comparison of baseline, both SFT epochs, and best/final RL. Qwen multi-harness SFT reaches 32.7% from 14.6%; LFM OpenCode SFT reaches 47.5%, mixed SFT 43.1%, and mixed RL finishes at 54.2%. Include native tool-call savings on matched successes, with sample sizes. Explicitly state that Qwen RL was correctness-only, LFM RL used an efficiency bonus, and SFT used imitation. Preserve overlap/incomplete flags and avoid causal claims because data and budgets differ. Use the HTML/JSON tables and exact source links, not numbers read from images. Check the current article branch before editing and do not overwrite newer material.

## SFT runtime metrics and public dashboard

Updated 29 September 2026: SFT has moved to its own public CPU-basic Space. The original comparison Space defaults to RL. Prior SFT imports remain archived in its raw database; no records were deleted.

- [SFT dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-sft-trackio)
- [SFT overview](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28eval_observed%2F%28pass_at_1%7Ctool_call_savings_pct%29%7Ctrain_sft%2Floss_mean50%7Ceval_coverage%2Fgraded_cells%29%24)
- [Training curves and runtime metrics](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28train_sft%7Ctrain_summary%29%2F)
- [Evaluation scores, tools and tokens](https://fineenvs-data-agent-sft-trackio.hf.space/?project=data-agent-sft-comparison&run_ids=f036025d14dd1296b49ce58fe500ec3d%2Ca4cf64db18a6b31a095ebbf9ab0a6b68%2C1bb4e31576c25d10314f48dc1a762a96%2C7dd987e57d95f45535307e5635ede2b0%2C7db29f7f7bf5dc725c86f83c2761a007%2Ca568c52ea8521b301f8155bed9b43826&smoothing=0&metric_filter=%5E%28eval_observed%7Ceval_coverage%7Ceval_usage%7Ceval_efficiency%29%2F)
- [Commit-pinned article citation](https://huggingface.co/spaces/FineEnvs/data-agent-sft-trackio/blob/fa45c93c5b987dfc51ee9758d2bc3786e5fdd8f0/SFT_RESULTS.md)
- [RL dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio)
- [Local results table](SFT_RESULTS.md)

All 12,806 SFT scalar records were verified by exact public API readback. This includes 12,782 optimizer updates, six runtime summaries and 18 evaluation records including repeated baselines. Dense plotting responses now retain roughly 800 training points per run; every evaluation and summary point remains visible. Raw data remains complete and can be queried or downloaded. The dedicated SFT Space restores its entire snapshot from a versioned archive on startup.

The original Space was RUNNING with HTTP 200 and no crash in the available logs when checked. Unlimited plotting responses were a plausible browser bottleneck, not a confirmed server crash. Separation and bounded plotting reduce the rendering load.

Runtime excludes queue, preprocessing and evaluation. Historical logging timestamps are ingestion times. Partial bash-907 evaluations and overlap-inclusive bash-4,677 evaluations remain flagged. No private rollout text or model weights were published. See TRACKIO_PUBLICATION_ORIGINAL.md for the earlier combined-dashboard publication audit.

RL dashboard UI update: the default page now opens six LFM overview charts, with navigation for Qwen, harness scores and tool/token usage. A real browser check verified the overview, Qwen and harness views with no JavaScript errors. The former unrestricted view rendered 426 canvases; all diagnostics remain accessible under All RL metrics.
