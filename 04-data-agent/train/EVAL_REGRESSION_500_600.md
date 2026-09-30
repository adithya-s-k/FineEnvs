# Checkpoint 500 → 600 evaluation diagnosis

Read-only analysis on 2026-09-16. No trainer, evaluator configuration, or score was changed.

The observed decline is concentrated in Claude Code and Codex on medium and hard tasks. The evidence supports a model-behavior regression under the fixed evaluation budget. Its underlying training cause remains unproven; neither overfitting nor harness weighting should be presented as established.

| Harness | Step 500 | Step 600 | Previously correct → wrong | Previously wrong → correct |
| --- | ---: | ---: | ---: | ---: |
| OpenCode | 32.8% | 34.0% | 27 | 30 |
| Claude Code | 44.8% | 30.0% | 46 | 9 |
| Codex | 39.2% | 32.4% | 30 | 13 |
| Mini-SWE-Agent | 31.2% | 30.8% | 24 | 23 |
| Overall | 37.0% | 31.8% | 127 | 75 |

Both evaluations contain all 1,000 graded cells. Easy is unchanged at 96/132; medium falls from 209/472 to 176/472; hard falls from 65/396 to 46/396. The comparison uses the first valid grade per task/harness, retaining scored failures.

## What was checked

- Protocol JSON and the complete task-manifest entries are identical. Both use the same fixed 250 tasks, harness versions, temperature 0.8, 17-turn limit, 600-second timeout, and 4,096-token per-completion serving cap.
- Checkpoint provenance identifies the appropriate distinct model weights. Tokenizer, chat template, model configuration and generation-configuration hashes match.
- All 1,000 selected trials in each evaluation have verified harness versions and pass the token audit. All retained graded records have `ok=true` and no native `exception_info`. Ungraded infrastructure attempts did occur and were retried; these checks do not prove infrastructure has no effect on behavior or selection.
- Both checkpoints came from training job 79083. The later migration/resume from checkpoint 684 cannot explain this earlier decline.
- LR remains 3e-6. Mean reported KL is 0.00231 → 0.00243, gradient norm 1.83 → 2.14, and staleness 1.52 → 1.47 across updates 401–500 versus 501–600. There is no obvious numerical instability in these averages.

## Observed behavioral changes

Claude Code's mean captured generated tokens per evaluation rollout rises from 3,943 to 4,780 (+21.2%); Codex rises from 3,376 to 4,121 (+22.1%). Mean turn counts remain similar. This establishes longer outputs, not that per-completion truncation caused the score decline. Many rollouts already reach the turn budget at both checkpoints.

Inspection of the first three lost task indices for each affected harness found concrete model-side failures:

- Claude Code, task index 12: checkpoint 500 leaves `Glucose` in the answer file and scores correctly. Checkpoint 600 also writes `Glucose`, then overwrites it with a correlation coefficient and scores zero.
- Claude Code and Codex, task index 17: checkpoint 500 submits the correct answer. Checkpoint 600 continues data-loading/debugging calls until the budget stop.
- Codex, task index 26: checkpoint 600 attempts `write_stdin` with a `cmd` argument, repeatedly writes an incorrect numerical answer, and scores zero; checkpoint 500 submits the correct category.

These examples establish failure mechanisms for those tasks, not their prevalence across the test set. Repeated answer writes also occur in successful checkpoint-500 traces, so repetition alone is not an explanation for the decline.

A follow-up inspected both captures for every lost Claude Code/Codex cell (152 captures, 76 pairs). None has a `length` finish reason. Budget stops occur for 74/76 checkpoint-500 successes and all 76 checkpoint-600 failures; hitting the budget alone does not distinguish success. At least three identical tool calls occur in 60/76 checkpoint-500 captures versus 26/76 checkpoint-600 captures, so an increase in exact repetition is not supported. The failed checkpoint-600 trajectories contain more generated tokens overall, but the underlying reasoning/tool-use errors require task-specific interpretation.

## Training-side hypotheses

Mean logged training reward rises from 0.330 to 0.425 across the two 100-update windows while held-out performance falls. This is consistent with poorer generalization, but changing task exposure makes it insufficient evidence for overfitting.

Actual optimizer receipts show only about 15–17 distinct task groups per harness in each 100-update window. Overall harness rollout counts remain broadly balanced. Codex's hard-task rollout count falls from 32 to 8, while its medium-task count rises from 80 to 104. Recent task mix is therefore a plausible contributor worth testing. It does not explain Claude Code by itself: Claude's hard-task rollout count remains 64 in both windows.

The loss normalizes over supervised tokens per update. Equal scheduled task counts therefore do not guarantee equal influence by harness. Raw training-row counts must not be interpreted as proportional gradient weights; the atomic rollout implementation and token normalization matter. No causal weighting diagnosis has been established here.

## Confidence and next checks

A paired bootstrap over 250 task clusters, retaining all four harnesses per task, gives a 95% interval of approximately −8.1 to −2.2 percentage points for the observed −5.2-point change (20,000 draws, seed 0). This is uncertainty across tasks in these observed rollouts, not a repeated-generation experiment, and does not correct for selecting checkpoint 500 after inspecting multiple checkpoints. Temperature 0.8/pass@1 still introduces generation variability.

Keep checkpoint 500 as the best fully audited checkpoint so far. The next discriminating check is a separate paired repeat of checkpoints 500 and 600 with matched seeds and unchanged budgets, followed by a diagnostic of checkpoint 550 to localize the transition. Longer-budget diagnostics or a lower-LR branch should remain separate from the canonical fixed-budget evaluation and current run. No new GPU jobs or training changes were made for this diagnosis.

Evidence is in `experiments/daytona_harness_comparison/logs/hf-20260915/three-run-monitor/status-check-20260916/regression-500-600/analysis.json` and `lost-task-captures.json`. Sources are checkpoint-500 job 80639, checkpoint-600 job 80644, their retained capture paths and retry ancestry, and job 79083 optimizer receipts/token audits/metrics.


## September 17 follow-up: initial decline versus late OpenCode failure

The completed multi-harness scores are 22.7% at checkpoint 900 and 26.3% at 1,000. These later points include 7/16 measured verifier zeros recovered from agent-budget failures after full capture/native-result checks. Earlier accepted grades retain their historical retry policy; a uniform retrospective audit is needed for strict first-attempt comparisons. This accounting issue does not erase the observed incorrect answers and missing answer files.

The late failure is different from the initial 500→600 decline. OpenCode falls from 23.2% at 800 to 2.4% at 900 and 5.6% at 1,000. Existing capture analysis found 225/250 checkpoint-900 tasks ended at the 4,096-token completion cap without an answer file; checkpoint 1,000 had 202/250 cap-plus-missing-answer cases. In contrast, none of the 76 lost Claude/Codex pairs at 500→600 had a `length` finish reason.

There is a concrete budget mismatch to test. From the Blackbox-matching training configuration onward, `max_completion_length=16384` is passed to `MAX_OUTPUT_TOKENS` in the frozen launcher; serving uses it as the per-completion cap. Canonical evaluation remained at 4,096. Actual optimizer input confirms the higher budget was used: step 809 contains a successful OpenCode rollout with a 7,625-token tool-call completion (`job-80608/audit/rollouts/3dffd4b78c5f4a43b3c82300592823a7.json`). That behavior cannot fit in one eval completion. The mismatch existed before checkpoint 500, so it is not a new configuration change at the decline. It is a plausible contributor to the later verbosity/truncation failure, not a proven explanation for the initial decline.

Training metrics remain finite through 1,000, with LR 3e-6 and maximum recorded staleness 4. Mean logged completion-length metrics increase from roughly 3,911 at steps 701–800 to 9,480 at 801–900 and 10,657 at 901–1,000 (68/70/68 observations respectively). These are asynchronous logged metrics across mixed harnesses, not a fixed per-harness rollout cohort. KL/gradient averages show no obvious blow-up. Local window statistics are saved in `experiments/multi4-regression-20260917/training_windows.json`.

Recommended diagnostic order: preserve checkpoint 500; audit timeout/retry accounting uniformly; repeat paired 500/550/600 evaluations under the fixed protocol to localize the initial regression; separately test the 800/900 OpenCode failure with matched train/eval output budgets. Lower or decaying LR and balanced task/token contribution are ablations to test, not established fixes. No training configuration or live evaluation job was changed for this diagnosis.
