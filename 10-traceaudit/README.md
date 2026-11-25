# TraceAudit

An auditor reads a completed agent trace and predicts whether the run failed. Sources supply
recorded evidence; the auditor never takes over the original task. The numeric gate policy is
pending sign-off.

The implementation is runnable offline. Production dataset readiness is **pending**: human review,
gated-source access, Who&When licence clarification, and a revised collection mix are outstanding.
No model training or paid collection has run. Human-reviewed labels still need sign-off from a named reviewer.

<!-- BEGIN:matrix -->
| Env | Tools | Backend | `verifiers` |
|---|---|---|---|
| **trace_audit** | 0 | `none` | ✅ |
<!-- END:matrix -->

## Run

From this directory, with Python 3.11–3.13:

```bash
python envs/trace_audit/verifiers/rollout.py
```

This uses 37 clearly marked synthetic plumbing fixtures, including recovered errors, alternative
successful paths, false completion and an unlabelled case. It writes six baseline reports and a
reliability plot to `results/local/`. These scores are **not benchmark results**.
The rollout also prints one rendered episode with its response and reward, and saves it in the
JSON report's `episode` field.

For the native adapter and checks:

```bash
cd envs/trace_audit/verifiers
uv sync --frozen --extra native --extra dev
uv run python ../../../tools/validate_harbor.py --fetch
uv run python -m pytest ../../../tests -q --basetemp ../../../.cache/pytest
uv run python rollout.py
```

Dependency installation and `--fetch` are setup steps that use the network. Tests, the default
rollout and subsequent validation are offline. The pinned upstream Harbor files are hash checked.
See [the port README](envs/trace_audit/verifiers/README.md) for native use.

## Episode and reward

The observation contains task instruction, policy, tool definitions and the complete numbered trace.
When the task exactly matches a user message, `task.instruction_step_id` points to that preserved
step instead of repeating the text. The task-only baseline resolves this pointer.
Known agent/model/source identities are masked; source metadata, gold outcomes, grader metrics,
annotations and reference success criteria are excluded. Domain vocabulary can still identify a
source statistically; the source probe reports this residual signal as a release concern.

The auditor emits one JSON object:

```json
{"p_failed": 0.85, "critical_step": 4, "category": "false_completion", "evidence_steps": [3, 4]}
```

Only a finite numeric `p_failed` in `[0,1]` is rewarded: `1 - (p_failed - failed)^2`.
Invalid required output earns zero. Invalid optional fields are logged and ignored. They never
change the reward. Each episode terminates on its first submission. Reset creates a new episode.
Unlabelled records are retained privately and excluded from scoring.

V1 exposes no inspection tools. Long traces are excluded whole, never truncated. The default 32k
budget uses UTF-8 bytes as a conservative token upper bound; the core accepts a target tokenizer
callable. Use the same counting policy for preparation, baselines and training. Multimodal traces
and unresolved continuation files require explicit conversion before admission.

For actual model-token counts, install `tiktoken` in your runner and use `--encoding cl100k_base`
(or the selected auditor's encoding) with data import/validation and rollout. Cache the encoding
once during setup before running offline. The native factory accepts the same `token_counter`
callable. The 50 fetched SWE setup traces all exceed 32k even under `cl100k_base`; they are review
inputs, with zero episodes admitted at this cap.

## Prepare data

Adapters normalize to our wrapper plus ATIF v1.7, preserve original step mappings privately and
reject ambiguous joins.

```bash
# Public source metadata and tau task selection; does not accept any access gate.
python tools/check_sources.py --fetch
python tools/tau2.py plan

# Restore retained ABCD inputs, then normalize them as unlabelled evidence.
python adapters/fetch_abcd.py
python tools/data.py import --source abcd --input data/sources/abcd/raw/abcd_v1.1.json.gz --output data/generated/abcd_v1.jsonl --revision 6b8700ce67c6b37b062dd7a60abc76d7ef832a97 --license MIT
python tools/validate_harbor.py data/generated/abcd_v1.jsonl

# Bounded public SWE sample (requires the native extra for datasets).
python adapters/fetch_swe.py --limit 50
python tools/data.py import --source swe --input data/generated/swe_native.jsonl --output data/generated/swe_v2.jsonl --revision 35455389ab51bf5e2306bfd436ef72d0f98bf882 --license CC-BY-4.0
python tools/validate_harbor.py data/generated/swe_v2.jsonl
```

All raw/generated data and local review packets are ignored by Git. Imports write a source/input
hash and per-source admission report. Do not treat ABCD human conversations as agent outcomes or
infer failure from its workflow path. Its workflow context is retained separately; review only
what the available task/policy evidence establishes.

Task family hashing keeps repeated runs together. Airline, telecom, SWE and ABCD use an 85/15 train/dev
split, with no task families sent to test. Scored test data is restricted to retail, AgentRx, TRAIL
and Who&When under the default protocol. `trace_audit.core.sampling.balanced_batch` draws equal success/failure
pairs within each source, domain, task family, agent model and user-simulator model, then enforces
the dataset source cap. Missing model provenance and strata lacking either outcome are excluded
and counted, with no fallback across models. Tasks receive pair tickets in proportion to their
smaller outcome count. `max_record_reuse=1` prevents replacement within a sampled dataset/batch;
larger explicit limits are enforced and reported alongside observed maximum reuse and unique count.
An impossible requested size raises an error instead of filling with scarce traces.
Reward remains unweighted. Evaluate calibration on the original dev/test distribution.
Pairs are indivisible: an odd rounded source cap leaves one slot unused, never a single-class slot.
Its default 50% cap still rejects one-source training; it does not silently relax data requirements.
Native Verifiers training and toolkit/default core training resets use this sampler after budget
admission. Default sample size is the largest feasible finite set of pairs under the cap and reuse
limit, with `training_size`, `source_cap`, `sampling_seed` and `max_record_reuse` available on the
framework factory. These limits apply per assembled sample, not across epochs or explicit episode
replays. Dev/test stay
unresampled. A trainer needing exact balance in each optimizer batch must call `balanced_batch` for
those batches; arbitrary reshuffling of the assembled dataset only preserves balance overall.

SWE grader-only training is blocked until 50 human-reviewed labels from the same pinned revision
are present with no observed grader disagreements. Reviews must retain their `apply-reviews`
history. If any audit disagrees, individually review all remaining training labels. A fully
human-reviewed source needs no grader-only admission. This enforces the pending audit; it does not
perform that human work or certify that tests capture every task and policy requirement.

Generating-model IDs stay in private `source.agent_model` and `source.user_model`; use the captured
simulator identity or explicit `not_applicable` when there was no simulator. Imports preserve null
for unknown metadata, which cannot participate in training. Tau2 Results metadata or explicit
`--agent-model`/`--user-model` collector arguments supply these fields; conflicting identities are
rejected. SWE stores its captured `source_model`. Both identities are masked from public text.

The rollout audits the exact paired training sample using the same sampler settings, reports raw
model-to-label associations, and labels the paired-model probe as a balance invariant (0.5 by
construction), not a residual-style detector. Write presence/count, length, tool count and closing
language are measured within task/model strata on unique sampled traces. Signed AUROCs pool across
strata before direction is summarized. A second, two-fold probe learns each task/model stratum's
feature direction on one half and evaluates it on the other, then swaps. This detects opposite
task-dependent rules without selecting direction on the evaluation outcomes. Strata need at least
two unique examples per outcome; skipped strata and evaluated records are reported. A high score
is diagnostic evidence and does not automatically reject the dataset.

The **owner-selected gate design** fits a write-presence baseline using unique
sampled train records only. Each required held-out cohort must show a task-mean Brier improvement
over it and AUROC above chance among runs that did write, with lower confidence bounds above zero
and 0.5 respectively. At least 10 unique pooled outcome pairs are required, including written
successes and failures. Missing classification or insufficient evidence is inconclusive. Intervals
use independent task-family clusters, so extra model variants and replays cannot increase apparent
task coverage. Intervals now use a deterministic percentile bootstrap of task means (2,000
resamples, seed 42), with a configurable minimum of eight task families before returning bounds.
This is approximate uncertainty, not a distribution-free guarantee. Sparse or degenerate bootstrap
support is reported explicitly; two retail tasks remain insufficient under the default guard.
The design (baseline plus written-run checks) is chosen; numeric thresholds (including the cluster
guard), human audits and release approval remain pending separately. A perfect synthetic auditor now passes the comparison at 8 and 20 task
families; no real-data performance or release readiness follows from those regression tests.
Use `--training-size`, `--source-cap`, `--sampling-seed` and `--max-record-reuse` to match a custom
trainer sample. With `return_report=True`, the native factory returns the environment and a sampling
report whose `shortcut_audit` measures its actual assembled dataset.
The core environment applies the same sample whitelist even to explicit training record IDs;
toolkit and core share one configured sample. Dev/test explicit resets use their original records.

Earlier normalized SWE exports contain an obsolete test split. Re-import the retained native rows
and create a fresh queue as above. The latest local export and queue use `swe_v2` filenames (export
generation only; the environment is still single-turn v1). Shell statuses now have a separate field; old
exports and review files were preserved. Record hashes prevent applying old approvals to new records.

## Human review

```bash
python tools/data.py review-queue data/generated/swe_v2.jsonl data/generated/swe_v2_review.jsonl
# A named reviewer reads the companion .traces.jsonl packet and edits the review queue.
python tools/data.py apply-reviews data/generated/swe_v2.jsonl data/generated/swe_v2_review.jsonl data/generated/swe_v2_reviewed.jsonl
```

A queue defaults to 50 randomly sampled records per source. Every entry starts `confirmed: false`,
with `failed: null`, no reviewer and no checker. Set `confirmed: true`, a boolean `failed`,
`label_source: expert` or `trace_evident`, `reviewer` and a concrete evidence/rubric `checker` only
after review. Optional `failures` must use normalized ATIF step IDs and the spec taxonomy.
The record hash prevents applying an old review to changed evidence. Prior outcomes remain in
private review history. The review report estimates disagreement against existing labels per
source; unlabelled records do not enter that denominator. MAST remains analysis-only in v1.

Legacy ABCD queues can still be applied to their original matching export. If a confirmed record
belongs to an ABCD test family, every member of that family moves to the deterministic train/dev
fold, with `split_history` and `split_migrations` in the report. All input hashes are checked before
any migration, the source split is retained, and other confirmations in the batch are processed.
Unconfirmed legacy test families stay unlabelled. New imports (`abcd_v1.jsonl`) avoid test entirely.
Create a new queue from the output before reviewing changed records again; do not reuse old hashes.

## Evaluation

```bash
python envs/trace_audit/verifiers/rollout.py --data data/generated/combined.jsonl --output results/local/real_baselines.json
```

Provide one validated file containing train, dev and test families, including both training outcomes.
The offline baselines are training prior, task-only TF-IDF, trace TF-IDF, closing keywords,
last-tool-error and a separate last-shell-nonzero-exit diagnostic. Nonzero shell status (e.g. grep
with no match or an intentionally failing test) is not labelled as a tool error. TF-IDF uses sparse
cosine class centroids; its scores are not calibrated posteriors. Thresholds are selected on dev,
with 80% precision/recall/specificity constraints when feasible;
fallbacks are explicitly reported. Test labels never choose thresholds.

Reports include per-source/label-source Brier, AUROC (null for single-class sets), precision, recall,
F1, calibration bins and SVG, false-success slices, source/length/tool-count/closing-language probes,
and optional human-gold localization/category diagnostics. A prompted judge can be compared from
saved JSONL `{record_id, response}` via `--predictions`; the output must cover every admitted record.
No judge is launched automatically. The RL gate stays pending until human audits, source coverage
and shortcut checks are reviewed; fixture success cannot open it.
Length/tool-count/closing-language probes report both raw AUROC and direction-free strength
`max(AUROC, 1-AUROC)`, so reverse shortcuts remain visible. Gate metrics use only the configured
held-out sources and exclude fixtures. Only tau2 retail and task-matched AgentRx retail are required
outcome cohorts. TRAIL and Who&When are reported diagnostics; missing or failure-only diagnostic
sets do not block the retail comparison. A missing or single-outcome required cohort is inconclusive.
AgentRx failure cases must share explicit canonical task families with verified successful tau2 retail
controls. Each selected task contributes equal successes and failures, using deterministic record-ID
selection independent of scores. The report lists matched tasks, selected/available counts, unused
controls and unpaired cases; no native task-ID equivalence is inferred. Reused controls make cohort
metrics dependent. The tau2 retail cohort is assessed separately. Task-only separation that clears
the numeric criteria makes the comparison inconclusive as a dataset shortcut.

A qualifying trace baseline needs 80% precision, recall and specificity plus AUROC above 0.5 in both
required cohorts. These thresholds await sign-off. The report's
`baseline_meets_minimums_all_outcome_cohorts` is scoped to the configured retail comparison; it makes
no all-source adequacy claim. The overall RL status remains pending policy, data and audits even
when the conditional comparison concludes. Retail controls are not borrowed for TRAIL or Who&When.
The tool-call probe counts both actors and embedded subagents. Calibration on matched case/control
cohorts describes their assembled class mix, not production prevalence. Balancing training also
changes its prior; assess calibration on the original dev/test distribution.

The source probe needs human assessment because semantic domain signals can survive metadata
masking. An admission report alone does not certify release readiness. Collection plans retain
native Results exports, policy and tool schemas; use final-state DBCheck to establish outcomes.
Read-only tasks need human review, and costs must include both agent and user-simulator models.
The pinned telecom source has no eligible DB-only tasks; retail has two.

From the repository root, run `python -X utf8 tools/build_index.py --check` on Windows so the index
tool's Unicode status output works independently of the console's default encoding.

## Results

Nothing has been trained yet, so there is no training curve. The rollout runs end to end on
synthetic fixtures only (`data/sample.jsonl`); those numbers check the plumbing, not the auditor.
The RL gate currently reports `inconclusive`: the held-out retail cohorts, the 50-label human
audits and sign-off on the gate thresholds are still pending. Next steps: collect tau2 runs,
finish the audits, compare the baselines with a prompted judge, then train.

## Layout

- `envs/trace_audit/core/`: validation, rendering, reward, review, sampling, metrics and shared evaluation;
  `tokenization.py` provides optional model-token counting.
- `envs/trace_audit/verifiers/`: single-turn native adapter, manual toolkit and runnable `rollout.py`.
- `adapters/`, `tools/`: source conversion, preparation, validation and review commands.
- `data/sample.jsonl`: synthetic fixtures; `data/source_checks.json`: pinned source evidence.
- `data/sources/abcd/`: retained source inventory/licence and ignored raw files.
- `tests/`: checks for the environment, data rules, reward and evaluation.

The original motivation is [Hamel's decision-model evaluation workflow](https://hamel.dev/blog/posts/decision-model-evals/):
human-aligned failure judgements, tuned thresholds and measured calibration. His private production
conversations are not copied into this project.
