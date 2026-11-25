# Step 0 · 22 October 2026

**Status: implementation verified offline; production dataset gates remain open.** Human review is
still pending. No labels have been approved, no model collection has launched and no training has run.
Evidence and immutable revisions are recorded in [source_checks.json](../data/source_checks.json).
`python tools/check_sources.py --fetch` refreshes public snapshots without accepting access gates.

## Source checks

| Source | Pinned revision | Licence/access observed | Import status |
|---|---|---|---|
| τ²-bench | `05736c27219b769d0be507574dd3135010916644` | [Repo MIT licence](https://github.com/sierra-research/tau2-bench/blob/05736c27219b769d0be507574dd3135010916644/LICENSE). This is task/framework licensing; generated exports also need collection provenance. No downloaded third-party runs are assumed licensed. | Native text Results join and explicit DB-hash exports supported; collection unlaunched. |
| SWE-rebench/OpenHands | `35455389ab51bf5e2306bfd436ef72d0f98bf882` | [Card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/blob/35455389ab51bf5e2306bfd436ef72d0f98bf882/README.md): CC-BY-4.0, public. | Native chat/0-or-1 resolved converter and bounded pinned fetcher implemented. |
| AgentRx | `88e871fecb58b2d090449f37ec80b8865594e0b5` | [Public metadata](https://huggingface.co/api/datasets/microsoft/AgentRx/revision/88e871fecb58b2d090449f37ec80b8865594e0b5): CC-BY-4.0; auto gate. Raw card access returned 401. | No gate accepted or rows downloaded. Explicit annotation/trace join supported; native gated joins unverified. |
| TRAIL | `b424ce63d5973d5dcd7169b1bc3c07ccdee276d1` | [Public metadata](https://huggingface.co/api/datasets/PatronusAI/TRAIL/revision/b424ce63d5973d5dcd7169b1bc3c07ccdee276d1): MIT; auto gate. Raw card access returned 401. | No gate accepted or rows downloaded. Same explicit join contract; native joins unverified. |
| Who&When | `59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c` | [Card](https://huggingface.co/datasets/Kevin355/Who_and_When/blob/59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c/README.md) has no licence declaration. | Licence unresolved; no admission or redistribution. Explicit join path prepared only. |
| MAST | `95118ac951421753cf1deb87ddea3b01e693c41b` | [Card](https://huggingface.co/datasets/mcemri/MAST-Data/blob/95118ac951421753cf1deb87ddea3b01e693c41b/README.md): CC-BY-4.0. | Analysis-only; model-produced annotations never become v1 reward labels. |
| ABCD | `6b8700ce67c6b37b062dd7a60abc76d7ef832a97` | Retained MIT licence, pinned hashes and raw workflow files in `data/sources/abcd/`. | All 10,042 conversations normalized and officially validated locally. All outcomes remain null. |

## τ²-bench eligibility: the planned domain mix is not yet viable

The pinned [evaluation documentation](https://github.com/sierra-research/tau2-bench/blob/05736c27219b769d0be507574dd3135010916644/docs/evaluation.md)
distinguishes final-state DB checks from action matching and semantic communication checks. We accept
`DB` alone, or `DB+COMMUNICATE` only when communication requirements are empty and therefore neutral.
We do not remove a criterion to make a task eligible.

| Domain | Tasks inspected | Eligible | Observation |
|---|---:|---:|---|
| airline | 50 | 44 | Six require communication judging. |
| telecom | 2,285 | 0 | 2,253 use ENV_ASSERTION; 32 use ENV_ASSERTION plus ACTION. Neither is DB-only. |
| retail | 114 | 2 | Only task IDs 33 and 34 qualify; other tasks include NL_ASSERTION. |

Task file URLs, hashes and the full selected ID lists are in `source_checks.json`. This is a complete
file-level eligibility check, not a count of generated runs. The spec's airline/telecom training mix
and broad retail holdout cannot be filled at this revision under its DB-only rule. An owner decision
is needed before collection: choose another verified revision/task set or explicitly revise the
allowed outcome checks. The implementation keeps the stated rule intact.

Labels use independent captured expected/actual DB hashes when available, otherwise the native
boolean `reward_info.db_check.db_match`, which represents final-state equality. Neither aggregate
`reward`, `db_reward` nor reference action matching establishes the label. Alternate paths remain valid.

The pinned [evaluator](https://github.com/sierra-research/tau2-bench/blob/05736c27219b769d0be507574dd3135010916644/src/tau2/evaluator/evaluator.py)
returns zero reward without a DB check for premature termination, including maximum steps or too
many errors. Those runs are retained with null outcomes and explicitly counted as
`unverified_early_stop:<reason>` unless independent final-state evidence is available. The stop
reason alone is not converted into a DB outcome label. Telecom user-side calls now import with
their actor and tool-result links preserved, while the tasks remain criteria-ineligible.

## SWE outcome semantics and remaining audit

The pinned card defines `resolved` as binary issue success, distinct from submission status and
generated-test metrics. [The authors' collection report](https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b)
describes test-validation infrastructure and filtering patches that fail to apply. We preserve the
published outcome as `checker.kind=published_test_outcome`, not an independently rerun proof.
Unknown/missing values stay null. Test-suite success can still hide incorrect patches. The exact
checker implementation, replay checks and 50-patch manual error-rate audit remain pending; no claim
of independent semantic correctness is made.

The first 50 pinned streaming rows were fetched, converted and officially validated locally: 27
published failures and 23 successes. They are a setup sample, not a representative evaluation set.
All 50 exceed the 32k v1 cap under `cl100k_base`: minimum 34,580, median 57,727, maximum 91,932 tokens
for the complete rendered observation plus auditor system prompt. They remain available for review
but none enters the v1 reward at that budget. The target auditor's tokenizer is not selected yet;
`--encoding` supports explicit model-token counts. No trace was truncated or budget silently widened.

The updated importer preserves 236 explicit nonzero shell-status footers as command status, not tool
errors. No explicit tool-error flags are present in this source sample; the last reported shell
status is nonzero in 4/50 traces. The new export has 46 train and 4 dev
records, zero test records; §7.1's source table is clarified to match §7.3's default source holdout.

The latest ABCD import has 8,496 train and 1,546 dev records, no test records, and all outcomes null.
Legacy ABCD test-family confirmations are migrated after the original review hashes are checked;
the migration is reported and original source splits remain private provenance.

The configured outcome comparison requires mixed tau2 retail and task-matched AgentRx retail;
TRAIL and Who&When remain diagnostics and cannot deadlock that retail comparison. AgentRx cases
and verified success controls must share explicit canonical task families, with equal outcomes per
selected task. Native IDs are not inferred to match. Only tasks 33 and 34 could currently supply
eligible controls; whether AgentRx has compatible cases remains unknown until access and mapping
are verified. Missing required matches makes the comparison inconclusive. Task-only separation
also makes it inconclusive. A qualifying trace baseline needs 80% specificity as well as
precision/recall, with AUROC above 0.5 in both required cohorts. These unchanged v2.1 numeric
criteria still await owner sign-off. Retail-only results cannot establish all-source adequacy.
Control reuse and case/control calibration limits remain explicit. Data shortages are unchanged.

## Actual attribution counts

AgentRx metadata exposes annotation splits `tau_retail` and `magentic_one`, plus separate trajectory
files, but no verified row counts. The 115-versus-170 discrepancy remains **unresolved** until an
authorized export can be counted and joined. TRAIL publicly advertises 117 GAIA and 31 SWE-bench rows;
these are **published metadata counts**, not local verified counts. Who&When's card advertises 184
failure tasks, also not locally counted. Do not turn paper/card counts into an ingestion claim.

## Conversion and validation

No upstream source-specific ATIF converter was used or certified. Local adapters normalize SWE chat,
τ² text results and ABCD turns. Attribution inputs require an explicit original-step-to-ATIF-step join;
no annotation offsets are guessed. Gated native joins remain pending access and schema verification.

The ATIF schema and official pure Pydantic models/validator are pinned to Harbor commit
`209a48a7424218828465c3c716b042f2b381dbed`. [Provenance](../data/schema/provenance.json) records all hashes;
the schema is restricted to ATIF-v1.7. The upstream Apache-2.0 licence is retained. Offline validation passed
for all **37 synthetic fixtures**, **10,042 normalized ABCD conversations**, and **50 SWE traces**. Source trajectories
are never executed by this project.

## Human review and release gates

- A review owner has been named. Queue creation leaves every confirmation false and every
  human label null. A 50-record ABCD queue and companion evidence packet were generated locally.
- Owner sign-off on SPEC v2.1's numeric gate policy is not recorded. Numeric policy approval is
  tracked in [SPEC](../SPEC.md); a conditional baseline result is not approval to train.
  The design (a write-presence baseline plus separate written-run checks) is chosen; numeric
  thresholds, audits and release approval remain separate.
- The required 50-label audit per scored source is **not done**. Existing verified labels need an
  independent correctness review, not just agreement with their stored outcome field.
- A 50-record SWE review queue and companion packet were also generated locally, with all human
  confirmations false. A first-N setup sample is not representative of the full benchmark; final
  auditing needs a documented representative draw. Its length exclusions further limit v1 coverage.
  For the enforced SWE training guard, count every revision-matched human review, whatever its
  split and whether or not it fits the rendering/token budget. Only revisions with grader-labelled
  records in the training pool need the audit. A reviewed record need not be selected into each
  particular sampled batch. Importing new provenance changes fingerprints:
  regenerate review queues rather than applying stale approvals.
- AgentRx and TRAIL require owner-approved access/export; Who&When needs its licence resolved.
- τ² generation needs both agent/user models, a budget-enforced runner and the eligibility decision above.
  The collection plan now targets repeated trials for each task/model pair, with mixed outcomes and
  both written successes and written failures. It reports raw counts, paired retention fraction and
  collection shortfalls; strict pairing retains only twice the smaller class count. Retain every
  raw run, stop at the collector's trial/budget limit and never manufacture a missing outcome.
  Broad held-out task coverage is also needed for the task-cluster bootstrap intervals;
  repeated trials on the two eligible retail tasks cannot substitute for more independent tasks.
  The current configurable guard needs eight task families before producing bounds. This numeric
  default awaits signoff. The previous Hoeffding bound is replaced; regression comparisons with
  strong signal can pass at eight or twenty synthetic tasks rather than requiring hundreds.
- Prompted-judge baseline, real heldout-domain/source coverage, source/length/tool-count/language
  shortcut checks and human label error rates remain pending. Source identity masking does not prove
  statistical anonymity: domain/tool vocabulary can remain predictive.
- No fixture baseline score satisfies a release or RL gate. There is no training run in this scope.
- Training still lacks an admissible second source: the sampler's 50% source cap correctly rejects
  airline-only batches. No source cap, label rule, or rendering budget has been relaxed to admit data.
