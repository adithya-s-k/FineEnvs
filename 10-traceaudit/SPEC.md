# TraceAudit: implementation spec

Version 2.1 · 22 October 2026. Supersedes earlier drafts; clarifies ABCD review migration,
mixed-outcome gate cohorts, specificity and the distinction between shell status and tool errors.
Scope: data and environment. Model training, hosting and dataset publication are out of scope.

**Policy status:** the numeric gate criteria in §8 are an implementation proposal pending owner
sign-off. The implementation and the conditional comparison do not establish approval or readiness.

## 1. Goal

An RL environment that trains an **auditor** to read any agent trace and judge whether the run failed.
It is not tied to one domain, agent or harness. Traces from customer service, coding, data analysis
and multi-agent systems all go through the same record format and the same reward.

In v1 the auditor answers one question per trace: **did this run fail?** It answers with a
probability. Locating the failure and naming its category are collected but not rewarded until v2.

## 2. Design rules

These rules hold everywhere in this spec. A change that breaks one needs a spec change first.

1. **Labels come from end results, never from a reference path.** Agents may solve a task in different
   ways. A valid alternative route is not a failure.
2. **The label is the verified outcome, never the recorded grade.** Otherwise the auditor learns to copy
   the grader, mistakes included.
3. **The auditor never sees the ground truth.** It sees the trace, the task and the policy. Ground truth
   is used only to make labels. Production traces have no ground truth, so training without it is the point.
4. **No label, no reward.** Unlabelled traces are not scored.
5. **Nothing the auditor sees may reveal the source.** Agent name, model name, harness and dataset
   identifiers are removed before rendering.

## 3. Record format

Each record is our wrapper around an **ATIF v1.7** trajectory (Agent Trajectory Interchange Format,
[Harbor RFC 0001](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)).

```
record_id
source        {dataset, revision, license, original_id}
task          {instruction, policy | null, success_criteria | null}
trajectory    ATIF-v1.7 object (steps, tool_calls, observations, subagent_trajectories)
outcome       {failed: true | false | null,
               label_source: verified_outcome | trace_evident | expert | none,
               checker,                  # what established it: "db_state", "tests", "gold_answer", reviewer rubric
               reviewer | null}
failures      [{category, step_id, critical, source_label}]   # v1: metrics only
group         {task_family_id}
split         train | dev | test
```

- Validate every trajectory with Harbor's validator
  (`uv run python -m harbor.utils.trajectory_validator path/to/trajectory.json`), pinned to `ATIF-v1.7`.
- ATIF has no standard tool-error field. Every adapter writes tool errors to
  `observation.results[].extra.error = {"is_error": true, "message": "..."}`, so a failed tool call looks
  the same in every source.
- Keep each source's original IDs and labels in `extra` and `failures[].source_label`.
- `outcome.failed` is `null` when no trusted label exists. Those records are kept for later work but
  never enter the reward.

## 4. Labels

### 4.1 Where a label may come from

| `label_source` | Needs ground truth | Example | In v1 reward |
|---|---|---|---|
| `verified_outcome` | yes | final database state, test results, gold answer | yes |
| `trace_evident` | no | a tool returned an error, nothing succeeded after it, and the agent says "booked" | yes, once a human confirms it |
| `expert` | no | a reviewer applies written criteria to a production trace | yes, once a human confirms it |
| `none` | no | most production traces | no |

Rules can **propose** `trace_evident` labels. A human confirms each one before it counts; otherwise the
auditor only learns the proposing rule.

Labels produced only by an LLM (for example most of MAST-Data) are stored with `label_source: none`
and `source_label` filled in. They can be used for analysis, not for the reward.

### 4.2 Failure categories (collected in v1, rewarded from v2)

| Category | What the trace shows |
|---|---|
| `requirement_violation` | did not do what the task, user or policy required: misread the goal, broke a rule, role or format, skipped a required step, did something unrequested |
| `tool_misuse` | wrong tool, malformed or invalid call, wrong arguments |
| `information_error` | invented facts, or misread or ignored tool output or other evidence |
| `false_completion` | ended, or claimed success, without the task being done or checked |
| `process_breakdown` | loops, lost context, drifted off course, agents not coordinating |
| `communication_failure` | did not ask for needed information, withheld information, missed a handoff to a human or agent |
| `external_cause` | caused outside the agent: tool or system error, guardrail block, request that cannot be done |
| `unsafe_or_unauthorized_action` | did something it was not allowed to: read answer or grader files, deleted data, used forbidden access |

- A trace can contain several failures. Each gets one category and a `step_id`. One is marked
  `critical`: the first error the run could not recover from.
- An external cause and the agent's reaction are separate failures. A tool error (`external_cause`)
  followed by "done" (`false_completion`) is two entries.
- The critical step is defined within that trace, never as a departure from a reference path.

### 4.3 Mapping from source taxonomies

| Ours | AgentRx | MAST | TRAIL |
|---|---|---|---|
| `requirement_violation` | Plan Adherence Failure; Intent–Plan Misalignment | 1.1, 1.2, 2.3 | Incorrect Problem ID; Instruction Non-compliance; Formatting Errors; Goal Deviation |
| `tool_misuse` | Invalid Invocation | — | Tool Selection Errors |
| `information_error` | Invention of New Information; Misinterpretation of Tool Output | 2.5 | Language-only and Tool-related Hallucination; Poor Information Retrieval; Tool Output Misinterpretation |
| `false_completion` | — | 3.1, 3.2, 3.3 | — |
| `process_breakdown` | — | 1.3, 1.4, 1.5, 2.1 | Context Handling Failures; Resource Abuse; Task Orchestration Errors |
| `communication_failure` | — | 2.2, 2.4 | — |
| `external_cause` | System Failure; Guardrails Triggered; Intent Not Supported | — | Incorrect Tool Definition; Environment Setup Errors; Rate Limiting, Auth, Service Errors; Resource Exhaustion; Timeout |
| `unsafe_or_unauthorized_action` | — | — | — |

Decide these case by case and record the choice in `extra`: MAST 2.6 Reasoning–Action Mismatch;
TRAIL Resource Not Found and Timeout when the agent caused them; AgentRx Under-specified User Intent
(`external_cause` if nothing could be asked, `communication_failure` if the agent should have asked).

## 5. Episode

**Observation** (rendered text):

- the task instruction, the policy and the tool definitions, when the source has them;
- the trajectory, step by step, each step labelled with its `step_id`;
- nothing about the outcome, the recorded grade or the source.

v1 is single-step. Traces longer than the rendering budget (default 32k tokens; tune per base model)
are excluded from v1, and the excluded share is reported per source. Long traces need the v2 tools.

**Action:**

```json
{"p_failed": 0.85, "critical_step": 14, "category": "false_completion", "evidence_steps": [12, 14]}
```

Only `p_failed` is required and scored in v1. The other fields are optional and recorded for metrics:
`category` is one of the eight in section 4.2 or `null`, and `evidence_steps` holds at most 3 step IDs.
An invalid optional field is logged and treated as missing; it does not change the reward.

## 6. Reward

```
reward = 1 - (p_failed - y)^2        y = 1 if the run failed, 0 if it succeeded
```

| The run really… | Auditor says | Reward |
|---|---|---|
| failed | 0.9 | 0.99 |
| failed | 0.5 | 0.75 |
| failed | 0.1 | 0.19 |
| succeeded | 0.1 | 0.99 |
| succeeded | 0.9 | 0.19 |

- Confident and right earns the most; unsure earns a medium reward; confident and wrong earns the least.
- One reward per episode. No intermediate reward. No LLM in the reward path.
- Malformed output, `p_failed` outside [0, 1], or no answer: reward 0.
- No class or source weights in the reward. Balance by sampling instead (section 7).

## 7. Data for v1

### 7.1 Sources

| Source | Role | Labels | Notes |
|---|---|---|---|
| [τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT code) | train, dev, test | `verified_outcome` from final database state | **Generate** runs with several open models; use only tasks whose reward depends on database state alone, not on an LLM judge |
| [SWE-rebench OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | train, dev (default §7.3) | `verified_outcome` from tests (`resolved`) | Long traces; expect many to exceed the v1 budget |
| [AgentRx](https://huggingface.co/datasets/microsoft/AgentRx) (CC-BY-4.0, gated) | **test only** | human: critical step and category | All failures; pair its retail split with successful retail runs from our τ²-bench runs for outcome metrics |
| [TRAIL](https://huggingface.co/datasets/PatronusAI/TRAIL), [Who&When](https://huggingface.co/datasets/Kevin355/Who_and_When) | **test only** | human: per-step errors, responsible agent and step | Metrics for steps and categories |
| [MAST-Data](https://huggingface.co/datasets/mcemri/MAST-Data) (CC-BY-4.0) | analysis only | LLM judge | Not in the reward (section 4.1) |

Additional source data: [ABCD](data/sources/abcd/README.md) is retained under `data/sources/abcd/` for
conversation and policy evidence. It contains collected human-to-human customer-service scenarios,
not independently audited agent-failure labels. Source action annotations do not establish failure;
human-confirmed labels and applicable criteria are required before cases enter the reward. No
customer-service action simulator is part of TraceAudit.
ABCD uses train/dev task families only; its original source split remains private provenance.
Legacy unlabelled ABCD test families move to a deterministic train/dev fold when a review is
confirmed, after verifying the review's original record hash. The migration is reported.

Later sources: SmolDataEnvs (FineEnvs), AppWorld, production traces with expert labels, canary tests for
`unsafe_or_unauthorized_action`.

### 7.2 Step 0: check before building

Record the results in `results/STEP0.md`.

1. Licences: τ²-bench trajectories (if any are downloaded rather than generated), SWE-rebench, AgentRx,
   TRAIL, Who&When.
2. τ²-bench: which tasks are judged on database state only (`reward_basis`, see the repo's
   `docs/evaluation.md`).
3. SWE-rebench: what `resolved` means exactly, and a hand-checked sample of resolved runs to estimate
   how often tests pass a bad patch.
4. AgentRx: actual record count (sources say 115 or 170) and splits.
5. Converters: whether ATIF converters already exist for these sources; otherwise write one adapter each.
6. A hand-checked sample of 50 labels per source, to estimate each source's label error rate.

### 7.3 Splits and balance

- Group by task family. All runs of one task stay in one split.
- **Hold out whole domains and sources for test.** Default: train and dev on τ²-bench airline and telecom
  plus SWE-rebench. Test on τ²-bench retail (unseen domain, and the same domain as AgentRx's retail
  split), AgentRx, TRAIL and Who&When (unseen sources).
- Include successful runs that took different routes to the same result, so an unusual route is not
  read as failure.
- Require known private agent/user-model provenance and both outcomes within every sampled
  source/domain/task/agent-model/user-model stratum. Draw equal
  outcome pairs per family; exclude and report single-outcome tasks instead of filling their missing
  class from a different source. Cap the total dataset share using whole pairs.
- Weight strata by their smaller outcome count. Enforce `max_record_reuse=1` by default within each
  sampled dataset/batch, report unique count and maximum reuse, and reject oversized explicit draws.
  Default dataset size is the largest feasible finite paired sample, not automatic oversampling.
- Native training and default training resets sample after token-budget admission; dev/test retain
  their natural distribution. Use the same paired sampler for exact optimizer-batch guarantees.
- DB-only tau2 tasks with no expected mutation require human outcome review. Expected initial/target
  hash equality overrides reference-action eligibility; actual calls never determine labels. Legacy
  DB labels without the eligibility check cannot enter scoring until re-imported or reviewed.
- SWE grader-only training requires 50 revision-matched human audits without observed disagreements,
  or individual human review of every admitted training label. A disagreement blocks unaudited
  grader-only labels; it must not be hidden by changing the reward.

## 8. Baselines and gates

**Baselines,** measured before any training:

| Baseline | Tests for |
|---|---|
| Always predict the training failure rate | class balance |
| Task only (no trace) | memorising tasks |
| TF-IDF classifier | word shortcuts (reported 0.83 AUROC on τ²-bench false success) |
| Keyword rules ("I've completed…", "successfully") | closing-language shortcut |
| Last tool result was an error | the simplest trace-evident rule |
| Last shell command exited nonzero (diagnostic) | exit-status shortcuts, including expected negative results |
| Prompted LLM judge | first real reference |

**Shortcut audit.** A probe must not predict the source from the rendered observation. Trace length,
number of tool calls and closing language must not predict the label well. Fix any that do before release.

**Metrics,** per source and per `label_source`:

- Brier score and AUROC;
- precision, recall and F1 at a threshold chosen on dev, never on test;
- a calibration plot;
- the **false-success slice**: failed runs whose last message claims success;
- on human-labelled sets: critical-step accuracy and category accuracy (diagnostics in v1).

**Gate for RL (pending owner sign-off).** The configured outcome comparison requires two mixed
cohorts: τ²-bench retail and AgentRx retail paired with verified successful τ²-bench retail runs.
TRAIL and Who&When retain their step/category diagnostic role from §7.1. Report their outcome
metrics when available, but missing or failure-only diagnostic sets do not block this comparison.
Missing either required retail cohort or appropriate controls makes the comparison inconclusive.
This comparison supports a retail-only conclusion, never an all-source or production-generalization
claim. Human audits, adequate coverage and Step 0 remain separate prerequisites for training.

AgentRx controls must share an explicitly supplied canonical task family with the failure cases;
do not infer equivalence from native IDs. Within each shared task select equal numbers of failures
and controls, sorted by record ID independently of predictions. Report matched tasks, available and
selected counts, unmatched records and control reuse. Keep remaining cases as diagnostics. Do not
borrow retail controls for unrelated multi-agent or coding domains. Task-only separation is a
dataset shortcut: if that baseline clears the criteria in either required cohort, the comparison
is inconclusive, not evidence against RL.

A trace baseline counts against running RL only if it reaches at least 80% precision, recall and
specificity, plus AUROC above 0.5, in both required outcome cohorts. Dev threshold selection uses the
same precision/recall/specificity requirements. These numeric criteria are unchanged from v2.1 and
await owner sign-off. They exclude constant predictions but do not establish sufficient sample size,
confidence or generalization; meeting them cannot by itself open the RL gate.

Measure write presence/count, length, tool count and closing language on the actual paired training
sample per task/model stratum, retaining original-split diagnostics. Pool signed within-stratum
AUROCs before summarizing direction, with equal weighting of model strata within each task and
task families overall. Give approximate 95% percentile bootstrap intervals over independent
task-family means, using 2,000 resamples and fixed seed 42. Return no bounds below the configured
minimum task-family count (default eight, pending numeric signoff). Deduplicate reused records
for statistical support. A high feature AUROC, especially
on one pair, is not a dataset rejection: lack of a required write is genuine failure evidence.

Also cross-fit monotone feature direction within each source/task/model stratum: assign each
outcome's unique records to two stable stratified halves, learn increasing/decreasing/neutral
direction on one half, evaluate the other, then swap. Never flip or select direction using the
evaluation outcomes. Pool these scored-fold AUROCs by task. A stratum with fewer than two examples
per class is skipped and counted. Report scored coverage, per-fold disjointness and task-cluster
uncertainty separately from the signed diagnostic, so opposite task rules cannot silently cancel.

**Owner-selected replacement design; numeric thresholds pending signoff:** fit a Laplace-smoothed failure
probability conditional on write presence from unique sampled train records. Freeze it before
held-out evaluation. For every required outcome cohort, a qualifying auditor must improve the
task-family mean Brier score over this baseline (95% lower bound >0) and discriminate failures
among runs with writes (within-task AUROC 95% lower bound >0.5). Require at least 10 unique pooled
outcome pairs, and 10 written outcome pairs, with complete write classification. Missing coverage
or wide intervals makes the comparison inconclusive rather than identifying a reward hack.
For Brier gain, bootstrap the mean of paired per-task loss differences with fixed auditor and
train-fitted baseline predictions. For written-run AUROC, bootstrap task-level AUROCs without
fitting their direction. Intervals are clipped to [-1,1] and [0,1] respectively. They describe the
represented task distribution, not production prevalence or training-process uncertainty.
Bootstrap coverage is approximate, especially with few tasks; a minimum of eight is a configurable
coverage guard, not a guarantee of nominal coverage. Report degenerate distributions explicitly.
The two retail tasks at the current pin remain insufficient for this population-level comparison.
The design (a write-presence baseline plus written-run checks) is chosen; the numeric defaults
still await sign-off. Passing this comparison still does not replace human audits or release
approval.

Report model-to-label associations on original splits. On the paired mix the same probe is only a
balance invariant expected to read 0.5; it cannot diagnose residual writing style. Source model names
and simulator names stay private. Core training resets, including explicit record IDs, enforce the
same configured sampled-record whitelist as the toolkit.

The model-label diagnostic uses each model pair's observed failure frequency as an in-sample
lookup. It measures label association on the original distribution rather than held-out style
recognition. For training probes and baseline fitting, repeated sampled records count once;
the sampled model-balance invariant includes every draw. The bootstrap uses the
[percentile method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
with task statistics sorted before seeded resampling so input order does not affect the interval.

A shell's nonzero exit status alone does not establish a tool error or task failure. Preserve it as
`observation.results[].extra.command_status = {exit_code: int, nonzero_exit: bool}`. Reserve `extra.error`
for an explicitly reported tool error. The tool-call shortcut counts agent and user-side calls,
including calls in embedded subagent trajectories.

## 9. Implementation

Follow the repository layout in CONTRIBUTING:

```
08-traceaudit/
├── README.md, project.yaml, SPEC.md
├── adapters/          one per source: download or generate, convert to ATIF, validate, label
├── data/              pinned small sample for tests; full data built by adapters
├── envs/trace_audit/
│   ├── core/          validation, rendering, reward, evaluation and optional token counting
│   └── verifiers/     native adapter and runnable rollout.py
├── tests/
└── results/
```

- TRL exposes every public method of the environment class except `reset` as a tool. Labels and
  hidden record fields live in private attributes only.
- CI runs the tests and `rollout.py` without network access.
- Run `python3 tools/build_index.py --check` from the repo root and `uv run python rollout.py`
  from `envs/trace_audit/verifiers/` before opening the PR.

**Acceptance tests:** a reward is emitted once per episode; resets are isolated; no public method
on the model-facing toolkit returns a label or source identity; malformed actions score 0; unlabelled records never reach the
reward; every stored trajectory passes the ATIF validator.

## 10. Later versions

- **v2:** tools for long traces (`read_steps`, `search_trace`, with bounded reads, explicit truncation
  and continuation offsets); critical step joins the reward once an automatic step-label source
  exists that does not depend on position (for example injected faults confirmed by replay).
- **v3:** category joins the reward; canary tests supply `unsafe_or_unauthorized_action`; SmolDataEnvs,
  AppWorld and production traces as further sources.
- **Training,** only after the v1 gates hold.

## 11. Open questions

1. Which models generate the τ²-bench runs, and what is the budget?
2. A reviewer is named; which rubric do they apply to confirm `trace_evident` and `expert` labels?
3. The rendering budget for v1 (default 32k tokens).
4. Approval of §8's v2.1 numeric gate policy; no approval is recorded yet.

## References

- Hamel Husain, [Evaluating decision models](https://hamel.dev/blog/posts/decision-model-evals/), and
  Parlance Labs, [auto-evals](https://parlance-labs.com/blog/posts/auto-evals/): the trace-level failure
  task, binary labels, threshold tuning and calibration this spec follows.
- [False success in LLM agents (arXiv 2606.09863)](https://arxiv.org/abs/2606.09863): false-success rates
  on τ²-bench and AppWorld, judge and TF-IDF results.
- [ATIF documentation](https://docs.harborframework.com/agents/atif) and
  [validator](https://github.com/harbor-framework/harbor/blob/main/src/harbor/utils/trajectory_validator.py).
- Taxonomies: [AgentRx](https://microsoft.com/en-us/research/blog/systematic-debugging-for-ai-agents-introducing-the-agentrx-framework),
  [MAST](https://huggingface.co/datasets/mcemri/MAST-Data/blob/main/README.md),
  [TRAIL](https://arxiv.org/abs/2505.08638).
