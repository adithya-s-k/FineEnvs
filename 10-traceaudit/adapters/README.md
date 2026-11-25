# Source adapters

Every import is offline and requires a pinned revision and an explicitly verified data licence.
`tools/data.py import` consumes UTF-8 JSONL, except ABCD's original split dictionary/gzip file.
Imports fail without writing a partial output when a row is malformed. Full data is ignored by Git.
Run both `tools/data.py validate` and the pinned official `tools/validate_harbor.py` after importing.

| Source | Input contract | Outcome |
|---|---|---|
| SWE-rebench | Native `trajectory_id`, `instance_id`, chat `trajectory`, `resolved`, global `tools` | Boolean/integer 0/1 `resolved`; missing is unknown. Submission status and generated tests are ignored. |
| τ²-bench | Joined `task`, `simulation`, `domain`, actual visible `instruction`, `policy`, `tools` | DB-only criteria with a required DB change, plus `db_verification` completed hashes or native boolean `reward_info.db_check.db_match`. Total reward/db_reward are ignored. |
| ABCD | Original `{train: [...], dev: [...], test: [...]}` with `convo_id` and `original` turns | Always unknown. Dialogue actions remain observations, not executable service tools. |
| AgentRx/TRAIL/Who&When | Explicit annotation-to-trace join below | Unknown until a named reviewer confirms a boolean outcome. Test-only. |
| MAST | Explicit trace join with original taxonomy labels | Unknown/analysis-only regardless of model annotations. |

`chat_to_atif` joins tool results by ID, records original message-index mappings, preserves malformed
tool arguments as raw evidence, normalizes explicit error flags and preserves unobserved calls.
It never guesses a tool error from English wording. Text v1 rejects multimodal message blocks;
continued traces and external subagent references must be resolved before admission.

SWE's linked `execute_bash` replies additionally normalize the terminal OpenHands footer
`[Command finished with exit code N]` into `extra.command_status = {exit_code, nonzero_exit}`.
A nonzero status is not enough to set `extra.error`: grep with no match and expected test failures
are ordinary status evidence. Explicit source tool-error flags still normalize to `extra.error`.
Full output remains in `content`. Quoted examples, nonterminal text and other tools do not trigger
the shell parser. Neither status nor a tool error changes the trace's `resolved` outcome.

ABCD imports preserve its archive split in `source.source_split`, but assign task families to
train/dev. Existing unlabelled test records remain loadable for legacy reviews; confirmation checks
their original hashes then migrates the whole family to train/dev and reports the change. ABCD
does not become another held-out test group. A family shared canonically with another source needs
an explicit split decision and cannot be migrated automatically across that boundary.

## τ²-bench

`tools/tau2.py convert INPUT OUTPUT --revision SHA --domain airline --policy POLICY.txt --tools TOOLS.json`
joins the native monolithic `Results.tasks` and `Results.simulations` arrays. It uses the captured
first user message as instruction and the captured policy when present. It supports text half-duplex
assistant and user-side tool calls. User calls live on their user step in
`extra.participant_tool_calls` (ATIF restricts native `tool_calls` to agents); linked observations carry
`extra.participant_call_id` and `requestor: user`. The renderer exposes these calls with their actual
actor and opaque IDs. Identical native IDs from different requestors do not collide. Voice/tick
exports still need an explicit converter. Source messages, action lists and aggregate rewards are
not replayed or compared with an agent's route. Missing DB outcomes remain unknown.

Each record now preserves private `source.agent_model` and `source.user_model`. Joined JSONL accepts
explicit `agent_model`/`user_model` on the row or simulation. Results conversion reads
`info.agent_info.llm` and `info.user_info.llm`, with optional `--agent-model` and `--user-model`
collector arguments that must agree with captured metadata. Missing identities remain null rather
than being guessed. Generic imports also accept those collector flags. SWE preserves `source_model`
as `agent_model` and records `user_model: not_applicable`. Sampling requires both identities and
equal outcomes within each task/model combination; it cannot silently pool unknown or different
models. These identities are masked in public rendering. Add provenance before making review queues:
record fingerprints change when provenance is added, and old review approvals then become stale.

DB equality on a read-only task does not prove that the agent answered correctly or followed policy.
Such tasks now stay unlabelled with `db_only_no_verified_required_change`, for either native outcome,
until a named human reviews the full task and trace. The necessary eligibility screen checks for a
known mutating action in the task criteria, never a write call in the actual run. Independently
captured `db_verification.initial_db_hash` and `expected_db_hash` take precedence: equal hashes
exclude the task, differing valid hashes establish a required change without a reference route.
Unknown tool names cannot establish eligibility. A mutating reference action alone does not prove
that all informational and policy requirements were checked; the human label audit remains required.

New private checkers record `task_requires_db_change` and the evidence basis. Legacy normalized
DB-only labels lacking this eligibility evidence remain readable for review but are excluded by
preparation, runtime, sampling and held-out gates. Re-import native data or apply human reviews;
do not add the marker to an old label without checking its original task.

Premature termination without final-state evidence stays unlabelled, including `max_steps` and
`too_many_errors`. Its `source.label_exclusion_reason` is `unverified_early_stop:<reason>`; preparation
and runtime count it explicitly. Criteria-ineligible runs use `ineligible_outcome_criteria`, and
ordinary runs missing DB evidence use `missing_db_outcome`. A stop reason or zero reward is not a
verified DB outcome. Independent final-state evidence may establish either outcome after an early stop.

Only `DB`, or `DB+COMMUNICATE` with no communication requirement, qualifies. `ACTION`, `NL_ASSERTION`
and `ENV_ASSERTION` criteria do not qualify. No criterion is removed to manufacture eligibility.
The pinned revision has airline 44/50, telecom 0/2285 and retail 2/114 DB-only tasks. Of airline's 44,
23 have no expected write, leaving 21 mutation-screened tasks; retail retains 2 and telecom 0.
The source check reports both counts, and collection plans use `db_change_task_ids`. The unlaunched
plan records this shortage. It does not enforce spending limits for an external collector; choose
models and a budget-enforced runner before any generation.

## Attribution joins

These exports are deliberately explicit because source annotation rows may be separate from raw
traces and use different step numbering. No offsets are guessed. An accepted join looks like:

```json
{"original_id":"source-id","source_split":"tau_retail","task_family_id":"shared-task-family","task":{"instruction":"...","policy":null,"success_criteria":null},"trajectory":{"schema_version":"ATIF-v1.7","agent":{"name":"original-agent","version":"source"},"steps":[{"step_id":1,"source":"user","message":"..."}]},"step_map":{"14":1},"failures":[{"step_number":14,"failure_category":"Invalid Invocation","critical":true}],"verified_failed":true,"human_confirmed":false,"reviewer":null}
```

`step_map` maps original annotation steps to normalized root ATIF IDs. Preserve originals in private
extras. The adapter maps unambiguous taxonomy labels; supply `mapped_category` and `adjudication_reason` for ambiguous cases
after adjudication. Human confirmation must be explicit. The eight categories and critical step are
diagnostics, not reward terms. Native source-specific trace joins remain an access/data-preparation
step for the gated corpora; this adapter does not claim their gated schema has been exercised.

Local `task_family_id` values are namespaced by dataset, with the original kept privately in
`source.original_task_family_id`. If exports from different sources are known to describe the same
underlying task, provide an explicit shared `canonical_task_family_id`, such as `tau2:retail:33`.
That key replaces the local namespace and split validation still rejects cross-split overlap.
The supplied mapping is also preserved in `source.canonical_task_family_id`. AgentRx retail outcome
pairing requires this explicit shared identity and a matching verified tau2 retail success; local
IDs alone never establish a match. Pairing selects equal numbers per outcome within each matched
task, sorted by record ID independently of predictions. Unmatched cases remain diagnostic records.
The actual gated-source task mapping has not been verified and is still required before comparison.

Who&When's dataset card lacks a licence. An import path is not permission to redistribute the data;
resolve the licence before admitting that source. AgentRx and TRAIL gates have not been accepted.
Source card licence metadata is recorded separately from the framework code's licence in
`data/source_checks.json` and [Step 0](../results/STEP0.md).
