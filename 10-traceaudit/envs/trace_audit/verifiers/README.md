# TraceAudit · Verifiers

`create_verifiers_env(data_path, allow_fixtures=False)` builds a `vf.SingleTurnEnv`. Prompts contain
only public observations. Opaque random answer keys join to private verified outcomes in the reward
closure. No label or source ID enters the model messages. One Brier reward has weight 1; framework
monitor metrics have weight 0. There are no model-facing inspection tools in v1.

The training dataset is sampled after rendering admission with equal outcomes per source/domain/task/model pair
and a 50% dataset cap. Missing outcome pairs fail closed; fixtures require the explicit fixture flag.
`training_size=None` selects the largest feasible sample under the source cap and finite paired
capacity. `max_record_reuse=1` disallows repeat records within that sample; increasing it is explicit.
`source_cap` and `sampling_seed` configure the sample. Set `return_report=True` to receive
`(env, sampling_report)`. The report includes observed reuse, unique counts and a `shortcut_audit`
of that exact sample. Missing generating-model provenance excludes records from sampling.
The audit includes pooled training length, tool-count and closing-language probes as well as writes.
High feature predictivity is diagnostic, not an automatic rejection. The owner-selected gate design compares
held-out auditor predictions with a train-fitted write baseline and requires discrimination among
written runs, subject to task-cluster uncertainty and pending numeric signoff. A training report alone never
approves this comparison. Explicit core resets enforce the same configured whitelist as the toolkit.
Training diagnostics also cross-fit each task/model's feature direction across disjoint halves;
singleton pairs are skipped. Intervals use a fixed-seed task-cluster percentile bootstrap, with
no bounds below eight task families by default. This configurable numeric guard remains pending
signoff, and bootstrap coverage is approximate.
Evaluation always uses the original dev/test distribution. Exact balance in each optimizer batch
requires that trainer to draw batches with `trace_audit.core.sampling.balanced_batch`; random dataset slicing
preserves these guarantees only across the full assembled sample. SWE grader labels additionally
require the revision-bound 50-label human audit described in the project README.

```bash
uv sync --frozen --extra native --extra dev
uv run python rollout.py
uv run python -m pytest ../../../tests -q --basetemp ../../../.cache/pytest
```

The offline rollout uses fixtures explicitly and makes no API calls. Native integration tests run
the actual framework loop with a local fake response, then invoke its rubric. They exercise prompt
formatting, one-turn termination and scoring without network access.

```python
from env import create_verifiers_env

env = create_verifiers_env("../../../data/generated/combined.jsonl")
# env.evaluate(client, model) is available to a separately authorized model runner.
```

Verifiers is pinned to **0.1.9**, the legacy API used by this repository. Current upstream v1 has
a different taskset API. The lockfile records the tested dependency resolution. This is a source
checkout port (`package=false`), not an independently installable wheel.

Source entry points add `envs/` with a guarded path insert and import shared code through
`trace_audit.core`. This keeps the local framework folder separate from the installed `verifiers`
package during native imports.

For manual/TRL consumption, `TraceAuditToolkit(records, split=..., allow_fixtures=False)` exposes
`reset()` and `submit_analysis(response)`. Submission returns a terminal acknowledgement only.
The trusted trainer calls the module-level `consume_toolkit_reward(toolkit)` after submission (or
after generation ends without one). It calls the core's `consume_reward()`; missing, repeated
or already-consumed submissions earn zero. Missing submissions also close the episode. This helper
is not a method on the model-facing toolkit and must never be registered as a tool. Do not expose
core internals, datasets or private record objects through tool introspection. `TOOL_FUNCTIONS=[]`;
submit the final answer as text for native single-turn evaluation. No module-global shared episode
state exists, so independent toolkit instances are isolated.

Toolkit construction selects the paired sample without running the statistical shortcut audit.
The native dataset builder and offline evaluator run that audit once for their reports and reuse
the core's cached prompts. Their reports do not modify the core's sampling report.
