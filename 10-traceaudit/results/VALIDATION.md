# Local validation · 22 October 2026

Executed on Windows with the locked Python 3.12.9 Verifiers environment. These are implementation
checks, not production model results. The GitHub workflow is added; no remote CI run is claimed.

| Check | Observed result |
|---|---|
| Full test suite, external network connections blocked | **155 passed**, including realistic task-count bootstrap comparisons and cross-fitted task-dependent rules |
| Actual Verifiers SingleTurnEnv loop with local fake completion | One assistant turn, Brier reward 0.75 at p=0.5; repeated scoring returns 0 |
| Offline baseline rollout | Six baselines, dev-only thresholds, metrics JSON and calibration SVG written |
| Official pinned Harbor validator: synthetic sample | **37 passed** |
| Official pinned Harbor validator: complete normalized ABCD archive | **10,042 passed**, all outcome labels null |
| Official pinned Harbor validator: bounded SWE sample | **50 passed**, 27 published failures and 23 published successes; human review pending |
| SWE length check with cl100k_base | **50/50 excluded** at 32k; 34,580 / 57,727 / 91,932 min/median/max tokens |
| Updated SWE split and command status | **46 train / 4 dev / 0 test**, 236 nonzero shell statuses, 0 explicit tool errors, 4 last-shell-nonzero positives |
| Updated ABCD split | **8,496 train / 1,546 dev / 0 test**, original archive splits preserved |
| Legacy ABCD apply-reviews | Confirmed test families migrate after hash verification; batch and CLI regressions pass |
| Degenerate outcome comparison | Always-failed predictions fail specificity/AUROC; required single-outcome cohorts are inconclusive; failure-only diagnostics do not block retail comparison |
| AgentRx task matching and shortcut audit | Explicit shared canonical tasks only, equal outcomes per task; task-ID oracle AUROC 0.5; qualifying task-only scores make comparison inconclusive |
| Public τ² task-file eligibility | DB-only: airline **44/50**, telecom **0/2,285**, retail **2/114**; required-write screen: **21 / 0 / 2** |
| Reward shortcut protections | Equal outcomes per source/domain/task; legacy/no-op DB labels excluded; SWE audit guard and trusted trainer reward hook tested |
| Follow-up training distribution checks | Same agent/simulator identities per pair; no reuse by default; exhaustive finite-capacity oracle; core explicit resets enforce the sampled whitelist |
| Evidence audit revision | Pooled training probes with task-cluster intervals; high write predictivity is a baseline rather than a rejection; proposed held-out Brier/written-run comparison tested |
| Statistical follow-up | Fixed-seed task-cluster bootstrap; 8/20-family strong-signal comparisons pass, 2 remain inconclusive; opposite task directions detected out of fold without evaluation-label flipping |
| Review preparation | 50 pending ABCD reviews and 50 pending SWE reviews; none confirmed |
| Root generated index and whitespace checks | Passed |

The SWE sample is the first 50 streaming rows at revision
`35455389ab51bf5e2306bfd436ef72d0f98bf882`, not a representative draw. Native export SHA-256:
`f7277d97db687721d9364d15bfcfcace13b1d6ad5d29873e8546dadd520f1794`.
No model API was called. Only public data/dependencies/tokenizer setup used the network.

Reproduce the offline checks from `envs/trace_audit/verifiers/` after the documented dependency and
Harbor setup:

```bash
uv run --offline --frozen --extra native --extra dev python ../../../tools/offline.py -m pytest ../../../tests -q --basetemp ../../../.cache/pytest
uv run --offline --frozen --extra native --extra dev python ../../../tools/offline.py rollout.py
uv run --offline --frozen --extra native --extra dev python ../../../tools/offline.py ../../../tools/validate_harbor.py
```

Full local imports and review packets remain under ignored `data/generated/`. The required 50-label
human audit per scored source and actual production baseline comparison are not complete.
See [Step 0](STEP0.md) for access, licence, collection and coverage limits.
