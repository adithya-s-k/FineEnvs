# 04 · RetroEnv

RetroEnv is a small, runnable retrosynthesis RL environment built from real
PaRoutes v2 route trees and its n1 stock. It already reaches the complete loop:

```text
raw licensed routes → normalized references → strict split → hidden tasks
→ bounded multi-turn tools → renderable graphs → deterministic reward → eval
```

This is an initial evaluation set, not yet a large scientific benchmark. RDKit and hidden
reaction records can establish structural consistency and dataset support; they
cannot establish that a synthesis will work experimentally.

## Current artifact

`benchmark/retroeval-v1/` contains 100 verified alternate-route tasks mined
from the downloaded `all_loaded_routes.json.gz` archive. `sample/` remains the
12-task development fixture.

| Split | Tasks | References visible to policy? |
|---|---:|---|
| train | 40 | No |
| dev | 20 | No |
| eval | 20 | No |
| stress | 20 | No |

Each task has 2–5 distinct first-cut references, at most three reaction nodes,
and terminal leaves found by exact lookup in the frozen 13,432-molecule n1
stock. The split audit reports zero crossings for canonical targets, hidden
route products, their Murcko scaffolds, route IDs, reaction IDs, patent groups,
and route-product near-duplicates at Morgan Tanimoto ≥0.90. Private references live only under
`benchmark/retroeval-v1/tasks-private`; reference-free HF-ready rows are under
`benchmark/retroeval-v1/tasks-public`.

Rebuild it deterministically from the downloaded raw artifact:

```bash
uv run python dataset/build_training_sample.py \
  --sample-size 100 --ratios 0.4 0.2 0.2 0.2 \
  --output-dir benchmark/retroeval-v1

uv run python dataset/audit_benchmark.py \
  --benchmark-dir benchmark/retroeval-v1 \
  --expected-eval-tasks 20 \
  --output benchmark/retroeval-v1/audit.json
```

The build manifest pins the archive and stock SHA-256 values, source URL,
license, selection counts, verifier replay requirement, split strategy, and
leakage audit. Of 457,166 raw trees, 125 targets had at least two stock-closed
routes with distinct first cuts under the three-step cap.

## Harness

`reset(split=..., index=...)` is deterministic and returns only the target,
budgets, stock ID, prompt, and tool names. It never returns a reference route,
reference count, patent ID, or answer-bearing metadata. Stock is not embedded
in the prompt.

| Tool | Deterministic rollout behavior |
|---|---|
| `inspect_molecule` | RDKit formula, scaffold, rings, charge, and stereo |
| `pubchem_lookup` | Canonicalize SMILES locally; name/CAS requires a frozen cache |
| `stock_retrieve` | Only stock access; exact/InChIKey/class/SMARTS/similarity, cap 20 |
| `reaction_precedent_search` | Capped analogues from the **train split only** |
| `validate_disconnection` | Check an agent-supplied cut without returning a route |
| `reaction_class_lookup` | Class attached to a supported supplied cut, or `unclassified` |
| `reaction_conditions_search` | Frozen conditions from evidence/analogues |
| `search_literature` | Frozen patent/citation metadata; never live web during rollout |
| `emit_routes` | The only terminal action; parse, verify, and score 1–5 trees |

Live web and supplier-price search are intentionally absent from the current
rollout boundary. Network changes would make episodes non-replayable. They can
be used during dataset hydration and snapshotted before a future release.
Name/CAS support follows that rule: run
`python dataset/hydrate_pubchem.py queries.txt cache/pubchem.json`, then set
`RETROENV_PUBCHEM_CACHE=cache/pubchem.json` when serving.

The final submission is always renderable graph JSON:

```json
{
  "schema_version": "retro-route-graph-v1",
  "routes": [
    {
      "type": "mol",
      "smiles": "CCOC(C)=O",
      "in_stock": false,
      "children": [
        {
          "type": "reaction",
          "is_reaction": true,
          "metadata": {
            "explanation": "Dataset-supported ester disconnection.",
            "reaction_class": "esterification",
            "confidence": 0.9,
            "literature": [],
            "precursor_roles": {"CCO": "alcohol", "CC(=O)O": "acid"}
          },
          "children": [
            {"type": "mol", "smiles": "CCO", "in_stock": true, "children": []},
            {"type": "mol", "smiles": "CC(=O)O", "in_stock": true, "children": []}
          ]
        }
      ]
    }
  ]
}
```

A leaf claim is rewarded only when it agrees with exact stock membership. The
route score shown per tree is its weakest verified step, while the episode
reward covers the whole submitted route set.

## Reward

The terminal verifier is tolerant enough to train weak policies: malformed or
partial outputs do not collapse all groups to the same value. Passing status is
still strict—route-count compliance and at least `min_routes` fully supported,
stock-closed routes with distinct first cuts are required.

| Component | Weight |
|---|---:|
| JSON parse validity | 0.05 |
| Valid RDKit molecules | 0.10 |
| Alternating, connected target-rooted graph | 0.10 |
| Supported/atom-conserving steps | 0.20 |
| Truthful, complete stock leaves | 0.10 |
| Best reference-route similarity | 0.10 |
| Exact match to any reference | 0.10 |
| Verified distinct first cuts | 0.10 |
| Required route-set cardinality | 0.15 |

The legacy flat-route verifier remains hard-gated for compatibility. New
training uses `emit_routes` and the dense graph score. Unsupported chemistry,
valid-looking SMILES, or fluent explanations cannot earn the high-value
correctness components.

Sanity baselines are generated from private references only to test plumbing:

```bash
uv run python eval/run_baselines.py
```

Expected result: oracle ceiling `1.000/pass`, one-route ablation `0.788/fail`,
and empty graph floor `0.050/fail`. These are not model baselines.

## Run and evaluate

```bash
uv sync --extra dev --extra eval
uv run pytest

RETROENV_TASKS_DIR=sample/tasks-private \
RETROENV_STOCKS_DIR=sample/stocks \
uv run retroenv-server
```

For any OpenAI-compatible endpoint with native tool calling:

```bash
uv run python eval/run_model.py \
  --endpoint http://127.0.0.1:8001/v1 \
  --model Qwen/Qwen3.5-4B \
  --split dev --attempts 4 \
  --output sample/model-runs/qwen-dev.jsonl
```

The runner records raw transcripts separately, stores submissions, writes a
hash-pinned run manifest, checkpoints each paid attempt, and can resume after a
provider failure. It forces the final turn to expose only `emit_routes`, then
recomputes Pass@1/Pass@k and every component offline against private truth.
Reports include Wilson 95% intervals for pass rates. To score an existing
prediction file:

```bash
uv run retroenv-eval \
  --tasks-dir sample/tasks-private \
  --stocks-dir sample/stocks \
  --predictions predictions.jsonl \
  --split eval --k 1 4 8
```

The current 20-task model board is in
[`benchmark/retroeval-v1/model-runs/board-v1/RESULTS.md`](benchmark/retroeval-v1/model-runs/board-v1/RESULTS.md).

The static benchmark explainer and trajectory explorer is under `web/`. It
contains all 160 completed episodes, model/task selectors, compact tool traces,
and interactive submitted-route DAGs without exposing private references:

```bash
uv run python web/build_data.py
uv run python -m http.server 8080 --directory web
```

Open <http://localhost:8080> or open `web/index.html` directly.

For the first GRPO experiment, overfit the six `sample/` training tasks before scaling:
use deterministic `(split, index)` reset, 4–8 rollouts per group, and monitor
within-task reward standard deviation—not only mean reward. The environment
server used for training must be the same image and private task bundle used for
evaluation. A trainer should persist raw tool episodes so graph/reward failures
can be replayed.

## Full data pipeline

The workspace already has about 6.5 GB of raw artifacts covering ORD, Lowe
USPTO, CRD, USPTO-LLM, CREED/CREED-CCV, FREA/RxnVerif, PaRoutes, and SynRXN.
The registry distinguishes approved, non-commercial, research-only, and pending
sources; “download everything” never means scraping proprietary or
license-unknown data.

```bash
python dataset/download_raw.py --include-research-only
python dataset/inventory_raw.py

retroenv-prepare-source paroutes_v2 \
  data/raw/paroutes-v2-benchmark build/paroutes-n1.routes.jsonl --release n1

retroenv-build-tasks \
  --routes build/paroutes-n1.routes.jsonl \
  --stock-file data/raw/paroutes-v2-benchmark/stock_n1.txt \
  --stock-id paroutes-v2-n1 \
  --output-dir build/paroutes-n1-tasks
```

The larger dry run produced 7,123 provenance-backed 1–3 step tasks with zero
audited crossings. See `dataset/README.md` for normalization, deduplication,
evidence labels, licensing, and why flat reaction rows are never globally
joined into invented multi-step routes.

## Design basis

The implementation follows the data-first lessons in the
[FineEnvs GeoGuesser environment article](https://huggingface.co/spaces/FineEnvs/geoguesser-article):
freeze a small eval early, define contamination at the correct grouping unit,
make reset indexable, keep truth inside the environment, provide continuous
reward, simulate full rollouts before GPU training, overfit a few tasks first,
use the same hosted environment for train/eval, and preserve raw episodes.

See `RESEARCH.md` for source and verifier evidence, and `dataset/README.md` for
the corpus contract.
