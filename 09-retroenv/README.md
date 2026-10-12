# 09 · RetroEnv

**The job: given a molecule, work backwards to molecules you can buy.**

That is retrosynthesis. A chemist looks at a target compound and asks "what two simpler things
could I have combined to make this?", then asks the same of those, and keeps going until
everything on the shopping list is something a supplier sells. RetroEnv turns that into an RL
environment. The agent gets a target molecule and a budget, uses tools to inspect molecules and
look up reactions, and finally submits one or more synthesis trees.

A deterministic verifier grades the answer. It does **not** compare against a hidden answer key.
It checks each proposed reaction against a frozen library of real reactions and reaction
templates, and checks that every leaf of the tree is really in the stock list. So any route that
holds up chemically counts, even one no chemist has published. Matching a known patent route
only adds a small bonus.

One honest caveat: "the library supports this step" means a real reaction like it is on record,
not that the reaction will work in a lab. No yield, no selectivity, no safety claim.

## Quick start

```bash
uv sync --extra dev --extra eval
uv run pytest                                      # runs on tests/fixtures/mini-release

# Run the server on the curated dataset
uv run hf download LiteFold/RetroEnv-Curated --repo-type dataset --local-dir data/release/RetroEnv-Curated
RETROENV_BENCHMARK_DIR=data/release/RetroEnv-Curated uv run uvicorn retroenv_openenv.server:app --port 8000

# Or the hosted copy, and a local browser for the data
uv run bash envs/retro_route/openenv/start.sh      # http://localhost:8000/web/
uv run python explorer/server.py                   # http://127.0.0.1:8050
```

| Hosted | Where |
|---|---|
| The environment server and playground | [FineEnvs/retroenv](https://huggingface.co/spaces/FineEnvs/retroenv) |
| The pinned serving copy and its prebuilt index | [FineEnvs/retroenv-bucket](https://huggingface.co/buckets/FineEnvs/retroenv-bucket) |

## The data

Use **[LiteFold/RetroEnv-Curated](https://huggingface.co/datasets/LiteFold/RetroEnv-Curated)**.
Four splits, nothing to choose between. It is self-contained: tasks, the stock list and the
reaction library in one 68 MB download, so pointing `RETROENV_BENCHMARK_DIR` at it is all the
setup there is.

| Split | Tasks | easy / medium / hard | What it is for |
|---|---:|---|---|
| `train` | 13,200 | 6,812 / 5,153 / 1,235 | RL rollouts and SFT |
| `validation` | 150 | 50 / 50 / 50 | checking a checkpoint mid-training |
| `test` | 150 | 50 / 50 / 50 | the board you iterate against |
| `heldout` | 300 | 100 / 100 / 100 | the sealed final test; run it once |

The four splits never share a task. Held-out targets share no patent, molecule, reaction,
scaffold group or close structural near-duplicate with `train` or with each other, and library
reactions that touch a held-out task are hidden from the agent's tools. Difficulty tiers come
from the release's own labels: route length, how unusual the chemistry is, whether the route is
convergent, and how rare its templates are.

`train` carries 1,200 constraint variants on top of its 12,000 standard tasks, because the
constraint rules are only learnable from tasks that impose them. The split must keep the name
`train`: the environment hides a task's own reactions from the tools only when
`task.split == "train"`, so renaming it would let a rollout look up its own answer.

The full source release, with all 78,505 tasks, is
[LiteFold/RetroEnv](https://huggingface.co/datasets/LiteFold/RetroEnv). The curated splits are
cut from it by [`dataset/build_curated.py`](dataset/build_curated.py), which takes about six
seconds. To rebuild everything from scratch instead:

```bash
uv run python dataset/download_raw.py --source paroutes-v2-benchmark   # 346 MB
uv run python -m dataset.build_release        # the full release, about an hour on 32 cores
uv run python -m dataset.audit_release        # leakage, solvability, reward probes
uv run python -m dataset.build_task_sets      # freeze the curated selections
uv run python -m dataset.build_curated --publish LiteFold/RetroEnv-Curated
```

## Input: what a task looks like

One JSON row. The whole job is in `target_smiles`.

```json
{"task_id": "retro_003174b322aa68b3", "split": "test", "variant": "standard",
 "target_smiles": "CNCc1cn(S(=O)(=O)c2ccc(C)cc2)c(-c2ccccc2)n1",
 "max_depth": 4, "min_routes": 1, "max_routes": 5,
 "stock_id": "paroutes-archive-leaves",
 "constraints": {"forbidden_classes": [], "excluded_stock": []}}
```

In plain terms: *here is a molecule. Work back to things you can buy. Your longest chain may be
4 reactions. Give me between 1 and 5 different routes.*

The stock list is deliberately **not** in the prompt. The only way to find out whether a
molecule is buyable is to call `stock_retrieve`. Some tasks add a constraint: a reaction class
you may not use, or specific building blocks removed from the shelf.

## Output: what the agent sends back

One `emit_routes` call. Each route is a tree that alternates molecule → reaction → molecule.

```json
{"routes": [
  {"type": "mol", "smiles": "<the target>", "in_stock": false,
   "children": [
     {"type": "reaction", "is_reaction": true,
      "metadata": {"reaction_class": "reductive amination",
                   "explanation": "...", "confidence": 0.75,
                   "literature": [], "precursor_roles": {"CN": "nucleophile"}},
      "children": [
        {"type": "mol", "smiles": "CN", "in_stock": true, "children": []},
        {"type": "mol", "smiles": "<intermediate>", "in_stock": false,
         "children": [ ... keep going ... ]}
      ]}]}]}
```

Read it top-down: this molecule is made by this reaction, from these precursors. A molecule with
`children: []` is where the agent stopped, and `in_stock` is its claim that you can buy it.
Recurse until every leaf is in stock.

Three things decide the grade: the root must be the target, every reaction must be one the
library accepts, and every leaf must really be in stock. Claiming `in_stock: true` for something
that is not is the one mistake that caps the score hard.

## The tools

| Tool | Answers from |
|---|---|
| `inspect_molecule`, `pubchem_lookup` | RDKit on the SMILES you pass |
| `stock_retrieve` | The task's stock: exact, InChIKey, class, SMARTS or similarity, 20 results max |
| `reaction_precedent_search`, `reaction_conditions_search`, `search_literature` | Train-visible corpus reactions |
| `validate_disconnection` | Train-visible reactions and frequent templates |
| `reaction_class_lookup` | The verifier's own reaction classifier |
| `emit_routes` | The verifier. Terminal: the episode ends here |

No tool can read a task's hidden known routes, and a training task's own reactions are hidden
from search and validation so a rollout cannot look up its own answer.
`RETROENV_TOOLSET=unaided` drops `validate_disconnection` as an ablation.

## How scoring works

A single route is **valid** when all of this holds: it starts at the target, it is a well-formed
tree inside the depth budget, every step is supported by the library, every leaf is in stock, no
leaf lies about being in stock, and the constraints hold. A task is **solved** when enough valid
routes with different first disconnections were submitted.

Alongside that pass/fail, the environment returns a reward between 0 and 1. The reward currently
shipped is a weighted sum of nine parts:

| Part | Weight | What it measures |
|---|---:|---|
| Parse | 0.02 | the submission is valid JSON |
| Structure | 0.08 | well-formed tree, valid molecules, reaction metadata present |
| Supported steps | 0.35 | the share of reactions the library accepts |
| Leaves in stock | 0.15 | leaves are buyable and honestly claimed |
| Constraints | 0.10 | no forbidden class, no excluded building block, within depth |
| Efficiency | 0.10 | shortest known route depth ÷ your depth |
| Diversity | 0.05 | distinct valid first disconnections |
| Similarity to a known route | 0.10 | overlap with a real patent route |
| Exact known-route match | 0.05 | you found the patent route |

Per-route parts are averaged over the routes submitted, so duplicates and junk dilute the score.
A false in-stock claim caps the reward at 0.40; a submission that never starts at the target
caps at 0.10.

**This reward is measurably gameable, and a replacement has been designed and tested.** Short
version: it pays 0.35 for "the library accepted my reaction" without ever requiring the route to
reach something buyable, plus 0.20 for well-formed JSON. So taking the first disconnection of a
known route and stopping there scores **0.603**, against about 0.88 for actually solving the
task. Across 889 real submissions, 26% of failures scored higher than the worst pass. The
proposed replacement scores that same stop-early submission at 0.055 and puts every failure
below every pass. It lives in [`research/reward/`](research/reward) and is **not wired into the
verifier yet**. [`RESEARCH.md`](RESEARCH.md) has the full study, the evidence and the numbers.

Training applies its own mitigation in the meantime: `sync_grpo.py --partial-credit 0.5` halves
a failed submission's reward, which closes about half the gap.

## Running an evaluation

```bash
uv run python eval/run_eval.py --provider openrouter --model openai/gpt-4.1 \
  --split test --output runs/my-run
```

Every tool call and the reward come from the server, the same one training uses. Each episode is
written to `runs/my-run/episodes/` with its full transcript and submission, and `summary.json`
holds pass@1, mean reward and cost. Reruns resume.

Because the reward is terminal and depends only on `(task, submission)`, stored episodes can be
re-scored under a different reward offline, with no model calls. That is how the reward study
was done.

## Layout

| Path | What it is |
|---|---|
| [`envs/retro_route/core`](envs/retro_route/core) | The `retroenv` package: chemistry, reaction library, classifier, leakage keys, verifier, tools, session |
| [`envs/retro_route/openenv`](envs/retro_route/openenv) | OpenEnv server, client, agent loops, hand-play UI, deployment |
| [`dataset/`](dataset) | Source intake, `build_release.py`, `build_curated.py`, `audit_release.py` |
| [`eval/`](eval) | `run_eval.py` through the server, boards and summaries |
| [`train/`](train) | SFT and GRPO recipes, HF Jobs launchers, the difficulty sweep |
| [`research/reward/`](research/reward) | The reward study: evidence cache, candidate rewards, hack probes, ablation |
| [`explorer/`](explorer) | Local browser for a release, with route graphs and live play |
| [`tests/`](tests) | Unit and integration tests, and the committed mini-release fixture |

[`RESEARCH.md`](RESEARCH.md) has the data-source evidence, the verifier's boundary, and the
reward study.
