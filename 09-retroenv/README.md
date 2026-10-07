# 09 · RetroEnv

Plan a retrosynthesis: given a target molecule, work back to molecules you can buy. The
agent inspects molecules, searches a frozen stock and train-visible reactions, checks
disconnections, and submits molecule/reaction trees. A deterministic verifier judges every
step against a frozen reaction library, not against a hidden answer, so any chemically
supported route that ends in stock counts; known patent routes only add a bonus.

Library support (a recorded reaction, or a retro-template seen at least five times) is
evidence, not proof that a synthesis works in the lab.

## Quick start

```bash
uv sync --extra dev --extra eval
uv run pytest                                            # runs on tests/fixtures/mini-release

# The release, fully open: https://huggingface.co/datasets/LiteFold/RetroEnv
# (every split's tasks and known routes, stock, reaction library, eval runs)
uv run hf download LiteFold/RetroEnv --repo-type dataset --local-dir data/release/RetroEnv-RL

# Or rebuild it from PaRoutes v2 (346 MB download, about an hour on 32 cores)
uv run python dataset/download_raw.py --source paroutes-v2-benchmark
uv run python -m dataset.build_release                   # data/release/RetroEnv-RL (git-ignored)
uv run python -m dataset.audit_release                   # leakage, solvability, reward probes
uv run python -m dataset.publish_release --runs runs     # publish to LiteFold/RetroEnv

RETROENV_BENCHMARK_DIR=data/release/RetroEnv-RL uv run uvicorn retroenv_openenv.server:app --port 8000
uv run python explorer/server.py                         # http://127.0.0.1:8050
```

## The release

Built from PaRoutes v2 (457,160 patent routes, CC-BY-4.0). The stock is every leaf of every
archive route (102,915 molecules). Targets have 10–60 heavy atoms, no metal, are not in
stock, and have a shortest stock-closed route of 2–8 reactions through clean corpus
reactions.

| Split | Targets | Tasks | Shortest route | Use |
|---|---:|---:|---|---|
| train | 70,028 | 75,224 | 2–8 (73% two-step) | RL rollouts, SFT bootstrap |
| dev | 1,000 | 1,079 | 2–7, flattened | checkpoints, reward calibration |
| test_id | 1,000 | 1,083 | 2–8, flattened | unseen targets, training distribution |
| test_hard | 1,000 | 1,119 | 4–8 | novel (nearest other-patent Tanimoto < 0.6) and convergent, rare-template or complex-ring chemistry |

Held-out targets share no patent, route molecule, reaction, scaffold group or Tanimoto ≥
0.90 near-duplicate with train or with each other, including the molecules of variant
witness routes. Library reactions that touch a held-out key never reach the tools.

Variants (each with a witness route proving it solvable): `max_depth` (budget equal to the
shortest known depth while the patent route is longer), `restricted_stock` and
`forbidden_class` (break both the shortest witness and the patent route), and `diversity`
(two or three routes with different first disconnections).

## The environment

| Tool | Answers from |
|---|---|
| `inspect_molecule`, `pubchem_lookup` | RDKit on the given SMILES (names need a frozen cache) |
| `stock_retrieve` | The task's stock (excluded building blocks absent): exact, InChIKey, class, SMARTS or similarity, at most 20 results |
| `reaction_precedent_search`, `reaction_conditions_search`, `search_literature` | Train-visible corpus reactions; a train task's own keys are hidden too |
| `validate_disconnection` | Train-visible reactions and frequent templates (the verifier uses the full library) |
| `reaction_class_lookup` | The verifier's own rule-based reaction classifier |
| `emit_routes` | The verifier; terminal |

No tool reads a task's hidden routes. `RETROENV_TOOLSET=unaided` drops
`validate_disconnection` as an ablation.

## Reward

A route is valid when it starts at the target, is a well-formed tree within the depth
budget, every step is library-supported, every leaf is in the task's stock, no leaf falsely
claims stock, and the constraints hold. A task is solved with enough valid routes with
distinct first disconnections.

| Component | Weight |
|---|---:|
| Parse | 0.02 |
| Structure (target root, tree, valid molecules, metadata) | 0.08 |
| Library-supported steps | 0.35 |
| Leaves in stock, claimed truthfully | 0.15 |
| Constraint compliance | 0.10 |
| Efficiency (shortest known depth / route depth) | 0.10 |
| Distinct valid first disconnections | 0.05 |
| Similarity to a compliant known route | 0.10 |
| Exact known-route match | 0.05 |

Per-route components are averaged over the submitted routes, so duplicates and junk routes
dilute the score. A false in-stock claim caps the reward at 0.40, and a submission that never
starts at the target at 0.10.

## Layout

| Path | What it is |
|---|---|
| [`envs/retro_route/core`](envs/retro_route/core) | The `retroenv` package: chemistry, reaction library, classifier, leakage keys, verifier, tools and session |
| [`envs/retro_route/openenv`](envs/retro_route/openenv) | OpenEnv server, client, agent loops (including the scripted chemist), hand-play UI, Docker/Space deployment |
| [`dataset/`](dataset) | Raw-source intake, `build_release.py` with its `pipeline/` stages, and `audit_release.py` |
| [`eval/`](eval) | `run_eval.py` through the server, and summaries |
| [`train/`](train) | SFT and GRPO recipes |
| [`explorer/`](explorer) | Local browser for a release, with route graphs and live play |
| [`tests/`](tests) | Unit and integration tests, and the committed mini-release fixture |

Earlier benchmarks (v1–v3), their model runs and SFT data are archived outside the repo.
[`RESEARCH.md`](RESEARCH.md) has the source and verifier evidence.
