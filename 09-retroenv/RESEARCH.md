# Research and design notes

Two halves. First, where the data comes from and what the verifier is allowed to claim.
Second, the reward study: what was wrong with the reward, how it was measured, and what
replaces it.

---

# Part 1 · The reward study

## What we set out to do

The reward was too easy to game. A review of real model runs found two problems:

1. **Failing paid almost as well as succeeding.** A failed submission earned roughly 0.55–0.65
   against about 0.88 for a pass. Stopping after one accepted disconnection was a cheap floor
   that an RL policy would find immediately.
2. **The reaction templates accepted bad chemistry.** They only checked the atoms at the bond
   being made, so they waved through reactions at the wrong position on a ring or on the wrong
   functional group.

The goal was a reward that is (a) grounded in the real pass rate, (b) hard to game, and (c)
still dense enough to learn from. Those three pull against each other, so the work was
empirical rather than a guess.

## What we did

Collected **889 real submissions from 16 models** across 79 held-out tasks, by running an
OpenAI capability ladder (gpt-4.1-nano through gpt-5) and combining it with the existing board
runs. Then built **587 synthetic probes**: degenerate submissions that cost nothing to produce
and should never score well, plus "staircases" that truncate a known route at every depth.

The key trick: the reward is terminal and depends only on `(task, submission)`. Both are stored
in every episode, so **any candidate reward can be re-scored offline with no model calls**. That
made it cheap to compare 26 candidates on identical data. The harness lives in
[`research/reward/`](research/reward):

| File | Does |
|---|---|
| `evidence.py` | Distils each submission once into the raw facts (slow: RDKit, template scans) |
| `variants.py` | Scores 26 candidate rewards from that cache, in microseconds each |
| `probes.py` | Generates the hack battery and the truncation staircases |
| `ablate.py` | `build`, `report`, and `verify` |

`ablate verify` replays all 889 stored grades and asserts they match the server exactly. It
catches harness bugs; it found one real one, where templates were being run on the stereo-free
molecule instead of the molecule as submitted.

## The old reward, in plain language

Nine parts added up with weights. Read as a sentence, it paid for:

> Valid JSON (0.02), a well-formed tree (0.08), the share of your reactions the library accepts
> (0.35), leaves that are buyable and honestly claimed (0.15), breaking no constraint (0.10), a
> short route (0.10), distinct first cuts (0.05), looking like a known route (0.10), being a
> known route (0.05).

**What it actually captured was effort, not synthesis.** Three things combine badly:

- *Parse, structure and constraints are nearly free.* Any syntactically valid tree gets 0.20,
  because you cannot violate a constraint on a route you never attempted.
- *Supported steps are scored per step, with no requirement that the route reach anything
  buyable.* One accepted disconnection is 1.0 on a 0.35 component.
- *Everything that needs a complete route* — efficiency, diversity, route similarity — is 0
  until you actually solve the task.

So the floor for "propose one cut the library likes, then stop" is about 0.55, and the ceiling
for solving is about 0.88. The gap is 0.33, and the cheap behaviour is far easier to find.

### The evidence

Measured over the 889 submissions:

| | Old reward |
|---|---:|
| Failures scoring above the worst pass | **26.0%** |
| Best failure vs worst pass | 0.684 vs 0.468 |
| Take the first cut of a known route, then stop | **0.603** |
| Gain from that first cut to the full route | 0.397 |
| Mean \|reward − pass rate\| per model | 0.188 |
| Worst model's \|reward − pass rate\| | 0.374 |

The clearest single result is a ranking inversion on an identical 30-task set:

| Model | pass@1 | old reward |
|---|---:|---:|
| gpt-4o | **0.100** | 0.326 |
| gpt-4.1 | 0.025 | 0.349 |
| gpt-5-mini | 0.025 | 0.399 |

The reward ranks the two models that almost never solve anything **above** the one that solves
four times as often. A reward column that does this cannot be read as capability, and a policy
trained on it is being pushed toward the wrong behaviour.

## The new reward, in plain language

The design follows the pattern that held up across the literature on verifiable rewards: make
the **outcome** dominate, and let partial credit only order the failures among themselves.

> **If you solved the task**, you start at 0.70 and earn the rest for being efficient, giving
> genuinely different routes, and agreeing with a known route. Padding with duplicates or junk
> scales that down.
>
> **If you did not**, you get at most 0.20, and your position inside that band is set by how
> much of your tree is actually grounded in buyable material.

The two bands never meet. That one change is what makes mean reward readable as a pass rate.

What "grounded" means precisely: walk the tree from the target down. A leaf scores 1 if it is
really in stock and 0 otherwise. A molecule whose reaction the library rejects scores 0, and so
does everything under it. Otherwise it is the heavy-atom weighted average of its precursors.
**An accepted cut that never reaches anything buyable therefore scores exactly 0.** That is what
removes the cheap floor.

A small extra term handles a real gap: on a purely linear route you reach no stock at all until
the very last step, so progress alone gives nothing to climb. The frontier term gives bounded
credit for the hardest unsolved molecule getting closer to something buyable, measured as a gain
over the target itself so that doing nothing scores zero. It is capped at 0.04 of total reward
by construction.

| | Old | `--partial-credit 0.5` | New |
|---|---:|---:|---:|
| First cut of a known route, then stop | 0.603 | 0.301 | **0.055** |
| One unsupported cut into two stock molecules | 0.350 | 0.175 | **0.030** |
| A leaf that lies about being in stock | 0.400 | 0.253 | **0.051** |
| A real solve, submitted three times | 0.480 | 0.480 | 0.667 |
| A clean single-route solve | 1.000 | 1.000 | 1.000 |
| Failures scoring above the worst pass | 26.0% | 0.0% | **0.0%** |
| Worst pass minus best failure | −0.216 | +0.126 | **+0.477** |
| Gain from first cut to full route | 0.397 | 0.699 | **0.945** |
| Mean \|reward − pass rate\| per model | 0.188 | 0.090 | **0.029** |
| Rank correlation with pass@1 | 0.961 | 0.987 | **0.993** |

`--partial-credit 0.5` is the mitigation already shipped in `train/sync_grpo.py`: it halves a
failed submission's reward. It is a genuine improvement and costs nothing, but it leaves
stopping early at 0.301, still two thirds of the worst real pass.

## The template fix

The second problem was separate: the library accepted a reaction as soon as a retro-template
seen 5+ times mapped the product onto a subset of the proposed precursors. That only constrains
the atoms at the bond being made.

The fix is a **round-trip**: run the matched template *forwards* over the precursors it proposed
and require the target back as the **only** product. A regioselective reaction gives exactly
one; a wrong-site reaction gives several, and is rejected. The round-trip regenerates the target
99.3% of the time, so the mechanism is sound rather than a source of false rejections. Examples
it catches: chlorosulfonation of benzaldehyde (the template fires at 3 ring positions), an SNAr
diaryl ether on a dichloroarene (2 chlorides).

**This contradicts one of the original suggestions.** The review proposed raising the minimum
template frequency above 5. The data says don't. At a minimum count of 25 the rule drops 8.1% of
template-only steps, and reading them shows they are textbook chemistry being punished for being
uncommon: Riley oxidation (template seen 9×), Horner–Wadsworth–Emmons olefination (19×),
Newman–Kwart rearrangement (16×), Buchwald–Hartwig amination (17×). The sole-product rule drops
a comparable 7.3% and drops the right ones. **Keep the frequency floor at 5; add the round-trip.**

Importantly, oracle solvability stays at **1.000 under every strictness setting**. Known patent
routes are recorded reactions and bypass templates entirely, so tightening costs nothing on
ground truth. It removes 4–6% of *model* passes, which were resting on an ambiguous step.

## The eval runs

| Run | What |
|---|---|
| `runs/openrouter-30` | 8 models on 30 easy `test_id` tasks, with and without submission repair |
| `runs/rewardlab` | The OpenAI ladder: 8 models, 50 random `test_id` tasks plus the same easy 30 |
| `research/reward/report.json` | 26 reward variants scored over all of the above plus 587 probes |

The 30-task board, re-scored under both rewards. `gap` is reward minus pass@1:

| Model | pass@1 old | pass@1 new | reward old | reward new | gap old | gap new |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 5.5 | 0.833 | 0.800 | 0.850 | 0.761 | +0.017 | −0.039 |
| Claude Opus 5.5 | 0.800 | 0.800 | 0.798 | 0.753 | −0.002 | −0.047 |
| GPT-5.6 Sol | 0.767 | 0.767 | 0.797 | 0.699 | +0.030 | −0.068 |
| GPT-5.6 Terra | 0.567 | 0.500 | 0.704 | 0.480 | +0.137 | −0.020 |
| GPT-5.6 Luna | 0.433 | 0.400 | 0.617 | 0.381 | +0.184 | −0.019 |
| Intern-S2 Preview 35B | 0.267 | 0.267 | 0.427 | 0.259 | +0.160 | −0.008 |
| Qwen3.8 27B (high) | 0.200 | 0.200 | 0.467 | 0.208 | +0.267 | +0.008 |

Every old gap is positive and grows as the model gets weaker. That is the reward paying for
effort. Under the new reward the gaps are small and slightly negative, because a strong model
now has to earn the top of its band instead of being handed it. Pass@1 also moves a little
(0.525 → 0.504 overall) as the round-trip rule removes routes resting on an ambiguous step.

## Status and limits

**The new reward is not wired into `verifier.py`.** It lives in `research/reward/variants.py` as
`v8-FINAL` and is scored offline. Adopting it is a deliberate decision, because it changes both
the reward and `valid` itself.

Known limits, stated plainly:

- The truncation staircase is strictly increasing on 46 of 60 routes. Five are flat ties and
  nine wobble by at most −0.013. The frontier term can move the reward by at most 0.04 by
  construction, so the wobble is inside the noise a group-relative objective normalises away.
- 6.3% of tasks give every model the same reward, against 0% under the old one. These are tasks
  nobody makes progress on, and GRPO gets no advantage signal there. More structure credit
  restores spread but also raises the floor a hack can reach; 0.15 is the measured compromise.
- The frontier term needs the stock fingerprint index at reward time, which the verifier does
  not currently receive. Dropping it costs 14 staircase steps and nothing else.
- Hackability was measured against **fixed trajectories**, not a trained policy. The probes
  cover the exploits visible today. Reward hacking co-evolves with optimisation pressure, so the
  battery should be re-run against high-reward rollouts once something is training on this.

Reproduce with:

```bash
uv run python -m research.reward.ablate build --runs runs/rewardlab runs/openrouter-30
uv run python -m research.reward.ablate verify --runs runs/rewardlab runs/openrouter-30
uv run python -m research.reward.ablate report
```

---

# Part 2 · Sources and the verifier's boundary

Research checked on 2026-09-22. Primary sources are linked so dataset and verifier claims can
be audited.

## Decisions grounded in the sources

- RDKit sanitization checks representability and allowed valences, and reaction SMARTS can be
  executed deterministically. This supports structure and template gates, not experimental
  feasibility. The distinction follows the
  [RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html#chemical-reaction-handling)
  and its [sanitization/valence documentation](https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization).
- Bemis–Murcko scaffolds are computed using RDKit's documented
  [`MurckoScaffold`](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html)
  implementation. Acyclic targets use their canonical product as the grouping key because an
  empty scaffold would collapse every acyclic compound into one split.
- ORD is the safest first public corpus: its official repository stores data as Parquet with
  serialized protobuf reactions and explicitly licenses the data CC-BY-SA-4.0 and code
  Apache-2.0. See the
  [official `ord-data` repository](https://github.com/open-reaction-database/ord-data).
- PaRoutes is useful for route and stock evaluation: its official repository provides two
  10,000-route sets, two stock sets, and top-N route recovery tooling. The software repository
  is Apache-2.0, while the official v2 Zenodo record independently marks the benchmark artifacts
  CC-BY-4.0. They are kept as separate intake entries. See
  [MolecularAI/PaRoutes](https://github.com/MolecularAI/PaRoutes)
  and [PaRoutes v2 on Zenodo](https://zenodo.org/records/7341155).
- OpenEnv documents MCP as the agent boundary and the Gym-like reset/step API as the
  infrastructure boundary. RetroEnv exposes both, following the
  [official MCP lifecycle](https://github.com/huggingface/openenv/blob/main/docs/source/guides/mcp-environment-lifecycle.md).

## Corpus intake policy

The source list in the proposal is a discovery list, not an allowlist. A source enters a release
only after all of these are recorded:

1. exact upstream artifact and immutable version;
2. data license (not merely the code-repository license);
3. observed/generated/simulated evidence kind;
4. extraction and atom-mapping method;
5. redistribution and attribution obligations;
6. overlap audit against existing sources.

ORD, Lowe USPTO, CRD, CREED/CREED-CCV, and FREA now have pinned intake entries; CREED remains
isolated as non-commercial. PaRoutes v2 benchmark artifacts and USPTO-LLM are tracked under
their respective Zenodo CC-BY-4.0 records, while SynRXN remains a per-artifact input.
RXNGraphormer does not publish its stated 13M-reaction pretraining artifact in the official
repository. AbSynth routes require institutional-email access and carry a no-derivatives
license, so they cannot enter a transformable HF release. The pipeline will not invent a license
or erase evidence labels.

## Verifier boundary

The following are deterministic claims:

- SMILES parses and sanitizes under the pinned RDKit build;
- mapped atoms preserve element/isotope identity where mappings exist;
- every product's elemental atom inventory is contained in its proposed precursors even when the
  source record is unmapped;
- a hidden trusted template generates the canonical product from the proposed precursors;
- a proposal matches a hidden observed reaction record;
- the submitted route is a connected acyclic graph ending in the selected stock set.

The following are explicitly not claimed:

- that a reaction has useful yield or selectivity;
- that listed conditions are complete or compatible across the route;
- that a template match proves kinetics, mechanism, scale-up, safety, or laboratory feasibility;
- that a stock identifier guarantees current commercial availability or price.

Results therefore use `dataset_supported` and `template_supported`, never
`experimentally_validated`, unless a future source carries explicit experimental evidence and
the evaluation reports it separately.

## Recommended release order

1. Normalize 10,000 observed, license-audited single-step records.
2. Freeze global stock and scaffold-OOD splits before model experiments.
3. Run the leakage, duplicate, reward, and replay checks in this project.
4. Publish prompt-only development tasks and keep evaluation references server side.
5. Add provenance-backed 2–3 step routes and report them separately from single-step prediction.
6. Add template/search/open-model baselines only after the data split is frozen.
