# 04 · Wordle

The cheapest multi-turn environment in this repo, and a GRPO loop that can tell a failed group from a collapsed one.

Project 00 shipped Wordle six ways and never trained against it. Project 03 trained against GeoGuesser and then wrote down, in `LEARNINGS.md`, that `frac_reward_zero_std` is the number that decides whether a step taught anything. TRL still has no dynamic sampling. This project is that number, treated as a training failure rather than a chart to watch after the run dies.

<!-- BEGIN:matrix -->
| Env | Tools | Backend | `inprocess` |
|---|---|---|---|
| **wordle** | 1 | `none` | ✅ |
<!-- END:matrix -->

## The result

A 32-d pointer over the 2,309-word answer list, GRPO with groups of 8, 200 steps, one seed, greedy eval on a 200-word holdout the policy never trained on. Three to six minutes a run on CPU. The estimator is the same one `grpo_wordle.py` hands to TRL.

| arm | best solve | at step | final solve | dead groups at end |
|---|---:|---:|---:|---:|
| sparse + GRPO | 0.770 | 75 | 0.745 | 0.20 |
| sparse + alive | 0.790 | 50 | 0.700 | 0.05 |
| process + GRPO | 0.800 | 200 | **0.800** | 0.11 |
| process + alive | **0.820** | 175 | 0.760 | **0.03** |

`python train/tiny_grpo.py --steps 200 --seed 0`

Two things happened, and they are not the same thing.

1. **The reward is the gain.** Paying realised information gain instead of win/loss is +0.055 at step 200 against sparse GRPO (0.800 against 0.745), and process-alive's peak is 0.820. Sparse GRPO peaked at step 75 and gave some of it back.
2. **Sparse GRPO then did what GeoGuesser run 1 did.** Dead groups went 0.02 → 0.20. Collapse — eight identical guess sequences — went 0 → 0.08. Alive on the same sparse reward cut the dead fraction to 0.05. It did not save the solve rate by step 200. Skipping a zero-std group is not a substitute for a reward that can see the difference between two failures.

Single seed, n = 200. A 0.02 gap is noise. The dead-group column (0.20 → 0.03) is the one that transfers.

The Qwen recipe is `train/grpo_wordle.py`. It has not been spent on a GPU in this contribution. That is a gap, not a claim.

## What a dead group is

A GRPO group of G rollouts produces no gradient when they share a reward. Two ways that happens, and they need opposite responses.

**Cliff.** Different trajectories, identical rewards. GeoGuesser's subtract-and-floor scored a 3,324 km miss the same as an 18,723 km miss — 77 of 200 episodes at exactly 0.0. The reward is blind. Resample the task, densify the reward, or skip the backward pass.

**Collapse.** The same trajectory, G times. Run 1's policy converged to a one-glance city guess; group std fell to 0.001 and the next 750 steps spent about $70 moving 0.013. Resampling draws the same action. Raise temperature or stop.

`classify_group` tells those apart. TRL's `frac_reward_zero_std` does not.

A third fact, cheap to miss: 29.5% of untrained GeoGuesser episodes scoring zero does **not** fill a group of 8. Drawn independently that is 0.295⁸ ≈ 5.7×10⁻⁵. Groups die when the eight rollouts of *one location* all land on the same side of the cliff. It is a within-task property. That is why `ACCUM=2` (one task per step) and `ACCUM=4` (two tasks, between-task variance in the std) are not the same optimiser.

## What GeoGuesser's published numbers already said

20,000 simulated groups of 8, task-level error log-normal around the published median of 1,226 km, 29.5% missing guesses, mean cost 0.13. `python train/analyse_geoguesser.py`.

| regime | dead groups | of which cliff | GRPO scale, median / p95 |
|---|---:|---:|---:|
| game curve, subtract-and-floor | 6.43% | 6.42% | 6.5× / 10,000× |
| after one *same-task* resample | 4.56% | — | — |
| mixture × (1 − cost), the training reward | 0.01% | 0.00% | 4.5× / 15× |
| run 1, eight identical city guesses | **100%** | 0% (all collapse) | **10,000×** |
| run 1, 15 km of jitter around 662 km | 0% | 0% | **277× / 461×** |

The mixture they already ship fixes the cliff on an untrained policy. It does not fix the amplifier once the policy has made up its mind. Eight identical guesses hit TRL's `1e-4` floor and scale the residual by ten thousand. Fifteen kilometres of jitter around the published ckpt1000 median is enough to keep the group technically live and still divide by ~0.004.

Centered ranks on the same groups are in [−0.5, 0.5] whether the kilometre gaps are 15 or 15,000. That is the whole point of them.

## What this environment is

The original 2,309-word answer list. Six guesses. Any 5-letter string is legal. The observation is the coloured history and nothing else — the answer is not written into a failed episode, which is the GeoGuesser reveal bug in miniature. Information gain is computed on the remaining set behind the reward; the policy never sees that set.

Project 00's Wordle is 50 words. Training against that is memorising a list.

```
04-wordle/
├── envs/wordle/     game, 2,309-word list, information gain, rollout
├── train/           estimator, CPU trainer, TRL recipe, GeoGuesser classifier
└── results/         the two JSON files the tables above were printed from
```

## Try it

```bash
cd 04-wordle
python envs/wordle/rollout.py zonal
python -m pytest -q
python train/tiny_grpo.py --steps 200 --out results/tiny-ablation.json
```

`rollout.py zonal` plays CRANE → … → ZONAL in five guesses. The answer does not appear in the observation except as a guess the player typed.

## Train a language model against it

In-process, no Space, no panorama store.

```bash
hf jobs uv run --flavor a100-large --timeout 6h --image huggingface/trl \
  --secrets HF_TOKEN \
  -e MODEL=Qwen/Qwen3.5-4B \
  -e REWARD_SHAPE=process \
  -e SCALE_REWARDS=none \
  -e MAX_STEPS=200 \
  -e NUM_GENERATIONS=8 \
  train/grpo_wordle.py
```

Watch `alive/frac_dead` against TRL's `frac_reward_zero_std`. They should agree — both count every zero-std group. Watch `alive/frac_collapse` separately; TRL does not split cliff from collapse. If collapse is high and solve rate has been flat for 50 steps, stop. That is the $70.

Dynamic sampling — keep drawing until the group is live — lives in `tiny_grpo.py` and not in the TRL script. TRL marks it unsupported. Forking `GRPOTrainer` to add it is a TRL patch.

## A GeoGuesser run 4

`train/geoguesser_run4.py` is the classifier plus the env we would actually launch:

```
SCALE_REWARDS=none   # ranks sit on top of a mean-baseline; do not divide by std
BETA=0
ACCUM=2              # one task per step
COST_SCALE=0.2       # keep the policy looking
MAX_STEPS=250        # run 1 plateaued here
```

It is not a fourth GeoGuesser run until someone launches it. What we would treat as success: dead groups stay below 5%, turns stay above ~2, eval at 250 is not worse than eval at 200. What we would treat as failure: the one-glance city policy coming back, which is `COST_SCALE=1.0` plus `scale_rewards=group`, which is run 1.

## Layout of the claim

The algorithm is not the hard part, until the environment's reward cannot tell eight rollouts apart, at which point it is the only part. GeoGuesser measured that and then trained through it for 750 steps. This folder is the loop that would have stopped, or switched reward, instead.
