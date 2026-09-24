# Watercolour

Train `Qwen/Qwen3.5-35B-A3B` to paint watercolours by writing
[p5.brush](https://github.com/acamposuribe/p5.brush) sketches. The sketch is
rendered in headless Chromium and the render is scored, so **the reward is a
picture rather than a test suite**.

Reproduces the idea in [Surya Narreddi's "RL'ing Qwen to paint with
code"](https://surya.website/rling-qwen-to-paint-with-code)
([tweet](https://x.com/kickingkeys/status/2091570990048276897)). The idea is his.
His results look better than these. His write-up ships no dataset, scripts or
model, which is the gap this fills.

The full story, from the reward design to what the three runs actually learned, is
in the blog post: [Training a coding model to paint watercolours with TRL and
OpenEnv](https://huggingface.co/blog/train-to-paint-with-code).

## What is here

| | |
|---|---|
| `envs/watercolour/core/` | the gate, the renderer, the two judges, the reward, the domains |
| `envs/watercolour/openenv/` | the OpenEnv port: client, models, environment, app, Dockerfile |
| `envs/watercolour/hpsv3/` | the HPSv3 scorer as its own deployable Space. **Deploy this first** |
| `train/watercolour_grpo.py` | the GRPO trainer |
| `train/pool_photos.py` → `pool_generate.py` → `pool_rate.py` | how the reference pool was built, in order |
| `train/pool_calibrate.py` | sorts candidates by how often the policy already beats them |
| `results/` | the per-step curves as CSV, so they outlive the dashboards |

## The reward

| term | default | what it measures |
|---|---|---|
| `gate` | 0.05 | compiles, paints something, does not cheat |
| `length` | 0.05 | a ramp towards elaboration |
| pairwise judge | 0.60 | style, against references drawn from a pool |
| [HPSv3](https://huggingface.co/MizzenAI/HPSv3) | 0.30 | aesthetic preference on the render |

Every weight is an environment variable, so **one deployment covers any split**.
The three runs below differ only in these numbers.

**What the pool contains is the reward function.** Swap
[the dataset](https://huggingface.co/datasets/HuggingEnvs/watercolour-reference-pool)
and you have changed what the agent is rewarded for, without touching code.

## Hardware, and what it costs

**Read this before starting.** Three paid pieces are needed at the same time.

| piece | hardware | why |
|---|---|---|
| the trainer | 1x **H200** | a 35B with LoRA and 8 generations per step |
| HPSv3 | 1x **a100-large** Space | a 7B preference model. Deploy `envs/watercolour/hpsv3/` |
| the pairwise judge | HF inference quota | `Qwen/Qwen3-VL-30B-A3B-Instruct` through the router |
| the env | **cpu-upgrade** Space | headless Chromium. `cpu-basic` cannot render in time |

### What one run costs

```
60 steps    17h46m on one H200      (measured, hps-only)
110 steps   32h15m and 34h49m       (measured, hps-led and judge-led)
```

Plus, **for the whole duration of the run**, an `a100-large` Space for HPSv3 and a
`cpu-upgrade` Space for the env. That is the part that surprises people: a scorer
left running costs more than you expect, and it has to stay up while you train.

A step is 8 rollouts and takes 15 to 18 minutes, of which **70 to 80% is rendering**.

### What the dataset costs

The pool is a separate, one-off cost, and the scripts are in `train/`:

| step | script | what it costs |
|---|---|---|
| 1. reference photographs | `pool_photos.py` | free. 371 `cc0`/`cc-by`/`cc-by-sa` observations from the iNaturalist API |
| 2. generate candidates | `pool_generate.py` | inference quota for four models, plus a VLM judge giving written feedback each round |
| 3. rate into tiers | `pool_rate.py` | your time. Every render looked at one at a time |

What came out: **206 paintings over three refinement rounds** (77 in round 0 with no
photograph, 68 in round 1, 61 in round 2), of which 178 ship as `love` and `okay`.
Another 421 renders were generated and set aside along the way.

Step 3 has no shortcut, and it is the step that defines the reward. If you skip it
you have not built a pool, you have collected images.

## Run it

```bash
hf jobs uv run train/watercolour_grpo.py --flavor h200 --timeout 24h --secrets HF_TOKEN -- \
  --env-url https://<you>-watercolour-env.hf.space \
  --model Qwen/Qwen3.5-35B-A3B --lora --all-linear --bf16 --gradient-checkpointing \
  --subject 'a peach hibiscus' --references 4 \
  --top-p 0.95 --top-k 20 \
  --lr 5e-5 --lr-scheduler constant_with_warmup --warmup-steps 5 \
  --scale-rewards none \
  --steps 60 --n-episodes 240 --num-generations 8 \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 \
  --max-completion-length 8192 --probe-samples 40 --film \
  --run-tag hps-only --out <you>/watercolour-grpo-hps-only --push-to-hub
```

The three runs use that same command. **What changes is two environment variables on
the env Space, and the `--run-tag`:**

| run | `WATERCOLOUR_JUDGE_WEIGHT` | `WATERCOLOUR_QUALITY_WEIGHT` | `--run-tag` | `--steps` |
|---|---|---|---|---|
| `hps-only` | `0.00` | `0.90` | `hps-only` | 60 |
| `judge-led` | `0.60` | `0.30` | `judge-led` | 200 |
| `hps-led` | `0.30` | `0.60` | `hps-led` | 200 |

`judge-led` is the split Narreddi's write-up converged on. `hps-only` switches the
pairwise judge off entirely, which is why it is the cheapest to reproduce: no
inference quota for the judge, though HPSv3 still has to be up.

Weights are read at import, so **restart the env Space after changing them** or it
keeps scoring with the old ones.

### Before that, deploy the scorer

A paused copy is published at
[`HuggingEnvs/watercolour-hpsv3`](https://huggingface.co/spaces/HuggingEnvs/watercolour-hpsv3):
duplicate it and give your copy `a100-large` hardware. It sits paused because a 7B
needs a GPU and there is no free tier that fits one. The four files that are the
whole thing also live in `envs/watercolour/hpsv3/`, and `tools/deploy.py` ships them:

```bash
python3 tools/deploy.py 02-watercolour/envs/watercolour/hpsv3 <you>/watercolour-hpsv3
```

Then give the Space GPU hardware and point the environment at it:

```
WATERCOLOUR_HPSV3_URL=https://<you>-watercolour-hpsv3.hf.space
```

The environment itself deploys the same way, from `envs/watercolour/openenv/`.

The env Space must also be running on paid hardware and have `HF_TOKEN` set. Without
the scorer the reward loses its 0.30 (or 0.90) silently, and without the token the
pairwise judge loses its 0.60. Both failures show up in the breakdown rather than as
an error, so check one rollout before starting a run.

**The `uv` header pins no versions** (`trl`, `peft`, `transformers`, `torch`), so a
run today resolves different ones than ours did. That is a real reproducibility gap.

## Results

Three runs, differing only in how the reward splits:

| run | judge | HPSv3 | steps | reward slope |
|---|---|---|---|---|
| `hps-only` | 0.00 | 0.90 | 60 | t = +6.41 |
| `judge-led` | 0.60 | 0.30 | 110 | t = +10.5 |
| `hps-led` | 0.30 | 0.60 | 110 | **t = +15.6** |

The judge runs were launched for 200 steps and stopped at 110, still climbing.
All three learn, and `frac_reward_zero_std` stayed at 0.000 in every run, so no
step ever lost its gradient. The full per-step numbers, per run, are in
[`results/`](results/), and every artifact of every run:

| run | adapter | rollouts | curve |
|---|---|---|---|
| `hps-only` | [`watercolour-grpo-hps-only`](https://huggingface.co/HuggingEnvs/watercolour-grpo-hps-only) | [`watercolour-rollouts-hps-only`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-hps-only) | `results/curve-hps-only.csv` |
| `judge-led` | [`watercolour-grpo-judge-led`](https://huggingface.co/HuggingEnvs/watercolour-grpo-judge-led) | [`watercolour-rollouts-judge-led`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-judge-led) | `results/curve-judge-led.csv` |
| `hps-led` | [`watercolour-grpo-hps-led`](https://huggingface.co/HuggingEnvs/watercolour-grpo-hps-led) | [`watercolour-rollouts-hps-led`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-hps-led) | `results/curve-hps-led.csv` |

The section below reads the validation run, `hps-only`, in detail. It is the
simplest of the three (one dense generic signal, judge off), so its mechanics are
the cleanest to decompose. The judge runs change the picture in one specific way:
with the judge on, the top of the distribution moves too, and paint coverage
roughly doubles where `hps-only` barely moved it. Their cards carry the numbers.

### The honest version

**What it learned is to stop producing bad paintings, not to paint better ones.**
Decomposed: +0.0290 of the rise comes from failing less often, +0.0017 from the
paintings that did render being better. Rollouts scoring under 0.3 fall from 33 to
11 across the run while the best of each group barely moves.

**Earlier runs were flat.** Three separate experiments on the reward
(removing the judge, changing the pool, removing renderer noise) all produced the
same line. The bottleneck was the optimiser step, not the signal: a sanity task with
no browser and no judges rose 4.4x in 13 steps at lr 3e-4 and did not move at 2e-5.

**The reward cannot reach 1.0, and the ceiling is lower than that.** 0.901 is the
absolute maximum (all eight rollouts at the best one this run produced) and **0.771**
is what "every rollout as good as the current good ones" would give. 1.0 would need
an infinite HPSv3 score.

**Pigment is not rewarded, structurally.** Correlation between reward and paint
coverage is +0.378 overall but **-0.045 inside the reward >= 0.65 band**. At equal
high reward, coverage ranges from 0.071 to 0.232. Once HPSv3 sees petals around a
centre, it stops caring whether they are painted with a wash or with paste. So the
paintings get more reliable without getting denser.

**The shortcut we did not intend**: the system prompt asks for fifteen to thirty
filled shapes, and `n_shapes` has a **+0.000** correlation with reward over 470
rollouts, at a real mean of 7.6. The policy is not paid for following that
instruction, and it does not.

**It was cut by the step counter, not by running out of progress.** The last 15
steps have a steeper slope (+0.0084/step) than the run as a whole (+0.0035). The
mechanism above projects to exhausting bad rollouts around step 94.

Everything above is recomputable from
[the rollouts dataset](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-hps-only).

## Artifacts

Everything is in the
[**Paint with Code** collection](https://huggingface.co/collections/HuggingEnvs/paint-with-code-6a955b79d63f67f1631d9be6).

| artifact | what it is |
|---|---|
| `envs/watercolour/openenv/` | the environment, deployed with `tools/deploy.py`. Needs `cpu-upgrade`, it cannot render on free hardware |
| [`watercolour-reference-pool`](https://huggingface.co/datasets/HuggingEnvs/watercolour-reference-pool) | the 178 paintings that define the reward, with the sketch behind each |
| [`watercolour-grpo-hps-only`](https://huggingface.co/HuggingEnvs/watercolour-grpo-hps-only) | the trained adapter. **Read the card**: the obvious way to load it fails silently |
| [`watercolour-rollouts-hps-only`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-hps-only) | every rollout of the run: 470 paintings, sketches and rewards, by step |
| [`watercolour-grpo-judge-led`](https://huggingface.co/HuggingEnvs/watercolour-grpo-judge-led) · [`watercolour-grpo-hps-led`](https://huggingface.co/HuggingEnvs/watercolour-grpo-hps-led) | the sibling adapters, with their rollouts datasets linked from each card |
| [`watercolour-hpsv3`](https://huggingface.co/spaces/HuggingEnvs/watercolour-hpsv3) | the scorer Space, paused on free hardware. Duplicate it on `a100-large` for a run. The model itself is [HPSv3](https://huggingface.co/MizzenAI/HPSv3) |
| [`watercolour-trackio-judge-led`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-judge-led) · [`hps-led`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-hps-led) · [`hps-only`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-hps-only) | the live training dashboards, one per run |
| `results/curve-*.csv` | the training curves as data, one CSV per run, so nothing depends on a Space staying up |

Nothing here depends on a Space being switched on. The curves are CSV, the rollouts
are a dataset, the pool is a dataset, and the environment is a Dockerfile.

## Two things that will bite you

**The env serves one session at a time** and refuses the rest with
`CAPACITY_REACHED`. It also does not reclaim a session whose client died, so a
crashed trainer leaves it blocked until the Space restarts. Do not probe it while
training against it.

**A render takes 69 to 96 seconds** against a 90 second deadline, and it is 70 to
80% of each training step. Nobody has looked at why.

## Another domain

The method is aesthetic reward over generated code, not flowers.
[Alex Yango did animals with the same mechanism](https://x.com/alexyango/status/2091696296931574217),
and [Brendan Hogan did canvas animations](https://x.com/brendanh0gan/status/2092650655789855222)
and independently found the same pattern: his working animations went from 31% to
90%, which is the same mechanism measured here.

For a domain of your own: write a `Domain` in `core/domains.py` (the subjects, the
composition, the judge criteria) and **build a pool**. The first is an afternoon.
The second is 90% of the work. `core/domains.py` ships a second domain,
`JELLYFISH`, complete except for its pool, and left out for exactly that reason.

## Citation

```bibtex
@misc{fineenvs,
  author = {Kolavi, Adithya S},
  title  = {FineEnvs: Open Source RL Environments for LLM Agents},
  year   = {2026},
  url    = {https://github.com/adithya-s-k/FineEnvs}
}
```
