<div align="center">

<img src="./banner.png" alt="SmolDataEnvs" width="100%">

# SmolDataEnvs

[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-SmolDataEnvs-FFD21E?style=for-the-badge&labelColor=1a1a1a)](https://huggingface.co/collections/FineEnvs/smoldataenvs)
[![Harbor Visualiser](https://img.shields.io/badge/%F0%9F%A4%97%20Browse-the%20tasks-FFD21E?style=for-the-badge&labelColor=1a1a1a)](https://huggingface.co/spaces/HuggingFaceH4/harbor-visualiser?dataset=FineEnvs/SmolDataEnvs-harbor-train)

</div>

> **5.5K+ RL tasks for hill-climbing small models in code and data science.**

<div align="center">

<img src="./curves.gif" alt="Reward and held-out pass@k climbing over 1,119 GRPO steps" width="100%">

<sub>A 2B model on these tasks. Left: what it optimises. Right: 144 held-out tasks it never trains on.<br>
Two runs over the same 5,000 tasks — <b>shuffled</b> against a <b>curriculum</b> ordered easiest to hardest.</sub>

</div>

## What this is

A small model that can write code is not the same thing as a small model that can *use* code to
answer a question about data. The second one has to open a file it has never seen, work out what is
in it, decide what to compute, run something, read the result, and commit to an answer. That is a
long-horizon task with a short, checkable answer at the end — which is exactly the shape RL wants.

SmolDataEnvs is 5,394 of those tasks. Each one is a real Kaggle dataset, a question a human actually
asked about it in a notebook, and the answer that human computed. The agent gets a sandbox with the
files in it and has to produce the answer.

**Verified, not judged.** Grading is an exact comparison against a known answer, through a ladder of
checks: exact match → numeric with per-task tolerances → list and percent normalisation → symbolic
equivalence. No LLM sits in the reward path. The reward cannot drift because there is nothing in it
to drift.

Every task was built the hard way, too: strong agent models had to solve it in a live sandbox and
reproduce the gold answer before it was allowed in. Anything ambiguous or un-checkable was dropped.
So a zero from a task is a statement about your model, not about the task.

## What's in it

| Split | Tasks | Easy | Medium | Hard |
|---|---|---|---|---|
| train | 5,000 | 1,433 | 2,845 | 722 |
| test | 250 | 33 | 118 | 99 |
| eval | 144 | 16 | 74 | 54 |

Drawn from 471 Kaggle datasets. Answer types run numeric (2,906), short label (1,409), list (367),
free-form (152), yes/no (127) and csv-list (39) — the match mode is per task, and so are the numeric
tolerances.

**The held-out splits are deliberately harder than train.** Train is 29% easy and 14% hard; test and
eval are 11–13% easy and 38–40% hard. Worth knowing before reading any eval number from this dataset,
including ours.

## Five repos, three ways in

| Repo | Use it when |
|---|---|
| [`SmolDataEnvs`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) | you want the tasks as rows: question, answer, match mode, data pointer. Plus `grader.py` |
| [`SmolDataEnvs-sft`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft) | you want 4,677 verified agent trajectories to fine-tune on |
| [`SmolDataEnvs-harbor-train`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-train) | you want to actually run RL: 5,000 sandboxed environments |
| [`SmolDataEnvs-harbor-test`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-test) | you want a number to quote: 250 held-out tasks |
| [`SmolDataEnvs-harbor-eval`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-eval) | you want to watch a run: 144 tasks, cheap enough to run often |

## Start here

Two notebooks. Open either in Colab and run it top to bottom.

| | | |
|---|---|---|
| **1 · SFT** | imitate 4,677 verified trajectories | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adithya-s-k/FineEnvs/blob/main/04-smoldataenvs/notebooks/01_sft.ipynb) |
| **2 · RL** | build the environment, then GRPO against the grader | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adithya-s-k/FineEnvs/blob/main/04-smoldataenvs/notebooks/02_rl.ipynb) |

Both finish on a free T4 with their defaults, and both end with the command that runs the same thing
properly on a Hugging Face Jobs GPU.

## How a rollout works

One turn, no environment framework:

```
prompt → model writes a Python program → a sandbox runs it against the real tables
       → whatever it printed last is the answer → the dataset's grader says 1.0 or 0.0
```

```
scripts/rollout.py        the environment: prompt in, reward out
scripts/train_sft.py      imitate the verified trajectories
scripts/train_grpo.py     RL against the grader
scripts/eval_pass1.py     score any model on the held-out split
```

You can watch the whole thing happen without a GPU, on the live sandbox:

```bash
uv run scripts/rollout.py                # gold code scores 1.0, wrong code scores 0.0
uv run scripts/train_grpo.py --dry-run   # the training rewards, without the training
uv run scripts/eval_pass1.py --dry-run   # the scoring path, without the GPU
```

## Run it

```bash
# 1. measure the base model first, or the rest is a story without a control
./scripts/run_on_hf_jobs.sh eval Qwen/Qwen3.5-2B

# 2. warm start on trajectories that are known to be correct
./scripts/run_on_hf_jobs.sh sft you/smoldataenvs-sft-2b

# 3. RL against the grader
MODEL=you/smoldataenvs-sft-2b ./scripts/run_on_hf_jobs.sh grpo you/smoldataenvs-grpo-2b

# 4. the same measurement as step 1, so the numbers are comparable
./scripts/run_on_hf_jobs.sh eval you/smoldataenvs-grpo-2b
```

Everything runs on Hugging Face Jobs with the `huggingface/trl` image. Generation during
GRPO is colocated in the training process, so there is no inference server to stand up
alongside it. The rollout sandboxes are Jobs too, which is why the training job needs your
token forwarded (`--secrets HF_TOKEN`).

The container is deleted when a job ends, so every training mode insists on a Hub model id:
a run that does not push is a run you cannot keep.

## Notes on the environment

- **One sandbox per process, not per rollout.** A sandbox starts in 6–10s, which would be the
  entire per-rollout budget. `rollout.py` starts one and swaps the contents of `/home/user/input`
  per task; GRPO asks for N attempts at the same task in a row, so the data pull happens once.
- **A crashing program is not an error.** It is a reward of zero and a gradient. The runner returns
  tracebacks as text rather than raising.
- **Everything runs non-thinking.** Qwen3.5 opens with a `<think>` block by default; a single
  program does not need one, and SFT, RL and eval must render the same template or you train one
  model and measure another.
- **The shaping reward is exploitable, and it got exploited.** `+0.1` for a program that runs exists
  because early on almost nothing is correct and an all-zero group carries no gradient. The first
  version paid it for "no traceback" — which a program that does nothing also satisfies. Within 70
  steps the policy was emitting an empty `<think></think>`, collecting 0.1 per rollout, with zero
  variance and therefore zero learning. It now requires the program to have **printed** something,
  and a completion that is not a program is rejected by `compile()` before a sandbox is spent on it.

## Measure before you train

Run the eval once on the untrained model. Without that number nothing afterwards
means anything, and it takes about ten minutes:

```bash
./scripts/run_on_hf_jobs.sh eval Qwen/Qwen3.5-2B
```

Two things worth knowing before you read any result of your own:

- **Use the whole split.** A subset large enough to be quick is not large enough to
  separate 4% from 6%. On 144 tasks the confidence interval is still wide; on 24 it
  is wider than any effect a short run will produce.
- **Train on a mix of tiers.** `DIFFICULTY=easy` makes rewards dense, which is useful
  for getting the loop moving, but a model trained only on easy tasks is a model that
  is good at easy tasks. `DIFFICULTY=easy,medium` or no filter at all is what you
  want before believing a number.

Runs land in [a Trackio Space](https://huggingface.co/spaces/AdithyaSK/smoldataenvs-trackio),
one project, training and eval together.

## Provenance

Built from the [jupyter-agent dataset](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset)
(real data-science notebooks over Kaggle datasets), then verified end to end in a sandbox.

## Citation

```bibtex
@misc{fineenvs,
  author = {Kolavi, Adithya S},
  title  = {FineEnvs: Open Source RL Environments for LLM Agents},
  year   = {2026},
  url    = {https://github.com/adithya-s-k/FineEnvs}
}
```
