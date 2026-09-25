---
title: Data Agent
emoji: 📊
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
---

# blackbox-opencode

**Train a policy to actually analyse data.** An agent is given a question and a directory of real
tables, works in a sandbox with its own tools, and files its answer to a file. You get back the
**engine's own token ids** for every model call it made, a per-token loss mask, and the task's reward.

## Overview

The agent's loop belongs to the harness, not to the trainer. opencode runs inside the sandbox with
bash, read, edit and grep; it decides how many turns to take and when it is done. This environment
records that loop through a **capture proxy** the agent talks to instead of the engine, so what comes
back is what the model actually saw, token for token.

That last part is the whole point. A trainer that re-renders each turn's prompt with
`apply_chat_template` is training on a *different* string from the one the engine scored — measured on
Qwen3.5-4B, the re-rendered prompt matched the engine on **0 of 28 turns**, which fragmented one long
conversation into many short ones and collapsed the run at its first weight update. Here the engine
returns `prompt_token_ids` and turn *k+1*'s prompt **is** the canonical tokenisation of everything
before it, so turns link by exact token prefix and nothing is ever tokenised locally.

## Quick start

```bash
uv sync
SPLITS=train:medium LLM_URL=http://127.0.0.1:8455/v1 MODEL=Qwen/Qwen3.5-2B ./serve.sh
```

`serve.sh` is the same configuration a deployment uses, so a client cannot tell the two apart.
Credentials come from the environment (`E2B_API_KEY`, `HF_TOKEN`) and are never arguments.

```python
from data_agent_env import DataAgentEnv

# The folder is `blackbox-opencode`, which is not a legal Python identifier; `uv sync` installs it
# under the package name `data_agent_env` via the package-dir mapping in pyproject.toml.
env = DataAgentEnv("http://127.0.0.1:8200")
print(env.capabilities())  # usable sandboxes, splits, concurrency budget
print(env.splits())  # train / test / eval, and per-difficulty variants

result = env.run_rollout(
    split="train:medium",
    index=0,
    llm_url="http://127.0.0.1:8000/v1",
    model="Qwen/Qwen3.5-2B",
)
print(result.reward, result.answer, len(result.turns))
print(result.turns[0].prompt_token_ids[:8])  # the engine's tokenisation, not ours
```

## Training with TRL

The environment does the work; the training script stays short. It hosts vLLM and orchestrates —
everything else (dataset, sandbox, agent, grading, tokenisation, loss mask) is on this side.

```python
from data_agent_env import DataAgentSessionFactory
from datasets import Dataset
from trl.experimental.async_grpo import AsyncGRPOTrainer, HarnessRolloutWorker

factory = DataAgentSessionFactory(
    "http://127.0.0.1:8200", split="train:medium", llm_url=VLLM_URL, model=MODEL
)
dataset = Dataset.from_list(factory.prompt_rows())
worker = HarnessRolloutWorker(harness_session_factory=factory, harness_adapter=None, ...)
AsyncGRPOTrainer(model=MODEL, args=config, train_dataset=dataset, rollout_worker=worker).train()
```

`harness_adapter=None` selects the **loop-owning** path: the trainer blocks on the rollout and reads
the recorded trace, rather than driving turns itself.

> **Serve the engine with `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`.**
> Without them capture degrades to text level *silently*: every rollout looks completely normal and
> carries nothing to train on. The session mint probes for this and refuses, rather than letting a run
> spend hours discovering it. An eval run may pass `require_tokens=False` — a text-only endpoint is a
> perfectly good eval backend.

## Splits

Splits come from [`HuggingEnvs/data-agent`](https://huggingface.co/datasets/HuggingEnvs/data-agent),
a flat row-based dataset. A difficulty is a **named split**, `train:medium`, not a filter argument:
a filter shifts every index after it, and the index is task identity everywhere downstream.

| split | meaning |
| --- | --- |
| `train`, `test`, `eval` | the whole split |
| `train:easy`, `train:medium`, `train:hard` | one difficulty tier of it |

## Sandboxes

`e2b` and `hf`, chosen per rollout (`run_rollout(sandbox=...)`), so switching is one word and needs no
redeploy. Install only the backend you will use; the import is lazy.

The one thing that must never be hardcoded is the **home directory**: E2B runs the agent as `user`
(`/home/user`), Hugging Face sandboxes run as root (`/root`). Get it wrong and opencode writes its
provider config where it cannot read it back, so the agent starts with **no model configured** and
makes zero model calls — which arrives as a flat-zero reward that looks exactly like a policy that
cannot do the task. `sandbox/sandbox_home()` is the single place that knows.

## Reward

`correctness` comes from the task's own grader: exact match, then numeric within `atol`/`rtol`, then
list comparison. On top of it:

- **Filing the answer is what counts.** An answer only stated in chat gets partial credit (0.3), never
  full. A string that merely *narrates* the submission (`echo -n "2.14" > answer.txt`) gets nothing —
  42% of partial credit once went to exactly that.
- **The efficiency bonus is gated on a solve and is never a penalty.** Ungated, "make zero tool calls"
  becomes the highest-scoring move available to a policy that cannot solve the task.
- **An ungraded rollout returns `reward=None`, never `0.0`.** `None` means the infrastructure failed
  and the trainer drops it from the group baseline; `0.0` means the policy was wrong. Collapsing the
  two silently turns a flaky sandbox into a training signal.

## Concurrency

Three limits stack, and the tightest is not the obvious one:

| limit | value | why |
| --- | --- | --- |
| capture proxy | **the real ceiling** | single uvicorn process; `/health` starved at ~200 concurrent, crashed at 320 (3,525 fds, 542 threads, 6.7 GB) |
| `DATA_AGENT_MAX_CONCURRENT` | 64 | rollouts executing; a semaphore, so an over-limit rollout waits rather than failing |
| `MAX_CONCURRENT_ENVS` | 128 | WebSocket sessions; must exceed `num_generations` or rollouts queue at the door |
| E2B account | 500 | sandboxes; capture gives out first |

Sessions release slowly, not instantly, and killing a client leaks its sessions — leftovers collide
with the next run's claim and surface as a burst of `CAPACITY_REACHED`. A training run and an
evaluation run share one deployment, and the eval must not be able to starve training out.

## Step limits

`agent_step_limit` is enforced **in the capture proxy**, which is the only component that sees every
model call. It is not enforced by the agent's own config: measured on opencode 1.18.30 against a fake
engine, `agent.build.steps=3`, `maxSteps=3` and no setting at all each produced **61** model calls.

At the cap the proxy answers a terminal completion itself — the agent's loop ends cleanly and
`opencode run` exits 0 — and because that happens before capture ingests anything, no turn the model
never generated can enter the training data.

## Environment variables

| variable | meaning |
| --- | --- |
| `DATA_AGENT_SPLITS` | comma-separated splits to serve (default `train`) |
| `DATA_AGENT_SANDBOX` | default backend, `e2b` or `hf` |
| `DATA_AGENT_MAX_CONCURRENT` | rollouts executing at once (default 64) |
| `DATA_AGENT_CAPTURE_PORT` | port the capture proxy binds (default 8300) |
| `CAPTURE_PUBLIC_URL` | how the **sandbox** reaches that port, when it is not localhost |
| `OPENENV_LLM_URL` / `OPENENV_MODEL` | default engine; optional, since a rollout may name its own |
| `HF_TOKEN` | reads the dataset and stages each task's tables |
| `E2B_API_KEY` | required for the `e2b` backend |

`OPENENV_LLM_URL` is optional on purpose. The dataset and its prebuilt sandbox templates are the
expensive things to host; an engine restarts every training run, and a train-tier engine and an
eval-tier one are usually both wanted against the same tasks at once.
