# blackbox-harbor

The same data-analysis tasks as [`blackbox-opencode`](../blackbox-opencode), served the other way: as
a **Harbor task catalog** rather than a flat dataset. There is no environment package here —
OpenEnv's `harbor_env` already serves this shape, so this is a recipe for the CLI plus a script that
proves the rollouts come back trainable.

## Which one should I use?

| | [`blackbox-opencode`](../blackbox-opencode) | this (`harbor_env` + the CLI) |
| --- | --- | --- |
| dataset | [`HuggingEnvs/data-agent`](https://huggingface.co/datasets/HuggingEnvs/data-agent) — flat, one row per task | `HuggingEnvs/data-agent-harbor-{train,test,eval}` — one directory + `task.toml` per task |
| task setup | the env stages tables from a bucket | the task's own Dockerfile and healthcheck |
| agent | opencode | any of Harbor's ~30 harnesses, per rollout |
| grading | the env's verifier | the task's own verifier |
| to change a task | edit a dataset row | edit a task directory |

Neither replaces the other. Use the flat env when the task definition is data and you want to
iterate on it quickly; use Harbor when the task ships its own container and verifier, or when you
want to train one policy against several agent harnesses.

## Serve it

```bash
openenv harbor serve \
  --dataset HuggingEnvs/data-agent-harbor-train \
  --llm-url http://127.0.0.1:8455/v1 \
  --model Qwen/Qwen3.5-2B \
  --port 8000 --capture-port 8100 \
  --expose gradio
```

`--expose` is how the **sandbox** reaches the capture proxy, which is not the same as how the trainer
reaches the env server. The sandbox is on another machine, so `direct` only works when this host is
already routable from it; `gradio` and `cloudflare` mint a public URL.

> **Serve the engine with `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`.**
> Without them the engine probes as eval tier and every rollout comes back with reward and a trace
> but nothing trainable — silently. `openenv harbor info` reports the measured tier before you start.

One engine is not a requirement of the deployment. `--llm-url` sets a *default*; each rollout may name
its own, which is what lets a training run and an evaluation run share one server. The dataset and its
prebuilt sandbox templates are the expensive things to host; the engine is the cheap, changing part.

## One rollout, no trainer

```bash
openenv harbor rollout \
  --dataset HuggingEnvs/data-agent-harbor-train \
  --llm-url http://127.0.0.1:8455/v1 --model Qwen/Qwen3.5-2B \
  --harness opencode --sandbox e2b --indices 0,1,2
```

This runs the whole path — engine, capture proxy, sandbox, harness, task verifier — with **no env
server in the way**. When something breaks, that halves the search space immediately: if `rollout`
works and `serve` does not, the fault is in the serving layer and nothing below it.

## Check the rollouts are trainable

```bash
uv run python rollout.py --server http://127.0.0.1:8000 \
  --llm-url http://127.0.0.1:8455/v1 --model Qwen/Qwen3.5-2B
```

It asserts the three things that are silent when wrong: `rollout_type == "train"`, every turn carries
non-empty `prompt_token_ids`, and turn *k+1*'s prompt equals turn *k*'s prompt plus its completion.
That last one is the contract — it is what makes turns link by exact token prefix instead of by a
re-rendered string.

## Train on it

```python
from harbor_env.harness import HarborSessionFactory

factory = HarborSessionFactory(
    server_url="http://127.0.0.1:8000",
    split="HuggingEnvs/data-agent-harbor-train",
    llm_url=VLLM_URL,          # the same engine the trainer syncs weights into
    model=MODEL,
    harness="opencode",
    sandbox="e2b",
    agent_step_limit=10,
)
```

`agent_step_limit` is worth setting. AsyncGRPO packs every turn of a rollout into one training row and
each turn re-sends the whole conversation, so packed length grows with the **square** of the turn
count — an unbounded rollout can OOM the loss step while every rollout log line looks healthy.

## The suite declares too little memory, and it OOM-kills rollouts

**Every task in `HuggingEnvs/data-agent-harbor-{train,test,eval}` declares `cpus = 1` and
`memory_mb = 1024`.** Harbor passes both straight into `AsyncTemplate.build`
(`harbor/environments/e2b.py:192`), and `_resource_value` reads them from `task.toml` with no
environment override (`harbor/environments/base.py:315`) -- so that is the size of the sandbox the
agent actually gets.

It is not enough. Task 0's bucket is 0.31 GB and pandas wants roughly 3-5x a file's size resident, so
1 GB is marginal at best. Measured on `0000_324_324276_qa_3` with Qwen3.5-2B:

| | `cpus=1, memory_mb=1024` (as shipped) | `cpus=2, memory_mb=4096` |
| --- | --- | --- |
| outcome | `FAILED (NonZeroAgentExitCodeError)`, **exit 137**, 517 s | `ok`, 158 s |
| capture | 0/1 usable | **1/1 usable** |
| turns / trainable tokens | 17 / 3252, discarded | 11 / 231 |
| graded | no -- the verifier never ran | yes |

Exit 137 is SIGKILL, i.e. the OOM killer. The expensive part is that an OOM-killed rollout files no
answer, which scores **identically to a model that could not do the task** -- so a training run reads
as a policy that never learns, with nothing in the reward to say otherwise.

**The fix belongs in the dataset**, not here: bump `[environment] cpus` and `memory_mb` in the task
files. Patching the downloaded cache works for one run and is undone the moment the server refetches.

## Warm the sandbox templates first

E2B builds one template per `env_hash`, from the task's **own Dockerfile** — `docker_image` in the
task config is ignored whenever a Dockerfile exists. Concurrent first use races that build and fails
with `404 tag default`, so run a single serial rollout once before any parallel run.

## Concurrency

The capture proxy is a single uvicorn process and is the real ceiling: `/health` starved at ~200
concurrent sessions and the process crashed outright at 320 (3,525 fds, 542 threads, 6.7 GB). E2B
allows 500 sandboxes per account, so capture gives out first. Size `MAX_CONCURRENT_ENVS` well under
that, and remember a training run and an eval run share one deployment.
