<div align="center">

<h1>Data Agent</h1>

<h3>Give an agent a question and a directory of real tables, and train it on what it actually did</h3>

<p>Two black-box environments over the same data-analysis tasks, and the token-level contract that makes an agent's own loop trainable.</p>

<a href="https://huggingface.co/datasets/HuggingEnvs/data-agent"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Flat%20tasks-4F46E5?style=for-the-badge&labelColor=1a1a1a" alt="Flat dataset" height="32"></a>
<a href="https://huggingface.co/datasets/HuggingEnvs/data-agent-harbor-train"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Harbor%20catalog-7C3AED?style=for-the-badge&labelColor=1a1a1a" alt="Harbor catalog" height="32"></a>
<a href="https://github.com/huggingface/OpenEnv"><img src="https://img.shields.io/badge/framework-OpenEnv-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="OpenEnv" height="32"></a>

</div>

---

## The problem

The agent's loop is not yours. opencode runs inside a sandbox with bash, read, edit and grep; it
decides how many turns to take, when to look at the data and when it is done. A trainer never drives
a turn. So everything you need in order to train — the tokens, the logprobs, the loss mask — has to be
recovered by *observing* the model calls rather than by making them.

The obvious way to recover them is wrong, and wrong silently. Rebuild each turn's prompt with
`apply_chat_template` and you get a **different string** from the one the engine scored. Measured on
Qwen3.5-4B, the rebuilt prompt matched the engine on **0 of 28 turns**. Nothing errors. The rollout
log looks healthy, the reward curve looks plausible, and the run collapses at its first weight update
— because what fragmented was the conversation itself: turns that should have chained by exact token
prefix instead looked like unrelated short rollouts, and every fragment still trained.

The fix is to never tokenise locally. A capture proxy sits between the agent and the engine and keeps
what the engine returned: `prompt_token_ids` for every call. Turn *k+1*'s prompt **is** the canonical
tokenisation of everything before it, so turns link by exact token prefix, by construction.

## Two environments

Both are black box in the same sense — the agent owns its loop. They differ in where a task lives and
who grades it.

| | [`blackbox-opencode`](envs/blackbox-opencode) | [`blackbox-harbor`](envs/blackbox-harbor) |
| --- | --- | --- |
| task is | a **row** in [`HuggingEnvs/data-agent`](https://huggingface.co/datasets/HuggingEnvs/data-agent) | a **directory** with `task.toml`, `Dockerfile`, `tests/` |
| setup | the env stages tables from a Hub bucket | the task's own Dockerfile + healthcheck |
| agent | opencode | any of Harbor's harnesses, chosen per rollout |
| grading | the env's verifier | the task's own `tests/grader.py` |
| change a task by | editing a dataset row | editing a task directory |
| served by | this package | OpenEnv's `harbor_env`, via the CLI |

Neither replaces the other. The flat one iterates fast, because a task is data. The Harbor one is what
you want when a task must ship its own container, or when you want one policy trained against several
agent harnesses so it does not learn a single harness's habits.

They are not independent, which is worth knowing before you edit either: every Harbor task directory
carries a `tests/grader.py` that is **the same grader** as the flat env's — 120 of its 130 non-comment
lines are identical. The catalog was baked from it.

## Run one

```bash
cd envs/blackbox-opencode
uv sync
./serve.sh &                       # http://localhost:8200/web/
uv run python rollout.py --llm-url http://127.0.0.1:8455/v1 --model Qwen/Qwen3.5-2B
```

`rollout.py` does not just print a reward. It asserts the three things that are silent when wrong:
the rollout is train tier, every turn carries `prompt_token_ids`, and turn *k+1*'s prompt equals turn
*k*'s prompt plus its completion.

For the Harbor variant, see [`envs/blackbox-harbor`](envs/blackbox-harbor) — it is a CLI recipe rather
than a package.

## Serve the engine correctly

```bash
vllm serve Qwen/Qwen3.5-2B --port 8455 \
    --return-tokens-as-token-ids --logprobs-mode processed_logprobs
```

Without those two flags capture degrades to text level **silently**: rollouts come back with a reward
and a transcript and nothing to train on. The environment probes the engine when it mints a capture
session and refuses a training rollout rather than letting a run discover this hours in. An evaluation
rollout may pass `require_tokens=False` — a text-only endpoint is a perfectly good eval backend, and
refusing it would rule out every hosted provider.

## The reward

`correctness` comes from the grader: exact match, then numeric within `atol`/`rtol`, then list
comparison. Three rules on top of it, each of which exists because of a measured failure:

- **Filing the answer is what counts.** An answer only stated in chat gets partial credit, never full.
  A string that merely *narrates* the submission — `echo -n "2.14" > answer.txt` — gets nothing; 42% of
  partial credit once went to exactly that.
- **The efficiency bonus is gated on a solve, and is never a penalty.** Ungated, "make zero tool calls"
  becomes the highest-scoring move available to a policy that cannot solve the task.
- **An ungraded rollout returns `reward=None`, never `0.0`.** `None` means the infrastructure failed
  and the trainer drops it from the group baseline; `0.0` says the policy was wrong. Collapsing the
  two turns a flaky sandbox into a training signal.

## Step limits

`agent_step_limit` is enforced **in the capture proxy**, because that is the only component that sees
every model call. It is *not* enforced by the agent's own config: measured on opencode 1.18.30 against
a fake engine that always asks for one more tool call, `agent.build.steps=3`, `maxSteps=3` and no
setting at all each produced **61** model calls.

It matters beyond cost. AsyncGRPO packs a whole rollout into one training row and every turn re-sends
the entire conversation, so packed length grows with the **square** of the turn count — and one
runaway rollout holds its whole GRPO group hostage.

## Status

The environments are landing; nothing here is verified end to end yet, and no training results are
published in this project. Both depend on core changes still in review upstream:

- **OpenEnv** — `TraceEntry` carrying `prompt_token_ids` and `loss_mask`, `CaptureServer` promoted
  into `openenv.core.harness.capture`, and a per-session model-call budget in the proxy.
- **TRL** — `async_grpo` consuming those ids instead of re-rendering, and `TurnRecord.output_mask`.
