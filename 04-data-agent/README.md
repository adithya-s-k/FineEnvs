<div align="center">

<h1>Data Agent</h1>

<h3>Give an agent a question and a directory of real tables, and train it on what it actually did</h3>

<p>Two black-box environments over the same data-analysis tasks, and the token-level contract that makes an agent's own loop trainable.</p>

<a href="https://huggingface.co/datasets/HuggingEnvs/data-agent"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Flat%20tasks-4F46E5?style=for-the-badge&labelColor=1a1a1a" alt="Flat dataset" height="32"></a>
<a href="https://huggingface.co/datasets/HuggingEnvs/data-agent-harbor-train"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Harbor%20catalog-7C3AED?style=for-the-badge&labelColor=1a1a1a" alt="Harbor catalog" height="32"></a>
<a href="https://github.com/huggingface/OpenEnv"><img src="https://img.shields.io/badge/framework-OpenEnv-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="OpenEnv" height="32"></a>

</div>

---

## Current evidence and source snapshots

The [SFT/RL evidence packet](reports/sft-rl-article-handoff-20260929/HANDOFF.md) covers Qwen3.5-2B and LFM2.5-2.6B, including epoch/checkpoint scores, runtimes, tool efficiency and comparison caveats. Public dashboards: [SmolDataEnv RL](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio) and [SmolDataEnv SFT](https://huggingface.co/spaces/FineEnvs/data-agent-sft-trackio).

Preserved implementations: [SFT experiment source](model_runs/sft/README.md), [dashboard source](hf/dashboards/README.md), and [local validation status](reports/LOCAL_SNAPSHOT_20260930.md). Dated reports below describe the earlier runs and their original configurations.

## Completed comparison — 2026-09-17

All three async Qwen3.5-2B runs reached **1,000 optimizer steps**, and every scheduled
checkpoint evaluation is complete: **250 fixed tasks × four harnesses × pass@1**.
Best measured scores were **37.0% at step 500** for Harbor multi-harness,
**29.8% at step 1,000** for native OpenCode, and **39.5% at step 700** for Harbor
OpenCode-only. The two Harbor runs declined after their peaks.

See [results and training/eval observations](results.md), the
[detailed evaluation analysis](reports/three-run-analysis-20260917/REPORT.md), and
the [public comparison dashboard](https://huggingface.co/spaces/HuggingEnvs/data-agent-training-comparison-trackio).
The comparison is observational: task exposure, token budgets, backends and resume
histories differ. Historical notes below describe earlier states of these experiments.

## Historical training notes — 2026-09-15

Job **78956** resumes checkpoint **53** of job 78831. Qwen3.5-2B uses LR `3e-6`,
eight generations, **GAS4**, max staleness **4**, a **40,960-token packing target**
and compiled vLLM. The worker ceiling remains 32, with **16 total outstanding
rollouts** covering generation and the training queue. This addresses the large
stale backlog observed when matching the blackbox reference without backpressure.
All four harnesses remain enabled; the 1,000-task pool is 150 easy / 600 medium / 250 hard,
one harness per task per pass.

All rows of an admitted rollout finish before the optimizer changes weights.
Exact long rows retain context up to 131,072 tokens; loss uses one supervised-token
mean across the update. Save every **50 optimizer steps**, plus hourly recovery
checkpoints. Fixed **1,000-cell pass@1** evaluations run every **100 steps** and at
the true final checkpoint on separate TP1/DP2 GPUs. The step-100 milestone is achieved; the long run and checkpoint evaluations continue.

**Checkpoint 100 is complete** (06:56 UTC), and training has continued beyond it.
Across the 47 updates since checkpoint 53, **246/246 admitted rollouts passed TiTO**,
with zero stale drops, no nonfinite loss/gradient/ratio values, and maximum
staleness 3. Thirty-four updates had nonzero gradients. One Mini-SWE-Agent
rollout deleted its working directory; its missing reward received zero advantage
under the verified native TRL scorer.

Checkpoint eval **78987** is running on two separate H100s with **TP1/DP2 and
100 concurrent slots**. Its eight initial rollouts passed TiTO before the full
1,000-cell pass@1 evaluation began. Training and eval scores are separate; the
checkpoint score is still pending. The watchdog now checks every ten minutes.
See [configuration and validation](train/TRAINING_SETUP.md), [logging details](train/LOGGING.md),
and the [private Trackio dashboard](https://huggingface.co/spaces/AdithyaSK/multi4-qwen35-2b-trackio).

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

The capture proxy keeps the engine's actual `prompt_token_ids`, sampled token IDs and logprobs for
every call. Turns can share a training sequence when the next prompt extends the previous tokens
exactly. A harness may rewrite or compact history; those turns start separate sequences so each
sampled output keeps its original causal context. Rebuilding or realigning captured history would
lose that guarantee.

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

It matters beyond cost. A rollout can produce one or several training sequences. When each turn
forks, those sequences repeat historical context and total processed tokens can grow quadratically
with turn count. A runaway rollout also delays scoring of its GRPO group.

## Historical status — 2026-09-15

The Qwen3.5-2B four-harness baseline completed on **2026-09-14**: **250 fixed test tasks × four
harnesses × pass@1 = 1,000 evaluations**, with **14.6% overall pass@1**. OpenCode scored 10.8%,
Claude Code 16.8%, Codex 16.4%, and mini-swe-agent 14.4%. All 1,000 selected captures passed the
exact-context TiTO audit. See [the baseline results and protocol](eval/BASELINE_PASS_AT_1.md).

The prepared training pool is **150 easy / 600 medium / 250 hard** tasks, starting with
32 easy tasks, then shuffling the rest. See [training configuration, diagnostic and
checkpoint evaluations](train/TRAINING_SETUP.md). The long run migrated to job **79083** on `hopper-prod`, resuming checkpoint **196** from job 78956; checkpoint 200 has been saved and its independent eval is job 79092.

Checkpoint 100 completed the same **1,000-cell pass@1 evaluation at 24.8% overall**, versus
14.6% for the base model. All selected captures passed TiTO and harness versions match; see the
[completed checkpoint comparison](eval/CHECKPOINT_100_PASS_AT_1.md). The local training recipe uses
atomic rollout admission and records optimizer consumption; upstream normalization work is tracked
in [TRL issue #7206](https://github.com/huggingface/trl/issues/7206).

Separate [Daytona blackbox and whitebox comparisons](eval/DAYTONA_COMPARISON.md) have passed provider,
tool and model-rollout smokes. Their full baselines are running; their optimizer smokes and long
comparison runs have not started. The local stack depends on core changes still in review upstream:

- **OpenEnv** — `TraceEntry` carrying `prompt_token_ids` and `loss_mask`, `CaptureServer` promoted
  into `openenv.core.harness.capture`, and a per-session model-call budget in the proxy.
- **TRL** — `async_grpo` consuming those ids instead of re-rendering, and `TurnRecord.output_mask`.
