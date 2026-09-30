# SmolDataEnv: whitebox and multi-harness RL

Train the same data-analysis policy through three agent interfaces. Use **LFM2.5-2.6B** or **Qwen3.5-2B**, with thinking disabled, the same task split, and correctness plus a small tool-efficiency bonus.

This is a focused successor to the archived `04-data-agent` experiment. Start with [REPRODUCE.md](REPRODUCE.md) for HF Jobs and Slurm commands. The new recipe is under qualification; historical runs do not establish that this new non-thinking configuration is trained or validated.

| Mode | Who drives the agent? | Training | Evaluation |
|---|---|---|---|
| `whitebox` | TRL calls native bash/SETA methods | Synchronous GRPO | 250 tasks with the same bash/SETA tools |
| `opencode` | Native OpenCode, without Harbor execution | AsyncGRPO | 250 tasks through each of four Harbor harnesses |
| `multi-harness` | Harbor runs OpenCode, Claude Code, Codex or Mini-SWE-Agent | AsyncGRPO | The same 250 tasks through all four harnesses |

The OpenEnv server runs **inside each training or evaluation job**. Daytona runs the isolated task containers. No environment Space is needed. Training and evaluation use separate GPU allocations, inference engines and environment processes; they still share your sandbox quota and storage bandwidth.

## Shared experiment

| Setting | Value |
|---|---|
| Training data | Fixed 1,000 tasks: 400 medium, 600 hard; no test task or notebook overlap |
| Schedule | Two passes; one harness per task group; rotate the harness on the second pass |
| Rollouts per group | 8, all using the same task and harness |
| Learning rate / sampling | `3e-6`; temperature `0.8`, top-p `1`, top-k disabled |
| Async limits | 32 inflight; 16 outstanding rollouts including queued work; staleness 4 |
| Optimizer | Full fine-tuning, paged AdamW 8-bit, BF16, gradient checkpointing |
| Async batching | Batch 4 × accumulation 4; whole rollouts admitted together; 40,960-token packing target |
| Synchronous batching | Microbatch 1 × accumulation 16 to limit logits memory |
| Episode limits | 17 model/tool iterations, 600 seconds; 4,096 output tokens per model call |
| Trainer limits | 16,384 completion tokens; 131,072 context; 1,000 optimizer-step ceiling |
| Save / evaluate | Save every 50 updates; evaluate every 100 and the final checkpoint |
| Evaluation | Pass@1; TP1/DP2; default concurrency 35, configurable |

Task groups and optimizer updates are different units. Async packing, discarded stale rollouts, and groups without reward contrast affect how many groups produce an update. Training stops at schedule exhaustion or the step ceiling, whichever comes first. Two scheduled passes do not guarantee two completed passes; checkpoints record the groups actually used.

The reward is `correctness × (1 + 0.1 × 15 / (15 + tool_calls))`. Incorrect answers earn zero. A zero or unverified tool count receives no bonus. Infrastructure failures remain ungraded. Native action records provide the counts; the sandbox grader returns correctness only. Pass@1 reports binary correctness separately from shaping and always includes coverage.

Qwen uses `enable_thinking=False`. The pinned LFM template ignores that flag, so this recipe explicitly closes its generation-time thinking block. Serving omits reasoning parsing: a streaming reasoning parser can otherwise hide tool calls from the harness. Startup checks that both streaming and ordinary replies are visible. Earlier LFM RL results used thinking and are not a baseline for this setting.

## Code comparison

The learning boundary is small:

```python
# Whitebox: the trainer generates, calls tools and masks tool results.
GRPOTrainer(environment_factory=white_box_bash_env(...), ...)

# Native OpenCode: the native environment runs OpenCode and captures its model calls.
DataAgentSession(...)

# Multi-harness: Harbor owns the agent loop and returns captured model calls.
HarborSession(harness=group["harness"], ...)
```

Both blackbox modes use the same finite AsyncGRPO worker and whole-rollout admission. Engine token IDs, sampled logprobs and loss masks are preserved, including when a harness rewrites its history. Those rewrites can create several training rows from one rollout, so row-averaged training reward is not the same as rollout-level correctness. LFM's packed convolution states are isolated between sequences.

| File | Purpose |
|---|---|
| [configs/default.json](configs/default.json) | Shared hyperparameters and model/harness pins |
| [prepare.py](prepare.py) | Fixed task selection and corrected Harbor grader checks |
| [train/whitebox.py](train/whitebox.py) | Synchronous tool-based training |
| [train/blackbox.py](train/blackbox.py), [train/adapters.py](train/adapters.py) | Async training and native/Harbor session boundary |
| [eval/evaluate.py](eval/evaluate.py) | Resumable pass@1 evaluation and usage metrics |
| [eval/watch.py](eval/watch.py) | Checkpoint verification and separate evaluation jobs |
| [runtime/launch.py](runtime/launch.py) | HF Jobs and Slurm submission; dry-run by default |

`runtime/bootstrap.py` fetches 45 existing environment/admission files and pinned TRL/OpenEnv checkouts into ignored `.runtime/`. Their source hashes are in [runtime-lock.json](configs/runtime-lock.json). Small native-environment adaptations are explicit in [runtime/patches.py](runtime/patches.py). The recipe does not require an old local experiment checkout. The pinned TRL integration is experimental; dependency upgrades require another smoke.

## Logs and previous results

Each run keeps its resolved configuration, task schedule, scalar `metrics.jsonl`, local Trackio data, rollout reward/count evidence, service logs and full optimizer checkpoints. Each eval keeps a record per task/harness, coverage, missing pairs, pass@1, tool calls and generated/prompt tokens. Set `trackio_space_id` in the config to mirror to an online dashboard.

[SmolDataEnv RL dashboard](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio) contains the historical experiments. Their full record remains on [the archived branch](https://github.com/adithya-s-k/FineEnvs/tree/archive/data-agent-experiments-20260930/04-data-agent). This recipe has no new published scores yet.

The September 30 Harbor grader update is pinned here: null tool-efficiency fields must not discard a valid correctness reward. [SmolDataEnvs prompt updates](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs/discussions/2) also change some single-program baselines. Do not combine those scores with an older protocol without identifying the revisions.

Historical impact depends on each run's frozen files. The corrected Hub tasks match this recipe's instruction hashes; 302 selected training tasks use a JSON reward wrapper where the old local copy used `reward.txt`. The recipe checks the corrected grader before serving either split. See [validation notes](VALIDATION.md) for smoke coverage and remaining checks.
