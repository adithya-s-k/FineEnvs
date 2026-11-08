# OpenEnv RL

[Tinker](https://tinker-docs.thinkingmachines.ai/tinker/) is Thinking Machines' training API: your loop runs on
a CPU machine and Tinker runs LoRA training and sampling on its GPUs. This recipe connects it to
[OpenEnv](https://github.com/huggingface/OpenEnv), so the cookbook RL loop can train on any OpenEnv environment,
including Harbor tasks. It uses only OpenEnv's public client APIs (`openenv>=0.7`); OpenEnv needs no changes.

```bash
pip install "openenv>=0.7" tinker tinker-cookbook
export TINKER_API_KEY=...
```

The integration follows OpenEnv's two harness modes ([RFC 005](https://github.com/huggingface/OpenEnv/tree/main/rfcs),
RFC 006):

| | White-box | Black-box |
|---|---|---|
| Who runs the agent loop | the cookbook (Tinker samples, the renderer parses tool calls) | an agent harness in a sandbox (OpenCode, Claude Code, ...) |
| What OpenEnv provides | a `ResourceSession` per episode | a loop-owning session with a captured `TrainingTrace` |
| Entry points | `OpenEnvEnvGroupBuilder`, `OpenEnvRLDatasetBuilder` | `TinkerChatServer`, `run_black_box_group`, `train_step` |
| Works with | every `envs/<env>/harness.py` factory, and any MCP env via `MCPSessionFactory` | `HarborSessionFactory` and other `TrainableSession`s |
| Images in observations | yes (tool results may carry screenshots) | no (prompts must be token ids) |

Rewards are always the environment's own: read from `ResourceSession.verify()` or the tool results, never
recomputed by the integration. An episode the environment never graded raises `MissingRewardError` instead of
counting as 0.

## White-box: any environment, the cookbook's RL loop

```python
import asyncio
from tinker_cookbook.rl import train
from openenv_rl import ImageWindow, MCPSessionFactory, OpenEnvRLDatasetBuilder

# Any ResourceSessionFactory works; MCPSessionFactory covers MCP environments without a harness.py.
factory = MCPSessionFactory("https://my-env.hf.space", exclude_tools=("start_task",))

dataset = OpenEnvRLDatasetBuilder(
    session_factory=factory,
    tasks=[{"split": "train", "index": i} for i in range(256)],  # reset() kwargs per task
    model_name="Qwen/Qwen3.6-35B-A3B",
    renderer_name="qwen3_5_disable_thinking",
    batch_size=16,
    group_size=8,  # 16 tasks x 8 attempts per step (GRPO groups)
    max_turns=40,
    finalize_tool="submit",  # grade episodes that hit max_turns
    image_window=ImageWindow(max_images=8, keep_images=3),  # screenshot-heavy envs
)

config = train.Config(
    model_name="Qwen/Qwen3.6-35B-A3B",
    dataset_builder=dataset,
    learning_rate=1e-5,
    max_tokens=1024,
    log_path="/tmp/openenv-tinker",
)
asyncio.run(train.main(config))
```

`OpenEnvEnvGroupBuilder` builds each episode with the cookbook's `build_agent_tool_env`, so the cookbook's
parse-error policies, token budgets and rollout strategies all apply. Its options:

- `reward_source="verify"` (default) grades at the end with `session.verify()`, cross-checked against the last
  tool reward; `"sum"` pays each tool result's reward as a step reward, for environments that emit dense deltas.
- `finalize_tool` is called once when an episode ends without a tool reporting `done`.
- `image_window` drops old screenshots in chunks, so consecutive turns still extend each other between drops.
- Override `compute_group_rewards` in a subclass for rewards that compare a group's attempts (e.g. an LLM judge
  ranking them).

`make_renderer(model_name)` attaches the model's image processor and, by default, keeps earlier turns' thinking
in the rendered history (`extension=True`), so a multi-turn episode trains as one sequence.

## Black-box: Harbor tasks

The agent runs inside a sandbox and talks to an OpenAI-compatible endpoint through OpenEnv's capture proxy.
`TinkerChatServer` is that endpoint: it samples with Tinker and returns `prompt_token_ids`, `choices[].token_ids`
and their logprobs, which the capture proxy needs to record trainable turns.

```python
import tinker
from openenv_rl import TinkerChatServer, make_renderer, run_black_box_group, train_step
from harbor_env.harness import HarborSessionFactory

model = "Qwen/Qwen3.6-35B-A3B"
service = tinker.ServiceClient()
training = service.create_lora_training_client(base_model=model, rank=32)
server = TinkerChatServer(training.save_weights_and_get_sampling_client(), make_renderer(model))
llm_url = await server.start(host="0.0.0.0", port=8300)  # must be reachable from the sandbox

factory = HarborSessionFactory(harbor_server_url, llm_url=llm_url, model=model, harness="opencode", sandbox="e2b")
for step, batch in enumerate(task_batches):
    groups = [await run_black_box_group(factory, row, group_size=8) for row in batch]
    metrics = await train_step(training, groups, learning_rate=1e-5)
    server.set_sampling_client(training.save_weights_and_get_sampling_client())
```

`save_weights_and_get_sampling_client()` checkpoints are ephemeral; use `save_weights_for_sampler(name=...)` plus
`create_sampling_client(model_path=...)` for a checkpoint you keep.

Each `TrainingTrace` becomes Tinker datums with the trace's own loss masks; turns whose prompt extends the
previous turn merge into one sequence. Failed or ungraded rollouts are left out of their group, and groups whose
rewards are all equal are skipped.

## CLI

```bash
python -m openenv_rl.train env_url=http://127.0.0.1:8000 tasks_path=tasks/train.jsonl \
    model_name=Qwen/Qwen3.6-35B-A3B finalize_tool=submit max_images=8 keep_images=3
```

`tasks_path` is JSONL with one `reset()` kwargs dict per line; `initial_messages=module:function` overrides how the
opening messages are built from the reset observation.

## Evaluation

Sample from any checkpoint (`tinker://.../sampler_weights/...`) or base model with the same builders and the
cookbook's `do_group_rollout`, with `group_size=1`. `TinkerChatServer` also serves a checkpoint to any
OpenAI-compatible harness.

## Known limitation

Qwen3.5 renderers in tinker-cookbook 0.5.7 re-render an assistant tool call with one more newline before
`<tool_call>` than the model sampled. Each such turn starts a new training sequence. Training stays correct
(every datum is exactly what the model saw), it just uses more tokens.
