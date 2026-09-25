# whitebox-bash

A **white-box** agent environment: bash plus the SETA tool surface over one sandbox, driven by
TRL's **synchronous** `GRPOTrainer`.

## Which one should I use?

The two sibling environments here are black box — an agent owns its own loop inside a sandbox, the
trainer never drives a turn, and everything trainable has to be recovered by *observing* model calls
through a capture proxy. This one inverts that:

| | blackbox-opencode / blackbox-harbor | **whitebox-bash** |
| --- | --- | --- |
| Who owns the loop | the agent (opencode / a Harbor harness) | **TRL** |
| Trainer | `AsyncGRPOTrainer` | **`GRPOTrainer`** (sync) |
| Tools | inside the sandbox, not on the wire | **MCP calls, individually observable** |
| Token ids | recovered by a capture proxy | TRL already has them — it generated them |
| Loss mask | reconstructed per turn | TRL masks tool results itself |
| Needs a tunnel | yes (sandbox must reach capture) | **no** |

The black-box pair is what you want when the thing being trained *is* a real agent you did not write.
This one is what you want when you want to see and shape every step.

## Install

The trainer needs only the client half:

```bash
pip install -e .                 # client: the class TRL introspects
pip install -e '.[server]'       # server: sandbox stack, only where you host it
```

## Train

```python
from trl import GRPOTrainer
from whitebox_bash import white_box_bash_env

trainer = GRPOTrainer(
    model="Qwen/Qwen3.5-2B",
    args=config,
    train_dataset=dataset,
    environment_factory=white_box_bash_env(
        "https://your-space.hf.space",
        toolsets="bash,seta",   # the default; "bash" alone for a minimal terminal agent
        step_limit=20,
    ),
)
```

TRL calls the factory once per rollout, introspects the instance, and puts its public methods in the
model's tool schema. `reset()` returns the task text; `get_reward()` scores the episode.

## Serve

```bash
E2B_API_KEY=... uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## The toolsets

| set | tools | notes |
| --- | --- | --- |
| `bash` | `bash` | always included |
| `seta` | `read`, `write`, `edit`, `grep`, `glob`, `ls` | SETA's surface, same names |
| — | `submit_solution` | always present; ends the episode |

Default is `("bash", "seta")` — full SETA parity.

Both run on **one sandbox** and share its filesystem. That is the point: a file written by `write`
must be visible to `bash` in the very next call. Splitting the toolsets across servers would mean
replicating state between sandboxes, and every divergence would surface as an agent that wrote a file
and then could not find it — which reads as a model failure and is not one.

**There is deliberately no second execution model.** An earlier revision also offered a persistent
Jupyter kernel; it was dropped because two tools could then do the same job under different state
semantics (kernel names persist, shell state does not), which is easy for a small model to conflate
and is one more unvalidated variable in an environment that has not trained yet.

`submit_solution` is always present rather than living in `seta`, so a `bash`-only agent still has a
way to finish; it takes SETA's name so a task written against SETA reads unchanged here.

## Why toolset selection varies the class, not a flag

TRL turns **every public method** of the instance into a tool:

```python
for member_name, member in inspect.getmembers(instance, predicate=inspect.ismethod):
    if member_name == "reset":        has_reset = True
    elif member_name == "get_reward": has_reward = True
    elif not member_name.startswith("_"): methods.append(member)
```

So a runtime flag could not shrink the surface — the model would still be offered every tool and
would call ones the server does not serve. `white_box_bash_env()` therefore composes a class from
mixins, one per toolset. Two consequences worth remembering when editing:

* **every helper must be `_`-prefixed**, or it silently becomes a tool;
* **type hints and docstrings are the tool schema**, not documentation. Write them for the model.

`tests/test_surface_matches.py` asserts all three descriptions of the surface agree — the registry in
`tools.py`, the client's methods, and the server's registered tools. If they drift the model calls a
tool nobody implements, gets an error, and the run reads as a policy that cannot use tools. Nothing
else would report it.

## Tasks

The Task API is implemented structurally: `list_splits`, `num_tasks`, `list_tasks`, `get_task`,
`get_task_range`. Difficulty is part of the **split name** (`train:medium`), never a filter — a
filter would shift every index after it, and the index is the task's identity everywhere downstream.

A small built-in suite ships so the environment is testable with no network. Point
`WHITE_BOX_BASH_DATASET` at a Hub dataset to replace it.

## Reward

`grade` runs server-side, where the sandbox and the gold answer live. Three rules carried over from
the black-box work, each learned expensively:

* an **ungraded** rollout is `None`, never `0.0` — a dead sandbox and a wrong answer are different
  events, and collapsing them teaches the model the dead sandbox was its fault;
* the **efficiency bonus is gated on a solve and is never a penalty** — ungated, "make no tool calls"
  becomes the best move for a policy that cannot solve the task;
* a submission that is really a **shell command** earns nothing — 42% of partial credit once went to
  strings like `echo -n "2.14" > answer.txt`.

## Not yet done

* **SETA's task suite.** Its *tool surface* and terminator name are implemented here, so a task
  written against SETA reads the same. Its 1,376-task suite is a different matter: it grades with
  weighted pytest inside its own Ubuntu 24.04 image and speaks ORS. That lands later as a split (if
  its image runs in our sandbox) or as a sibling env — a data and protocol decision, not a tools one.
* **A trained run.** The surfaces are verified and the grader is unit-tested; no GRPO run has used
  this yet.
