---
name: rl-env-from-description
description: Turns a user's plain-English description of an RL training environment into runnable code across the four target frameworks — OpenEnv, OpenReward (ORS), Verifiers, and NeMo Gym. Use whenever someone describes an environment they want to build ("I want to train an agent that does X", "make an env where the model has to Y"), asks to scaffold a new env, asks to port an existing env to one of these frameworks, or asks how to design tools/rewards/state for a new env. Use even when the user does not explicitly say "RL environment" — descriptions like "agent that browses the web", "tool-calling agent for SQL", or "game-playing agent" all qualify. Drives the full flow — clarifying interview, env-name selection, shared-domain extraction, per-framework implementation, and rollout-based smoke tests.
---

# RL Env From Description

Convert a plain-English description of an RL training environment into runnable code across **OpenEnv**, **OpenReward (ORS)**, **Verifiers**, and **NeMo Gym**. Two other framework variants (SkyRL Gym, GEM) are *secondary* and only relevant for text-action-with-tag-parsing envs — produce them only if the user asks.

## When to use

- A user describes an env in plain English (a goal, an action surface, or a reward shape) and wants code.
- A user asks to "build an env for X", "scaffold an RL env", "port this env to OpenEnv/ORS/Verifiers/NeMo Gym".
- A user already has a runnable env in one framework and wants the same env in others.
- A user asks "what's the right way to design my reward / state / tool surface for this task" — start with the **interview** below, then implement.

Do **not** use for: training runs (TRL/GRPO config), evaluation harness work, or general agent-design questions that don't end with new env code.

## Recommended layout (suggest, don't impose)

A clean shape that scales well — but the user gets to pick the actual paths:

```
<env_dir>/                          # whatever the user names it
├── <domain>.py                     # SHARED pure logic (e.g. game.py)
├── tasks.py                        # SHARED list of task dicts (optional)
├── openenv/                        # OpenEnv variant (HTTP, MCP)
├── ors/                            # ORS variant (HTTP, REST + SSE)
├── verifiers/                      # Verifiers variant (in-process)
└── nemo_gym/                       # NeMo Gym variant (HTTP, REST + cookies)
```

Inside each framework folder, the public contract is:
- `pyproject.toml` — framework-specific deps
- `__init__.py`
- One implementation file (`server.py` for ORS, `server/<env>_environment.py` for OpenEnv, `env.py` for Verifiers, `server.py` for NeMo Gym)
- `rollout.py` — runs an LLM against the env end-to-end
- `README.md` — one-page consumption guide

**Always ask the user where they want files written.** If they don't have a preference, propose the layout above. Don't force it.

## The four-step flow

### Step 1 — Interview the user (focused, not exhaustive)

Ask only the questions that determine architecture. The full bank lives in `references/interview.md`; the must-cover set is:

1. **What does the agent DO?** One sentence describing the goal and the loop.
2. **Action surface** — structured tool calls (most cases) or free text with tag parsing (rare; only when the model has no tool-calling support).
3. **State** — does anything persist across turns? Per-session sandbox? In-memory dict? Nothing?
4. **Reward** — when does it fire (per-step, on terminate, post-episode)? What's the success criterion?
5. **External backends** — sandbox (E2B), web service, none?
6. **Termination** — fixed turn cap, model-emits-`terminate`, or a derived condition.
7. **Where should the files live?** Project-relative path; never assume.

If the user has already given enough signal in their description (e.g. they cited an existing env they want to mirror), skip questions whose answers are obvious. Don't make people repeat themselves.

When in doubt about an architectural choice, propose a default with a one-line rationale and let the user veto.

### Step 2 — Pick the closest archetype

Match the user's description to one of these archetypes; tell them which archetype you're using and why:

| Archetype | Hallmarks | Typical reward shape |
|---|---|---|
| **Pure-Python game** | Deterministic, single tool, no external services, multi-turn | Terminal reward (1.0/0.0) or per-step from game state |
| **Stateful sandbox** | Real backend (E2B Code Interpreter, browser, DB), structured tool calls, state persists across calls | External grader (string match, unit tests, LLM judge) |
| **Vision / computer-use** | Screenshots + mouse/keyboard, 19-tool action surface modelled on Anthropic's `computer_20251124` | Terminal reward via `terminate(status)` tool |
| **Text-action with parsing** | Model emits free text containing tags; env parses (use only if the model has no tool-calling support) | Per-step from parsed action results |

### Step 3 — Implement in dependency order

The shared module first, then per-framework variants. Order doesn't matter between frameworks.

1. **`<env_dir>/<domain>.py`** + **`<env_dir>/tasks.py`** — the *only* file that contains domain logic. Frameworks just wrap it. Keep it pure-Python; no framework imports.
2. **OpenEnv** — read `references/openenv.md` (planner-level) or trigger `generate-openenv-env` skill (full workflow). Use `MCPEnvironment` + `@mcp.tool` + `create_app(...)` in `server/app.py`.
3. **ORS** — read `references/ors.md` or trigger `generate-ors-env`. Use `Environment` + `@tool` methods + `ToolOutput(blocks=[...], reward=..., finished=...)`. Per-tool-call reward is the framework's defining feature.
4. **Verifiers** — read `references/verifiers.md` or trigger `generate-verifiers-env`. Plain Python tool functions on a toolkit class; `vf.ToolEnv` + `vf.Rubric` for native consumption.
5. **NeMo Gym** — read `references/nemo_gym.md` or trigger `generate-nemo-gym-env`. `SimpleResourcesServer` with one `app.post("/<tool>")` per tool; cookie sessions; post-episode `/verify` reward.

### Step 4 — Validate end-to-end before declaring done

Each framework folder gets ONE smoke rollout against a small LLM (Qwen via HF Router by default, or OpenAI if `OPENAI_API_KEY` is set). The rollout must:

- Discover the tools the env exposes (don't hardcode names — except for NeMo Gym, which has no `list_tools()`).
- Drive a 3–5 turn loop and print every tool call + result.
- Fail loudly if a tool call errors. (`MAX_TURNS=3` for the smoke check.)

If the env needs an external backend (E2B, etc.), check for the relevant secret in `.env` and stop with a clear error if it's missing.

## What success looks like

A user typing "make me an env where the agent plays connect-four at `path/to/connect_four/`" should end with:
- `path/to/connect_four/game.py` (the shared engine), `tasks.py` (a few starting positions)
- `path/to/connect_four/openenv/`, `.../ors/`, `.../verifiers/`, `.../nemo_gym/` all runnable
- 4 green rollout smoke tests (NeMo Gym tested via deployed Space if local Ray init fails on shared nodes)
- Whatever README convention the project uses, updated

…in one continuous flow, with the user only answering 5–7 questions along the way.

## Reference docs

- `references/interview.md` — full question bank with example answers
- `references/openenv.md` — OpenEnv-specific implementation notes (planner-level; defers to `generate-openenv-env`)
- `references/ors.md` — ORS planner-level (defers to `generate-ors-env`)
- `references/verifiers.md` — Verifiers planner-level (defers to `generate-verifiers-env`)
- `references/nemo_gym.md` — NeMo Gym planner-level (defers to `generate-nemo-gym-env`)

When the user wants only one framework variant, trigger the framework-specific skill directly: `generate-openenv-env`, `generate-ors-env`, `generate-verifiers-env`, or `generate-nemo-gym-env`.

## Hard guardrails

- **Don't impose a folder layout.** Suggest the recommended one once; respect the user's choice if they want different paths.
- **Don't skip the shared domain module.** Cross-framework consistency is impossible without it. Every framework variant must wrap the same `<domain>.py` — never duplicate logic.
- **Don't run training.** This skill ends with rollouts. Training/eval is a separate concern.
- **Don't invent APIs.** When unsure about a framework's actual call shape, read its `references/architecture.md` (in the framework-specific skill) before writing code.
- **Coordinate spaces matter for vision envs.** Declare the convention (pixel vs normalized 0–1000) in the prompt. Qwen2.5-VL emits 0–1000 normalized; the rollout adapter must rescale.

## Official documentation

- **OpenEnv:** [openenv/OpenEnv](https://github.com/openenv/OpenEnv) · [docs](https://huggingface.co/docs/openenv)
- **OpenReward (ORS):** [openrewardstandard.io](https://openrewardstandard.io) · [docs.openreward.ai](https://docs.openreward.ai) · [openreward on PyPI](https://pypi.org/project/openreward/)
- **Verifiers:** [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) · [docs](https://docs.primeintellect.ai/verifiers/overview)
- **NeMo Gym:** [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) · [docs](https://docs.nvidia.com/nemo/gym/latest/)
