# Tooling

Apps for exploring and evaluating RL environments on the Hub. Each one is a Docker Space, deployed from its folder.

| Tool | What it does | Space |
|---|---|---|
| **[HF RL Explorer](./hf-rl-explorer/)** | Every RL environment on the Hub in one place: Harbor datasets, OpenEnv Spaces, NeMo Gym, Verifiers, verl and SkyRL datasets, and the MiMo release. What each task asks, how it is graded and what it runs in; rollouts with an agent and a model of your choice, graded and compared; MCP for coding agents. | [FineEnvs/RL-Explorer](https://huggingface.co/spaces/FineEnvs/RL-Explorer) |
| **[MiMo RL Environment Explorer](./mimo-rl-explorer/)** | The 7,780 MiMo-V2.6-RL environments by domain, rollouts on MiMo's own harness, compared and shared; and `mimo_harbor`, the adapter that publishes them as Harbor datasets. | [FineEnvs/MiMo-RL-Envs-Explorer](https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer) |

## HF RL Explorer

```bash
cd tooling/hf-rl-explorer
uv sync
uv run uvicorn app.main:app --port 8060 --reload          # the explorer, as you (your HF token)
uv run uvicorn app.admin_app:app --port 8061 --reload     # its admin
uv run python -m pytest tests -q                          # security, contract, frameworks, data layer (no network)
uv run python scripts/deploy_spaces.py --apply --listing-from-store   # both Spaces
```

Two Spaces from one image: the public explorer and a private admin (FineEnvs members only). They share one
bucket, `FineEnvs/rl-explorer-data`; an hourly Job (`scripts/schedule_indexer.py`) indexes the Hub into an immutable
SQLite snapshot there, which the Spaces hot-swap. [DESIGN.md](./hf-rl-explorer/DESIGN.md) has the architecture,
the data layer, security and the production checklist; [CUSTOM_ENVS.md](./hf-rl-explorer/CUSTOM_ENVS.md) how to
add a format.

## MiMo RL Environment Explorer

```bash
cd tooling/mimo-rl-explorer
uv sync
uv run uvicorn app.main:app --port 8000                   # http://localhost:8000
```

[CONTRIBUTING.md](./mimo-rl-explorer/CONTRIBUTING.md) covers rollouts, validation and the Space deploy;
[mimo_harbor/](./mimo-rl-explorer/mimo_harbor/) the Harbor conversion and its parity results.

The HF RL Explorer carries the MiMo explorer's backend in `hf-rl-explorer/app/mimo` (the same code, adapted to
its store), so a MiMo task and its Harbor conversion are one task page there.
