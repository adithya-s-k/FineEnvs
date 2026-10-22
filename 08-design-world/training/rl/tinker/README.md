# Tinker training for DesignGym

RL on the DesignGym OpenEnv environment (`envs/designgym`) with [Tinker](https://tinker-docs.thinkingmachines.ai/tinker/).
Main models: `Qwen/Qwen3.6-35B-A3B` (renderer `qwen3_5_disable_thinking`) and `Qwen/Qwen3.8-27B`
(`qwen3_8_disable_thinking`).

- `openenv_rl/` is a generic adapter (any OpenEnv environment, plus Harbor), laid out as a future
  `tinker_cookbook/recipes/openenv_rl/` recipe. It only uses OpenEnv's public client APIs, so OpenEnv is untouched.
  See `openenv_rl/README.md`.
- `designgym_rl.py` builds the opening messages (instructions, plus the target image in reference mode) and
  writes task lists per stage (`reference`, `description`) and surface (`tools`, `html`, `gui`) from an env split
  (`rl` by default).
- `train_designgym.py` fills in the DesignGym settings (server URL, opening messages, `submit` as the finaliser,
  Qwen3.6-35B-A3B, an 8/3 screenshot window); any `openenv_rl.train` option can be passed as `key=value` to override
  them.
- `eval_designgym.py` scores a base model or a `tinker://` checkpoint on the DesignWorldBench evalset with the
  same episodes as `envs/designgym/rollout.py`, sampled on Tinker.
- `hf_jobs/` runs the same pieces on Hugging Face Jobs: `launch.py rl` (env server + Tinker driver on a CPU job),
  `launch.py export` (Tinker LoRA -> PEFT adapter or merged weights on the Hub), `launch.py eval` (vLLM + DesignGym +
  `rollout.py` on a GPU node) and `watch_rl.py` (one status line with stop rules for a running RL job).

This project has its own environment so the environment package (and its Docker image) stays free of the training
stack (Tinker, the cookbook, torch). Both use the same OpenEnv release; the adapter is a client and talks to the
environment server over WebSocket.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python "openenv>=0.7.0" tinker tinker-cookbook pytest ruff
.venv/bin/python -m pytest openenv_rl                     # offline tests
DESIGN_CANVAS_REWARD=3 designgym serve --port 8007 &          # the env on :8007 (from envs/designgym)
.venv/bin/python designgym_rl.py --split rl --mode reference --surface tools --out tasks/rl-reference-tools.jsonl
.venv/bin/python train_designgym.py tasks_path=tasks/rl-reference-tools.jsonl
.venv/bin/python train_designgym.py tasks_path=tasks/rl-reference-tools.jsonl model_name=Qwen/Qwen3.8-27B \
    renderer_name=qwen3_8_disable_thinking
.venv/bin/python eval_designgym.py --model Qwen/Qwen3.6-35B-A3B --renderer qwen3_5_disable_thinking --out evals/q36-base
```

`train_designgym.py` and `eval_designgym.py` read `TINKER_API_KEY` from the environment, else from a `.env`
in the working directory or up the tree; `workspace_env.py` does the loading and never prints the key. The HF Jobs
launcher reads `HF_TOKEN` (and `TINKER_API_KEY`) the same way and passes them as Job secrets. Set
`DESIGNWORLD_RESULTS_REPO` (a dataset you own) and optionally `HF_JOBS_NAMESPACE` before launching jobs.

Training runs spend Tinker credit.
