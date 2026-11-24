"""Train on DesignGym with the openenv_rl recipe: loads TINKER_API_KEY from the environment or a .env up the tree and
fills in the DesignGym settings. Any `openenv_rl.train` option can be passed as `key=value`, e.g.

    .venv/bin/python train_designgym.py tasks_path=tasks/rl-reference-tools.jsonl model_name=Qwen/Qwen3.6-35B-A3B
    .venv/bin/python train_designgym.py tasks_path=tasks/rl-reference-gui.jsonl model_name=Qwen/Qwen3.8-27B \
        renderer_name=qwen3_8_disable_thinking
"""

from __future__ import annotations

import asyncio
import os
import sys

import chz

import workspace_env
from openenv_rl.train import CLIConfig, cli_main

DEFAULTS = [
    "env_url=http://127.0.0.1:8007",
    "initial_messages=designgym_rl:initial_messages",
    "exclude_tools=start_task",
    "finalize_tool=submit",
    "model_name=Qwen/Qwen3.6-35B-A3B",
    "renderer_name=qwen3_5_disable_thinking",
    "max_images=8",
    "keep_images=3",
]

if __name__ == "__main__":
    if not workspace_env.load():
        raise SystemExit("TINKER_API_KEY not found in the environment or a .env up the tree")
    given = {a.split("=", 1)[0] for a in sys.argv[1:]}
    argv = [d for d in DEFAULTS if d.split("=", 1)[0] not in given] + sys.argv[1:]
    # the GUI opening message fetches the editor's first screenshot from the env server (designgym_rl._screen)
    os.environ.setdefault("DESIGNGYM_ENV_URL", next(a.split("=", 1)[1] for a in argv if a.startswith("env_url=")))
    asyncio.run(cli_main(chz.entrypoint(CLIConfig, argv=argv)))
