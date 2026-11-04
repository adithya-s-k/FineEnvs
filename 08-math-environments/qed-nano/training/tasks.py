"""Prepared QED-Nano tasks, and one Harbor session per rollout, all through our harness."""

from __future__ import annotations

import tomllib
from pathlib import Path

HARNESS = "qed-proof"
AGENT_TIMEOUT_SEC = 900
# Our harness makes 1 + rounds model calls (see harness/qed_proof_agent/loop.py); the cap is a backstop.
AGENT_STEP_LIMIT = 6


def load_tasks(data, split):
    """Task folders in the order the Harbor server indexes them (sorted by name)."""
    root = Path(data).resolve() / "datasets" / split / "tasks"
    tasks = []
    for folder in sorted(root.iterdir()):
        spec = tomllib.loads((folder / "task.toml").read_text())
        tasks.append({
            "name": folder.name,
            "folder": str(folder),
            "instruction": (folder / "instruction.md").read_text(),
            "base_reward": spec["metadata"].get("reward_mean"),
        })
    if not tasks or len({t["instruction"] for t in tasks}) != len(tasks):
        raise ValueError("Tasks must have unique instructions; run prepare.py first")
    return tasks


def session_for(args, index, task, split_path, sampling):
    from harbor_env import HarborEnv
    from harbor_env.harness import HarborSession

    session = HarborSession(
        env=HarborEnv(args.server, message_timeout_s=1800),
        owns_env=True,
        split=split_path,
        task_index=index,
        instruction=task["instruction"],
        harness=HARNESS,
        sandbox="daytona",
        llm_url=getattr(args, "env_llm_url", None) or args.vllm_url,
        model=args.model,
        sampling=sampling,
        reward_key="reward",
        agent_step_limit=AGENT_STEP_LIMIT,
        agent_timeout_sec=AGENT_TIMEOUT_SEC,
    )
    session.trial_root = Path(args.trials)
    return session


def split_paths(server):
    import httpx

    response = httpx.get(server.rstrip("/") + "/qed/splits", timeout=30)
    response.raise_for_status()
    return response.json()


class TaskFactory:
    """What TRL's HarnessRolloutWorker calls: `factory(sampling=...)`, then `.create(prompt, ...)` per rollout."""

    def __init__(self, args, tasks, *, split="train", sampling):
        self.args, self.sampling = args, sampling
        self.split = split_paths(args.server)[split]
        self.tasks = {task["instruction"]: (i, task) for i, task in enumerate(tasks)}

    def create(self, task, seed=None, episode_id=None):
        # Match by prompt, not by the worker's seed: seeds can change after a restart.
        index, row = self.tasks[task[-1]["content"]]
        return session_for(self.args, index, row, self.split, self.sampling)
