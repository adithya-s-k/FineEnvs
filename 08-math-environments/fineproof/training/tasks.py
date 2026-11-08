"""Prepared FineProofEnvs tasks, and one Harbor session per rollout, all through our harness."""

from __future__ import annotations

import tomllib
from pathlib import Path

# Our Harbor agent, by import path: OpenEnv's `--harness module:Class` points a host-side agent at the capture
# proxy through its `api_base`/`api_key` kwargs. Options the server cannot pass come from FINEPROOF_* env vars.
HARNESS = "fineproof_agent.agent:FineProofAgent"
AGENT_TIMEOUT_SEC = 900
# Our harness makes 1 + rounds model calls (see harness/fineproof_agent/loop.py); the cap is a backstop.
AGENT_STEP_LIMIT = 6


def dataset_dir(data, split):
    """The folder `openenv harbor serve --dataset` is given for a split. The server names the split by it."""
    return str(Path(data).resolve() / "datasets" / split)


def load_tasks(dataset):
    """Task folders of one served dataset, in the order the Harbor server indexes them (sorted by name)."""
    root = Path(dataset) / "tasks"
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


def _session_class():
    """HarborSession, except that a rollout the server did not finish cleanly has no grade.

    When the agent times out (a slow engine, say), Harbor still runs the verifier, which scores the missing proof 0:
    a wrong answer as far as GRPO can tell. Such a rollout says nothing about the policy, so it is left ungraded and
    drops out of its group, like a judge failure.
    """
    from harbor_env.harness import HarborSession

    class FineProofSession(HarborSession):
        def verify(self, transcript, final_state=None):
            verdict = super().verify(transcript, final_state)
            result = self.result
            if result is not None and (not result.ok or result.exception_type == "AgentTimeoutError"):
                return type(verdict)(env_reward=None, done=True)
            return verdict

    return FineProofSession


def session_for(args, index, task, split, sampling):
    """`sampling=None` is an eval rollout on the server's own engine (a hosted provider is eval-only)."""
    from harbor_env import HarborEnv

    session = _session_class()(
        env=HarborEnv(args.server, message_timeout_s=1800),
        owns_env=True,
        split=split,
        task_index=index,
        instruction=task["instruction"],
        harness=HARNESS,
        sandbox=getattr(args, "sandbox", None) or "daytona",
        llm_url=getattr(args, "env_llm_url", None) or args.vllm_url or "",
        model=args.model,
        sampling=sampling,
        purpose="auto" if sampling is not None else "eval",
        reward_key="reward",
        agent_step_limit=AGENT_STEP_LIMIT,
        agent_timeout_sec=AGENT_TIMEOUT_SEC,
    )
    session.trial_root = Path(args.trials)
    return session


class TaskFactory:
    """What TRL's HarnessRolloutWorker calls: `factory(sampling=...)`, then `.create(prompt, ...)` per rollout."""

    def __init__(self, args, tasks, *, split="train", sampling):
        self.args, self.sampling = args, sampling
        self.split = dataset_dir(args.data, split)
        self.tasks = {task["instruction"]: (i, task) for i, task in enumerate(tasks)}

    def create(self, task, seed=None, episode_id=None):
        # Match by prompt, not by the worker's seed: seeds can change after a restart.
        index, row = self.tasks[task[-1]["content"]]
        return session_for(self.args, index, row, self.split, self.sampling)
