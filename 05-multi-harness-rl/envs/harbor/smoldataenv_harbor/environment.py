"""Assign SmolDataEnvs tasks to Harbor harnesses and count verified native actions."""

import json
from pathlib import Path

from harbor_env import HarborEnv
from harbor_env.harness import HarborSession

HARNESSES = ("opencode", "claude-code", "codex", "mini-swe-agent")


def count_tools(path):
    try:
        trajectory = json.loads(path.read_text())
        if not trajectory["schema_version"].startswith("ATIF-"):
            raise ValueError("Unknown trajectory format")
        ids = [
            call["tool_call_id"]
            for step in trajectory["steps"]
            if step.get("source") == "agent"
            for call in step.get("tool_calls", []) or []
        ]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Missing or repeated tool action IDs")
        calls = len(ids)
    except (OSError, ValueError, KeyError, TypeError):
        calls = None
    return calls


class TaskSession(HarborSession):
    def fetch_training_trace(self):
        trace = super().fetch_training_trace()
        # ATIF counts actual tool actions, not model requests or training rows.
        path = self.trial_root / self.result.trial_name / "agent/trajectory.json"
        if getattr(self, "server", None):
            import httpx

            response = httpx.get(
                self.server.rstrip("/")
                + f"/smoldataenv/trials/{self.result.trial_name}/tool-count",
                timeout=30,
            )
            response.raise_for_status()
            calls = response.json()["native_tool_calls"]
        else:
            calls = count_tools(path)
        for turn in trace.turns:
            turn.metadata["native_tool_calls"] = calls
        return trace


class TaskFactory:
    def __init__(self, args, tasks, *, sampling):
        import httpx

        self.args, self.sampling = args, sampling
        response = httpx.get(
            args.server.rstrip("/") + "/smoldataenv/splits", timeout=30
        )
        response.raise_for_status()
        self.split = response.json()["train"]
        self.tasks = {task["instruction"]: (i, task) for i, task in enumerate(tasks)}

    def create(self, task, seed=None, episode_id=None):
        # Match by prompt, not by the worker seed: seeds can change after a restart.
        index, row = self.tasks[task[-1]["content"]]
        session = TaskSession(
            env=HarborEnv(self.args.server, message_timeout_s=1800),
            owns_env=True,
            split=self.split,
            task_index=index,
            instruction=row["instruction"],
            harness=HARNESSES[index % len(HARNESSES)],
            sandbox="daytona",
            llm_url=getattr(self.args, "env_llm_url", None) or self.args.vllm_url,
            model=self.args.model,
            sampling=self.sampling,
            reward_key="correctness,reward",
            agent_step_limit=17,
            agent_timeout_sec=600,
        )
        session.trial_root = Path(self.args.trials)
        session.server = self.args.server
        return session
