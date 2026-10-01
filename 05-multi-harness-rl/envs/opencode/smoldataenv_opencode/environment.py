"""Native OpenCode sessions on the fixed SmolDataEnvs tasks."""

import json
import os

from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.task import OpenCodeTask
from openenv.core.harness import VerifyResult

from .daytona import DaytonaSandboxBackend
from .tasks import grade_answer, stage_task


class TaskSession:
    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        return getattr(self.session, name)

    def fetch_training_trace(self):
        trace = self.session.fetch_training_trace()
        try:
            events = [
                json.loads(line)
                for line in self.session.fetch_trace().splitlines()
                if line.strip()
            ]
            parts = [
                event["part"] for event in events if event.get("type") == "tool_use"
            ]
            ids = [part["callID"] for part in parts]
            if not any(event.get("type") == "step_finish" for event in events):
                raise ValueError("Incomplete OpenCode event stream")
            if any(not value for value in ids) or len(set(ids)) != len(ids):
                raise ValueError("Missing or repeated tool action IDs")
            if any(
                part.get("state", {}).get("status") not in {"completed", "error"}
                for part in parts
            ):
                raise ValueError("Unfinished tool action")
            calls = len(ids)
        except (OSError, ValueError, KeyError, TypeError):
            calls = None
        for turn in trace.turns:
            turn.metadata["native_tool_calls"] = calls
        return trace


def verify(sandbox, task):
    return VerifyResult(
        env_reward=grade_answer(sandbox, task.metadata["folder"]), done=True
    )


class TaskFactory:
    def __init__(self, args, tasks, *, sampling):
        self.tasks = {task["instruction"]: task for task in tasks}
        config = OpenCodeConfig(
            base_url=os.environ["SANDBOX_VLLM_URL"].rstrip("/") + "/v1",
            api_key=os.environ["SANDBOX_VLLM_KEY"],
            model=args.model,
            opencode_version="1.18.31",
            sandbox_home="/root",
            proxy_disable_thinking=True,
            proxy_max_tokens_cap=4096,
            agent_timeout_s=600,
            run_format="json",
            extra_setup_shell="mkdir -p /workdir && rmdir /root/workdir && ln -s /workdir /root/workdir",
            disabled_tools=["webfetch", "question", "task"],
            extra_opencode_json={"permission": {"*": "allow"}},
        )
        self.factory = OpenCodeSessionFactory(
            config=config,
            sampling=sampling,
            verifier=verify,
            mode="transparent_proxy",
            sandbox_backend=DaytonaSandboxBackend(
                image="docker.io/savatar101/env-data-agent-train:base"
            ),
        )

    def create(self, task, seed=None, episode_id=None):
        row = self.tasks[task[-1]["content"]]
        task = OpenCodeTask(
            instruction=row["instruction"], metadata={"folder": row["folder"]}
        )
        session = self.factory.create(
            task, seed=seed, episode_id=episode_id, start_agent=False
        )
        try:
            stage_task(session.sandbox, row["folder"])
            session.start_agent()
            return TaskSession(session)
        except BaseException:
            session.close()
            raise
