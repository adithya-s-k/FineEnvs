"""The public OpenEnv training session contract over a remote native OpenCode server."""

from pathlib import Path

from openenv.core.harness import (
    ResourceSession,
    ToolResult,
    TrainingTrace,
    VerifyResult,
)
from openenv.core.mcp_client import MCPToolClient


class RemoteSession(ResourceSession):
    def __init__(self, server, row, model, sampling):
        self.client = MCPToolClient(server, message_timeout_s=1800).sync()
        self.row, self.model, self.sampling = row, model, sampling
        self.result = None

    def initial_messages(self):
        return [{"role": "user", "content": self.row["instruction"]}]

    def list_tools(self):
        return []

    def call_tool(self, name, arguments):
        return ToolResult(error="OpenCode owns its tool loop")

    def wait_for_completion(self, timeout_s=None):
        self.client.reset()
        folder = Path(self.row["folder"])
        self.result = self.client.call_tool(
            "run_rollout",
            split=folder.parent.parent.name,
            task_name=folder.name,
            model=self.model,
            sampling=self.sampling,
        )
        return 0

    def fetch_training_trace(self):
        if self.result is None:
            raise RuntimeError("Rollout has not completed")
        if "capture_error" in self.result:
            raise ValueError(self.result["capture_error"])
        return TrainingTrace.model_validate(self.result["training_trace"])

    def verify(self, transcript, final_state=None):
        return VerifyResult(
            env_reward=self.result["correctness"] if self.result else None, done=True
        )

    def close(self):
        self.client.close()


class RemoteTaskFactory:
    def __init__(self, args, tasks, *, sampling):
        self.args, self.sampling = args, sampling
        self.tasks = {row["instruction"]: row for row in tasks}

    def create(self, task, seed=None, episode_id=None):
        return RemoteSession(
            self.args.server,
            self.tasks[task[-1]["content"]],
            self.args.model,
            self.sampling,
        )
