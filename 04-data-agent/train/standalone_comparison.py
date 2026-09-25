"""Native OpenCode sessions with the reference task schedule and binary reward.

Only the session boundary differs from the multi-harness trainer. GPU loss,
whole-rollout admission, backpressure and checkpoint callbacks are shared.
"""
from __future__ import annotations

from data_agent_env import DataAgentSessionFactory
from data_agent_env.harness import DataAgentSession, _instruction_of
from data_agent_env.task import instruction_id
from openenv.core.harness import VerifyResult


class ComparisonSession(DataAgentSession):
    @property
    def result(self):
        return self._result

    @property
    def _task_index(self):
        return self._index

    def verify(self, transcript, final_state=None):
        native = super().verify(transcript, final_state)
        correctness = self._result.correctness if self._result is not None else None
        # The comparison trains binary task success, like the reference and SETA.
        # Keep partial chat credit and efficiency bonuses in the raw artifact only.
        reward = None if correctness is None else float(correctness >= 1.0)
        return VerifyResult(env_reward=reward, done=True, metrics=native.metrics,
                            artifacts={**native.artifacts, "reward_policy": "binary_correctness"})


class ScheduledOpenCodeFactory(DataAgentSessionFactory):
    def __init__(self, server, *, harnesses, schedule, group_offset=0, split="train",
                 sandbox="daytona", llm_url, model, sampling, reward_key="",
                 api_key="", agent_timeout_sec=600, agent_step_limit=17,
                 indices=None, num_tasks=None):
        from harness_schedule import validate_schedule
        validate_schedule(schedule)
        if harnesses != ["opencode"] or schedule["harnesses"] != harnesses:
            raise ValueError("Standalone training requires an OpenCode-only schedule")
        if sampling != {"temperature": 0.8, "top_p": 1.0, "top_k": 0}:
            raise ValueError("Standalone comparison requires the pinned sampling policy")
        if group_offset < 0:
            raise ValueError("Negative schedule offset")
        super().__init__(server, split="train", llm_url=llm_url, model=model,
                         sandbox=sandbox, api_key=api_key, agent_step_limit=agent_step_limit,
                         agent_timeout_s=agent_timeout_sec, sampling=sampling)
        self.harnesses, self.schedule, self.group_offset = harnesses, schedule, group_offset
        self._rows, self._by_instruction = None, None

    def harness_for(self, seed):
        return "opencode"

    def _new_client(self):
        from data_agent_env import DataAgentEnv
        return DataAgentEnv(self._server, message_timeout_s=1800)

    def prompt_rows(self):
        if self._rows is None:
            client = self._new_client()
            try:
                tasks = client.get_task_range("train")
            finally:
                client.close()
            by_index = {task["index"]: task for task in tasks}
            rows, lookup = [], {}
            for expected in self.schedule["tasks"]:
                task = by_index[expected["task_index"]]
                if task["task_id"] != expected["name"]:
                    raise ValueError("Native Space task identity differs from the frozen schedule")
                key = instruction_id(task["instruction"])
                if key in lookup:
                    raise ValueError("Ambiguous training instruction")
                lookup[key] = expected["task_index"]
                rows.append({"prompt": [{"role": "user", "content": task["instruction"]}],
                             "task_name": expected["name"], "task_index": expected["task_index"]})
            self._rows, self._by_instruction = rows, lookup
        return self._rows

    def create(self, task, seed=None, episode_id=None):
        self.prompt_rows()
        instruction = _instruction_of(task)
        index = self._by_instruction.get(instruction_id(instruction))
        absolute = (seed or 0) + self.group_offset
        expected = self.schedule["groups"][absolute % len(self.schedule["groups"])]
        if index != expected["task_index"]:
            raise ValueError(f"Native schedule mismatch at group {absolute}: {index}")
        return ComparisonSession(self._new_client(), "train", index, instruction,
                                 **self._rollout_kwargs)
