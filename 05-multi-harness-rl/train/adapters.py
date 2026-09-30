"""One task identity and reward boundary for both blackbox implementations."""
from __future__ import annotations

import json
from pathlib import Path

from recipe import reward, write_json


def atif_count(path):
    data = json.loads(Path(path).read_text())
    if not str(data.get("schema_version", "")).startswith("ATIF-"):
        raise ValueError("Unknown native trajectory format")
    calls = [c for s in data["steps"] if s.get("source") == "agent" for c in s.get("tool_calls", []) or []]
    ids = [c.get("tool_call_id") for c in calls]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing or repeated native action IDs")
    return len(ids)


def opencode_count(text):
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or not any(r.get("type") == "step_finish" for r in rows):
        raise ValueError("Incomplete native OpenCode event stream")
    parts = [r["part"] for r in rows if r.get("type") == "tool_use"]
    ids = [p["callID"] for p in parts]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ValueError("Ambiguous native tool actions")
    if any(p.get("state", {}).get("status") not in {"completed", "error"} for p in parts):
        raise ValueError("Unfinished tool action")
    return len(ids)


class Session:
    def __init__(self, inner, cfg, evidence, trial_root):
        self.inner, self.cfg = inner, cfg
        self.evidence, self.trial_root = evidence, Path(trial_root)

    def __getattr__(self, name):
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    @property
    def result(self):
        return getattr(self.inner, "result", getattr(self.inner, "_result", None))

    def fetch_proxy_trace(self):
        self.trace = self.inner.fetch_proxy_trace()
        return self.trace

    def verify(self, transcript, final_state=None):
        from openenv.core.harness import VerifyResult

        original = self.inner.verify(transcript, final_state)
        result = self.result
        correctness = original.env_reward
        calls, error = None, None
        if result is not None:
            if self.cfg["mode"] == "opencode":
                value = result.correctness
                correctness = None if value is None else float(value >= 1)
                calls = result.metadata.get("verified_native_actions")
            else:
                try:
                    calls = atif_count(self.trial_root / result.trial_name / "agent/trajectory.json")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    error = type(exc).__name__
        shaped = reward(correctness, calls, verified=calls is not None,
                        weight=self.cfg["efficiency_weight"], budget=self.cfg["tool_budget"])
        record = {**self.evidence, "correctness": correctness, "tool_calls": calls,
                  "tool_count_verified": calls is not None, "tool_count_error": error,
                  "reward": shaped, "bonus": None if shaped is None else shaped - correctness}
        if getattr(self, "trace", None):
            self.trace[0]["recipe_reward"] = record
        write_json(Path(self.cfg["output"]) / "rollouts" / f'{self.evidence["episode_id"]}.json', record)
        return VerifyResult(env_reward=correctness, done=original.done, metrics={**original.metrics, **record},
                            artifacts=original.artifacts)


def rollout_reward(outcome):
    if not outcome.trace:
        return None
    return outcome.trace[0]["recipe_reward"]["reward"]


class Factory:
    def __init__(self, cfg, data, server, vllm, groups, trial_root, split="train"):
        self.cfg, self.data, self.server, self.vllm = cfg, str(data), server, vllm
        self.groups, self.trial_root, self.split = groups, str(trial_root), split
        manifest = json.loads((Path(data) / f"{split}_manifest.json").read_text())
        self.tasks = sorted(manifest["tasks"], key=lambda row: row["name"])
        self.instructions = [(Path(data) / "datasets" / split / "tasks" / r["name"] / "instruction.md").read_text()
                             for r in self.tasks]

    def rows(self):
        return [{"prompt": [{"role": "user", "content": self.instructions[g["task_index"]]}],
                 "task_name": g["task_name"], "harness": g["harness"]} for g in self.groups]

    def create(self, task, seed=None, episode_id=None):
        from harbor_env.harness import HarborSession, _instruction_of

        group_id = 0 if seed is None else seed
        group = self.groups[group_id]
        index = group["task_index"]
        instruction = self.instructions[index]
        if _instruction_of(task).strip() != instruction.strip():
            raise ValueError("Prompt and frozen task schedule disagree")
        sampling = {k: self.cfg[k] for k in ("temperature", "top_p", "top_k")}
        if self.cfg["mode"] == "opencode":
            from data_agent_env import DataAgentEnv
            from data_agent_env.harness import DataAgentSession

            inner = DataAgentSession(DataAgentEnv(self.server, message_timeout_s=1800), self.split,
                index, instruction, llm_url=self.vllm, model=self.cfg["profile"]["id"],
                sandbox=self.cfg["sandbox"], sampling=sampling,
                agent_step_limit=self.cfg["agent_step_limit"], agent_timeout_s=self.cfg["agent_timeout_sec"])
        else:
            from harbor_env import HarborEnv

            inner = HarborSession(env=HarborEnv(self.server, message_timeout_s=1800), owns_env=True,
                split=str(Path(self.data) / "datasets" / self.split), task_index=index, instruction=instruction,
                harness=group["harness"], sandbox=self.cfg["sandbox"], llm_url=self.vllm,
                model=self.cfg["profile"]["id"], sampling=sampling, reward_key="correctness,reward",
                agent_step_limit=self.cfg["agent_step_limit"], agent_timeout_sec=self.cfg["agent_timeout_sec"])
        evidence = {"group_id": group_id, "task_name": group["task_name"], "harness": group["harness"],
                    "episode_id": episode_id}
        return Session(inner, self.cfg, evidence, self.trial_root)
