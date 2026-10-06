"""The OpenEnv environment: one berth-planning episode per session, three MCP tools.

    get_situation()      the quay, notices, ships already alongside, closures and the ships to berth
    check_plan(plan)     rule violations and cost of the agent's own plan (budgeted; never a score or a reference)
    submit_plan(plan)    the final answer: ends the episode; the rubric grades it once

`reset(task_id=...)` or `reset(split=, index=)` (or `seed=` for a random task) picks a task. Tasks are served through
OpenEnv's Task API (`list_splits`, `num_tasks`, `get_task`, `get_task_range`). The reward is `BerthPlanRubric`
applied to the `submit_plan` action; every other step has reward 0.0. An episode that runs out of tool calls ends
with reward 0.0.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from collections import OrderedDict
from typing import Annotated, Any

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

from berth_core import PROMPT_VERSION, Task, TaskPack, evaluate, load_pack, parse_plan, plan_to_list, rules, situation
from berth_core.model import PlanError

try:
    from .rubric import SUBMIT_TOOL, BerthPlanRubric
except ImportError:  # flat layout
    from rubric import SUBMIT_TOOL, BerthPlanRubric

MAX_CHECKS = int(os.environ.get("BERTH_MAX_CHECKS", "10"))
MAX_TOOL_CALLS = int(os.environ.get("BERTH_MAX_TOOL_CALLS", "24"))


class PlanEntry(BaseModel):
    ship: int = Field(description="ship id from the table")
    berth_hour: int = Field(description="hour the ship berths (>= its arrival hour)")
    section: int = Field(description="first (lowest-numbered) section the ship occupies")
    cranes: int | None = Field(default=None, description="quay cranes working the ship (tasks with crane rules; "
                                                         "defaults to the ship's planned cranes)")


PlanArg = Annotated[list[PlanEntry] | str, Field(
    description='one entry per ship: [{"ship": 0, "berth_hour": 36, "section": 22}, ...]')]


class BerthState(State):
    task_id: str | None = None
    split: str | None = None
    checks_used: int = 0
    tool_calls: int = 0
    done: bool = False
    reward: float | None = None
    grade: dict | None = None


class Episode:
    """Everything one episode did, kept for the viewer's live page (bounded store, newest first)."""

    def __init__(self, task: Task, episode_id: str):
        self.task = task
        self.episode_id = episode_id
        self.started = time.time()
        self.steps: list[dict] = []
        self.done = False
        self.grade: dict | None = None
        self.end_reason: str | None = None

    def summary(self) -> dict:
        return {"episode_id": self.episode_id, "task_id": self.task.task_id, "started": self.started,
                "done": self.done, "end_reason": self.end_reason, "steps": self.steps, "grade": self.grade}


class _Store:
    def __init__(self, cap: int = 200):
        self.cap = cap
        self._d: OrderedDict[str, Episode] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, ep: Episode):
        with self._lock:
            self._d[ep.episode_id] = ep
            while len(self._d) > self.cap:
                self._d.popitem(last=False)

    def get(self, eid: str) -> Episode | None:
        return self._d.get(eid)

    def recent(self, n: int = 50) -> list[Episode]:
        return list(reversed(self._d.values()))[:n]


STORE = _Store()


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _plan_result(task: Task, plan) -> dict:
    r = evaluate(task, plan)
    return {
        "feasible": r.feasible,
        "violations": r.violations,
        "cost": r.cost, "delay_cost": r.delay_cost, "moves": r.moves,
        "ships": [{"ship": s.ship, "departure": s.departure, "delay_h": s.delay_h, "moved": s.moved, "cost": s.cost}
                  | ({"cranes": s.cranes, "hours_alongside": s.departure - s.berth_hour} if s.cranes is not None else {})
                  for s in r.ships if s.berth_hour is not None],
    }


class BerthPlanningEnvironment(MCPEnvironment):
    """Re-plan a real week of container berthings at a Barcelona quay after a disruption."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, pack: TaskPack | None = None):
        self.pack = pack or load_pack()
        self._episode: Episode | None = None
        self._state = BerthState()
        mcp = FastMCP("berth_planning")

        @mcp.tool
        def get_situation() -> str:
            """The quay, notices, ships already alongside, closed sections and the table of ships to berth."""
            ep = self._require()
            return situation(ep.task)

        @mcp.tool
        def check_plan(plan: PlanArg) -> str:
            """Check your plan: rule violations per ship, departure and delay per ship, and the plan's cost.
            Uses one of your checks. It does not grade the plan."""
            ep = self._require(open_only=True)
            if self._state.checks_used >= MAX_CHECKS:
                return _json({"error": f"no checks left (you had {MAX_CHECKS}); submit_plan when ready"})
            self._state.checks_used += 1
            try:
                parsed, problems = parse_plan(ep.task, _raw(plan))
            except PlanError as e:
                out = {"error": str(e), "checks_left": MAX_CHECKS - self._state.checks_used}
                ep.steps.append({"turn": self._state.tool_calls, "tool": "check_plan", "plan": None, "result": out})
                return _json(out)
            out = _plan_result(ep.task, parsed)
            if problems:
                out["entry_problems"] = problems
                out["feasible"] = False
            out["checks_left"] = MAX_CHECKS - self._state.checks_used
            ep.steps.append({"turn": self._state.tool_calls, "tool": "check_plan", "plan": plan_to_list(parsed),
                             "result": out})
            return _json(out)

        @mcp.tool
        def submit_plan(plan: PlanArg) -> str:
            """Submit your final berth plan. Ends the episode; the plan is graded once."""
            ep = self._require(open_only=True)
            try:
                parsed, problems = parse_plan(ep.task, _raw(plan))
            except PlanError as e:
                parsed, problems = {}, [str(e)]
            # The rubric grades the same arguments again in step(); this result is what the agent reads.
            g = self.rubric.graded(CallToolAction(tool_name=SUBMIT_TOOL, arguments={"plan": _raw(plan)}))
            ep.done = True
            ep.end_reason = "submitted"
            ep.grade = g.as_dict() if g else None
            ep.steps.append({"turn": self._state.tool_calls, "tool": "submit_plan", "plan": plan_to_list(parsed),
                             "result": {"grade": ep.grade}})
            out = {"submitted": True, "feasible": g.feasible if g else False, "reward": g.reward if g else 0.0}
            if g and g.feasible:
                out.update(cost=g.cost, delay_cost=g.delay_cost, moves=g.moves)
            elif g:
                out.update(violations=g.violations[:20], entry_problems=g.parse_problems)
            return _json(out)

        super().__init__(mcp)
        self.rubric = BerthPlanRubric()

    # ------------------------------------------------------------ Task API

    def list_splits(self):
        return [{"name": s, "type": s, "num_tasks": self.pack.count(s)} for s in self.pack.splits()]

    def num_tasks(self, split):
        return self.pack.count(split)

    def get_task(self, split, index):
        return self.pack.public(self.pack.at(split, int(index)))

    def get_task_range(self, split, start=None, stop=None):
        n = self.pack.count(split)
        start, stop = int(start or 0), int(n if stop is None else stop)
        return [self.pack.public(self.pack.at(split, i)) for i in range(max(0, start), min(n, stop))]

    def list_tasks(self, split):
        return self.get_task_range(split)

    # ------------------------------------------------------------ lifecycle

    def reset(self, seed: int | None = None, episode_id: str | None = None, split: str | None = None,
              index: int | None = None, task_id: str | None = None, **kwargs: Any) -> Observation:
        if task_id is not None:
            task = self.pack.get(task_id)
        else:
            split = split or ("train" if "train" in self.pack.splits() else self.pack.splits()[0])
            n = self.pack.count(split)
            index = random.Random(seed).randrange(n) if index is None else int(index)
            task = self.pack.at(split, index)
        ep = Episode(task, str(episode_id)[:64] if episode_id else uuid.uuid4().hex[:12])
        STORE.add(ep)
        self._episode = ep
        self._state = BerthState(episode_id=ep.episode_id, step_count=0, task_id=task.task_id, split=task.split)
        self._reset_rubric()
        self.rubric.set_task(task)
        return Observation(done=False, reward=None, metadata={
            "episode_id": ep.episode_id,
            "task_id": task.task_id,
            "split": task.split,
            "difficulty": task.difficulty,
            "instructions": rules(task, MAX_CHECKS),
            "situation": situation(task),
            "max_checks": MAX_CHECKS,
            "max_tool_calls": MAX_TOOL_CALLS,
            "tools": ["get_situation", "check_plan", "submit_plan"],
            "prompt_version": PROMPT_VERSION,
            "viewer_url": f"/viewer/#/live/{ep.episode_id}",
        })

    def _require(self, open_only: bool = False) -> Episode:
        if self._episode is None:
            raise ValueError("no episode: call reset first")
        if open_only and self._episode.done:
            raise ValueError("the episode is over: the plan was already submitted")
        return self._episode

    @property
    def state(self) -> BerthState:
        return self._state

    # ------------------------------------------------------------ stepping

    def _after_step(self, action: Action, obs: Observation) -> Observation:
        ep = self._episode
        if ep is None or not isinstance(action, CallToolAction):
            return obs
        self._state.tool_calls += 1
        self._state.step_count += 1
        obs.reward = self._apply_rubric(action, obs)  # 0.0 unless this was submit_plan
        if ep.done:
            g = self.rubric.last_grade
            obs.done = True
            obs.metadata = {**(obs.metadata or {}), "grade": g.as_dict() if g else ep.grade,
                            "rubric": {name: r.last_score for name, r in self.rubric.named_rubrics()},
                            "episode_id": ep.episode_id}
            self._state.done, self._state.reward = True, obs.reward
            self._state.grade = obs.metadata["grade"]
        elif self._state.tool_calls >= MAX_TOOL_CALLS:
            ep.done, ep.end_reason = True, "tool_call_limit"
            obs.done, obs.reward = True, 0.0
            obs.metadata = {**(obs.metadata or {}), "end_reason": "tool_call_limit", "episode_id": ep.episode_id}
            self._state.done, self._state.reward = True, 0.0
        return obs

    def step(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return self._after_step(action, super().step(action, timeout_s=timeout_s, **kwargs))

    async def step_async(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return self._after_step(action, await super().step_async(action, timeout_s=timeout_s, **kwargs))

    def _step_impl(self, action: Action, timeout_s: float | None = None, **kwargs) -> Observation:
        return Observation(done=False, reward=0.0, metadata={"error": "use the MCP tools: get_situation, check_plan, submit_plan"})


def _raw(plan):
    """Tool arguments arrive as PlanEntry models (or a JSON string); the core parser takes plain data."""
    if isinstance(plan, list):
        return [p.model_dump(exclude_none=True) if isinstance(p, BaseModel) else p for p in plan]
    return plan
