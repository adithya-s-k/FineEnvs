"""PortSimEnv v1's three MCP tools, their text and their limits (as ../openenv serves them), one episode per server.

urn:portsim:load-task/v1 starts an episode; data/get reports it and never the task's reference plans. The task packs
come from BERTH_TASKS_DIR, else from berth_core's tasks/ (dock-v1-eval and dock-v1-train)."""

import json
import os
from typing import Annotated, Any

from agentenv_protocol import AgentEnvEnvironment, DataPart, environment_card, extension, get_data, reset_data, tool
from berth_core import Task, evaluate, grade, load_pack, parse_plan, plan_to_list, situation
from berth_core.model import PlanError
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from pydantic import BaseModel, Field, TypeAdapter

MAX_CHECKS = 10
MAX_TOOL_CALLS = 24
TOOL_ORDER = ["get_situation", "check_plan", "submit_plan"]


class PlanEntry(BaseModel):
    ship: int = Field(description="ship id from the table")
    berth_hour: int = Field(description="hour the ship berths (>= its arrival hour)")
    section: int = Field(description="first (lowest-numbered) section the ship occupies")
    cranes: int | None = Field(default=None, description="quay cranes working the ship (tasks with crane rules; "
                                                         "defaults to the ship's planned cranes)")


PlanArg = Annotated[list[PlanEntry] | str, Field(
    description='one entry per ship: [{"ship": 0, "berth_hour": 36, "section": 22}, ...]')]
PLAN = TypeAdapter(PlanArg)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _raw(plan):
    if isinstance(plan, list):
        return [p.model_dump(exclude_none=True) if isinstance(p, BaseModel) else p for p in plan]
    return plan


def _plan_result(task: Task, plan) -> dict:
    r = evaluate(task, plan)
    return {
        "feasible": r.feasible,
        "violations": r.violations,
        "cost": r.cost, "delay_cost": r.delay_cost, "moves": r.moves,
        "ships": [{"ship": s.ship, "departure": s.departure, "delay_h": s.delay_h, "moved": s.moved, "cost": s.cost}
                  | ({"cranes": s.cranes, "hours_alongside": s.departure - s.berth_hour}
                     if s.cranes is not None else {})
                  for s in r.ships if s.berth_hour is not None],
    }


@environment_card(name="portsim")
class PortSimEnv(AgentEnvEnvironment):
    def __init__(self):
        self.pack = load_pack()
        self.reset()

    def create_app(self) -> FastMCP:
        return self.mount(FastMCP("portsim", middleware=[Episode(self)]))

    @tool()
    def get_situation(self) -> str:
        """The quay, notices, ships already alongside, closed sections and the table of ships to berth."""
        return situation(self._require())

    @tool()
    def check_plan(self, plan: PlanArg) -> str:
        """Check your plan: rule violations per ship, departure and delay per ship, and the plan's cost.
        Uses one of your checks. It does not grade the plan."""
        task = self._require(open_only=True)
        if self.checks_used >= MAX_CHECKS:
            return _json({"error": f"no checks left (you had {MAX_CHECKS}); submit_plan when ready"})
        self.checks_used += 1
        try:
            parsed, problems = parse_plan(task, _raw(plan))
        except PlanError as e:
            return _json({"error": str(e), "checks_left": MAX_CHECKS - self.checks_used})
        out = _plan_result(task, parsed)
        if problems:
            out["entry_problems"] = problems
            out["feasible"] = False
        out["checks_left"] = MAX_CHECKS - self.checks_used
        return _json(out)

    @tool()
    def submit_plan(self, plan: PlanArg) -> str:
        """Submit your final berth plan. Ends the episode; the plan is graded once."""
        return _json(self._submit(plan))

    @extension("urn:portsim:load-task/v1", description="Start an episode on the task with this id.")
    def load_task(self, task_id: str) -> dict:
        try:
            task = self.pack.get(task_id)
        except KeyError:
            raise ValueError(f"unknown task id {task_id!r}") from None
        self.reset()
        self.task = task
        return {"task_id": task_id}

    @extension("urn:portsim:submit-plan/v1",
               description="Submit a plan as the submit_plan tool does, outside the agent's tool calls.")
    def submit(self, plan: PlanArg) -> dict:
        return self._submit(PLAN.validate_python(plan))

    @reset_data
    def reset(self) -> None:
        self.task = None
        self.checks_used = 0
        self.calls_used = 0
        self.done = False
        self.end_reason = None
        self.plan = None
        self.grade = None
        self.reward = None

    @get_data
    def get(self) -> list[DataPart]:
        return [DataPart(data={"task_id": self.task.task_id if self.task else None, "checks_used": self.checks_used,
                               "calls_used": self.calls_used, "done": self.done, "end_reason": self.end_reason,
                               "plan": self.plan, "grade": self.grade, "reward": self.reward})]

    def count_call(self) -> None:
        self.calls_used += 1
        if self.calls_used >= MAX_TOOL_CALLS and not self.done:
            self.done, self.end_reason, self.reward = True, "tool_call_limit", 0.0

    def _require(self, open_only: bool = False) -> Task:
        if self.task is None:
            raise ValueError("no episode: call reset first")
        if open_only and self.done:
            raise ValueError("the episode is over: the plan was already submitted")
        return self.task

    def _submit(self, plan) -> dict:
        task = self._require(open_only=True)
        try:
            parsed, problems = parse_plan(task, _raw(plan))
        except PlanError as e:
            parsed, problems = {}, [str(e)]
        g = grade(task, parsed, problems)
        self.done, self.end_reason, self.reward = True, "submitted", g.reward
        self.plan, self.grade = plan_to_list(parsed), g.as_dict()
        out = {"submitted": True, "feasible": g.feasible, "reward": g.reward}
        if g.feasible:
            out.update(cost=g.cost, delay_cost=g.delay_cost, moves=g.moves)
        else:
            out.update(violations=g.violations[:20], entry_problems=g.parse_problems)
        return out


class Episode(Middleware):
    """Counts every tools/call, including those that fail validation or name no tool, and lists the tools in
    upstream's order (the SDK registers them alphabetically)."""

    def __init__(self, env: PortSimEnv):
        self.env = env

    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        finally:
            self.env.count_call()

    async def on_list_tools(self, context, call_next):
        return sorted(await call_next(context), key=lambda t: TOOL_ORDER.index(t.name))


def main() -> None:
    PortSimEnv().create_app().run(transport="http", host=os.environ.get("MCP_HOST", "0.0.0.0"),
                                  port=int(os.environ.get("MCP_PORT", "18765")), show_banner=False)


if __name__ == "__main__":
    main()
