"""AirOps live: one storm day at a JetBlue hub, played in hourly watches on the gateway's virtual clock.

urn:airops:live-load/v1 starts a day; each advance re-arms the clock at the next watch, and the gateway's triggers
deliver the watch's news through the hidden airops_notice tool. urn:airops:end-day/v1 runs the rest of the day and
grades it. data/get never reports the references or the truth the desk doesn't know yet."""

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from agentenv_protocol import AgentEnvEnvironment, DataPart, environment_card, extension, get_data, reset_data, tool
from airops_core.core.model import Task, load_pack
from airops_core.day import Day
from airops_core.schedule import (
    CLOCK_URI,
    END_DAY_URI,
    ENV,
    LIVE_LOAD_URI,
    NOTICE_TOOL,
    PLANNING_CALLS,
    TOOLS,
    virtual_time,
)
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from pydantic import BaseModel, ConfigDict, Field

PACK = Path(__file__).resolve().parents[2] / "tasks" / "b6-v1"
EMPTY = {"task_id": None, "watches": None, "watch": None, "t": None, "done": False, "end_reason": None,
         "calls_used": 0, "planning_calls": [], "plan": [], "frozen": [], "notices": [], "freezes": [], "changes": [],
         "refusals": [], "estimates": [], "audit": None, "grade": None, "reward": None}


def _strict(schema: dict) -> None:
    """OpenAI's strict mode holds a tool to every field of its schema, so all are required and none has a default;
    a call may still leave not_before out."""
    schema["required"] = list(schema["properties"])
    for prop in schema["properties"].values():
        prop.pop("default", None)


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_strict)

    flight: str = Field(description='flight id from the table, e.g. "f12"')
    action: Literal["cancel", "restore", "hold"] = Field(
        description="cancel the flight; restore a cancelled flight; hold a flight until not_before (restoring it if "
                    "cancelled)")
    not_before: str | None = Field(
        default=None,
        description='local time the flight may not leave before, "HH:MM" ("00:30+1" after midnight); null to let it '
                    "leave as early as it can, and with cancel")


ChangesArg = Annotated[list[Change] | str, Field(
    description='[{"flight": "f12", "action": "cancel", "not_before": null}, {"flight": "f30", "action": "hold", '
                '"not_before": "15:40"}, ...]; flights left out keep their plan')]


def tasks_by_id() -> dict[str, Task]:
    """The tasks of AIROPS_TASKS_DIR (pack folders or .jsonl files, separated by ':'), else the packaged pack."""
    paths = os.environ.get("AIROPS_TASKS_DIR")
    tasks = [t for p in paths.split(":") for t in load_pack(Path(p))] if paths else load_pack(Path(str(PACK)))
    return {t.task_id: t for t in tasks}


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _raw(changes):
    return [c.model_dump() if isinstance(c, BaseModel) else c for c in changes] if isinstance(changes, list) \
        else changes


def _reply(day: Day, result: dict) -> str:
    return _json({**result, "messages": day.drain()})


@environment_card(name=ENV)
class AirOpsLiveEnv(AgentEnvEnvironment):
    def __init__(self):
        self.tasks = tasks_by_id()
        self.set_time_url: str | None = None
        self.loaded = False
        self.reset()

    def create_app(self) -> FastMCP:
        return self.mount(FastMCP(ENV, middleware=[LiveDay(self)]))

    @tool(description="The current watch: the time, the freeze line, the slot caps as known for the next hours, the "
                      "FAA programs in effect, the latest METAR and TAF, every flight (its status, its delay once "
                      "known, your plan, where your plan has it leave and land on the day as known, and the aircraft "
                      "it takes), where the aircraft are, and new messages.")
    def get_situation(self) -> str:
        day = self._day()
        return _reply(day, day.situation())

    @tool(description="Check changes without confirming them: your changes over your plan, run on the day as known "
                      "now. Gives the plan's cost, the flights whose plan or times change, the automatic "
                      "cancellations, the flights the env would hold to the freeze line, and the changes "
                      f"confirm_changes would refuse. Uses one of this watch's {PLANNING_CALLS} planning calls.")
    def check_plan(self, changes: ChangesArg) -> str:
        day = self._day(open_only=True)
        return _reply(day, day.check(_raw(changes)))

    @tool(description="Commit changes to your plan; flights you leave out keep theirs. From watch 1 on, frozen flights "
                      "(those your plan has leaving before the freeze line, an hour from now) can't be changed, and "
                      "the env holds any other flight your plan would have leaving before the freeze line to it. "
                      f"Uses one of this watch's {PLANNING_CALLS} planning calls.")
    def confirm_changes(self, changes: ChangesArg) -> str:
        day = self._day(open_only=True)
        return _reply(day, day.confirm(_raw(changes)))

    @tool(description="End this watch: the clock moves an hour to the next watch, flights leave as your plan has "
                      "them, and flights your plan has leaving before the new freeze line freeze. The watch's news "
                      "arrives with your next tool result. After the last watch, the rest of the day runs on your "
                      "plan and is graded.")
    async def advance(self) -> str:
        day = self._day(open_only=True)
        if day.last:
            return _reply(day, day.finish("done"))
        if self.set_time_url is None:
            raise ValueError(f"the gateway clock is not synced: call {CLOCK_URI} sync_time first")
        async with httpx.AsyncClient() as http:
            response = await http.put(self.set_time_url, timeout=10, json={
                "virtual_time": virtual_time(day.task, day.watches[day.watch + 1].t),
                "virtual_seconds_per_real_second": 0})
            response.raise_for_status()
        return _reply(day, day.open())

    @tool(description="Deliver one news item (an FAA advisory, a TAF, a METAR or a forecast discussion) to the desk "
                      "at this watch. Called by the env's triggers.")
    def airops_notice(self, news_id: str) -> str:
        return _json(self._day(open_only=True).notice(news_id, "trigger"))

    @extension(LIVE_LOAD_URI, description="Start a live day on the task with this id; an env plays one day.")
    def live_load(self, task_id: str) -> dict:
        """Loading again would replay the day with its news already seen, unseen by the audit."""
        if self.loaded:
            raise ValueError("a day was already loaded in this env: each live day needs a new deploy")
        if task_id not in self.tasks:
            raise ValueError(f"unknown task id {task_id!r}")
        self.reset()
        self.day = Day(self.tasks[task_id])
        self.loaded = True
        return {"task_id": task_id, "watches": len(self.day.watches)}

    @extension(END_DAY_URI, description="Run the rest of the day on the plan, grade it on the truth and audit it.")
    def end_day(self) -> dict:
        return self._day().end()

    @extension(CLOCK_URI, description="Follow the gateway's virtual clock; advance re-arms it at each watch.")
    async def sync_time(self, env_get_time_url: str) -> dict:
        self.set_time_url = env_get_time_url.removesuffix("/clock/time") + "/clock/set-time"
        async with httpx.AsyncClient() as http:
            response = await http.get(env_get_time_url, timeout=10)
            response.raise_for_status()
        return {"env_get_time_url": env_get_time_url, "virtual_time": response.json()["virtual_time"]}

    @reset_data
    def reset(self) -> None:
        self.day: Day | None = None

    @get_data
    def get(self) -> list[DataPart]:
        return [DataPart(data=self.day.data() if self.day else EMPTY)]

    def _day(self, open_only: bool = False) -> Day:
        if self.day is None:
            raise ValueError(f"no day loaded: load one with {LIVE_LOAD_URI}")
        if open_only and self.day.done:
            raise ValueError("the day is over")
        return self.day


class LiveDay(Middleware):
    """Counts the planner's tools/calls, failed ones included (airops_notice calls are the triggers'), and lists the
    planner's tools first, in the order of the rules."""

    def __init__(self, env: AirOpsLiveEnv):
        self.env = env

    async def on_call_tool(self, context, call_next):
        if context.message.name != NOTICE_TOOL and self.env.day is not None:
            self.env.day.calls += 1
        return await call_next(context)

    async def on_list_tools(self, context, call_next):
        order = [*TOOLS, NOTICE_TOOL]
        return sorted(await call_next(context), key=lambda t: order.index(t.name))
