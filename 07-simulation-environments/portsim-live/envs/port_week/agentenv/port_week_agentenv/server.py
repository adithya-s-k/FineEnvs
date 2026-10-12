"""PortSim live: one week at one quay, played in watches on the gateway's virtual clock; PortSim marine: the same
week with the port's pilots and tugs; PortSim wind: the marine week in a real weather week, with the forecast at every
watch; PortSim lines: the wind week with the ships' agents, who report ETAs, answer the planner and agree arrival times
through two tools only their role sees.

urn:portsim:live-load/v1 starts a week; each advance re-arms the clock at the next bulletin, and the gateway's
triggers deliver the parties' notices through the hidden port_notice tool. data/get never reports reference plans."""

import json
import os
import sys
from typing import Annotated, Any

import httpx
from agentenv_protocol import AgentEnvEnvironment, DataPart, environment_card, extension, get_data, reset_data, tool
from berth_core import TaskPack
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from port_week_core import lines, marine, packs, wind
from port_week_core.lines import LinesWeek
from port_week_core.marine import MarineWeek
from port_week_core.schedule import virtual_time
from port_week_core.wind import WindWeek
from port_week_core.world import Week
from pydantic import BaseModel, Field

from .contract import (
    CLOCK_URI,
    END_WEEK_URI,
    LINE_TOOLS,
    LINES_ENV,
    LIVE_ENV,
    LIVE_LOAD_URI,
    MARINE_ENV,
    NOTICE_TOOL,
    TOOLS,
    WIND_ENV,
)

EMPTY = {"task_id": None, "watches": None, "watch": None, "hour": None, "done": False, "end_reason": None,
         "calls_used": 0, "planning_calls": [], "plan": [], "notices": [], "frozen": [], "refusals": [],
         "excused": {"problems": [], "cost": []}, "audit": None, "grade": None, "reward": None}


class Window(BaseModel):
    ship: int = Field(description="ship id from the table")
    berth_hour: int = Field(description="hour the ship berths (>= its arrival; from watch 1 on, >= the freeze line)")
    section: int = Field(description="first (lowest-numbered) section the ship occupies")
    cranes: int | None = Field(default=None, description="quay cranes working the ship (default: its planned cranes)")


WindowsArg = Annotated[list[Window] | str, Field(
    description='berth windows: [{"ship": 7, "berth_hour": 95, "section": 4, "cranes": 2}, ...]; '
                'ships left out keep their confirmed windows')]


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def entries(plan):
    if isinstance(plan, list):
        return [p.model_dump(exclude_none=True) if isinstance(p, BaseModel) else p for p in plan]
    return plan


def _reply(week: Week, result: dict) -> str:
    return dumps({**result, "messages": week.drain()})


@environment_card(name=LIVE_ENV)
class PortSimLiveEnv(AgentEnvEnvironment):
    ENV = LIVE_ENV
    WEEK = Week

    def __init__(self):
        self.pack = self.task_pack()
        self.set_time_url: str | None = None
        self.loaded = False
        self.reset()

    @staticmethod
    def task_pack() -> TaskPack:
        return packs.live_pack()

    def create_app(self) -> FastMCP:
        return self.mount(FastMCP(self.ENV, middleware=[LiveEpisode(self)]))

    @tool(description="The current watch: the hour, the week as the port knows it now, your confirmed berth windows "
                      "(departed, berthed, frozen or open), the ships still without a window, and new messages.")
    def get_situation(self) -> str:
        week = self._week()
        return _reply(week, week.situation())

    @tool(description="Check berth windows without confirming them: your entries over your confirmed windows, on the "
                      "week as known now. Lists the ships with a rule problem or a cost, the plan's cost, and the "
                      "entries confirm_berths would refuse. Problems and cost that news brought to a frozen window are "
                      "listed as excused and not counted. Uses one of this watch's 3 planning calls.")
    def check_plan(self, plan: WindowsArg) -> str:
        week = self._week(open_only=True)
        return _reply(week, week.check(entries(plan)))

    @tool(description="Confirm berth windows with the ships: each entry sets or changes one ship's window; ships left "
                      "out keep theirs. From watch 1 on, windows starting before the freeze line (6 hours from now) "
                      "are frozen, and new ones must start at or after it. Uses one of this watch's 3 planning calls.")
    def confirm_berths(self, plan: WindowsArg) -> str:
        week = self._week(open_only=True)
        return _reply(week, week.confirm(entries(plan)))

    @tool(description="End this watch: the port moves to the next bulletin, ships berth and sail on their confirmed "
                      "windows, and windows starting before the new freeze line freeze. The bulletin's messages arrive "
                      "with your next tool result. After the last watch, the rest of the week runs on your windows and "
                      "is graded.")
    async def advance(self) -> str:
        week = self._week(open_only=True)
        if week.last:
            return _reply(week, week.finish("done"))
        if self.set_time_url is None:
            raise ValueError(f"the gateway clock is not synced: call {CLOCK_URI} sync_time first")
        async with httpx.AsyncClient() as http:
            response = await http.put(self.set_time_url, timeout=10, json={
                "virtual_time": virtual_time(week.task, week.watches[week.watch + 1].hour),
                "virtual_seconds_per_real_second": 0})
            response.raise_for_status()
        return _reply(week, week.open())

    @tool(description="Deliver one party's notice for this watch: the port applies the news and queues the message. "
                      "Called by the env's triggers.")
    def port_notice(self, event_id: str, name: str, text: str) -> str:
        return dumps(self._week(open_only=True).notice(event_id, name, text, "trigger"))

    @extension(LIVE_LOAD_URI, description="Start a live week on the one-week task with this id; an env plays one week.")
    def live_load(self, task_id: str) -> dict:
        """Loading again would replay the week with its news already seen, unseen by the audit."""
        if self.loaded:
            raise ValueError("a week was already loaded in this env: each live week needs a new deploy")
        try:
            task = self.pack.get(task_id)
        except KeyError:
            raise ValueError(f"unknown task id {task_id!r}") from None
        self.reset()
        self.week = self.WEEK(task)
        self.loaded = True
        return {"task_id": task_id, "watches": len(self.week.watches)}

    @extension(END_WEEK_URI, description="Run the rest of the week on the confirmed windows, grade it and audit it.")
    def end_week(self) -> dict:
        return self._week().end()

    @extension(CLOCK_URI, description="Follow the gateway's virtual clock; advance re-arms it at each bulletin.")
    async def sync_time(self, env_get_time_url: str) -> dict:
        self.set_time_url = env_get_time_url.removesuffix("/clock/time") + "/clock/set-time"
        async with httpx.AsyncClient() as http:
            response = await http.get(env_get_time_url, timeout=10)
            response.raise_for_status()
        return {"env_get_time_url": env_get_time_url, "virtual_time": response.json()["virtual_time"]}

    @reset_data
    def reset(self) -> None:
        self.week: Week | None = None

    @get_data
    def get(self) -> list[DataPart]:
        return [DataPart(data=self.week.data() if self.week else EMPTY)]

    def _week(self, open_only: bool = False) -> Week:
        if self.week is None:
            raise ValueError(f"no week loaded: load one with {LIVE_LOAD_URI}")
        if open_only and self.week.done:
            raise ValueError("the week is over")
        return self.week


@environment_card(name=MARINE_ENV)
class PortSimMarineEnv(PortSimLiveEnv):
    ENV = MARINE_ENV
    WEEK = MarineWeek
    task_pack = staticmethod(marine.pack)


@environment_card(name=WIND_ENV)
class PortSimWindEnv(PortSimMarineEnv):
    ENV = WIND_ENV
    WEEK = WindWeek
    task_pack = staticmethod(wind.pack)


@environment_card(name=LINES_ENV)
class PortSimLinesEnv(PortSimWindEnv):
    ENV = LINES_ENV
    WEEK = LinesWeek
    task_pack = staticmethod(lines.pack)

    @tool(description="Confirm berth windows with the ships: each entry sets or changes one ship's window; ships left "
                      "out keep theirs. From watch 1 on, windows starting before the freeze line (6 hours from now) "
                      "are frozen, and new ones must start at or after it. Returns the confirmed plan's problems and "
                      "cost, as check_plan does. Uses one of this watch's 3 planning calls.")
    def confirm_berths(self, plan: WindowsArg) -> str:
        week = self._week(open_only=True)
        return _reply(week, week.confirm(entries(plan)))

    @tool(description="A ship's state as its agent knows it now: its ETA, or the arrival agreed and its turn in the "
                      "queue, or when it arrived. Records what the agent tells the planner. For the ships' agents.")
    def ship_status(self, agent: str, ship: int) -> str:
        return dumps(self._lines().status(agent, ship))

    @tool(description="Agree an arrival time for a ship when it can make it, at cruising speed or by slowing; the ship "
                      "then arrives at that hour and keeps its turn. For the ships' agents.")
    def agree_arrival(self, agent: str, ship: int, arrival: int) -> str:
        return dumps(self._lines().request(agent, ship, arrival))

    def _lines(self) -> LinesWeek:
        week = self._week(open_only=True)
        if not isinstance(week, LinesWeek):
            raise ValueError("this env has no ships' agents")
        return week


class LiveEpisode(Middleware):
    """Counts the planner's tools/calls, failed ones included; port_notice calls are the triggers', and ship_status and
    agree_arrival the ships' agents'."""

    def __init__(self, env: PortSimLiveEnv):
        self.env = env

    async def on_call_tool(self, context, call_next):
        if context.message.name not in (NOTICE_TOOL, *LINE_TOOLS) and self.env.week is not None:
            self.env.week.calls += 1
        return await call_next(context)

    async def on_list_tools(self, context, call_next):
        order = [*TOOLS, NOTICE_TOOL, *LINE_TOOLS]
        return sorted(await call_next(context), key=lambda t: order.index(t.name))


ENVS = {LIVE_ENV: PortSimLiveEnv, MARINE_ENV: PortSimMarineEnv, WIND_ENV: PortSimWindEnv, LINES_ENV: PortSimLinesEnv}


def main() -> None:
    name = os.environ.get("ENVIRONMENT_NAME", LIVE_ENV)
    if name not in ENVS:
        sys.exit(f"ENVIRONMENT_NAME {name!r} is not an env this server serves: {', '.join(ENVS)}")
    try:
        env = ENVS[name]()
    except wind.NotBuilt as e:
        sys.exit(str(e))
    env.create_app().run(transport="http", host=os.environ.get("MCP_HOST", "0.0.0.0"),
                         port=int(os.environ.get("MCP_PORT", "18765")), show_banner=False)


if __name__ == "__main__":
    main()
