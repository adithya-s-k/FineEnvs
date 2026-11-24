"""The ops control center live: one storm day played by an ops desk and a crew desk (or one desk running both) in
hourly watches on the gateway's virtual clock.

urn:airops:occ-load/v1 starts a day. A desk reads its page with wait_turn, which long-polls until its turn is granted
(a re-poll costs no model call), acts through its tools, and ends its turn with end_turn, which returns at once so the
gateway's trigger can deliver the next watch's FAA advisories through the hidden airops_notice; the env moves the
clock when a watch opens. A desk's caller role (ops, crew or both) picks its desk. urn:airops:end-day/v1 runs the rest
of the day and grades it; data/get serves the record, never the truth ahead of the watch."""

import asyncio
import json
import time
from typing import Annotated, Any, Literal

import httpx
from agentenv_protocol import (
    AgentEnvEnvironment,
    DataPart,
    ToolContext,
    environment_card,
    extension,
    get_data,
    reset_data,
    tool,
)
from airops_core.occ.day import TEXT_CHARS, OccDay, Refused
from airops_core.occ.pack import Pack
from airops_core.schedule import virtual_time
from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

ENV = "airops-occ"
LOAD_URI = "urn:airops:occ-load/v1"
END_DAY_URI = "urn:airops:end-day/v1"
CLOCK_URI = "urn:agentenv:clock/v1"
NOTICE_TOOL = "airops_notice"
POLL_SECONDS = 240.0
NOTICE_SECONDS = 60.0
TICK = 1.0
EMPTY = {"task_id": None, "mode": None, "done": False, "calls": [], "turns": [], "requests": [], "audit": None,
         "grade": None, "reward": None}
Time = Annotated[str | None, Field(description='local "HH:MM" ("01:10+1" after midnight), or null')]
Flight = Annotated[str | None, Field(description='flight id, e.g. "f12", or null')]
Text = Annotated[str | None, Field(description=f"at most {TEXT_CHARS} characters, or null")]


def _strict(schema: dict) -> None:
    """OpenAI's strict mode holds a tool to every field of its schema: all required, none with a default."""
    schema["required"] = list(schema["properties"])
    for prop in schema["properties"].values():
        prop.pop("default", None)


class OpsChange(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_strict)

    flight: str = Field(description='flight id, e.g. "f12"')
    action: Literal["cancel", "cancel_pair", "hold", "release", "claim", "swap_slots"] = Field(
        description="cancel it; cancel it and its round-trip partner; hold it to `time`; release its hold and claim; "
                    "claim the slot at `time` (its arrival); swap slots with `other`")
    time: Time = None
    other: Flight = None
    ref: str | None = Field(default=None, description="the request this answers, e.g. R3, or null")


class CrewMove(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_strict)

    action: Literal["assign", "call_reserve", "swap", "deadhead", "rest", "release"] = Field(
        description="assign `unit` to fly `flight`; call short-call reserve `unit` to report at `time` (null: in time "
                    "for `flight`) and assign it to `flight` if given; swap the remaining trips of `unit` and `other`, "
                    "two units at one station; deadhead `unit` on `flight`; rest `unit` to report again at `time` "
                    "(null: as soon as rested; a hotel away from base); release `unit` from `flight`, which goes back "
                    "to the crew system (or keep `unit` off a ride its trip plans on `flight`: it stays)")
    unit: str = Field(description='crew unit id, e.g. "P-A32X-0391"')
    flight: Flight = None
    other: str | None = Field(default=None, description="the other unit of a swap, or null")
    kind: Literal["pilot", "cabin"] | None = Field(default=None, description="null: the unit's own")
    time: Time = None
    ref: str | None = Field(default=None, description="the request this answers, e.g. R3, or null")


Changes = Annotated[list[OpsChange], Field(description="changes in order; one that is refused leaves the others")]
Moves = Annotated[list[CrewMove], Field(description="moves in order; one that is refused leaves the others")]
Status = Literal["open", "attention", "frozen", "flown", "cancelled"]


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _rows(items) -> list[dict]:
    return [x.model_dump() if isinstance(x, BaseModel) else x for x in items]


@environment_card(name=ENV)
class OccEnv(AgentEnvEnvironment):
    DAY, PACK = OccDay, Pack

    def __init__(self):
        self.pack = self.PACK()
        self.set_time_url: str | None = None
        self.loaded = False
        self.turn_timeout = 1800.0
        self.changed: asyncio.Event | None = None
        self.clock_k: int | None = None
        self.since: dict[int, float] = {}
        self.opened_at = 0.0
        self.reset()

    def create_app(self) -> FastMCP:
        return self.mount(FastMCP(ENV))

    # ------------------------------------------------------------------ the day's machinery

    def _day(self) -> OccDay:
        if self.day is None:
            raise ValueError(f"no day loaded: load one with {LOAD_URI}")
        return self.day

    def _event(self) -> asyncio.Event:
        if self.changed is None:
            self.changed = asyncio.Event()
        return self.changed

    def police(self) -> None:
        """The backstops: a watch's advisories the trigger hasn't brought in NOTICE_SECONDS the env delivers, and a
        desk that has held its turn past the turn timeout is off."""
        day, now = self._day(), time.monotonic()
        if day.done:
            return
        if day.pending and now - self.opened_at > NOTICE_SECONDS:
            day.missed("late")
        if day.turn is not None:
            started = self.since.setdefault(day.turn["n"], now)
            if now - started > self.turn_timeout:
                day.time_out(day.turn["desk"])

    async def after(self) -> None:
        day = self._day()
        if day.k != self.clock_k:
            self.clock_k, self.opened_at = day.k, time.monotonic()
            if self.set_time_url is not None:
                async with httpx.AsyncClient() as http:
                    response = await http.put(self.set_time_url, timeout=10, json={
                        "virtual_time": virtual_time(day.task, day.t), "virtual_seconds_per_real_second": 0})
                    response.raise_for_status()
        if day.turn is not None:
            self.since.setdefault(day.turn["n"], time.monotonic())
        event = self._event()
        event.set()
        self.changed = asyncio.Event()

    async def act(self, ctx: ToolContext, name: str, **args) -> str:
        day = self._day()
        self.police()
        try:
            result = day.call(ctx.role, name, args)
        except Refused as e:
            raise ValueError(str(e)) from None
        await self.after()
        return _json(result)

    # ------------------------------------------------------------------ shared tools

    @tool(description="Wait for your turn and read your page: returns your page once your turn is granted, else after "
                      "a few minutes where the day stands ({waiting_for}); {done: true} once the day is over for you.")
    async def wait_turn(self, ctx: ToolContext) -> str:
        day = self._day()
        deadline = time.monotonic() + POLL_SECONDS
        while True:
            self.police()
            result = day.call(ctx.role, "wait_turn", {})
            await self.after()
            if "waiting_for" not in result or time.monotonic() >= deadline:
                return _json(result)
            try:
                await asyncio.wait_for(self._event().wait(), timeout=TICK)
            except TimeoutError:
                pass

    @tool(description="End your turn. The next desk's turn, or the next watch, follows; your next page comes when your "
                      "turn comes again. stand_down: true skips every turn you have left today.")
    async def end_turn(self, ctx: ToolContext, note: Annotated[str | None, Field(
            description="a note for the record, or null")], stand_down: bool) -> str:
        return await self.act(ctx, "end_turn", note=note, stand_down=stand_down)

    @tool(description="Leave the day: your desk takes no more turns. The harness calls it as it stops.")
    async def resign(self, ctx: ToolContext, reason: Literal[
            "done", "cost_cap", "context_full", "turn_limit", "time_limit", "error", "crash"]) -> str:
        return await self.act(ctx, "resign", reason=reason)

    # ------------------------------------------------------------------ ops tools

    @tool(description="Flights by scheduled departure, as many as fit 4k characters, filtered by any of ids, station, "
                      "status and "
                      "scheduled departure window: schedule, pax, type, its own delay once known (else ?), your plan, "
                      "where the projection has it leave and land and how late, its slot, the crew flags, deadline, "
                      "the cancellation's price now and its round-trip partner.")
    async def ops_flights(self, ctx: ToolContext, ids: list[str] | None, station: str | None, status: Status | None,
                          start: Time, end: Time) -> str:
        return await self.act(ctx, "ops_flights", ids=ids, station=station, status=status, start=start, end=end)

    @tool(description="A hub's FAA programs as issued: the advisories, JetBlue's slots in time order with who holds "
                      "each (* claimed), the flights the program controls with no slot, and ground stops.")
    async def ops_program(self, ctx: ToolContext, hub: Literal["JFK", "BOS"]) -> str:
        return await self.act(ctx, "ops_program", hub=hub)

    @tool(description="Aircraft by station and type: where each is (or is going), ready when, its next projected "
                      "legs; tomorrow's needs and the ferries the projection sends tonight.")
    async def ops_aircraft(self, ctx: ToolContext, station: str | None) -> str:
        return await self.act(ctx, "ops_aircraft", station=station)

    @tool(description="Check changes without committing them: the projected cost and its change by part, new "
                      "automatic cancellations, every flight leaving over 30 min earlier or later or cancelled "
                      "(warnings), the flights that move most, crew flags that change, and the changes refused "
                      "(committed or cancelled flights, FAA CDM).")
    async def ops_check(self, ctx: ToolContext, changes: Changes) -> str:
        return await self.act(ctx, "ops_check", changes=_rows(changes))

    @tool(description="Commit changes, decided now. Cancellations are final; holds go no earlier than the freeze line; "
                      "flights committed at an earlier watch can't be changed.")
    async def ops_confirm(self, ctx: ToolContext, changes: Changes) -> str:
        return await self.act(ctx, "ops_confirm", changes=_rows(changes))

    @tool(description="Ask the crew desk: COVER (crew `flight` by `time`) or TEXT (at most 400 characters).")
    async def ops_request(self, ctx: ToolContext, type: Literal["COVER", "TEXT"], flight: Flight, time: Time,
                          text: Text) -> str:
        return await self.act(ctx, "ops_request", type=type, flight=flight, time=time, text=text)

    @tool(description="Answer a crew request: accept carries it out now as your decision (HOLD holds the flight, no "
                      "earlier than the freeze line; CANCEL and CANCEL_PAIR cancel), decline leaves it.")
    async def ops_reply(self, ctx: ToolContext, request: str, answer: Literal["accept", "decline"],
                        text: Text) -> str:
        return await self.act(ctx, "ops_reply", request=request, answer=answer, text=text)

    # ------------------------------------------------------------------ crew tools

    @tool(description="As ops_flights, with the crews the projection has on each flight; status attention adds every "
                      "flight the crew system has a word on.")
    async def crew_flights(self, ctx: ToolContext, ids: list[str] | None, station: str | None, status: Status | None,
                           start: Time, end: Time) -> str:
        return await self.act(ctx, "crew_flights", ids=ids, station=station, status=status, start=start, end=end)

    @tool(description="Crew units, at most 40, filtered by any of unit, station, base, kind, group and state: where, "
                      "ready when, report, legs and flight time so far, the last legal block-in (pilots with the "
                      "extension), release and rest, the next legs of its trip, reserve status.")
    async def crew_crews(self, ctx: ToolContext, unit: str | None, station: str | None, base: str | None,
                         kind: Literal["pilot", "cabin"] | None, group: Literal["A32X", "A220", "E190"] | None,
                         state: Literal["on_duty", "resting", "reserve", "off"] | None) -> str:
        return await self.act(ctx, "crew_crews", unit=unit, station=station, base=base, kind=kind, group=group,
                              state=state)

    @tool(description="Reserves by base: short-call (earliest report, called by when) and airport standby (until "
                      "when).")
    async def crew_reserves(self, ctx: ToolContext, base: str | None) -> str:
        return await self.act(ctx, "crew_reserves", base=base)

    @tool(description="A hub's FAA programs as issued, read-only: the advisories, JetBlue's slots in time order with "
                      "who holds each (* claimed), the flights the program controls with no slot, and ground stops.")
    async def crew_program(self, ctx: ToolContext, hub: Literal["JFK", "BOS"]) -> str:
        return await self.act(ctx, "crew_program", hub=hub)

    @tool(description="The standard crew desk's options for a flight's crew of one kind now, priced by the wait, the "
                      "deadhead, the hotel and the end of the day: (a) its own crew, (b) a reserve, (c) a swap, "
                      "(d) a deadhead, (e) the pilots' extension.")
    async def crew_options(self, ctx: ToolContext, flight: str, kind: Literal["pilot", "cabin"]) -> str:
        return await self.act(ctx, "crew_options", flight=flight, kind=kind)

    @tool(description="Check crew moves without committing them: the projected cost and its change, the flights whose "
                      "crew or departure changes, every flight leaving over 30 min earlier or later or cancelled "
                      "(warnings), what each unit named then flies and its legality, the legs they hand back to the "
                      "crew system (handed_back), and refusals.")
    async def crew_check(self, ctx: ToolContext, moves: Moves) -> str:
        return await self.act(ctx, "crew_check", moves=_rows(moves))

    @tool(description="Commit crew moves, decided now. Flights you leave alone stay with the crew system.")
    async def crew_confirm(self, ctx: ToolContext, moves: Moves) -> str:
        return await self.act(ctx, "crew_confirm", moves=_rows(moves))

    @tool(description="Ask the ops desk: HOLD (hold `flight` until `time`), CANCEL (`flight`), CANCEL_PAIR "
                      "(`flight` and its round-trip partner) or TEXT (at most 400 characters).")
    async def crew_request(self, ctx: ToolContext, type: Literal["HOLD", "CANCEL", "CANCEL_PAIR", "TEXT"],
                           flight: Flight, time: Time,
                           text: Text) -> str:
        return await self.act(ctx, "crew_request", type=type, flight=flight, time=time, text=text)

    @tool(description="Answer an ops request: accept commits you to it (COVER: crew the flight by its time; your moves "
                      "follow with ref), decline refuses it.")
    async def crew_reply(self, ctx: ToolContext, request: str, answer: Literal["accept", "decline"],
                         text: Text) -> str:
        return await self.act(ctx, "crew_reply", request=request, answer=answer, text=text)

    # ------------------------------------------------------------------ the env's own

    @tool(description="Deliver one FAA advisory at this watch. Called by the env's triggers.")
    async def airops_notice(self, news_id: str) -> str:
        result = self._day().notice(news_id, "trigger")
        await self.after()
        return _json(result)

    @extension(LOAD_URI, description="Start a day on the task with this id, played by two desks (joint) or one (one).")
    def occ_load(self, task_id: str, mode: str = "joint", turn_timeout_s: float = 1800.0) -> dict:
        """Loading again would replay the day with its advisories already seen, unseen by the audit."""
        if self.loaded:
            raise ValueError("a day was already loaded in this env: each day needs a new deploy")
        if task_id not in self.pack.tasks:
            raise ValueError(f"unknown task id {task_id!r}")
        self.reset()
        self.day = self.DAY(self.pack.world(task_id), mode)
        self.loaded, self.turn_timeout = True, float(turn_timeout_s)
        self.clock_k = 0
        return {"task_id": task_id, "mode": mode, "watches": len(self.day.times), "max_turns": self.day.max_turns}

    @extension(END_DAY_URI, description="Run the rest of the day without the desks, grade it on the truth, audit it.")
    def end_day(self) -> dict:
        return self._day().end()

    @extension(CLOCK_URI, description="Follow the gateway's virtual clock; the env moves it when a watch opens.")
    async def sync_time(self, env_get_time_url: str) -> dict:
        self.set_time_url = env_get_time_url.removesuffix("/clock/time") + "/clock/set-time"
        async with httpx.AsyncClient() as http:
            response = await http.get(env_get_time_url, timeout=10)
            response.raise_for_status()
        return {"env_get_time_url": env_get_time_url, "virtual_time": response.json()["virtual_time"]}

    @reset_data
    def reset(self) -> None:
        self.day: OccDay | None = None

    @get_data
    def get(self) -> list[DataPart]:
        return [DataPart(data=self.day.data() if self.day else EMPTY)]
