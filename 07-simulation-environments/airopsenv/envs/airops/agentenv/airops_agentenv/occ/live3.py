"""The ops control center with passengers live (airops-occ3): v2's env (live.py) on a b6-v3 day, every day costing P,
with a third desk in the role customer. The customer desk reads its page with wait_turn, looks passengers up with
cust_flights and cust_flight, and asks ops with cust_request; ops answers with ops_reply and texts the customer desk
with ops_request. Rebooking stays automatic: no tool rebooks."""

from typing import Annotated, Literal

from agentenv_protocol import ToolContext, environment_card, extension, tool
from airops_core.occ.day3 import OccDay3
from airops_core.occ.pack import Pack3
from fastmcp import FastMCP
from pydantic import Field

from .live import LOAD_URI, Flight, OccEnv, Status, Text, Time

ENV3 = "airops-occ3"
Note = Annotated[str | None, Field(description="at most 400 characters, or null")]


@environment_card(name=ENV3)
class OccEnv3(OccEnv):
    DAY, PACK = OccDay3, Pack3

    def create_app(self) -> FastMCP:
        return self.mount(FastMCP(ENV3))

    # ------------------------------------------------------------------ ops tools

    @tool(description="Ask or tell another desk: the crew desk COVER (crew `flight` by `time`) or TEXT, the customer "
                      "desk TEXT (at most 400 characters).")
    async def ops_request(self, ctx: ToolContext, to: Literal["crew", "customer"], type: Literal["COVER", "TEXT"],
                          flight: Flight, time: Time, text: Text) -> str:
        return await self.act(ctx, "ops_request", type=type, flight=flight, time=time, text=text, to=to)

    @tool(description="Answer a request: accept carries a HOLD out now as your hold (no earlier than the freeze line), "
                      "and a crew CANCEL or CANCEL_PAIR as your cancellation; a customer PROTECT or SUGGEST_CANCEL "
                      "accepted is your word only (act on it with ops_confirm); decline leaves it.")
    async def ops_reply(self, ctx: ToolContext, request: str, answer: Literal["accept", "decline"],
                        text: Text) -> str:
        return await self.act(ctx, "ops_reply", request=request, answer=answer, text=text)

    # ------------------------------------------------------------------ customer tools

    @tool(description="Flights by scheduled departure, as many as fit 4k characters, filtered by any of ids, station, "
                      "status and scheduled departure window: booked passengers and of them connecting onto and off "
                      "it, where the projection has it leave and land, its connecting passengers who miss it, and its "
                      "passengers with no later JetBlue seat today. Status attention: connections at risk and flights "
                      "ops might cancel.")
    async def cust_flights(self, ctx: ToolContext, ids: list[str] | None, station: str | None,
                           status: Status | None, start: Time, end: Time) -> str:
        return await self.act(ctx, "cust_flights", ids=ids, station=station, status=status, start=start, end=end)

    @tool(description="One flight's passengers: booked, local and connecting; each connection onto it and off it on "
                      "the projection with its margin; its passengers by final destination with the later JetBlue "
                      "itineraries today that have seats; where they would go were it cancelled now (the passenger "
                      "system's rebooking: same day, a later day, refund); and how they fare as projected.")
    async def cust_flight(self, ctx: ToolContext, flight: str) -> str:
        return await self.act(ctx, "cust_flight", flight=flight)

    @tool(description="Ask the ops desk: HOLD (`flight` until `until`, for connecting passengers), PROTECT (don't "
                      "cancel `flight`), SUGGEST_CANCEL (cancel `flight`) or TEXT (`note`, at most 400 characters); "
                      "a note may come with any.")
    async def cust_request(self, ctx: ToolContext, type: Literal["HOLD", "PROTECT", "SUGGEST_CANCEL", "TEXT"],
                           flight: Flight, until: Time, note: Note) -> str:
        return await self.act(ctx, "cust_request", type=type, flight=flight, until=until, note=note)

    # ------------------------------------------------------------------ the env's own

    @extension(LOAD_URI, description="Start a day on the task with this id, played by three desks (joint) or one agent "
                                     "at all three (one).")
    def occ_load(self, task_id: str, mode: str = "joint", turn_timeout_s: float = 1800.0) -> dict:
        return super().occ_load(task_id, mode, turn_timeout_s)
