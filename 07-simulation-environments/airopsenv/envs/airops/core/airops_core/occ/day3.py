"""One day of the ops control center with passengers (v3, airops-occ3): v2's day (day.py) on a b6-v3 world, whose
passengers fly on every day it plays, so every day costs P, with a third desk, the customer desk.

Each watch at minute t runs a crew turn, a customer turn and an ops turn; then a crew reply turn if ops sent crew a
request or a text in its turn, and a customer reply turn if it sent the customer desk a text. One agent may run all
three desks (one turn a watch, no one to ask). The customer desk reads the passengers on the projection (customer.py)
and asks ops: HOLD (a departure until a time, for connecting passengers), PROTECT (don't cancel a flight),
SUGGEST_CANCEL (cancel one) or TEXT. An accepted HOLD is ops's hold, as a crew HOLD is; PROTECT and SUGGEST_CANCEL are
advice, which ops carries out, if at all, with its own tools. Rebooking stays the passenger system's, automatic
(pax.fly: earliest arrival first, first come first served): no desk rebooks. A new request on a flight replaces an older
one from the customer desk, or one between ops and crew, never one across the two. The grade reads the pack's references
under P: the stage-2 reference (best), the standard crew desk with aircraft doing nothing (B0) and the capacity-aware
bound."""

from ..programs import pax
from . import customer, pages
from .day import ASKS, CREW, ONE, OPS, OccDay, Refused, digest, flight_id
from .pack import World

CUSTOMER = "customer"
ROLES = {"ops": OPS, "crew": CREW, "customer": CUSTOMER, "both": ONE}
PREFIXES = {OPS: ("ops_",), CREW: ("crew_",), CUSTOMER: ("cust_",), ONE: ("ops_", "crew_", "cust_")}
MODES = {"joint": (CREW, CUSTOMER, OPS), "one": (ONE,), "idle": ()}
ORDER = {"joint": ("crew", "customer", "ops", "reply", "creply"), "one": ("one",), "idle": ()}
ASKS3 = {CREW: ASKS[CREW], OPS: ASKS[OPS], CUSTOMER: ("HOLD", "PROTECT", "SUGGEST_CANCEL", "TEXT")}
TURNS = {"joint": 6, "one": 10, "idle": 0}
TO_CUSTOMER = ("TEXT",)


class OccDay3(OccDay):
    ROLES, PREFIXES, MODES, ORDER, ASKS, TURNS = ROLES, PREFIXES, MODES, ORDER, ASKS3, TURNS
    DESKS = (OPS, CREW, CUSTOMER, ONE)
    REPLIES = {"reply": CREW, "creply": CUSTOMER}

    def __init__(self, world: World, mode: str = "joint"):
        if world.passengers is None:
            raise ValueError("a day with passengers needs a b6-v3 world")
        self.book = customer.Bookings(world.task, world.passengers)
        super().__init__(world, mode)

    def compose(self, desk: str, proj, flags) -> str:
        parts = [pages.header(self, desk)]
        if desk in (OPS, ONE):
            parts.append(pages.cost_line(proj, "ops"))
        if desk in (CREW, ONE):
            parts.append(pages.cost_line(proj, "crew"))
        if desk in (CUSTOMER, ONE):
            parts.append(customer.cost_line(proj))
        parts.append(pages.messages(self, desk))
        if desk in (OPS, ONE):
            parts += [pages.attention(self, proj, flags), pages.programs(self, proj), pages.spares(self, proj)]
        if desk in (CREW, ONE):
            parts += [pages.alerts(self, desk, proj, flags), pages.near(self, proj), pages.reserve_line(self, proj)]
        if desk in (CUSTOMER, ONE):
            parts += customer.page(self, proj, flags)
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------ requests

    def supersedes(self, desk: str, r: dict) -> bool:
        return (desk == CUSTOMER) == (r["sender"] == CUSTOMER)

    def ask(self, desk: str, to: str, type, flight, time, text) -> dict:
        if self.mode == "one":
            raise Refused("one desk runs every job: there is no one to ask")
        out = super().ask(desk, to, type, flight, time, text)
        r = self.requests[-1]
        if (n := self.about(r)) is not None:
            r["pax"] = n
            out["pax"] = n
        return out

    def about(self, r: dict) -> int | None:
        """The passengers a customer request is about: a HOLD's, the connecting passengers who miss the departure as
        projected and make it at the hold; PROTECT's and SUGGEST_CANCEL's, the flight's booked."""
        if r["sender"] != CUSTOMER or "flight" not in r["params"]:
            return None
        j = r["params"]["flight"]
        if r["type"] != "HOLD":
            return self.book.booked[j] if r["type"] != "TEXT" else None
        runs = self.project(self.plan).day.run.flights
        x, gap = runs[j], pax.mct(self.task.flight(j).orig)
        if x.dep is None:
            return 0
        return sum(n for i, n in self.book.into[j].items() if runs[i].dep is not None and
                   x.dep < runs[i].arr + gap <= r["params"]["time"])

    def ops_request(self, desk: str, type, flight, time, text, to=CREW) -> dict:
        if to not in (CREW, CUSTOMER):
            raise Refused("ops asks the crew desk or the customer desk")
        if to == CUSTOMER and type not in TO_CUSTOMER:
            raise Refused("ops tells the customer desk by TEXT")
        return self.ask(desk, to, type, flight, time, text)

    # ------------------------------------------------------------------ customer tools

    def cust_flights(self, desk: str, ids, station, status, start, end) -> dict:
        return customer.flights(self, ids, station, status, start, end)

    def cust_flight(self, desk: str, flight) -> dict:
        return customer.flight(self, flight_id(self.task, flight))

    def cust_request(self, desk: str, type, flight, until, note) -> dict:
        return self.ask(desk, OPS, type, flight, until, note)

    # ------------------------------------------------------------------ the end

    def _finish(self) -> None:
        super()._finish()
        d = self.truth.run.pax
        self.grade["passengers"] = {**d.outcomes(), "misconnected": d.misconnected(), "delay": d.delay,
                                    "lodging": d.lodging, "fare": d.fare}

    def audit(self) -> dict:
        """v2's A1-A7, A5 replaying the passengers too: the pass played again over the day's flights and riders
        costs what the grade read."""
        out = super().audit()
        if self.done:
            day = self.truth
            again = pax.fly(self.task, self.w.passengers, day.run.flights,
                            {i: sum(self.units[u].size for u in us) for i, us in day.riders.items() if us})
            if (again.delay, again.lodging, again.fare) != (day.run.pax.delay, day.run.pax.lodging, day.run.pax.fare):
                out["problems"].setdefault("A5", []).append("the passengers played again cost otherwise")
                out["ok"] = False
        return out

    def digest(self) -> str:
        """v2's digest of the run, with the passengers' day once done."""
        body = [super().digest()]
        if self.done:
            body.append(self.truth.run.pax.to_dict())
        return digest(body)

    def data(self) -> dict:
        out = super().data()
        out["passengers"] = self.truth.run.pax.to_dict() if self.done else None
        return out
