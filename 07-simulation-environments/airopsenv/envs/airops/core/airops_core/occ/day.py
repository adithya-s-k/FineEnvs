"""One day of the ops control center, pure and deterministic: the turn machine, each desk's decisions taken at the watch
minute, the freeze, the typed requests and their replies, the pages and lookups, check and confirm on the projection,
the FAA's advisories and CDM replies, the grade on the truth, the audit and the record data/get serves.

Each watch at minute t runs a crew turn, an ops turn and, when ops sent crew a request or a text in its turn, a crew
reply turn (one turn in one-agent mode); then the next watch opens: the clock moves an hour and its FAA advisories must
arrive (through the env's trigger on the end_turn that opened it, else the env delivers them) before its first turn is
granted. A desk that resigns or stands down is skipped from then on; once no flight is open the day ends. Every decision
of a turn is decided at t and reaches the truth from t on (sim.Dispatch, crewsim.Pool), so nothing before t changes,
which each commit checks (run.history). A desk acts only on its turn and through its own tools: ops never names crews,
crew never cancels or holds, and a request accepted is the receiver's decision."""

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping

from ..core.known import FREEZE
from ..core.model import hhmm, parse_hhmm
from ..core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from ..programs import crewday, crewplan, crews, crewsim, occ, sim
from . import pages, run
from .pack import World

OPS, CREW, ONE = "ops", "crew", "one"
ROLES = {"ops": OPS, "crew": CREW, "both": ONE}
PREFIXES = {OPS: ("ops_",), CREW: ("crew_",), ONE: ("ops_", "crew_")}
MODES = {"joint": (CREW, OPS), "one": (ONE,), "idle": ()}
OPS_ACTIONS = ("cancel", "cancel_pair", "hold", "release", "claim", "swap_slots")
CREW_ACTIONS = ("assign", "call_reserve", "swap", "deadhead", "rest", "release")
OPS_DOES = ("cancel", "hold", "release", "claim")
ASKS = {CREW: ("HOLD", "CANCEL", "CANCEL_PAIR", "TEXT"), OPS: ("COVER", "TEXT")}
ANSWERS = ("accept", "decline")
TEXT_CHARS = 400
WARN = 30
GRADED = ("done", "stand_down", "cost_cap", "context_full", "turn_limit", "time_limit")
REASONS = (*GRADED, "error", "crash", "timeout", "missing")
TURNS = {"joint": 6, "one": 8, "idle": 0}
EXTRA_TURNS = 10
EARLY = 720
SHARED = ("wait_turn", "end_turn", "resign")
MUTATING = ("ops_confirm", "ops_request", "ops_reply", "crew_confirm", "crew_request", "crew_reply", "end_turn")
ORDER = {"joint": ("crew", "ops", "reply"), "one": ("one",), "idle": ()}


class Refused(ValueError):
    """A change the env won't take, and why."""


def fid(i: int) -> str:
    return f"f{i}"


def flight_id(task, value) -> int:
    text = str(value).strip().lower()
    text = text[1:] if text.startswith("f") else text
    if not text.isdigit() or int(text) not in task.by_id:
        raise Refused(f"unknown flight {json.dumps(value)}: use the ids on the pages (f0, f1, ...)")
    return int(text)


def minute_of(value, what: str = "time") -> int:
    try:
        return parse_hhmm(str(value))
    except ValueError:
        raise Refused(f"{what} {json.dumps(value)} is not a time: write HH:MM, or HH:MM+1 after midnight") from None


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def reward(cost: float, best: float, naive: float) -> float:
    """0.2 + 0.8 exp(-g / 0.5), g the cost over the best plan in units of max(B0 - best, best / 10): 1.0 at the best
    plan or under it, about 0.31 at B0."""
    return 0.2 + 0.8 * math.exp(-max(0.0, cost - best) / max(naive - best, 0.1 * best) / 0.5)


def costs(day: crewsim.CrewDay) -> dict[str, float]:
    c = day.run.cost
    return {"total": round(crewplan.worth(day, False), 1), "delay": c.delay, "cancel": c.cancel, "notice": c.notice,
            "aircraft": c.gate + c.overrun - c.saved + c.ferry, **{k: round(v, 1) for k, v in day.cost.items()}}


def timetable(task) -> dict[int, list[str]]:
    """Each watch's FAA advisories: those signed after the last watch and by it (the first watch's, by the start)."""
    times = occ.watches(task)
    due: dict[int, list[str]] = {k: [] for k in range(len(times))}
    for n in sorted(pages.news(task).values(), key=lambda n: (n["at"], n["id"])):
        k = next((k for k, t in enumerate(times) if t >= n["at"]), None)
        if k is not None:
            due[k].append(n["id"])
    return due


class OccDay:
    def __init__(self, world: World, mode: str = "joint"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        self.w, self.task, self.mode = world, world.task, mode
        self.world = sim.truth(self.task)
        self.known: dict[int, sim.World] = {}
        self.units = {u.id: u for u in world.units}
        self.times = occ.watches(self.task)
        self.news = pages.news(self.task)
        self.due = timetable(self.task)
        self.k = self.seq = self.version = 0
        self.plan: dict = {}
        self.truth = run.truth(world, self.plan, self.world)
        self.projections: dict[tuple, run.Projection] = {}
        self.committed: set[int] = set()
        self.queue: list[str] = list(MODES[mode])
        self.turn: dict | None = None
        self.turns: list[dict] = []
        self.opened_by: dict[int, str] = {0: "load"}
        self.calls: list[dict] = []
        self.refusals: list[dict] = []
        self.changes: list[dict] = []
        self.requests: list[dict] = []
        self.notices: list[dict] = []
        self.pending: list[str] = []
        self.absent: dict[str, dict] = {}
        self.inbox: dict[str, list[dict]] = {d: [] for d in (OPS, CREW, ONE)}
        self.seen: dict[str, set] = {d: set() for d in (OPS, CREW, ONE)}
        self.shown: list[dict] = []
        self.flag_log: list[dict] = []
        self.estimates: list[dict] = []
        self.acausal: list[dict] = []
        self.done, self.grade = False, None
        for n in self.due[0]:
            self.notice(n, "load")
        self._advance("load")

    # ------------------------------------------------------------------ clock and turns

    @property
    def t(self) -> int:
        return self.times[self.k]

    @property
    def desks(self) -> tuple[str, ...]:
        return MODES[self.mode]

    @property
    def line(self) -> int | None:
        """The freeze line: no hold goes earlier; none at the first watch, where the desks set the day up."""
        return None if self.k == 0 else self.t + FREEZE

    @property
    def max_turns(self) -> int:
        return TURNS[self.mode] * len(self.times) + EXTRA_TURNS

    def known_at(self, t: int) -> sim.World:
        if t not in self.known:
            self.known[t] = sim.known(self.task, t)
        return self.known[t]

    def open_flights(self, t: int) -> list[int]:
        return [i for i in self.task.by_id if run.opened(self.truth, i, t) and not self.plan.get(i, {}).get("cancel")]

    def _advance(self, via: str) -> int | None:
        """The next turn: the next desk present in this watch's queue, else the next watch's, none once an absence
        leaves the run unscorable; the watch it opened when that watch awaits its advisories through the trigger on
        this call."""
        opened = None
        while True:
            while self.queue:
                kind = self.queue.pop(0)
                desk = CREW if kind == "reply" else kind
                if desk not in self.absent and not self.unscorable():
                    self.turn = {"n": len(self.turns), "watch": self.k, "t": self.t, "desk": desk, "kind": kind,
                                 "granted": None, "ended": None, "calls": 0}
                    self.turns.append(self.turn)
                    return opened
            self.turn = None
            self.missed()
            self._close()
            if self.k + 1 >= len(self.times) or not self.open_flights(self.times[self.k + 1]):
                self._finish()
                return None
            self.k += 1
            by = via if via == "end_turn" and opened is None else "env"
            opened = self.k if by == "end_turn" else None
            self._open(by)
            self.queue = list(self.desks)

    def _close(self) -> None:
        if self.desks:
            self.estimates.append({"watch": self.k, "t": self.t, **costs(self.project(self.plan).day)})
        self.committed |= run.commits(self.task, self.truth, self.plan, self.t, self.known_at(self.t))

    def _open(self, by: str) -> None:
        self.opened_by[self.k] = by
        for r in self.requests:
            if r["status"] == "open" and (why := self.lapsed(r)):
                self.close(r, "lapsed", why)
        self.pending = list(self.due[self.k])
        if by != "end_turn":
            self.missed()

    def notice(self, news_id: str, via: str) -> dict:
        """One FAA advisory delivered at this watch: by the trigger, or by the env; a second delivery is ignored."""
        if news_id not in self.news:
            raise ValueError(f"unknown news {news_id!r}")
        if any(n["news_id"] == news_id for n in self.notices):
            return {"delivered": news_id, "watch": self.k, "again": True}
        self.seq += 1
        self.notices.append({"news_id": news_id, "watch": self.k, "t": self.t, "via": via, "seq": self.seq})
        if news_id in self.pending:
            self.pending.remove(news_id)
        return {"delivered": news_id, "watch": self.k}

    def missed(self, via: str = "env") -> list[str]:
        """The env delivers the watch's advisories no trigger brought ("late" when it gave up waiting for them)."""
        out = list(self.pending)
        for n in out:
            self.notice(n, via)
        return out

    def unscorable(self) -> bool:
        return any(a["reason"] not in GRADED for a in self.absent.values())

    def grantable(self, desk: str) -> bool:
        return self.turn is not None and self.turn["desk"] == desk and not self.pending

    def whose(self) -> str:
        if self.done:
            return "the day is over"
        return f"the {self.turn['desk']} desk's {self.turn['kind']} turn at watch {self.k} ({hhmm(self.t)})"

    # ------------------------------------------------------------------ calls

    def call(self, role: str | None, tool: str, args: Mapping) -> dict:
        """A desk's tool call, logged with its result's digest: refused off its turn (wait_turn and resign aside), and
        to a desk that is off."""
        desk = ROLES.get(role or "")
        if desk is None or desk not in self.desks:
            raise Refused(f"no desk for role {role!r} in {self.mode} mode")
        if tool not in SHARED and not tool.startswith(PREFIXES[desk]):
            raise Refused(f"{tool} is not a tool of the {desk} desk")
        if tool == "wait_turn":
            return self.wait_turn(desk)
        if tool == "resign":
            return self._logged(desk, tool, args, self.resign(desk, args.get("reason") or "done"))
        if desk in self.absent:
            return {"error": f"the {desk} desk is off ({self.absent[desk]['reason']})", "done": True}
        if self.done:
            return {"error": "the day is over", "done": True}
        if self.turn is None or self.turn["desk"] != desk or self.turn["granted"] is None:
            self.refusals.append({"seq": self.seq, "desk": desk, "tool": tool, "watch": self.k,
                                  "turn": self.whose(), "mutating": tool in MUTATING})
            return {"error": f"not your turn: it is {self.whose()}; call wait_turn"}
        self.turn["calls"] += 1
        turn, k, t = self.turn["n"], self.k, self.t
        try:
            result = getattr(self, tool)(desk, **args)
        except Refused as e:
            result = {"error": str(e)}
        return self._logged(desk, tool, args, result, turn, k, t)

    def _logged(self, desk: str, tool: str, args: Mapping, result: dict, turn: int | None = None,
                k: int | None = None, t: int | None = None) -> dict:
        self.seq += 1
        self.calls.append({"seq": self.seq, "desk": desk, "tool": tool, "watch": self.k if k is None else k,
                           "t": self.t if t is None else t, "turn": turn, "args": dict(args),
                           "result": digest(result)})
        return result

    def wait_turn(self, desk: str) -> dict:
        """The desk's page once its turn is granted, else where the day stands."""
        if desk in self.absent:
            return {"done": True, "absent": self.absent[desk]["reason"]}
        if self.done:
            return {"done": True, "reward": self.grade["reward"]}
        if not self.grantable(desk):
            return {"waiting_for": self.whose()}
        if self.turn["granted"] is None:
            page = {"watch": self.k, "watches": len(self.times), "time": hhmm(self.t), "turn": self.turn["kind"],
                    "page": self.page(desk)}
            self.seq += 1
            self.turn["granted"], self.turn["page"] = self.seq, page
            self.calls.append({"seq": self.seq, "desk": desk, "tool": "wait_turn", "watch": self.k, "t": self.t,
                               "turn": self.turn["n"], "args": {}, "result": digest(page)})
        return self.turn["page"]

    def end_turn(self, desk: str, note: str | None = None, stand_down: bool = False) -> dict:
        turn = self.turn
        turn["ended"], turn["note"] = self.seq + 1, note
        if desk == OPS and CREW not in self.absent and any(r["sender"] == OPS and r["turn"] == turn["n"]
                                                           for r in self.requests):
            self.queue.insert(0, "reply")
        if stand_down:
            self.absent[desk] = {"reason": "stand_down", "watch": self.k, "t": self.t, "seq": self.seq + 1}
        opened = self._advance("end_turn")
        return {"ended": turn["kind"], "watch": turn["watch"], "opened": opened, "done": self.done,
                "next": self.whose()}

    def resign(self, desk: str, reason: str) -> dict:
        if desk not in self.absent:
            self.absent[desk] = {"reason": reason if reason in REASONS else "error", "watch": self.k, "t": self.t,
                                 "seq": self.seq + 1}
            if not self.done and self.turn is not None and (self.turn["desk"] == desk or self.unscorable()):
                self.turn["ended"] = self.seq + 1
                self.turn["note"] = f"resigned: {reason}" if self.turn["desk"] == desk else f"{desk} desk off: {reason}"
                self._advance("resign")
        return {"resigned": desk, "reason": self.absent[desk]["reason"], "done": self.done}

    def time_out(self, desk: str) -> dict:
        """A desk that overran its turn is off from now on; the run can't be scored."""
        return self._logged(desk, "resign", {"reason": "timeout"}, self.resign(desk, "timeout"))

    # ------------------------------------------------------------------ projection and pages

    def project(self, plan: dict) -> run.Projection:
        key = (self.version if plan is self.plan else digest(pages.jsonable(plan)), self.t)
        if key not in self.projections:
            if len(self.projections) > 8:
                self.projections.clear()
            self.projections[key] = run.project(self.w, plan, self.t, self.world, self.known_at(self.t))
        return self.projections[key]

    def page(self, desk: str) -> str:
        proj = self.project(self.plan)
        flags = run.flags(proj, self.task)
        self.flag_log.append({"watch": self.k, "t": self.t, "desk": desk, "flags": {
            i: {k: list(v) for k, v in f.items()} for i, f in flags.items() if any(v[0] != run.LEGAL
                                                                                  for v in f.values())}})
        text = pages.page(self, desk, proj, flags)
        self.shown.append({"watch": self.k, "t": self.t, "desk": desk, "chars": len(text), "text": text})
        return text

    # ------------------------------------------------------------------ the desks' plans

    def gone(self, i: int) -> bool:
        x = self.truth.run.flights[i]
        return x.dep is not None and x.dep < self.t

    def cancelled(self, i: int) -> bool:
        x = self.truth.run.flights[i]
        return bool(self.plan.get(i, {}).get("cancel")) or (x.status != FLOWN and x.decided < self.t)

    def why_not(self, i: int, frozen: bool = True) -> str | None:
        f = self.task.flight(i)
        if self.gone(i):
            return f"{fid(i)} ({f.flight}) left at {hhmm(self.truth.run.flights[i].dep)}"
        if self.cancelled(i):
            return f"{fid(i)} ({f.flight}) is cancelled; cancellations are final"
        if frozen and i in self.committed:
            return f"{fid(i)} ({f.flight}) is committed: it was due out before an earlier freeze line"
        return None

    def when(self, minute: int, i: int) -> int:
        """A time read near the flight's schedule: "01:10" for a flight after midnight is "01:10+1"."""
        due = self.task.flight(i).sched_dep
        while minute < due - EARLY:
            minute += 1440
        return minute

    def ops_entry(self, plan: dict, change: Mapping, frozen: bool = True) -> list[tuple]:
        """One ops change into `plan` at t: (flight, action, detail) for each flight it touches; a CANCEL the crew desk
        asked for and ops accepted passes the freeze (its flight can't be crewed, so it isn't leaving)."""
        t = self.t
        i = flight_id(self.task, change.get("flight"))
        action = change.get("action")
        if action not in OPS_ACTIONS:
            raise Refused(f"{fid(i)}: action must be one of {', '.join(OPS_ACTIONS)}")
        if why := self.why_not(i, frozen):
            raise Refused(why)
        if action in ("cancel", "cancel_pair"):
            pair = [i]
            if action == "cancel_pair":
                j = self.w.partner.get(i)
                if j is None:
                    raise Refused(f"{fid(i)} has no round-trip partner on its planned rotation")
                if why := self.why_not(j, frozen):
                    raise Refused(f"its partner: {why}")
                pair.append(j)
            for j in pair:
                plan.setdefault(j, {}).update(cancel=True, decided=t)
            return [(j, "cancel", None) for j in pair]
        e = plan.setdefault(i, {})
        if action == "hold":
            nb = self.when(minute_of(self.need(change, "time", "a hold")), i)
            floor = t if self.line is None else self.line
            if nb < floor:
                raise Refused(f"{fid(i)}: no hold before {'now' if self.line is None else 'the freeze line'}, "
                              f"{hhmm(floor)}")
            e.setdefault("holds", []).append([t, nb])
            return [(i, "hold", hhmm(nb))]
        if action == "release":
            e.setdefault("holds", []).append([t, None])
            e.setdefault("claims", []).append([t, None])
            return [(i, "release", None)]
        if action == "claim":
            slot = self.when(minute_of(self.need(change, "time", "a claim")), i)
            self.cdm(plan, i, slot)
            e.setdefault("claims", []).append([t, slot])
            return [(i, "claim", hhmm(slot))]
        j = flight_id(self.task, self.need(change, "other", "a swap"))
        if why := self.why_not(j):
            raise Refused(f"the other flight: {why}")
        now = self.project(self.plan).day.run.flights
        a, b = now[i], now[j]
        if a.slot is None or b.slot is None or a.program != b.program:
            raise Refused(f"FAA CDM: to swap slots both flights hold one in the same program ({fid(i)}: "
                          f"{pages.slot_text(a)}; {fid(j)}: {pages.slot_text(b)})")
        e.setdefault("claims", []).append([t, b.slot])
        plan.setdefault(j, {}).setdefault("claims", []).append([t, a.slot])
        return [(i, "claim", hhmm(b.slot)), (j, "claim", hhmm(a.slot))]

    @staticmethod
    def need(row: Mapping, key: str, what: str):
        if row.get(key) is None:
            raise Refused(f"{what} needs {'a' if key != 'other' else 'the'} {key}")
        return row[key]

    def cdm(self, plan: dict, i: int, slot: int) -> None:
        """FAA CDM's check of a claim on the programs as issued: a slot of the pool that governs the flight leaving at
        it less its scheduled block, used by no flight gone and claimed by no other."""
        f = self.task.flight(i)
        seg = sim.Rules(self.task, self.known_at(self.t)).governs(f, slot - f.sched_block)
        if seg is None:
            raise Refused(f"FAA CDM: {fid(i)} leaving at {hhmm(slot - f.sched_block)} is in no program's pool")
        if slot not in seg.slots:
            near = [hhmm(s) for s in seg.slots if abs(s - slot) <= 60]
            raise Refused(f"FAA CDM: {hhmm(slot)} is not one of JetBlue's slots in {seg.id}"
                          + (f" (near it: {' '.join(near)})" if near else ""))
        if any(self.gone(j) and (x.program, x.slot) == (seg.id, slot) for j, x in self.truth.run.flights.items()):
            raise Refused(f"FAA CDM: slot {hhmm(slot)} of {seg.id} went to a flight that left")
        for j, e in plan.items():
            if j != i and isinstance(j, int) and not self.gone(j) and not self.cancelled(j) and \
                    sim.claim_at(e, self.t) == slot:
                raise Refused(f"FAA CDM: slot {hhmm(slot)} is claimed by {fid(j)}")

    def unit(self, uid) -> crews.Unit:
        u = self.units.get(str(uid))
        if u is None:
            raise Refused(f"unknown unit {json.dumps(uid)}: use the ids on the pages")
        return u

    def crew_entry(self, plan: dict, move: Mapping) -> list[tuple]:
        """One crew move into `plan` at t: (flight or None, action, unit) for each flight it touches."""
        t = self.t
        action = move.get("action")
        if action not in CREW_ACTIONS:
            raise Refused(f"action must be one of {', '.join(CREW_ACTIONS)}")
        u = self.unit(move.get("unit"))
        now = self.project(self.plan)
        state = now.units[u.id]
        i = None
        if move.get("flight") is not None:
            i = flight_id(self.task, move["flight"])
            f = self.task.flight(i)
            if why := self.why_not(i, False):
                raise Refused(why)
            if action in ("assign", "call_reserve") and crews.TYPE[f.family] != u.group:
                raise Refused(f"{u.id} flies the {u.group} group; {fid(i)} is an {f.family}")
        if action in ("assign", "deadhead") and i is None:
            raise Refused(f"{action} needs a flight")
        if action == "assign":
            self.name(plan, i, u)
            return [(i, "assign", u.id)]
        if action == "deadhead":
            plan.setdefault(i, {}).setdefault("rides", []).append([t, None, u.id])
            return [(i, "deadhead", u.id)]
        if action == "release":
            return self.release(plan, u, i, state)
        if action == "call_reserve":
            free = t == self.task.start or (state["report"] is None and state["held"] is None)
            if u.reserve != "short" or not free or any(c[1] == u.id for c in plan.get("calls", ())):
                raise Refused(f"{u.id} is not a short-call reserve free to call (the crew system may have called it: "
                              "assign it instead)")
            first = crews.on_hour(max(t, u.rap, self.task.start) + u.lead)
            if move.get("time") is not None:
                r = minute_of(move["time"])
                r = r if r >= first - EARLY else r + 1440
            elif i is not None:
                r = max(first, ((now.day.run.flights[i].dep or self.task.flight(i).sched_dep) - crews.BRIEF) // 60 * 60)
            else:
                r = first
            if r % 60 or r < first or r - u.lead > u.until:
                raise Refused(f"{u.id} reports on a whole hour from {hhmm(first)}, called by {hhmm(u.until)}: not "
                              f"{hhmm(r)}")
            plan.setdefault("calls", []).append([t, u.id, r])
            if i is not None:
                self.name(plan, i, u)
            return [(i, "call_reserve", u.id)]
        if action == "rest":
            if state["period"] != 1 or (u.reserve == "short" and state["report"] is None and t > self.task.start):
                raise Refused(f"{u.id} can't start a second duty period today")
            first = crews.on_hour(t if state["release"] is None else max(state["release"] + crews.rest(u.kind), t))
            if t == self.task.start:
                first = crews.on_hour(t)
            r = first if move.get("time") is None else minute_of(move["time"])
            r = r if r >= first - EARLY else r + 1440
            if r % 60 or r < first:
                raise Refused(f"{u.id} is rested from {hhmm(first)}: report on a whole hour from then")
            plan.setdefault("calls", []).append([t, f"{u.id}#2", r])
            return [(None, "rest", u.id)]
        v = self.unit(self.need(move, "other", "a swap"))
        other = now.units[v.id]
        if (v.kind, v.group) != (u.kind, u.group) or other["station"] != state["station"] or v.id == u.id:
            raise Refused(f"a swap takes two units of one kind and group at one station: {u.id} at "
                          f"{state['station']}, {v.id} at {other['station']}")
        plan.setdefault("swaps", []).append([t, u.id, v.id])
        out = [(j, "swap", b.id) for a, b in ((state, v), (other, u)) for j in a["trip"]
               if not (self.gone(j) or self.cancelled(j))]
        return out or [(None, "swap", u.id)]

    def release(self, plan: dict, u: crews.Unit, i: int | None, state: Mapping) -> list[tuple]:
        """Lets go of u's moves: on a flight its name or ride there (and the call that came with it), with none its
        rest, call and swaps; undone as if never made while nothing they changed has happened yet (always within the
        turn), else a name or ride ends now. With a flight and none of those, u is kept off a ride its trip plans."""
        t = self.t
        if i is None:
            moves = [(("calls",), x) for x in plan.get("calls", ()) if x[1] in (u.id, f"{u.id}#2")] + \
                [(("swaps",), x) for x in plan.get("swaps", ()) if u.id in x[1:]]
        else:
            moves = [((i, k), x) for k in ("names", "rides") for x in plan.get(i, {}).get(k, ())
                     if x[-1] == u.id and x[1] is None]
            moves += [(("calls",), x) for x in plan.get("calls", ()) if x[1] == u.id and moves and
                      x[0] in {m[0] for _, m in moves}]
        if not moves:
            if i is not None and i in state["deadheads"] and [u.id, i] not in [k[1:] for k in plan.get("keeps", ())]:
                plan.setdefault("keeps", []).append([t, u.id, i])
                return [(i, "release", u.id)]
            raise Refused(f"{u.id} has no move of yours to release{' on ' + fid(i) if i is not None else ''} and "
                          "doesn't ride it as planned: release lets go of a unit you assigned or sent to ride (with no "
                          "flight: its rest, call or swap), or keeps one off a ride its trip plans")

        def drop(p: dict) -> None:
            for path, x in moves:
                box = p[path[0]] if len(path) == 1 else p[path[0]][path[1]]
                box.remove(x)
        trial = copy.deepcopy(plan)
        drop(trial)
        if all(x[0] == t for _, x in moves) or run.acted(run.truth(self.w, trial, self.world), t) == \
                run.acted(self.truth, t):
            drop(plan)
        elif i is None:
            raise Refused(f"{u.id}'s rest, call or swap has already acted: it can't be undone now")
        else:
            for path, x in moves:
                if len(path) == 2:
                    x[1] = t
        return [(i, "release", u.id)]

    def name(self, plan: dict, i: int, u: crews.Unit) -> None:
        e = plan.setdefault(i, {})
        for x in list(e.get("names", ())):
            if x[2] == u.kind and x[1] is None:
                if x[0] == self.t:
                    e["names"].remove(x)
                else:
                    x[1] = self.t
        e.setdefault("names", []).append([self.t, None, u.kind, u.id])

    def changed(self, rows, apply: Callable) -> tuple[dict, list[dict], list[tuple]]:
        """The rows applied in order on a copy of the plan: the plan, the rows refused and why, what each touched."""
        if not isinstance(rows, list):
            raise Refused("send a list")
        plan, refused, touched = copy.deepcopy(self.plan), [], []
        for row in rows:
            if not isinstance(row, Mapping):
                refused.append({"change": row, "reason": "not an object"})
                continue
            trial = copy.deepcopy(plan)
            try:
                touched += [(row.get("ref"), *x) for x in apply(trial, row)]
                plan = trial
            except Refused as e:
                refused.append({"change": dict(row), "reason": str(e)})
        return plan, refused, touched

    def review(self, plan: dict) -> dict:
        """A plan on the projection against the current one: its costs and their change, the new automatic
        cancellations, every flight it moves by more than WARN minutes or cancels (warnings), the flights it moves or
        crews otherwise (by impact) and the crew flags it changes."""
        now, then = self.project(self.plan), self.project(plan)
        a, b = costs(now.day), costs(then.day)
        moved, warnings = [], []
        for i in sorted(then.day.run.flights):
            x, y = now.day.run.flights[i], then.day.run.flights[i]
            if x.status != y.status and y.status != FLOWN:
                warnings.append(f"{fid(i)} {pages.proj_text(y, self.task.flight(i))}")
            elif x.dep is not None and y.dep is not None and abs(y.dep - x.dep) > WARN:
                warnings.append(f"{fid(i)} leaves {y.dep - x.dep:+d} min")
            cx, cy = now.day.crews.get(i, {}), then.day.crews.get(i, {})
            if (x.status, x.dep, x.arr) != (y.status, y.dep, y.arr) or cx != cy:
                f = self.task.flight(i)
                impact = f.pax * (abs((y.arr or 0) - (x.arr or 0)) + 60 * (x.status != y.status)) + (cx != cy)
                moved.append((-impact, i, f"{fid(i)} {pages.proj_text(y, f)}{pages.crew_text(cy)} (was "
                                          f"{pages.proj_text(x, f)}{pages.crew_text(cx)})"))
        fa, fb = run.flags(now, self.task), run.flags(then, self.task)
        flags = [f"{fid(i)} {pages.flag_text(fb[i])} (was {pages.flag_text(fa.get(i, {}))})" for i in sorted(fb)
                 if fb[i] != fa.get(i)]
        return {"cost": b, "delta": {k: round(b[k] - a[k], 1) for k in b},
                "auto_cancelled": [fid(i) for i, x in then.day.run.flights.items() if x.status == AUTO_CANCELLED
                                   and now.day.run.flights[i].status != AUTO_CANCELLED],
                "warnings": warnings[:20] + ([f"{len(warnings) - 20} more"] if len(warnings) > 20 else []),
                "flights": [m for *_, m in sorted(moved)[:12]] + ([f"{len(moved) - 12} more"] if len(moved) > 12
                                                                   else []),
                "flags": flags[:12] + ([f"{len(flags) - 12} more"] if len(flags) > 12 else [])}

    def commit(self, desk: str, plan: dict, touched: list[tuple], via: str) -> str | None:
        """The plan in force from t, the truth played again; refused if anything before t would change."""
        truth = run.truth(self.w, plan, self.world)
        if run.acted(truth, self.t) != run.acted(self.truth, self.t):
            self.acausal.append({"watch": self.k, "t": self.t, "seq": self.seq + 1, "desk": desk, "via": via})
            return f"these changes would change what happened before {hhmm(self.t)}: nothing was committed"
        self.plan, self.truth = plan, truth
        self.version += 1
        for ref, target, action, detail in touched:
            self.changes.append({"seq": self.seq + 1, "turn": self.turn["n"], "desk": desk, "watch": self.k,
                                 "t": self.t, "target": target, "action": action, "detail": detail, "ref": ref,
                                 "via": via})
        return None

    def confirmed(self, desk: str, rows, apply: Callable, extra: Callable | None = None) -> dict:
        plan, refused, touched = self.changed(rows, apply)
        review = {**self.review(plan), **(extra(plan, touched) if extra else {})}
        if touched and (why := self.commit(desk, plan, touched, "confirm")):
            return {"error": why, "refused": refused}
        return {"confirmed": [f"{fid(i) + ' ' if i is not None else ''}{a}{' ' + d if d else ''}"
                              for _, i, a, d in touched], **review, "refused": refused}

    # ------------------------------------------------------------------ ops tools

    def ops_flights(self, desk: str, ids, station, status, start, end) -> dict:
        return pages.flights(self, ids, station, status, start, end, False)

    def ops_program(self, desk: str, hub) -> dict:
        return pages.program(self, hub)

    def ops_aircraft(self, desk: str, station) -> dict:
        return pages.aircraft(self, station)

    def ops_check(self, desk: str, changes) -> dict:
        plan, refused, _ = self.changed(changes, self.ops_entry)
        return {**self.review(plan), "refused": refused}

    def ops_confirm(self, desk: str, changes) -> dict:
        return self.confirmed(desk, changes, self.ops_entry)

    # ------------------------------------------------------------------ crew tools

    def crew_flights(self, desk: str, ids, station, status, start, end) -> dict:
        return pages.flights(self, ids, station, status, start, end, True)

    def crew_crews(self, desk: str, unit, station, base, kind, group, state) -> dict:
        return pages.crew_list(self, unit, station, base, kind, group, state)

    def crew_program(self, desk: str, hub) -> dict:
        return pages.program(self, hub)

    def crew_reserves(self, desk: str, base) -> dict:
        return pages.reserves(self, base)

    def crew_options(self, desk: str, flight, kind) -> dict:
        return pages.options(self, flight_id(self.task, flight), kind)

    def crew_check(self, desk: str, moves) -> dict:
        plan, refused, touched = self.changed(moves, self.crew_entry)
        return {**self.review(plan), **pages.legality(self, plan, moves, touched), "refused": refused}

    def crew_confirm(self, desk: str, moves) -> dict:
        return self.confirmed(desk, moves, self.crew_entry,
                              lambda plan, touched: pages.legality(self, plan, moves, touched))

    # ------------------------------------------------------------------ requests

    def ops_request(self, desk: str, type, flight, time, text) -> dict:
        return self.ask(desk, CREW, type, flight, time, text)

    def crew_request(self, desk: str, type, flight, time, text) -> dict:
        return self.ask(desk, OPS, type, flight, time, text)

    def ops_reply(self, desk: str, request, answer, text) -> dict:
        return self.answer(desk, request, answer, text)

    def crew_reply(self, desk: str, request, answer, text) -> dict:
        return self.answer(desk, request, answer, text)

    def ask(self, desk: str, to: str, type, flight, time, text) -> dict:
        if self.mode == "one":
            raise Refused("one desk runs both jobs: there is no one to ask")
        if type not in ASKS[desk]:
            raise Refused(f"the {desk} desk asks {', '.join(ASKS[desk])}")
        if text is not None and len(str(text)) > TEXT_CHARS:
            raise Refused(f"a text is at most {TEXT_CHARS} characters")
        params: dict = {}
        if flight is not None:
            i = params["flight"] = flight_id(self.task, flight)
            if type != "TEXT" and (why := self.why_not(i, False)):
                raise Refused(why)
        if type == "TEXT" and not text:
            raise Refused("a TEXT needs its text")
        if type != "TEXT" and flight is None:
            raise Refused(f"a {type} needs a flight")
        if type == "CANCEL_PAIR" and self.w.partner.get(params["flight"]) is None:
            raise Refused(f"{fid(params['flight'])} has no round-trip partner on its planned rotation")
        if type in ("HOLD", "COVER"):
            params["time"] = self.when(minute_of(self.need({"time": time}, "time", f"a {type}")), params["flight"])
        rid = f"R{len(self.requests) + 1}"
        if type != "TEXT":
            for r in self.requests:
                if r["status"] == "open" and r["params"].get("flight") == params["flight"]:
                    self.close(r, "superseded", f"superseded by {rid}")
        else:
            self.inbox[to].append({"request": rid, "status": f"text from {desk}" + (
                f" on {fid(params['flight'])}" if "flight" in params else ""), "text": text})
        self.requests.append({"id": rid, "sender": desk, "receiver": to, "type": type, "params": params,
                              "text": text or "", "sent": self.t, "watch": self.k, "turn": self.turn["n"],
                              "status": "delivered" if type == "TEXT" else "open", "reply": None, "replied": None,
                              "reply_text": None, "actions": []})
        return {"request": rid, "to": to, "type": type}

    def close(self, r: dict, status: str, why: str) -> None:
        r.update(status=status, replied=self.t, reply_text=why)
        self.inbox[r["sender"]].append({"request": r["id"], "status": status, "why": why})

    def answer(self, desk: str, request, answer, text) -> dict:
        r = next((r for r in self.requests if r["id"] == str(request)), None)
        if r is None or r["receiver"] != desk:
            raise Refused(f"no request {json.dumps(request)} to the {desk} desk")
        if r["status"] != "open":
            raise Refused(f"{r['id']} is {r['status']}")
        if answer not in ANSWERS:
            raise Refused("answer accept or decline")
        if text is not None and len(str(text)) > TEXT_CHARS:
            raise Refused(f"a text is at most {TEXT_CHARS} characters")
        out = {"request": r["id"], "answer": answer}
        if answer == "accept" and r["type"] in ("HOLD", "CANCEL", "CANCEL_PAIR"):
            if why := self.lapsed(r):
                self.close(r, "lapsed", why)
                return {**out, "lapsed": why}
            i = r["params"]["flight"]
            row = {"flight": fid(i), "action": {"HOLD": "hold", "CANCEL": "cancel", "CANCEL_PAIR": "cancel_pair"}[
                r["type"]], "time": None, "other": None, "ref": r["id"]}
            if r["type"] == "HOLD":
                row["time"] = hhmm(max(r["params"]["time"], self.line or self.t))
            plan, refused, touched = self.changed([row], lambda p, c: self.ops_entry(p, c, frozen=False))
            if refused:
                self.close(r, "lapsed", refused[0]["reason"])
                return {**out, "lapsed": refused[0]["reason"]}
            review = self.review(plan)
            if why := self.commit(desk, plan, touched, "accept"):
                return {**out, "error": why}
            r["actions"] += [{"desk": desk, "action": a, "target": j, "minute": self.t, "link": "accept"}
                             for _, j, a, _ in touched]
            out |= review
        r.update(status="accepted" if answer == "accept" else "declined", reply=answer, replied=self.t,
                 reply_text=text or "")
        self.inbox[r["sender"]].append({"request": r["id"], "status": r["status"], "text": text or ""})
        return out

    def lapsed(self, r: dict) -> str | None:
        """Why an open request no longer stands: its flight gone or cancelled, a HOLD's flight committed, or its time
        passed; else None."""
        i, when = r["params"].get("flight"), r["params"].get("time")
        if i is None or r["type"] == "TEXT":
            return None
        if self.gone(i) or self.cancelled(i):
            return self.why_not(i, False)
        if r["type"] == "HOLD" and i in self.committed:
            return self.why_not(i)
        if when is not None and when < self.t:
            return f"{hhmm(when)} has passed"
        return None

    def links(self) -> None:
        """Each request's follow-on actions: by reference, and the receiver's on the same flight in its next turn."""
        for r in self.requests:
            i = r["params"].get("flight")
            nxt = next((t["n"] for t in self.turns if t["desk"] == r["receiver"] and t["n"] > r["turn"]), None)
            have = {(a["action"], a["target"], a["minute"]) for a in r["actions"]}
            for c in self.changes:
                if (c["action"], c["target"], c["t"]) in have:
                    continue
                if c["ref"] == r["id"]:
                    link = "ref"
                elif i is not None and c["target"] == i and c["desk"] == r["receiver"] and c["turn"] == nxt:
                    link = "same flight"
                else:
                    continue
                r["actions"].append({"desk": c["desk"], "action": c["action"], "target": c["target"],
                                     "minute": c["t"], "link": link})

    # ------------------------------------------------------------------ the end

    def end(self) -> dict:
        """end-day: whatever the desks left runs on its own to the end of the day, graded."""
        if not self.done:
            for d in self.desks:
                if d not in self.absent:
                    self.absent[d] = {"reason": "missing", "watch": self.k, "t": self.t, "seq": self.seq}
            if self.turn is not None:
                self.turn["ended"] = self.seq
            self.queue = []
            self._advance("env")
        return {"end_reason": self.grade["end_reason"], "cost": self.grade["cost"], "reward": self.grade["reward"],
                "audit": self.audit()}

    def _finish(self) -> None:
        for k in range(len(self.times)):
            for n in self.due[k]:
                self.notice(n, "end")
        for r in self.requests:
            if r["status"] == "open":
                r.update(status="expired", reply_text="open at the end of the day")
        self.done, self.turn, self.done_seq = True, None, self.seq
        self.links()
        day, refs = self.truth, self.w.refs
        c = crewplan.worth(day, False)
        best, bound, b0 = refs["best"], refs["bound"], refs["B0"]["notice"]
        script = refs.get("scripted", {}).get(self.mode)
        present = [d for d in self.desks if d not in self.absent or self.absent[d]["reason"] in GRADED]
        self.grade = {"end_reason": "done" if len(present) == len(self.desks) else "absent", "cost": round(c, 1),
                      "flat": round(crewplan.worth(day, True), 1), "reward": round(reward(c, best, b0), 6),
                      "reward_L": round(reward(c, bound, b0), 6), "best": best, "bound": bound,
                      "gap_pct": refs["gap_pct"], "B0": b0, "over_best_pct": round(100 * (c / best - 1), 4),
                      "over_bound_pct": round(100 * (c / bound - 1), 4),
                      "avoidable_share": None if b0 <= best else round((b0 - c) / (b0 - best), 4),
                      "above_script": None if script is None or script <= best else round((script - c) /
                                                                                           (script - best), 4),
                      "script": script, "below_best": c < best - 0.5, "costs": costs(day), "counts": self.counts(),
                      "executed": day.run.rows()}

    def counts(self) -> dict:
        flights = self.truth.run.flights
        cause = {c["target"]: c["via"] for c in self.changes if c["action"] == "cancel"}
        return {"cancelled": dict(Counter(cause.get(i, "?") for i, x in flights.items() if x.status == CANCELLED)),
                "auto_cancelled": sum(x.status == AUTO_CANCELLED for x in flights.values()),
                "holds": sum(c["action"] == "hold" for c in self.changes),
                "claims": sum(c["action"] == "claim" for c in self.changes),
                "crew_moves": dict(Counter(c["action"] for c in self.changes if c["desk"] in (CREW, ONE)
                                           and c["action"] in CREW_ACTIONS)),
                "requests": dict(Counter(f"{r['type']} {r['status']}" for r in self.requests)),
                "alerts": dict(Counter(m["type"] for m in self.truth.messages)),
                "latency": [r["replied"] - r["sent"] for r in self.requests if r["reply"] is not None]}

    def audit(self) -> dict:
        """A1 news, A2 turns, A3 causality, A4 roles, A5 legality, A6 the bound, A7 presence."""
        p: dict[str, list[str]] = {f"A{k}": [] for k in range(1, 8)}
        first = {}
        for turn in self.turns:
            if turn["granted"] is not None:
                first.setdefault(turn["watch"], turn["granted"])
        for n, k in ((n, k) for k, ids in self.due.items() for n in ids):
            got = [x for x in self.notices if x["news_id"] == n]
            if len(got) != 1:
                p["A1"].append(f"{n} delivered {len(got)} times")
            elif k in first and (got[0]["watch"] != k or got[0]["seq"] > first[k]):
                p["A1"].append(f"{n} reached watch {k} after its first turn")
            elif got[0]["via"] == "late":
                p["A1"].append(f"{n}: watch {k}'s trigger never brought it")
        for k in sorted({t["watch"] for t in self.turns}):
            kinds = [t["kind"] for t in self.turns if t["watch"] == k]
            order = [x for x in ORDER[self.mode] if x in kinds]
            if kinds != order:
                p["A2"].append(f"watch {k}: turns {kinds}")
        p["A2"] += [f"call {c['seq']} ({c['tool']}) outside its desk's turn" for c in self.calls
                    if c["tool"] not in ("resign", "wait_turn") and (c["turn"] is None or
                                                                     self.turns[c["turn"]]["desk"] != c["desk"])]
        for k, e in self.plan.items():
            if k not in sim.EXTRA and set(e) - {"cancel", "decided", "holds", "claims", "names", "rides"}:
                p["A4"].append(f"{fid(k)}: {sorted(e)}")
        for c in self.changes:
            if c["desk"] == OPS and c["action"] not in OPS_DOES:
                p["A4"].append(f"ops {c['action']} on {c['target']}")
            if c["desk"] == CREW and c["action"] not in CREW_ACTIONS:
                p["A4"].append(f"crew {c['action']} on {c['target']}")
        if self.done:
            day = self.truth
            c = crewplan.worth(day, False)
            if day.uncrewed:
                p["A5"].append(f"flown without a legal crew: {list(day.uncrewed)[:10]}")
            named = crewday.named(day)
            replay = crewsim.execute(self.task, named, self.w.units, self.world, self.w.local, fill=False,
                                     airports=self.w.airports)
            if abs(crewplan.worth(replay, False) - c) > 0.5:
                p["A5"].append(f"the named replay costs {crewplan.worth(replay, False):.1f}, not {c:.1f}")
            deps = {i: x.dep for i, x in day.run.flights.items() if x.status == FLOWN}
            p["A5"] += crewsim.audit(self.task, self.w.units, named, deps, {i: day.run.flights[i].arr - d
                                                                            for i, d in deps.items()},
                                     self.w.local, self.w.airports)[:10]
            if c < self.w.refs["bound"] - 0.5:
                p["A6"].append(f"cost {c:.1f} is under the bound {self.w.refs['bound']}")
        p["A7"] += [f"{d}: {a['reason']}" for d, a in sorted(self.absent.items()) if a["reason"] not in GRADED]
        return {"ok": not any(p.values()), "problems": {k: v for k, v in p.items() if v},
                "refused_off_turn": len(self.refusals), "acausal_refused": len(self.acausal)}

    def digest(self) -> str:
        """The run: every call and its result, the plan, the requests and, once done, the day as it happened."""
        calls = [c for c in self.calls if not self.done or c["seq"] <= self.done_seq]
        body = [[[c["desk"], c["tool"], c["args"], c["result"]] for c in calls], pages.jsonable(self.plan),
                self.requests]
        if self.done:
            body += [self.truth.run.rows(), sorted(self.truth.crews.items()), sorted(self.truth.riders.items())]
        return digest(body)

    def data(self) -> dict:
        return {"task_id": self.task.task_id, "mode": self.mode, "watches": len(self.times), "watch": self.k,
                "t": self.t, "done": self.done, "desks": list(self.desks), "absent": self.absent,
                "turns": [{k: v for k, v in t.items() if k != "page"} for t in self.turns], "calls": self.calls,
                "refusals": self.refusals, "plan": pages.jsonable(self.plan), "changes": self.changes,
                "requests": self.requests, "notices": self.notices, "pages": self.shown, "flags": self.flag_log,
                "estimates": self.estimates, "committed": sorted(self.committed), "acausal": self.acausal,
                "crews": self.truth.units if self.done else None,
                "status": crewsim.statuses(self.truth) if self.done else None,
                "audit": self.audit() if self.done else None, "grade": self.grade,
                "reward": self.grade["reward"] if self.grade else None,
                "digest": self.digest() if self.done else None}
