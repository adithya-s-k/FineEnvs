"""PortSim lines: the wind week with the shipping lines answering back. Each ship's agent, the firm that deals with the
port for its line, reports the ship's ETA 48 and 24 hours out and calls in an hour before it arrives; the published
schedule is the same estimate from 72 hours out, and the ETAs are optimistic and firm up as the ship nears. The
planner can ask a firm for a ship's ETA and request an arrival time, which the firm agrees when the ship can make it
by slowing down, or answers with the earliest or latest hour it can make. Until the port's news about a delayed ship,
its agent works from the plan as it stood; the news keeps an earlier agreement only if the ship can still make it. A
ship that gets in after its window starts is berthed by the port at the next hour a free stretch of quay fits it,
nobody else moving; one that gets in early waits at anchor for its window. Ships berth first come, first served: none
may start while a ship earlier in the queue, with an agreed arrival, waits at anchor, unless the wind holds that ship;
a ship's turn is its arrival at cruising speed, so a ship slowed on request keeps it. An hour at anchor costs what an
hour late does.

``rules["lines"]`` holds each ship's firm, its previous port's distance, its ETA bias, its scheduled arrival, its
arrival before its news (for a delayed ship) and its v4 arrival, from which the news is timed; the truth is each
ship's ``arrival`` (at cruising speed) and the arrival times agreed during the week."""

import json
import math
import re
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from berth_core import Plan, Task, TaskPack
from berth_core.check import PlanResult
from berth_core.model import MOVE_PENALTY

from . import marine, packs, wind, world
from .schedule import Notice, Watch, label
from .schedule import schedule as v4_schedule
from .wind import WindWeek

CRUISE_KN = 17
SLOWEST_KN = 10
STRETCH = CRUISE_KN / SLOWEST_KN
REPORT_LEADS = (48, 24)
CALL_LEAD = 1
ETA_ERROR = 0.2
ETA_HORIZON = 72
COMMIT_LEAD = 48
DISPATCH_HOURS = 168
SPEAKER = "Ship's agent"
REPORTS = ("eta", "call")
NEWS_ETA = ("late", "bunching", "extra")
REFERENCES = packs.references("lines")


def _built(path: Path) -> Path:
    """The lines weeks derive from the wind weeks, so they are built locally too and never committed."""
    if not path.exists():
        raise wind.NotBuilt(f"lines weeks not built ({path} is missing): run python tools/build_lines.py in "
                            "07-simulation-environments/portsim-live/envs/port_week, after tools/build_wind.py")
    return path


def pack_dir() -> Path:
    return packs.tasks_root() / packs.LINES_PACK


def pack() -> TaskPack:
    return TaskPack(_built(pack_dir()))


def references() -> list[dict]:
    return [json.loads(line) for line in _built(REFERENCES).read_text().splitlines()]


def task_ids() -> list[str]:
    return [r["task_id"] for r in references() if r["qualifies"]]


def meta(task: Task, sid: int) -> dict:
    return task.rules["lines"]["ships"][sid]


def firm(task: Task, sid: int) -> str:
    return meta(task, sid)["firm"]


def firm_name(task: Task, firm_id: str) -> str:
    return task.rules["lines"]["firms"][firm_id]


def leg_hours(task: Task, sid: int) -> float | None:
    nm = meta(task, sid)["leg_nm"]
    return None if nm is None else nm / CRUISE_KN


def departure(task: Task, sid: int, arrival: int | None = None) -> float | None:
    """The hour the ship leaves its previous port to arrive at ``arrival`` (its true arrival by default) at cruising
    speed; None when that is beyond the week's reach."""
    leg = leg_hours(task, sid)
    return None if leg is None else (task.ships[sid].arrival if arrival is None else arrival) - leg


def slowest(task: Task, sid: int, hour: int, arrival: int) -> int:
    """The latest an unslowed ship due at ``arrival`` can arrive, slowing from ``hour`` or once it leaves its previous
    port."""
    dep = departure(task, sid, arrival)
    start = hour if dep is None else max(hour, dep)
    return math.floor(start + (arrival - start) * STRETCH + 1e-9)


def estimate(task: Task, sid: int, hour: int, arrival: int | None = None) -> int:
    """The firm's ETA at ``hour`` for an unslowed ship due at ``arrival`` (its true arrival by default): off by a share
    of the hours left, as from 72 hours out at most, mostly early (the ship arrives later); never later than the
    slowest it can arrive, and exact from the arrival call."""
    a = task.ships[sid].arrival if arrival is None else arrival
    if a - hour <= CALL_LEAD:
        return a
    eta = a - round(ETA_ERROR * meta(task, sid)["bias"] * min(a - hour, ETA_HORIZON))
    return min(max(hour + 1, eta), slowest(task, sid, hour, a))


def commits(task: Task, sid: int, hour: int, arrival: int) -> bool:
    """Whether the ship's agent can commit to an arrival hour at ``hour``: once the ship has sailed from its previous
    port, or once its estimate is within 48 hours, the berth-planning horizon. The estimate, not an hour the agent has
    offered since, so an offered hour can be agreed whichever was asked first."""
    dep = departure(task, sid, arrival)
    return dep is None or hour >= dep or estimate(task, sid, hour, arrival) - hour <= COMMIT_LEAD


def as_v4(task: Task) -> Task:
    """The task with v4's arrivals, from which v4's news is timed, so the watches don't move."""
    ships = task.rules["lines"]["ships"]
    return replace(task, ships=[replace(s, arrival=ships[s.id].get("v4_arrival", s.arrival)) for s in task.ships])


def scheduled(task: Task, sid: int) -> int | None:
    """The arrival the published schedule gives, the line's estimate from 72 hours out; None for an unscheduled call."""
    return meta(task, sid).get("scheduled")


def before_news(task: Task, sid: int) -> int | None:
    """When a delayed or bunched ship would have arrived without the news about it."""
    return meta(task, sid).get("before_news")


def news_of(task: Task) -> dict[int, str]:
    """The event that brings each delayed or bunched ship's news."""
    out = {}
    for i, e in enumerate(task.disruptions):
        if e["type"] in ("late", "bunching"):
            out |= {s: f"{e['type']}-{i}" for s in ([e["ship"]] if e["type"] == "late" else e["ships"])}
    return out


def first_known(task: Task) -> dict[int, int]:
    """The hour of the watch that first brings a ship's own news, for the ships whose arrival v4 announced: an
    unscheduled call when its notice reveals it, a late or bunched ship when the notice about it comes."""
    out = {}
    for w in v4_schedule(as_v4(task)):
        for n in w.notices:
            if n.kind in NEWS_ETA:
                for s in [n.event["ship"]] if n.kind != "bunching" else n.event["ships"]:
                    out[s] = max(out.get(s, w.hour), w.hour)
    return out


def timetable(task: Task) -> list[Watch]:
    """v4's watches, unchanged, each also carrying the ETA reports and arrival calls made since the watch before:
    routine reports add no watch, so they wait for the next bulletin, and the latest one made before hour 0 opens the
    week. A delayed or bunched ship's reports before its news come from the plan as it stood, and only those that
    reach the planner before the news; an unscheduled call's start with its notice."""
    watches = v4_schedule(as_v4(task))
    hours = [w.hour for w in watches]
    news = first_known(task)
    reports: dict[int, list[Notice]] = {k: [] for k in range(len(watches))}
    for s in task.ships:
        made = [(s.arrival - lead, "eta", lead, "") for lead in REPORT_LEADS] + [
            (s.arrival - CALL_LEAD, "call", CALL_LEAD, "")]
        if s.id in news:
            made = [m for m in made if m[0] > news[s.id]]
            if (planned := before_news(task, s.id)) is not None:
                made += [m for m in _opening([(planned - lead, "eta", lead, "-plan") for lead in REPORT_LEADS])
                         if next((h for h in hours if h >= m[0]), news[s.id]) < news[s.id]]
        else:
            made = _opening(made)
        for hour, kind, lead, tag in made:
            k = next((k for k, h in enumerate(hours) if h >= hour), None)
            if k is None:
                continue
            at, arrival = max(hour, 0), before_news(task, s.id) if tag else s.arrival
            event = {"type": kind, "ship": s.id, "hour": at, "lead": lead,
                     "eta": arrival if kind == "call" else estimate(task, s.id, at, arrival)}
            reports[k].append(Notice(f"{kind}-{s.id}-{lead}{tag}", -1, kind, firm_name(task, firm(task, s.id)), at,
                                     arrival, report_text(task, event), event))
    return [Watch(w.index, w.hour, w.notices + tuple(sorted(reports[w.index], key=lambda n: (n.hour, n.event_id))))
            for w in watches]


def _opening(made: list[tuple]) -> list[tuple]:
    """The reports made during the week, and the latest made before it, which opens it."""
    before = [m for m in made if m[0] <= 0]
    return [m for m in made if m[0] > 0] + ([max(before)] if before else [])


def report_text(task: Task, event: dict) -> str:
    s = task.ships[event["ship"]]
    if event["type"] == "call":
        return f"{s.name} (ship {s.id}) calls in: at buoy S, arriving hour {event['eta']} ({label(event['eta'])})."
    return (f"{s.name} (ship {s.id}): ETA {label(event['eta'])} (hour {event['eta']}), as of "
            f"{label(event['hour'])}.")


def _waiting(task: Task, sid: int, hour: int, window: tuple[float, int, int]) -> bool:
    """Whether the ship is at anchor at ``hour``, in by then and berthing later in ``window``, unheld by the wind and
    with a stretch of quay the closures leave clear for its stay."""
    ship, (start, _, cranes) = task.ships[sid], window
    return (ship.arrival <= hour < start and not any(a <= hour < b for a, b in task.no_move_windows(ship))
            and _room(task, ship.sections, hour, ship.handling_for(cranes)))


def _room(task: Task, sections: int, hour: int, stay: int) -> bool:
    """Whether the closures leave a stretch of that many sections clear from ``hour`` for ``stay`` hours."""
    shut = [(b.first, b.last + 1) for b in task.blocks if b.kind == "closed" and b.start < hour + stay and hour < b.end]
    return any(not any(x < hi and lo < x + sections for lo, hi in shut)
               for x in range(task.first_section, task.last_section - sections + 2))


def _queued(task: Task, plan: Plan, res: PlanResult) -> None:
    """Each ship that starts berthing while a ship earlier in the queue, with an agreed arrival, waits at anchor breaks
    the queue rule, unless the wind holds that ship or the closures leave no stretch long enough for it; the first such
    ship is named. A wait the planner couldn't foresee, an unagreed ship arriving off its ETA, costs its hours at anchor
    instead."""
    turns = task.rules["lines"]["turns"]
    agreed = set(task.rules["lines"].get("agreed", range(len(task.ships))))
    starts = {r.ship: r.berth_hour for r in res.ships if r.berth_hour is not None}
    for y, ty in sorted(starts.items()):
        for x in sorted(starts, key=lambda x: (turns[x], x)):
            if turns[x] >= turns[y]:
                break
            if x in agreed and _waiting(task, x, ty, plan[x]):
                res.ships[y].problems.append(f"berths at hour {ty} while ship {x} {task.ships[x].name}, earlier in "
                                             f"the queue, waits at anchor")
                break


def executed(task: Task, plan: Plan) -> Plan:
    """The windows as the port runs them. A ship that arrives before its window waits for it. One that arrives after
    its window starts is berthed at the first hour from its arrival when a free stretch of quay fits it, with the same
    cranes, around every other ship's window without breaking a rule and ahead of no earlier ship with an agreed
    arrival still waiting; the stretch nearest its planned sections first, and nobody else moves. Hour by hour, the
    late ships in by then are placed in queue order; with no such hour within a week, a window starts when its ship
    gets in, or once the wind lets it move."""
    turns = task.rules["lines"]["turns"]
    late = {s for s, w in plan.items() if 0 <= s < len(task.ships) and w[0] < task.ships[s].arrival}
    out = {s: w for s, w in plan.items() if s not in late}
    if not late:
        return out
    base = None
    for hour in range(min(task.ships[s].arrival for s in late), max(task.ships[s].arrival for s in late)
                      + DISPATCH_HOURS + 1):
        for sid in sorted((s for s in late if s not in out and task.ships[s].arrival <= hour),
                          key=lambda s: (turns[s], s)):
            ship, (_, section, cranes) = task.ships[sid], plan[sid]
            if hour == ship.arrival + DISPATCH_HOURS:
                out[sid], base = _slid(task, sid, plan[sid]), None
                continue
            if any(a <= hour < b for a, b in task.no_move_windows(ship)):
                continue
            if base is None:
                base = marine.evaluate(task, out)
                before = {r.ship: set(r.problems) for r in base.ships}
                taken = [(r.section, r.section + task.ships[r.ship].sections, r.berth_hour, r.departure)
                         for r in base.ships if r.berth_hour is not None and r.departure is not None]
                taken += [(b.first, b.last + 1, b.start, b.end) for b in task.blocks]
            stay = ship.handling_for(cranes)
            sections = sorted(range(task.first_section, task.last_section - ship.sections + 2),
                              key=lambda x: (abs(x - section), x))
            waiting = {s: (math.inf, *plan[s][1:]) for s in late - set(out)}
            window = next(((hour, x, cranes) for x in sections
                           if not any(x < hi and lo < x + ship.sections and hour < end and start < hour + stay
                                      for lo, hi, start, end in taken)
                           and _fits(task, out, waiting, sid, (hour, x, cranes), before)), None)
            if window:
                out[sid], base = window, None
    return out


def _fits(task: Task, out: Plan, waiting: dict[int, tuple[float, int, int]], sid: int, window: tuple[int, int, int],
          before: dict[int, set[str]]) -> bool:
    """Whether the port can berth the late ship in this window: no earlier ship with an agreed arrival waits, whether
    berthing later in its window or a late ship not yet placed, and the window adds no problem."""
    turns = task.rules["lines"]["turns"]
    agreed = set(task.rules["lines"].get("agreed", range(len(task.ships))))
    windows = {x: w for x, w in out.items() if 0 <= x < len(task.ships)} | {
        x: w for x, w in waiting.items() if x != sid}
    if any(turns[x] < turns[sid] and x in agreed and _waiting(task, x, window[0], w) for x, w in windows.items()):
        return False
    res = marine.evaluate(task, out | {sid: window})
    return not res.ships[sid].problems and all(
        set(r.problems) <= before.get(r.ship, set()) for r in res.ships if r.ship != sid)


def _slid(task: Task, sid: int, window: tuple[int, int, int]) -> tuple[int, int, int]:
    """The window started when its ship gets in, or once the wind lets it move."""
    ship, (_, section, cranes) = task.ships[sid], window
    start = ship.arrival
    while any(a <= start < b for a, b in task.no_move_windows(ship)):
        start += 1
    return start, section, cranes


def evaluate(task: Task, plan: Plan) -> PlanResult:
    """marine.evaluate of the windows as run, with the queue rule, and each hour a ship waits at anchor charged as an
    hour late is: its sections times the hours between its arrival and its berthing."""
    plan = executed(task, plan)
    res = marine.evaluate(task, plan)
    for r in res.ships:
        arrival = task.ships[r.ship].arrival
        if r.berth_hour is not None and r.berth_hour > arrival:
            r.cost += task.ships[r.ship].sections * (r.berth_hour - arrival)
    _queued(task, plan, res)
    if any(r.problems for r in res.ships):
        return PlanResult(False, res.ships, None, None, None)
    cost = sum(r.cost for r in res.ships)
    moves = sum(r.moved for r in res.ships)
    return PlanResult(True, res.ships, cost, cost - MOVE_PENALTY * moves, moves)


def anchor_hours(task: Task, res: PlanResult) -> int:
    return sum(r.berth_hour - task.ships[r.ship].arrival for r in res.ships
               if r.berth_hour is not None and r.berth_hour > task.ships[r.ship].arrival)


def with_arrivals(task: Task, arrivals: dict[int, int], turns: dict[int, int],
                  agreed: Iterable[int] | None = None) -> Task:
    """The task with these arrivals and queue turns, and the ships whose arrival is agreed (every ship's when not
    given), whom the queue rule protects."""
    ships = [replace(s, arrival=arrivals[s.id]) for s in task.ships]
    lines = task.rules["lines"] | {"turns": [turns[s.id] for s in task.ships]}
    if agreed is not None:
        lines["agreed"] = sorted(agreed)
    return replace(task, ships=ships, rules=task.rules | {"lines": lines})


class LinesWeek(WindWeek):
    timetable = staticmethod(timetable)
    evaluate = staticmethod(evaluate)
    untimed = REPORTS

    anchor_rows = True

    def __init__(self, task: Task):
        self.locks: dict[int, tuple[float, float, int]] = {}
        self.told: dict[int, tuple[int, int]] = {}
        self.arrived: set[int] = set()
        self.exchanges: list[dict] = []
        self.news = news_of(task)
        self.outcomes: dict[str, list[str]] = {}
        self.floor: dict[int, int] = {}
        self.news_texts: dict[str, str] = {}
        self.paces: dict[int, list[tuple[float, float]]] = {}
        self.resumed: dict[int, int] = {}
        super().__init__(task)

    def open(self) -> dict:
        """The next watch, without the ships berthed, departed and still without a window: those are news its bulletin
        brings, and get_situation shows them once it has."""
        return {k: v for k, v in super().open().items() if k not in ("berthed", "departed", "unconfirmed")}

    def confirm(self, raw) -> dict:
        """Confirms, and reports the confirmed plan's problems and cost as check_plan would, at no extra call."""
        out = super().confirm(raw)
        if "error" in out:
            return out
        return ({k: v for k, v in out.items() if k != "planning_calls_left"} | self.assess(self.view, self.plan)
                | {"planning_calls_left": out["planning_calls_left"]})

    def draft_excuses(self, plan: Plan) -> tuple[set[tuple[int, str]], dict[int, int]]:
        """The excuses the draft itself would earn, so a problem the confirmed plan has and the draft fixes isn't
        charged to the draft."""
        return self.excuses(plan)

    def revealing(self, notice: Notice) -> None:
        """A delayed or bunched ship's news: an agreement made before it stands if the ship can still make that hour,
        else it lapses and the ship cruises in from where it is, which its agent gives as the ETA; without an
        agreement, what its agent last said moved later by the delay. The news's ETA becomes what the planner knows."""
        event = notice.event
        if event["type"] == "extra":
            self._tell(event["ship"], self.hour, event["eta"])
        if event["type"] not in ("late", "bunching"):
            return
        said, etas = [], {}
        for sid in self._news_ships(event):
            truth, before = self.task.ships[sid].arrival, before_news(self.task, sid)
            last = max(estimate(self.task, sid, self.hour, before), self.floor.pop(sid, 0))
            etas[sid] = min(max(self.hour + 1, last + truth - before), slowest(self.task, sid, self.hour, truth))
            if sid in self.locks:
                at = self.locks[sid][2]
                start, left = self.progress(sid, self.hour, truth)
                earliest = math.ceil(start + left - 1e-9)
                if earliest <= at <= math.floor(start + left * STRETCH + 1e-9):
                    self.locks[sid] = (start, left, at)
                    self.paces[sid] = [(start, left / (at - start) if at > start else 1.0)]
                    etas[sid] = at
                    said.append(f"The earlier agreement stands: {self.agreement(sid)}")
                    continue
                del self.locks[sid], self.paces[sid]
                if earliest > truth:
                    self.resumed[sid] = earliest
                etas[sid] = max(estimate(self.task, sid, self.hour, self.basis(sid)), earliest)
                said.append(f"{self.task.ships[sid].name} (ship {sid}) can no longer arrive at hour {at}: the earlier "
                            "agreement no longer holds.")
            self.floor[sid] = min(etas[sid], self.actual(sid))
            self._tell(sid, self.hour, etas[sid])
        self.outcomes[notice.event_id] = said
        self.news_texts[notice.text] = (
            re.sub(r"gives ETA hour \d+", f"gives ETA hour {etas[event['ship']]}", notice.text)
            if event["type"] == "late" else
            re.sub(r"ship (\d+) hour \d+", lambda m: f"ship {m[1]} hour {etas[int(m[1])]}", notice.text))

    def progress(self, sid: int, hour: int, truth: int) -> tuple[float, float]:
        """(the hour from which the ship can change pace, the cruising hours it still needs then) for a ship that
        agreed an hour before its news, on its true track: it left its previous port when the truth has it leave, and
        has since sailed at the pace each agreement set in turn, from the first."""
        segments = self.paces[sid]
        dep = departure(self.task, sid, truth)
        since = segments[0][0] if dep is None else max(segments[0][0], dep)
        start = max(hour, since)
        ends = [begin for begin, _ in segments[1:]] + [start]
        sailed = sum(pace * max(0.0, min(end, start) - max(begin, since))
                     for (begin, pace), end in zip(segments, ends, strict=True))
        return start, truth - since - sailed

    def notice(self, event_id: str, name: str, text: str, via: str) -> dict:
        out = super().notice(event_id, name, text, via)
        event = self.scheduled[event_id].event
        reported = event["type"] in REPORTS
        made = (estimate(self.task, event["ship"], event["hour"], self.resumed[event["ship"]])
                if event["type"] == "eta" and event["ship"] in self.resumed else event.get("eta"))
        eta = max(made, self.floor.get(event["ship"], 0)) if reported else None
        if reported:
            self._tell(event["ship"], event["hour"], eta)
            if event["type"] == "call" and event["ship"] not in self.locks:
                self.arrived.add(event["ship"])
        delivered = text
        if via != "load":
            last = self.messages[-1]
            if reported and event["ship"] in self.locks:
                last["text"] = self.agreement(event["ship"])
            elif reported and eta != event["eta"]:
                last["text"] = report_text(self.task, event | {"eta": eta})
            elif event_id in self.outcomes:
                last["text"] = " ".join([self.news_texts.get(last["text"], last["text"]), *self.outcomes[event_id]])
            delivered = last["text"]
            if (reported and event["ship"] in self.locks and self.actual(event["ship"]) <= self.hour) or any(
                    m["hour"] == self.hour and m["text"] == delivered for m in self.messages[:-1]):
                self.messages.pop()
                delivered = None
        self.log[-1]["text"] = delivered
        return out

    def agreement(self, sid: int) -> str:
        return (f"{self.task.ships[sid].name} (ship {sid}): arriving hour {self.actual(sid)} "
                f"({label(self.actual(sid))}), as agreed; turn hour {self.turn(sid)}.")

    def _tell(self, sid: int, as_of: int, eta: int) -> None:
        if sid not in self.locks and (sid not in self.told or self.told[sid][0] <= as_of):
            self.told[sid] = (as_of, eta)

    def actual(self, sid: int) -> int:
        """When the ship really arrives: at the agreed time, else at cruising speed, from where it was when an
        agreement it had slowed for lapsed."""
        return self.locks[sid][2] if sid in self.locks else self.resumed.get(sid, self.task.ships[sid].arrival)

    def pending(self, sid: int) -> bool:
        """Whether the ship's news has yet to reach the planner, so its agent still works from the plan as it stood."""
        return sid in self.news and self.news[sid] not in self.revealed

    def basis(self, sid: int) -> int:
        """The cruising-speed arrival the ship's agent works from: the plan as it stood until its news, then the
        truth, or where the ship cruises in from once a lapsed agreement had slowed it."""
        return before_news(self.task, sid) if self.pending(sid) else self.resumed.get(sid, self.task.ships[sid].arrival)

    def turn(self, sid: int) -> int:
        """The ship's place in the queue: its arrival at cruising speed, so a ship slowed on request keeps it."""
        return before_news(self.task, sid) if self.pending(sid) else self.task.ships[sid].arrival

    def known_arrival(self, view: Task, sid: int) -> int:
        """The arrival as the planner knows it: agreed or called in, else the latest ETA or the published schedule, and
        no earlier than the next hour for a ship that hasn't called in."""
        if sid in self.locks or sid in self.arrived:
            return self.actual(sid)
        eta = self.told[sid][1] if sid in self.told else scheduled(self.task, sid)
        return max(view.ships[sid].arrival if eta is None else eta, self.hour + 1)

    def known_turn(self, view: Task, sid: int) -> int:
        return self.turn(sid) if sid in self.locks or sid in self.arrived else self.known_arrival(view, sid)

    @property
    def view(self) -> Task:
        view = wind.known(self.task, self.revealed)
        visible = {s.id for s in view.ships}
        notices = [self.news_texts.get(text, text)
                   + "".join(f" Since then: {self.agreement(s)}" for s in self._news_ships(e) if s in self.locks)
                   for e, text in zip(view.disruptions, view.notices, strict=True)]
        return with_arrivals(replace(view, notices=notices), {s: self.known_arrival(view, s) for s in visible},
                             {s: self.known_turn(view, s) for s in visible}, visible & set(self.locks))

    @staticmethod
    def _news_ships(event: dict) -> list[int]:
        return ([event["ship"]] if event["type"] in ("late", "extra") else event["ships"]
                if event["type"] == "bunching" else [])

    def truth(self) -> Task:
        ids = range(len(self.task.ships))
        return with_arrivals(self.task, {s: self.actual(s) for s in ids}, {s: self.turn(s) for s in ids}, self.locks)

    def past(self, revealed: list[str], target: Task) -> Task:
        """The week as known then, with the target's arrivals and turns: an ETA that proved wrong is no news, so the
        excuses never waive it."""
        view = wind.known(self.task, revealed)
        arrivals = {s.id: s.arrival for s in target.ships}
        turns = dict(zip(arrivals, target.rules["lines"]["turns"], strict=True))
        agreed = {s.id for s in view.ships} & set(target.rules["lines"].get("agreed", arrivals))
        return with_arrivals(view, {s.id: arrivals[s.id] for s in view.ships}, {s.id: turns[s.id] for s in view.ships},
                             agreed)

    def target(self) -> Task:
        return self.truth() if self.done else self.view

    def finish(self, end_reason: str) -> dict:
        self.done, self.end_reason = True, end_reason
        truth = self.truth()
        unslowed = with_arrivals(self.task, {s.id: s.arrival for s in self.task.ships},
                                 {s.id: s.arrival for s in self.task.ships})
        self.grade = world.grade_week(unslowed, self.plan, *self.excuses(),
                                      evaluate=lambda _task, plan: evaluate(truth, plan))
        self.grade["anchor_hours"] = anchor_hours(truth, evaluate(truth, self.plan))
        return {"done": True, "feasible": self.grade["feasible"], "cost": self.grade["cost"],
                "reward": self.grade["reward"]}

    def situation(self) -> dict:
        out = super().situation()
        return out | {"situation": out["situation"] + "\n\n" + section(self)}

    def remaining(self, sid: int, hour: int, arrival: int | None = None) -> tuple[float, float]:
        """(the hour from which the ship can change pace, the hours it still needs at cruising speed then), as agreed or
        from its agent's basis, or for an unslowed ship due at ``arrival``."""
        a = self.basis(sid) if arrival is None else arrival
        dep = departure(self.task, sid, a)
        start = hour if dep is None else max(hour, dep)
        if sid in self.locks and arrival is None:
            ref, left, at = self.locks[sid]
            start = max(start, ref)
            return start, left * (at - start) / (at - ref) if at > ref else 0.0
        return start, a - start

    def reach(self, sid: int, hour: int, arrival: int | None = None) -> tuple[int, int]:
        """The earliest and latest hours the ship can arrive, from ``hour`` on."""
        start, left = self.remaining(sid, hour, arrival)
        return math.ceil(start + left - 1e-9), math.floor(start + left * STRETCH + 1e-9)

    def _ship(self, firm_id: str, sid) -> int:
        view = self.view
        if not isinstance(sid, int) or isinstance(sid, bool) or sid not in {s.id for s in view.ships}:
            raise ValueError(f"no ship {sid!r} in the week")
        if firm(self.task, sid) != firm_id:
            raise ValueError(f"ship {sid} {self.task.ships[sid].name} is not this agent's: "
                             f"{firm_name(self.task, firm(self.task, sid))} handles it")
        return sid

    def status(self, firm_id: str, sid) -> dict:
        sid = self._ship(firm_id, sid)
        name, hour = self.task.ships[sid].name, self.hour
        if not self.pending(sid) and self.actual(sid) <= hour:
            out = {"ship": sid, "name": name, "arrived": self.actual(sid), "turn": self.turn(sid)}
            self.arrived.add(sid)
        elif sid in self.locks:
            out = {"ship": sid, "name": name, "eta": self.actual(sid), "agreed": True, "turn": self.turn(sid)}
        else:
            eta = max(estimate(self.task, sid, hour, self.basis(sid)), self.floor.get(sid, 0))
            self._tell(sid, hour, eta)
            out = {"ship": sid, "name": name, "eta": eta, "as_of": hour}
        self.exchanges.append({"watch": self.watch, "hour": hour, "firm": firm_id, "ship": sid, "ask": "eta",
                               "call": self.calls, "reply": out})
        return out

    def request(self, firm_id: str, sid, arrival) -> dict:
        sid = self._ship(firm_id, sid)
        if not isinstance(arrival, int) or isinstance(arrival, bool):
            raise ValueError(f"request_arrival must be an hour, got {arrival!r}")
        name, hour = self.task.ships[sid].name, self.hour
        if not self.pending(sid) and self.actual(sid) <= hour:
            out = {"ship": sid, "name": name, "agreed": False, "reason": f"already arrived at hour {self.actual(sid)}"}
        elif sid not in self.locks and not commits(self.task, sid, hour, self.basis(sid)):
            eta = max(estimate(self.task, sid, hour, self.basis(sid)), self.floor.get(sid, 0))
            self._tell(sid, hour, eta)
            out = {"ship": sid, "name": name, "agreed": False, "eta": eta,
                   "reason": f"can't commit yet: it hasn't sailed from {self.task.ships[sid].from_port} and is more "
                             f"than {COMMIT_LEAD} hours out"}
        else:
            earliest, latest = self.reach(sid, hour)
            kept = {"agreed_arrival": self.locks[sid][2]} if sid in self.locks else {}
            if earliest <= arrival <= latest:
                start, left = self.remaining(sid, hour)
                self.locks[sid] = (start, left, arrival)
                self.paces.setdefault(sid, []).append((start, left / (arrival - start) if arrival > start else 1.0))
                self.told[sid] = (hour, arrival)
                out = {"ship": sid, "name": name, "agreed": True, "arrival": arrival, "turn": self.turn(sid),
                       "slowing": arrival > earliest}
            elif arrival < earliest:
                if sid not in self.locks:
                    self.floor[sid] = max(self.floor.get(sid, 0), earliest)
                self._tell(sid, hour, earliest)
                out = {"ship": sid, "name": name, "agreed": False, "reason": "can't arrive that early",
                       "earliest": earliest} | kept
            else:
                out = {"ship": sid, "name": name, "agreed": False, "reason": "can't slow down that much",
                       "latest": latest} | kept
        self.exchanges.append({"watch": self.watch, "hour": hour, "firm": firm_id, "ship": sid,
                               "ask": "request_arrival", "arrival": arrival, "call": self.calls, "reply": out})
        return out

    def audit(self) -> dict:
        out = super().audit()
        problems = list(out["problems"])
        truth = self.truth()
        for sid, (_, _, at) in self.locks.items():
            if truth.ships[sid].arrival != at:
                problems.append(f"ship {sid} agreed hour {at} but arrives at {truth.ships[sid].arrival}")
        return {"ok": not problems, "problems": problems}

    def data(self) -> dict:
        return super().data() | {
            "lines": {"exchanges": self.exchanges,
                      "agreed": [{"ship": s, "arrival": at, "turn": self.turn(s)} for s, (_, _, at) in
                                 sorted(self.locks.items())],
                      "arrivals": [{"ship": s.id, "arrival": self.actual(s.id), "turn": self.turn(s.id)}
                                   for s in self.task.ships] if self.done else []}}


def firms_in(view: Task) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for s in view.ships:
        out.setdefault(firm(view, s.id), []).append(s.id)
    return out


def section(week: LinesWeek) -> str:
    view = week.view
    lines = [
        "## Arrivals and the ships' agents",
        "- Each ship's agent reports its ETA 48 and 24 hours out, and the ship calls in an hour before it arrives; "
        "reports made between watches arrive with the next one. The published schedule is the lines' estimate from "
        "about three days out. Both are optimistic: most ships arrive later than they say, by less as they near.",
        "- Message a ship's agent to ask for its ETA or to request an arrival time. Once the ship has sailed from its "
        f"previous port or its agent's estimate is within {COMMIT_LEAD} hours, the agent agrees when it can make the "
        f"hour at {CRUISE_KN} kn or by slowing to as little as {SLOWEST_KN} kn once it has sailed, and the ship then "
        "arrives at that hour; otherwise it answers with the earliest or the latest hour the ship can make. Before "
        "that, the agent can't commit to an hour and gives its ETA.",
        "- Until the port's news about a delayed ship, its agent works from the plan as it stood. The news keeps an "
        "earlier agreement only if the ship can still make that hour; otherwise the agreement lapses.",
        "- A ship can't berth before it arrives. If it gets in after its window starts, the port berths it at the "
        "first hour a free stretch of quay fits it, with the same cranes, around every other ship's window and the "
        "rules, never ahead of an earlier ship whose arrival you agreed still waiting; nothing else moves. A ship that "
        "gets in before its window waits at anchor for it.",
        "- Ships berth first come, first served: no ship may start berthing while a ship earlier in the queue whose "
        "arrival you agreed waits at anchor, unless the wind holds that ship or the closures leave no stretch of quay "
        "long enough for it. A ship's turn is when it would arrive at cruising speed, so a ship slowed at your request "
        "keeps its turn.",
        "- Each hour a ship waits at anchor between its arrival and its berthing costs its sections, as an hour late "
        "does.",
        "- Agents, and their ships: " + "; ".join(
            f"{firm_name(view, f)} ({f}): ships {', '.join(map(str, ids))}"
            for f, ids in sorted(firms_in(view).items())) + "."]
    rows = []
    for s in view.ships:
        if s.id in week.locks:
            state = f"agreed hour {week.actual(s.id)}, turn {week.turn(s.id)}"
        elif s.id in week.arrived:
            verb = "arrived" if week.actual(s.id) <= week.hour else "called in, arriving"
            state = f"{verb} hour {week.actual(s.id)}, turn {week.turn(s.id)}"
        elif s.id in week.told:
            as_of, eta = week.told[s.id]
            state = f"ETA hour {eta}, as of {label(as_of)}"
        else:
            state = "published schedule"
        rows.append(f"- ship {s.id} {s.name}: arrival hour {s.arrival}, {state}; from {s.from_port or 'unknown'}")
    return "\n".join(lines + ["- Arrivals as known now:"] + rows)


def step_lines(week: LinesWeek) -> dict:
    view = week.view
    return {"agreed": [{"ship": s, "arrival": at} for s, (_, _, at) in sorted(week.locks.items())],
            "arrivals": [{"ship": s.id, "arrival": s.arrival, "known": s.id in week.locks or s.id in week.arrived}
                         for s in view.ships],
            "exchanges": [e for e in week.exchanges if e["watch"] == week.watch]}

