"""One live day, pure and deterministic: the watch and its clock minute, the plan, the freeze, the news queue, the
situation the planner reads, check and confirm on the day as known, the grade on the truth and the audit.

From the second watch on, a flight that isn't frozen can't leave before the freeze line: the env holds a flight the
plan would have leaving earlier to the freeze line, so a decision never reaches back past it."""

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import date

from .core.known import CAP_LOOK, FREEZE, frozen, known_caps, lb_known, view
from .core.model import News, Task, hhmm, parse_hhmm
from .core.reward import reward
from .core.sim import AUTO_CANCELLED, CANCELLED, FLOWN, Execution, FlightRun, execute
from .schedule import PLANNING_CALLS, timetable

ACTIONS = ("cancel", "restore", "hold")
CAP_HOURS = 8
EARLY = 720
DESK = "Duty manager"
NOTE = "This watch's news arrives with your next tool result."
CANCEL = {"cancel": True, "not_before": None}
FLY = {"cancel": False, "not_before": None}
USAGE = '[{"flight": "f12", "action": "cancel", "not_before": null}, {"flight": "f30", "action": "hold", ' \
        '"not_before": "15:40"}]'
PROGRAM_KINDS = {"gs": "ground stop", "gdp": "ground delay program", "gdp_proposed": "proposed ground delay program"}
PERIODS = {"gs": ("gs_period", "program_period", "event_time"), "gdp": ("program_period", "arrivals_for"),
           "gdp_proposed": ("proposed_period", "arrivals_for"), "other": ("event_time",)}
CLEARS = {"gs_cnx": ("gs",), "gdp_cnx": ("gdp", "gdp_proposed"), "gdp": ("gdp_proposed",)}


class ChangeError(ValueError):
    """Changes that can't be read at all."""


def fid(i: int) -> str:
    return f"f{i}"


class Day:
    def __init__(self, task: Task):
        self.task = task
        self.watches = timetable(task)
        self.news = {n.id: n for n in task.news}
        self.scheduled = {n: w.index for w in self.watches for n in w.news}
        self.watch = 0
        self.plan: dict[int, dict] = {}
        self.frozen: set[int] = set()
        self.calls = 0
        self.planning_calls = [0]
        self.revealed: list[str] = []
        self.log: list[dict] = []
        self.messages: list[dict] = []
        self.freezes = [{"watch": 0, "t": task.start, "line": None, "call": 0, "frozen": []}]
        self.changes: list[dict] = []
        self.refusals: list[dict] = []
        self.estimates: list[dict] = []
        self.done = False
        self.end_reason: str | None = None
        self.grade: dict | None = None
        for n in self.watches[0].news:
            self.notice(n, "load")

    @property
    def t(self) -> int:
        return self.watches[self.watch].t

    @property
    def line(self) -> int | None:
        """The freeze line; none at the first watch, where the planner sets the day up."""
        return None if self.watch == 0 else self.t + FREEZE

    @property
    def last(self) -> bool:
        return self.watch == len(self.watches) - 1

    @property
    def calls_left(self) -> int:
        return PLANNING_CALLS - self.planning_calls[-1]

    def project(self, plan: Mapping[int, Mapping]) -> Execution:
        """``plan`` run on the day as known now."""
        return execute(self.task, plan, *view(self.task, self.t))

    def situation(self) -> dict:
        return {"watch": self.watch, "time": hhmm(self.t), "freeze_line": _time(self.line),
                "planning_calls_left": self.calls_left, "situation": self.text()}

    def briefing(self) -> str:
        """The opening prompt: the first watch's situation and the news so far."""
        news = "\n\n".join(_news_text(self.news[n]) for n in self.watches[0].news)
        return (f"{self.text()}\n\n## News so far\n{news or 'None.'}\n\nIt is the first watch, {hhmm(self.t)}. "
                "Nothing is frozen yet: set the day up with confirm_changes, then call advance.")

    def text(self) -> str:
        task, t, run = self.task, self.t, self.project(self.plan)
        hub = task.airports[task.hub]
        if self.line is None:
            freeze = "First watch: nothing is frozen; changes take effect at once."
        else:
            freeze = (f"Freeze line {hhmm(self.line)}: flights your plan has leaving before it are frozen "
                      f"({len(self.frozen)} now); no other flight leaves before it.")
        c = run.cost
        counts = run.counts
        held = sum(1 for e in self.plan.values() if not e["cancel"] and e["not_before"] is not None)
        parts = [
            f"# JetBlue operations desk, {task.hub} ({hub.name}), {_weekday(task.date)} {task.date}; times are local "
            f"({task.tz}).",
            f"Now {hhmm(t)}, watch {self.watch}. {freeze} Planning calls left this watch: {self.calls_left}.",
            f"Your plan on the day as known costs {c.total:,} passenger-minutes (arrival delay {c.pax_delay:,}, "
            f"cancellations {c.cancel:,}, ferries {c.ferry:,}): {counts[CANCELLED]} cancelled, "
            f"{counts[AUTO_CANCELLED]} cancelled automatically, {held} held.",
            self.caps_text(), self.programs_text(), self.weather_text(), self.flights_text(run),
            self.aircraft_text(run)]
        return "\n\n".join(parts)

    def caps_text(self) -> str:
        task, t = self.task, self.t
        caps = known_caps(task, t)
        touched = {a for f in task.flights for a in (f.orig, f.dest)}
        keys = sorted((k for k in caps if k.split(":")[0] in touched), key=lambda k: (k.split(":")[0] != task.hub, k))
        hours = range(t // 60, t // 60 + CAP_HOURS)
        known = (t + CAP_LOOK) // 60
        lines = [f"## Slot caps: JetBlue movements per hour, known through the {known % 24:02d}:00 hour; later hours "
                 "show JetBlue's own schedule (- = no limit)", "hour " + " ".join(f"{h % 24:02d}" for h in hours)]
        lines += [f"{k.replace(':', ' ')} " + " ".join(str(caps[k][h]) if h in caps[k] else "-" for h in hours)
                  for k in keys]
        return "\n".join(lines)

    def programs_text(self) -> str:
        lines = programs([self.news[n] for n in self.revealed], self.t)
        return "## Programs in effect (FAA Command Center)\n" + ("\n".join(lines) if lines else "None.")

    def weather_text(self) -> str:
        out = []
        for kind, label in (("metar", "Latest METAR"), ("taf", "Latest TAF")):
            seen = [self.news[n] for n in self.revealed if self.news[n].kind == kind]
            if seen:
                n = max(seen, key=lambda n: (n.at, n.id))
                out.append(f"{label} ({hhmm(n.at)}): {n.text}")
        return "## Weather\n" + ("\n".join(out) if out else "None yet.")

    def flights_text(self, run: Execution) -> str:
        task, t = self.task, self.t
        rows = ["## Flights (sched = scheduled dep-arr; exp = earliest departure once its delay is known, ? before; "
                "plan = cancel or hold@HH:MM; proj = where your plan has it leave and land on the day as known; late = "
                "arrival delay; tail = the aircraft it takes)",
                "id flight route sched pax type status exp plan proj late tail"]
        for f in sorted(task.flights, key=lambda f: (f.sched_dep, f.id)):
            r = run.flights[f.id]
            exp = hhmm(f.sched_dep + f.lb) if lb_known(f.sched_dep, t) else "?"
            rows.append(f"{fid(f.id)} {f.flight.split()[-1]} {f.orig}-{f.dest} {hhmm(f.sched_dep)}-{hhmm(f.sched_arr)} "
                        f"{f.pax} {f.family} {self.status(f.id, r)} {exp} {_plan_text(self.plan.get(f.id))} "
                        f"{_proj(r)}")
        return "\n".join(rows)

    def aircraft_text(self, run: Execution) -> str:
        task, t = self.task, self.t
        at: dict[tuple[str, str], list[str]] = defaultdict(list)
        airborne = []
        for tail in task.tails:
            station, ready = tail.pos, tail.ready
            for i in run.legs[tail.tail]:
                r = run.flights[i]
                if r.dep > t:
                    break
                f = task.flight(i)
                if r.arr > t:
                    airborne.append(f"{tail.tail} {fid(i)}>{f.dest}@{hhmm(r.arr)}")
                    station = None
                    break
                station, ready = f.dest, r.arr + task.turn_for(f.family)
            if station is not None:
                at[(station, tail.family)].append(tail.tail + (f"@{hhmm(ready)}" if ready > t else ""))
        lines = ["## Aircraft now, as your plan has them (@ = ready at)"]
        stations = sorted(at, key=lambda k: (k[0] != task.hub, k))
        lines += [f"{s} {fam}: {' '.join(at[(s, fam)])}" for s, fam in stations]
        lines.append("airborne: " + (" ".join(airborne) or "none"))
        return "\n".join(lines)

    def status(self, i: int, r: FlightRun) -> str:
        if i not in self.frozen:
            return "cancelled" if r.status == CANCELLED else "open"
        if r.status == FLOWN and r.dep <= self.t:
            return "flown" if r.arr <= self.t else "airborne"
        return "frozen"

    def _planning_call(self) -> dict | None:
        if not self.calls_left:
            return {"error": f"no planning calls left in this watch ({PLANNING_CALLS} used): call advance",
                    "planning_calls_left": 0}
        self.planning_calls[-1] += 1
        return None

    def _merge(self, raw) -> tuple[dict, list[int], list[int], list[dict], list[str], list[int]]:
        rows, problems = read_changes(self.task, raw)
        plan, confirmed, unchanged, refused = dict(self.plan), [], [], []
        run = None
        for i, entry in rows:
            if _entry(self.plan.get(i)) == entry:
                unchanged.append(i)
            elif i in self.frozen:
                run = run or self.project(self.plan)
                refused.append({"flight": fid(i), "from": DESK, "reason": self._frozen_reason(i, run.flights[i])})
            else:
                plan[i] = entry
                confirmed.append(i)
        return plan, confirmed, unchanged, refused, problems, self._settle(plan)

    def _frozen_reason(self, i: int, r: FlightRun) -> str:
        f = self.task.flight(i)
        if r.status == FLOWN:
            how = f"your plan has it leaving at {hhmm(r.dep)}"
        elif r.status == CANCELLED:
            how = f"it is cancelled and was due out at {hhmm(f.sched_dep)}"
        else:
            how = f"it was due out at {hhmm(f.sched_dep)} and can't leave in time"
        return f"{fid(i)} ({f.flight} {f.orig}-{f.dest}) is frozen: {how}, before the freeze line at " \
               f"{hhmm(self.line)}; it stands."

    def _settle(self, plan: dict) -> list[int]:
        """Holds to the freeze line every open flight ``plan`` has leaving before it; returns them."""
        if self.line is None:
            return []
        held: set[int] = set()
        while early := [i for i, r in self.project(plan).flights.items()
                        if r.status == FLOWN and i not in self.frozen and r.dep < self.line]:
            for i in early:
                plan[i] = {"cancel": False, "not_before": self.line}
            held.update(early)
        return sorted(held)

    def check(self, raw) -> dict:
        if error := self._planning_call():
            return error
        try:
            plan, _, _, refused, problems, held = self._merge(raw)
        except ChangeError as e:
            return {"error": str(e), "planning_calls_left": self.calls_left}
        now, then = self.project(self.plan), self.project(plan)
        return {"cost": then.cost.to_dict(), "cost_now": now.cost.total,
                "flights": self.moved(now, then, plan), "auto_cancelled": _auto(then),
                "held_to_freeze_line": [fid(i) for i in held], "refused": refused, "problems": problems,
                "planning_calls_left": self.calls_left}

    def confirm(self, raw) -> dict:
        if error := self._planning_call():
            return error
        try:
            plan, confirmed, unchanged, refused, problems, held = self._merge(raw)
        except ChangeError as e:
            return {"error": str(e), "planning_calls_left": self.calls_left}
        for i in sorted(set(confirmed) | set(held)):
            self.changes.append({"watch": self.watch, "t": self.t, "flight": i, "was": _entry(self.plan.get(i)),
                                 "now": plan[i], "by": "planner" if i in confirmed else "freeze_line",
                                 "call": self.calls})
        self.plan = {i: e for i, e in sorted(plan.items()) if e != FLY}
        self.refusals += [{"watch": self.watch, **r} for r in refused]
        run = self.project(self.plan)
        return {"confirmed": [fid(i) for i in confirmed], "unchanged": [fid(i) for i in unchanged],
                "held_to_freeze_line": [fid(i) for i in held], "refused": refused, "problems": problems,
                "cost": run.cost.to_dict(), "auto_cancelled": _auto(run), "planning_calls_left": self.calls_left}

    def moved(self, now: Execution, then: Execution, plan: Mapping[int, dict]) -> list[str]:
        """The flights whose plan or times the changes alter, as they would be (was: as they are); a swap of
        interchangeable aircraft alone isn't a change."""
        out = []
        for i in sorted(now.flights):
            a, b = now.flights[i], then.flights[i]
            if (a.status, a.dep, a.arr) != (b.status, b.dep, b.arr) or _entry(self.plan.get(i)) != _entry(plan.get(i)):
                f = self.task.flight(i)
                out.append(f"{fid(i)} {f.flight} {f.orig}-{f.dest} {_plan_text(plan.get(i))} {_proj(b)} "
                           f"(was {_plan_text(self.plan.get(i))} {_proj(a)})")
        return out

    def open(self) -> dict:
        self.estimates.append({"watch": self.watch, "t": self.t, "cost": self.project(self.plan).cost.to_dict()})
        self.watch += 1
        self.planning_calls.append(0)
        self.frozen |= frozen(self.task, self.project(self.plan), self.t)
        self.freezes.append({"watch": self.watch, "t": self.t, "line": self.line, "call": self.calls,
                             "frozen": sorted(self.frozen)})
        return {"watch": self.watch, "time": hhmm(self.t), "freeze_line": hhmm(self.line), "frozen": len(self.frozen),
                "note": NOTE}

    def notice(self, news_id: str, via: str) -> dict:
        if news_id not in self.scheduled:
            raise ValueError(f"unknown news {news_id!r}")
        if self.done:
            raise ValueError("the day is over")
        n = self.news[news_id]
        self.log.append({"news_id": news_id, "kind": n.kind, "watch": self.watch, "t": self.t, "via": via,
                         "call": self.calls})
        if news_id not in self.revealed:
            self.revealed.append(news_id)
        if via != "load":
            self.messages.append({"at": hhmm(n.at), "from": n.speaker, "kind": n.kind, "text": n.text})
        return {"delivered": news_id, "watch": self.watch}

    def drain(self) -> list[dict]:
        messages, self.messages = self.messages, []
        return messages

    def finish(self, end_reason: str) -> dict:
        self.estimates.append({"watch": self.watch, "t": self.t, "cost": self.project(self.plan).cost.to_dict()})
        self.done, self.end_reason = True, end_reason
        run = execute(self.task, self.plan)
        ref = self.task.reference
        score = round(reward(run.cost.total, ref["anchor"], ref["floor"]), 6) if "anchor" in ref else None
        self.grade = {"cost": run.cost.to_dict(), "counts": run.counts, "reward": score, "executed": run.rows(),
                      "ferries": [x.to_dict() for x in run.ferries]}
        return {"done": True, "cost": run.cost.total, "reward": score}

    def end(self) -> dict:
        if not self.done:
            while not self.last:
                self.open()
                for n in self.watches[self.watch].news:
                    self.notice(n, "env")
            self.finish("end_day")
        return {"end_reason": self.end_reason, "watch": self.watch, "cost": self.grade["cost"]["total"],
                "reward": self.grade["reward"], "audit": self.audit()}

    def audit(self) -> dict:
        problems = []
        for k in range(self.watch + 1):
            got = [e["news_id"] for e in self.log if e["watch"] == k]
            if got != list(self.watches[k].news):
                problems.append(f"watch {k} got {got} instead of {list(self.watches[k].news)}")
        problems += [f"{e['news_id']} arrived after the planner's next call in watch {e['watch']}" for e in self.log
                     if e["via"] == "trigger" and e["call"] != self.freezes[e["watch"]]["call"]]
        if self.done:
            problems += [f"{n} delivered {k} times" for n in self.scheduled
                         if (k := sum(e["news_id"] == n for e in self.log)) != 1]
        return {"ok": not problems, "problems": problems}

    def data(self) -> dict:
        return {"task_id": self.task.task_id, "watches": len(self.watches), "watch": self.watch, "t": self.t,
                "done": self.done, "end_reason": self.end_reason, "calls_used": self.calls,
                "planning_calls": self.planning_calls,
                "plan": [{"flight": i, **e} for i, e in sorted(self.plan.items())], "frozen": sorted(self.frozen),
                "notices": self.log, "freezes": self.freezes, "changes": self.changes, "refusals": self.refusals,
                "estimates": self.estimates, "audit": self.audit(), "grade": self.grade,
                "reward": self.grade["reward"] if self.grade else None}


def read_changes(task: Task, raw) -> tuple[list[tuple[int, dict]], list[str]]:
    """The plan entries ``raw`` asks for, by flight, and the problems with the entries left out."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ChangeError(f"changes are not valid JSON ({e.msg}): send a list like {USAGE}") from None
    if isinstance(raw, dict) and isinstance(raw.get("changes"), list):
        raw = raw["changes"]
    if not isinstance(raw, list):
        raise ChangeError(f"changes must be a list like {USAGE}")
    entries: dict[int, dict] = {}
    problems = []
    for c in raw:
        if not isinstance(c, dict):
            problems.append(f"not a change: {json.dumps(c)}")
            continue
        flight, action, nb = c.get("flight"), c.get("action"), c.get("not_before")
        i = _flight(task, flight)
        if i is None:
            problems.append(f"unknown flight {json.dumps(flight)}: use the ids from the flight table (f0, f1, ...)")
            continue
        if action not in ACTIONS:
            problems.append(f"{fid(i)}: action must be cancel, restore or hold, not {json.dumps(action)}")
            continue
        if action == "cancel":
            entry = dict(CANCEL)
        elif nb is None:
            entry = dict(FLY)
        else:
            try:
                minute = parse_hhmm(str(nb))
            except ValueError:
                problems.append(f"{fid(i)}: not_before {json.dumps(nb)} is not a time: write HH:MM, or HH:MM+1 after "
                                "midnight")
                continue
            due = task.flight(i).sched_dep
            if minute < due - EARLY:
                problems.append(f"{fid(i)}: not_before {nb} is {due - minute} minutes before its scheduled departure "
                                f"{hhmm(due)}; write {nb}+1 for after midnight")
                continue
            entry = {"cancel": False, "not_before": minute}
        if i in entries:
            problems.append(f"{fid(i)} appears more than once: the last change counts")
        entries[i] = entry
    return sorted(entries.items()), problems


def programs(news: Iterable[News], t: int) -> list[str]:
    """The FAA programs in effect or announced at t, from the advisories received: a later advisory on the same
    element replaces or cancels the earlier one."""
    current: dict[tuple, News] = {}
    for n in sorted((n for n in news if n.kind == "advisory"), key=lambda n: (n.at, n.id)):
        kind, element = n.data.get("kind"), n.data.get("element")
        for cleared in CLEARS.get(kind, ()):
            current.pop((cleared, element), None)
        if kind in PROGRAM_KINDS:
            current[(kind, element)] = n
        elif kind == "other":
            current[(kind, element, n.data.get("title"))] = n
    out = []
    for n in sorted(current.values(), key=lambda n: (n.at, n.id)):
        d = n.data
        period = next((d[k] for k in PERIODS[d["kind"]] if d.get(k)), None)
        if period and period[1] < t:
            continue
        element = d.get("element") or ""
        title = (d.get("title") or "advisory").removeprefix(element).strip().lower()
        line = f"- {element} {PROGRAM_KINDS.get(d['kind'], title)}"
        if period:
            line += f" {hhmm(period[0])}-{hhmm(period[1])}"
        if rates := d.get("program_rate"):
            line += f", rate {'/'.join(map(str, rates))} arrivals/h (all carriers)"
        if d.get("avg_delay") is not None:
            line += f", average delay {d['avg_delay']} min, max {d.get('max_delay')}"
        if d.get("probability_of_extension"):
            line += f", probability of extension {d['probability_of_extension'].lower()}"
        if d.get("condition"):
            line += f" ({d['condition'].lower()})"
        out.append(line + f" [advisory {d.get('advzy')}, {hhmm(n.at)}]")
    return out


def changes_to(day: Day, target: Mapping) -> list[dict]:
    """The changes that make the day's plan ``target`` ({flight id: {"cancel", "not_before"}}) on the flights
    that aren't frozen."""
    want = {int(i): _entry(e) for i, e in target.items()}
    rows = []
    for i in sorted(set(day.plan) | set(want)):
        a, b = _entry(day.plan.get(i)), want.get(i, FLY)
        if i in day.frozen or a == b:
            continue
        action = "cancel" if b["cancel"] else "restore" if a["cancel"] else "hold"
        rows.append({"flight": fid(i), "action": action, "not_before": _time(b["not_before"])})
    return rows


def play(task: Task, policy: Callable[[Day], Mapping]) -> Day:
    """The day a policy (the plan it wants at each watch) plays through confirm and advance, news delivered as the
    triggers deliver it."""
    day = Day(task)
    while True:
        if rows := changes_to(day, policy(day)):
            day.confirm(rows)
        if day.last:
            day.finish("done")
            return day
        day.open()
        for n in day.watches[day.watch].news:
            day.notice(n, "trigger")


def _flight(task: Task, value) -> int | None:
    text = str(value).strip().lower()
    text = text[1:] if text.startswith("f") else text
    return int(text) if text.isdigit() and int(text) in task.by_id else None


def _entry(e: Mapping | None) -> dict:
    if not e:
        return dict(FLY)
    cancel = bool(e.get("cancel"))
    return {"cancel": cancel, "not_before": None if cancel else e.get("not_before")}


def _time(minute: int | None) -> str | None:
    return None if minute is None else hhmm(minute)


def _plan_text(e: Mapping | None) -> str:
    e = _entry(e)
    return "cancel" if e["cancel"] else "-" if e["not_before"] is None else f"hold@{hhmm(e['not_before'])}"


def _proj(r: FlightRun) -> str:
    if r.status == FLOWN:
        return f"{hhmm(r.dep)}-{hhmm(r.arr)} +{r.arr_delay} {r.tail}"
    return "cancelled - -" if r.status == CANCELLED else "auto-cancel - -"


def _auto(run: Execution) -> list[str]:
    return [fid(i) for i, r in run.flights.items() if r.status == AUTO_CANCELLED]


def _news_text(n: News) -> str:
    return f"[{hhmm(n.at)}] {n.speaker}: {n.text}"


def _weekday(day: str) -> str:
    return date.fromisoformat(day).strftime("%A")
