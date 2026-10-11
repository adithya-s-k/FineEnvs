"""One live week, pure and deterministic: the week as known so far, the freeze, excuses, the audit and the grade."""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

from berth_core import Plan, PlanError, Task, evaluate, parse_plan, plan_to_list, situation
from berth_core.check import PlanResult, ShipResult, unavoidable_cost
from berth_core.model import MOVE_PENALTY
from berth_core.reward import score_v3
from berth_core.solve import _Ledger, naive_replan

from .schedule import FREEZE_HOURS, HARBOUR_MASTER, PLANNING_CALLS, label, schedule

RULES = [("missing", r"^missing from the plan$"), ("no_move", r"inside the no-movement window"),
         ("cranes", r"^gets -?\d+ cranes"), ("arrival", r"cannot arrive before hour"),
         ("quay", r"but the quay has sections"), ("closed", r"^overlaps closed sections"),
         ("alongside", r"\(alongside\)"), ("ship", r"^overlaps ship (\d+) "), ("crane_pool", r"^cranes over the pool"),
         ("moves", r"ships move \(limit"), ("pilots", r"^pilots short at hour"), ("tugs", r"^tugs short at hour")]
NOTE = "This watch's bulletin arrives with your next tool result."


def known(task: Task, revealed: Iterable[str]) -> Task:
    ids = set(revealed)
    shown = [i for i, e in enumerate(task.disruptions) if f"{e['type']}-{i}" in ids]
    events = [task.disruptions[i] for i in shown]
    hidden = {e["ship"] for i, e in enumerate(task.disruptions) if e["type"] == "extra" and i not in shown}
    original = {}
    for e in task.disruptions:
        if e["type"] == "late":
            original[e["ship"]] = task.ships[e["ship"]].arrival - e["hours"]
        elif e["type"] == "bunching":
            original |= {s: task.ships[s].planned_hour for s in e["ships"]}
    ships = [replace(s, arrival=original.get(s.id, s.arrival), weight=1, berth_deadline=None, deadline_penalty=0)
             for s in task.ships if s.id not in hidden]
    for e in events:
        if e["type"] == "late":
            ships[e["ship"]].arrival = task.ships[e["ship"]].arrival
        elif e["type"] == "bunching":
            for s in e["ships"]:
                ships[s].arrival = task.ships[s].arrival
        elif e["type"] == "emergency":
            ships[e["ship"]].berth_deadline = task.ships[e["ship"]].berth_deadline
            ships[e["ship"]].deadline_penalty = task.ships[e["ship"]].deadline_penalty
        elif e["type"] == "priority":
            ships[e["ship"]].weight = task.ships[e["ship"]].weight
    closures = {(e["first"], e["last"], e["start"], e["end"]) for e in events if e["type"] == "closure"}
    outages = {(e["start"], e["end"], e["cranes"]) for e in events if e["type"] == "crane_outage"}
    windows = {w for e in events if e["type"] == "gale"
               for w in ((e["start"], e["end"]), (e["core_start"], e["core_end"]))}
    rules = dict(task.rules, no_moves=[w for w in task.rules["no_moves"] if (w["start"], w["end"]) in windows],
                 crane_outages=[o for o in task.rules["crane_outages"]
                                if (o["start"], o["end"], o["cranes"]) in outages])
    blocks = [b for b in task.blocks if b.kind == "alongside" or (b.first, b.last, b.start, b.end) in closures]
    return replace(task, ships=ships, blocks=blocks, rules=rules, notices=[task.notices[i] for i in shown],
                   disruptions=events, reference={})


def rule(problem: str) -> str:
    for key, pattern in RULES:
        if m := re.search(pattern, problem):
            return f"ship {m[1]}" if key == "ship" else key


def excuse(before: Task, after: Task, frozen: Plan, evaluate: Callable[[Task, Plan], PlanResult] = evaluate
           ) -> tuple[list[tuple[int, str, str]], dict[int, int]]:
    """What the news alone did to the frozen entries: the (ship, rule) problems and the cost it added."""
    was = {r.ship: r for r in evaluate(before, frozen).ships if r.ship in frozen}
    problems, cost = [], {}
    for r in evaluate(after, frozen).ships:
        if r.ship in frozen:
            seen = {rule(p) for p in was[r.ship].problems}
            problems += [(r.ship, rule(p), p) for p in r.problems if rule(p) not in seen]
            if r.cost > was[r.ship].cost:
                cost[r.ship] = r.cost - was[r.ship].cost
    return problems, cost


def charge(res: PlanResult, excused_problems: Iterable[tuple[int, str]],
           excused_cost: Mapping[int, int]) -> list[tuple[ShipResult, list[str], list[str], int]]:
    """Each ship's result with its charged problems, its excused problems and its excused cost, as the grade counts
    them."""
    allowed = set(excused_problems)
    return [(r, [p for p in r.problems if (r.ship, rule(p)) not in allowed],
             [p for p in r.problems if (r.ship, rule(p)) in allowed], min(r.cost, excused_cost.get(r.ship, 0)))
            for r in res.ships]


def grade_week(task: Task, plan: Plan, excused_problems: Iterable[tuple[int, str]], excused_cost: Mapping[int, int],
               evaluate: Callable[[Task, Plan], PlanResult] = evaluate) -> dict:
    ships = charge(evaluate(task, plan), excused_problems, excused_cost)
    violations = [{"ship": r.ship, "problem": p} for r, charged, _, _ in ships for p in charged]
    excused = [{"ship": r.ship, "problem": p} for r, _, forgiven, _ in ships for p in forgiven]
    feasible = not violations
    clean = sum(not charged for _, charged, _, _ in ships) / len(task.ships)
    raw_cost = sum(r.cost for r, *_ in ships)
    waived = sum(w for *_, w in ships)
    cost = raw_cost - waived if feasible else None
    optimal, floor = task.reference["optimal_cost"], unavoidable_cost(task)
    reward, quality = score_v3(cost, feasible, clean, optimal, int(task.rules.get("gap_k", 100)), floor)
    return {"reward": round(reward, 6), "feasible": feasible, "clean_fraction": round(clean, 4),
            "quality": round(quality, 6), "cost": cost, "raw_cost": raw_cost, "excused_cost": waived,
            "optimal_cost": optimal, "unavoidable_cost": floor, "regret": cost - optimal if feasible else None,
            "violations": violations, "excused": excused}


class Week:
    evaluate = staticmethod(evaluate)
    notice_excuse = True

    def __init__(self, task: Task):
        self.task = task
        self.watches = schedule(task)
        self.scheduled = {n.event_id: n for w in self.watches for n in w.notices}
        self.watch = 0
        self.revealed: list[str] = []
        self.plan: Plan = {}
        self.calls = 0
        self.planning_calls = [0]
        self.log: list[dict] = []
        self.frozen = [{"watch": 0, "hour": 0, "before": 0, "call": 0, "ships": []}]
        self.refusals: list[dict] = []
        self.excused_problems: list[dict] = []
        self.excused_cost: list[dict] = []
        self.messages: list[dict] = []
        self.done = False
        self.end_reason: str | None = None
        self.grade: dict | None = None
        for n in self.watches[0].notices:
            self.notice(n.event_id, n.name, n.text, "load")

    @property
    def hour(self) -> int:
        return self.watches[self.watch].hour

    @property
    def before(self) -> int:
        return 0 if self.watch == 0 else self.hour + FREEZE_HOURS

    @property
    def last(self) -> bool:
        return self.watch == len(self.watches) - 1

    @property
    def view(self) -> Task:
        return known(self.task, self.revealed)

    @property
    def fixed(self) -> Plan:
        return {} if self.watch == 0 else {s: e for s, e in self.plan.items() if e[0] < self.before}

    @property
    def calls_left(self) -> int:
        return PLANNING_CALLS - self.planning_calls[-1]

    def windows(self, view: Task) -> list[dict]:
        fixed = self.fixed
        departures = {r.ship: r.departure for r in self.evaluate(view, self.plan).ships}
        rows = []
        for sid, (h, sec, cranes) in sorted(self.plan.items()):
            dep = departures[sid]
            status = ("open" if sid not in fixed else "departed" if dep <= self.hour
                      else "berthed" if h <= self.hour else "frozen")
            rows.append({"ship": sid, "berth_hour": h, "section": sec, "cranes": cranes, "departure": dep,
                         "status": status})
        return rows

    def unconfirmed(self, view: Task) -> list[int]:
        return [s.id for s in view.ships if s.id not in self.plan]

    def situation(self) -> dict:
        view = self.view
        return {"watch": self.watch, "hour": self.hour, "time": label(self.hour), "frozen_before": self.before,
                "situation": situation(view), "windows": self.windows(view), "unconfirmed": self.unconfirmed(view),
                "planning_calls_left": self.calls_left}

    def _planning_call(self) -> dict | None:
        if not self.calls_left:
            return {"error": f"no planning calls left in this watch ({PLANNING_CALLS} used): call advance",
                    "planning_calls_left": 0}
        self.planning_calls[-1] += 1
        return None

    def _merge(self, raw) -> tuple[Task, Plan, list[int], list[int], list[dict], list[str]]:
        view = self.view
        parsed, entry_problems = parse_plan(view, raw)
        fixed, plan = self.fixed, dict(self.plan)
        confirmed, unchanged, refused = [], [], []
        for sid, entry in sorted(parsed.items()):
            name = view.ships[sid].name
            if self.plan.get(sid) == entry:
                unchanged.append(sid)
            elif sid in fixed:
                refused.append({"ship": sid, "from": HARBOUR_MASTER, "reason": (
                    f"ship {sid} {name} is frozen: its window starts at hour {fixed[sid][0]}, before the freeze line "
                    f"at hour {self.before}; it stands.")})
            elif self.watch >= 1 and entry[0] < self.before:
                refused.append({"ship": sid, "from": HARBOUR_MASTER, "reason": (
                    f"ship {sid} {name} can't start at hour {entry[0]}: windows must start at or after the freeze "
                    f"line at hour {self.before}.")})
            else:
                plan[sid] = entry
                confirmed.append(sid)
        return view, plan, confirmed, unchanged, refused, entry_problems

    def check(self, raw) -> dict:
        if error := self._planning_call():
            return error
        try:
            view, plan, _, _, refused, entry_problems = self._merge(raw)
        except PlanError as e:
            return {"error": str(e), "planning_calls_left": self.calls_left}
        ships = charge(self.evaluate(view, plan), *self.excuses())
        rows = []
        for r, charged, forgiven, waived in ships:
            if r.berth_hour is None:
                rows.append({"ship": r.ship, "problems": charged})
            elif r.problems or r.cost > 0:
                rows.append({"ship": r.ship, "berth_hour": r.berth_hour, "section": r.section, "cranes": r.cranes,
                             "departure": r.departure, "delay_h": r.delay_h, "moved": r.moved, "cost": r.cost,
                             "problems": charged}
                            | ({"excused": forgiven, "excused_cost": waived} if forgiven or waived else {}))
        feasible = not any(charged for _, charged, _, _ in ships)
        excused_cost = sum(w for *_, w in ships)
        moves = sum(r.moved for r, *_ in ships)
        cost = sum(r.cost for r, *_ in ships) - excused_cost
        return {"feasible": feasible, "cost": cost if feasible else None,
                "delay_cost": cost - MOVE_PENALTY * moves if feasible else None, "moves": moves if feasible else None,
                "excused_cost": excused_cost, "ships": rows, "refused": refused, "entry_problems": entry_problems,
                "planning_calls_left": self.calls_left}

    def confirm(self, raw) -> dict:
        if error := self._planning_call():
            return error
        try:
            _, self.plan, confirmed, unchanged, refused, entry_problems = self._merge(raw)
        except PlanError as e:
            return {"error": str(e), "planning_calls_left": self.calls_left}
        self.refusals += [{"watch": self.watch, "ship": r["ship"], "reason": r["reason"]} for r in refused]
        return {"confirmed": confirmed, "unchanged": unchanged, "refused": refused, "entry_problems": entry_problems,
                "planning_calls_left": self.calls_left}

    def open(self) -> dict:
        self.watch += 1
        self.planning_calls.append(0)
        self.frozen.append({"watch": self.watch, "hour": self.hour, "before": self.before, "call": self.calls,
                            "ships": sorted(self.fixed)})
        view = self.view
        windows = self.windows(view)
        return {"watch": self.watch, "hour": self.hour, "time": label(self.hour),
                "berthed": [w["ship"] for w in windows if w["status"] == "berthed"],
                "departed": [w["ship"] for w in windows if w["status"] == "departed"],
                "frozen_before": self.before, "unconfirmed": self.unconfirmed(view), "note": NOTE}

    def notice(self, event_id: str, name: str, text: str, via: str) -> dict:
        if event_id not in self.scheduled:
            raise ValueError(f"unknown event {event_id!r}")
        if self.done:
            raise ValueError("the week is over")
        scheduled = self.scheduled[event_id]
        self.log.append({"event_id": event_id, "kind": scheduled.kind, "name": name, "watch": self.watch,
                         "hour": self.hour, "via": via, "call": self.calls,
                         "matches": name == scheduled.name and text == scheduled.text})
        if event_id not in self.revealed:
            fixed, before = self.fixed, self.view
            self.revealed.append(event_id)
            if fixed and self.notice_excuse:
                problems, cost = excuse(before, self.view, fixed, evaluate=self.evaluate)
                self.excused_problems += [{"event_id": event_id, "ship": s, "rule": r, "problem": p}
                                          for s, r, p in problems]
                self.excused_cost += [{"event_id": event_id, "ship": s, "cost": c} for s, c in cost.items()]
        if via != "load":
            self.messages.append({"hour": self.hour, "from": name, "text": text})
        return {"applied": event_id, "watch": self.watch}

    def excuses(self) -> tuple[set[tuple[int, str]], dict[int, int]]:
        cost: dict[int, int] = {}
        for c in self.excused_cost:
            cost[c["ship"]] = cost.get(c["ship"], 0) + c["cost"]
        return {(p["ship"], p["rule"]) for p in self.excused_problems}, cost

    def finish(self, end_reason: str) -> dict:
        self.done, self.end_reason = True, end_reason
        self.grade = grade_week(self.task, self.plan, *self.excuses(), evaluate=self.evaluate)
        return {"done": True, "feasible": self.grade["feasible"], "cost": self.grade["cost"],
                "reward": self.grade["reward"]}

    def end(self) -> dict:
        if not self.done:
            while not self.last:
                self.open()
                for n in self.watches[self.watch].notices:
                    self.notice(n.event_id, n.name, n.text, "env")
            self.finish("end_week")
        return {"end_reason": self.end_reason, "watch": self.watch, "feasible": self.grade["feasible"],
                "cost": self.grade["cost"], "reward": self.grade["reward"], "audit": self.audit()}

    def drain(self) -> list[dict]:
        messages, self.messages = self.messages, []
        return messages

    def audit(self) -> dict:
        problems = []
        for k in range(self.watch + 1):
            got = [e["event_id"] for e in self.log if e["watch"] == k]
            expected = [n.event_id for n in self.watches[k].notices]
            if got != expected:
                problems.append(f"watch {k} got {got} instead of {expected}")
        problems += [f"{e['event_id']} arrived after the agent's next call in watch {e['watch']}" for e in self.log
                     if e["via"] == "trigger" and e["call"] != self.frozen[e["watch"]]["call"]]
        problems += [f"{e['event_id']}: the message differs from the schedule" for e in self.log if not e["matches"]]
        for e in self.log:
            lead = self.scheduled[e["event_id"]].event_hour - e["hour"]
            if e["hour"] > 0 and lead < FREEZE_HOURS:
                problems.append(f"{e['event_id']}: revealed {lead} h ahead at hour {e['hour']}")
        if self.done:
            for event_id in self.scheduled:
                n = sum(e["event_id"] == event_id for e in self.log)
                if n != 1:
                    problems.append(f"{event_id} revealed {n} times")
        return {"ok": not problems, "problems": problems}

    def data(self) -> dict:
        return {"task_id": self.task.task_id, "watches": len(self.watches), "watch": self.watch, "hour": self.hour,
                "done": self.done, "end_reason": self.end_reason, "calls_used": self.calls,
                "planning_calls": self.planning_calls, "plan": plan_to_list(self.plan), "notices": self.log,
                "frozen": self.frozen, "refusals": self.refusals,
                "excused": {"problems": self.excused_problems, "cost": self.excused_cost}, "audit": self.audit(),
                "grade": self.grade, "reward": self.grade["reward"] if self.grade else None}


Policy = Callable[[Week], Plan]


def _published(s) -> tuple[int, int]:
    return (s.planned_hour if s.planned_hour is not None else s.arrival, s.id)


def naive(week: Week) -> Plan:
    """Keep each window that still fits; move the rest to the earliest legal hour after the freeze line, on the
    nearest section, with the same cranes."""
    view = week.view
    if week.watch == 0:
        return naive_replan(view)
    fixed = week.fixed
    led = _Ledger(view)
    for sid, entry in fixed.items():
        led.add(view.ships[sid], *entry)
    plan = dict(fixed)
    for s in sorted(view.ships, key=_published):
        if s.id in fixed:
            continue
        cur = week.plan.get(s.id)
        if cur and cur[0] >= max(s.arrival, week.before) and led.fits(s, *cur):
            led.add(s, *cur)
            plan[s.id] = cur
            continue
        cranes = cur[2] if cur else s.std_cranes
        anchor = cur[1] if cur else (s.planned_section if s.planned_section is not None else view.first_section)
        h = max(s.arrival, week.before, cur[0] if cur else 0)
        sections = sorted(range(view.first_section, view.last_section - s.sections + 2),
                          key=lambda k: (abs(k - anchor), k))
        while (sec := next((k for k in sections if led.fits(s, h, k, cranes)), None)) is None:
            h += 1
        led.add(s, h, sec, cranes)
        plan[s.id] = (h, sec, cranes)
    return plan


def play(task: Task, policy: Policy, week: type[Week] = Week) -> Week:
    week = week(task)
    while True:
        target = policy(week)
        if rows := [r for r in plan_to_list(target) if week.plan.get(r["ship"]) != target[r["ship"]]]:
            week.confirm(rows)
        if week.last:
            week.finish("done")
            return week
        week.open()
        for n in week.watches[week.watch].notices:
            week.notice(n.event_id, n.name, n.text, "trigger")
