"""The day under the two desks' plan, as the env runs it: the truth (the task's day, the standard crew desk crewing what
the crew desk leaves alone and, unable to cancel, asking it to), its history before a minute (what a decision then
must not change), and the projection at a watch: the truth played to the watch and carried on from there in the day as
known (sim.known), so the history is the truth's by construction, with the crew flags per kind it implies."""

from dataclasses import dataclass

from ..core.known import FREEZE
from ..core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from ..programs import crews, crewsim, occ, sim
from ..programs.crewsim import CrewDay
from .pack import World

LEGAL, NO_CREW = "legal", "no crew"
SOON = 60


@dataclass(frozen=True)
class Projection:
    """The day as the desks see it at minute t: `day` played on the truth to t and on the day as known after, the
    units and aircraft as they stand at t."""

    t: int
    day: CrewDay
    units: dict[str, dict]
    aircraft: dict[str, tuple[str, int, bool]]


def truth(w: World, plan: dict, world: sim.World) -> CrewDay:
    return crewsim.execute(w.task, plan, w.units, world, w.local, airports=w.airports, desk=True, drop=False)


def project(w: World, plan: dict, t: int, world: sim.World, known: sim.World) -> Projection:
    pool, day = crewsim.start(w.task, plan, w.units, world, w.local, airports=w.airports, desk=True, drop=False)
    day.until(t)
    units = {uid: standing(s, w.task.start) for uid, s in pool.state.items()}
    aircraft = {}
    for (_, station), tails in day.at.items():
        for p in tails:
            legs = day.legs[p]
            aircraft[p] = (station, day.ready[p], bool(legs) and day.runs[legs[-1]].arr > t)
    day.switch(known, t)
    return Projection(t, crewsim.finish(pool, day), units, aircraft)


def standing(s: crewsim.State, start: int) -> dict:
    """A unit as it stands: where, free from when, its duty so far and its legal limit, its trip and deadheads left,
    and as a reserve whether it was called and until when it may be."""
    u = s.unit
    out = {"station": s.station, "ready": s.ready, "report": s.report, "flown": s.flown, "minutes": s.minutes,
           "period": s.period, "trip": list(s.trip), "deadheads": list(s.deadheads), "held": s.claim,
           "call": s.call, "landed": s.landed, "release": s.release(start), "limit": None}
    if s.report is not None:
        out["limit"] = crews.pilot_limit(s.report, s.hour, s.flown + 1, s.rap) if u.kind == "pilot" else \
            s.report + crews.CABIN_DUTY
    if u.reserve == "short" and s.report is None:
        out["earliest"] = s.earliest
    return out


def history(day: CrewDay, t: int) -> tuple:
    """What the day did before t (occ.past's departures, cancellations, crew flags and messages), with the crews and
    riders of the flights gone and each unit's duties so far and its call (a reserve called but released before it
    reported is logged as never called, as crewsim does)."""
    gone = {i for i, x in day.run.flights.items() if x.dep is not None and x.dep < t}
    duties = {}
    for uid, rec in sorted(day.units.items()):
        done = []
        for d in rec["duties"]:
            legs = [x for x in d["legs"] if x[1] < t]
            rides = [x for x in d["deadheads"] if x[1] < t]
            if legs or rides:
                done.append((d["report"], d["at"], d["begins"], tuple(legs), tuple(rides)))
        if done:
            duties[uid] = (rec["call"], tuple(done))
    return (occ.past(day, t), tuple(sorted((i, tuple(sorted(c.items()))) for i, c in day.crews.items() if i in gone)),
            tuple(sorted((i, r) for i, r in day.riders.items() if i in gone)), tuple(sorted(duties.items())))


def acted(day: CrewDay, t: int) -> tuple:
    """What the day did before t that no decision can take back: `history` less the crew flags and messages."""
    past, *rest = history(day, t)
    return (past[:2], *rest)


def flags(p: Projection, task) -> dict[int, dict[str, tuple[str, int | None]]]:
    """Per open flight and kind its crew flag on the projection: "no crew" where it auto-cancels having waited for a
    crew of the kind, "crew" (the minute it leaves) where it leaves after waiting for one, "times out" (the crew's last
    legal departure, with the pilots' extension) where it leaves within SOON of that, else "legal"."""
    day, tr = p.day, p.day.trace
    out = {}
    for i, r in day.run.flights.items():
        if r.status == CANCELLED or (r.dep is not None and r.dep < p.t) or (r.status == AUTO_CANCELLED and
                                                                           r.decided < p.t):
            continue
        kinds = {}
        for kind in crews.KINDS:
            waited = (i, kind) in tr["waited"]
            if r.status != FLOWN:
                kinds[kind] = (NO_CREW, None) if waited else (LEGAL, None)
                continue
            last = tr["limits"].get(i, {}).get(kind, (None, None, None))[2]
            if waited and tr["waited"][(i, kind)] < r.dep:
                kinds[kind] = ("crew", r.dep)
            elif last is not None and r.dep >= last - SOON:
                kinds[kind] = ("times out", last)
            else:
                kinds[kind] = (LEGAL, None)
        out[i] = kinds
    return out


def opened(day: CrewDay, i: int, t: int) -> bool:
    """Whether flight i is open at t on the day: not gone before t and not cancelled before t."""
    x = day.run.flights[i]
    return (x.dep is None or x.dep >= t) and (x.status == FLOWN or x.decided >= t)


def commits(task, day: CrewDay, plan: dict, t: int, known: sim.World) -> set[int]:
    """The flights ops commits at the end of watch t (occ.watch's rule): those the day as known, aircraft only, has
    leaving before t + FREEZE, the history kept and every open one no earlier than t; none at the first watch."""
    if t <= task.start:
        return set()
    view = {}
    for f in task.flights:
        x, e = day.run.flights[f.id], plan.get(f.id, {})
        if x.dep is not None and x.dep < t:
            view[f.id] = {"left": x.dep}
        elif e.get("cancel") or (x.status == CANCELLED and x.decided < t):
            view[f.id] = {"cancel": True, "decided": e.get("decided", x.decided)}
        else:
            view[f.id] = {"not_before": max(sim.hold(e, t), t), **({"claim": c} if (c := sim.claim_at(e, t)) else {})}
    run = sim.execute(task, view, known)
    return {i for i, x in run.flights.items() if x.status == FLOWN and x.dep < t + FREEZE}
