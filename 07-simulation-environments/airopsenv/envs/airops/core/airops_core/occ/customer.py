"""The customer desk (v3): the day's passengers on the projection at the watch, for the customer desk's page and its
lookups, and the passenger counts the other desks read. Rebooking is the passenger system's own, automatic (pax.fly: the
default rebooking, earliest arrival first, first come first served); the customer desk asks ops to hold a departure for
connecting passengers, to protect a flight or to cancel one, and texts. Nothing here reads the day past the watch: the
projection is the day as known (run.project), and the bookings and the following days' seats are the airline's own."""

from collections import Counter, defaultdict
from dataclasses import replace

from ..core.model import hhmm
from ..core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from ..programs import pax
from ..programs.cost import RATES, REFUND, late
from . import pages, run

AHEAD, CANCEL_AHEAD, MARGIN, WORST, LATE_CANCEL = 240, 360, 15, 120, 180
RISK_ROWS, CANCEL_ROWS, DETAIL_ROWS = 15, 10, 12
SAME_DAY, LATER, REFUNDS = "same day", "later day", "refund"


class Bookings:
    """A day's bookings by flight: booked, the connections onto it and off it, its passengers by final destination,
    and its seats free as booked (its recorded tail's seats, the family's smallest where unknown, less its booked)."""

    def __init__(self, task, passengers: pax.Passengers):
        self.task, self.passengers = task, passengers
        self.booked = passengers.booked()
        self.into: dict[int, Counter] = defaultdict(Counter)
        self.onward: dict[int, Counter] = defaultdict(Counter)
        self.dest: dict[int, Counter] = defaultdict(Counter)
        for g in passengers.groups:
            d = task.flight(g.legs[-1]).dest
            for k in g.legs:
                self.dest[k][d] += g.pax
            if len(g.legs) == 2:
                self.into[g.legs[1]][g.legs[0]] += g.pax
                self.onward[g.legs[0]][g.legs[1]] += g.pax
        least, tails = pax.smallest(task), {t.tail: t.seats for t in task.tails}
        self.free = {f.id: tails.get(f.actual.tail, least[f.family]) - self.booked[f.id] for f in task.flights}
        self.table = pax.Timetable(task.flights)

    def pax_text(self, i: int) -> str:
        """Booked, and of them connecting onto it and off it."""
        return f"{self.booked[i]}({sum(self.into[i].values())}/{sum(self.onward[i].values())})"


def riders(day, proj) -> dict[int, int]:
    return {i: sum(day.units[u].size for u in us) for i, us in proj.day.riders.items() if us}


def no_later(day, proj, i: int, dests: Counter | None = None) -> int:
    """Of flight i's passengers (or `dests`, passengers by final destination), those with no later JetBlue itinerary
    that day to their destination: none leaving after its scheduled departure, direct or through one connecting station
    as sold, every leg flying on the projection with a seat free as booked."""
    b, runs, f = day.book, proj.day.run.flights, day.task.flight(i)

    def ok(g) -> bool:
        return g.id != i and g.sched_dep > f.sched_dep and runs[g.id].status == FLOWN and b.free[g.id] > 0

    def then(g) -> bool:
        return g.id != i and runs[g.id].status == FLOWN and b.free[g.id] > 0
    return sum(n for d, n in (b.dest[i] if dests is None else dests).items()
               if not b.table.itineraries(f.orig, d, ok, then))


def if_cancelled(day, proj, i: int) -> Counter:
    """Where flight i's booked passengers would end up were it cancelled now, the rest of the day as projected, by the
    passenger system's default rebooking: same day, a later day, or refunded."""
    runs = dict(proj.day.run.flights)
    runs[i] = replace(runs[i], status=CANCELLED, tail=None, dep=None, arr=None, slot=None, program=None,
                      decided=day.t)
    out = pax.fly(day.task, day.w.passengers, runs, riders(day, proj))
    mine = {g.id for g in day.w.passengers.groups if i in g.legs}
    kinds = {pax.FLEW: SAME_DAY, pax.REBOOKED: SAME_DAY, pax.STRANDED: LATER, pax.REFUNDED: REFUNDS}
    got = Counter()
    for p in out.parts:
        if p.group in mine:
            got[kinds[p.kind]] += p.pax
    return got


def at_risk(day, proj) -> list[dict]:
    """Open departures leaving in the next AHEAD minutes on the projection whose booked connecting passengers make it
    with MARGIN minutes or less to spare, or miss it by WORST or less (a hold could still make them), both flights
    flying: each inbound with its passengers and margin, the minute a hold makes them all, and how many of those who
    miss have no later seat today."""
    t, runs, b = day.t, proj.day.run.flights, day.book
    out = []
    for j, ins in b.into.items():
        x = runs[j]
        if x.status != FLOWN or not t <= x.dep < t + AHEAD or day.gone(j) or day.cancelled(j):
            continue
        gap = pax.mct(day.task.flight(j).orig)
        rows = [(i, n, runs[i].arr, x.dep - runs[i].arr - gap) for i, n in sorted(ins.items())
                if runs[i].status == FLOWN and -WORST <= x.dep - runs[i].arr - gap <= MARGIN]
        if not rows:
            continue
        miss = sum(n for _, n, _, m in rows if m < 0)
        need = max((a + gap for _, _, a, m in rows if m < 0), default=None)
        stuck = no_later(day, proj, j, Counter({day.task.flight(j).dest: miss})) if miss else 0
        out.append({"flight": j, "dep": x.dep, "rows": rows, "miss": miss, "need": need, "stuck": stuck})
    return sorted(out, key=lambda r: (r["dep"], r["flight"]))


def might_cancel(day, proj, flags) -> list[tuple[int, str]]:
    """Open flights in the next CANCEL_AHEAD minutes ops might cancel: auto-cancelling on the projection, no crew, a
    cancellation asked, or LATE_CANCEL minutes late or more."""
    t, runs = day.t, proj.day.run.flights
    asked = {r["params"].get("flight") for r in day.requests if r["status"] == "open"
             and r["type"] in ("CANCEL", "CANCEL_PAIR", "SUGGEST_CANCEL")}
    out = []
    for f in sorted(day.task.flights, key=lambda f: (f.sched_dep, f.id)):
        i, x = f.id, runs[f.id]
        if day.gone(i) or day.cancelled(i) or not day.book.booked[i] or \
                min(f.sched_dep, x.dep if x.dep is not None else f.sched_dep) >= t + CANCEL_AHEAD:
            continue
        why = []
        if x.status == AUTO_CANCELLED:
            why.append(f"auto-cancels {hhmm(x.decided)}")
        if any(v[0] == run.NO_CREW for v in flags.get(i, {}).values()):
            why.append("no crew")
        if i in asked:
            why.append("cancel asked")
        if x.status == FLOWN and x.arr - f.sched_arr >= LATE_CANCEL:
            why.append(f"{x.arr - f.sched_arr} min late")
        if why:
            out.append((i, ", ".join(why)))
    return out


def outcomes_text(got: Counter) -> str:
    return f"{got[SAME_DAY]} {SAME_DAY} / {got[LATER]} {LATER} / {got[REFUNDS]} {REFUNDS}"


def cost_line(proj) -> str:
    c, d = proj.day.run.cost, proj.day.run.pax
    o = d.outcomes()
    return (f"Passengers as projected: delay {c.delay:,} pax-min, refund exposure {c.cancel:,} ({REFUND} a cancelled "
            f"passenger), nights away {c.lodging:,}, refunded fares {c.fare:,}; of {sum(o.values()):,} booked today "
            f"{d.misconnected():,} misconnect, {o[pax.REBOOKED]:,} rebooked the same day, {o[pax.STRANDED]:,} on a "
            f"later day, {o[pax.REFUNDED]:,} refunded.")


def risk_text(day, r: dict) -> str:
    f = day.task.flight(r["flight"])
    ins = ", ".join(f"{pages.fid(i)} {day.task.flight(i).orig} in {hhmm(a)} {m:+d} {n}" for i, n, a, m in r["rows"])
    out = (f"{pages.fid(f.id)} {f.flight.split()[-1]} {f.orig}-{f.dest} {hhmm(r['dep'])}"
           f"{' committed' if f.id in day.committed else ''}: {ins}")
    if r["miss"]:
        out += f"; misses {r['miss']}, hold to {hhmm(r['need'])} makes them, {r['stuck']} have no later seat today"
    return out


def cancel_text(day, proj, i: int, why: str) -> str:
    f = day.task.flight(i)
    return (f"{pages.fid(i)} {f.flight.split()[-1]} {f.orig}-{f.dest} {hhmm(f.sched_dep)} {day.book.pax_text(i)} "
            f"{why}; cancelled now: {outcomes_text(if_cancelled(day, proj, i))}; no later seat today "
            f"{no_later(day, proj, i)}")


def page(day, proj, flags) -> list[str]:
    """The customer desk's sections of a page (shorter for one agent at every desk)."""
    risk_rows, cancel_rows = (RISK_ROWS, CANCEL_ROWS) if day.mode != "one" else (RISK_ROWS // 2, CANCEL_ROWS // 2)
    risks = at_risk(day, proj)
    out = ["## Connections at risk: departures in the next 4 h whose booked connecting passengers make it with 15 min "
           "or less, or miss it by 2 h or less (negative), both flights flying as projected",
           "id flight route dep [committed]: inbound station in(arrival) margin pax; misses, the hold that makes "
           "them, no later seat"]
    out += [risk_text(day, r) for r in risks[:risk_rows]] or ["None."]
    if len(risks) > risk_rows:
        out.append(f'{len(risks) - risk_rows} more: cust_flights(status="attention")')
    cuts = might_cancel(day, proj, flags)
    out += ["", "## Flights ops might cancel in the next 6 h: auto-cancelling, no crew, a cancellation asked, or 3 h "
                "late or more as projected",
            "id flight route sched pax(conn in/out) why; its booked passengers if cancelled now (the passenger "
            "system's rebooking); no later seat today"]
    out += [cancel_text(day, proj, i, why) for i, why in cuts[:cancel_rows]] or ["None."]
    if len(cuts) > cancel_rows:
        out.append(f'{len(cuts) - cancel_rows} more: cust_flights(status="attention")')
    return ["\n".join(out)]


def flights(day, ids, station, status, start, end) -> dict:
    """The customer desk's flight lookup: bookings, the projection, connections at risk and seats left."""
    proj = day.project(day.plan)
    if status is not None and status not in pages.STATUSES:
        return {"error": f"status is one of {', '.join(pages.STATUSES)}"}
    try:
        lo, hi = pages.window(start, None), pages.window(end, None)
    except ValueError as e:
        return {"error": str(e)}
    pick = None
    if ids is not None:
        try:
            pick = {int(str(x).strip().lower().lstrip("f")) for x in ids}
        except ValueError:
            return {"error": "ids are flight ids like f12"}
    flags = run.flags(proj, day.task)
    risky = {r["flight"]: r for r in at_risk(day, proj)}
    attention = set(risky) | {i for i, _ in might_cancel(day, proj, flags)}
    rows = []
    for f in sorted(day.task.flights, key=lambda f: (f.sched_dep, f.id)):
        i = f.id
        if pick is not None and i not in pick or station is not None and station not in (f.orig, f.dest):
            continue
        if lo is not None and f.sched_dep < lo or hi is not None and f.sched_dep > hi:
            continue
        gone, cancelled = day.gone(i), day.cancelled(i)
        state = "flown" if gone else "cancelled" if cancelled else "frozen" if i in day.committed else "open"
        if status == "attention" and i not in attention or status not in (None, "attention", state) and \
                not (status == "open" and state == "frozen"):
            continue
        x = proj.day.run.flights[i]
        line = (f"{pages.fid(i)} {f.flight.split()[-1]} {f.orig}-{f.dest} {hhmm(f.sched_dep)} {day.book.pax_text(i)} "
                f"{pages.proj_text(x, f)}")
        if i in risky and risky[i]["miss"]:
            line += f"; misses {risky[i]['miss']} connecting"
        if not gone and not cancelled:
            line += f"; no later seat {no_later(day, proj, i)}"
        rows.append(f"{line} [{state}]")
    return {"columns": "id flight route sched pax(conn in/out) proj(dep-arr +late); connecting who miss it; its "
                       "passengers with no later seat today [status]",
            **pages.fit(rows, "ids, station, status, start or end")}


def flight(day, i: int) -> dict:
    """One flight's passengers: bookings, connections onto it and off it on the projection, its passengers by final
    destination with the later itineraries today, where they would go were it cancelled now, and how its passengers
    fare as projected."""
    proj = day.project(day.plan)
    task, runs, b = day.task, proj.day.run.flights, day.book
    f, x = task.flight(i), runs[i]

    def conn(a: int, c: int, n: int) -> str:
        p, q = runs[a], runs[c]
        if p.status != FLOWN or q.status != FLOWN:
            return f"{pages.fid(a)}->{pages.fid(c)} {n}: {'cancelled' if p.status != FLOWN else 'onward cancelled'}"
        m = q.dep - p.arr - pax.mct(task.flight(c).orig)
        return f"{pages.fid(a)} in {hhmm(p.arr)} -> {pages.fid(c)} out {hhmm(q.dep)}: {n} pax, margin {m:+d}"
    later = []
    for d, n in b.dest[i].most_common(DETAIL_ROWS):
        found = b.table.itineraries(f.orig, d, lambda g: g.id != i and g.sched_dep > f.sched_dep and
                                    runs[g.id].status == FLOWN and b.free[g.id] > 0,
                                    lambda g: g.id != i and runs[g.id].status == FLOWN and b.free[g.id] > 0)
        nxt = " ".join("+".join(pages.fid(g.id) for g in legs) + f" {hhmm(legs[0].sched_dep)}" for legs in found[:3])
        later.append(f"{d} {n}: " + (f"{len(found)} later today, first {nxt}" if found else "no later seat today"))
    d = proj.day.run.pax
    mine = {g.id for g in day.w.passengers.groups if i in g.legs}
    fate = Counter()
    for p in d.parts:
        if p.group in mine:
            fate[p.kind] += p.pax
    out = {"flight": f"{pages.fid(i)} {f.flight} {f.orig}-{f.dest} {hhmm(f.sched_dep)}-{hhmm(f.sched_arr)}",
           "projected": pages.proj_text(x, f), "booked": b.booked[i],
           "local": b.booked[i] - sum(b.into[i].values()) - sum(b.onward[i].values()),
           "connecting_in": [conn(a, i, n) for a, n in sorted(b.into[i].items())][:DETAIL_ROWS],
           "connecting_out": [conn(i, c, n) for c, n in sorted(b.onward[i].items())][:DETAIL_ROWS],
           "by_destination": later,
           "as_projected": {k: fate[k] for k in (pax.FLEW, pax.REBOOKED, pax.STRANDED, pax.REFUNDED)}}
    if not day.gone(i) and not day.cancelled(i):
        out["if_cancelled_now"] = outcomes_text(if_cancelled(day, proj, i))
        out["cancel_now"] = f"{price(f, day.t):,} pax-min before its passengers' rebooking delays, nights and fares"
    return out


def price(f, t: int) -> int:
    """A cancellation now under P before its passengers' outcomes: their refund exposure and notice, less the
    saving."""
    return f.pax * (REFUND + late(f, t)) - RATES[f.family].saved * f.sched_block
