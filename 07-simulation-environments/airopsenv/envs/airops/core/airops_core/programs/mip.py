"""The hindsight MIP (HiGHS): v1's time-line network (refs.mip) by split family, inside the programs.

Each flight gets a copy per departure minute (every `step` minutes, at the EDCT of every slot into its destination, the
slot less its scheduled block, and at the end of its crew's rest) from its earliest to its deadline, the hub cutoff
included, less the minutes the world forbids: those a ground stop holds it, those at which a segment governs it with
its slot time (the minute plus the scheduled block) inside the period but on none of JetBlue's slots, and those after
its crew deadline before its crew's rest ends. A copy on a slot counts in that slot's row, at most one flight per slot;
any other copy into a hub counts in the nominal row of its arrival hour (the minute plus the block). A copy costs its
pax x minutes late at arrival and its aircraft's minutes (cost.flown_price: the minute's gate delay, and the world's
block overrun, the same for every copy of a flight); a cancellation costs `price`, the notice price at the solve time,
which keeps the model linear, less the cancellation's saving, and a caller pins a cancellation the sim decides (an
auto-cancel, `lapsed`) at the price the sim gives it (Task.auto_at). With `flat` every cancellation, the desk's or an
auto-cancel, costs the flat price less the saving, as Cost.flat prices it. Pinned flights keep their minute, their
cancellation or their auto-cancel, and use up the slots and nominal room they take.

Aircraft flow as in v1, over nodes per (family, station) at every event time; a copy of a leg with a busy block
reaches its destination's node at the block's ready time when that is later than its turn. A leg JetBlue turned into
its next one faster than the family's turn (Task.quick) lands, where that is earlier, on a lane that releases its
aircraft to the station's node after the turn or, at JetBlue's ground time, to a gate only that next leg's copies
draw from; the station's node feeds the gate at each of them. Moves are arcs on their
tail (sim.routes), no aircraft created or taken: each tail with a break has a track, a node per break and time. The
aircraft a break binds enters its track by a redirect out of the node its carrier lands at, allowed and then forced
when every leg between the carrier and the break is cancelled, or by the tail's own aircraft leaving its start when
every leg before the first break is. From a break it goes to the Move's destination and stays there, or on to the
next break when every leg between the two is cancelled, a row on those legs' cancel columns. A carrier JetBlue turned
fast into the leg its Moves deliver it to may also go from its lane straight to that leg's gate, at its turned time
through the Moves, under the same rows. On such a tail a leg the sim lets auto-cancel has its own column, at its auto
price: it leaves a leg not cancelled that carries no aircraft, and a plan holds the flight past its deadline for the
sim to decide it. Overnight nodes feed v1's ferries onto tomorrow's needs (Task.needed). Every solution is a day
sim.execute reproduces from its departure minutes as holds, at the same cost, and every day sim.execute runs is a
solution (`Network.vector`), which also starts the search.

Proven to GAP (0.01%); `canonical` then re-solves among the optimal plans for the one with the earliest weighted
departures (v1's second pass, started from the first), so the plan, not only its cost, is the same on every run. Within
the gap the first plan need not be the cheapest, so the second may cost less: the objective is the cost of the plan
returned. A second pass that runs out of time leaves the proven cost solve's status and plan, flagged as not
canonical. With `fix` a canonical solve first solves the LP relaxation: an integral LP optimum is the proven optimum,
and its reduced costs fix, for the second pass, every column no plan within the optimum's cost can move (an LP
duality bound, so the plan is the one the second pass would find without it).

With passengers (`pax`, a pax.Proxy) the model prices them as the proxy does. A copy carries the delay of the
passengers its flight takes to their destination; a cancellation (the desk's or an auto-cancel) its passengers'
refund exposure and notice. Each flight a booked connection uses gets its departure minute as a column (the sum of its
copies' minutes, which the cover row makes the minute it leaves, 0 when cancelled), and each booked connection a binary
y that its inbound flown and its outbound cancelled, y >= cancelled(j) - cancelled(i), or a gap under the MCT, dep_j -
dep_i - block_i + M y >= MCT (its cancelled terms raised so that it binds only when both fly, M the widest gap short
of the MCT their copies allow, no row where none falls short), force to 1. Valid cuts tighten the relaxation: with
columns counting each such flight's copies by minutes on a GRID-minute grid, y >= left(j, a) - left(i, a - MCT -
block_i, rounded up) - cancelled(i) for every grid minute a in the RISK minutes after the outbound's earliest one.
Every way a group is disrupted (its first leg's cancel column, its auto-cancel column, or its connection's y) rebooks
it, as a column per option, onto the proxy's itineraries within each flight's seats as booked (none on a cancelled
one) and the following days' seats left, or refunds it at the proxy's refund price (a last resort, pax.last_resort).
Rebooking costs at least 0, so a
y at 1 that nothing forces never lowers the cost: every day the sim runs is a solution at its proxy cost
(pax.Proxy.cost, the rebooking an LP), every optimum replays at its cost, and the bound is the proxy's.
"""

import time
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import highspy
import numpy as np

from ..core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from ..refs.mip import GAP, _Model
from . import sim
from .cost import cancel_price, ferry_cost, flat_price, flown_price
from .model import Flight, Tail, Task
from .pax import Proxy, mct

INF = highspy.kHighsInf
INTEGRAL = 1e-6
GRID, RISK = 5, 180


@dataclass(frozen=True)
class Solution:
    status: str
    objective: int
    bound: float
    gap_pct: float
    wall_s: float
    deps: dict[int, int | None]
    cols: int
    rows: int
    copies: int
    started: bool
    canonical: bool
    lapsed: dict[int, int]

    @property
    def proven(self) -> bool:
        return self.status == "Optimal" and self.gap_pct <= 100 * GAP

    def plan(self, decided: int) -> dict[int, dict]:
        """The solution as holds to each departure minute, its cancellations decided at `decided`; a flight it lets
        auto-cancel is held past its deadline."""
        return {i: {"not_before": self.lapsed[i]} if i in self.lapsed else {"cancel": True, "decided": decided}
                if d is None else {"not_before": d} for i, d in self.deps.items()}

    def to_dict(self) -> dict:
        return {"status": self.status, "objective": self.objective, "bound": round(self.bound, 1),
                "gap_pct": self.gap_pct, "wall_s": self.wall_s, "cols": self.cols, "rows": self.rows,
                "copies": self.copies, "started": self.started, "canonical": self.canonical}


def solve(task: Task, world: sim.World | None = None, price: Mapping[int, int] | None = None,
          fixed: Mapping[int, int | None] | None = None, earliest: int | None = None, step: int = 1,
          time_limit: float = 600.0, gap: float = GAP, threads: int = 1, canonical: bool = True,
          start: sim.Execution | None = None, lapsed: Collection[int] = (), fix: bool = True,
          flat: bool = False, pax: Proxy | None = None) -> Solution:
    """The optimum in `world` (the truth by default) with each cancel arc at `price` (default: notice-priced at the
    first watch, or flat with `flat`). `fixed` pins flights to a departure minute (None: cancelled, or auto-cancelled
    for those in `lapsed`); the others leave at or after `earliest`. `start` is a day sim.execute ran in `world` that
    the model holds, to start the search from; `pax` prices the passengers by their proxy."""
    net = Network(task, world, price, fixed, earliest, step, lapsed, flat, pax)
    return net.solve(time_limit, gap, threads, canonical, None if start is None else net.vector(start), fix)


@dataclass(frozen=True)
class Line:
    """A tail with a break: its legs in order, and the positions of those a Move follows."""

    tail: Tail
    legs: tuple[int, ...]
    breaks: tuple[int, ...]

    def first(self, p: int) -> int | None:
        """The first break at or after position p."""
        return next((e for e, k in enumerate(self.breaks) if k >= p), None)

    def after(self, e: int) -> tuple[int, ...]:
        """The legs between break e and the next break, through the leg that one follows, or the last leg."""
        return self.legs[self.breaks[e] + 1:self.breaks[e + 1] + 1 if e + 1 < len(self.breaks) else len(self.legs)]


class Network:
    """The MIP of a task in a world, its columns kept so that any day the sim runs can be written as a solution."""

    def __init__(self, task: Task, world: sim.World | None = None, price: Mapping[int, int] | None = None,
                 fixed: Mapping[int, int | None] | None = None, earliest: int | None = None, step: int = 1,
                 lapsed: Collection[int] = (), flat: bool = False, pax: Proxy | None = None):
        self.task, self.pax = task, pax
        self.world = world = sim.truth(task) if world is None else world
        if price is None:
            price = {f.id: pax.price(f, task.start, flat) if pax else flat_price(f) if flat else
                     cancel_price(f, task.start) for f in task.flights}
        fixed = fixed or {}
        rules = sim.Rules(task, world)
        pools = {s.id: set(s.slots) for segs in world.segments.values() for s in segs}
        self.m = m = _Started()
        copies: dict[int, list[tuple[int, tuple]]] = {}
        events: dict[tuple[str, str], set[int]] = defaultdict(set)
        for f in task.flights:
            if f.id in fixed:
                d = fixed[f.id]
                copies[f.id] = [] if d is None else [(d, _place(f, d, rules, pools) or ("free", None))]
            else:
                lo = max(task.earliest(f, world.lb[f.id]), earliest or 0)
                jumps = {s - f.sched_block for seg in world.segments.get(f.dest, ()) for s in seg.slots}
                if f.crew and f.crew.rest is not None:
                    jumps.add(f.crew.rest)
                minutes = set(range(lo, f.deadline + 1, step)) | {d for d in jumps if lo <= d <= f.deadline}
                copies[f.id] = [(d, at) for d in sorted(minutes)
                                if rules.held(f, d) == d and f.legal(d) == d and (at := _place(f, d, rules, pools))]
            for d, _ in copies[f.id]:
                events[(f.family, f.orig)].add(d)
                events[(f.family, f.dest)].add(task.back(f, d + world.block[f.id]))
        for t in task.tails:
            events[(t.family, t.pos)].add(t.ready)
        self.lines = lines = [Line(t, t.legs, tuple(p for p, i in enumerate(t.legs) if i in task.moved))
                              for t in task.tails if any(i in task.moved for i in t.legs)]
        ready = {(ln.tail.tail, e): task.moved[ln.legs[k]].ready for ln in lines for e, k in enumerate(ln.breaks)}
        lands: dict[int, dict[int, list[int]]] = {f.id: defaultdict(list) for f in task.flights}
        for f in task.flights:
            for d, _ in copies[f.id]:
                lands[f.id][task.back(f, d + world.block[f.id])].append(d)
        transit: dict[tuple[str, int], list[int]] = {}
        for ln in lines:
            name, fam, last = ln.tail.tail, ln.tail.family, set()
            for e, k in enumerate(ln.breaks):
                r = ready[(name, e)]
                ts = {max(t, r) for t in last} | {max(t, r) for p in range(ln.breaks[e - 1] + 1 if e else 0, k + 1)
                                                  for t in lands[ln.legs[p]]}
                if not e:
                    ts.add(max(ln.tail.ready, r))
                events[(fam, task.moved[ln.legs[k]].dest)] |= ts
                transit[(name, e)] = sorted(ts)
                last = ts
        self.times = {key: sorted(ts) for key, ts in events.items()}
        node = {(key, x): None for key, ts in self.times.items() for x in ts}
        node = {k: n for n, k in enumerate(node)}
        over = {key: len(node) + n for n, key in enumerate(self.times)}
        track = {(key, x): len(node) + len(over) + n
                 for n, (key, x) in enumerate((key, x) for key, ts in transit.items() for x in ts)}
        last = {j: max((d for d, _ in copies[j]), default=-1) for j, _ in task.quick.values()}
        quick = {}
        for i, (j, _) in task.quick.items():
            f = task.flight(i)
            for d, _ in copies[i]:
                t = task.turned(f, d + world.block[i])
                if t < task.back(f, d + world.block[i]) and t <= last[j]:
                    quick[(i, d)] = t
        place = {i: (ln, p) for ln in lines for p, i in enumerate(ln.legs)}
        swift: dict[tuple[int, int, int], int] = {}
        for (i, d), t in quick.items():
            ln, p = place.get(i, (None, 0))
            e = ln.first(p) if ln else None
            if e is None:
                continue
            f, g = task.flight(i), task.flight(task.quick[i][0])
            back, r = task.back(f, d + world.block[i]), None
            for k in ln.breaks[e:]:
                move = task.moved[ln.legs[k]]
                r = move.ready if r is None else max(r, move.ready)
                if move.dest == g.orig and max(t, r) < max(back, r) and max(t, r) <= last[g.id]:
                    swift[(i, d, move.leg)] = max(t, r)
        gates: dict[int, set[int]] = defaultdict(set)
        for (i, _), t in [*quick.items(), *((k[:2], t) for k, t in swift.items())]:
            gates[task.quick[i][0]].add(t)
        self.gates = {j: sorted(ts | {d for d, _ in copies[j] if d >= min(ts)}) for j, ts in gates.items()}
        base = len(node) + len(over) + len(track)
        lane = {k: base + n for n, k in enumerate(quick)}
        gate = {k: base + len(lane) + n for n, k in enumerate((j, t) for j, ts in self.gates.items() for t in ts)}
        flow: dict[int, dict[int, int]] = defaultdict(dict)

        def arc(u: int | None, v: int | None, c: int) -> None:
            if u is not None:
                flow[u][c] = flow[u].get(c, 0) - 1
            if v is not None:
                flow[v][c] = flow[v].get(c, 0) + 1

        on_line = {i for ln in lines for i in ln.legs}
        self.xs: dict[tuple[int, int], int] = {}
        self.cancel: dict[int, int] = {}
        self.lapsed: dict[int, int] = {}
        rows: dict[tuple, dict[int, int]] = defaultdict(dict)
        pinned: Counter = Counter()
        cover = []
        for f in task.flights:
            w = _weight(task, f.id)
            r = {}
            for d, (kind, key) in copies[f.id]:
                arr = d + world.block[f.id]
                c = m.col(pax.flown(f, d, arr) if pax else flown_price(f, d, arr), 1, True,
                          (d - f.sched_dep + 1) * w)
                self.xs[(f.id, d)] = c
                arc(gate.get((f.id, d), node[((f.family, f.orig), d)]),
                    lane.get((f.id, d), node[((f.family, f.dest), task.back(f, arr))]), c)
                r[c] = 1
                if kind != "free":
                    if f.id in fixed:
                        pinned[(kind, key)] += 1
                    else:
                        rows[(kind, key)][c] = 1
            ub = 0 if f.id in fixed and fixed[f.id] is not None else 1
            lapses = f.id in on_line and (f.id not in fixed or f.id in lapsed)
            self.cancel[f.id] = c = m.col(price[f.id], 0 if lapses and f.id in fixed else ub, True,
                                          (f.deadline - f.sched_dep + 2) * w)
            r[c] = 1
            if f.id in on_line:
                auto = pax.price(f, task.auto_at(f), flat) if pax else flat_price(f) if flat else \
                    cancel_price(f, task.auto_at(f))
                self.lapsed[f.id] = c = m.col(auto, ub if lapses else 0, True, (f.deadline - f.sched_dep + 3) * w)
                r[c] = 1
            cover.append(r)
        self.landed: dict[tuple[int, int], int] = dict(self.xs)
        self.turn: dict[tuple[int, int], int] = {}
        for (i, d), t in quick.items():
            f = task.flight(i)
            self.landed[(i, d)] = c = m.col(0, 1, False)
            arc(lane[(i, d)], node[((f.family, f.dest), task.back(f, d + world.block[i]))], c)
            self.turn[(i, d)] = c = m.col(0, 1, True)
            arc(lane[(i, d)], gate[(task.quick[i][0], t)], c)
        self.through: dict[tuple[int, int, int], int] = {}
        for (i, d, leg), t in swift.items():
            self.through[(i, d, leg)] = c = m.col(0, 1, False)
            arc(lane[(i, d)], gate[(task.quick[i][0], t)], c)
        self.feed: dict[tuple[int, int], int] = {}
        self.hold: dict[tuple[int, int], int] = {}
        for j, ts in self.gates.items():
            g = task.flight(j)
            for a, b in zip(ts, ts[1:], strict=False):
                self.hold[(j, a)] = c = m.col(0, 1, False)
                arc(gate[(j, a)], gate[(j, b)], c)
            for d, _ in copies[j]:
                if (j, d) in gate:
                    self.feed[(j, d)] = c = m.col(0, 1, False)
                    arc(node[((g.family, g.orig), d)], gate[(j, d)], c)
        self.redirect: dict[tuple[int, int], int] = {}
        self.token: dict[str, int] = {}
        self.deliver: dict[tuple[str, int, int], int] = {}
        self.onward: dict[tuple[str, int, int], int] = {}
        fast_to: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        for (i, _, leg), c in self.through.items():
            fast_to[(i, leg)][c] = 1
        for ln in lines:
            name, fam = ln.tail.tail, ln.tail.family
            pc = {i: self.cancel[i] for i in ln.legs}
            for p, i in enumerate(ln.legs):
                e = ln.first(p)
                if e is None:
                    continue
                f = task.flight(i)
                between = ln.legs[p + 1:ln.breaks[e] + 1]
                went = {}
                for t, ds in lands[i].items():
                    self.redirect[(i, t)] = c = m.col(0, 1, False)
                    arc(node[((fam, f.dest), t)], track[((name, e), max(t, ready[(name, e)]))], c)
                    m.row({c: 1, **{self.landed[(i, d)]: -1 for d in ds}}, -INF, 0)
                    went[c] = 1
                for q in range(e, len(ln.breaks)):
                    fast = fast_to.get((i, ln.legs[ln.breaks[q]]))
                    if not fast:
                        continue
                    went |= fast
                    for j in ln.legs[ln.breaks[e] + 1:ln.breaks[q] + 1]:
                        m.row({**fast, pc[j]: -1}, -INF, 0)
                    if q + 1 < len(ln.breaks):
                        m.row({**fast, **{pc[j]: 1 for j in ln.after(q)}}, -INF, len(ln.after(q)))
                for j in between:
                    m.row({**went, pc[j]: -1}, -INF, 0)
                flown = {self.xs[(i, d)]: 1 for d, _ in copies[i]}
                m.row({**flown, **{c: -1 for c in went}, **{pc[j]: 1 for j in between}}, -INF, len(between))
            r = ready[(name, 0)]
            self.token[name] = c = m.col(0, 1, False)
            arc(node[((fam, ln.tail.pos), ln.tail.ready)], track[((name, 0), max(ln.tail.ready, r))], c)
            before = ln.legs[:ln.breaks[0] + 1]
            for j in before:
                m.row({c: 1, pc[j]: -1}, -INF, 0)
            m.row({c: 1, **{pc[j]: -1 for j in before}}, 1 - len(before), INF)
            for e, k in enumerate(ln.breaks):
                stay, on = {}, {}
                for t in transit[(name, e)]:
                    u = track[((name, e), t)]
                    self.deliver[(name, e, t)] = c = m.col(0, 1, False)
                    arc(u, node[((fam, task.moved[ln.legs[k]].dest), t)], c)
                    stay[c] = 1
                    if e + 1 < len(ln.breaks):
                        self.onward[(name, e, t)] = c = m.col(0, 1, False)
                        arc(u, track[((name, e + 1), max(t, ready[(name, e + 1)]))], c)
                        on[c] = 1
                if on:
                    after = ln.after(e)
                    for j in after:
                        m.row({**on, pc[j]: -1}, -INF, 0)
                    m.row({**stay, **{pc[j]: 1 for j in after}}, -INF, len(after))
        tokens = Counter(t.family for t in task.tails)
        self.ground: dict[tuple, list[int]] = {}
        for key, ts in self.times.items():
            cs = []
            for a, b in zip(ts, ts[1:], strict=False):
                c = m.col(0, tokens[key[0]], False)
                arc(node[(key, a)], node[(key, b)], c)
                cs.append(c)
            cs.append(m.col(0, tokens[key[0]], False))
            arc(node[(key, ts[-1])], over[key], cs[-1])
            self.ground[key] = cs
        supply: Counter = Counter()
        for t in task.tails:
            supply[node[((t.family, t.pos), t.ready)]] += 1
        self.need = need = task.needed
        self.ferry: dict[tuple[str, str, str], int] = {}
        into: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        for fam, a in self.times:
            for fam2, b in need:
                if fam2 == fam:
                    self.ferry[(fam, a, b)] = c = m.col(ferry_cost(task, fam, a, b), tokens[fam], True)
                    arc(over[(fam, a)], None, c)
                    into[(fam, b)][c] = 1
        for key, n in need.items():
            m.row(into[key], n, n)
        for v in range(base + len(lane) + len(gate)):
            m.row(flow[v], -supply[v], -supply[v])
        for r in cover:
            m.row(r, 1, 1)
        for (kind, key), r in rows.items():
            room = 1 if kind == "slot" else world.nominal[key[0]][key[1]]
            m.row(r, -INF, max(0, room - pinned[(kind, key)]))
        self.copies = copies
        self.at: dict[int, int] = {}
        self.links: dict[tuple[int, int], int] = {}
        self.upto: dict[tuple[int, int], int] = {}
        self.rebook: dict[tuple[int, int], int] = {}
        self.refund: dict[int, int] = {}
        if pax is not None:
            self._passengers(pax)

    def _passengers(self, pax: Proxy) -> None:
        m, copies, task = self.m, self.copies, self.task
        gone = {f.id: {c: 1 for c in (self.cancel[f.id], self.lapsed.get(f.id)) if c is not None}
                for f in task.flights}
        for k in sorted({k for x in pax.links for k in (x.i, x.j)}):
            self.at[k] = c = m.col(0, max((d for d, _ in copies[k]), default=0), False)
            m.row({c: 1, **{self.xs[(k, d)]: -d for d, _ in copies[k]}}, 0, 0)
        cuts, need = [], defaultdict(set)
        for x in pax.links:
            gi, gj = gone[x.i], gone[x.j]
            self.links[(x.i, x.j)] = y = m.col(0, 1, True)
            m.row({y: 1, **{c: -1 for c in gj}, **gi}, 0, INF)
            if not copies[x.i] or not copies[x.j]:
                continue
            b, gap = self.world.block[x.i], mct(x.hub)
            lo, hi = min(d for d, _ in copies[x.j]), max(d for d, _ in copies[x.i]) + b
            if gap + hi - lo > 0:
                m.row({self.at[x.j]: 1, self.at[x.i]: -1, **{c: b for c in gi}, **{c: lo for c in gj},
                       y: gap + hi - lo}, gap + b, INF)
            for a in range(-(-lo // GRID) * GRID, min(max(d for d, _ in copies[x.j]), hi + gap - 1, lo + RISK) + 1,
                           GRID):
                t = -(-(a - gap - b) // GRID) * GRID
                cuts.append((x, a, t))
                need[x.j].add(a)
                need[x.i].add(t)
        for k, ts in sorted(need.items()):
            ds, pos, prev = sorted(d for d, _ in copies[k]), 0, None
            for t in sorted(ts):
                self.upto[(k, t)] = c = m.col(0, 1, False)
                row = {c: 1} if prev is None else {c: 1, self.upto[(k, prev)]: -1}
                while pos < len(ds) and ds[pos] <= t:
                    row[self.xs[(k, ds[pos])]] = -1
                    pos += 1
                m.row(row, 0, 0)
                prev = t
        for x, a, t in cuts:
            m.row({self.links[(x.i, x.j)]: 1, self.upto[(x.j, a)]: -1, self.upto[(x.i, t)]: 1, **gone[x.i]}, 0, INF)
        room: dict[int | str, dict[int, int]] = defaultdict(dict)
        for n, mode in enumerate(pax.modes):
            if mode.link is not None:
                hit = {self.links[mode.link]: 1}
            else:
                hit = {self.lapsed[mode.cut]: 1} if mode.lapse else {self.cancel[mode.cut]: 1}
            row = {c: -mode.pax for c in hit}
            for o, (legs, price) in enumerate(mode.options):
                self.rebook[(n, o)] = c = m.col(price, mode.pax, False)
                row[c] = 1
                for k in legs:
                    if k in pax.free:
                        room[k][c] = 1
            self.refund[n] = c = m.col(mode.refund, mode.pax, False)
            row[c] = 1
            m.row(row, 0, 0)
        for k, r in sorted(room.items(), key=lambda kv: str(kv[0])):
            m.row({**r, **({c: pax.free[k] for c in gone[k]} if isinstance(k, int) else {})}, -INF, pax.free[k])

    def solve(self, time_limit: float = 600.0, gap: float = GAP, threads: int = 1, canonical: bool = True,
              start: np.ndarray | None = None, fix: bool = True, seed: int | None = None) -> Solution:
        """`seed` sets HiGHS's random seed, for independent searches of one model."""
        m = self.m
        m.seed = seed
        lp = m.relax(time_limit, threads) if fix and canonical else None
        if lp is not None and lp.integral:
            status, objective, bound, mip_gap, wall, x = "Optimal", lp.objective, lp.objective, 0.0, lp.wall, lp.x
        else:
            status, objective, bound, mip_gap, wall, x = m.run(time_limit, gap, threads, start)
            wall += lp.wall if lp else 0.0
        settled = False
        if canonical and status == "Optimal":
            lower, upper = lp.bounds(m, round(objective) + 0.5) if lp else (None, None)
            m.row({c: v for c, v in enumerate(m.cost) if v}, -INF, round(objective) + 0.5)
            m.cost, main = m.second, m.cost
            tied, _, _, _, second_wall, y = m.run(time_limit, 0.0, threads, x, lower, upper)
            m.cost, wall = main, wall + second_wall
            m.rows.pop()
            if tied == "Optimal":
                x, settled, objective = y, True, float(np.dot(main, y))
        deps = {}
        for f in self.task.flights:
            flown = [d for d, _ in self.copies[f.id] if x[self.xs[(f.id, d)]] > 0.5]
            deps[f.id] = flown[0] if flown else None
        lapsed = {i: self.task.flight(i).deadline + 1 for i, c in self.lapsed.items() if x[c] > 0.5}
        return Solution(status, round(objective), bound, round(100 * mip_gap, 4), round(wall, 2), deps, len(m.cost),
                        len(m.rows), sum(len(v) for v in self.copies.values()), start is not None, settled, lapsed)

    def rounded(self, time_limit: float = 600.0, threads: int = 1) -> dict[int, dict] | None:
        """The LP relaxation's optimum rounded to a plan the sim can play, a start for the search: a flight more
        cancelled than not is cancelled at the start (held past its deadline where the LP lets it auto-cancel more than
        the desk cancels it), any other held to the minute by which half of it has left; None if the LP isn't
        reached."""
        h = self.m._highs(time_limit, threads)
        h.run()
        if h.modelStatusToString(h.getModelStatus()) != "Optimal":
            return None
        x, task = np.array(h.getSolution().col_value), self.task
        plan = {}
        for f in task.flights:
            cut, auto = x[self.cancel[f.id]], x[self.lapsed[f.id]] if f.id in self.lapsed else 0.0
            if cut + auto >= 0.5:
                plan[f.id] = {"not_before": f.deadline + 1} if auto > cut else {"cancel": True, "decided": task.start}
                continue
            left = 0.0
            for d, _ in self.copies[f.id]:
                left += x[self.xs[(f.id, d)]]
                if left >= 0.5 * (1 - cut - auto):
                    plan[f.id] = {"not_before": d}
                    break
        return plan

    def vector(self, run: sim.Execution) -> np.ndarray | None:
        """`run` (a day sim.execute ran in this world, holding every pin) as a solution of the model, or None if a
        departure has no copy here."""
        task, world = self.task, self.world
        x = np.zeros(len(self.m.cost))
        delta: dict[tuple, Counter] = defaultdict(Counter)
        carried, _ = sim.routes(task, {i for i, r in run.flights.items() if r.status == CANCELLED})
        nxt = {a: b for legs in run.legs.values() for a, b in zip(legs, legs[1:], strict=False)}
        via, fast = {}, set()
        for f in task.flights:
            r = run.flights[f.id]
            if r.status != FLOWN:
                x[self.lapsed[f.id] if r.status == AUTO_CANCELLED and f.id in self.lapsed else self.cancel[f.id]] = 1
                continue
            k = (f.id, r.dep)
            if k not in self.xs:
                return None
            x[self.xs[k]] = 1
            j = task.quick.get(f.id, (None,))[0]
            swift = (*k, carried[f.id][1][-1].leg) if f.id in carried else None
            if swift in self.through and nxt.get(f.id) == j:
                x[self.through[swift]] = 1
                via[j] = max([task.turned(f, r.dep + world.block[f.id]), *(m.ready for m in carried[f.id][1])])
                fast.add(f.id)
            elif k in self.turn and f.id not in carried and nxt.get(f.id) == j:
                x[self.turn[k]] = 1
                via[j] = task.turned(f, r.dep + world.block[f.id])
            else:
                x[self.landed[k]] = 1
                delta[(f.family, f.dest)][task.back(f, r.dep + world.block[f.id])] += 1
        for f in task.flights:
            r = run.flights[f.id]
            if r.status != FLOWN:
                continue
            if f.id in via:
                for a in self.gates[f.id]:
                    if via[f.id] <= a < r.dep:
                        x[self.hold[(f.id, a)]] = 1
                continue
            if (f.id, r.dep) in self.feed:
                x[self.feed[(f.id, r.dep)]] = 1
            delta[(f.family, f.orig)][r.dep] -= 1
        for t in task.tails:
            delta[(t.family, t.pos)][t.ready] += 1
        for ln in self.lines:
            self._carry(ln, run, x, delta, fast)
        ends: dict[tuple, int] = {}
        for key, ts in self.times.items():
            held = 0
            for t, c in zip(ts, self.ground[key], strict=True):
                held += delta[key][t]
                x[c] = held
            ends[key] = held
        if self.pax is not None:
            self._riders(run, x)
        out = Counter((fy.family, fy.orig, fy.dest) for fy in run.ferries)
        for (fam, a), n in ends.items():
            stay = min(n, self.need[(fam, a)])
            if stay:
                x[self.ferry[(fam, a, a)]] = stay
            for (g, o, b), v in out.items():
                if (g, o) == (fam, a):
                    x[self.ferry[(fam, a, b)]] = v
        return x

    def _riders(self, run: sim.Execution, x: np.ndarray) -> None:
        """`run`'s passengers into x: departure minutes, delays, broken connections and the rebooking LP's
        assignment."""
        pax, runs = self.pax, run.flights
        for k, c in self.at.items():
            x[c] = runs[k].dep if runs[k].status == FLOWN else 0
        for (k, t), c in self.upto.items():
            x[c] = runs[k].status == FLOWN and runs[k].dep <= t
        for link in pax.links:
            x[self.links[(link.i, link.j)]] = pax.broken(link, runs)
        _, rebooked, refunded = pax.assign(runs)
        for key, v in rebooked.items():
            x[self.rebook[key]] = v
        for n, v in refunded.items():
            x[self.refund[n]] = v

    def _carry(self, ln: Line, run: sim.Execution, x: np.ndarray, delta: dict[tuple, Counter],
               fast: set[int]) -> None:
        """`run`'s aircraft through the breaks of `ln` into x, walking its legs as sim.routes does; a carrier in
        `fast` went to its next leg's gate instead."""
        task, world = self.task, self.world
        name, fam = ln.tail.tail, ln.tail.family
        carrier, void, at = None, False, None
        for p, i in enumerate(ln.legs):
            if run.flights[i].status != CANCELLED:
                if at is not None:
                    self._deliver(ln, at, x, delta)
                carrier, void, at = i, False, None
            if p not in ln.breaks:
                continue
            e = ln.breaks.index(p)
            r = task.moved[i].ready
            if at is not None:
                x[self.onward[(name, e - 1, at[1])]] = 1
                at = (e, max(at[1], r))
            elif carrier is None:
                x[self.token[name]] = 1
                delta[(fam, ln.tail.pos)][ln.tail.ready] -= 1
                at = (e, max(ln.tail.ready, r))
            elif not void and run.flights[carrier].status == FLOWN and carrier not in fast:
                f = task.flight(carrier)
                t = task.back(f, run.flights[carrier].dep + world.block[carrier])
                x[self.redirect[(carrier, t)]] = 1
                delta[(fam, f.dest)][t] -= 1
                at = (e, max(t, r))
            else:
                void = True
        if at is not None:
            self._deliver(ln, at, x, delta)

    def _deliver(self, ln: Line, at: tuple[int, int], x: np.ndarray, delta: dict[tuple, Counter]) -> None:
        e, t = at
        x[self.deliver[(ln.tail.tail, e, t)]] = 1
        delta[(ln.tail.family, self.task.moved[ln.legs[ln.breaks[e]]].dest)][t] += 1


class _Started(_Model):
    """v1's model, run from a start solution when one is given, within column bounds when given, or relaxed, under a
    HiGHS random seed when one is set."""

    seed: int | None = None

    def run(self, time_limit: float, gap: float, threads: int, start: np.ndarray | None = None,
            lower: np.ndarray | None = None,
            upper: np.ndarray | None = None) -> tuple[str, float, float, float, float, np.ndarray]:
        h = self._highs(time_limit, threads, lower, upper)
        h.setOptionValue("mip_rel_gap", gap)
        if self.seed is not None:
            h.setOptionValue("random_seed", self.seed)
        n = len(self.cost)
        ints = np.array([c for c in range(n) if self.integer[c]], dtype=np.int32)
        h.changeColsIntegrality(len(ints), ints, np.array([highspy.HighsVarType.kInteger] * len(ints)))
        if start is not None:
            h.setSolution(n, np.arange(n, dtype=np.int32), np.asarray(start, dtype=float))
        t0 = time.monotonic()
        h.run()
        wall = time.monotonic() - t0
        status = h.modelStatusToString(h.getModelStatus())
        info = h.getInfo()
        if info.primal_solution_status != 2:
            raise RuntimeError(f"no feasible solution: {status}")
        return (status, info.objective_function_value, info.mip_dual_bound, info.mip_gap, wall,
                np.array(h.getSolution().col_value))

    def relax(self, time_limit: float, threads: int) -> "Relaxed | None":
        """The LP relaxation's optimum, or None if it isn't reached."""
        h = self._highs(time_limit, threads)
        t0 = time.monotonic()
        h.run()
        wall = time.monotonic() - t0
        if h.modelStatusToString(h.getModelStatus()) != "Optimal":
            return None
        sol = h.getSolution()
        x, integer = np.array(sol.col_value), np.array(self.integer)
        integral = bool(np.all(np.abs(x[integer] - np.round(x[integer])) <= INTEGRAL))
        return Relaxed(h.getInfo().objective_function_value, np.where(integer, np.round(x), x),
                       np.array(sol.col_dual), integral, wall)

    def _highs(self, time_limit: float, threads: int, lower: np.ndarray | None = None,
               upper: np.ndarray | None = None) -> highspy.Highs:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("threads", threads)
        n = len(self.cost)
        h.addVars(n, np.zeros(n) if lower is None else lower,
                  np.array(self.ub, dtype=float) if upper is None else upper)
        h.changeColsCost(n, np.arange(n, dtype=np.int32), np.array(self.cost, dtype=float))
        starts, idx, val = [], [], []
        for coefs, _, _ in self.rows:
            starts.append(len(idx))
            for c, v in coefs.items():
                if v:
                    idx.append(c)
                    val.append(v)
        h.addRows(len(self.rows), np.array([r[1] for r in self.rows], dtype=float),
                  np.array([r[2] for r in self.rows], dtype=float), len(idx), np.array(starts, dtype=np.int32),
                  np.array(idx, dtype=np.int32), np.array(val, dtype=float))
        return h


@dataclass(frozen=True)
class Relaxed:
    """The LP relaxation's optimum, its solution (integer columns rounded), reduced costs and whether it is
    integral."""

    objective: float
    x: np.ndarray
    reduced: np.ndarray
    integral: bool
    wall: float

    def bounds(self, m: _Model, cap: float) -> tuple[np.ndarray, np.ndarray]:
        """Column bounds every solution costing at most `cap` keeps: one with x_j moved k off the bound its reduced
        cost d_j holds it at costs at least the LP optimum plus k|d_j| (one unit of slack for the LP's tolerances)."""
        d, ub, integer = self.reduced, np.array(m.ub, dtype=float), np.array(m.integer)
        room = cap - self.objective + 1.0 + 1e-6 * abs(cap)
        up = np.where(d > 0, room / np.where(d > 0, d, 1.0), np.inf)
        down = np.where(d < 0, room / np.where(d < 0, -d, 1.0), np.inf)
        upper = np.minimum(ub, np.where(integer, np.floor(up), up))
        lower = np.maximum(0.0, ub - np.where(integer, np.floor(down), down))
        return lower, upper


def _place(f: Flight, d: int, rules: sim.Rules, pools: Mapping[str, set[int]]) -> tuple[str, tuple | None] | None:
    """The row a copy of f leaving at d counts in: its slot, its nominal hour, or none; None when d is forbidden."""
    seg = rules.governs(f, d)
    if seg is not None:
        cta = d + f.sched_block
        if cta <= seg.end:
            return ("slot", (seg.id, cta)) if cta in pools[seg.id] else None
    cap = rules.hour(f, d)
    return ("hour", cap[0]) if cap else ("free", None)


def _weight(task: Task, i: int) -> float:
    return 1 + i / (len(task.flights) + 1)
