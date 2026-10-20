"""The storm day with crews: sim.execute's dispatch, where a flight ready to leave also needs a legal pilot unit and a
legal cabin unit of its type group at its origin, and the crew event log.

A plan entry may name its crews ({"crew": {"pilot": id, "cabin": id}}) and the units that deadhead on it ({"deadhead":
[id, ...]}); plan["reports"] maps a short-call reserve to its report and "id#2" to a unit's second duty period's report.
Those act from the start; a crew desk's decisions during the day act from their minute (`entries`: "names", "rides",
plan["calls"], plan["keeps"] and plan["swaps"], two units exchanging their remaining trips) and change only what they
name: a unit one moves keeps the legs and rides its trip plans that still connect (`prune`), the rest going back to the
standard desk, which leaves a flight's named kinds alone and doesn't take the unit while a flight it is named to fly or
ride is still to come.
A named unit flies its flight: the flight waits until the unit is at its origin, free and legal, and auto-cancels at its
deadline if it never is. With `fill` (the default rule, the trip rule), a flight with a kind unnamed takes the unit
whose trip has it if that unit is there, free and legal, else the first one ready of the other units of the kind and
group there, free and legal and named nowhere in the plan with no trip left (a crew done with its trip, an
airport-standby reserve, a called reserve) or the short-call reserve called for it; no crew leaves its trip. With none,
the flight waits for the first of them to be free or legal again, or for a crew to arrive, or calls a short-call reserve
of the base if one could report sooner, held for it until it reports (if another crew takes the flight, the reserve is
as if never called); else it waits for a crew to arrive. A flight with "left" (the day so far) leaves at its minute with
the crews it finds, and one without a legal crew of a kind is logged so. A rider rides if it is at the origin, free,
legal for the deadhead and the flight has its seats free (the family's smallest seats less the passengers); with `fill`
a unit named nowhere, or moved only by a crew desk, also rides its trip's planned deadheads.

With `desk`, the flights a plan leaves unnamed are crewed as a standard crew desk does it, the baseline the best plan is
measured against. At the start and after every move of a crew whose flights it watches, for each flight and kind whose
planned crew can't leave by the schedule (or at once, if later) without the pilots' extension, it lists its options,
each a crew legal for the flight that can leave by its deadline: (a) the planned crew, waited for; (b) an
airport-standby or called reserve with no trip at the origin, or an uncalled short-call reserve of the base called to
report on the last whole hour that leaves BRIEF before the departure, or as soon after as its lead allows; (c) a swap,
with a crew at the origin with no trip left or one whose own remaining trip the planned crew can fly, legal as
scheduled, in its stead; (d) a crew or reserve with no trip elsewhere, deadheading to the origin on a flight with a
seat free; (e) any of those crews legal for the flight only with the extension. An option costs the flight's
passengers' wait, the deadhead's price, the hotel of a rest away from base the crew needs first, and what the move adds
to the end of the day as the day prices it (per kind, type group and base, every unit where its day then leads), all
as scheduled from then. The desk takes the cheapest option, ties to the earlier letter, then the earlier departure,
then the unit id; so it takes a crew off its trip, or relieves the planned crew of its own, only when that costs less
than waiting. A unit taken for a flight takes it and the rest of its planned crew's trip and is held for it; on a swap
the planned crew takes that unit's trip. When the flight is ready to leave the desk decides again from that minute.
Once for one of its kinds no option leaves by the deadline and no crew's trip brings one to the origin by then (the
planned crew's only before the flight), the flight is cancelled at that minute, priced at that minute's notice; without
`drop` the desk can't cancel, so it asks ops to (CANCEL) and the flight stays on the board, waiting for a crew or its
deadline. Each decision on a flight is also the desk's word to ops, from what it computed and as scheduled from then:
CANCEL where it is stuck, else HOLD(T) when the crew it takes or waits for can't leave before T, later than it wanted,
or when none can until the first crew a trip brings there lands (T that arrival plus CONNECT), T the latest over the
kinds; the flight's latest word stands, and a new one is logged only when its type or minute changes.

The rules are crews.py's, judged at takeoff (crewreplay.py prices the same ones): a unit's period continues while the
leg fits it; an uncalled short-call reserve reports, and a rested unit starts its second period, on the last whole hour
that leaves BRIEF before the departure, the reserve no earlier than LEAD after the start or its period's start and
called by its period's end, the second period at least the rest after release, a hotel away from base. A ride in the
first period that leaves a full rest after the release starts the second period that way (its night away is a hotel).
A unit connects in CONNECT, is released DEBRIEF after its last block-in, and its flight time counts actual blocks.

Costs, in passenger-minutes: deadheads off a unit's trip, nights away from base, and the end of the day: per kind, type
group and base, the units where the day leaves them sent to where their trips have them tonight at the least cost
(crews.Ends: a unit may take any of its base's trips tomorrow). The log has per unit each period's report time and
place, its legs and deadheads with their times, its release, hotels, and a reserve's call and report; per flight its
crew status over time: "legal", "times out" at a minute (its crew's last legal departure), or "no crew" (no unit
assigned, or its unit stranded or out of time), then "flown", "flown without a legal crew" or "cancelled", judged
again whenever its crews move or the desk gives it a crew (a crew with a deadhead booked to its origin judged as if
there after it); and the desk's messages to ops.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from ..core.model import Airport
from . import crews, sim
from .crews import Unit
from .model import Flight, Task

LEGAL, TIMES_OUT, NO_CREW = "legal", "times out", "no crew"
CANCEL, HOLD = "CANCEL", "HOLD"


@dataclass
class State:
    unit: Unit
    station: str
    ready: int
    report: int | None
    hour: int | None
    rap: int | None
    flown: int
    minutes: int
    landed: int | None
    trip: list[int]
    deadheads: list[int]
    period: int = 1
    moved: bool = False
    earliest: int | None = None
    claim: int | None = None
    duties: list[dict] = field(default_factory=list)
    hotels: list[dict] = field(default_factory=list)
    call: int | None = None

    def release(self, start: int) -> int | None:
        """When its duty period ends if it flies no more: after its last block-in, at its shift's end for an unused
        standby reserve, else at its report or the start, whichever is later."""
        u = self.unit
        if self.landed is not None:
            return self.landed + crews.DEBRIEF
        if u.reserve == "standby" and not self.moved and u.until is not None:
            return u.until
        return None if self.report is None else max(self.report, start)


@dataclass(frozen=True)
class CrewDay:
    run: sim.Execution
    crews: dict[int, dict[str, str]]
    riders: dict[int, tuple[str, ...]]
    uncrewed: tuple[int, ...]
    cost: dict[str, float]
    units: dict[str, dict]
    status: dict[int, list[tuple]]
    messages: tuple[dict, ...] = ()
    dropped: dict[int, int] = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

    @property
    def crew_cost(self) -> float:
        return sum(self.cost.values())

    @property
    def total(self) -> float:
        return self.run.cost.total + self.crew_cost


def execute(task: Task, plan: Mapping, units: Sequence[Unit], world: sim.World | None = None,
            local: crews.Local | None = None, fill: bool = True, log: bool = True,
            airports: Mapping[str, Airport] | None = None, desk: bool = False, drop: bool = True) -> CrewDay:
    """The day under `plan` with `units` at the start (the truth by default); `local` reads a station's clock (the
    hub's by default), `airports` hold every station a unit may end at or be wanted at (the task's by default); with
    `desk` the standard crew desk crews the flights the plan leaves unnamed, else the trip rule, and without `drop` it
    asks ops to cancel where it would cancel."""
    return finish(*start(task, plan, units, world, local, fill, log, airports, desk, drop))


def start(task: Task, plan: Mapping, units: Sequence[Unit], world: sim.World | None = None,
          local: crews.Local | None = None, fill: bool = True, log: bool = True,
          airports: Mapping[str, Airport] | None = None, desk: bool = False, drop: bool = True
          ) -> tuple["Pool", sim.Dispatch]:
    """`execute`'s day before it runs: its pool and its dispatch, to play to a minute (sim.Dispatch.until)."""
    world = sim.truth(task) if world is None else world
    pool = Pool(task, units, local or (lambda s, m: m), plan, fill, log, airports or task.airports, desk=desk,
                drop=drop)
    return pool, sim.Dispatch(task, plan, world, pool)


def finish(pool: "Pool", day: sim.Dispatch) -> CrewDay:
    """The rest of a started day played, and the day; with the log, `trace` holds when each flight first waited for
    a crew of a kind, the legal limits of the crews that flew it and the standard desk's moves."""
    run = day.execution()
    spent = pool.spent()
    trace = {"waited": pool.waited, "limits": pool.limits, "fixes": pool.fixes} if pool.log else {}
    return CrewDay(run, dict(pool.crewed), {i: tuple(v) for i, v in pool.riders.items()}, tuple(sorted(pool.uncrewed)),
                   {k: sum(v[k] for v in spent.values()) for k in ("deadhead", "hotel", "end")}, pool.unit_log(spent),
                   pool.status, tuple(pool.messages), dict(pool.dropped), trace)


def audit(task: Task, units: Sequence[Unit], plan: Mapping, deps: Mapping[int, int], blocks: Mapping[int, int],
          local: crews.Local | None = None, airports: Mapping[str, Airport] | None = None,
          ends: Mapping[int, str] | None = None) -> list[str]:
    """A plan's crews and riders over fixed departures and blocks (and where each flight ended, for a diversion),
    checked move by move against the sim's rules in departure order (no aircraft): the moves the sim would refuse."""
    pool = Pool(task, units, local or (lambda s, m: m), plan, False, False, airports or task.airports, ends or {})
    out = []
    for i in sorted(deps, key=lambda i: (deps[i], i)):
        f, t = task.flight(i), deps[i]
        entry = plan.get(i, {})
        picks = {}
        for kind, uid in entry.get("crew", {}).items():
            how = pool.operate(pool.state[uid], f, t)
            if how is None:
                out.append(f"{f.flight} {f.orig}-{f.dest} at {t}: {kind} {uid}")
            else:
                picks[kind] = (uid, how)
        pool.picks[i] = picks
        pool.fly(f, t, t + blocks[i], entry)
        out += [f"{f.flight} {f.orig}-{f.dest} at {t}: rider {u}" for u in entry.get("deadhead", ())
                if u not in pool.riders.get(i, ())]
    return out


class Pool:
    def __init__(self, task: Task, units: Sequence[Unit], local: crews.Local, plan: Mapping, fill: bool, log: bool,
                 airports: Mapping[str, Airport], ends: Mapping[int, str] | None = None, desk: bool = False,
                 drop: bool = True):
        self.task, self.local, self.fill, self.log, self.desk, self.drop = task, local, fill, log, desk, drop
        self.lb = task.lb
        self.messages: list[dict] = []
        self.standing: dict[int, tuple] = {}
        self.notes: list[tuple] = []
        self.airports, self.diverted = airports, ends or {}
        self.uncalled: dict[str, tuple[int, int]] = {}
        self.held: dict[int, list[str]] = defaultdict(list)
        self.reports: dict[str, int] = {}
        self.named: dict[tuple[str, int], str] = {}
        self.riding: dict[int, list[str]] = defaultdict(list)
        self.refs: Counter = Counter()
        self.mine: Counter = Counter()
        self.commits: dict[str, set[int]] = defaultdict(set)
        self.kept: dict[str, set[int]] = defaultdict(set)
        self.swaps: list[tuple[str, str]] = []
        self.lost: dict[int, dict[str, list[tuple[int, bool]]]] = defaultdict(dict)
        self.replaced: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
        self.pending: dict[int, list[tuple]] = defaultdict(list)
        for when, change in entries(plan):
            if when is None or when <= task.start:
                self.apply(change, when is not None)
            else:
                self.pending[when].append(change)
        self.scripted = {u for u, n in self.refs.items() if n > 0}
        self.owned = {u for u, n in self.mine.items() if n > 0}
        self.done: set[int] = set()
        self.cut = {int(k): task.start if v.get("decided") is None else v["decided"] for k, v in plan.items()
                    if k not in sim.EXTRA and v.get("cancel")}
        self.state = {}
        for u in units:
            s = State(u, u.station, u.ready, u.report, u.hour, u.rap if u.reserve == "short" else None, u.flown,
                      u.minutes, u.landed, list(u.trip), list(u.deadheads), moved=u.landed is not None)
            if u.reserve == "short" and u.report is None:
                s.earliest = crews.on_hour(max(u.rap, task.start) + u.lead)
                self.uncalled[u.id] = (s.earliest, s.ready)
            elif u.reserve == "short":
                s.call = u.report - u.lead
            if u.report is not None:
                s.duties.append(duty(u.report, u.at or None, "on duty"))
            self.state[u.id] = s
        self.at: dict[tuple, set[str]] = defaultdict(set)
        for u in units:
            self.at[(u.kind, u.group, u.station)].add(u.id)
        self.planned = {(u.kind, i): u.id for u in units for i in u.trip}
        self.watch: dict[str, set[int]] = defaultdict(set)
        for (_, i), uid in [*self.planned.items(), *self.named.items()]:
            self.watch[uid].add(i)
        seats: dict[str, int] = {}
        for t in task.tails:
            seats[t.family] = min(seats.get(t.family, t.seats), t.seats)
        self.free = {f.id: max(0, seats[f.family] - f.pax) for f in task.flights}
        self.picks: dict[int, dict[str, tuple]] = {}
        self.ends = crews.Ends(units, airports, task.date)
        self.waiting: dict[tuple, set[int]] = defaultdict(set)
        self.wake: list[tuple[int, int]] = []
        self.crewed: dict[int, dict[str, str]] = defaultdict(dict)
        self.riders: dict[int, list[str]] = defaultdict(list)
        self.uncrewed: list[int] = []
        self.status: dict[int, list[tuple]] = defaultdict(list)
        self.dropped: dict[int, int] = {}
        self.waited: dict[tuple[int, str], int] = {}
        self.limits: dict[int, dict[str, tuple]] = defaultdict(dict)
        self.fixes: list[dict] = []
        self.booked: Counter = Counter()
        self.rides: dict[int, tuple[str, int]] = {}
        self.kin: dict[tuple, list[str]] = defaultdict(list)
        for u in sorted(units, key=lambda u: u.id):
            self.kin[(u.kind, u.group)].append(u.id)
        self.route: dict[tuple, list[Flight]] = defaultdict(list)
        for f in sorted(task.flights, key=lambda f: (f.sched_dep, f.id)):
            self.route[(f.orig, f.dest)].append(f)
        replaced = [(uid, self.planned[key], key[1]) for key, uid in self.named.items() if key[1] in self.commits[uid]
                    and self.planned.get(key) not in (None, uid)]
        self.settle(task.start, self.owned | {y for _, y, _ in replaced}, replaced)
        if log:
            for f in task.flights:
                self.judge(f, task.start)
        if desk:
            for f in sorted(task.flights, key=lambda f: (f.sched_dep, f.id)):
                self.check(f, task.start)

    def apply(self, change: tuple, timed: bool = False) -> int | None:
        """One crew entry of the plan taking effect: a unit named on a flight or no longer, riding one or no longer,
        kept off a ride its trip plans, or a report set; the flight it touches. A timed entry (a crew desk's move)
        commits the unit to the flights it names it on or lets it ride."""
        what, key, uid, value = change
        counts = (self.refs, self.mine) if timed else (self.refs,)
        if what == "swap":
            self.swaps.append((uid, key))
            for c in counts:
                c[uid] += 1
                c[key] += 1
            return None
        if what in ("keep", "report"):
            if what == "report":
                self.reports[key] = value
            elif timed:
                self.kept[uid].add(key)
            for c in counts:
                c[uid] += 1
            return None
        if what in ("name", "unname"):
            if what == "name":
                if (was := self.named.get(key)) is not None:
                    for c in counts:
                        c[was] -= 1
                    self.commits[was].discard(key[1])
                self.named[key] = uid
            elif self.named.get(key) == uid:
                del self.named[key]
            else:
                return None
            for c in counts:
                c[uid] += 1 if what == "name" else -1
            if timed:
                (self.commits[uid].add if what == "name" else self.commits[uid].discard)(key[1])
            return key[1]
        if what == "ride":
            self.riding[key].append(uid)
        elif uid in self.riding.get(key, ()):
            self.riding[key].remove(uid)
        else:
            return None
        for c in counts:
            c[uid] += 1 if what == "ride" else -1
        if timed:
            (self.commits[uid].add if what == "ride" else self.commits[uid].discard)(key)
        return key

    def changes(self) -> list[int]:
        """The minutes after the start at which a crew entry of the plan takes effect."""
        return sorted(self.pending)

    def decide(self, t: int) -> list[int]:
        """The plan's crew entries decided at t, in effect from now (`settle`), and the desk re-plans the flights of the
        units they move and of the crews they replace; the flights they touch, to try again."""
        touched, flights, replaced, ended = set(), [], [], []
        for change in sorted(self.pending.pop(t, ()), key=lambda c: c[0] not in ("unname", "unride")):
            touched |= {change[2], change[1]} if change[0] == "swap" else {change[2]}
            if change[0] == "name" and self.planned.get(change[1]) not in (None, change[2]):
                touched.add(self.planned[change[1]])
                replaced.append((change[2], self.planned[change[1]], change[1][1]))
            if change[0] in ("unname", "unride"):
                ended.append((change[3], change[2], change[1][1] if change[0] == "unname" else change[1]))
            i = self.apply(change, True)
            if i is None:
                continue
            flights.append(i)
            if change[0] == "name":
                self.watch[change[2]].add(i)
        for frm, x, i in ended:
            touched |= self.restore(frm, x, i, t)
        self.settle(t, touched, replaced)
        flights = list(dict.fromkeys(flights))
        if self.log:
            for i in flights:
                if i not in self.done:
                    self.judge(self.task.flight(i), t)
        self.recheck(touched, t)
        return flights

    def settle(self, t: int, touched: set[str], replaced: list[tuple[str, str, int]]) -> None:
        """The crew desk's moves of a minute in place: swaps traded, each unit moved or replaced pruned, and a named
        unit with no trip left taking the later legs its flight's planned crew dropped (as the standard desk's
        `hand`)."""
        self.trade()
        self.rescript()
        self.replaced[t] += replaced
        dropped = {u: self.prune(u, t) for u in sorted(touched)}
        for x, y, i in replaced:
            s, after = self.state[x], self.task.flight(i).sched_dep
            rest = sorted((j for j in dropped.get(y, ()) if self.task.flight(j).sched_dep > after and
                           (s.unit.kind, j) not in self.named), key=lambda j: (self.task.flight(j).sched_dep, j))
            if rest and not s.trip and self.named.get((s.unit.kind, i)) == x:
                s.trip = [i, *rest]
                for j in s.trip:
                    self.planned[(s.unit.kind, j)] = x
                    self.watch[x].add(j)
                self.prune(x, t)

    def restore(self, frm: int, x: str, i: int, t: int) -> set[str]:
        """A crew desk's name or ride of x on flight i, taken at frm, ending at t: x, and the crew it replaced on i,
        get back the legs and rides its taking dropped, those still to come that no one else is named on, from
        whoever the standard desk gave them to; the units given back to."""
        units = {x} | {y for xx, y, ii in self.replaced.get(frm, ()) if (xx, ii) == (x, i)}
        for uid in units:
            s = self.state[uid]
            kind = s.unit.kind
            for j, leg in self.lost[frm].pop(uid, ()):
                if j in self.done or self.cut.get(j, t + 1) <= t:
                    continue
                if not leg:
                    if j not in s.deadheads:
                        s.deadheads.append(j)
                    continue
                if self.named.get((kind, j), uid) != uid:
                    continue
                if (cur := self.planned.get((kind, j))) not in (None, uid):
                    if j in self.state[cur].trip:
                        self.state[cur].trip.remove(j)
                    if self.state[cur].claim == j:
                        self.unhold(cur, j)
                if j not in s.trip:
                    s.trip = sorted([*s.trip, j], key=lambda g: (self.task.flight(g).sched_dep, g))
                self.planned[(kind, j)] = uid
                self.watch[uid].add(j)
        return units

    def unhold(self, uid: str, i: int) -> None:
        """uid no longer held for flight i: an uncalled short-call reserve as if never called."""
        s = self.state[uid]
        self.held[i].remove(uid)
        s.claim = None
        if s.report is None:
            s.call = None
            s.earliest, s.ready = self.uncalled[uid]

    def rescript(self) -> None:
        """The units the plan's entries move (`owned`: a crew desk's) and those the standard desk leaves alone
        (`scripted`): named from the start, or committed to a flight still to come."""
        self.owned = {u for u, n in self.mine.items() if n > 0}
        self.scripted = {u for u, n in self.refs.items() if n > self.mine[u]} | \
            {u for u in self.owned if self.commits[u] - self.done}

    def trade(self) -> None:
        """The swaps taking effect: two units exchange their remaining trips and the rides those plan, the standard
        desk's to crew as any trip; one held for a flight it hands over is free."""
        for a, b in self.swaps:
            sa, sb = self.state[a], self.state[b]
            booked = {(v, g) for v, g in self.rides.values()}
            ra, rb = ([i for i in s.deadheads if (u, i) not in booked] for u, s in ((a, sa), (b, sb)))
            sa.trip, sb.trip = sb.trip, sa.trip
            sa.deadheads = [i for i in sa.deadheads if i not in ra] + rb
            sb.deadheads = [i for i in sb.deadheads if i not in rb] + ra
            for uid, s, o in ((a, sa, sb), (b, sb, sa)):
                for i in s.trip:
                    self.planned[(s.unit.kind, i)] = uid
                    self.watch[uid].add(i)
                if s.claim is not None and s.claim in o.trip:
                    self.held[s.claim].remove(uid)
                    s.claim = None
        self.swaps = []

    def prune(self, uid: str, t: int) -> list[int]:
        """The legs and rides of uid's trip it no longer makes, decided at t once a crew desk moved it, dropped (back
        to the standard desk): those named to another unit, those kept off, and, walking its trip with the flights
        it is committed to from where it is, those that don't leave from where it is then or, before a commitment,
        don't bring it to that flight's origin by its departure; the legs dropped."""
        s = self.state[uid]
        kind, gone, legs = s.unit.kind, self.done | {i for i, d in self.cut.items() if d <= t}, set(s.trip)
        fixed = self.commits[uid] - gone
        planned = {*s.trip, *s.deadheads}
        out = [i for i in s.trip if self.named.get((kind, i), uid) != uid] + \
            [i for i in s.deadheads if i in self.kept[uid]]
        rest = sorted((self.task.flight(i) for i in planned - fixed - gone - set(out)),
                      key=lambda g: (g.sched_dep, g.id))
        steps = sorted([(g, True) for g in map(self.task.flight, fixed)] + [(g, False) for g in rest],
                       key=lambda x: (x[0].sched_dep, x[0].id))
        keep, chain, at = set(), [], s.station
        for g, commit in [*steps, (None, True)]:
            if not commit:
                if g.orig == (self.lands(chain[-1].id) if chain else at):
                    chain.append(g)
                continue
            while chain and g is not None and (self.lands(chain[-1].id) != g.orig or (
                    g.id not in planned and chain[-1].sched_arr + crews.CONNECT > g.sched_dep)):
                chain.pop()
            keep.update(x.id for x in chain)
            if g is not None:
                chain, at = [], self.lands(g.id)
        out += [x.id for x in rest if x.id not in keep]
        lost = self.lost[t].setdefault(uid, [])
        for i in out:
            lost.append((i, i in legs))
            if s.claim == i:
                self.unhold(uid, i)
            if i in s.trip:
                s.trip.remove(i)
                if self.planned.get((kind, i)) == uid:
                    del self.planned[(kind, i)]
            if i in s.deadheads:
                s.deadheads.remove(i)
                for j, (v, g) in list(self.rides.items()):
                    if (v, g) == (uid, i):
                        del self.rides[j]
                        self.booked[i] -= s.unit.size
        return [i for i in out if i in legs]

    def board(self, f: Flight, t: int, entry: Mapping, left: bool) -> int | None:
        """t if f's crews are there now (held for `fly`), else the minute to try again (None: when woken)."""
        group = crews.TYPE[f.family]
        picks, later = {}, []
        for kind in crews.KINDS:
            name = self.named.get((kind, f.id))
            got, retry = self.attend(kind, group, f, t) if self.desk and self.fill and not name and not left else \
                self.choose(kind, group, f, t, name)
            if any(n[1] == CANCEL for n in self.notes):
                if self.log:
                    self.waited.setdefault((f.id, kind), t)
                self.say(f, t)
                return None
            if got is not None:
                picks[kind] = got
            elif retry is not None:
                later.append(retry)
        self.say(f, t)
        if left or len(picks) == len(crews.KINDS):
            self.picks[f.id] = picks
            return t
        for kind in crews.KINDS:
            if kind not in picks:
                self.waiting[(kind, group, f.orig)].add(f.id)
                if self.log:
                    self.waited.setdefault((f.id, kind), t)
        return max(later) if later else None

    def choose(self, kind: str, group: str, f: Flight, t: int, name: str | None) -> tuple:
        """((unit, how it starts), None) for the unit that takes f at t, or (None, the minute to try again or None)."""
        if name:
            s = self.state[name]
            if s.station != f.orig:
                return None, None
            if s.ready > t:
                return None, s.ready
            how = self.operate(s, f, t)
            return ((name, how), None) if how else (None, self.later(s, f, t))
        if not self.fill:
            return None, None
        everyone = self.at[(kind, group, f.orig)]
        mine = self.planned.get((kind, f.id))
        spare = [u for u in everyone - self.scripted if u != mine and not self.state[u].trip
                 and self.available(self.state[u], f)]
        spare.sort(key=lambda u: (self.state[u].ready, u))
        if mine in everyone and self.available(self.state[mine], f):
            spare.insert(0, mine)
        for u in spare:
            s = self.state[u]
            if s.ready <= t and (how := self.operate(s, f, t)):
                return (u, how), None
        waits = [w for u in spare if (w := self.later(self.state[u], f, t)) is not None]
        reserve = self.reserve(kind, group, f, t)
        if reserve is not None and (not waits or reserve[1] < min(waits)):
            s = self.state[reserve[0]]
            s.claim, s.call, s.earliest = f.id, max(t, s.unit.rap, self.task.start), reserve[1] - crews.BRIEF
            s.ready = reserve[1]
            self.held[f.id].append(reserve[0])
            return None, reserve[1]
        return None, min(waits) if waits else None

    def attend(self, kind: str, group: str, f: Flight, t: int) -> tuple:
        """The standard crew desk's crew of `kind` for f, ready to leave at t: the cheapest of its `options`, ties to
        the earlier letter, then the earlier departure, then the unit id; with none and no crew coming (`coming`), f is
        cancelled now (`wait`)."""
        options = self.options(kind, group, f, t, t)
        if not options:
            self.wait(kind, group, f, t, False)
            return None, None
        cost, rank, d, uid, act = min(options, key=lambda o: o[:4])
        self.fixed(t, f, kind, cost, rank, d, uid, act)
        self.take(kind, f, t, uid, act)
        if d > t:
            self.notes.append((kind, HOLD, d, "later"))
            return None, d
        return (uid, self.operate(self.state[uid], f, t)), None

    def options(self, kind: str, group: str, f: Flight, t: int, t0: int) -> list[tuple]:
        """The desk's ways, decided at t, to give f a crew of `kind` for a departure from t0, each (cost, rank,
        departure, unit, action), departing by f's deadline and legal for f: (a) its planned crew; (b) an
        airport-standby or called reserve with no trip at its origin, else an uncalled short-call reserve of the base
        called to report in time; (c) a swap, with a crew at its origin with no trip left, or one whose own trip the
        planned crew can fly instead; (d) a crew or reserve with no trip elsewhere, deadheading there on a flight with
        a free seat; (e) any of these legal for f only with the pilots' extension. The cost is f's passengers' wait
        from t0, the deadhead's price, the night away from base the crew must rest to take it, and what the move adds
        to the end of the day (`shift`). A planned crew a crew desk committed to a flight still to come keeps its leg
        whenever it can fly it, and takes no other trip."""
        out: list[tuple] = []
        now: dict[tuple, dict[str, str]] = {}

        def offer(rank: int, uid: str, act: tuple, plain: int | None, extended: int | None, price: float = 0.0,
                  night: float | None = None) -> None:
            moved = None
            for r, d, ext in ((rank, plain, False), (5, extended, True)):
                if d is None or d > f.deadline or (ext and plain is not None and d >= plain):
                    continue
                if moved is None:
                    moved = 0.0 if act[0] == "own" else self.shift(kind, f, uid, act[0] == "swap", now)
                rest = self.night(self.state[uid], f, d, ext) if night is None else night
                out.append((f.pax * (d - t0) + price + rest + moved, r, d, uid, act))

        mine = self.planned.get((kind, f.id))
        y = self.state.get(mine) if mine is not None else None
        held = mine in self.owned and mine in self.scripted
        if y is not None and self.available(y, f):
            if y.station == f.orig:
                offer(0, mine, ("own",), self.when(y, f, t0, False), self.when(y, f, t0, True))
            else:
                arrives = self.arrives(y, f, t)
                offer(0, mine, ("own",), None if arrives is None else max(t0, arrives), None, night=0.0)
        if held and out:
            return out
        for uid in sorted(self.at[(kind, group, f.orig)] - self.scripted - {mine}):
            x = self.state[uid]
            if not self.available(x, f):
                continue
            if x.unit.reserve and not x.trip:
                offer(1, uid, ("standby",), self.when(x, f, t0, False), self.when(x, f, t0, True))
            elif not x.trip or (y is not None and not held and self.carries(y, x.trip)):
                offer(3, uid, ("swap",), self.when(x, f, t0, False), self.when(x, f, t0, True))
        for uid, r, plain, extended in self.callable(kind, group, f, t, t0):
            offer(2, uid, ("call", r), plain, extended, night=0.0)
        for uid, ways in self.deadheads(kind, group, f, t, t0).items():
            plain, extended = ways
            if plain is not None:
                offer(4, uid, ("deadhead", plain[1]), plain[0], None, *plain[2:])
            if extended is not None and (plain is None or extended[0] < plain[0]):
                offer(4, uid, ("deadhead", extended[1]), None, extended[0], *extended[2:])
        return out

    def fixed(self, t: int, f: Flight, kind: str, cost: float, rank: int, d: int, uid: str, act: tuple) -> None:
        """With the log, the desk's move on f's `kind` at t: its option's letter, unit, cost and departure."""
        if self.log:
            self.fixes.append({"t": t, "flight": f.id, "kind": kind, "option": "abbcde"[rank], "unit": uid,
                               "act": act[0], "via": act[1] if len(act) > 1 else None, "cost": round(cost, 1),
                               "dep": d})

    def night(self, s: State, f: Flight, d: int, extend: bool) -> float:
        """The hotel s pays to take f at d if that needs a rest away from its base first."""
        how = self.operate(s, f, d, extend)
        if how is None or how[0] != "rest" or s.station == s.unit.home:
            return 0.0
        return crews.hotel(s.station, self.task.date) * s.unit.size

    def shift(self, kind: str, f: Flight, uid: str, swap: bool, now: dict[tuple, dict[str, str]]) -> float:
        """What `hand` putting uid on f would add to the end of the day as the run prices it (crews.Ends, per kind,
        type group and base): uid ends where the trip it takes does, and f's planned crew where it is, where its trip
        before f ends, or on a swap where uid's own trip does; every other unit where its day leads (`heading`).
        `now` keeps each base's headings for the decision."""
        y = self.planned.get((kind, f.id))
        x = self.state[uid]
        rest, after = [f.id], {}
        if y is not None and y != uid:
            sy = self.state[y]
            k = sy.trip.index(f.id) if f.id in sy.trip else len(sy.trip)
            rest = sy.trip[k:] or [f.id]
            left = sy.trip[:k] + (x.trip if swap else [])
            after[y] = self.lands(left[-1]) if left else sy.station
        after[uid] = self.lands(rest[-1])
        out = 0.0
        for key in {self.ends.of[v] for v in after}:
            if key not in now:
                now[key] = {v: self.heading(self.state[v]) for v in self.ends.members[key]}
            then = {**now[key], **{v: a for v, a in after.items() if v in now[key]}}
            out += self.ends.spread(key, tuple(sorted(then.values()))) - \
                self.ends.spread(key, tuple(sorted(now[key].values())))
        return out

    def heading(self, s: State) -> str:
        """Where s's day leads as things stand: where its last remaining leg lands, else its last planned deadhead,
        else where it is."""
        for legs in (s.trip, s.deadheads):
            fs = [self.task.flight(i) for i in legs if i in self.free]
            if fs:
                return self.lands(max(fs, key=lambda g: (g.sched_dep, g.id)).id)
        return s.station

    def lands(self, i: int) -> str:
        return self.diverted.get(i, self.task.flight(i).dest)

    def when(self, s: State, f: Flight, t: int, extend: bool) -> int | None:
        """The first minute from t, by f's deadline, at which s, at f's origin, may operate f."""
        d = max(t, s.ready)
        while d <= f.deadline:
            if self.operate(s, f, d, extend):
                return d
            nxt = self.later(s, f, d)
            if nxt is None or nxt <= d:
                return None
            d = nxt
        return None

    def arrives(self, s: State, f: Flight, t: int) -> int | None:
        """When s, not yet at f's origin, is there and free as scheduled from t: by its trip's legs before f, legal
        along them, or by the deadhead booked for it; None if neither brings it."""
        ride = self.rides.get(f.id)
        if f.id in s.trip and s.trip.index(f.id) > 0:
            path = [self.task.flight(i) for i in s.trip[:s.trip.index(f.id)]]
            if self.outlook(s, f)[0] == NO_CREW:
                return None
        elif ride is not None and ride[0] == s.unit.id and ride[1] not in self.done:
            path = [self.task.flight(ride[1])]
        else:
            return None
        e = s.ready
        for g in path:
            e = max(e, g.sched_dep, t) + g.sched_block + crews.CONNECT
        return e if path[0].orig == s.station and path[-1].dest == f.orig else None

    def carries(self, s: State, legs: Sequence[int]) -> bool:
        """Whether s, from where it is, can fly `legs` in order as scheduled, legal without the extension, in its
        period or, rested by the first, in a new one."""
        fs = [self.task.flight(i) for i in legs]
        if s.station != fs[0].orig or s.ready > fs[0].sched_dep or \
                any(a.dest != b.orig for a, b in zip(fs, fs[1:], strict=False)):
            return False
        u = s.unit
        starts = [(s.report, s.hour, s.flown, s.minutes, s.rap)] if s.report is not None else []
        release, r = s.release(self.task.start), (fs[0].sched_dep - crews.BRIEF) // 60 * 60
        if s.period == 1 and release is not None and r >= release + crews.rest(u.kind):
            starts.append((r, crews.local_hour(self.local, s.station, r), 0, 0, None))
        return any(all(self.fits(u, report, hour, flown + k, minutes + sum(g.sched_block for g in fs[:k]), g,
                                 g.sched_dep, False, rap, False) for k, g in enumerate(fs))
                   for report, hour, flown, minutes, rap in starts)

    def callable(self, kind: str, group: str, f: Flight, t: int, t0: int) -> list[tuple]:
        """The uncalled short-call reserves of the base at f's origin that, called from t, report on the last whole
        hour leaving BRIEF before t0 (or as soon after as they can): (unit, report, departure legal without the
        extension, departure legal with it), each None if not legal then."""
        out = []
        for uid in sorted(self.at[(kind, group, f.orig)] - self.scripted):
            s = self.state[uid]
            u = s.unit
            if u.reserve != "short" or s.report is not None or s.claim is not None:
                continue
            r = max(crews.on_hour(max(t, u.rap, self.task.start) + u.lead), (t0 - crews.BRIEF) // 60 * 60)
            d = max(t0, r + crews.BRIEF)
            if r - u.lead <= u.until:
                legal = [d if self.fresh(s, r, f, d, extend) else None for extend in (False, True)]
                if any(legal):
                    out.append((uid, r, *legal))
        return out

    def deadheads(self, kind: str, group: str, f: Flight, t: int, t0: int) -> dict[str, list]:
        """Per crew or reserve of the kind and group elsewhere with no trip, not held, its earliest ways to f's origin
        as scheduled from t, riding a flight there with a seat free, legal to ride and then to fly f, without the
        extension and with it: each (departure, flight ridden, the deadhead's price, the hotel of a rest away from base
        it starts to ride) or None."""
        out = {}
        for uid in self.kin[(kind, group)]:
            s = self.state[uid]
            if s.station == f.orig or s.trip or s.claim is not None or uid in self.scripted:
                continue
            best: list = [None, None]
            for g in self.route[(s.station, f.orig)]:
                e = max(g.sched_dep, t)
                if g.id in self.done or e < s.ready or self.free[g.id] - self.booked[g.id] < s.unit.size:
                    continue
                period = self.boards(s, g, e, t)
                if period is None:
                    continue
                d = max(t0, e + g.sched_block + crews.CONNECT)
                night = crews.hotel(s.station, self.task.date) * s.unit.size if period[5] and \
                    s.station != s.unit.home else 0.0
                for k, extend in enumerate((False, True)):
                    if self.fits(s.unit, *period[:4], f, d, False, period[4], extend) and (best[k] is None or
                                                                                            d < best[k][0]):
                        best[k] = (d, g.id, crews.deadhead(kind, group, g.sched_block, self.task.date), night)
            if best[0] or best[1]:
                out[uid] = best
        return out

    def boards(self, s: State, g: Flight, e: int, t: int) -> tuple | None:
        """The period s rides g in, leaving at e, called at t if an uncalled short-call reserve, as `ride` would start
        it: (report, hour, segments flown, minutes flown, short-call period start, whether it starts after rest), or
        None if it may not ride."""
        u = s.unit
        r = (e - crews.BRIEF) // 60 * 60
        if u.reserve == "short" and s.report is None:
            if r < crews.on_hour(max(t, u.rap, self.task.start) + u.lead) or r - u.lead > u.until:
                return None
            return r, crews.local_hour(self.local, u.home, r), 0, 0, u.rap, False
        release = s.release(self.task.start)
        if s.period == 1 and release is not None and r >= release + crews.rest(u.kind):
            return r, crews.local_hour(self.local, s.station, r), 0, 0, None, True
        if s.report is not None and not self.over(s, e) and (u.kind == "pilot" or
                                                             crews.cabin_fits(s.report, e, g.sched_block)):
            return s.report, s.hour, s.flown, s.minutes, s.rap, False
        return None

    def fresh(self, s: State, r: int, f: Flight, d: int, extend: bool) -> bool:
        """Whether s, reporting at r at f's origin, may operate f leaving at d (with the extension if `extend`)."""
        u = s.unit
        return self.fits(u, r, crews.local_hour(self.local, f.orig, r), 0, 0, f, d, False,
                         u.rap if u.reserve == "short" else None, extend)

    def take(self, kind: str, f: Flight, t: int, uid: str, act: tuple) -> None:
        """Puts uid on f's `kind` by the desk's `act`, decided at t: a short-call reserve is called to report at its
        report, a deadhead booked, and the unit takes f and the rest of its planned crew's trip (`hand`), held for f
        until it leaves."""
        if act[0] in ("own", "extend"):
            return
        s = self.state[uid]
        u = s.unit
        if act[0] == "call":
            s.call, s.earliest, s.ready = act[1] - u.lead, act[1], act[1] + crews.BRIEF
        elif act[0] == "deadhead":
            s.deadheads.append(act[1])
            self.booked[act[1]] += u.size
            self.rides[f.id] = (uid, act[1])
            if u.reserve == "short" and s.report is None:
                s.call = max(t, u.rap, self.task.start)
                s.earliest = crews.on_hour(s.call + u.lead)
        y = self.planned.get((kind, f.id))
        self.hand(kind, f, uid, act[0] == "swap")
        s.claim = f.id
        self.held[f.id].append(uid)
        if self.log:
            self.rejudge({uid} if y is None else {uid, y}, t)

    def hand(self, kind: str, f: Flight, uid: str, swap: bool) -> None:
        """uid takes f and the rest of the trip of f's planned crew from f on; on a swap, that crew takes uid's own
        trip."""
        y = self.planned.get((kind, f.id))
        x = self.state[uid]
        rest, own = [f.id], list(x.trip)
        if y is not None and y != uid:
            sy = self.state[y]
            if f.id in sy.trip:
                k = sy.trip.index(f.id)
                rest, sy.trip = sy.trip[k:], sy.trip[:k]
            if swap and own:
                sy.trip += own
                for i in own:
                    self.planned[(kind, i)] = y
                    self.watch[y].add(i)
        x.trip = rest
        for i in rest:
            self.planned[(kind, i)] = uid
            self.watch[uid].add(i)

    def coming(self, kind: str, group: str, f: Flight) -> int | None:
        """The first scheduled arrival at f's origin by f's deadline of a trip leg of a crew of the kind and group,
        wherever it is (f's planned crew's only before f), or None."""
        mine = self.planned.get((kind, f.id))
        out = None
        for uid in self.kin[(kind, group)]:
            trip = self.state[uid].trip
            if uid == mine and f.id in trip:
                trip = trip[:trip.index(f.id)]
            for i in trip:
                g = self.task.flight(i)
                if g.dest == f.orig and g.sched_arr <= f.deadline and (out is None or g.sched_arr < out):
                    out = g.sched_arr
        return out

    def wait(self, kind: str, group: str, f: Flight, t: int, wake: bool) -> bool:
        """With no option for f's `kind` at t: HOLD until the first crew a trip brings to its origin is there, else f
        is stuck, cancelled now or, without `drop`, a CANCEL asked of ops; whether it is stuck."""
        come = self.coming(kind, group, f)
        if come is not None:
            self.notes.append((kind, HOLD, come + crews.CONNECT, "coming"))
            return False
        self.notes.append((kind, CANCEL, None, "none"))
        if self.drop:
            self.dropped[f.id] = t
            if wake:
                self.wake.append((f.id, t))
        return True

    def say(self, f: Flight, t: int) -> None:
        """The desk's word on f from the decision at t, logged when its type or minute changes: CANCEL when a kind is
        stuck (with `drop` the cancellation itself, accepted at once), else HOLD until the latest minute a kind's crew
        can be there; in the request log's shape (sender, receiver, type, params, free text, sent, reply, replied,
        actions)."""
        notes, self.notes = self.notes, []
        if not notes:
            return
        kind, said, minute, reason = next((n for n in notes if n[1] == CANCEL), None) or max(notes, key=lambda n: n[2])
        if self.standing.get(f.id) == (said, minute):
            return
        self.standing[f.id] = (said, minute)
        done = said == CANCEL and self.drop
        self.messages.append({"sender": "crew", "receiver": "ops", "type": said,
                              "params": {"flight": f.id, "kind": kind, "minute": minute, "reason": reason},
                              "text": "", "sent": t, "reply": "accept" if done else None,
                              "replied": t if done else None, "actions": [{"cancel": f.id, "at": t}] if done else []})

    def check(self, f: Flight, t: int) -> None:
        """The desk at t on f, from its earliest departure t0 in the world the day runs in (`lb`: the truth's, the
        day as known after sim.Dispatch.switch): for each kind whose planned crew can't leave by t0, or its schedule if
        later, without the extension, it takes the cheapest of its `options` now unless that is waiting for that crew
        (HOLD if that leaves later); with none and no crew coming it cancels f now (`wait`). A kind the plan names it
        leaves to the crew desk until f boards; a flight the plan cancels by t (its first pass at the start included)
        it leaves alone."""
        if f.id in self.done or f.id in self.dropped or self.cut.get(f.id, t + 1) <= t:
            return
        group, t0 = crews.TYPE[f.family], max(t, self.task.earliest(f, min(0, self.lb[f.id])))
        on_time = max(t0, f.sched_dep)
        for kind in crews.KINDS:
            mine = self.planned.get((kind, f.id))
            if (kind, f.id) in self.named:
                continue
            y = self.state.get(mine) if mine is not None else None
            if y is not None and self.available(y, f):
                d = self.when(y, f, t0, False) if y.station == f.orig else self.arrives(y, f, t)
                if d is not None and d <= on_time:
                    continue
            options = self.options(kind, group, f, t, t0)
            if not options:
                if self.wait(kind, group, f, t, True):
                    break
                continue
            cost, rank, d, uid, act = min(options, key=lambda o: o[:4])
            self.fixed(t, f, kind, cost, rank, d, uid, act)
            if d > on_time:
                self.notes.append((kind, HOLD, d, "later"))
            self.take(kind, f, t, uid, act)
        self.say(f, t)

    def available(self, s: State, f: Flight) -> bool:
        """Whether the default rule may give s flight f: not held for another flight, and no uncalled short-call
        reserve unless held for f."""
        if s.claim is not None:
            return s.claim == f.id
        return not (s.unit.reserve == "short" and s.report is None)

    def reserve(self, kind: str, group: str, f: Flight, t: int) -> tuple[str, int] | None:
        """An uncalled short-call reserve of the base, unnamed, and the earliest it could leave if called now."""
        best = None
        for u in sorted(self.at[(kind, group, f.orig)] - self.scripted):
            s = self.state[u]
            if s.unit.reserve != "short" or s.report is not None or s.claim is not None:
                continue
            call = max(t, s.unit.rap, self.task.start)
            report = crews.on_hour(call + s.unit.lead)
            if report - s.unit.lead > s.unit.until:
                continue
            go = report + crews.BRIEF
            if best is None or go < best[1]:
                best = (u, go)
        return best

    def operate(self, s: State, f: Flight, t: int, extend: bool = True) -> tuple | None:
        """How s may operate f leaving at t: ("on",) in its period, ("report", r, hour) or ("rest", r, hour) starting
        one; None if it may not (a pilot within the extension if `extend`)."""
        u = s.unit
        if s.station != f.orig or s.ready > t:
            return None
        own = s.period == 1 and f.id in u.trip
        if u.id + "#2" in self.reports and s.period == 1 and (start := self.begin(s, t)) is not None:
            return start[:3] if self.fits(u, start[1], start[2], 0, 0, f, t, False, start[3], extend) else None
        if s.report is not None and not self.over(s, t) and \
                self.fits(u, s.report, s.hour, s.flown, s.minutes, f, t, own, s.rap, extend):
            return ("on",)
        start = self.begin(s, t)
        if start is not None and self.fits(u, start[1], start[2], 0, 0, f, t, False, start[3], extend):
            return start[:3]
        return None

    def begin(self, s: State, t: int) -> tuple | None:
        """The period s would start to leave at t: ("report"|"rest", report, hour, short-call period start)."""
        u = s.unit
        if u.reserve == "short" and s.report is None:
            r = self.reports.get(u.id, (t - crews.BRIEF) // 60 * 60)
            if r % 60 or r < s.earliest or r - u.lead > u.until or not r + crews.BRIEF <= t < r + crews.BRIEF + 60:
                return None
            return "report", r, crews.local_hour(self.local, u.home, r), u.rap
        release = s.release(self.task.start)
        if s.period != 1 or release is None:
            return None
        r = self.reports.get(u.id + "#2", (t - crews.BRIEF) // 60 * 60)
        if r % 60 or r < release + crews.rest(u.kind) or not r + crews.BRIEF <= t < r + crews.BRIEF + 60:
            return None
        return "rest", r, crews.local_hour(self.local, s.station, r), None

    def later(self, s: State, f: Flight, t: int) -> int | None:
        """The next minute after t at which s, there, might take f: when it is free, reports or is rested."""
        u = s.unit
        if s.ready > t:
            return s.ready
        if u.reserve == "short" and s.report is None:
            r = self.reports.get(u.id, max(s.earliest, (t - crews.BRIEF) // 60 * 60))
            go = r + crews.BRIEF
            return go if go > t else None
        release = s.release(self.task.start)
        if s.period != 1 or release is None:
            return None
        r = self.reports.get(u.id + "#2", crews.on_hour(release + crews.rest(u.kind)))
        go = max(r + crews.BRIEF, crews.on_hour(release + crews.rest(u.kind)) + crews.BRIEF)
        return go if go > t else None

    def fits(self, u: Unit, report: int, hour: int, flown: int, minutes: int, f: Flight, t: int, own: bool,
             rap: int | None, extend: bool = True) -> bool:
        if u.kind == "pilot":
            return crews.pilot_fits(report, hour, flown, minutes, t, f.sched_block, rap, extend)
        return crews.cabin_fits(report, t, f.sched_block, own)

    def over(self, s: State, t: int) -> bool:
        """An airport-standby reserve that has not moved may leave only until its shift ends."""
        u = s.unit
        return u.reserve == "standby" and not s.moved and u.until is not None and t > u.until

    def ride(self, s: State, f: Flight, t: int) -> tuple | None:
        u = s.unit
        if s.station != f.orig or s.ready > t:
            return None
        start = self.begin(s, t)
        if s.report is None or (start is not None and start[0] == "rest"):
            return start[:3] if start is not None else None
        if not self.over(s, t) and (u.kind == "pilot" or (s.period == 1 and f.id in u.deadheads)
                                    or crews.cabin_fits(s.report, t, f.sched_block)):
            return ("on",)
        return None

    def fly(self, f: Flight, t: int, arr: int, entry: Mapping) -> None:
        picks = self.picks.pop(f.id, {})
        for kind in crews.KINDS:
            if kind not in picks:
                self.uncrewed.append(f.id)
                break
        touched = set()
        for kind, (uid, how) in picks.items():
            s = self.state[uid]
            self.start(s, how, f.orig)
            if self.log:
                self.limits[f.id][kind] = (uid, *self.last(s, f))
            self.move(s, f, t, arr, True)
            self.crewed[f.id][kind] = uid
            touched.add(uid)
        room = self.free[f.id]
        riders = list(self.riding.get(f.id, ()))
        if self.fill:
            riders += sorted(u for u, s in self.state.items() if f.id in s.deadheads and
                             (u not in self.scripted or u in self.owned) and u not in riders)
        for uid in riders:
            s = self.state[uid]
            how = self.ride(s, f, t)
            if how is None or s.unit.size > room:
                continue
            room -= s.unit.size
            self.start(s, how, f.orig)
            self.move(s, f, t, arr, False)
            self.riders[f.id].append(uid)
            touched.add(uid)
        for kind in crews.KINDS:
            uid = self.planned.get((kind, f.id))
            if uid is not None and f.id in self.state[uid].trip:
                self.state[uid].trip.remove(f.id)
                touched.add(uid)
        self.release(f.id, t)
        self.done.add(f.id)
        if self.owned:
            self.rescript()
        if self.log:
            self.status[f.id].append((t, "flown" if f.id not in self.uncrewed else "flown without a legal crew",
                                      None))
            self.rejudge(touched, t)
        self.recheck(touched, t)

    def last(self, s: State, f: Flight) -> tuple[int | None, int | None]:
        """The last minute s, in its period, may take f off the ground without and with the pilots' extension (None
        for a team's own trip leg: (b)(14))."""
        u = s.unit
        if u.kind == "pilot":
            return tuple(crews.pilot_limit(s.report, s.hour, s.flown + 1, s.rap, extend) - f.sched_block
                         for extend in (False, True))
        if s.period == 1 and f.id in u.trip:
            return None, None
        last = s.report + crews.CABIN_DUTY - crews.DEBRIEF - f.sched_block
        return last, last

    def release(self, i: int, t: int | None = None) -> None:
        """The units held for flight i, which flew (at t) or was cancelled: a short-call reserve that never reported is
        as if never called, a flight waiting at its base tries again at t, and a deadhead booked for i and not taken is
        given up."""
        ride = self.rides.pop(i, None)
        if ride is not None and ride[1] in self.state[ride[0]].deadheads:
            self.state[ride[0]].deadheads.remove(ride[1])
            self.booked[ride[1]] -= self.state[ride[0]].unit.size
        for uid in self.held.pop(i, ()):
            s = self.state[uid]
            if s.claim == i:
                s.claim = None
                if s.report is None:
                    s.call = None
                    s.earliest, s.ready = self.uncalled[s.unit.id]
                    u = s.unit
                    for j in self.waiting.pop((u.kind, u.group, s.station), ()) if t is not None else ():
                        self.wake.append((j, t))

    def start(self, s: State, how: tuple, station: str) -> None:
        if how[0] == "on":
            return
        _, r, hour = how
        u = s.unit
        if how[0] == "rest":
            release = s.release(self.task.start)
            if s.duties:
                s.duties[-1]["release"] = release
            if s.station != u.home:
                s.hotels.append({"station": s.station, "from": release, "to": r})
            s.period, s.rap = 2, None
        else:
            s.call = s.call if s.call is not None else r - u.lead
            s.rap = u.rap
        s.report, s.hour, s.flown, s.minutes, s.landed = r, hour, 0, 0, None
        s.duties.append(duty(r, station, "rest" if how[0] == "rest" else "call"))

    def move(self, s: State, f: Flight, t: int, arr: int, op: bool) -> None:
        u = s.unit
        dest = self.diverted.get(f.id, f.dest)
        self.at[(u.kind, u.group, s.station)].discard(u.id)
        s.station, s.ready, s.landed, s.moved = dest, arr + crews.CONNECT, arr, True
        if op:
            s.flown, s.minutes = s.flown + 1, s.minutes + (arr - t)
        else:
            if f.id in s.deadheads:
                s.deadheads.remove(f.id)
        if not s.duties:
            s.duties.append(duty(s.report, u.at or None, "on duty"))
        s.duties[-1]["legs" if op else "deadheads"].append((f.id, t, arr))
        s.duties[-1]["release"] = arr + crews.DEBRIEF
        self.at[(u.kind, u.group, dest)].add(u.id)
        for i in self.waiting.pop((u.kind, u.group, dest), ()):
            self.wake.append((i, s.ready))

    def woken(self) -> list[tuple[int, int]]:
        out, self.wake = self.wake, []
        return out

    def cancel(self, i: int, decided: int) -> None:
        touched = set()
        for kind in crews.KINDS:
            uid = self.planned.get((kind, i))
            if uid is not None and i in self.state[uid].trip:
                self.state[uid].trip.remove(i)
                touched.add(uid)
        self.release(i)
        self.done.add(i)
        if self.owned:
            self.rescript()
        if self.log:
            self.status[i].append((decided, "cancelled", None))
            self.rejudge(touched, decided)
        self.recheck(touched, decided)

    def recheck(self, units: set[str], t: int) -> None:
        """With the desk, the flights of the units a move touched, checked for certain cancellation at t."""
        if self.desk:
            for i in sorted({i for uid in units for i in self.watch[uid]} - self.done):
                self.check(self.task.flight(i), t)

    def rejudge(self, units: set[str], t: int) -> None:
        for uid in units:
            for i in sorted(self.watch[uid]):
                if i not in self.done:
                    self.judge(self.task.flight(i), t)

    def judge(self, f: Flight, t: int) -> None:
        """f's crew status from minute t, logged when it changes."""
        worst = (LEGAL, None)
        for kind in crews.KINDS:
            uid = self.named.get((kind, f.id)) or self.planned.get((kind, f.id))
            got = (NO_CREW, None) if uid is None else self.flag(self.state[uid], f, t)
            if got[0] == NO_CREW or (got[0] == TIMES_OUT and (worst[0] == LEGAL or got[1] < worst[1])):
                worst = got
            if worst[0] == NO_CREW:
                break
        log = self.status[f.id]
        if not log or log[-1][1:] != worst:
            log.append((t, *worst))

    def flag(self, s: State, f: Flight, t: int) -> tuple[str, int | None]:
        """s's `outlook` for f as the status log shows it at t: riding a deadhead booked to f's origin, as it will be
        there in the period the ride leaves it in."""
        got = self.outlook(s, f)
        rides = [self.task.flight(i) for i in s.deadheads if i in self.free and i not in self.done]
        rides = [g for g in rides if g.orig == s.station and g.dest == f.orig]
        if got[0] != NO_CREW or not rides:
            return got
        g = min(rides, key=lambda g: (g.sched_dep, g.id))
        e = max(g.sched_dep, t)
        period = self.boards(s, g, e, t)
        if period is None:
            return got
        report, hour, flown, minutes, rap, rested = period
        there = replace(s, station=f.orig, ready=e + g.sched_block + crews.CONNECT, report=report, hour=hour,
                        flown=flown, minutes=minutes, rap=rap, period=2 if rested else s.period)
        return self.outlook(there, f)

    def outlook(self, s: State, f: Flight) -> tuple[str, int | None]:
        """Whether s, following its trip to f, can take it: legal, timing out at its last legal departure, or not."""
        u = s.unit
        trip = s.trip
        if f.id in trip:
            before = trip[:trip.index(f.id)]
            path = [self.task.flight(i) for i in before]
            station = path[0].orig if path else f.orig
            chain = [*path, f]
            for a, b in zip(chain, chain[1:], strict=False):
                if a.dest != b.orig:
                    return NO_CREW, None
        else:
            before, path, station = [], [], f.orig
        if s.station != station:
            return NO_CREW, None
        if s.report is None:
            return LEGAL, None
        own = s.period == 1 and f.id in u.trip
        if u.kind == "pilot":
            minutes = s.minutes + sum(g.sched_block for g in path) + f.sched_block
            if minutes > crews.flight_time(s.hour):
                return NO_CREW, None
            last = crews.pilot_limit(s.report, s.hour, s.flown + len(path) + 1, s.rap) - f.sched_block
        elif own:
            return LEGAL, None
        else:
            last = s.report + crews.CABIN_DUTY - crews.DEBRIEF - f.sched_block
        if last < max(f.sched_dep, s.ready):
            return NO_CREW, None
        return TIMES_OUT, last

    def spent(self) -> dict[str, dict[str, float]]:
        """Each unit's crew cost: its deadheads off its trip, its nights away from base, its share of the end of the
        day."""
        d = self.task.date
        end = self.ends.cost({uid: s.station for uid, s in self.state.items()})
        out = {}
        for uid, s in self.state.items():
            u = s.unit
            planned = set(u.deadheads)
            dh = sum(crews.deadhead(u.kind, u.group, arr - t, d) for duty in s.duties for i, t, arr in
                     duty["deadheads"] if i not in planned)
            out[uid] = {"deadhead": dh, "hotel": sum(crews.hotel(h["station"], d) * u.size for h in s.hotels),
                        "end": end[uid]}
        return out

    def unit_log(self, spent: Mapping[str, dict[str, float]]) -> dict[str, dict]:
        out = {}
        for uid, s in self.state.items():
            if not s.duties and s.call is None:
                continue
            out[uid] = {"kind": s.unit.kind, "group": s.unit.group, "home": s.unit.home, "reserve": s.unit.reserve,
                        "call": s.call, "duties": s.duties, "hotels": s.hotels, "station": s.station,
                        "cost": spent[uid]}
        return out


def entries(plan: Mapping) -> list[tuple[int | None, tuple]]:
    """A plan's crew entries, each with the minute it takes effect (None: from the start), in plan order: the units it
    names (`crew`), lets ride (`deadhead`) and has report (`reports`) from the start, and those it names, lets ride or
    calls from a minute on (`names` [from, to or None, kind, unit], `rides` [from, to or None, unit], `calls` [from,
    key, report]), a name or a ride ending at its `to`, the planned rides it keeps units off (`keeps` [from, unit,
    flight]) and the units whose trips it swaps (`swaps` [from, unit, unit])."""
    out = []
    for k, v in plan.items():
        if k in sim.EXTRA:
            continue
        i = int(k)
        out += [(None, ("name", (kind, i), uid, None)) for kind, uid in v.get("crew", {}).items()]
        out += [(None, ("ride", i, uid, None)) for uid in v.get("deadhead", ())]
        for frm, to, kind, uid in v.get("names", ()):
            out += [(frm, ("name", (kind, i), uid, None))] + ([(to, ("unname", (kind, i), uid, frm))] if to else [])
        for frm, to, uid in v.get("rides", ()):
            out += [(frm, ("ride", i, uid, None))] + ([(to, ("unride", i, uid, frm))] if to else [])
    out += [(None, ("report", key, key.split("#")[0], r)) for key, r in plan.get("reports", {}).items()]
    out += [(frm, ("report", key, key.split("#")[0], r)) for frm, key, r in plan.get("calls", ())]
    out += [(frm, ("keep", i, uid, None)) for frm, uid, i in plan.get("keeps", ())]
    out += [(frm, ("swap", b, a, None)) for frm, a, b in plan.get("swaps", ())]
    return out


def duty(report: int | None, at: str | None, begins: str) -> dict:
    """A duty period in the log: its report, where, how it began (on duty at the start, a reserve's call, after
    rest), its legs and deadheads (flight, departure, arrival) and its release."""
    return {"report": report, "at": at, "begins": begins, "legs": [], "deadheads": [], "release": None}


def clock(minute: int | None) -> str | None:
    """A task minute as HH:MM on the hub's clock, +1 for the next day."""
    if minute is None:
        return None
    day, m = divmod(minute, 1440)
    return f"{m // 60:02d}:{m % 60:02d}" + (f"+{day}" if day else "")


def statuses(day: CrewDay) -> dict[int, list[dict]]:
    return {i: [{"at": clock(t), "status": st, "until": clock(until)} for t, st, until in log]
            for i, log in sorted(day.status.items())}

