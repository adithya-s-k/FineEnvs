"""Small hand-made days for the sim, MIP and reference tests: a few flights and tails at JFK with the programs given,
every program covering every origin."""

import random
from dataclasses import replace

from program_helpers import EVERY, adv

from airops_core.core.model import Airport
from airops_core.programs import network
from airops_core.programs.advisories import HUBS, Advisory, Stop
from airops_core.programs.model import (
    Busy,
    Connection,
    Crew,
    Flight,
    Move,
    Outcome,
    Programs,
    Segment,
    Tail,
    Task,
)

PLACES = {"JFK": (40.64, -73.78), "BOS": (42.36, -71.01), "BUF": (42.94, -78.73), "DCA": (38.85, -77.04),
          "MCO": (28.43, -81.31), "PWM": (43.65, -70.31)}
TURN = 30


def flight(i: int, orig: str, dest: str, dep: int, block: int, pax: int = 100, lb: int = 0,
           deadline: int | None = None, sched_block: int | None = None, actual: Outcome | None = None) -> Flight:
    sb = block if sched_block is None else sched_block
    return Flight(id=i, flight=f"B6 {100 + i}", orig=orig, dest=dest, sched_dep=dep, sched_arr=dep + sb,
                  sched_block=sb, block=block, pax=pax, pax_raw=pax, family="A320", lb=lb, nas_removed=0,
                  controlled=None, gs_held=None, exempt=False, pair=None,
                  deadline=dep + 240 if deadline is None else deadline,
                  actual=actual or Outcome(False, False, "", dep, dep + block, dest, "N1"))


def crewed(f: Flight, deadline: int | None) -> Flight:
    """f with a crew whose deadline is `deadline`, resting 10 h after it (None: a departure from a base)."""
    return replace(f, crew=Crew(f"{f.actual.tail}/0", f.sched_dep - 45, f.orig, 2, 720, deadline,
                                None if deadline is None else deadline + 600))


def tail(name: str, pos: str, ready: int = 0, end: str | None = None, repositions: tuple[Move, ...] = (),
         busy: tuple[Busy, ...] = (), seats: int = 150, legs: tuple[int, ...] = ()) -> Tail:
    return Tail(name, "A320", "A320-232", seats, pos, ready, legs, busy, repositions, end or pos, end or pos)


def segment(id: str, since: int, until: int, start: int, end: int, slots: tuple[int, ...]) -> Segment:
    return Segment(id, "JFK", since, until, start, end, 0.0, slots, ())


def stop(start: int, end: int, hub: str = "JFK") -> Stop:
    return Stop("gs", hub, start, end, EVERY)


def day(flights: list[Flight], tails: list[Tail], segments: tuple[Segment, ...] = (), stops: tuple[Stop, ...] = (),
        aar: int = 99, start: int = 0, cutoff: int = 1620, advisories: tuple[Advisory, ...] = ()) -> Task:
    """`advisories` (at JFK) add to one GDP advisory per segment; the forecast reads them, the truth `segments` and
    `stops`."""
    advisories = tuple(adv(s.id, "gdp", s.since, s.start, s.end, (30,) * (s.end // 60 - s.start // 60 + 1))
                       for s in segments) + advisories
    programs = {h: Programs(h, aar, 1.0, 0.0, {}, advisories if h == "JFK" else (),
                            tuple(s for s in segments if s.hub == h), tuple(s for s in stops if s.hub == h), ())
                for h in HUBS}
    return Task(task_id="b6-jfk-20240101-0100", split="", carrier="B6", hub="JFK", date="2024-01-01",
                tz="America/New_York", base=0, window=(start + 60, start + 360), start=start, slide=0, dmax=900,
                cutoff=cutoff, turn={"A320": TURN},
                airports={c: Airport(c, c, lat, lon, "America/New_York") for c, (lat, lon) in PLACES.items()},
                artcc={}, tails=tuple(tails), flights=tuple(flights), programs=programs, clauses={}, sources={})


GDP = segment("g", 0, 1000, 150, 400, (180, 240, 300))


def three_into_a_program(**kw) -> Task:
    """Three flights into JFK, each scheduled to land at 160 inside a program with slots at 180, 240 and 300."""
    flights = [flight(0, "BUF", "JFK", 100, 60), flight(1, "DCA", "JFK", 100, 60),
               flight(2, "MCO", "JFK", 100, 60, deadline=600, **kw)]
    return day(flights, [tail("N1", "BUF", end="JFK"), tail("N2", "DCA", end="JFK"), tail("N3", "MCO", end="JFK")],
               segments=(GDP,))


def shortened(task: Task, slack: int = 150) -> Task:
    """The task with every deadline at most `slack` after the earliest departure, to keep a test's MIP small."""
    return replace(task, flights=tuple(replace(f, deadline=min(f.deadline, task.earliest(f, f.lb) + slack))
                                       for f in task.flights))


def busy_day() -> Task:
    """N2 lands late at JFK at 220 and is busy there until 400, while N1 idles at JFK for the 170 departure."""
    flights = [flight(0, "BUF", "JFK", 100, 60, lb=60), flight(1, "JFK", "DCA", 170, 60),
               flight(2, "JFK", "MCO", 430, 150)]
    return day(flights, [tail("N1", "JFK", end="DCA"), tail("N2", "BUF", busy=(Busy(0, 400),), end="MCO")])


def moved_day(lb: int = 0) -> Task:
    """N1 flies BUF-JFK and JFK-BUF, then a Move takes it from BUF to MCO for MCO-JFK at 330; N2 idles at BUF, N3,
    ready at JFK from 140, flies JFK-DCA at 150."""
    flights = [flight(0, "BUF", "JFK", 20, 60), flight(1, "JFK", "BUF", 120, 60, lb=lb),
               flight(2, "MCO", "JFK", 330, 150), flight(3, "JFK", "DCA", 150, 60)]
    move = Move(1, "BUF", "MCO", 180, 300)
    return day(flights, [tail("N1", "BUF", end="JFK", repositions=(move,), legs=(0, 1, 2)), tail("N2", "BUF"),
                         tail("N3", "JFK", 140, end="DCA", legs=(3,))])


def broken_day() -> Task:
    """N1's chain breaks twice: BUF-JFK, a Move JFK-PWM for PWM-BOS, a Move BOS-DCA for DCA-MCO and MCO-JFK."""
    flights = [flight(0, "BUF", "JFK", 20, 60), flight(1, "PWM", "BOS", 200, 50), flight(2, "DCA", "MCO", 400, 150),
               flight(3, "MCO", "JFK", 640, 150)]
    moves = (Move(0, "JFK", "PWM", 80, 170), Move(1, "BOS", "DCA", 250, 370))
    return day(flights, [tail("N1", "BUF", end="JFK", repositions=moves, legs=(0, 1, 2, 3))])


def quick_day(ground: int | None = 10) -> Task:
    """N1 lands BUF-JFK at 160 and JetBlue turned it into JFK-DCA at 170, `ground` minutes after (None: no such
    turn); JFK-MCO also leaves at 170, and N2 idles at JFK."""
    flights = [flight(0, "BUF", "JFK", 100, 60), flight(1, "JFK", "DCA", 170, 60), flight(2, "JFK", "MCO", 170, 150)]
    turned = () if ground is None else (Connection(0, 1, ground),)
    return day(flights, [replace(tail("N1", "BUF", end="DCA", legs=(0, 1)), connections=turned),
                         tail("N2", "JFK", end="MCO", legs=(2,))])


def turned_through_day(ground: int | None = 20) -> Task:
    """N1 lands BUF-JFK at 80; BTS left it on JFK-BOS at 95, which a Move closes back to JFK, but JetBlue turned it
    into JFK-DCA at 100, `ground` minutes after landing (None: no such turn)."""
    flights = [flight(0, "BUF", "JFK", 20, 60), flight(1, "JFK", "BOS", 95, 50), flight(2, "JFK", "DCA", 100, 60)]
    turned = () if ground is None else (Connection(0, 2, ground),)
    move = Move(1, "BOS", "JFK", 145, 100 - TURN)
    return day(flights, [replace(tail("N1", "BUF", end="DCA", repositions=(move,), legs=(0, 1, 2)),
                                 connections=turned)])


def network_day(rng: random.Random, n_tails: int = 4) -> Task:
    """A small random whole-network day: each tail's legs are the ones it flew, in the order it flew them, a station
    gap between two a Move bound to the leg before it, some long daytime gaps busy blocks; legs JetBlue cancelled sit
    on no tail; tomorrow's needs are pooled from the schedule (network.needs); a GDP, maybe a stop, and some turns
    JetBlue flew faster than the rule."""
    flights, tails = [], []
    for k in range(n_tails):
        here, t, legs, moves, busy = rng.choice(list(PLACES)), rng.randrange(0, 120), [], [], []
        start = here
        for _ in range(rng.randrange(1, 5)):
            if legs and rng.random() < 0.3:
                there = rng.choice([c for c in PLACES if c != here])
                t += rng.randrange(0, 60)
                moves.append(Move(legs[-1], here, there, flights[-1].sched_arr, t - TURN))
                here = there
            elif legs and rng.random() < 0.15:
                t += 240
                busy.append(Busy(legs[-1], t - TURN))
            dest = rng.choice([c for c in PLACES if c != here])
            block = rng.randrange(50, 150)
            flights.append(flight(len(flights), here, dest, t, block + rng.choice([0, 0, 20]),
                                  pax=rng.randrange(50, 151), lb=rng.choice([0, 0, 30, 90, 200]),
                                  deadline=t + rng.randrange(150, 300), sched_block=block,
                                  actual=Outcome(False, False, "", t, t + block, dest, f"N{k}")))
            legs.append(len(flights) - 1)
            here, t = dest, t + block + rng.randrange(TURN - 10, 90)
        tails.append(tail(f"N{k}", start, ready=0, end=here, repositions=tuple(moves), busy=tuple(busy),
                          legs=tuple(legs)))
    for _ in range(rng.randrange(1, 4)):
        orig, dest = rng.sample(list(PLACES), 2)
        t, block = rng.randrange(0, 400), rng.randrange(50, 150)
        flights.append(flight(len(flights), orig, dest, t, block, pax=rng.randrange(50, 151), deadline=t + 240,
                              actual=Outcome(True, False, "A", None, None, None, rng.choice(tails).tail)))
    flights.sort(key=lambda f: (f.sched_dep, f.id))
    ids = {f.id: n for n, f in enumerate(flights)}
    flights = [replace(f, id=ids[f.id]) for f in flights]
    tails = [replace(t, legs=tuple(ids[i] for i in t.legs), busy=tuple(replace(b, leg=ids[b.leg]) for b in t.busy),
                     repositions=tuple(replace(m, leg=ids[m.leg]) for m in t.repositions)) for t in tails]
    task = day(flights, tails, segments=(segment("g", 0, 1000, 150, 400, (180, 200, 240, 300, 330)),),
               stops=(stop(250, 290),) if rng.random() < 0.5 else ())
    tails = [replace(t, connections=tuple(Connection(a, b, rng.randrange(0, 2 * TURN))
                                          for a, b in zip(t.legs, t.legs[1:], strict=False)
                                          if task.flight(a).dest == task.flight(b).orig and rng.random() < 0.6))
             for t in tails]
    return replace(task, tails=tuple(tails), needs=network.needs(tails, flights)[0])


def random_day(rng: random.Random, n_tails: int = 3) -> Task:
    """A small random day at JFK: chains of two to four legs whose breaks are Moves, delays, a GDP and a stop, and
    some turns JetBlue flew faster than the rule."""
    stations = [c for c in PLACES if c != "JFK"]
    flights, tails = [], []
    for k in range(n_tails):
        here, t, legs, moves = rng.choice(list(PLACES)), rng.randrange(0, 120), [], []
        start = here
        for _ in range(rng.randrange(2, 5)):
            if legs and rng.random() < 0.4:
                there = rng.choice([c for c in PLACES if c != here])
                t += rng.randrange(-40, 60)
                moves.append(Move(legs[-1], here, there, flights[-1].sched_arr, t - TURN))
                here = there
            dest = "JFK" if here != "JFK" and rng.random() < 0.6 else rng.choice([c for c in stations if c != here])
            block = rng.randrange(50, 150)
            i = len(flights)
            flights.append(flight(i, here, dest, t, block + rng.choice([0, 0, 20]), pax=rng.randrange(50, 151),
                                  lb=rng.choice([0, 0, 30, 90, 200]), deadline=t + rng.randrange(150, 300),
                                  sched_block=block))
            legs.append(i)
            here, t = dest, t + block + TURN + rng.randrange(0, 90)
        tails.append(tail(f"N{k}", start, ready=0, end=rng.choice([here, start]), repositions=tuple(moves),
                          legs=tuple(legs)))
    flights.sort(key=lambda f: (f.sched_dep, f.id))
    ids = {f.id: n for n, f in enumerate(flights)}
    flights = [replace(f, id=ids[f.id]) for f in flights]
    tails = [replace(t, legs=tuple(ids[i] for i in t.legs),
                     repositions=tuple(replace(m, leg=ids[m.leg]) for m in t.repositions)) for t in tails]
    task = day(flights, tails, segments=(segment("g", 0, 1000, 150, 400, (180, 200, 240, 300, 330)),),
               stops=(stop(250, 290),) if rng.random() < 0.5 else ())
    return replace(task, tails=tuple(replace(t, connections=tuple(
        Connection(a, b, rng.randrange(0, 2 * TURN)) for a, b in zip(t.legs, t.legs[1:], strict=False)
        if task.flight(a).dest == task.flight(b).orig and rng.random() < 0.6)) for t in task.tails))
