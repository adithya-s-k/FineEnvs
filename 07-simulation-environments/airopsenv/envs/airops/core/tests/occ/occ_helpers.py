"""A small storm day for the ops control center by hand, as a pack world, and a driver that plays it turn by turn."""

import gzip
import importlib.util
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import uvicorn
from day_helpers import day, flight, tail
from program_helpers import adv

from airops_core.occ.day import OccDay
from airops_core.occ.pack import World
from airops_core.programs import crewplan, crews, crewsim, occ, sim
from airops_core.programs.crews import Unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("airops_desk", ROOT / "agentenv" / "agents" / "airops-desk" / "agent.py")
desk_agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(desk_agent)
ROLE = {"ops": "ops", "crew": "crew", "one": "both"}


def unit(uid: str, kind: str, station: str, report: int, trip: tuple = ()) -> Unit:
    return Unit(uid, kind, "A32X", crews.size(kind, "A32X"), "JFK", station, station, report + crews.BRIEF, report,
                report // 60 % 24, at=station, trip=trip)


def crew(n: int, trip: tuple, station: str = "JFK", report: int = 540) -> list[Unit]:
    return [unit(f"P{n}", "pilot", station, report, trip), unit(f"C{n}", "cabin", station, report, trip)]


def storm() -> World:
    """JFK-BOS f0 held 600 minutes by its own delay and BOS-JFK f1, a round trip on one aircraft; PWM-DCA f2, its crew
    stranded at BUF (the crew system asks for it to be cancelled); JFK-MCO f3, its crew reporting at 10:00; a stop
    advisory that binds no one at the start and another at 01:30."""
    flights = [flight(0, "JFK", "BOS", 300, 60, lb=600, deadline=1500),
               flight(1, "BOS", "JFK", 420, 60, deadline=1500),
               flight(2, "PWM", "DCA", 600, 60, deadline=840), flight(3, "JFK", "MCO", 500, 150)]
    advisories = (adv("adv-a", "gs", -10, 0, 120, only="CAT"), adv("adv-b", "gs", 90, 90, 200, only="DAL"))
    task = day(flights, [tail("N1", "JFK", legs=(0, 1)), tail("N2", "PWM", legs=(2,)), tail("N3", "JFK", legs=(3,))],
               advisories=advisories)
    units = tuple(crew(1, (0, 1), report=840) + crew(2, (2,), "BUF") + crew(3, (3,), report=600))
    world = sim.truth(task)
    b0 = crewsim.execute(task, {}, units, world, desk=True)
    idle = crewsim.execute(task, {}, units, world, desk=True, drop=False)
    best = crewplan.worth(b0, False) * 0.9
    refs = {"best": round(best, 1), "bound": round(best * 0.95, 1), "gap_pct": 5.0,
            "B0": {"notice": round(crewplan.worth(b0, False), 1), "flat": round(crewplan.worth(b0, True), 1)},
            "idle": round(crewplan.worth(idle, False), 1)}
    return World(task, units, task.airports, ((0, 1),), refs, {}, lambda s, m: m)


def case(w: World) -> occ.Case:
    return occ.Case(w.task, w.units, w.local, w.airports, w.trips, sim.truth(w.task))


def play(w: World, mode: str = "joint", policies: dict | None = None, triggers: bool = True) -> OccDay:
    """The day played by `policies` (desk: function of the day, the desk and its page), every watch's advisories
    delivered by the trigger, every desk resigning when it is over."""
    d = OccDay(w, mode)
    while not d.done:
        desk = d.turn["desk"]
        if triggers:
            for n in list(d.pending):
                d.notice(n, "trigger")
        page = d.call(ROLE[desk], "wait_turn", {})
        (policies or {}).get(desk, lambda *_: None)(d, desk, page["page"])
        d.call(ROLE[desk], "end_turn", {"note": None, "stand_down": False})
    for desk in d.desks:
        d.call(ROLE[desk], "resign", {"reason": "done"})
    return d


def scripted(*desks: str) -> dict:
    """agentenv/agents/airops-desk's scripts as policies."""
    def policy(desk):
        script = desk_agent.Script(desk)

        def act(d, _, page):
            async def call(name, args):
                return json.loads(json.dumps(d.call(ROLE[desk], name, args)))
            done(script.turn(page, call))
        return act
    return {d: policy(d) for d in desks}


def done(coro) -> None:
    """A coroutine that never waits on anything, run to its end without an event loop."""
    try:
        while True:
            coro.send(None)
    except StopIteration:
        pass


def write_pack(path: Path, w: World) -> Path:
    """The world as a one-date pack folder."""
    (path / "dates").mkdir(parents=True)
    body = {"task": w.task.to_dict(), "units": [u.to_dict() for u in w.units],
            "airports": {c: a.to_dict() for c, a in w.airports.items()}, "trips": [list(p) for p in w.trips],
            "refs": w.refs, "best_plan": {str(k): v for k, v in w.best_plan.items()}}
    (path / "dates" / f"{w.task.date}.json.gz").write_bytes(gzip.compress(json.dumps(body).encode()))
    (path / "manifest.json").write_text(json.dumps({"tasks": {w.task.task_id: {"date": w.task.date, "split": "eval",
                                                                                 "best_from": "run"}}}))
    return path


KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "anyOf", "description", "const",
            "$defs", "$ref"}


def strict(schema: dict, where: str = "") -> list[str]:
    """Where a schema breaks OpenAI's strict mode: an object must close its properties and require every one; an
    optional field is a nullable one; no defaults, and no keyword strict mode doesn't take."""
    problems = [f"{where}: {k} is not supported" for k in sorted(set(schema) - KEYWORDS)]
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            problems.append(f"{where}: additionalProperties is not false")
        if set(schema.get("required", [])) != set(props):
            problems.append(f"{where}: required {schema.get('required')} is not every property")
        for name, sub in props.items():
            problems += strict(sub, f"{where}.{name}")
    for sub in schema.get("anyOf", []):
        problems += strict(sub, where)
    if "items" in schema:
        problems += strict(schema["items"], f"{where}[]")
    for sub in schema.get("$defs", {}).values():
        problems += strict(sub, f"{where}$defs")
    return problems


@contextmanager
def serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join()
