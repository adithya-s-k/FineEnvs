"""The task: one hub window on a real day, as stored in data/<pack>/tasks.jsonl (see DESIGN.md)."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


@dataclass(frozen=True)
class Airport:
    code: str
    name: str
    lat: float
    lon: float
    tz: str

    @classmethod
    def from_dict(cls, code: str, d: Mapping) -> "Airport":
        return cls(code, d["name"], d["lat"], d["lon"], d["tz"])

    def to_dict(self) -> dict:
        return {"name": self.name, "lat": self.lat, "lon": self.lon, "tz": self.tz}


@dataclass(frozen=True)
class Tail:
    tail: str
    family: str
    type: str
    seats: int
    pos: str
    ready: int
    end_req: str
    actual_end: str

    @classmethod
    def from_dict(cls, d: Mapping) -> "Tail":
        return cls(d["tail"], d["family"], d["type"], d["seats"], d["pos"], d["ready"], d["end_req"], d["actual_end"])

    def to_dict(self) -> dict:
        return {"tail": self.tail, "family": self.family, "type": self.type, "seats": self.seats, "pos": self.pos,
                "ready": self.ready, "end_req": self.end_req, "actual_end": self.actual_end}


@dataclass(frozen=True)
class Actual:
    cancelled: bool
    cancel_code: str
    dep: int | None
    arr: int | None
    end: str | None
    tail: str | None

    @classmethod
    def from_dict(cls, d: Mapping) -> "Actual":
        return cls(bool(d["cancelled"]), d["cancel_code"], d["dep"], d["arr"], d["end"], d["tail"])

    def to_dict(self) -> dict:
        return {"cancelled": int(self.cancelled), "cancel_code": self.cancel_code, "dep": self.dep, "arr": self.arr,
                "end": self.end, "tail": self.tail}


@dataclass(frozen=True)
class Flight:
    id: int
    flight: str
    orig: str
    dest: str
    sched_dep: int
    sched_arr: int
    block: int
    pax: int
    family: str
    lb: int
    dmax: int
    actual: Actual

    @classmethod
    def from_dict(cls, d: Mapping) -> "Flight":
        return cls(d["id"], d["flight"], d["orig"], d["dest"], d["sched_dep"], d["sched_arr"], d["block"], d["pax"],
                   d["family"], d["lb"], d["dmax"], Actual.from_dict(d["actual"]))

    def to_dict(self) -> dict:
        return {"id": self.id, "flight": self.flight, "orig": self.orig, "dest": self.dest,
                "sched_dep": self.sched_dep, "sched_arr": self.sched_arr, "block": self.block, "pax": self.pax,
                "family": self.family, "lb": self.lb, "dmax": self.dmax, "actual": self.actual.to_dict()}


@dataclass(frozen=True)
class News:
    id: str
    at: int
    kind: str
    speaker: str
    text: str
    data: dict

    @classmethod
    def from_dict(cls, d: Mapping) -> "News":
        return cls(d["id"], d["at"], d["kind"], d["speaker"], d["text"], d["data"])

    def to_dict(self) -> dict:
        return {"id": self.id, "at": self.at, "kind": self.kind, "speaker": self.speaker, "text": self.text,
                "data": self.data}


Caps = dict[str, dict[int, int]]


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    carrier: str
    hub: str
    date: str
    tz: str
    window: tuple[int, int]
    start: int
    turn: dict[str, int]
    turn_all: int
    dmax: int
    airports: dict[str, Airport]
    tails: tuple[Tail, ...]
    flights: tuple[Flight, ...]
    caps: Caps
    nominal: Caps
    news: tuple[News, ...]
    frozen: tuple[dict, ...]
    reference: dict
    sources: dict

    @classmethod
    def from_dict(cls, d: Mapping) -> "Task":
        return cls(d["task_id"], d["split"], d["carrier"], d["hub"], d["date"], d["tz"], tuple(d["window"]),
                   d["start"], dict(d["turn"]), d["turn_all"], d["dmax"],
                   {c: Airport.from_dict(c, a) for c, a in d["airports"].items()},
                   tuple(Tail.from_dict(t) for t in d["tails"]), tuple(Flight.from_dict(f) for f in d["flights"]),
                   _caps(d["caps"]), _caps(d["nominal"]), tuple(News.from_dict(n) for n in d["news"]),
                   tuple(d["frozen"]), dict(d["reference"]), dict(d["sources"]))

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "split": self.split, "carrier": self.carrier, "hub": self.hub,
                "date": self.date, "tz": self.tz, "window": list(self.window), "start": self.start,
                "turn": dict(self.turn), "turn_all": self.turn_all, "dmax": self.dmax,
                "airports": {c: a.to_dict() for c, a in self.airports.items()},
                "tails": [t.to_dict() for t in self.tails], "flights": [f.to_dict() for f in self.flights],
                "caps": _caps_out(self.caps), "nominal": _caps_out(self.nominal),
                "news": [n.to_dict() for n in self.news], "frozen": [dict(f) for f in self.frozen],
                "reference": self.reference, "sources": self.sources}

    @cached_property
    def by_id(self) -> dict[int, Flight]:
        return {f.id: f for f in self.flights}

    def flight(self, flight_id: int) -> Flight:
        return self.by_id[flight_id]

    def turn_for(self, family: str) -> int:
        return self.turn.get(family, self.turn_all)

    def earliest(self, f: Flight, lb: int) -> int:
        """Nothing in scope leaves before the first watch, whatever its delay."""
        return max(f.sched_dep + lb, self.start)

    def deadline(self, f: Flight, lb: int) -> int:
        """Past it the flight is cancelled; a flight whose earliest departure is already later cannot fly today."""
        return f.sched_dep + f.dmax

    @property
    def lb(self) -> dict[int, int]:
        return {f.id: f.lb for f in self.flights}


def load_pack(path: str | Path) -> list[Task]:
    """The tasks of a pack directory (its tasks.jsonl) or of a .jsonl file."""
    p = Path(path)
    p = p / "tasks.jsonl" if p.is_dir() else p
    return [Task.from_dict(json.loads(line)) for line in p.read_text().splitlines() if line.strip()]


def hhmm(minute: int) -> str:
    day, m = divmod(minute, 1440)
    return f"{m // 60:02d}:{m % 60:02d}" + (f"{day:+d}" if day else "")


def parse_hhmm(text: str) -> int:
    """Inverse of hhmm: "14:05" is 845, "01:10+1" is 1510."""
    clock, plus, day = text.strip().partition("+")
    h, colon, m = clock.partition(":")
    if not (colon and h.isdigit() and m.isdigit() and len(m) == 2 and int(h) < 24 and int(m) < 60
            and (day.isdigit() or not plus)):
        raise ValueError(f"not a time: {text!r}")
    return int(h) * 60 + int(m) + 1440 * int(day or 0)


def _caps(d: Mapping) -> Caps:
    return {k: {int(h): v for h, v in hours.items()} for k, hours in d.items()}


def _caps_out(caps: Caps) -> dict:
    return {k: {str(h): v for h, v in hours.items()} for k, hours in caps.items()}
