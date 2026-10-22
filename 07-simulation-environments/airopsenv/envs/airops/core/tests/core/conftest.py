import json
from pathlib import Path

import pytest

from airops_core.core.model import Task, load_pack

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tasks.jsonl"
AIRPORTS = {"JFK": (40.639928, -73.778692, "America/New_York"), "BOS": (42.362944, -71.006389, "America/New_York"),
            "DCA": (38.8521, -77.037697, "America/New_York"), "MCO": (28.429394, -81.308993, "America/New_York")}


@pytest.fixture(scope="session")
def pack() -> list[Task]:
    return load_pack(FIXTURES)


@pytest.fixture(scope="session")
def fixtures_path() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def lines() -> list[str]:
    return FIXTURES.read_text().splitlines()


@pytest.fixture
def make_task():
    return build


def build(flights: list[dict], tails: list[dict], caps: dict | None = None, nominal: dict | None = None,
          start: int = 600, news: list[dict] | None = None, turn: int = 30) -> Task:
    """A small task: flights as (orig, dest, sched_dep, block) plus overrides, tails as (tail, pos, ready, end_req)."""
    fl = []
    for i, f in enumerate(flights):
        d = {"id": i, "flight": f"B6 {100 + i}", "orig": f["orig"], "dest": f["dest"], "sched_dep": f["dep"],
             "sched_arr": f["dep"] + f.get("block", 60), "block": f.get("block", 60), "pax": f.get("pax", 100),
             "family": f.get("family", "A32X"), "lb": f.get("lb", 0), "dmax": f.get("dmax", 360),
             "actual": {"cancelled": 0, "cancel_code": "", "dep": f["dep"], "arr": f["dep"] + f.get("block", 60),
                        "end": f["dest"], "tail": "N1"}}
        fl.append(d)
    tl = [{"tail": t["tail"], "family": t.get("family", "A32X"), "type": "A320-232", "seats": 162, "pos": t["pos"],
           "ready": t.get("ready", 0), "end_req": t.get("end_req", t["pos"]), "actual_end": t["pos"]} for t in tails]
    airports = {c: {"name": c, "lat": la, "lon": lo, "tz": tz} for c, (la, lo, tz) in AIRPORTS.items()}
    return Task.from_dict(json.loads(json.dumps({
        "task_id": "t", "split": "eval", "carrier": "B6", "hub": "JFK", "date": "2023-06-26", "tz": "America/New_York",
        "window": [start + 60, start + 420], "start": start, "turn": {"A32X": turn, "E190": turn}, "turn_all": turn,
        "dmax": 360, "airports": airports, "tails": tl, "flights": fl, "caps": caps or {}, "nominal": nominal or {},
        "news": news or [], "frozen": [], "reference": {}, "sources": {}})))
