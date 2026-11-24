import json
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from airops_core.build import bts, fleet, pack

INPUTS = Path(__file__).resolve().parent / "fixtures" / "inputs"
ROOT = Path(__file__).resolve().parents[3]
FIELDS = ["task_id", "split", "carrier", "hub", "date", "tz", "window", "start", "turn", "turn_all", "dmax", "airports",
          "tails", "flights", "caps", "nominal", "news", "frozen", "reference", "sources"]


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> pd.DataFrame:
    """The fixture's B6 legs of the day with local minutes, read independently of the scope code."""
    df = bts.read_ontime(bts.ontime_path(INPUTS, "2023-06"), tmp_path_factory.mktemp("raw"))
    base = bts.epoch_minutes(pd.Timestamp("2023-06-26").tz_localize("America/New_York").tz_convert("UTC")
                             .to_pydatetime())
    df = df[(df.Reporting_Airline == "B6") & (df.FlightDate == "2023-06-26")]
    return df.assign(sd=df.sched_dep - base, sa=df.sched_arr - base, ad=df.act_dep - base)


def test_one_task_in_the_design_schema(task):
    assert list(task) == FIELDS
    assert task["task_id"] == "b6-jfk-20230626-{:02d}{:02d}".format(*divmod(task["window"][0], 60))
    assert (task["split"], task["carrier"], task["hub"], task["tz"]) == ("eval", "B6", "JFK", "America/New_York")
    w0, w1 = task["window"]
    assert w0 % 60 == 0 and w1 % 60 == 0 and 180 <= w1 - w0 <= 360 and task["start"] == w0 - 60
    assert task["reference"] == {} and task["dmax"] == 360


def test_scope_is_the_tails_touching_the_hub_in_the_window(task, raw):
    w0, w1 = task["window"]
    touching = set()
    for r in raw.itertuples():
        if (r.Origin == "JFK" and w0 <= r.sd < w1) or (r.Dest == "JFK" and w0 <= r.sa < w1):
            touching.add(r.Tail_Number)
    assert {t["tail"] for t in task["tails"]} == touching
    start = task["start"]
    for f in task["flights"]:
        a = f["actual"]
        assert a["tail"] in touching
        assert (a["dep"] if a["dep"] is not None else f["sched_dep"]) >= start
    assert all(leg["dep"] < start for leg in task["frozen"])
    expected = sum(1 for r in raw.itertuples() if r.Tail_Number in touching
                   and ((r.Cancelled == 0 and r.ad >= start) or (r.Cancelled == 1 and r.sd >= start)))
    assert len(task["flights"]) == expected
    assert [f["id"] for f in task["flights"]] == list(range(len(task["flights"])))
    assert [f["sched_dep"] for f in task["flights"]] == sorted(f["sched_dep"] for f in task["flights"])


def test_lb_pax_and_tails(task, raw):
    by_key = {(f"B6 {int(r.Flight_Number_Reporting_Airline)}", int(r.sd)): r for r in raw.itertuples()}
    lf = fleet.load_factors("B6", "2023-06", INPUTS)
    seats = {t["tail"]: t["seats"] for t in task["tails"]}
    for f in task["flights"]:
        r = by_key[f["flight"], f["sched_dep"]]
        if r.Cancelled == 0:
            late = 0 if pd.isna(r.LateAircraftDelay) else r.LateAircraftDelay
            assert f["lb"] == max(0, max(r.DepDelay, 0) - late)
            assert f["actual"]["dep"] == int(r.ad)
        else:
            assert f["actual"]["cancel_code"] == r.CancellationCode and f["actual"]["dep"] is None
        assert f["pax"] == fleet.pax(seats[f["actual"]["tail"]], lf, f["orig"], f["dest"])
        assert f["dmax"] >= 360
    typed = {t["tail"]: t for t in task["tails"]}
    assert typed["N729JB"]["type"] == "A320-232" and typed["N729JB"]["family"] == "A32X"
    assert all(t["pos"] in task["airports"] and t["end_req"] in task["airports"] for t in task["tails"])


def test_caps_never_below_what_the_scope_moved(task):
    moved: dict[str, dict[str, int]] = {}
    for f in task["flights"]:
        a = f["actual"]
        if a["dep"] is not None:
            key = f"{f['orig']}:dep"
            moved.setdefault(key, {}).setdefault(str(a["dep"] // 60), 0)
            moved[key][str(a["dep"] // 60)] += 1
    for key, hours in task["caps"].items():
        assert key.split(":")[0] in pack.CONSTRAINED
        assert list(hours) == [str(h) for h in range(task["start"] // 60, 24)] == list(task["nominal"][key])
        for h, n in moved.get(key, {}).items():
            if h in hours:
                assert hours[h] >= n


def test_news_is_ordered_and_scoped(task):
    news = task["news"]
    keys = [(n["at"], pack.KINDS.index(n["kind"]), n["id"]) for n in news]
    assert keys == sorted(keys) and len({n["id"] for n in news}) == len(news)
    assert {n["kind"] for n in news} == set(pack.KINDS)
    advisories = [n for n in news if n["kind"] == "advisory"]
    assert [n["data"]["advzy"] for n in advisories] == [46, 50]
    assert all("LGA" not in n["text"].split("\n")[0] for n in advisories)
    start = task["start"]
    assert sum(1 for n in news if n["kind"] == "taf" and n["at"] <= start) == 1
    assert sum(1 for n in news if n["kind"] == "outlook" and n["at"] <= start) == 1
    assert min(n["at"] for n in news if n["kind"] == "metar") >= start - pack.METAR_BEFORE
    speakers = {n["kind"]: n["speaker"] for n in news}
    assert speakers == {"advisory": "FAA Command Center", "taf": "NWS JFK TAF", "metar": "JFK METAR",
                        "outlook": "NWS New York forecast discussion"}


def test_sources_and_manifest(built, task):
    manifest = json.loads((built / "manifest.json").read_text())
    assert manifest["splits"] == {"eval": 1, "control": 0, "train": 0}
    assert manifest["inputs"]["bts/ontime_2023_06.zip"]["sha256"] == task["sources"]["bts_ontime"]
    assert manifest["inputs"]["atcscc/2023-06-26/JFK"]["count"] == 2
    assert task["sources"]["faa_registry"] == manifest["inputs"]["faa/ReleasableAircraft.zip"]["sha256"]
    assert set(task["sources"]["advisories"]) <= {"2023-06-25", "2023-06-26", "2023-06-27", "2023-06-28"}
    names = sorted(p.parent.name for p in (built / "sources").glob("*/SOURCE.md"))
    assert names == ["airportsdata", "bts-b43", "bts-ontime", "bts-t100", "faa-atcscc", "faa-registry", "iem"]


def test_a_rebuild_writes_the_same_bytes(built, tmp_path, run_build):
    again = tmp_path / "b" / "b6-v1"
    assert run_build(again, tmp_path / "derived") == 0
    first = sorted(p.relative_to(built) for p in built.rglob("*") if p.is_file())
    assert first == sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file())
    for rel in first:
        assert (built / rel).read_bytes() == (again / rel).read_bytes(), rel
    assert run_build(again, tmp_path / "derived") == 0
    assert (again / "tasks.jsonl").read_bytes() == (built / "tasks.jsonl").read_bytes()


def test_a_changed_input_fails_its_pin(built, tmp_path, run_build):
    inputs = tmp_path / "inputs"
    shutil.copytree(INPUTS, inputs)
    out = tmp_path / "pack" / "b6-v1"
    shutil.copytree(built, out)
    metar = inputs / "iem" / "metar" / "JFK_2023_06.csv"
    metar.write_text(metar.read_text().replace("A2976", "A2977"))
    assert run_build(out, tmp_path / "derived", inputs) == 1
    assert run_build(out, tmp_path / "derived", inputs, "--repin") == 0


@pytest.mark.network
def test_the_pinned_bts_months_are_still_published_as_pinned():
    manifest = json.loads((ROOT / "tasks" / "b6-v1" / "manifest.json").read_text())
    for month in manifest["months"]:
        pin = manifest["inputs"][str(bts.ontime_path(Path("."), month))]
        req = urllib.request.Request(bts.ontime_url(month), method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert int(resp.headers["Content-Length"]) == pin["bytes"]
