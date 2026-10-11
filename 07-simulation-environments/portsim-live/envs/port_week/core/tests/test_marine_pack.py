import csv
import hashlib
import io
import json
import os
import urllib.request
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import berth_core
import pytest
from berth_core import TaskPack, plan_from_list

from port_week_core import marine, packs, schedule, world
from port_week_core.marine import CONTAINER

PACK = marine.pack()
EVAL_DIR = next(r for r in packs.live_pack().roots if r.name == "dock-v1-eval")
V2 = [t for t in TaskPack(EVAL_DIR).tasks if schedule.ONE_WEEK.match(t.task_id)]
MANIFEST = json.loads((marine.pack_dir() / "manifest.json").read_text())
CUTS = {"tug_outage": "tugs", "pilot_shortage": "pilots"}


def test_the_pack_sits_in_the_tasks_root_or_under_port_week_tasks_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("PORT_WEEK_TASKS_DIR", raising=False)
    assert packs.TASKS_ROOT == Path(packs.__file__).resolve().parents[2] / "tasks"
    assert marine.pack_dir() == packs.TASKS_ROOT / "dock-v1-marine"
    (tmp_path / "dock-v1-marine").symlink_to(packs.TASKS_ROOT / "dock-v1-marine")
    monkeypatch.setenv("PORT_WEEK_TASKS_DIR", str(tmp_path))
    assert marine.pack_dir() == tmp_path / "dock-v1-marine"
    assert [t.task_id for t in marine.pack().tasks] == [t.task_id for t in PACK.tasks]
    monkeypatch.setenv("BERTH_TASKS_DIR", str(EVAL_DIR))
    assert marine.pack_dir() == tmp_path / "dock-v1-marine"


def test_the_manifest():
    body = (marine.pack_dir() / "tasks.jsonl").read_bytes()
    assert list(MANIFEST) == ["name", "split", "tasks", "tiers", "sha256", "weeks", "source", "quays", "raw", "marine",
                              "calibration", "note"]
    assert {key: MANIFEST[key] for key in ("name", "split", "tasks", "tiers", "weeks", "quays")} == {
        "name": "dock-v1-marine", "split": "eval", "tasks": 18, "tiers": {"busy": 9, "standard": 9},
        "weeks": [5, 6, 7, 10, 15, 16, 17, 35, 37], "quays": {"24B": 6, "36A": 12}}
    assert MANIFEST["sha256"] == hashlib.sha256(body).hexdigest()
    assert body == b"".join(json.dumps(t.to_dict(), ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
                            for t in PACK.tasks)
    assert MANIFEST["raw"] == {
        "repository": "https://github.com/alberto-santini/berth-allocation-problems",
        "file": "generator/raw/Barcelona_2024.csv", "commit": "8e726a47625da99eec72f911b16a0b76a78a8d88",
        "sha256": "123012c078e66a5978d55bcedd4203ef85c4282ceefc0e3d6b01ebdc94207d94", "licence": "CC BY-SA 4.0"}
    assert list(MANIFEST["marine"]) == ["pilot", "service", "pilots", "tugs", "tugs_by_length", "wind", "gales",
                                        "other_traffic", "hours", "tug_outage", "pilot_shortage"]
    for key, item in MANIFEST["marine"].items():
        assert list(item) == ["value", "basis"]
        assert key == "hours" or "https://" in item["basis"] or "ssumption" in item["basis"], key
    assert (MANIFEST["marine"]["pilots"]["value"], MANIFEST["marine"]["tugs"]["value"]) == (7, 8)
    assert list(MANIFEST["calibration"]) == ["hours", "movements", "over_pool_percent"]
    assert MANIFEST["calibration"]["hours"] == 8784
    assert "CC BY-SA 4.0" in MANIFEST["source"]


def test_the_weeks_are_dock_v1_evals_one_week_weeks_with_only_the_marine_fields_changed():
    assert [t.task_id for t in PACK.tasks] == [t.task_id for t in V2]
    for task, v2 in zip(PACK.tasks, V2, strict=True):
        assert task.rules == v2.rules | {"marine": task.rules["marine"]}
        assert task.disruptions[:-2] == v2.disruptions
        assert task.reference == v2.reference | {key: task.reference[key] for key in (
            "optimal_cost", "optimal_plan", "lower_bound", "proven_optimal")}
        assert task.reference["lower_bound"] == task.reference["optimal_cost"] and task.reference["proven_optimal"]
        assert replace(task, rules={}, disruptions=[], notices=[], reference={}) == replace(
            v2, rules={}, disruptions=[], notices=[], reference={})


def test_the_marine_rules():
    for task in PACK.tasks:
        rules = task.rules["marine"]
        assert list(rules) == ["pilots", "tugs", "hours", "movements"]
        assert (rules["pilots"], rules["tugs"], rules["hours"]) == (7, 8, 264)
        moves = rules["movements"]
        assert moves == sorted(moves) and all(0 <= h < 264 and length_m >= 45 for h, _, length_m in moves)
        for b in task.blocks:
            if b.kind == "alongside":
                assert any(h == b.end and kind == CONTAINER for h, kind, _ in moves)
    assert sum(b.kind == "alongside" for t in PACK.tasks for b in t.blocks) == 36


def demand(task) -> dict[str, Counter]:
    out = {"pilots": Counter(), "tugs": Counter()}
    for _, hour, tugs in marine.own_moves(task, berth_core.evaluate(task, plan_from_list(
            task.reference["optimal_plan"]))):
        out["pilots"][hour] += 1
        out["tugs"][hour] += tugs
    return out


def test_each_cut_lands_on_a_bulletin_a_day_ahead_over_the_v2_optimums_busiest_window():
    for task, v2 in zip(PACK.tasks, V2, strict=True):
        tug, pilot = task.disruptions[-2:]
        assert (tug["type"], tug["end"] - tug["start"], tug["tugs"]) == ("tug_outage", 24, 2)
        assert (pilot["type"], pilot["end"] - pilot["start"], pilot["pilots"]) == ("pilot_shortage", 12, 2)
        bulletins = [w.hour for w in schedule.schedule(v2)][1:]
        need = demand(v2)
        for e in (tug, pilot):
            pool, length = CUTS[e["type"]], e["end"] - e["start"]
            window = {b: sum(need[pool][h] for h in range(b + 24, b + 24 + length)) for b in bulletins}
            assert e["start"] - 24 == min(b for b in bulletins if window[b] == max(window.values()))
        watches = schedule.schedule(task)
        assert [w.hour for w in watches] == [w.hour for w in schedule.schedule(v2)]
        n = len(v2.disruptions)
        assert {x.event_id: x.hour for w in watches for x in w.notices if x.index >= n} == {
            f"tug_outage-{n}": tug["start"] - 24, f"pilot_shortage-{n + 1}": pilot["start"] - 24}


def test_the_notices():
    for task, v2 in zip(PACK.tasks, V2, strict=True):
        tug, pilot = task.disruptions[-2:]
        assert task.notices[-2:] == [
            f"2 of the port's 8 tugs are out of service from hour {tug['start']} to hour {tug['end']}: 6 tugs in that "
            "window.",
            f"2 of the port's 7 pilots on duty are unavailable from hour {pilot['start']} to hour {pilot['end']}: 5 "
            "pilots in that window."]
        for text, before, e in zip(task.notices[:-2], v2.notices, v2.disruptions, strict=True):
            assert text == (before + f" Ships that berth or leave from hour {e['start']} to hour {e['end']} take one "
                            "more tug." if e["type"] == "gale" else before)


def test_the_example_weeks_other_traffic_at_hour_54():
    task = PACK.get("dock-24B-w07x1-busy-0")
    assert [m for m in task.rules["marine"]["movements"] if m[0] == 54] == [
        [54, "Portacontenidors", 139.0], [54, "Portacontenidors", 148.0], [54, "Ro-Ro", 238.0]]


def test_the_marine_pack_never_loads_with_dock_v1_eval():
    with pytest.raises(ValueError, match="more than one pack"):
        TaskPack([marine.pack_dir(), EVAL_DIR])


def test_the_known_week_carries_the_rules_whole_and_only_the_announced_cuts():
    task = PACK.get("dock-24B-w07x1-busy-0")
    view = world.known(task, [])
    assert view.rules["marine"] == task.rules["marine"] and view.rules["no_moves"] == []
    assert not any(e["type"] in CUTS for e in view.disruptions)
    view = world.known(task, ["tug_outage-8"])
    assert [e["type"] for e in view.disruptions] == ["tug_outage"]


def raw() -> bytes:
    """RAW from the cache marine_references.py fills, else downloaded."""
    pin = MANIFEST["raw"]
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "portsim-live" / "raw"
    path = cache / pin["commit"][:7] / Path(pin["file"]).name
    if path.is_file():
        body = path.read_bytes()
    else:
        url = pin["repository"].replace("github.com", "raw.githubusercontent.com") + f"/{pin['commit']}/{pin['file']}"
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read()
    assert hashlib.sha256(body).hexdigest() == pin["sha256"]
    return body


@pytest.mark.network
def test_the_other_traffic_is_raws_calls_away_from_the_quay_counted_once_at_each_berth():
    rows = list(csv.DictReader(io.StringIO(raw().decode())))
    calls: dict[str, dict[tuple, dict]] = {}
    for r in rows:
        calls.setdefault(r["ESCALANUM"].split("-")[0], {}).setdefault(
            (r["ESCALANUM"], r["MOLLCODI"], r["ETAUTC"], r["ETDUTC"]), r)
    assert len(rows) - sum(map(len, calls.values())) == 116

    def when(text):
        return datetime.fromisoformat(text).replace(tzinfo=UTC)

    def length(r):
        return float(r["ESLORA_METRES"].replace(",", "."))

    left_out = Counter()
    for task in PACK.tasks:
        start = datetime.fromisoformat(task.week_start_utc.removesuffix("Z")).replace(tzinfo=UTC)
        counted = []
        for call in calls.values():
            call = sorted(call.values(), key=lambda r: r["ETAUTC"])
            at_quay = any(r["MOLLCODI"] == task.quay for r in call)
            for i, r in enumerate(call):
                leaves = i + 1 == len(call) or call[i + 1]["MOLLCODI"] == "90A"
                for text, shift in ((r["ETAUTC"], False), (r["ETDUTC"], not leaves)):
                    move = [round((when(text) - start).total_seconds() / 3600), r["VAIXELLTIPUS"], length(r)]
                    if 0 <= move[0] < 264 and move[2] >= 45:
                        why = "quay" if at_quay else "anchorage" if r["MOLLCODI"] == "90A" else "shift" if shift else ""
                        if why:
                            left_out[why] += 1
                        else:
                            counted.append(move)
        for b in task.blocks:
            if b.kind == "alongside":
                [r] = [r for r in rows if r["MOLLCODI"] == task.quay and r["VAIXELLNOM"] == b.label
                       and when(r["ETAUTC"]) < start <= when(r["ETDUTC"])]
                counted.append([b.end, CONTAINER, length(r)])
        assert task.rules["marine"]["movements"] == sorted(counted), task.task_id
    assert all(left_out[why] for why in ("quay", "anchorage", "shift"))
