"""tools/build_wind.py and tools/wind_weather.py: the pins, the fetch and its resume, the month states, the weather
derivation on invented readings and forecasts, the output check; and, once built, the wind weeks byte for byte the ones
the published wind runs were played on (agentenv-portsim v0.4.5). tools/build_lines.py's pins and output check, and
once built the lines weeks the published lines runs were played on (agentenv-portsim 83eca5c). No network, no solve,
no readings."""

import gzip
import hashlib
import io
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from lines_fixtures import BUILT as LINES_BUILT
from wind_fixtures import REAL_BUSTS, REAL_TASKS, REAL_WEEKS, real

from port_week_core import packs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import build_lines  # noqa: E402
import build_wind  # noqa: E402
import wind_weather  # noqa: E402

PUBLISHED = {"dock-v1-wind/tasks.jsonl": "4362816b3657373063afab2f06c97cf76f941cb4e5b2b593b0a754e4ba9da0c9",
             "wind/references.jsonl": "0bb7929eecd5f0e70c291f0596b6a5df4c5c5f81da8209e79acc19271c7e0e13"}
WEEK, NEXT = "2024-W13", "2024-W14"
HOUR = timedelta(hours=1)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access in a test")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(wind_weather.http.client, "HTTPSConnection", refuse)


def pins() -> dict:
    return json.loads(wind_weather.PINS.read_text())


def test_the_constants_are_the_published_weeks_and_deal():
    assert list(wind_weather.WEEKS) == sorted(REAL_WEEKS)
    assert [w for w, kind in wind_weather.WEEKS.items() if kind == "bust"] == list(REAL_BUSTS)
    assert wind_weather.BUST_WEEKS == REAL_BUSTS
    assert [f"{s}-{w}" for s, w in build_wind.DEAL] == REAL_TASKS
    assert build_wind.DEAL_STATS["dealt"] == len(build_wind.DEAL) == 15


def test_the_pins_cover_every_month_and_the_published_outputs():
    p = pins()
    assert list(p["y7"]["months"]) == wind_weather.months()
    assert all(set(m) == {"rows", "sha256", "values_sha256"} for m in p["y7"]["months"].values())
    assert {rel: p["outputs"][rel]["sha256"] for rel in PUBLISHED} == PUBLISHED
    for rel in PUBLISHED:
        assert list(p["outputs"][rel]["lines"]) == REAL_TASKS
    assert list(p["outputs"]["wind/weather.jsonl"]["lines"]) == [f"e{i:02d}" for i in range(len(REAL_WEEKS))]
    assert set(build_wind.OUTPUTS.values()) | set(build_lines.OUTPUTS.values()) == set(p["outputs"])


def test_the_ecmwf_points_are_the_pinned_ones_and_cover_the_plan():
    body = wind_weather.POINTS.read_bytes()
    pin = pins()["ecmwf"]
    assert hashlib.sha256(body).hexdigest() == pin["sha256"]
    assert hashlib.sha256(gzip.decompress(body)).hexdigest() == pin["body_sha256"]
    runs = wind_weather.read_points(wind_weather.POINTS)
    work = wind_weather.plan(wind_weather.WEEKS)
    assert set(runs) == set(work)
    assert all(set(map(int, runs[r]["steps"])) == set(steps) for r, steps in work.items())
    assert sum(len(s) for r in runs.values() for s in r["steps"].values()) == pin["messages"]


def row(t: datetime, variable: str, value: float, state: str | None = "V") -> dict:
    r = {"id": f"Y7{variable}{t:%d%m%y%H%M}", "codi_estacio": "Y7", "codi_variable": variable,
         "data_lectura": f"{t:%Y-%m-%dT%H:%M:%S}.000", "valor_lectura": f"{value:.1f}", "codi_base": "SH"}
    return r | ({"codi_estat": state} if state else {})


def month_rows(month: str, gale: bool = True) -> list[dict]:
    """Invented Y7 readings for a month: a slow swell of wind with a gale on the 25th-26th of March and a short gust."""
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    end = (start + timedelta(days=32)).replace(day=1)
    out, t = [], start
    while t < end:
        k = (t - start) / HOUR
        mean = 6 + 4 * math.sin(k / 9) + (11 if gale and t.month == 3 and t.day in (25, 26) else 0)
        out += [row(t, "30", mean, "V" if t.day % 3 else None), row(t, "50", mean * 1.6, "V" if t.day % 3 else None)]
        t += HOUR / 2
    return out


class Response(io.BytesIO):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Last-Modified": "Thu, 08 Oct 2026 15:46:13 GMT"}


def portal(served: dict[str, list[dict]], calls: list[str], down: set[str] = frozenset()):
    def urlopen(url: str, timeout: int) -> Response:
        month = next(m for m in served if f"'{m}-01T00:00:00'" in urllib.parse.unquote_plus(url))
        calls.append(month)
        if month in down:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return Response(json.dumps(served[month]).encode())
    return urlopen


@pytest.fixture
def two_months(monkeypatch):
    served = {"2024-03": month_rows("2024-03"), "2024-04": month_rows("2024-04")}
    monkeypatch.setattr(wind_weather, "months", lambda: list(served))
    monkeypatch.setattr(wind_weather, "backoff", lambda attempt: None)
    return served


def test_the_fetch_resumes_with_only_the_missing_months(tmp_path, monkeypatch, two_months):
    calls: list[str] = []
    monkeypatch.setattr(wind_weather.urllib.request, "urlopen", portal(two_months, calls, down={"2024-04"}))
    metas, failures = wind_weather.fetch_y7(tmp_path)
    assert list(metas) == ["2024-03"] and failures == ["Y7 2024-04: HTTP 404"]
    assert metas["2024-03"]["sha256"] == wind_weather.digest(two_months["2024-03"])
    assert metas["2024-03"]["last_modified"] == "2026-10-08T15:46:13Z"
    calls.clear()
    monkeypatch.setattr(wind_weather.urllib.request, "urlopen", portal(two_months, calls))
    metas, failures = wind_weather.fetch_y7(tmp_path)
    assert calls == ["2024-04"] and failures == [] and list(metas) == ["2024-03", "2024-04"]
    assert json.loads((tmp_path / "2024-04.json").read_bytes()) == two_months["2024-04"]


def test_a_throttled_month_is_retried(tmp_path, monkeypatch, two_months):
    calls: list[str] = []
    serve = portal(two_months, calls)

    def throttled(url: str, timeout: int) -> Response:
        if len(calls) < 3:
            calls.append("throttled")
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return serve(url, timeout)
    monkeypatch.setattr(wind_weather, "months", lambda: ["2024-03"])
    monkeypatch.setattr(wind_weather.urllib.request, "urlopen", throttled)
    metas, failures = wind_weather.fetch_y7(tmp_path)
    assert calls == ["throttled"] * 3 + ["2024-03"] and failures == [] and list(metas) == ["2024-03"]


def test_each_month_is_classified_against_its_pin(tmp_path, monkeypatch, two_months):
    monkeypatch.setattr(wind_weather.urllib.request, "urlopen", portal(two_months, []))
    wind_weather.fetch_y7(tmp_path)
    rows = two_months["2024-03"]
    validated = [r | {"codi_estat": "V"} for r in rows]
    corrected = [r | {"valor_lectura": "99.9"} if i == 0 else r for i, r in enumerate(two_months["2024-04"])]
    assert wind_weather.values_digest(validated) == wind_weather.values_digest(rows)
    assert wind_weather.digest(validated) != wind_weather.digest(rows)

    def pin(rows: list[dict]) -> dict:
        return {"rows": len(rows), "sha256": wind_weather.digest(rows),
                "values_sha256": wind_weather.values_digest(rows)}
    months = {"2024-03": pin(rows), "2024-04": pin(two_months["2024-04"]), "2024-05": pin(rows)}
    assert build_wind.classify(tmp_path, months) == {"2024-03": "exact", "2024-04": "exact", "2024-05": "missing"}
    months |= {"2024-03": pin(validated), "2024-04": pin(corrected)}
    assert build_wind.classify(tmp_path, months) == {"2024-03": "values-match", "2024-04": "values-changed",
                                                     "2024-05": "missing"}


def test_the_fetch_command_fails_on_a_missing_month(tmp_path, monkeypatch, two_months, capsys):
    monkeypatch.setattr(wind_weather.urllib.request, "urlopen", portal(two_months, [], down={"2024-04"}))
    pinned = {m: {"rows": len(r), "sha256": wind_weather.digest(r), "values_sha256": wind_weather.values_digest(r)}
              for m, r in two_months.items()}
    monkeypatch.setattr(build_wind, "pins", lambda: {"y7": {"months": pinned}})
    args = build_wind.argparse.Namespace(cache=tmp_path)
    assert build_wind.fetch(args) == 1
    assert "1 exact" in capsys.readouterr().out
    monkeypatch.setattr(wind_weather.urllib.request, "urlopen", portal(two_months, []))
    assert build_wind.fetch(args) == 0
    assert "2 exact, 0 values-match, 0 values-changed, 0 missing" in capsys.readouterr().out


def invented_points(weeks: list[str]) -> dict:
    """Invented forecasts for every planned run of the weeks: each step's components a smooth function of its valid
    time, published 7 hours after the run starts, so every run is usable."""
    runs = {}
    for week in weeks:
        for init in wind_weather.week_runs(week):
            steps = {}
            for step in wind_weather.WEEK_STEPS:
                k = (init + step * HOUR - wind_weather.EPOCH) / HOUR
                u, v = 3 + 2 * math.sin(k / 7) + step / 40, 1 + math.cos(k / 11)
                modified = wind_weather.iso(init + 7 * HOUR)
                steps[str(step)] = {"10u": {"value": u, "last_modified": modified},
                                    "10v": {"value": v, "last_modified": modified}}
            runs[(wind_weather.week_grid(init), init)] = {"steps": steps}
    return runs


def derive(tmp_path: Path, gale: bool = True) -> tuple[bytes, list[dict]]:
    directory = tmp_path / ("gale" if gale else "calm")
    directory.mkdir()
    for month in ("2024-03", "2024-04"):
        (directory / f"{month}.json").write_text(json.dumps(month_rows(month, gale)))
    weeks = {WEEK: "storm", NEXT: "bust"}
    fits = {2024: ("0p25", wind_weather.monday(WEEK), wind_weather.monday(NEXT) + 6 * 24 * HOUR)}
    records, _ = wind_weather.weather(wind_weather.readings(directory), invented_points(list(weeks)), weeks, fits)
    return wind_weather.weather_body(records), records


def test_the_weather_derivation_is_deterministic(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    body, records = derive(tmp_path / "a")
    again, _ = derive(tmp_path / "b")
    assert body == again
    assert [(r["id"], r["iso_week"], r["kind"]) for r in records] == [("e00", WEEK, "storm"), ("e01", NEXT, "bust")]
    assert records[0]["windows"] and all(0 <= w["unvalidated"] <= 1 for w in records[0]["windows"])
    assert all(r["runs"] for r in records)


def test_the_ids_follow_the_fixed_weeks_not_the_storm_rule(tmp_path):
    _, gale = derive(tmp_path)
    _, calm = derive(tmp_path, gale=False)
    assert wind_weather.candidate_weeks(wind_weather.readings(tmp_path / "calm"))["storm"] == []
    assert [(r["id"], r["iso_week"]) for r in calm] == [(r["id"], r["iso_week"]) for r in gale]
    assert calm[0]["windows"] == [] and gale[0]["windows"]


def test_the_points_file_reads_as_the_fetch_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(wind_weather, "FITS", [])
    runs = invented_points([WEEK])
    first = next(iter(runs))
    runs[first]["steps"]["6"] = None
    runs[first]["steps"]["9"]["10v"] = None
    for (grid, init), run in runs.items():
        wind_weather.write_json(wind_weather.run_path(tmp_path, grid, init), run)
    path = tmp_path / "points.jsonl.gz"
    path.write_bytes(gzip.compress(wind_weather.points_body(wind_weather.point_rows(tmp_path, [WEEK])), 9, mtime=0))
    points = wind_weather.read_points(path)
    assert set(points) == set(runs)
    assert points[first]["steps"]["6"] == {"10u": None, "10v": None} and points[first]["steps"]["9"]["10v"] is None
    assert all(points[r] == runs[r] for r in runs if r != first)
    for grid, init in runs:
        cached = wind_weather.cached_run(tmp_path, grid, init)
        assert wind_weather.usable(points[grid, init], init) == wind_weather.usable(cached, init) == (
            (grid, init) != first)


def test_the_points_command_is_deterministic_and_keeps_the_committed_file_on_a_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(wind_weather, "WEEKS", {WEEK: "storm"})
    monkeypatch.setattr(wind_weather, "FITS", [])
    monkeypatch.setattr(wind_weather, "PINS", tmp_path / "pins.json")
    monkeypatch.setattr(wind_weather, "POINTS", tmp_path / "points.jsonl.gz")
    wind_weather.PINS.write_text(json.dumps({"ecmwf": {"body_sha256": "0" * 64}}))
    for (grid, init), run in invented_points([WEEK]).items():
        wind_weather.write_json(wind_weather.run_path(tmp_path, grid, init), run)
    assert wind_weather.ecmwf_points(tmp_path, wind_weather.POINTS, fetch=False) == 1
    assert not wind_weather.POINTS.exists()
    assert wind_weather.ecmwf_points(tmp_path, tmp_path / "a.gz", fetch=False) == 1
    body = gzip.decompress((tmp_path / "a.gz").read_bytes())
    wind_weather.PINS.write_text(json.dumps({"ecmwf": {"body_sha256": hashlib.sha256(body).hexdigest()}}))
    assert wind_weather.ecmwf_points(tmp_path, tmp_path / "b.gz", fetch=False) == 0
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()
    assert body.count(b"\n") == 45 * 25 * 2
    wind_weather.run_path(tmp_path, *next(iter(invented_points([WEEK])))).unlink()
    assert wind_weather.ecmwf_points(tmp_path, tmp_path / "c.gz", fetch=False) == 1


def built(root: Path, outputs: dict[str, list[dict]]) -> dict:
    """Write invented outputs under root and return pins for them."""
    out = {}
    for rel, records in outputs.items():
        body = b"".join(json.dumps(r).encode() + b"\n" for r in records)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(body)
        out[rel] = {"sha256": hashlib.sha256(body).hexdigest(), "lines": build_wind.line_digests(body)}
    return out


@pytest.fixture
def invented_build(tmp_path, monkeypatch):
    monkeypatch.setenv("PORT_WEEK_TASKS_DIR", str(tmp_path))
    outputs = {"dock-v1-wind/tasks.jsonl": [{"task_id": "a-e00", "x": 1}, {"task_id": "b-e01", "x": 2}],
               "wind/references.jsonl": [{"task_id": "a-e00", "y": 1}, {"task_id": "b-e01", "y": 2}],
               "wind/weather.jsonl": [{"id": "e00", "z": 1}, {"id": "e01", "z": 2}]}
    pinned = built(tmp_path, outputs)
    (tmp_path / "dock-v1-wind/manifest.json").write_text(json.dumps({"name": "dock-v1-wind"}))
    monkeypatch.setattr(build_wind, "pins", lambda: {"outputs": pinned})
    return tmp_path, outputs


def test_verify_passes_identical_outputs_and_ignores_weather_drift(invented_build, capsys):
    root, outputs = invented_build
    args = build_wind.argparse.Namespace(accept_drift=False)
    assert build_wind.verify(args) == 0
    built(root, {"wind/weather.jsonl": [{"id": "e00", "z": 1}, {"id": "e01", "z": 3}]})
    assert build_wind.verify(args) == 0
    assert "wind/weather.jsonl: differs in e01 (informational)" in capsys.readouterr().out


def test_verify_lists_the_drifted_tasks_and_accepts_them_on_request(invented_build, capsys):
    root, outputs = invented_build
    built(root, {"wind/references.jsonl": [{"task_id": "a-e00", "y": 1}, {"task_id": "b-e01", "y": 9}]})
    assert build_wind.verify(build_wind.argparse.Namespace(accept_drift=False)) == 1
    assert "wind/references.jsonl: differs in b-e01" in capsys.readouterr().out
    assert build_wind.verify(build_wind.argparse.Namespace(accept_drift=True)) == 0
    drift = json.loads((root / "dock-v1-wind/manifest.json").read_text())["drift"]
    assert drift["references"] == ["b-e01"] and "tasks" not in drift


def test_verify_fails_when_nothing_is_built(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PORT_WEEK_TASKS_DIR", str(tmp_path))
    assert build_wind.verify(build_wind.argparse.Namespace(accept_drift=True)) == 1
    assert "not built" in capsys.readouterr().out


@real
@pytest.mark.parametrize("rel", list(PUBLISHED))
def test_the_built_file_matches_its_pin(rel):
    path = packs.tasks_root() / rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pins()["outputs"][rel]["sha256"]


def test_the_lines_pins_are_the_published_lines_weeks():
    p = pins()["outputs"]
    published = "d3fe06b74a8dff25b0611171d380a06b47b2bc367ba0f06e0360e33a3e09b654"
    assert p["dock-v1-lines/tasks.jsonl"]["sha256"] == published
    for rel in ("dock-v1-lines/tasks.jsonl", "lines/references.jsonl"):
        assert list(p[rel]["lines"]) == REAL_TASKS
    assert list(p["lines/references.jsonl"]["reproducible_lines"]) == REAL_TASKS


def test_only_the_untied_8_worker_summaries_are_left_out_of_the_reproducible_record():
    row = {"task_id": "t", "configs": [{"workers": w, "tiebreak": tb, "cost": 1} for w, tb in
                                       [(1, True), (1, False), (8, True), (8, False)]],
           "silent": {"reward": 0.2, "configs": [{"workers": 8, "tiebreak": False, "reward": 0.2}]}}
    def digest(r):
        return build_lines.reproducible_digests((json.dumps(r) + "\n").encode())["t"]
    other = json.loads(json.dumps(row))
    other["configs"][3]["cost"] = other["silent"]["configs"][0]["reward"] = 2
    assert digest(other) == digest(row)
    other["configs"][2]["cost"] = 2
    assert digest(other) != digest(row)


@pytest.mark.skipif(not LINES_BUILT, reason="lines weeks not built: run tools/build_lines.py")
def test_the_built_lines_weeks_match_their_pins():
    diffs, _ = build_lines.compare(packs.tasks_root(), pins()["outputs"])
    assert diffs == {"tasks": [], "manifest": [], "references": []}
