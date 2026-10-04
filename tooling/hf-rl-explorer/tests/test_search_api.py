"""/api/search and /api/search/tasks (app/search_api.py) over a catalog snapshot (no network: the listing is fixture rows).

What is checked: declared framework taxonomy independent of editorial collections; metric-only trending with
separate featured picks; hidden environments left out of everything at query time; every answer (order, total, every
facet's counts) equal to a plain-Python port of the page's own filtering over the same rows, for hundreds of queries
(a parity test), through a published snapshot and through the live fallback alike; search text never read as FTS
syntax; the visitor's own private datasets merged in; cache headers and ETags.
"""

from __future__ import annotations

import gzip
import json
import random
import shutil
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import catalog, config, indexer, search_api, snapshot

app = FastAPI()
app.include_router(search_api.router)
client = TestClient(app)

TAGS = ["code", "agents", "math", "terminal", "swe", "sql", "vision", "a\"quote", "c++", "日本語", "NEAR", "tool-use"]
COLLS = [{"id": "fineenvs", "group": "FineEnvs on OpenEnv", "icon": "globe", "color": "amber", "about": "ours",
          "ids": ["space:fe/geo", "space:fe/latex", "space:fe/ors-one", "space:fe/plain"]},
         {"id": "terminal", "group": "Terminal-Bench", "icon": "terminal", "color": "cyber", "about": "terminal tasks",
          "ids": ["tb/bench-1", "tb/bench-2", "space:fe/plain"]}]   # fe/plain is in both: the first one counts
PINS = ["space:fe/latex", "ds/d07"]
HIDDEN = {"ds/d13", "space:sp/s05"}


def rows_fixture(n_ds: int = 30, n_sp: int = 40, seed: int = 7) -> list[dict]:
    """Listing rows as catalog.environments() returns them (datasets' framework already resolved), varied enough to
    exercise every facet, sort and tie."""
    rnd = random.Random(seed)
    fws = ["harbor", "harbor", "harbor", "verifiers", "nemo-gym", "openenv-data", "verl", "rows", None]
    out = []
    for i in range(n_ds):
        fw = fws[i % len(fws)]
        tasks = rnd.choice([None, None, 0, 5, 99, 100, 999, 1000, 4000]) if fw in ("harbor", None) else None
        out.append({"id": f"ds/d{i:02d}", "key": f"ds/d{i:02d}", "kind": "dataset", "framework": fw,
                    "heading": rnd.choice([None, f"Bench {i}", "Terminal tasks", "Data analysis"]),
                    "brief": rnd.choice(["", "Agents solve SQL tasks in a sandbox.", "Hard math, graded by tests.", "Tools and C++ builds."]),
                    "downloads": rnd.choice([0, 5, 50, 9999, 20000, 5]), "likes": rnd.choice([0, 1, 1, 3, 40]),
                    "trending": rnd.choice([0, 0, 1, 2, 5]), "updated": f"2026-0{rnd.randint(1, 9)}-1{rnd.randint(0, 9)}T10:00:00+00:00",
                    "created": rnd.choice([None, "2025-01-02T00:00:00.123+00:00", f"2026-0{rnd.randint(1, 9)}-01T00:00:00Z"]),
                    "badges": rnd.choice([[], ["RL Environment"], ["Benchmark"], ["RL Environment", "Benchmark"]]),
                    "tags": rnd.sample(TAGS, rnd.randint(0, 4)), "private": False, "gated": False,
                    "indexed": {"tasks": tasks, "graded": {"txt": 3}, "image": 1} if tasks is not None else None})
    out += [{"id": "tb/bench-1", "key": "tb/bench-1", "kind": "dataset", "framework": "harbor", "heading": "TB one", "brief": "Terminal",
             "downloads": 100, "likes": 2, "trending": 3, "badges": ["Benchmark"], "tags": ["terminal"], "indexed": {"tasks": 89},
             "updated": "2026-10-01T00:00:00+00:00"},
            {"id": "tb/bench-2", "key": "tb/bench-2", "kind": "dataset", "framework": "verifiers", "heading": None, "brief": "x" * 40,
             "downloads": 100, "likes": 2, "trending": 3, "badges": [], "tags": [], "indexed": None}]
    for i in range(n_sp):
        fw = rnd.choice(["openenv", "openenv", "openenv", "ors", "space"])
        out.append({"id": f"sp/s{i:02d}", "key": f"space:sp/s{i:02d}", "kind": "space", "framework": fw, "openenv": fw == "openenv",
                    "heading": rnd.choice([None, f"Game {i}", "Wordle"]), "brief": rnd.choice(["", "Play a game.", "An OCR env."]),
                    "downloads": 0, "likes": rnd.choice([0, 1, 2, 2, 10]), "trending": rnd.choice([0, 1, 1, 4]),
                    "updated": rnd.choice([None, f"2026-0{rnd.randint(1, 9)}-0{rnd.randint(1, 9)}T00:00:00+00:00"]),
                    "created": f"2025-1{rnd.randint(0, 2)}-01T00:00:00+00:00", "stage": rnd.choice(["RUNNING", "SLEEPING", None, "RUNTIME_ERROR"]),
                    "mcp": rnd.random() < 0.3, "openenv_version": rnd.choice([None, None, "0.2.1", "0.2.3"]) if fw == "openenv" else None,
                    "badges": ["OpenEnv"] if fw == "openenv" else ["ORS"] if fw == "ors" else [], "tags": rnd.sample(TAGS, rnd.randint(0, 3)),
                    "private": False, "gated": False, "hardware": "cpu-basic", "manifest": "openenv.yaml" if fw == "openenv" else None})
    for sid, fw in (("fe/geo", "openenv"), ("fe/latex", "space"), ("fe/ors-one", "ors"), ("fe/plain", "space")):
        out.append({"id": sid, "key": f"space:{sid}", "kind": "space", "framework": fw, "openenv": fw == "openenv", "heading": sid,
                    "brief": "FineEnvs server", "downloads": 0, "likes": 1, "trending": 0, "stage": "RUNNING", "mcp": True,
                    "badges": ["OpenEnv"] if fw == "openenv" else ["ORS"] if fw == "ors" else [], "tags": ["agents"]})
    out.append({"id": "priv/secret", "key": "priv/secret", "kind": "dataset", "framework": "harbor", "private": True, "heading": "Secret",
                "brief": "", "tags": [], "badges": [], "downloads": 0, "likes": 0, "trending": 99})
    return out


ROWS = rows_fixture()


def write_index(spec: str, tasks: list[dict], restricted: bool = False) -> None:
    idx = {"spec": spec, "sha": "abc", "built": 1.0, "version": catalog.INDEX_VERSION, "info": {"restricted": restricted},
           "tasks": tasks, "summary": {}}
    catalog.atomic_write(catalog._index_path(spec), gzip.compress(json.dumps(idx).encode()))


TASKS = {"ds/d00": [{"path": "tasks/sql-join", "title": "Join two tables in SQL", "brief": "Write a query.", "category": "sql",
                     "difficulty": "easy", "tags": ["sql"], "verifier": {"kind": "txt"}, "env": {}, "run": "image"},
                    {"path": "tasks/foo-only", "title": "Foo the bar", "brief": "Nothing else.", "category": "misc", "tags": [],
                     "verifier": {}, "env": {}}],
         "ds/d13": [{"path": "tasks/hidden-one", "title": "Join hidden tables", "brief": "", "tags": [], "verifier": {}, "env": {}}],
         "priv/secret": [{"path": "tasks/s", "title": "Join secret tables", "brief": "", "tags": [], "verifier": {}, "env": {}}]}


@pytest.fixture(autouse=True)
def stubs(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "collections", lambda: COLLS)
    monkeypatch.setattr(catalog, "pinned", lambda: list(PINS))
    monkeypatch.setattr(catalog, "hidden", lambda: set(HIDDEN))
    monkeypatch.setattr(catalog, "environments", lambda include_hidden=False: [r for r in ROWS if include_hidden or r["key"] not in HIDDEN])
    monkeypatch.setattr(search_api, "rollout_counts", lambda: {"ds/d03": 4, "ds/d05": 1, "sp/s01": 2})
    monkeypatch.setenv("RLX_SNAPSHOT_LOCAL", str(tmp_path / "local"))
    monkeypatch.setenv("RLX_SNAPSHOT_STORE", str(tmp_path / "store"))
    for spec, tasks in TASKS.items():
        write_index(spec, tasks)
    snapshot.reset()
    search_api._answers.clear()
    yield
    snapshot.reset()
    for spec in TASKS:
        catalog._index_path(spec).unlink(missing_ok=True)


def publish_fixture(tmp_path: Path) -> dict:
    """ROWS as a published snapshot in the test store (as the indexer would: raw rows, admin settings left out)."""
    store = snapshot.LocalStore(tmp_path / "store")
    db = tmp_path / "build.db"
    public = [r["id"] for r in ROWS if r["kind"] == "dataset"]
    counts = snapshot.build_db(db, ROWS, snapshot.iter_index_tasks(public, mimo=False))
    return indexer.publish(store, db, counts)


def fw_js(d: dict) -> str:
    """No runtime evidence in the base fixture: metadata only supplies candidates."""
    if d["kind"] == "space":
        return "unverified-space" if d.get("framework") == "openenv" or d.get("openenv") else "other-space"
    return d.get("framework") or "harbor"


def visible() -> list[dict]:
    return [r for r in ROWS if r["key"] not in HIDDEN and not r.get("private")]


# ── the taxonomy and the orders ──────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["snapshot", "live"])
def test_kind_taxonomy(tmp_path, mode):
    if mode == "snapshot":
        publish_fixture(tmp_path)
    d = client.get("/api/search?size=100").json()
    assert d["snapshot"]["source"] == mode
    expected: dict[str, int] = {}
    for r in visible():
        k = fw_js(r)
        expected[k] = expected.get(k, 0) + 1
    assert {k: v for k, v in d["facets"]["kind"].items() if k != "all"} == expected
    assert d["facets"]["kind"]["all"] == d["total"] == len(visible())
    assert expected["other-space"] > 0 and expected["unverified-space"] > 0
    cards = {c["key"]: c for c in d["rows"]}
    # Collections cannot supply framework evidence, including for FineEnvs.
    assert all(c["fw"] == "other-space" for c in cards.values() if c["kind"] == "space" and c["collection"] != "fineenvs" and c["framework"] == "ors")
    for key in ("space:fe/latex", "space:fe/ors-one", "space:fe/plain"):
        assert cards[key]["fw"] == "other-space" and not cards[key]["openenv"]
    assert cards["space:fe/plain"]["collection"] == "fineenvs"   # the first collection naming it
    assert cards["ds/d08"]["fw"] == "harbor"                      # no framework at all: Harbor, as the page reads it
    for k in expected:
        sub = client.get(f"/api/search?kind={k}&size=100").json()
        assert sub["total"] == expected[k] and all(c["fw"] == k for c in sub["rows"])
    assert client.get("/api/search?k=ors").json()["query"]["kind"] == "other-space"   # the page's old links


def test_trending_uses_metrics_and_featured_is_separate(tmp_path):
    publish_fixture(tmp_path)
    rows = client.get("/api/search?size=100").json()["rows"]
    want = sorted(visible(), key=lambda r: (-r["trending"], -r["likes"], -r["downloads"]))
    assert [r["key"] for r in rows] == [r["key"] for r in want]
    oe = client.get("/api/search?kind=unverified-space&size=100").json()["rows"]
    assert [r["key"] for r in oe] == [r["key"] for r in want if fw_js(r) == "unverified-space"]
    assert client.get("/api/search?kind=unverified-space&sort=likes&size=100").json()["rows"][0]["likes"] == max(r["likes"] for r in oe)
    trending = client.get("/api/search?trending=5&facets=0").json()["trending"]
    assert [r["key"] for r in trending] == [r["key"] for r in rows[:5]]
    assert [r["key"] for r in client.get("/api/search?trending=5").json()["featured"]] == PINS
    filtered = client.get("/api/search?kind=unverified-space&trending=48").json()
    assert filtered["featured"] == []
    assert [r["key"] for r in filtered["trending"]] == [r["key"] for r in oe]


def test_hidden_environments_are_left_out_everywhere(tmp_path):
    publish_fixture(tmp_path)
    d = client.get("/api/search?size=100&trending=48").json()
    keys = {r["key"] for r in d["rows"]} | {r["key"] for r in d["trending"]}
    assert not keys & HIDDEN and "priv/secret" not in keys
    assert d["stats"]["datasets"] + d["stats"]["spaces"] == len(visible())
    assert client.get("/api/search?q=d13").json()["total"] == 0
    assert client.get("/api/search/tasks?q=join").json()["rows"] == [
        {"env": "ds/d00", "ref": "tasks/sql-join", "title": "Join two tables in SQL", "brief": "Write a query.", "category": "sql",
         "difficulty": "easy", "grading": "txt", "run": "image"}]    # not the hidden one's, nor the private one's


def test_an_admin_hiding_shows_without_a_new_snapshot(tmp_path, monkeypatch):
    publish_fixture(tmp_path)
    before = client.get("/api/search?q=d01").json()["total"]
    assert client.get("/api/search?q=d01").json()["total"] == before == 1   # answered again (from memory)
    monkeypatch.setattr(catalog, "hidden", lambda: set(HIDDEN) | {"ds/d01"})
    assert client.get("/api/search?q=d01").json()["total"] == 0              # a settings change is a new question


def test_recent_openenv_checks_versions_and_trending_use_same_filters(tmp_path, monkeypatch):
    from app import space_checks
    import time
    publish_fixture(tmp_path)
    now = time.time()
    evidence = {"fe/geo": {"schema": 1, "id": "fe/geo", "stage": "RUNNING", "checked_at": now,
                            "status": "API checked", "mode": "Simulation", "tools": 0,
                            "version": {"value": "0.2.3", "source": "Lockfile"}}}
    monkeypatch.setattr(space_checks, "inventory", lambda: evidence)
    params = {"kind": "openenv", "f": "health:API%20checked;oe:0.2.3", "trending": "24"}
    data = client.get("/api/search", params=params).json()
    assert data["total"] == 1
    assert [r["id"] for r in data["rows"]] == [r["id"] for r in data["trending"]] == ["fe/geo"]
    assert data["rows"][0]["version_source"] == "Lockfile"
    assert data["rows"][0]["tools_status"] == "No tools discovered"
    assert data["facets"]["oe"] == [["0.2.3", 1]]
    assert client.get("/api/search", params={**params, "f": "health:API%20checked;tools:Tools%20discovered"}).json()["total"] == 0
    # Negative checks and expiration invalidate cached search responses too.
    evidence["fe/geo"]["status"] = "Checks failed"
    assert client.get("/api/search", params=params).json()["total"] == 0
    evidence["fe/geo"].update(status="API checked", checked_at=now - space_checks.FRESH - 1)
    assert client.get("/api/search", params=params).json()["total"] == 0
    expired = client.get("/api/search", params={"f": "health:Check%20expired"}).json()
    assert expired["total"] == 1 and expired["rows"][0]["api_status"] == "Check expired"


def test_sql_python_parity_for_api_evidence(tmp_path, monkeypatch):
    from app import space_checks
    import time
    publish_fixture(tmp_path)
    now = time.time()
    statuses = ["API checked", "Checks failed", "Not running", "Check unavailable"]
    evidence = {r["id"]: {"id": r["id"], "status": statuses[i % 4], "checked_at": now - (i % 5) * 1000,
                          "mode": "Simulation", "tools": i % 3, "version": {"value": ">=0.2.3", "source": "Dependency constraint"}}
                for i, r in enumerate(ROWS) if r.get("kind") == "space"}
    monkeypatch.setattr(space_checks, "inventory", lambda: evidence)
    ctx = search_api.context()
    cards = search_api.py_cards([r for r in ROWS if not r.get("private")], ctx)
    with snapshot.use() as (_, conn):
        for facet, value in [("health", "API checked"), ("health", "Check expired"), ("health", "Not checked"),
                             ("mode", "Simulation"), ("tools", "Tools discovered"), ("oe", ">=0.2.3"),
                             ("oe_source", "Dependency constraint")]:
            q = search_api.Query(size=100, sel={facet: [value]})
            actual, expected = search_api.sql_search(conn, q, ctx), search_api.py_search(cards, q)
            assert actual["total"] == expected["total"]
            assert actual["facets"] == expected["facets"]
            assert [r["id"] for r in actual["rows"]] == [r["id"] for r in expected["matches"]]


# ── parity with the page's own filtering ─────────────────────────────────────
def random_queries(n: int, seed: int = 3) -> list[search_api.Query]:
    rnd = random.Random(seed)
    words = ["", "", "a", "te", "sql", "bench", "game ocr", "s0", "C++", "日本", "\"", "near", "x" * 41, "d1 tasks"]
    out = []
    for _ in range(n):
        sel = {}
        for k, vals in (("type", ["Neither", "Benchmark", "OpenEnv", "ORS", "RL Environment"]),
                        ("size", ["Under 100", "100 to 1,000", "1,000 or more", "Not indexed yet"]),
                        ("stage", ["Running", "Sleeping", "Error", "Unknown"]), ("mcp", ["MCP tagged", "No MCP tag"]),
                        ("evidence", ["Manifest present", "Hub tag only"]),
                        ("oe", ["0.2.1", "0.2.3"]), ("tags", TAGS)):
            if rnd.random() < 0.2:
                sel[k] = rnd.sample(vals, rnd.randint(1, 2))
        out.append(search_api.Query(q=rnd.choice(words), kind=rnd.choice(search_api.KINDS), sort=rnd.choice(search_api.SORTS),
                                    coll=rnd.choice([None, None, "fineenvs", "terminal", "other"]), sel=sel))
    return out


@pytest.mark.parametrize("mode", ["snapshot", "live"])
def test_parity_with_the_pages_filtering(tmp_path, mode):
    if mode == "snapshot":
        publish_fixture(tmp_path)
    ctx = search_api.context()
    cards = search_api.py_cards([r for r in ROWS if not r.get("private")], ctx)
    with snapshot.use() as (snap, conn):
        assert snap.source == mode
        for q in random_queries(300):
            got = search_api.sql_search(conn, q, ctx, limit=1000, offset=0)
            want = search_api.py_search(cards, q)
            assert [r["key"] for r in got["rows"]] == [d["key"] for d in want["matches"]], q
            assert got["total"] == want["total"], q
            for k in ("kind", "collection", *search_api.FACETS):
                assert got["facets"][k] == want["facets"][k], (q, k)


def test_api_facets_match_the_reference(tmp_path):
    publish_fixture(tmp_path)
    ctx = search_api.context()
    cards = search_api.py_cards([r for r in ROWS if not r.get("private")], ctx)
    for qs, q in [("f=tags:sql|code;type:Neither", search_api.Query(sel={"tags": ["sql", "code"], "type": ["Neither"]})),
                  ("k=openenv&f=stage:Running&s=updated", search_api.Query(kind="openenv", sel={"stage": ["Running"]}, sort="updated")),
                  ("c=other&q=game&sort=new", search_api.Query(coll="other", q="game", sort="new")),
                  ("f=tags:a%22quote", search_api.Query(sel={"tags": ['a"quote']}))]:
        d = client.get(f"/api/search?{qs}&size=100").json()
        want = search_api.py_search(cards, q)
        assert d["total"] == want["total"] and [r["key"] for r in d["rows"]] == [x["key"] for x in want["matches"]][:100]
        for k in search_api.FACETS:
            assert d["facets"][k] == search_api._rank(want["facets"][k]), k
        assert d["facets"]["collection"] == want["facets"]["collection"]


def test_paging_covers_every_match_once(tmp_path):
    publish_fixture(tmp_path)
    total = client.get("/api/search?facets=0").json()["total"]
    seen = []
    for page in range(1, 20):
        rows = client.get(f"/api/search?sort=likes&page={page}&size=7&facets=0").json()["rows"]
        if not rows:
            break
        seen += [r["key"] for r in rows]
    assert len(seen) == len(set(seen)) == total


# ── search text is data, never FTS syntax ────────────────────────────────────
NASTY = ['"', '""', '"unbalanced', 'NEAR(a b)', 'NEAR(sql, 2)', 'a*', '*', '^sql', 'title:sql', 'brief:x', '{id heading}:sql',
         'AND', 'OR', 'NOT', 'sql OR math', 'sql NOT math', '(sql', 'sql)', '-sql', '+sql', "'; DROP TABLE envs; --",
         '\x00sql', 'c++', '日本語', '%', '_', '\\', 'a" OR "b', 'sql"*', ':', '"" ""', 'NEAR', 'near(']


@pytest.mark.parametrize("text", NASTY)
def test_search_text_cant_inject(tmp_path, text):
    publish_fixture(tmp_path)
    r = client.get("/api/search", params={"q": text, "size": 100})
    assert r.status_code == 200
    q = search_api.Query(q=text.strip())
    want = search_api.py_search(search_api.py_cards([x for x in ROWS if not x.get("private")], search_api.context()), q)
    assert [x["key"] for x in r.json()["rows"]] == [d["key"] for d in want["matches"]]
    t = client.get("/api/search/tasks", params={"q": text})
    assert t.status_code == 200 and isinstance(t.json()["rows"], list)


def test_fts_expressions_are_quoted():
    assert search_api.envs_fts_query(['a"b', "near(x", "ab"]) == '"a""b" "near(x"'
    assert search_api.tasks_fts_query('NEAR(sql OR "x*" title:join ^z)') == '"near"* "sql"* "or"* "x"* "title"* "join"* "z"*'
    assert search_api.tasks_fts_query('"*" -- ()') is None
    assert search_api.tasks_fts_query("a " * 50).count("*") == 8


def test_task_search_operators_are_words(tmp_path):
    publish_fixture(tmp_path)
    assert [r["ref"] for r in client.get("/api/search/tasks?q=foo").json()["rows"]] == ["tasks/foo-only"]
    assert client.get("/api/search/tasks?q=foo OR join").json()["total"] == 0     # OR is a word to find, not an operator
    assert client.get("/api/search/tasks?q=NEAR(foo bar)").json()["total"] == 0   # nor NEAR ("near" isn't in it)
    assert client.get("/api/search/tasks?q=NOT foo").json()["total"] == 1          # "not" finds "Nothing": a word
    assert client.get("/api/search/tasks?q=jo").json()["total"] == 1             # prefixes
    assert client.get("/api/search/tasks?q=join&env=ds/d01").json()["total"] == 0
    assert client.get("/api/search/tasks?q=join&env=../x").status_code == 400


# ── the visitor's own, headers ───────────────────────────────────────────────
def test_mine_merges_private_datasets(tmp_path, monkeypatch):
    publish_fixture(tmp_path)
    mine = [{"id": "me/private-ds", "key": "me/private-ds", "kind": "dataset", "framework": "harbor", "private": True, "heading": "Mine",
             "brief": "my sql tasks", "tags": ["sql"], "badges": [], "downloads": 0, "likes": 0, "trending": 50, "indexed": {"tasks": 3}},
            {**next(r for r in ROWS if r["id"] == "ds/d02"), "mine": True}]
    monkeypatch.setattr(search_api, "_mine_rows", lambda request: mine)
    d = client.get("/api/search?mine=1&c=mine&size=100").json()
    assert {r["key"] for r in d["rows"]} == {"me/private-ds", "ds/d02"} and d["total"] == 2
    assert d["facets"]["collection"]["mine"] == 2 and d["mine"] == 2
    top = client.get("/api/search?mine=1&trending=3").json()
    assert top["trending"][0]["key"] == "me/private-ds"
    assert [r["key"] for r in top["featured"]] == PINS
    own = client.get("/api/search?mine=1&c=mine&trending=24").json()
    assert {r["key"] for r in own["trending"]} == {"me/private-ds", "ds/d02"}
    assert top["total"] == len(visible()) + 1 and top["stats"]["datasets"] == sum(1 for r in visible() if r["kind"] == "dataset") + 1
    assert next(r for r in top["rows"] if r["key"] == "me/private-ds")["private"] is True
    r = client.get("/api/search?mine=1")
    assert r.headers["cache-control"].startswith("private") and r.headers["vary"] == "Cookie"
    monkeypatch.setattr(search_api, "_mine_rows", lambda request: None)   # signed out: "mine" is no filter
    assert client.get("/api/search?mine=1&c=mine").json()["query"]["collection"] is None


def test_cache_headers_and_etag(tmp_path):
    publish_fixture(tmp_path)
    r = client.get("/api/search?q=sql")
    assert r.headers["cache-control"].startswith("public, max-age=") and r.headers["etag"].startswith('W/"')
    again = client.get("/api/search?q=sql", headers={"If-None-Match": r.headers["etag"]})
    assert again.status_code == 304 and not again.content
    assert client.get("/api/search/tasks?q=sql").headers["cache-control"].startswith("public")


def test_responses_are_cards_only(tmp_path):
    publish_fixture(tmp_path)
    d = client.get("/api/search?trending=24").json()
    assert len(d["rows"]) == 40 and len(d["trending"]) == 24
    assert set(d["rows"][0]) == {"id", "key", "kind", "framework", "fw", "openenv", "collection", "heading", "brief", "downloads", "likes",
                                 "trending", "updated", "created", "stage", "mcp", "openenv_version", "manifest", "hardware", "badges",
                                 "tags", "pinned", "mine", "indexed", "rollouts", "private", "evidence", "stage_label",
                                 "api_check", "api_status", "api_mode", "declared_version", "version_source", "tools_status"}
    assert [c["id"] for c in d["collections"]] == ["fineenvs", "terminal"] and "ids" not in d["collections"][0]
    assert client.get("/api/search?q=ds/d03").json()["rows"][0]["rollouts"] == 4
    assert len(client.get("/api/search?trending=24").content) < 60_000


def test_rollouts_sort(tmp_path):
    publish_fixture(tmp_path)
    rows = client.get("/api/search?sort=rollouts&size=3").json()["rows"]
    assert [r["id"] for r in rows] == ["ds/d03", "sp/s01", "ds/d05"]


def teardown_module(module):
    shutil.rmtree(Path(config.STORAGE_DIR) / "snapshots", ignore_errors=True)


def test_environments_helper_for_whole_listing_readers(tmp_path, monkeypatch):
    publish_fixture(tmp_path)
    rows = search_api.environments()
    assert [r["key"] for r in rows] == [r["key"] for r in ROWS if r["key"] not in HIDDEN and not r.get("private")]   # listing order
    by = {r["key"]: r for r in rows}
    assert by["space:fe/plain"]["framework"] == "space" and by["space:fe/plain"]["collection"] == "fineenvs"
    assert by["ds/d07"]["pinned"] and not by["ds/d07"]["gated"] and not by["ds/d07"]["private"]
    assert {r["key"] for r in search_api.environments(include_hidden=True)} >= HIDDEN


def test_the_visitors_datasets_never_make_a_search_wait(monkeypatch):
    from app import auth

    calls = []

    def mine(token):
        calls.append(token)
        return [{"id": f"me/v{len(calls)}"}]

    monkeypatch.setattr(auth, "current_user", lambda request: {"name": "me", "token": "hf_secret_token"})
    monkeypatch.setattr(catalog, "mine", mine)
    search_api._mine.clear()
    assert search_api._mine_rows(None)[0]["id"] == "me/v1" and search_api._mine_rows(None)[0]["id"] == "me/v1" and len(calls) == 1
    who = next(iter(search_api._mine))
    assert "hf_secret_token" not in who
    search_api._mine[who] = (search_api._mine[who][0] - search_api.MINE_FRESH - 1, search_api._mine[who][1])
    assert search_api._mine_rows(None)[0]["id"] == "me/v1"          # stale: answered at once...
    for _ in range(100):
        if search_api._mine[who][1][0]["id"] == "me/v2":
            break
        time.sleep(0.02)
    assert search_api._mine[who][1][0]["id"] == "me/v2"             # ...and refreshed behind it
    monkeypatch.setattr(auth, "current_user", lambda request: None)
    assert search_api._mine_rows(None) is None


def test_a_spaces_rank_matches_the_trending_order_without_the_listing(tmp_path):
    publish_fixture(tmp_path)
    key = lambda d: (-d["trending"], -d["likes"], -(d.get("downloads") or 0))
    spaces = [r for r in visible() if r["kind"] == "space"]
    order = sorted(spaces, key=key)
    for i in (0, len(order) // 2, len(order) - 1):
        r = client.get(f"/api/search/rank?key={order[i]['key']}")
        assert r.status_code == 200 and r.json() == {"n": i + 1, "of": len(order)}, order[i]["key"]
    assert client.get(f"/api/search/rank?key={next(iter(HIDDEN))}").status_code == 404   # hidden: not ranked
    assert client.get("/api/search/rank?key=space:nobody/nothing").status_code == 404


def test_runtime_and_evidence_filters_and_census_agree(tmp_path):
    publish_fixture(tmp_path)
    d = client.get("/api/search?kind=unverified-space&size=100").json()
    assert sum(d["stats"]["openenv"].values()) == d["total"]
    for evidence, count in d["stats"]["openenv"].items():
        r = client.get("/api/search", params={"kind": "unverified-space", "f": f"evidence:{evidence}", "trending": 48, "size": 100}).json()
        assert r["total"] == count
        assert all(x["evidence"] == evidence for x in r["rows"] + r["trending"])
    running = client.get("/api/search?kind=unverified-space&f=stage:Running").json()
    assert running["total"] == d["stats"]["openenv_running"]
    errors = client.get("/api/search?f=stage:Error&size=100").json()
    assert errors["total"] and all(r["stage"] == "RUNTIME_ERROR" and r["stage_label"] == "Error" for r in errors["rows"])
    sleeping = client.get("/api/search?f=stage:Sleeping&size=100").json()
    assert sleeping["total"] and all(r["stage"] == "SLEEPING" for r in sleeping["rows"])
    old = client.get("/api/search?f=stage:Asleep%20or%20stopped;mcp:Has%20MCP%20tools").json()
    assert "Error" in old["query"]["f"]["stage"] and old["query"]["f"]["mcp"] == ["MCP tagged"]


def test_trending_priorities_cannot_overflow_into_each_other(tmp_path, monkeypatch):
    # Old 1e9/1e4 weights let popularity displace a larger recent-interest score,
    # and capped downloads made 10k and 10M downloads indistinguishable.
    custom = [dict(ROWS[0], id=f"rank/{i}", key=f"rank/{i}", trending=t, likes=l, downloads=d)
              for i, (t, l, d) in enumerate([(0, 10**9, 10**12), (0.1, 0, 0), (1, 3, 10000), (1, 3, 10**7)])]
    monkeypatch.setattr(__import__(__name__, fromlist=["ROWS"]), "ROWS", custom)
    monkeypatch.setattr(catalog, "pinned", lambda: ["rank/0"])
    publish_fixture(tmp_path)
    got = client.get("/api/search?trending=4").json()
    assert [r["id"] for r in got["rows"]] == ["rank/3", "rank/2", "rank/1", "rank/0"]
    assert got["trending"] == got["rows"]
    assert [r["id"] for r in got["featured"]] == ["rank/0"]
    assert got["ranking"]["order"] == ["trending", "likes", "downloads"]


def test_empty_filters_have_no_unrelated_trending_or_featured(tmp_path):
    publish_fixture(tmp_path)
    d = client.get("/api/search?q=absent-environment-xyz&trending=24").json()
    assert d["total"] == 0 and d["trending"] == [] and d["featured"] == []


@pytest.mark.parametrize("mode", ["published", "fallback"])
def test_owner_filter_is_exact_and_shared_by_results_facets_and_trending(tmp_path, monkeypatch, mode):
    import sys
    space = next(r for r in ROWS if r["kind"] == "space")
    dataset = ROWS[0]
    mixed = [*ROWS, {**space, "id": "ds/owned-space", "key": "space:ds/owned-space"},
             {**dataset, "id": "ds-copy/impostor", "key": "ds-copy/impostor", "heading": "ds environments"}]
    monkeypatch.setattr(sys.modules[__name__], "ROWS", mixed)
    if mode == "published":
        publish_fixture(tmp_path)
    expected = {r["key"] for r in visible() if r["id"].split("/")[0] == "ds"}
    data = client.get("/api/search", params={"owner": "DS", "size": 100, "trending": 48}).json()
    assert {r["key"] for r in data["rows"]} == expected
    assert {r["kind"] for r in data["rows"]} == {"dataset", "space"}
    assert data["total"] == data["facets"]["kind"]["all"] == len(expected)
    assert {r["key"] for r in data["trending"]} == expected
    assert all(r["id"].split("/")[0] == "ds" for r in data["featured"])
    query = search_api.Query(owner="DS")
    ref = search_api.py_search(search_api.py_cards([r for r in mixed if not r.get("private")], search_api.context()), query)
    assert {r["key"] for r in ref["matches"]} == expected
    pages = [client.get("/api/search", params={"owner": "ds", "size": 10, "page": page, "facets": 0}).json()["rows"]
             for page in range(1, (len(expected) + 9) // 10 + 1)]
    assert {r["key"] for page in pages for r in page} == expected
    assert sum(map(len, pages)) == len(expected)
    filtered = client.get("/api/search", params={"owner": "ds", "kind": "openenv", "q": "owned-space"}).json()
    assert all(r["id"] == "ds/owned-space" for r in filtered["rows"])
    for owner in ("d", "ds%", "ds/", "missing", "ds' OR 1=1 --"):
        empty = client.get("/api/search", params={"owner": owner, "trending": 24}).json()
        assert empty["total"] == 0 and empty["trending"] == [] and empty["featured"] == []


def test_owner_filter_applies_to_authorized_private_dataset_merge(tmp_path, monkeypatch):
    publish_fixture(tmp_path)
    mine = [{**ROWS[0], "id": "me/private", "key": "me/private", "private": True},
            {**ROWS[0], "id": "someone-else/private", "key": "someone-else/private", "private": True}]
    monkeypatch.setattr(search_api, "_mine_rows", lambda request: mine)
    public = client.get("/api/search", params={"owner": "me"}).json()
    assert public["total"] == 0
    response = client.get("/api/search", params={"owner": "ME", "mine": 1, "trending": 24})
    data = response.json()
    assert data["total"] == 1
    assert data["stats"]["datasets"] == 1
    assert data["stats"]["dataset_tasks"] == (mine[0].get("indexed") or {}).get("tasks", 0)
    assert [r["id"] for r in data["rows"]] == [r["id"] for r in data["trending"]] == ["me/private"]
    assert data["rows"][0]["private"] is True
    assert response.headers["cache-control"].startswith("private")


@pytest.mark.parametrize("status,stage,age,expected", [
    (None, "RUNNING", 0, "unverified-space"),
    ("Checks failed", "RUNNING", 0, "unverified-space"),
    ("API checked", "RUNNING", 0, "openenv"),
    ("API checked", "RUNNING", 3601, "unverified-space"),
    ("API checked", "SLEEPING", 0, "unverified-space"),
])
def test_manifest_filename_never_establishes_openenv_without_fresh_live_evidence(tmp_path, monkeypatch, status, stage, age, expected):
    from app import space_checks
    publish_fixture(tmp_path)
    target = next(r for r in ROWS if r.get("manifest"))
    record = {"id": target["id"], "status": status, "stage": stage, "checked_at": time.time() - age}
    monkeypatch.setattr(space_checks, "inventory", lambda: {target["id"]: record} if status else {})
    data = client.get("/api/search", params={"q": target["id"], "trending": 24}).json()
    row = next(r for r in data["rows"] if r["id"] == target["id"])
    assert row["fw"] == expected
    assert row["openenv"] is (expected == "openenv")
    if expected != "openenv":
        assert row["framework"] == "space" and "OpenEnv" not in row["badges"]
    filtered = client.get("/api/search", params={"q": target["id"], "kind": "openenv", "trending": 24}).json()
    assert bool(filtered["rows"]) is (expected == "openenv")
    assert bool(filtered["trending"]) is (expected == "openenv")
    assert filtered["facets"]["kind"].get(expected, 0) > 0


def test_verified_untagged_space_has_protocol_facets(tmp_path, monkeypatch):
    from app import space_checks
    publish_fixture(tmp_path)
    record = {"id": "fe/plain", "status": "API checked", "stage": "RUNNING", "checked_at": time.time(), "mode": "Simulation", "tools": 0}
    monkeypatch.setattr(space_checks, "inventory", lambda: {"fe/plain": record})
    data = client.get("/api/search", params={"kind": "openenv", "f": "health:API%20checked"}).json()
    assert [r["id"] for r in data["rows"]] == ["fe/plain"]
    ctx = search_api.context()
    cards = search_api.py_cards([r for r in ROWS if not r.get("private")], ctx)
    with snapshot.use() as (_, conn):
        q = search_api.Query(kind="openenv")
        assert search_api.sql_search(conn, q, ctx)["facets"] == search_api.py_search(cards, q)["facets"]


def test_ready_scope_counts_and_facets_share_live_evidence(tmp_path, monkeypatch):
    from app import space_checks
    publish_fixture(tmp_path)
    now = time.time()
    good = {"id": "fe/geo", "status": "API checked", "stage": "RUNNING", "checked_at": now,
            "task_catalog": {"checked_at": now, "tasks": 1234, "counted_splits": 1, "complete": True}}
    checks = {"fe/geo": good,
              "fe/latex": {**good, "id": "fe/latex", "checked_at": now - 3601, "task_catalog": {}},
              "fe/plain": {**good, "id": "fe/plain", "status": "Checks failed", "task_catalog": {}},
              "fe/ors-one": {**good, "id": "fe/ors-one", "status": "Checks failed", "interface": "ors", "task_catalog": {}}}
    monkeypatch.setattr(space_checks, "inventory", lambda: checks)
    data = client.get("/api/search", params={"scope": "ready", "owner": "fe", "size": 100, "trending": 24}).json()
    assert {r["id"] for r in data["rows"]} == {"fe/geo", "fe/ors-one"}
    assert data["stats"]["datasets"] == data["stats"]["dataset_tasks"] == 0
    assert data["stats"]["spaces"] == data["total"] == data["facets"]["kind"]["all"] == 2
    assert data["stats"]["space_tasks"] == 1234
    rank = client.get("/api/search/rank", params={"scope": "ready", "key": "space:fe/geo"}).json()
    assert rank["of"] == 2
    assert client.get("/api/search/rank", params={"scope": "ready", "key": "space:fe/plain"}).status_code == 404
    ctx = search_api.context()
    cards = search_api.py_cards([r for r in ROWS if not r.get("private")], ctx)
    with snapshot.use() as (_, conn):
        for q in (search_api.Query(scope="ready"), search_api.Query(scope="ready", owner="fe"),
                  search_api.Query(scope="ready", kind="openenv")):
            sql = search_api.sql_search(conn, q, ctx, limit=100)
            py = search_api.py_search(cards, q)
            assert sql["total"] == py["total"]
            assert sql["facets"] == py["facets"]
    datasets = client.get("/api/search", params={"scope": "ready", "owner": "ds"}).json()
    assert datasets["stats"]["datasets"] == datasets["total"] and datasets["stats"]["spaces"] == 0
    assert datasets["stats"]["dataset_tasks"] == len(TASKS["ds/d00"])
    empty = client.get("/api/search", params={"scope": "ready", "q": "nothing-matches"}).json()
    assert all(empty["stats"][k] == 0 for k in ("datasets", "spaces", "dataset_tasks", "space_tasks"))


def test_fineenvs_promoted_without_pins_and_without_changing_trending(tmp_path, monkeypatch):
    import sys
    from app import space_checks
    custom = [{**ROWS[0], "id": "FineEnvs/data", "key": "FineEnvs/data", "trending": 0},
              {**ROWS[1], "id": "other/popular", "key": "other/popular", "trending": 100},
              {**ROWS[-2], "id": "FineEnvs/server", "key": "space:FineEnvs/server", "kind": "space", "trending": 0}]
    monkeypatch.setattr(sys.modules[__name__], "ROWS", custom)
    monkeypatch.setattr(catalog, "pinned", list)
    monkeypatch.setattr(space_checks, "inventory", lambda: {"FineEnvs/server": {
        "id": "FineEnvs/server", "status": "API checked", "stage": "RUNNING", "checked_at": time.time()}})
    publish_fixture(tmp_path)
    data = client.get("/api/search?scope=ready&trending=24").json()
    assert [r["id"] for r in data["promoted"]] == ["FineEnvs/server", "FineEnvs/data"]
    assert data["trending"][0]["id"] == "other/popular"
    assert data["featured"] == []
    assert client.get("/api/search?scope=ready&owner=other&trending=24").json()["promoted"] == []
    assert client.get("/api/search?scope=ready&q=popular&trending=24").json()["promoted"] == []
    assert len(client.get("/api/search?scope=ready&owner=fineenvs&trending=24").json()["promoted"]) == 2
