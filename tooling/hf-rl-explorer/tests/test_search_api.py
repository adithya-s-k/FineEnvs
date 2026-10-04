"""/api/search and /api/search/tasks (app/search_api.py) over a catalog snapshot (no network: the listing is fixture rows).

What is checked: the Kind taxonomy the Explore page has always used (ORS with the other Spaces, FineEnvs' curated
Spaces as OpenEnv, Harbor whenever an index found tasks); pinned first in trending and FineEnvs' curated servers first
among OpenEnv Spaces; hidden environments left out of everything at query time; every answer (order, total, every
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
    """web/js/home.js `fw()`, verbatim, over catalog.environments()'s row (curated Spaces already OpenEnv there)."""
    if d["kind"] == "space":
        return "openenv" if d.get("framework") == "openenv" or d.get("openenv") else "other-space"
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
        k = fw_js({**r, "framework": "openenv"} if r["key"] in COLLS[0]["ids"] and r["kind"] == "space" else r)
        expected[k] = expected.get(k, 0) + 1
    assert {k: v for k, v in d["facets"]["kind"].items() if k != "all"} == expected
    assert d["facets"]["kind"]["all"] == d["total"] == len(visible())
    assert expected["other-space"] > 0 and expected["openenv"] > 0
    cards = {c["key"]: c for c in d["rows"]}
    # ORS sits with the other Spaces; FineEnvs' curated Spaces are OpenEnv whatever their manifest says
    assert all(c["fw"] == "other-space" for c in cards.values() if c["kind"] == "space" and c["collection"] != "fineenvs" and c["framework"] == "ors")
    for key in ("space:fe/latex", "space:fe/ors-one", "space:fe/plain"):
        assert cards[key]["fw"] == "openenv" and cards[key]["openenv"] and cards[key]["badges"] == ["OpenEnv"]
    assert cards["space:fe/plain"]["collection"] == "fineenvs"   # the first collection naming it
    assert cards["ds/d08"]["fw"] == "harbor"                      # no framework at all: Harbor, as the page reads it
    for k in expected:
        sub = client.get(f"/api/search?kind={k}&size=100").json()
        assert sub["total"] == expected[k] and all(c["fw"] == k for c in sub["rows"])
    assert client.get("/api/search?k=ors").json()["query"]["kind"] == "other-space"   # the page's old links


def test_pinned_first_and_curated_first(tmp_path):
    publish_fixture(tmp_path)
    rows = client.get("/api/search?size=100").json()["rows"]
    assert {r["key"] for r in rows[:2]} == set(PINS)
    assert all(r["pinned"] for r in rows[:2]) and not any(r["pinned"] for r in rows[2:])
    oe = client.get("/api/search?kind=openenv&size=100").json()["rows"]
    assert oe[0]["key"] == "space:fe/latex"                       # pinned beats curated
    curated = [r["collection"] == "fineenvs" for r in oe]
    assert curated[:4] == [True] * 4 and not any(curated[4:])    # FineEnvs' servers lead the OpenEnv Spaces
    assert client.get("/api/search?kind=openenv&sort=likes&size=100").json()["rows"][0]["likes"] == max(r["likes"] for r in oe)
    trending = client.get("/api/search?trending=5&facets=0").json()["trending"]
    assert [r["key"] for r in trending] == [r["key"] for r in rows[:5]]


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


# ── parity with the page's own filtering ─────────────────────────────────────
def random_queries(n: int, seed: int = 3) -> list[search_api.Query]:
    rnd = random.Random(seed)
    words = ["", "", "a", "te", "sql", "bench", "game ocr", "s0", "C++", "日本", "\"", "near", "x" * 41, "d1 tasks"]
    out = []
    for _ in range(n):
        sel = {}
        for k, vals in (("type", ["Neither", "Benchmark", "OpenEnv", "ORS", "RL Environment"]),
                        ("size", ["Under 100", "100 to 1,000", "1,000 or more", "Not indexed yet"]),
                        ("stage", ["Running", "Asleep or stopped", "Unknown"]), ("mcp", ["Has MCP tools", "No MCP tag"]),
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
    assert top["trending"][0]["key"] in PINS and "me/private-ds" in [r["key"] for r in top["trending"]]
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
                                 "tags", "pinned", "mine", "indexed", "rollouts", "private"}
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
    assert by["space:fe/plain"]["framework"] == "openenv" and by["space:fe/plain"]["collection"] == "fineenvs"
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
    key = lambda d: d["trending"] * 1e9 + d["likes"] * 1e4 + min(d.get("downloads") or 0, 9999)
    spaces = [r for r in visible() if r["kind"] == "space"]
    order = sorted(spaces, key=lambda d: -key(d))   # stable: listing order breaks ties, as on the page
    for i in (0, len(order) // 2, len(order) - 1):
        r = client.get(f"/api/search/rank?key={order[i]['key']}")
        assert r.status_code == 200 and r.json() == {"n": i + 1, "of": len(order)}, order[i]["key"]
    assert client.get(f"/api/search/rank?key={next(iter(HIDDEN))}").status_code == 404   # hidden: not ranked
    assert client.get("/api/search/rank?key=space:nobody/nothing").status_code == 404
