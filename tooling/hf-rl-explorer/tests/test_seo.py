"""Every page at its own address, findable and previewable (app/seo.py), without starting work or leaking anything
(no network: the catalog's listing is stubbed).

What is checked: each page answers with its own title, description, canonical address and schema.org data; what a
page shows comes escaped (a task title can't inject markup or end the JSON-LD script); pages that are personal or
unknown aren't indexed; old addresses move for good; robots.txt keeps crawlers off the API and rollouts; the
sitemap lists environments and indexed tasks; preview images are images; unknown paths are 404 pages.
"""

from __future__ import annotations

import json
import re
import time
import html as html_lib
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from app import main, seo, snapshot

client = TestClient(main.app, base_url="https://testserver")
EVIL = '</script><script>alert(1)</script>"<b>'
ROWS = [{"id": "org/ds", "key": "org/ds", "kind": "dataset", "heading": f"Bench {EVIL}", "brief": "A bench.", "tags": ["code"], "framework": "harbor",
         "indexed": {"tasks": 2}, "updated": "2026-10-01T00:00:00+00:00", "downloads": 10, "likes": 2, "trending": 1},
        {"id": "org/sp", "key": "space:org/sp", "kind": "space", "heading": "Game", "brief": "A game env.", "tags": ["openenv"], "trending": 0,
         "framework": "openenv", "likes": 3},
        {"id": "org/copy", "key": "space:org/copy", "kind": "space", "heading": "Echo copy", "tags": ["openenv"], "framework": "openenv", "likes": 0}]
TASKS = [{"path": "tasks/a", "title": f"Fix it {EVIL}", "brief": "Do it.", "category": "code"}, {"path": "tasks/b", "title": "B", "brief": "", "category": None}]


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    snapshot.reset()   # seo reads the listing through the snapshot (a live one here): built from these rows
    seo._listing.update(at=0.0, by={}, rows=[])
    seo._sitemap.update(at=0.0, pages=[], tasks=[])
    monkeypatch.setattr(seo.catalog, "environments", lambda include_hidden=False: ROWS)
    monkeypatch.setattr(seo, "index_rows", lambda spec: TASKS if spec == "org/ds" else [])
    monkeypatch.setattr(seo.space_checks, "inventory", lambda: {"org/sp": {"id": "org/sp", "stage": "RUNNING", "status": "API checked", "checked_at": time.time()}})
    monkeypatch.setattr(seo.spaces_live, "last_seen", lambda spec: None)


def head(html: str) -> dict:
    return {"title": re.search(r"<title>(.*?)</title>", html).group(1),
            "description": re.search(r'<meta name="description" content="([^"]*)"', html).group(1),
            "canonical": html_lib.unescape(re.search(r'rel="canonical" href="([^"]*)"', html).group(1)),
            "robots": re.search(r'name="robots" content="([^"]*)"', html).group(1),
            "ld": [json.loads(x) for x in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)]}


def test_every_page_has_its_own_head():
    for path, kind in [("/", "WebSite"), ("/d/org/ds", "Dataset"), ("/t/org/ds/tasks/a", "CreativeWork"), ("/s/org/sp", "SoftwareApplication")]:
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        h = head(r.text)
        assert h["canonical"] == "https://testserver" + path and "index" in h["robots"] and not h["robots"].startswith("noindex")
        assert kind in [x["@type"] for x in h["ld"]] and h["description"]
    assert "Bench" in head(client.get("/d/org/ds").text)["title"]
    assert client.get("/t/org/ds/tasks/a").text.count('href="/t/') == 0   # a task page links up, not sideways
    assert client.get("/d/org/ds").text.count('href="/t/org/ds/tasks/') == 2   # crawlable links to its tasks


def test_what_a_page_shows_is_escaped():
    for path in ("/d/org/ds", "/t/org/ds/tasks/a", "/"):
        html = client.get(path).text
        lds = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
        for x in lds:
            assert "</" not in x   # nothing in the data can end its script early
            json.loads(x)
        rest = re.sub(r'<script type="application/ld\+json">.*?</script>', "", html, flags=re.S)
        assert "<script>alert(1)" not in rest and '"<b>' not in rest, path
        assert "&lt;/script&gt;" in rest or path == "/"   # shown, as text


def test_personal_and_unknown_pages_arent_indexed():
    for path in ("/runs", "/run/20260101-000000-abcdef", "/compare/org/ds/tasks/a?r=x,y", "/d/someone/unknown", "/t/org/ds/tasks/zzz"):
        assert head(client.get(path).text)["robots"].startswith("noindex"), path
    r = client.get("/nowhere/at/all")
    assert r.status_code == 404 and head(r.text)["robots"].startswith("noindex")
    assert client.get("/api/nothing").json() == {"detail": "Not Found"}


def test_old_addresses_move_for_good():
    for old, new in [("/r/org/ds?c=default&s=train&i=3", "/t/org/ds/train/3"), ("/r/org/ds?c=code&s=train&i=6", "/t/org/ds/code/train/6"),
                     ("/task/music-1", f"/t/{seo.MIMO}/music-1"), ("/rewards", f"/d/{seo.MIMO}?rewards=1")]:
        r = client.get(old, follow_redirects=False)
        assert r.status_code == 301 and r.headers["location"] == new, old


def test_robots_and_sitemap():
    t = client.get("/robots.txt").text
    for line in ("Disallow: /api/", "Disallow: /mcp/", "Disallow: /run/", "Sitemap: https://testserver/sitemap.xml"):
        assert line in t
    idx = client.get("/sitemap.xml").text
    assert "<sitemapindex" in idx and "/sitemap-pages.xml" in idx and "/sitemap-tasks-1.xml" in idx
    pages = client.get("/sitemap-pages.xml").text
    assert "https://testserver/d/org/ds" in pages and "https://testserver/s/org/sp" in pages and "<lastmod>2026-10-01</lastmod>" in pages
    assert "https://testserver/s/org/copy" not in pages   # an unliked, sleeping copy isn't put forward
    tasks = client.get("/sitemap-tasks-1.xml").text
    assert "https://testserver/t/org/ds/tasks/a" in tasks
    assert client.get("/sitemap-tasks-99.xml").status_code == 404
    assert client.get("/api/me").headers["x-robots-tag"] == "noindex"


def test_preview_images_are_images():
    for path in ("/og.png", "/og/d/org/ds.png", "/og/t/org/ds/tasks/a.png", "/og/s/org/sp.png"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:8] == b"\x89PNG\r\n\x1a\n", path
    assert client.get("/og/d/org/ds.jpg").status_code == 404
    # nothing listed there: no image drawn (and its page points at the site's card instead)
    for path in ("/og/d/nobody/nothing.png", "/og/s/nobody/nothing.png", "/og/whatever.png", "/og/x/y/z.png"):
        assert client.get(path).status_code == 404, path
    assert 'content="https://testserver/social/rl-explorer.png"' in client.get("/d/nobody/nothing").text


def test_space_task_has_unique_canonical_prompt_and_preview(monkeypatch):
    monkeypatch.setattr(seo.seo_tasks, "space", lambda *args: {"title": "Find the landmark", "brief": f"Locate this place {EVIL}", "total": 5})
    path = "/s/org/sp?task=01&split=test+set&env=game&utm_source=test"
    response = client.get(path)
    h = head(response.text)
    assert response.status_code == 200
    assert h["canonical"] == "https://testserver/s/org/sp?env=game&split=test+set&task=1"
    assert "Find the landmark" in h["title"] and h["robots"].startswith("index")
    assert 'class="seo-prompt"' in response.text and "Locate this place" in response.text
    assert 'task=0' in response.text and 'task=2' in response.text
    assert '<script>alert(1)' not in response.text
    assert '/og/s/org/sp.png?env=game&amp;split=test+set&amp;task=1' in response.text
    assert "CreativeWork" in [item["@type"] for item in h["ld"]]


def test_space_task_missing_and_temporary_failure_have_real_statuses(monkeypatch):
    for status in (404, 503):
        def fail(*args):
            raise seo.seo_tasks.Unavailable(status)
        monkeypatch.setattr(seo.seo_tasks, "space", fail)
        response = client.get("/s/org/sp?env=game&split=train&task=0")
        assert response.status_code == status
        if status == 503:
            assert response.headers["retry-after"] == "60"
            assert not head(response.text)["robots"].startswith("noindex")
    for query in ("task=-1", "task=0", "task=no&env=game&split=train"):
        assert client.get("/s/org/sp?" + query).status_code == 404


def test_space_sitemaps_partition_ranges_without_fetching_tasks(monkeypatch):
    monkeypatch.setattr(seo, "SITEMAP_CHUNK", 2)
    monkeypatch.setattr(seo, "_space_ranges", lambda: [("org/sp", "game", "train & eval", 3), ("org/sp", "game", "test", 2)])
    def fail(*args):
        raise AssertionError("sitemaps must not fetch tasks")
    monkeypatch.setattr(seo.seo_tasks, "space", fail)
    assert "sitemap-space-tasks-3.xml" in client.get("/sitemap.xml").text
    urls = []
    for n in range(1, 4):
        r = client.get(f"/sitemap-space-tasks-{n}.xml")
        assert r.status_code == 200
        urls.extend(x.text for x in ET.fromstring(r.text).iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"))
    assert len(urls) == len(set(urls)) == 5
    assert urls[2].endswith("env=game&split=train+%26+eval&task=2")
    assert urls[3].endswith("env=game&split=test&task=0")
    assert client.get("/sitemap-space-tasks-4.xml").status_code == 404


def test_sitemaps_deduplicate_mimo_and_exclude_hidden_or_private(monkeypatch):
    rows = [*ROWS, {**ROWS[0], "id": seo.MIMO, "key": seo.MIMO}, {**ROWS[0], "id": "org/private", "key": "org/private", "private": True}]
    monkeypatch.setattr(seo, "listing", lambda: ({r["key"]: r for r in rows}, rows))
    monkeypatch.setattr(seo, "index_rows", lambda spec: TASKS)
    monkeypatch.setattr(seo.catalog, "hidden", lambda: ["org/ds"])
    tasks = client.get("/sitemap-tasks-1.xml").text
    assert tasks.count("MiMo-V2.6-RL-oss/tasks/a") == 1
    assert "org/private" not in tasks and "/t/org/ds/" not in tasks
    assert head(client.get("/t/org/private/tasks/a").text)["robots"].startswith("noindex")
    assert "Do it." not in client.get("/t/org/private/tasks/a").text


def test_public_row_task_is_rendered_without_login(monkeypatch):
    rows = [{**ROWS[0], "framework": "nemo-gym", "indexed": None}]
    monkeypatch.setattr(seo, "listing", lambda: ({r["key"]: r for r in rows}, rows))
    monkeypatch.setattr(seo, "index_rows", lambda spec: [])
    monkeypatch.setattr(seo.seo_tasks, "row", lambda *args: {"title": "Follow the instructions", "brief": "Write exactly three words."})
    response = client.get("/t/org/ds/train/0")
    assert response.status_code == 200 and "Write exactly three words." in response.text
    assert head(response.text)["robots"].startswith("index")


def test_static_social_card_is_available_without_catalog():
    import io
    from PIL import Image
    response = client.get("/social/rl-explorer.png")
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.content)).size == (1200, 630)


def test_home_reads_one_consistent_check_inventory(monkeypatch):
    calls = []
    rows = [{**ROWS[1], "id": f"org/space{i}", "key": f"space:org/space{i}"} for i in range(100)]
    monkeypatch.setattr(seo, "listing", lambda: ({r["key"]: r for r in rows}, rows))
    monkeypatch.setattr(seo.space_checks, "inventory", lambda: calls.append(True) or {})
    assert client.get("/").status_code == 200
    assert len(calls) == 1


def test_sitemap_uses_snapshot_tasks_without_bucket_reads(monkeypatch):
    from contextlib import contextmanager
    class Connection:
        def execute(self, sql, params):
            assert "org/ds" in json.loads(params[0])
            return self
        def fetchall(self):
            return [{"env": "org/ds", "ref": "tasks/from-snapshot"}]
    @contextmanager
    def use():
        yield None, Connection()
    monkeypatch.setattr(snapshot, "use", use)
    monkeypatch.setattr(seo, "listing", lambda: ({r["key"]: r for r in ROWS}, ROWS))
    monkeypatch.setattr(seo, "index_rows", lambda spec: pytest.fail("must not read bucket indexes"))
    assert "tasks/from-snapshot" in client.get("/sitemap-tasks-1.xml").text


def test_a_card_section_heading_is_not_a_datasets_name():
    from app.catalog import _card_text

    desc = "NVIDIA NeMo Gym knowledge MCQA dataset for training models on multiple choice questions across domains."
    assert _card_text(f"Dataset Description:\n\n{desc}") == (None, desc)
    assert _card_text(f"Overview\n\n{desc}")[0] is None
    assert _card_text(f"Dataset Card for Terminal Bench\n\n{desc}")[0] == "Terminal Bench"
    assert _card_text(f"Terminal-Bench 2.1\n\n{desc}")[0] == "Terminal-Bench 2.1"
