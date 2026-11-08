"""The env server's Daytona pool (realapps/daytona.py), against a fake Daytona: no account, no network.

The pool hands each desktop episode its own sandbox, keeps it awake while the episode runs, and gets it back once.
"""

from __future__ import annotations

import pytest

pytest.importorskip("daytona")

from designgym.realapps import daytona as dmod  # noqa: E402
from designgym.realapps.desktop import client as desktop_client  # noqa: E402


@pytest.fixture(autouse=True)
def _own_locks(tmp_path, monkeypatch):
    # desktops are locked through files shared by every process on the machine: each test gets its own folder
    monkeypatch.setattr(desktop_client, "LOCKS", tmp_path / "locks")


class _Box:
    def __init__(self, i):
        self.id, self.labels = f"sb{i}", {"role": "desktop"}


class _Apps:
    def __init__(self, n):
        self.made = [_Box(i) for i in range(n)]
        self.busy, self.prepared = {}, []

    def desktops(self, n, fonts_dir):
        return self.made[:n]

    def prepare_desktop(self, sb, fonts, check_agent=False):
        self.prepared.append(sb.id)

    def claim(self, sb):
        self.busy[sb.id] = self.busy.get(sb.id, 0) + 1

    def release(self, sb):
        self.busy[sb.id] -= 1

    @staticmethod
    def link(sb, port):
        return f"https://{port}-{sb.id}.proxy.test", {"x-daytona-preview-token": "tok"}


class _Reset(dmod.DesktopClient):
    def reset(self, *a, **kw):
        pass


def test_each_desktop_episode_gets_a_sandbox_kept_awake_until_released(monkeypatch):
    monkeypatch.setattr(dmod, "DesktopClient", _Reset)
    monkeypatch.setattr(dmod, "_fonts_bundle", lambda d: None)
    apps = _Apps(2)
    be = dmod.DaytonaDesktopBackend(apps, 2, None)
    a = be.new_episode("draw", 100, 100, {})
    b = be.new_episode("impress", 100, 100, {})
    assert {a.url, b.url} == {"sb0", "sb1"} and not be.free
    assert a.client.base == f"https://5005-{a.url}.proxy.test" and a.client.headers["x-daytona-preview-token"] == "tok"
    assert apps.busy == {"sb0": 1, "sb1": 1} and sorted(apps.prepared) == ["sb0", "sb1"]
    be.release(a.url)
    assert apps.busy[a.url] == 0 and be.free == [a.url]
    c = be.new_episode("inkscape-desktop", 100, 100, {})
    assert c.url == a.url and apps.busy[a.url] == 1


def test_a_failed_reset_gives_the_sandbox_back(monkeypatch):
    class _Broken(dmod.DesktopClient):
        def reset(self, *a, **kw):
            raise RuntimeError("app did not start")

    monkeypatch.setattr(dmod, "DesktopClient", _Broken)
    monkeypatch.setattr(dmod, "_fonts_bundle", lambda d: None)
    apps = _Apps(1)
    be = dmod.DaytonaDesktopBackend(apps, 1, None)
    with pytest.raises(RuntimeError):
        be.new_episode("draw", 100, 100, {})
    assert be.free == ["sb0"] and apps.busy["sb0"] == 0


def test_a_desktop_held_by_another_process_is_skipped(monkeypatch):
    monkeypatch.setattr(dmod, "DesktopClient", _Reset)
    monkeypatch.setattr(dmod, "_fonts_bundle", lambda d: None)
    other = desktop_client.DesktopBackend._lock("sb0")  # another worker is running an episode on sb0
    assert other is not None
    be = dmod.DaytonaDesktopBackend(_Apps(2), 2, None)
    assert be.new_episode("draw", 100, 100, {}).url == "sb1"
    assert desktop_client.DesktopBackend._lock("sb1") is None  # and now this process holds sb1


def test_a_named_pool_never_lists_another_pools_sandboxes(monkeypatch):
    class _Sb:
        def __init__(self, i, labels):
            self.id, self.labels, self.state = i, labels, "started"

    every = [_Sb("local", {"project": "design-canvas", "role": "desktop"}),
             _Sb("space", {"project": "design-canvas", "role": "desktop", "pool": "space"})]

    class _D:
        def list(self, query):  # Daytona's label query matches supersets: both come back for either pool
            return [s for s in every if all(s.labels.get(k) == v for k, v in query.labels.items())]

    apps = dmod.DaytonaApps.__new__(dmod.DaytonaApps)
    apps.d = _D()
    monkeypatch.delenv("DESIGN_CANVAS_DAYTONA_POOL", raising=False)
    assert [s.id for s in apps.sandboxes("desktop")] == ["local"]
    monkeypatch.setenv("DESIGN_CANVAS_DAYTONA_POOL", "space")
    assert [s.id for s in apps.sandboxes("desktop")] == ["space"]
    assert dmod._labels(role="desktop") == {"project": "design-canvas", "pool": "space", "role": "desktop"}
