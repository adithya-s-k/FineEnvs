"""Real-app registry and plugins (realapps/registry.py), the desktop agent's app modules, and the old package name:
no apps, no network."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
import types
from pathlib import Path

import pytest

from designgym.realapps import registry

AGENT = Path(__file__).resolve().parents[1] / "realapps" / "desktop" / "agent.py"


@pytest.fixture
def clean_registry(monkeypatch):
    """A registry with only the built-in apps, restored after the test."""
    monkeypatch.setattr(registry, "_APPS", dict(registry._APPS))
    monkeypatch.setattr(registry, "_DESKTOP", registry.Desktop())
    monkeypatch.setattr(registry, "_LOADED", False)
    monkeypatch.delenv("DESIGNGYM_APP_PLUGINS", raising=False)
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [])
    return registry


def test_builtin_apps(clean_registry):
    apps = registry.apps()
    assert set(apps) == {"impress", "draw", "inkscape-desktop"}
    assert {a.kind for a in apps.values()} == {"desktop"}
    assert apps["inkscape-desktop"].agent_app == "inkscape" and apps["inkscape-desktop"].text == "baseline"
    assert registry.desktop().dockerfile.name == "Dockerfile" and registry.desktop().agent_modules == []


def test_real_skins_and_prompts_follow_the_registry(clean_registry, tmp_path, monkeypatch):
    from designgym.core.prompts import instructions
    from designgym.server.episodes import REAL_SKINS

    mod = types.ModuleType("fake_plugin")

    def register():
        registry.register(registry.App(skin="fakeapp", name="FakeApp", about="a test app", agent_app="fake",
                                       natural_width=True, prompt="Drive FakeApp. {coords} Canvas {width}x{height}.",
                                       finish="Submit within {max_steps}."))
        registry.register_desktop(agent_modules=[tmp_path / "fake.py"])

    mod.register = register
    monkeypatch.setitem(sys.modules, "fake_plugin", mod)
    monkeypatch.setenv("DESIGNGYM_APP_PLUGINS", "fake_plugin:register")
    assert "fakeapp" in REAL_SKINS and REAL_SKINS["fakeapp"] == ("FakeApp", "a test app")
    assert "impress" in REAL_SKINS
    assert registry.desktop().agent_modules == [tmp_path / "fake.py"]
    info = {"canvas": {"width": 300, "height": 200}, "format": "Poster", "title": "T", "assets": [],
            "fonts_used": ["Montserrat"], "max_steps": 7}
    text = instructions("gui", "reference", info, skin="fakeapp")
    assert "Drive FakeApp." in text and "Canvas 300x200." in text and "Submit within 7." in text


def test_entry_points_are_loaded(clean_registry, monkeypatch):
    calls = []

    class _EP:
        def load(self):
            return lambda: calls.append("loaded")

    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda group=None: [_EP()] if group == registry.ENTRY_POINT_GROUP else [])
    registry.apps()
    registry.apps()
    assert calls == ["loaded"]  # once


def test_registration_is_checked(clean_registry):
    with pytest.raises(ValueError, match="agent_app"):
        registry.register(registry.App(skin="x", name="X", about=""))
    with pytest.raises(ValueError, match="backend"):
        registry.register(registry.App(skin="x", name="X", about="", kind="web"))
    with pytest.raises(ValueError, match="no real app"):
        registry.get("nope")


def test_desktop_backend_uses_the_app_spec(clean_registry, monkeypatch, tmp_path):
    from designgym.core.doc import DesignDoc, Element
    from designgym.realapps.desktop import client as dc

    monkeypatch.setattr(dc, "LOCKS", tmp_path / "locks")
    sent = {}

    class _Client(dc.DesktopClient):
        def reset(self, app, *a, **kw):
            sent["app"] = app

        def _post(self, path, body, timeout=60):
            sent["doc"] = body["doc"]

    monkeypatch.setattr(dc.DesktopBackend, "_client", lambda self, url: _Client(url))
    registry.register(registry.App(skin="fakeapp", name="FakeApp", about="", agent_app="fake", natural_width=True))

    class _Fonts:
        def metrics(self, *a):
            return 10.0, 3.0

    be = dc.DesktopBackend(["http://desk"])
    s = be.new_episode("fakeapp", 100, 100, {}, fonts=None)
    assert sent["app"] == "fake" and s.spec.skin == "fakeapp"
    be.release(s.url)
    s2 = be.new_episode("inkscape-desktop", 100, 100, {})
    assert sent["app"] == "inkscape" and s2.spec.text == "baseline"
    doc = DesignDoc(width=100, height=100, elements=[Element.model_validate(
        {"id": "e1", "kind": "image", "x": 0, "y": 0, "width": 10, "height": 10, "asset_id": "a1"})])
    be.write_doc(s2, doc)
    assert sent["doc"]["elements"][0]["asset_id"] == "a1"


def _agent():
    spec = importlib.util.spec_from_file_location("agent_under_test", AGENT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_agent_loads_app_modules(tmp_path, monkeypatch):
    agent = _agent()
    assert set(agent.APPS) == {"impress", "draw", "inkscape"}
    (tmp_path / "fake.py").write_text(textwrap.dedent('''
        import agent

        class Fake:
            def reset(self, width, height, background, paths):
                pass

        APPS = {"fake": Fake}
        KILL = ["fakeapp-bin"]
        ROUTES = {"/fake": lambda body: {"echo": body}}
        fonts_seen = []

        def kill():
            pass

        def fonts_changed():
            fonts_seen.append(1)
    '''))
    (tmp_path / "_helper.py").write_text("raise RuntimeError('not a plugin')\n")
    monkeypatch.setitem(sys.modules, "agent", agent)  # what load_app_modules arranges when the agent runs as a script
    assert agent.load_app_modules(tmp_path) == ["fake"]
    assert "fake" in agent.APPS and "fakeapp-bin" in agent.KILL and "/fake" in agent.ROUTES
    assert len(agent.KILL_HOOKS) == 1 and len(agent.FONT_HOOKS) == 1


def test_old_package_name_is_an_alias():
    import designgym.core.pack as new

    import design_canvas
    import design_canvas.core.pack as old

    assert old is new
    assert design_canvas.__version__
    from design_canvas.server.episodes import SKINS

    from designgym.server.episodes import SKINS as SKINS2

    assert SKINS is SKINS2
