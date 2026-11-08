"""Which real apps the GUI surface offers, and how other packages add more.

DesignGym ships three real desktop apps (LibreOffice Impress and Draw, Inkscape), run on the shared Linux desktops
(realapps/desktop). Other app backends register from outside, as a plugin: a package declares an entry point in
the group `designgym.apps` whose object is a function that calls `register(...)` (and, for desktop apps that need
more software or agent code on the desktop, `register_desktop(...)`):

    # the plugin's pyproject.toml
    [project.entry-points."designgym.apps"]
    my_apps = "my_package.apps:register"

    # my_package/apps.py
    from designgym.realapps.registry import App, register

    def register():
        register(App(skin="myapp", name="MyApp", about="the real MyApp, on a Linux desktop", agent_app="myapp",
                     prompt="You operate MyApp ... {coords} ... {width}x{height} ..."))

`DESIGNGYM_APP_PLUGINS` (comma list of `module:function`) loads plugins that are importable but not installed with
an entry point. Plugins load once, the first time the registry is read.

Two kinds of app:

- `desktop`: the app runs on a desktop (one X display per episode) next to the desktop agent (desktop/agent.py),
  which sets each episode up and reads the app's document back. `agent_app` names the agent's app; a plugin adds
  agent-side app classes with `register_desktop(agent_modules=[...])` (Python files the agent imports from its
  `apps/` folder, each with an `APPS = {name: AppClass}` dict) and, when the app is not in the public desktop image,
  `register_desktop(dockerfile=...)` (an image that has it).
- `web`: a web app the env's own browser opens. `backend(fonts_dir)` returns the shared backend; its
  `open_episode(env, ep, gt)` sets the episode up and opens the page (server/gui.py's `GuiDriver.open`), and its
  `read_doc(session)` / `write_doc(session, doc)` / `release(url)` work as the desktop backend's do.
"""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

ENTRY_POINT_GROUP = "designgym.apps"


@dataclass(frozen=True)
class App:
    skin: str  # what `reset(skin=...)` takes
    name: str  # shown to people (the playground, metadata)
    about: str
    kind: str = "desktop"  # "desktop" or "web"
    # desktop apps
    agent_app: str | None = None  # the desktop agent's name for the app (desktop/agent.py APPS)
    text: str = "top"  # how the agent places text: "top" (the first line's top) or "baseline" (SVG-style anchors)
    natural_width: bool = False  # text frames clip: the expert also sends each text's natural width
    # web apps
    backend: Callable[[Path | None], object] | None = None
    # the GUI instructions for this app: `prompt` takes {coords}, {width} and {height}; `finish` takes {max_steps}
    # (default: the submit tool hands in the current document)
    prompt: str | None = None
    finish: str | None = None
    # where a locally run app answers, for status(): (role, url, health path); None for the shared desktops
    endpoint: tuple[str, str, str] | None = None
    # run before the server's workers start when apps run on Daytona (e.g. start a shared sandbox once)
    prepare_daytona: Callable[[object], None] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Desktop:
    """What the desktops run: the image to build them from, and agent code beside the public agent."""

    dockerfile: Path = Path(__file__).parent / "desktop" / "Dockerfile"
    agent_modules: list[Path] = field(default_factory=list)


_APPS: dict[str, App] = {}
_DESKTOP = Desktop()
_LOCK = threading.RLock()
_LOADED = False


def register(app: App) -> App:
    """Add (or replace) a real app."""
    if app.kind not in ("desktop", "web"):
        raise ValueError(f"{app.skin}: kind must be desktop or web")
    if app.kind == "desktop" and not app.agent_app:
        raise ValueError(f"{app.skin}: a desktop app names its agent_app")
    if app.kind == "web" and app.backend is None:
        raise ValueError(f"{app.skin}: a web app needs a backend")
    with _LOCK:
        _APPS[app.skin] = app
    return app


def register_desktop(dockerfile: Path | str | None = None, agent_modules=()) -> Desktop:
    """Build the desktops from another image (one that has the plugin's apps), and/or give the desktop agent more
    app modules (uploaded to its `apps/` folder on Daytona; mounted there locally)."""
    with _LOCK:
        if dockerfile is not None:
            _DESKTOP.dockerfile = Path(dockerfile)
        for m in agent_modules:
            if Path(m) not in _DESKTOP.agent_modules:
                _DESKTOP.agent_modules.append(Path(m))
    return _DESKTOP


def load_plugins() -> None:
    """Run every registered plugin once: entry points in `designgym.apps`, then DESIGNGYM_APP_PLUGINS."""
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        _LOADED = True
        from importlib.metadata import entry_points

        for ep in entry_points(group=ENTRY_POINT_GROUP):
            ep.load()()
        for spec in filter(None, (s.strip() for s in os.environ.get("DESIGNGYM_APP_PLUGINS", "").split(","))):
            module, _, fn = spec.partition(":")
            getattr(importlib.import_module(module), fn or "register")()


def apps() -> dict[str, App]:
    load_plugins()
    with _LOCK:
        return dict(_APPS)


def get(skin: str) -> App:
    try:
        return apps()[skin]
    except KeyError:
        raise ValueError(f"no real app with skin {skin!r}") from None


def desktop() -> Desktop:
    load_plugins()
    return _DESKTOP


class RealSkins(Mapping):
    """skin -> (name, about) of every registered app, read live (plugins included)."""

    def __getitem__(self, skin: str) -> tuple[str, str]:
        a = apps()[skin]
        return a.name, a.about

    def __iter__(self) -> Iterator[str]:
        return iter(apps())

    def __len__(self) -> int:
        return len(apps())


# ------------------------------------------------------------------------------------------- the public apps

_DESKTOP_PROMPT = (
    "You operate {app}, a real desktop app on Linux, through screenshots, with the mouse and keyboard. {{coords}} "
    "{canvas} The task's images are files named after their asset ids (a1.png, a2.png, ...) in the Pictures folder "
    "of your home directory: {insert} Text is added with the text tool. Ctrl+Z undoes. Read the screen rather "
    "than assuming where controls are."
)


def desktop_prompt(app: str, canvas: str, insert: str) -> str:
    """The instructions of a desktop app, in the same words as the built-in ones (for plugins too)."""
    return _DESKTOP_PROMPT.format(app=app, canvas=canvas, insert=insert)


register(App(
    skin="impress", name="LibreOffice Impress", about="the real LibreOffice Impress, on a Linux desktop",
    agent_app="impress",
    prompt=desktop_prompt(
        app="LibreOffice Impress",
        canvas="The slide is the {width}x{height} canvas; anything off the slide is not scored. Impress shows "
               "lengths in inches: 1 px = 1/96 inch, so 96 px = 1\" (and 0.2646 mm).",
        insert="add one with Insert > Image... and pick it in the file window, then move and size it; Format > "
               "Position and Size (F4) takes exact values.")))
register(App(
    skin="draw", name="LibreOffice Draw", about="the real LibreOffice Draw, on a Linux desktop", agent_app="draw",
    prompt=desktop_prompt(
        app="LibreOffice Draw",
        canvas="The page is the {width}x{height} canvas; anything off the page is not scored. Draw shows lengths "
               "in inches: 1 px = 1/96 inch, so 96 px = 1\" (and 0.2646 mm).",
        insert="add one with Insert > Image... and pick it in the file window, then move and size it; Format > "
               "Position and Size (F4) takes exact values.")))
register(App(
    skin="inkscape-desktop", name="Inkscape", about="the real Inkscape, on a Linux desktop", agent_app="inkscape",
    text="baseline",
    prompt=desktop_prompt(
        app="Inkscape",
        canvas="The page is the {width}x{height} canvas in px; anything off the page is not scored.",
        insert="add one with File > Import... (Ctrl+I), pick it in the file window and accept the import dialog; "
               "the X/Y/W/H fields of the toolbar take exact values.")))
