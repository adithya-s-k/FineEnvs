"""Real design apps behind the DesignGym GUI interface (skin="impress", "draw", "inkscape-desktop", and any a plugin
registers: realapps/registry.py).

Each backend sets an episode up inside the app (a blank canvas at the task's size with its assets loaded), hands
the GUI driver a screen (a desktop) or a URL (a web app) to drive, and on submit reads the app's own document back
as a DesignDoc so the same reward scores it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from . import registry

_BACKENDS: dict = {}
_LOCK = threading.Lock()


def desktop_skins() -> tuple[str, ...]:
    """Skins that run on the shared desktops (realapps/desktop)."""
    return tuple(k for k, a in registry.apps().items() if a.kind == "desktop")


def get_backend(skin: str, fonts_dir: Path | None = None):
    """The shared backend for a real-app skin.

    With DESIGN_CANVAS_APPS=daytona the server runs the desktops in Daytona sandboxes it manages (realapps/daytona.py;
    DESIGN_CANVAS_DAYTONA_DESKTOPS desktops, default 2). Otherwise they are reached at fixed addresses:
    DESIGN_CANVAS_DESKTOP_URLS (desktop agents, one episode each at a time). All desktop apps share one pool of
    desktops; a web app's backend comes from its plugin (`App.backend`)."""
    app = registry.get(skin)
    with _LOCK:
        if skin not in _BACKENDS:
            if app.kind == "web":
                _BACKENDS[skin] = app.backend(fonts_dir)
            else:
                skins = desktop_skins()
                desk = next((b for k, b in _BACKENDS.items() if k in skins), None)  # one pool for all desktop apps
                if desk is None and os.environ.get("DESIGN_CANVAS_APPS", "").lower() == "daytona":
                    from .daytona import DaytonaDesktopBackend

                    desk = DaytonaDesktopBackend(daytona_apps(), int(os.environ.get("DESIGN_CANVAS_DAYTONA_DESKTOPS",
                                                                                     "2")), fonts_dir)
                elif desk is None:
                    from .desktop.client import DesktopBackend

                    desk = DesktopBackend()
                _BACKENDS[skin] = desk
        return _BACKENDS[skin]


_APPS = None


def daytona_apps():
    """The server's Daytona sandboxes (one manager per process, shared by every app that runs there)."""
    global _APPS
    if _APPS is None:
        from .daytona import DaytonaApps

        _APPS = DaytonaApps()
    return _APPS


_apps = daytona_apps  # the earlier name


def status(check: bool = False) -> dict:
    """Which real apps this server offers and where they run, for the playground and GET /api/apps.

    With check, each app's endpoint (or the Daytona pool) is also asked whether it is up; without it nothing is
    contacted, so the call is free."""
    daytona = os.environ.get("DESIGN_CANVAS_APPS", "").lower() == "daytona"
    out = {"runs_on": "daytona" if daytona else "local",
           "apps": [{"skin": k, "name": a.name, "about": a.about, "surface": "gui"}
                    for k, a in registry.apps().items()]}
    if daytona:
        out["desktops"] = int(os.environ.get("DESIGN_CANVAS_DAYTONA_DESKTOPS", "2"))
        if check:
            apps = daytona_apps()
            out["sandboxes"] = [{"role": sb.labels.get("role"), "id": sb.id[:8],
                                 "state": str(sb.state).split(".")[-1].lower(), "cpu": sb.cpu,
                                 "memory_gb": sb.memory, "episodes": apps.busy.get(sb.id, 0)}
                                for sb in apps.sandboxes() if sb.labels.get("role")]
    else:
        desktops = [u.strip() for u in os.environ.get("DESIGN_CANVAS_DESKTOP_URLS", "http://localhost:5005").split(",")
                    if u.strip()]
        web = {a.endpoint for a in registry.apps().values() if a.endpoint}  # (role, url, health path)
        out["endpoints"] = ([{"role": "desktop", "url": u, "health": "/health"} for u in desktops] +
                            [{"role": role, "url": url, "health": path} for role, url, path in sorted(web)])
        if check:
            for e in out["endpoints"]:
                e["up"] = _up(e["url"] + e["health"])
    return out


def _up(url: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=3):
            return True
    except urllib.error.HTTPError as e:  # an app's API may refuse a call without a session: it is up
        return e.code < 500
    except Exception:  # noqa: BLE001
        return False


_AVAILABLE: dict = {}


def available() -> tuple[bool, str]:
    """Whether this server can run real apps, and if not, why (cached for 30 s; cheap enough for every page load).

    On Daytona that needs the SDK and a key; locally, the desktop (and any web app's) endpoints must answer."""
    import time

    if _AVAILABLE and time.time() - _AVAILABLE["at"] < 30:
        return _AVAILABLE["ok"], _AVAILABLE["why"]
    if os.environ.get("DESIGN_CANVAS_APPS", "").lower() == "daytona":
        try:
            import daytona  # noqa: F401
        except ImportError:
            ok, why = False, "the daytona package is not installed (install the `daytona` extra)"
        else:
            ok, why = (bool(os.environ.get("DAYTONA_API_KEY")), "DAYTONA_API_KEY is not set")
    else:
        st = status(check=True)
        down = [e["url"] for e in st["endpoints"] if not e.get("up")]
        ok, why = not down, ("real apps run locally here and nothing answers at " + ", ".join(down) +
                             "; run them (realapps/) or set DESIGN_CANVAS_APPS=daytona")
    _AVAILABLE.update(at=time.time(), ok=ok, why="" if ok else why)
    return ok, _AVAILABLE["why"]
