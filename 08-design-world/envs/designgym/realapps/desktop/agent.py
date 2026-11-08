"""Desktop agent: one X display with a real design app, driven over HTTP (runs inside the desktop container).

    GET  /health
    GET  /screenshot                      -> PNG of the 1280x800 screen
    GET  /a11y                            -> the on-screen widgets (role, name, tooltip, box in screen pixels, state,
                                             text / value) from the accessibility tree, for scripted recorders
    POST /input   {"type": "click"|"double"|"triple"|"right"|"move"|"drag"|"type"|"key"|"scroll", ...}
    POST /reset   {"app": "impress"|"draw"|"inkscape"|<a plugin's app>, "width", "height", "background", "assets": {id: base64}}
    POST /read    -> {"width", "height", "background", "elements": [DesignDoc element dicts], "skipped": n}
    POST /expert  {"doc": DesignDoc dict}  -> builds the design in the app (for the score-1.0 check)

The env turns /read into a DesignDoc and scores it; everything app-specific lives in the App classes below.
Coordinates in documents are canvas pixels; each app's own unit is converted at this boundary.

More apps plug in as Python files in the agent's `apps/` folder (next to this file, or DESKTOP_AGENT_APPS), loaded
at start. Each module sees this one's names (`import agent`) and may define:

    APPS            {name: AppClass}: classes with reset(width, height, background, paths), read(), expert(doc)
    KILL            process names to kill before every reset
    kill()          more cleanup before every reset
    fonts_changed() called when the pack's fonts were (re)installed
    ROUTES          {"/path": fn(body) -> dict}: extra POST routes (an app's own control channel)
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

W, H = 1280, 800
DISPLAY = os.environ.get("DISPLAY", ":1")
HOME = Path.home()
PICS = HOME / "Pictures"
WORK = Path("/work")
LOCK = threading.Lock()


def run(cmd: list[str], check=False, timeout=30, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check,
                          env={**os.environ, "DISPLAY": DISPLAY}, **kw)


# ---------------------------------------------------------------------------------------------- display


def start_display():
    n = DISPLAY.lstrip(":")
    for stale in (Path(f"/tmp/.X{n}-lock"), Path(f"/tmp/.X11-unix/X{n}")):  # left by a restarted container
        stale.unlink(missing_ok=True)
    subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", f"{W}x{H}x24", "-nolisten", "tcp"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if run(["xdpyinfo"]).returncode == 0:
            break
        time.sleep(0.1)
    cfg = HOME / ".config/openbox"
    cfg.mkdir(parents=True, exist_ok=True)
    # every application window opens maximised and undecorated, like a full-screen app on a laptop
    # Super+E opens the file manager (where the image has one), as on most desktops: an app whose own file window
    # shows no preview leaves a person to look at the pictures there first
    (cfg / "rc.xml").write_text("""<?xml version="1.0"?><openbox_config xmlns="http://openbox.org/3.4/rc">
<keyboard><keybind key="W-e"><action name="Execute"><command>thunar</command></action></keybind></keyboard>
<applications><application class="*" type="normal"><maximized>yes</maximized><decor>no</decor></application>
</applications></openbox_config>""")
    subprocess.Popen(["openbox"], env={**os.environ, "DISPLAY": DISPLAY},
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fonts = Path(os.environ.get("DESIGN_CANVAS_FONTS", "/fonts"))  # mounted locally, uploaded in a sandbox
    if fonts.is_dir():  # the task pack's fonts, so the apps offer the families the designs use
        _link_fonts(fonts, HOME / ".local/share/fonts/pack")
        run(["fc-cache", "-f"], timeout=300)
    start_a11y()


FONT_HOOKS: list = []  # plugins' fonts_changed()


def _link_fonts(src: Path, dst: Path) -> None:
    """The pack's fonts as a real folder of hard links: some apps' own font scans do not follow a symlinked folder
    (they then draw every pack family with a stand-in), fontconfig's does either."""
    if dst.is_symlink():
        dst.unlink()
    fresh = False
    for f in src.rglob("*"):
        if f.is_file():
            to = dst / f.relative_to(src)
            same = to.exists() and (os.path.samefile(f, to) or to.read_bytes() == f.read_bytes())
            if not same:  # new, or a newer upload of the fonts
                to.parent.mkdir(parents=True, exist_ok=True)
                to.unlink(missing_ok=True)
                try:
                    os.link(f, to)
                except OSError:
                    shutil.copy2(f, to)
                fresh = True
    if fresh:  # apps that keep their own index of the fonts: have them scan again
        for hook in FONT_HOOKS:
            hook()


A11Y_CONF = Path("/usr/share/defaults/at-spi2/accessibility.conf")
A11Y_REGISTRY = Path("/usr/libexec/at-spi2-registryd")
A11Y_LAUNCHER = Path("/usr/libexec/at-spi-bus-launcher")


def start_a11y() -> bool:
    """A session bus, the AT-SPI bus and its registry, so the apps publish their widgets (GET /a11y lists them).
    The apps the agent starts inherit the bus addresses. Without at-spi2-core installed this is a no-op."""
    if not (A11Y_CONF.exists() and A11Y_REGISTRY.exists()):
        return False
    m = re.search(r"DBUS_SESSION_BUS_ADDRESS='([^']+)'", run(["dbus-launch", "--sh-syntax"]).stdout)
    if not m:
        return False
    os.environ["DBUS_SESSION_BUS_ADDRESS"] = m.group(1)
    env = {**os.environ, "DISPLAY": DISPLAY}
    if A11Y_LAUNCHER.exists():
        # the standard launcher: it owns org.a11y.Bus on the session bus, which is where Qt apps look for
        # the accessibility bus; GTK also reads AT_SPI_BUS_ADDRESS, set from the same answer
        subprocess.Popen([str(A11Y_LAUNCHER), "--launch-immediately"], env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        for _ in range(50):
            r = run(["dbus-send", "--session", "--print-reply", "--dest=org.a11y.Bus", "/org/a11y/bus",
                     "org.a11y.Bus.GetAddress"])
            m = re.search(r'string "([^"]+)"', r.stdout)
            if m:
                os.environ["AT_SPI_BUS_ADDRESS"] = m.group(1)
                break
            time.sleep(0.2)
    if "AT_SPI_BUS_ADDRESS" not in os.environ:
        bus = run(["dbus-daemon", f"--config-file={A11Y_CONF}", "--print-address", "--fork"]).stdout.strip().splitlines()
        if not bus:
            return False
        os.environ["AT_SPI_BUS_ADDRESS"] = bus[0]
    subprocess.Popen([str(A11Y_REGISTRY)], env={**os.environ, "DISPLAY": DISPLAY},
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    return True


# Runs in a fresh process per call, so no AT-SPI cache outlives an app restart. GTK widgets report true screen
# boxes; LibreOffice's own (VCL) widgets get the screen y wrong by the height of the native menu bar, but their
# window-relative boxes are right, so those get their toplevel's origin added.
A11Y_SNAPSHOT = r"""
import json, sys, gi
gi.require_version("Atspi", "2.0")
from gi.repository import Atspi
TOP = {"frame", "dialog", "window", "alert", "file chooser"}
ST = Atspi.StateType
WANT = {"focused": ST.FOCUSED, "checked": ST.CHECKED, "selected": ST.SELECTED, "pressed": ST.PRESSED,
        "sensitive": ST.SENSITIVE, "expanded": ST.EXPANDED, "editable": ST.EDITABLE, "active": ST.ACTIVE}
TEXTY = {"text", "spin button", "entry", "combo box", "password text", "paragraph", "label", "table cell",
         "list item", "menu item", "push button", "button", "toggle button", "page tab", "heading"}
out, MAX = [], (6000 if sys.argv[1] == "serve" else int(sys.argv[1]))
def node(a, depth, parent, origin, app):
    if len(out) >= MAX:
        return
    try:
        role, st = a.get_role_name(), a.get_state_set()
    except Exception:
        return
    if depth > 1 and not st.contains(ST.SHOWING):
        return
    # only ask for the interfaces it has: asking GTK widgets for others floods (and can upset) the app
    try:
        ifaces = set(a.get_interfaces())
    except Exception:
        ifaces = set()
    w = s = None
    if "Component" in ifaces:
        try:
            w, s = a.get_extents(Atspi.CoordType.WINDOW), a.get_extents(Atspi.CoordType.SCREEN)
        except Exception:
            w = s = None
    if role in TOP and s is not None and w is not None:
        origin = (s.x - w.x, s.y - w.y)
    try:
        gtk = a.get_attributes().get("toolkit") == "gtk"
    except Exception:
        gtk = False
    # GTK and Qt report true screen positions (Qt's window ones are wrong); LibreOffice's are right in its window
    e, off = (s, (0, 0)) if gtk or app in SCREEN_APPS else (w, origin)
    box = [e.x + off[0], e.y + off[1], e.width, e.height] if (
        e is not None and e.width > 0 and e.height > 0 and e.x > -10000) else None
    rec = {"i": len(out), "p": parent, "d": depth, "role": role, "name": a.get_name() or "", "app": app, "box": box,
           "st": [k for k, v in WANT.items() if st.contains(v)]}
    try:
        desc = a.get_description()
        if desc:
            rec["desc"] = desc
    except Exception:
        pass
    if "Text" in ifaces and (role in TEXTY or "editable" in rec["st"]):
        try:
            n = Atspi.Text.get_character_count(a)
            if 0 < n < 4000:
                rec["text"] = Atspi.Text.get_text(a, 0, n)
        except Exception:
            pass
    if "Value" in ifaces:
        try:
            rec["value"] = Atspi.Value.get_current_value(a)
        except Exception:
            pass
    out.append(rec)
    me = rec["i"]
    try:
        kids = a.get_child_count()
    except Exception:
        return
    for k in range(min(kids, 400)):
        try:
            c = a.get_child_at_index(k)
        except Exception:
            continue
        if c is not None:
            node(c, depth + 1, me, origin, app)
SCREEN_APPS = set()
def snapshot():
    out.clear()
    desk = Atspi.get_desktop(0)
    Atspi.Accessible.clear_cache(desk)  # a long-lived client: read everything afresh, as a new one would
    for i in range(desk.get_child_count()):
        app = desk.get_child_at_index(i)
        if app is not None:
            try:
                if (app.get_toolkit_name() or "").lower().startswith("qt"):
                    SCREEN_APPS.add(app.get_name() or "")
            except Exception:
                pass
            node(app, 0, -1, (0, 0), app.get_name() or "")
    return json.dumps(out, separators=(",", ":"))
if sys.argv[1] == "serve":  # one snapshot per line read
    for line in sys.stdin:
        MAX = int(line)
        sys.stdout.write(snapshot() + "\n")
        sys.stdout.flush()
else:
    sys.stdout.write(snapshot())
"""

# One long-lived snapshot client rather than one per call. A new AT-SPI client has every app list its whole object
# cache (Cache.GetItems), and Inkscape's bridge crashes in that walk when it still holds a widget GTK has destroyed;
# a client that stays only does it when it first meets an app. (Only while Inkscape runs: the others are fine.)
_SNAP: dict = {}
_SNAP_LOCK = threading.Lock()


def _snapshot_served(max_nodes: int) -> str:
    with _SNAP_LOCK:
        p = _SNAP.get("proc")
        if p is None or p.poll() is not None or _SNAP.get("n", 0) >= 500:  # renewed now and then: its cache grows
            if p is not None and p.poll() is None:
                p.kill()
            p = _SNAP["proc"] = subprocess.Popen(["python3", "-c", A11Y_SNAPSHOT, "serve"], stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            _SNAP["n"] = 0
        _SNAP["n"] += 1
        p.stdin.write(f"{max_nodes}\n")
        p.stdin.flush()
        got: list[str] = []
        reader = threading.Thread(target=lambda: got.append(p.stdout.readline()), daemon=True)
        reader.start()
        reader.join(90)
        if not got or not got[0].endswith("\n"):
            p.kill()
            raise RuntimeError("a11y snapshot failed: the snapshot client stopped answering")
        return got[0]


def a11y(max_nodes: int = 6000) -> dict:
    if "AT_SPI_BUS_ADDRESS" not in os.environ:
        raise RuntimeError("no accessibility bus on this desktop (at-spi2-core is not installed)")
    if subprocess.run(["pgrep", "-x", "inkscape"], capture_output=True).returncode == 0:  # the app it is for
        try:
            return {"nodes": json.loads(_snapshot_served(max_nodes))}
        except (RuntimeError, OSError, ValueError):
            pass
    r = run(["python3", "-c", A11Y_SNAPSHOT, str(max_nodes)], timeout=90)
    if r.returncode != 0:
        raise RuntimeError(f"a11y snapshot failed: {r.stderr[-800:]}")
    return {"nodes": json.loads(r.stdout)}


def screenshot(fmt: str = "png", quality: int = 90) -> bytes:
    """The screen as PNG (lossless, the default), or as WebP when asked (`?fmt=webp&q=90`): a recorder on a slow or
    metered link gets a screenshot four to six times smaller, at the quality it would store it in anyway."""
    from PIL import ImageGrab

    img = ImageGrab.grab(xdisplay=DISPLAY)
    buf = io.BytesIO()
    if fmt == "webp":
        img.convert("RGB").save(buf, "WEBP", quality=max(1, min(100, quality)), method=4)
    else:
        img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _shot_args(path: str) -> tuple[str, int]:
    """fmt and quality from a request's query string (/screenshot?fmt=webp&q=90)."""
    from urllib.parse import parse_qs, urlparse

    q = parse_qs(urlparse(path).query)
    fmt = (q.get("fmt") or ["png"])[0].lower()
    try:
        quality = int((q.get("q") or ["90"])[0])
    except ValueError:
        quality = 90
    return ("webp" if fmt == "webp" else "png"), quality


XKEYS = {"ControlOrMeta": "ctrl", "Control": "ctrl", "Meta": "super", "Alt": "alt", "Shift": "shift",
         "Enter": "Return", "Escape": "Escape", "Backspace": "BackSpace", "Delete": "Delete", "Tab": "Tab",
         " ": "space", "ArrowUp": "Up", "ArrowDown": "Down", "ArrowLeft": "Left", "ArrowRight": "Right",
         "Home": "Home", "End": "End", "PageUp": "Prior", "PageDown": "Next", "+": "plus", "-": "minus"}


def xkey(combo: str) -> str:
    """Playwright spelling ('ControlOrMeta+Shift+z') -> xdotool ('ctrl+shift+z')."""
    return "+".join(XKEYS.get(p, p) for p in combo.split("+") if p)


def do_input(a: dict) -> dict:
    t = a["type"]
    xd = ["xdotool"]
    if t in ("click", "double", "triple", "right"):
        mods = [xkey(m) for m in a.get("modifiers") or []]
        for m in mods:
            run(xd + ["keydown", m])
        n = {"click": 1, "double": 2, "triple": 3, "right": 1}[t]
        run(xd + ["mousemove", str(int(a["x"])), str(int(a["y"])), "click", "--repeat", str(n), "--delay", "60",
                  "3" if t == "right" else "1"])
        for m in reversed(mods):
            run(xd + ["keyup", m])
    elif t == "move":
        run(xd + ["mousemove", str(int(a["x"])), str(int(a["y"]))])
    elif t == "drag":
        (x0, y0), (x1, y1) = a["from"], a["to"]
        run(xd + ["mousemove", str(int(x0)), str(int(y0)), "mousedown", "1"])
        for k in range(1, 11):  # a real drag passes through intermediate points
            run(xd + ["mousemove", str(int(x0 + (x1 - x0) * k / 10)), str(int(y0 + (y1 - y0) * k / 10))])
            time.sleep(0.02)
        run(xd + ["mouseup", "1"])
    elif t == "type":
        text = a["text"]
        if all(ord(c) <= 0x7E for c in text) or bind_chars(text):
            run(xd + ["type", "--delay", "12", "--", text], timeout=120)
        else:  # no room in the map: give each rebinding time to reach the app
            run(xd + ["type", "--delay", "120", "--", text], timeout=300)
    elif t == "key":
        run(xd + ["key", "--delay", "40"] + [xkey(k) for k in a["keys"]], timeout=30 + 0.15 * len(a["keys"]))
    elif t == "scroll":
        run(xd + ["mousemove", str(int(a["x"])), str(int(a["y"]))])
        dy, dx = int(a.get("dy", 0)), int(a.get("dx", 0))
        for button, n in ((5 if dy > 0 else 4, abs(dy)), (7 if dx > 0 else 6, abs(dx))):
            if n:
                run(xd + ["click", "--repeat", str(n), "--delay", "30", str(button)])
    else:
        raise ValueError(f"unknown input type {t!r}")
    time.sleep(0.25)
    return {"ok": True}


_BOUND: set[str] = set()  # characters given a keycode of their own on this display


def _keysym(ch: str, x11=None) -> int:
    """The keysym for a character: Latin-1 as itself; Cyrillic / Greek letters by their classic X names
    (Cyrillic_EM, Greek_alpha: toolkits know their case pairs, which they do not reliably for Unicode keysyms, and
    capitals typed with shift were lost); anything else as a Unicode keysym."""
    o = ord(ch)
    if 0x20 <= o <= 0x7E or 0xA0 <= o <= 0xFF:
        return o
    if x11 is not None:
        import unicodedata

        m = re.fullmatch(r"(CYRILLIC|GREEK) (CAPITAL|SMALL) LETTER (\w+)", unicodedata.name(ch, ""))
        if m:
            letter = m.group(3).lower() if m.group(2) == "SMALL" else m.group(3)
            sym = x11.XStringToKeysym(f"{m.group(1).title()}_{letter}".encode())
            if sym:
                return sym
    return 0x01000000 + o


# the non-ASCII characters the designs use, most used first (Cyrillic, punctuation, accents): bound before an app
# that reads the keymap only at start is launched
PRESET_CHARS = '’оаистнервклдпмубя–йыьгзшчж—ц“х”юфщэ•‘éі\xa0öї°ё«»єъç·₽€¼ø№½ѝäʼêü…îè\u2009â©ð―\xadàº£ß−û®‒'  # every one the task pack's designs use (79 keys)


def bind_chars(text: str, partial: bool = False) -> bool:
    """Give every character of `text` that the keyboard map lacks a keycode of its own, for good. xdotool otherwise
    rebinds one spare key per character and undoes it right after the keystroke; a GTK app (Inkscape) that reads the
    new map late gets a different character or none (Cyrillic came out as a few stray letters). Returns False when
    the map has no room left, and the caller types slowly instead."""
    import ctypes

    need = list(dict.fromkeys(c for c in text if ord(c) > 0x7E and c not in _BOUND and c not in "\n\t"))
    if not need:
        return True
    x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XGetKeyboardMapping.restype = ctypes.POINTER(ctypes.c_ulong)
    x11.XGetKeyboardMapping.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    x11.XChangeKeyboardMapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_ulong), ctypes.c_int]
    x11.XStringToKeysym.restype = ctypes.c_ulong
    x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
    for f in ("XSync", "XCloseDisplay", "XFree", "XDisplayKeycodes"):
        getattr(x11, f).argtypes = None
    dpy = x11.XOpenDisplay(DISPLAY.encode())
    if not dpy:
        return False
    try:
        lo, hi = ctypes.c_int(), ctypes.c_int()
        x11.XDisplayKeycodes(ctypes.c_void_p(dpy), ctypes.byref(lo), ctypes.byref(hi))
        per = ctypes.c_int()
        count = hi.value - lo.value + 1
        syms = x11.XGetKeyboardMapping(ctypes.c_void_p(dpy), lo.value, count, ctypes.byref(per))
        table = [[syms[k * per.value + j] for j in range(per.value)] for k in range(count)]
        x11.XFree(syms)
        present = {sym for row in table for sym in row}
        # spare keys: unbound ones, then the multimedia keys (XF86 keysyms) no desktop here uses
        free = [lo.value + k for k, row in enumerate(table) if not any(row)] + \
            [lo.value + k for k, row in enumerate(table) if any(row) and all(
                sym == 0 or 0x1008FF00 <= sym <= 0x1008FFFF for sym in row)]
        todo = [c for c in need if _keysym(c, x11) not in present]
        _BOUND.update(c for c in need if c not in todo)
        # a letter with a case goes on one key as a keyboard has it: lower case plain, capital with shift (a key
        # holding only the capital is read through XKB's case rules, and the capital is lost)
        keys, seen = [], set()
        for c in todo:
            lo, up = c.lower(), c.upper()
            pair = (lo, up) if len(lo) == 1 and len(up) == 1 and lo != up else (c, c)
            if pair not in seen:
                seen.add(pair)
                keys.append(pair)
        if len(keys) > len(free):
            if not partial:
                return False
            keys = keys[:len(free)]  # (most used first: as many as there are spare keys)
        for (lo, up), code in zip(keys, free, strict=False):
            arr = (ctypes.c_ulong * 2)(_keysym(lo, x11), _keysym(up, x11))
            x11.XChangeKeyboardMapping(ctypes.c_void_p(dpy), code, 2, arr, 1)
            _BOUND.update((lo, up))
        x11.XSync(ctypes.c_void_p(dpy), 0)
    finally:
        x11.XCloseDisplay(ctypes.c_void_p(dpy))
    time.sleep(0.3)  # let the apps take in the new map
    return True


def wait_window(pattern: str, timeout: float = 90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = run(["xdotool", "search", "--onlyvisible", "--name", pattern])
        if r.returncode == 0 and r.stdout.strip():
            return True
        time.sleep(0.5)
    return False


def fill_screen(pattern: str) -> None:
    """Give the app's main window (its largest visible one) the whole screen: an app restores its own last window
    size whatever the window manager says, so the agent would otherwise see a different layout per desktop."""
    best = None
    for wid in run(["xdotool", "search", "--onlyvisible", "--name", pattern]).stdout.split():
        m = re.search(r"Geometry: (\d+)x(\d+)", run(["xdotool", "getwindowgeometry", wid]).stdout)
        if m and (best is None or int(m[1]) * int(m[2]) > best[1]):
            best = (wid, int(m[1]) * int(m[2]))
    if best and best[1] < W * H * 0.95:
        run(["xdotool", "windowmove", best[0], "0", "0"])
        run(["xdotool", "windowsize", best[0], str(W), str(H)])
        time.sleep(0.5)


def dismiss(titles: tuple[str, ...]) -> int:
    """Close windows by title (startup dialogs an app shows no matter its settings); returns how many."""
    n = 0
    for title in titles:
        for wid in run(["xdotool", "search", "--name", title]).stdout.split():
            run(["xdotool", "windowactivate", "--sync", wid])
            run(["xdotool", "key", "--window", wid, "Escape"])
            run(["xdotool", "windowclose", wid])
            n += 1
    if n:
        time.sleep(0.5)
    return n


KILL = ["soffice.bin", "inkscape", "thunar"]  # the apps' processes, ended before every reset (plugins add theirs)
KILL_HOOKS: list = []  # plugins' kill()


def kill_apps():
    for name in KILL:
        run(["pkill", "-9", "-x", name])
    for hook in KILL_HOOKS:
        hook()
    time.sleep(0.5)


# ---------------------------------------------------------------------------------------------- assets


# file name (stem) -> asset id of the files the episode wrote: an app that keeps the file a picture came from (its
# name, not its pixels) is mapped back through this (a workspace's everyday names are not asset ids)
FILE_ASSET: dict[str, str] = {}
HOME_FOLDERS = ("Downloads", "Desktop", "Pictures", "Documents")
XDG = {"DESKTOP": "Desktop", "DOWNLOAD": "Downloads", "TEMPLATES": "Templates", "PUBLICSHARE": "Public",
       "DOCUMENTS": "Documents", "MUSIC": "Music", "PICTURES": "Pictures", "VIDEOS": "Videos"}


def xdg_home() -> None:
    """The usual home folders, registered as XDG user dirs and GTK bookmarks, so file windows list them in their
    sidebar (Documents, Downloads, Pictures, ...) as on any desktop. Done before the apps start."""
    for d in XDG.values():
        (HOME / d).mkdir(parents=True, exist_ok=True)
    cfg = HOME / ".config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "user-dirs.dirs").write_text("".join(f'XDG_{k}_DIR="$HOME/{v}"\n' for k, v in XDG.items()))
    (cfg / "gtk-3.0").mkdir(exist_ok=True)
    (cfg / "gtk-3.0" / "bookmarks").write_text("".join(f"file://{HOME}/{d} {d}\n"
                                                     for d in ("Documents", "Downloads", "Pictures", "Music", "Videos")))
    # the file manager (Thunar, where installed) shows pictures as large thumbnails in its icon view
    xf = cfg / "xfce4" / "xfconf" / "xfce-perchannel-xml"
    xf.mkdir(parents=True, exist_ok=True)
    (xf / "thunar.xml").write_text("""<?xml version="1.0" encoding="UTF-8"?>
<channel name="thunar" version="1.0">
  <property name="last-view" type="string" value="ThunarIconView"/>
  <property name="last-icon-view-zoom-level" type="string" value="THUNAR_ZOOM_LEVEL_150_PERCENT"/>
  <property name="misc-thumbnail-mode" type="string" value="THUNAR_THUMBNAIL_MODE_ALWAYS"/>
  <property name="misc-thumbnail-max-file-size" type="uint64" value="0"/>
  <property name="last-show-hidden" type="bool" value="false"/>
  <property name="last-side-pane" type="string" value="ThunarShortcutsPane"/>
  <property name="misc-single-click" type="bool" value="false"/>
</channel>
""")


def _aged(path: Path, key: str, days: tuple[float, float]) -> None:
    """Give a file a plausible modified time, `days` ago (seeded by `key`: the same file, the same age)."""
    import random as _r

    r = _r.Random(key)
    t = time.time() - r.uniform(*days) * 86400
    os.utime(path, (t, t))


def _padded(name: str, data: bytes) -> bytes:
    """A clutter document at an everyday size (a PDF of a few hundred kB, not 200 bytes); still a valid file."""
    import random as _r

    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    r = _r.Random(name)
    size = {"pdf": r.randint(40_000, 900_000), "docx": r.randint(12_000, 160_000), "xlsx": r.randint(9_000, 80_000),
            "zip": r.randint(200_000, 4_000_000)}.get(ext)
    if not size or len(data) >= size:
        return data
    if ext == "pdf":  # a comment after %%EOF is ignored by readers
        return data + b"%" + bytes(r.getrandbits(8) | 0x20 for _ in range(min(size - len(data) - 2, 64))) + \
            b"\n" + b"%" * max(0, size - len(data) - 67)
    return bytes(size - len(data)) + data  # zip-based: leading bytes are skipped (as a self-extractor's are)


def _graphic_crop(shape) -> list[float] | None:
    """A LibreOffice picture's crop (GraphicCrop, 1/100 mm of the picture's own size) as fractions cut off its
    left, top, right and bottom; None when it shows all of it."""
    try:
        c = shape.GraphicCrop
        if not (c.Left or c.Top or c.Right or c.Bottom):
            return None
        size = shape.Graphic.Size100thMM
        w, h = size.Width, size.Height
        if not w or not h:  # (a picture with no physical size: its pixels at 96 dpi)
            px = shape.Graphic.SizePixel
            w, h = px.Width * 2540 / 96, px.Height * 2540 / 96
        cut = [max(0.0, c.Left / w), max(0.0, c.Top / h), max(0.0, c.Right / w), max(0.0, c.Bottom / h)]
        return cut if cut[0] + cut[2] < 0.999 and cut[1] + cut[3] < 0.999 else None
    except Exception:  # noqa: BLE001
        return None


def asset_by_name(name: str, paths: dict) -> str | None:
    """The asset a file (a path, URI or layer name) was loaded from, by its name; None if it is not one."""
    stem = Path(str(name or "").replace("file://", "")).stem
    if stem in paths:
        return stem
    return FILE_ASSET.get(stem)


def write_assets(assets: dict[str, str], workspace: dict | None = None) -> dict[str, Path]:
    """Task images as files: by default PNGs named after their asset ids in ~/Pictures (where file dialogs open);
    with a workspace (core.workspace), under everyday names in its folder, among decoy pictures and documents."""
    from PIL import Image

    FILE_ASSET.clear()
    shutil.rmtree(PICS, ignore_errors=True)
    PICS.mkdir(parents=True)
    (HOME / ".local/share/recently-used.xbel").unlink(missing_ok=True)
    out = {}
    if not workspace:
        for aid, b64 in assets.items():
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            p = PICS / f"{aid}.png"
            img.save(p, "PNG")
            out[aid] = p
            FILE_ASSET[p.stem] = aid
        return out
    for d in HOME_FOLDERS:  # a fresh home: the usual folders, as on any desktop
        if d != "Pictures":
            shutil.rmtree(HOME / d, ignore_errors=True)
        (HOME / d).mkdir(parents=True, exist_ok=True)
    xdg_home()
    folder = HOME / workspace["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    for f in workspace["files"]:
        img = Image.open(io.BytesIO(base64.b64decode(assets[f["asset"]])))
        here = HOME / f["folder"] if f.get("folder") else folder
        here.mkdir(parents=True, exist_ok=True)
        p = here / f["name"]
        if p.suffix.lower() in (".jpg", ".jpeg"):
            bg = Image.new("RGB", img.size, (255, 255, 255))  # (a JPEG has no transparency)
            bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").getchannel("A"))
            bg.save(p, "JPEG", quality=92)
        else:
            img.save(p, "PNG")
        out[f["asset"]] = p
        FILE_ASSET[p.stem] = f["asset"]
        # the design's own pictures came in lately; other pictures have been around for a while
        _aged(p, f["name"], (0.02, 1.5) if not re.fullmatch(r"x\d+", f["asset"]) else (2, 90))
    for o in workspace.get("others") or []:
        p = HOME / o["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_padded(p.name, base64.b64decode(o["b64"])))
        _aged(p, o["path"], (1, 120))
    # the desktop's own machinery (this agent, its fonts, app installs) is not part of anyone's home: file windows
    # that honour ~/.hidden (GTK) leave it out
    keep = set(HOME_FOLDERS) | set(XDG.values()) | {Path(workspace["folder"]).parts[0]} | {
        Path(o["path"]).parts[0] for o in workspace.get("others") or []}
    (HOME / ".hidden").write_text("".join(f"{q.name}\n" for q in sorted(HOME.iterdir())
                                          if not q.name.startswith(".") and q.name not in keep))
    return out


class Matcher:
    """Map an image an app embedded (no filename left) back to the task asset it came from, by pixels."""

    def __init__(self, paths: dict[str, Path]):
        from PIL import Image

        self.sig, self.size = {}, {}
        for aid, p in paths.items():
            img = Image.open(p)
            self.sig[aid], self.size[aid] = self._sig(img), img.size

    @staticmethod
    def _sig(img):
        import numpy as np

        return np.asarray(img.convert("RGBA").resize((24, 24)), dtype="float32") / 255.0

    def best(self, img) -> str | None:
        """Embedded images keep their pixel size, so an asset of exactly that size wins; otherwise the closest look,
        with the aspect ratio counted too (flat colour bars of one colour differ only in their proportions)."""
        import numpy as np

        s = self._sig(img)
        same = [aid for aid, sz in self.size.items() if sz == img.size]
        cands = same or list(self.sig)

        def score(aid):
            (w, h), (W, H) = img.size, self.size[aid]
            return float(np.abs(s - self.sig[aid]).mean()) + 0.5 * abs(math.log((w / max(h, 1)) / (W / max(H, 1))))

        scores = {aid: score(aid) for aid in cands}
        aid = min(scores, key=scores.get) if scores else None
        return aid if aid is not None and scores[aid] < 0.08 else None


# ---------------------------------------------------------------------------------------------- Impress (UNO)


MM100_PER_PX = 2540 / 96  # 1/100 mm per CSS pixel


class Impress:
    """LibreOffice Impress, one blank slide the size of the canvas; read back through the UNO API."""

    # what differs between LibreOffice's presentation and drawing apps (Draw subclasses this)
    LAUNCH, FACTORY, FILTER, EXT, TITLE = "--impress", "private:factory/simpress", "impress8", "odp", "Impress"

    def __init__(self):
        self.ctx = self.doc = None

    def _connect(self):
        import uno

        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        for _ in range(120):
            try:
                self.ctx = resolver.resolve("uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext")
                return
            except Exception:  # noqa: BLE001 - soffice still starting
                time.sleep(0.5)
        raise RuntimeError("LibreOffice did not open its UNO socket")

    def _desktop(self):
        return self.ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", self.ctx)

    PROFILE = Path("/tmp/lo-profile")
    # a fresh profile per episode: no stale lock from a killed instance, no first-run dialogs, the same UI every time
    SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema">
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="ShowTipOfTheDay" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="FirstRun" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Setup/Office"><prop oor:name="ooSetupInstCompleted" oor:op="fuse"><value>true</value></prop></item>
<item oor:path="/org.openoffice.Office.Impress/Misc"><prop oor:name="StartWithTemplate" oor:op="fuse"><value>false</value></prop></item>
</oor:items>"""

    def reset(self, width, height, background, paths):
        shutil.rmtree(self.PROFILE, ignore_errors=True)
        (self.PROFILE / "user").mkdir(parents=True)
        (self.PROFILE / "user" / "registrymodifications.xcu").write_text(self.SETTINGS)
        subprocess.Popen(["soffice", f"-env:UserInstallation={self.PROFILE.as_uri()}", self.LAUNCH, "--norestore",
                          "--nologo", "--nodefault", "--accept=socket,host=127.0.0.1,port=2002;urp;"],
                         env={**os.environ, "DISPLAY": DISPLAY, "SAL_USE_VCLPLUGIN": "gtk3"},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._connect()
        from com.sun.star.beans import PropertyValue  # type: ignore[import-not-found]

        self.doc = self._desktop().loadComponentFromURL(self.FACTORY, "_blank", 0, ())
        page = self.doc.DrawPages.getByIndex(0)
        if self.EXT == "odp":
            page.Layout = 20  # blank layout: no title / content placeholders
        while page.Count:
            page.remove(page.getByIndex(0))
        page.Width, page.Height = round(width * MM100_PER_PX), round(height * MM100_PER_PX)
        page.BorderLeft = page.BorderRight = page.BorderTop = page.BorderBottom = 0
        self._set_background(page, background)
        # LibreOffice sizes its work area (where objects may be placed or sized to) from the page the document
        # was opened with: grown through the API, the page is bigger but the work area is not, and the dialogs
        # then clamp anything below or right of the default slide. Reopen it so the work area is this page's.
        url = (WORK / f"design.{self.EXT}").as_uri()
        self.doc.storeToURL(url, (pv("FilterName", self.FILTER),))
        self.doc = self.doc.getCurrentController().getFrame().loadComponentFromURL(url, "_self", 0, ())
        self.paths = paths
        self.matcher = Matcher(paths)
        self.size = (width, height)
        del PropertyValue
        wait_window(self.TITLE, 60)
        time.sleep(1.5)
        fill_screen(self.TITLE)

    def _set_background(self, page, color):
        bg = self.doc.createInstance("com.sun.star.drawing.Background")
        bg.FillStyle = uno_enum("com.sun.star.drawing.FillStyle", "SOLID")
        bg.FillColor = int(color.lstrip("#"), 16)
        page.Background = bg

    def _graphic_asset(self, graphic) -> str | None:
        """Which task asset a placed graphic is: LibreOffice keeps the file it was loaded from (OriginURL);
        otherwise export it to a file and match it by pixels."""
        from PIL import Image

        origin = ""
        try:
            origin = graphic.OriginURL or ""
        except Exception:  # noqa: BLE001
            pass
        aid = asset_by_name(origin, self.paths) if origin else None
        if aid is not None:
            return aid
        gp = self.ctx.ServiceManager.createInstanceWithContext("com.sun.star.graphic.GraphicProvider", self.ctx)
        tmp = WORK / "graphic-export.png"
        tmp.unlink(missing_ok=True)
        gp.storeGraphic(graphic, (pv("URL", tmp.as_uri()), pv("MimeType", "image/png")))
        return self.matcher.best(Image.open(tmp)) if tmp.exists() and tmp.stat().st_size else None

    def read(self):

        page = self.doc.DrawPages.getByIndex(0)
        w, h = self.size
        bg = "#ffffff"
        try:
            b = page.Background
            if b is not None and int(b.FillColor) >= 0:
                bg = f"#{int(b.FillColor) & 0xFFFFFF:06x}"
        except Exception:  # noqa: BLE001
            pass
        els, skipped = [], 0

        def flat(container):  # groups are flattened; their members keep page coordinates
            for i in range(container.Count):
                s = container.getByIndex(i)
                if s.ShapeType == "com.sun.star.drawing.GroupShape":
                    yield from flat(s)
                else:
                    yield s

        for s in flat(page):
            kind = s.ShapeType
            if kind == "com.sun.star.drawing.CustomShape":  # what the Basic Shapes toolbar draws
                geo = {p.Name: p.Value for p in s.CustomShapeGeometry}.get("Type", "")
                kind = ("com.sun.star.drawing.EllipseShape" if geo in ("ellipse", "circle") else
                        "com.sun.star.drawing.RectangleShape" if geo in ("rectangle", "round-rectangle", "square",
                                                                          "round-square", "quadrat") else kind)
            common = get_geom(s)
            try:  # a picture's own transparency; other shapes' fill transparency
                tr = s.Transparency if kind == "com.sun.star.drawing.GraphicObjectShape" else s.FillTransparence
                common["opacity"] = 1 - (tr or 0) / 100.0
            except Exception:  # noqa: BLE001
                pass
            if kind == "com.sun.star.drawing.GraphicObjectShape":
                aid = self._graphic_asset(s.Graphic)
                if aid is None:
                    skipped += 1
                    continue
                el = {"kind": "image", "asset_id": aid, **common}
                crop = _graphic_crop(s)
                if crop:
                    el["crop"] = crop
                els.append(el)
            elif kind in ("com.sun.star.drawing.TextShape", "com.sun.star.presentation.TitleTextShape",
                          "com.sun.star.presentation.SubtitleShape", "com.sun.star.presentation.OutlinerShape"):
                text = s.getString().replace("\r\n", "\n").replace("\r", "\n")
                if not text.strip():
                    skipped += 1
                    continue
                els.append({"kind": "text", **_read_text(s, bg), **common})
            elif kind in ("com.sun.star.drawing.RectangleShape", "com.sun.star.drawing.EllipseShape"):
                fill = f"#{int(s.FillColor) & 0xFFFFFF:06x}"
                els.append({"kind": "shape", "shape": "ellipse" if "Ellipse" in kind else "rect", "fill": fill,
                            **common})
            else:
                skipped += 1
        return {"width": w, "height": h, "background": bg, "elements": els, "skipped": skipped}

    def expert(self, doc):
        page = self.doc.DrawPages.getByIndex(0)
        self._set_background(page, doc.get("background", "#ffffff"))
        for e in doc["elements"]:
            if e["kind"] == "image":
                s = self.doc.createInstance("com.sun.star.drawing.GraphicObjectShape")
                page.add(s)
                gp = self.ctx.ServiceManager.createInstanceWithContext("com.sun.star.graphic.GraphicProvider",
                                                                       self.ctx)
                s.Graphic = gp.queryGraphic((pv("URL", self.paths[e["asset_id"]].as_uri()),))
            elif e["kind"] == "shape":
                s = self.doc.createInstance("com.sun.star.drawing." + ("EllipseShape" if e.get("shape") == "ellipse"
                                                                       else "RectangleShape"))
                page.add(s)
                s.FillColor = int((e.get("fill") or "#000000").lstrip("#"), 16)
                s.LineStyle = uno_enum("com.sun.star.drawing.LineStyle", "NONE")
            else:
                s = self.doc.createInstance("com.sun.star.drawing.TextShape")
                page.add(s)
                s.TextAutoGrowWidth = True
                s.TextAutoGrowHeight = True
                _write_text(s, e)
                # the box is the design's width (or the text's natural width if wider, so nothing wraps), placed
                # so the alignment edge (the anchor) is where it should be; its height follows the lines
                set_geom(s, {**e, "angle": 0})
                w = max(e["width"], s.Size.Width / MM100_PER_PX)
                s.TextAutoGrowWidth = False
                e = {**e, "x": e["x"] + {"left": 0, "center": 0.5, "right": 1}[e.get("align", "left")] * (
                    e["width"] - w), "width": w}
                set_geom(s, {**e, "angle": 0})
                e = {**e, "height": s.Size.Height / MM100_PER_PX}
            if e["kind"] == "image" and e.get("opacity", 1) < 1:
                s.Transparency = round((1 - e["opacity"]) * 100)
            set_geom(s, e)
        time.sleep(0.5)



class Draw(Impress):
    """LibreOffice Draw: one blank page the size of the canvas; same document model and read-back as Impress."""

    LAUNCH, FACTORY, FILTER, EXT, TITLE = "--draw", "private:factory/sdraw", "draw8", "odg", "Draw"

def _luminance(color: str) -> float:
    """Relative luminance of an sRGB colour (WCAG: channels linearised, 0.2126 R + 0.7152 G + 0.0722 B), 0..1."""
    v = int(color.lstrip("#")[:6], 16)

    def lin(c: int) -> float:
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(v >> 16 & 0xFF) + 0.7152 * lin(v >> 8 & 0xFF) + 0.0722 * lin(v & 0xFF)


AUTO_DARK = 0.345  # (measured on LibreOffice 25.2.3: #9e9e9e / #00b800, 0.343, draw white; #9f9f9f / #00bc00, 0.347, black)


def _auto_color(background: str) -> str:
    """The colour LibreOffice draws text set to Automatic (CharColor -1) with over `background`: white on a dark
    one, black otherwise. Measured on LibreOffice 25.2.3 (Impress and Draw, 40 page colours): the boundary is the
    background's relative luminance (sRGB linearised) at about 0.345, not its 8-bit luminance (pure green, 150 of
    255, draws black; grey 158 draws white). Only the page's, or the text shape's own, fill colour counts: a
    picture under the text does not."""
    return "#ffffff" if _luminance(background) < AUTO_DARK else "#000000"


def _hexcolor(v, auto: str = "#000000") -> str:
    """A UNO colour as hex; a negative value is Automatic, read as the colour it is drawn with (`auto`)."""
    return f"#{int(v) & 0xFFFFFF:06x}" if int(v) >= 0 else auto


def _style_of(r, auto: str = "#000000") -> dict:
    return {"color": _hexcolor(r.CharColor, auto), "bold": float(r.CharWeight) >= 150,
            "italic": _ev(r.CharPosture) not in (0, "NONE")}


def _text_background(shape, page_background: str) -> str:
    """What is behind a text shape's letters for the Automatic colour: the shape's own solid fill (LibreOffice
    draws the text over it; not measured here, the env's text boxes have none), else the page."""
    try:
        if _ev(shape.FillStyle) in (1, "SOLID") and int(shape.FillColor) >= 0:
            return f"#{int(shape.FillColor) & 0xFFFFFF:06x}"
    except Exception:  # noqa: BLE001
        pass
    return page_background


def _read_text(shape, page_background: str = "#ffffff") -> dict:
    """A text shape's string, typography and styled runs (one run per portion that differs in colour/weight).
    Text left on the Automatic colour reads as the colour LibreOffice draws it with over `page_background`."""
    auto = _auto_color(_text_background(shape, page_background))
    paras = shape.Text.createEnumeration()
    runs, first_para = [], None
    while paras.hasMoreElements():
        para = paras.nextElement()
        if first_para is None:
            first_para = para
        elif runs:
            runs.append({"text": "\n", **{k: runs[-1][k] for k in ("color", "bold", "italic")}})
        portions = para.createEnumeration()
        while portions.hasMoreElements():
            por = portions.nextElement()
            t = por.getString()
            if t:
                runs.append({"text": t, **_style_of(por, auto)})
    text = "".join(r["text"] for r in runs)
    cur = shape.Text.createTextCursor()
    cur.gotoStart(False)
    font_px = float(cur.CharHeight) * 96 / 72
    ls = getattr(first_para, "ParaLineSpacing", None) if first_para is not None else None
    mode = _ev(ls.Mode) if ls is not None else 0
    if mode in (3, "FIX") and ls.Height:  # a fixed line height in 1/100 mm
        lh = ls.Height / MM100_PER_PX / font_px
    elif mode in (0, "PROP") and ls is not None:  # proportional: 100% is the font's own line height (~1.17 em)
        lh = ls.Height / 100.0 * 1.17
    else:
        lh = 1.17
    adjust = {0: "left", 1: "right", 3: "center", "LEFT": "left", "RIGHT": "right", "CENTER": "center"}.get(
        _ev(getattr(first_para, "ParaAdjust", 0)), "left")
    styles = {(r["color"], r["bold"], r["italic"]) for r in runs if r["text"].strip()}
    return {"text": text, "font_family": cur.CharFontName, "font_size": font_px, **(_style_of(cur, auto)),
            "underline": int(_ev(cur.CharUnderline) or 0) != 0, "align": adjust, "line_height": lh,
            "letter_spacing": float(getattr(cur, "CharKerning", 0) or 0) / MM100_PER_PX,
            "uppercase": _ev(getattr(cur, "CharCaseMap", 0)) in (1, "UPPERCASE"),
            "runs": runs if len(styles) > 1 else None}


def _write_text(shape, e: dict) -> None:
    import uno

    shape.setString(e["text"])
    font_px = float(e.get("font_size", 48))
    cur = shape.Text.createTextCursor()
    cur.gotoStart(False)
    cur.gotoEnd(True)
    cur.CharFontName = e.get("font_family") or "DejaVu Sans"
    cur.CharHeight = font_px * 72 / 96
    cur.CharColor = int((e.get("color") or "#000000").lstrip("#"), 16)
    cur.CharWeight = 150.0 if e.get("bold") else 100.0
    cur.CharPosture = uno.Enum("com.sun.star.awt.FontSlant", "ITALIC" if e.get("italic") else "NONE")
    cur.CharKerning = round(float(e.get("letter_spacing", 0)) * MM100_PER_PX)
    cur.CharCaseMap = 1 if e.get("uppercase") else 0
    cur.ParaAdjust = {"left": 0, "right": 1, "center": 3}[e.get("align", "left")]
    spacing = uno.createUnoStruct("com.sun.star.style.LineSpacing")
    spacing.Mode, spacing.Height = 3, round(float(e.get("line_height", 1.0)) * font_px * MM100_PER_PX)
    cur.ParaLineSpacing = spacing
    pos = 0
    for run in e.get("runs") or []:  # colour / weight per styled stretch
        n = len(run["text"])
        c = shape.Text.createTextCursor()
        c.gotoStart(False)
        c.goRight(pos, False)
        c.goRight(n, True)
        if run.get("color"):
            c.CharColor = int(run["color"].lstrip("#"), 16)
        if run.get("bold") is not None:
            c.CharWeight = 150.0 if run["bold"] else 100.0
        if run.get("italic") is not None:
            c.CharPosture = uno.Enum("com.sun.star.awt.FontSlant", "ITALIC" if run["italic"] else "NONE")
        pos += n


def _ev(v):
    """A UNO enum's name, or the plain value (UNO hands back some properties as enums, some as ints)."""
    return v.value if hasattr(v, "value") and not isinstance(v, (int, float, str)) else v


def pv(name, value):
    import uno

    p = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    p.Name, p.Value = name, value
    return p


def uno_enum(typ, value):
    import uno

    return uno.Enum(typ, value)


def set_geom(shape, e):
    """Unrotated box (px) + clockwise angle about its centre -> the shape's transformation (1/100 mm)."""
    import uno

    k = MM100_PER_PX
    th = math.radians(float(e.get("angle") or 0))
    w, h = e["width"] * k, e["height"] * k
    a, b, c, d = w * math.cos(th), w * math.sin(th), -h * math.sin(th), h * math.cos(th)
    cx, cy = (e["x"] + e["width"] / 2) * k, (e["y"] + e["height"] / 2) * k
    def line(*v):  # pyuno hands nested structs back by value: build each row, then assign it
        r = uno.createUnoStruct("com.sun.star.drawing.HomogenMatrixLine3")
        r.Column1, r.Column2, r.Column3 = v
        return r

    m = uno.createUnoStruct("com.sun.star.drawing.HomogenMatrix3")
    m.Line1, m.Line2, m.Line3 = line(a, c, cx - (a + c) / 2), line(b, d, cy - (b + d) / 2), line(0.0, 0.0, 1.0)
    shape.Transformation = m


def get_geom(shape) -> dict:
    """The shape's transformation -> unrotated box (px) + clockwise angle about its centre. (Position and Size
    alone are not enough: a rotated shape's Position is the corner of its bounding box.)"""
    t, k = shape.Transformation, MM100_PER_PX
    a, c, e = t.Line1.Column1, t.Line1.Column2, t.Line1.Column3
    b, d, f = t.Line2.Column1, t.Line2.Column2, t.Line2.Column3
    w, h = math.hypot(a, b), math.hypot(c, d)
    cx, cy = e + (a + c) / 2, f + (b + d) / 2
    return {"x": (cx - w / 2) / k, "y": (cy - h / 2) / k, "width": max(w / k, 0.5), "height": max(h / k, 0.5),
            "angle": math.degrees(math.atan2(b, a))}


# ---------------------------------------------------------------------------------------------- Inkscape (SVG)


SVG = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"


class Inkscape:
    """Inkscape with an SVG page the size of the canvas; read back by saving (Ctrl+S) and parsing the SVG."""

    path = WORK / "design.svg"

    def reset(self, width, height, background, paths):
        self.size, self.paths, self.matcher = (width, height), paths, Matcher(paths)
        # Inkscape remembers docked dialogs, the last folder and import choices: start every episode from defaults
        shutil.rmtree(HOME / ".config/inkscape", ignore_errors=True)
        self.path.write_text(_svg(width, height, background, []))
        subprocess.Popen(["inkscape", str(self.path)], env={**os.environ, "DISPLAY": DISPLAY},
                         stdout=subprocess.DEVNULL, stderr=open("/tmp/inkscape.log", "ab"))  # its crash backtraces
        wait_window("Inkscape", 120)
        time.sleep(4)
        fill_screen("Inkscape")

    def save(self):
        before = self.path.stat().st_mtime
        win = run(["xdotool", "search", "--onlyvisible", "--name", "Inkscape"]).stdout.split()
        if win:
            run(["xdotool", "windowactivate", "--sync", win[-1]])
        run(["xdotool", "key", "ctrl+s"])
        for _ in range(40):
            time.sleep(0.25)
            if self.path.stat().st_mtime != before:
                break
        time.sleep(0.5)

    def read(self):
        self.save()
        text = self.path.read_text()
        out = parse_svg(text, self.size, self.matcher)
        # where the canvas is on screen (zoom, and the document point at the view's centre), for scripted recorders
        nv = ET.fromstring(text).find("{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}namedview")
        ink = "{http://www.inkscape.org/namespaces/inkscape}"
        if nv is not None and nv.get(ink + "zoom"):
            out["view"] = {k: float(nv.get(ink + k)) for k in ("zoom", "cx", "cy") if nv.get(ink + k)}
        return out

    def expert(self, doc):
        # write the design as SVG and reopen it in Inkscape (then a read saves through the app and re-parses)
        kill_apps()
        self.path.write_text(_svg(*self.size, doc.get("background", "#ffffff"), doc["elements"], self.paths))
        subprocess.Popen(["inkscape", str(self.path)], env={**os.environ, "DISPLAY": DISPLAY},
                         stdout=subprocess.DEVNULL, stderr=open("/tmp/inkscape.log", "ab"))  # its crash backtraces
        wait_window("Inkscape", 120)
        time.sleep(4)
        fill_screen("Inkscape")


def _svg(width, height, background, elements, paths=None) -> str:
    parts = []
    for e in elements:
        tr = ""
        if e.get("angle"):
            cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
            tr = f' transform="rotate({e["angle"]} {cx} {cy})"'
        op = f' opacity="{e.get("opacity", 1)}"' if e.get("opacity", 1) != 1 else ""
        if e["kind"] == "image":
            parts.append(f'<image x="{e["x"]}" y="{e["y"]}" width="{e["width"]}" height="{e["height"]}" '
                         f'preserveAspectRatio="none" xlink:href="{paths[e["asset_id"]].as_uri()}"{tr}{op}/>')
        elif e["kind"] == "shape":
            tag = "ellipse" if e.get("shape") == "ellipse" else "rect"
            geo = (f'cx="{e["x"] + e["width"] / 2}" cy="{e["y"] + e["height"] / 2}" rx="{e["width"] / 2}" '
                   f'ry="{e["height"] / 2}"' if tag == "ellipse" else
                   f'x="{e["x"]}" y="{e["y"]}" width="{e["width"]}" height="{e["height"]}"')
            parts.append(f'<{tag} {geo} fill="{e.get("fill") or "#000000"}"{tr}{op}/>')
        else:
            fs = float(e.get("font_size", 48))
            step = float(e.get("_step") or float(e.get("line_height", 1.0)) * fs)
            base = float(e.get("_baseline") or e["y"] + fs * 0.8)
            align = e.get("align", "left")
            anchor = {"left": "start", "center": "middle", "right": "end"}[align]
            ax = float(e.get("_anchor_x") or {"left": e["x"], "center": e["x"] + e["width"] / 2,
                                               "right": e["x"] + e["width"]}[align])
            style = (f'font-family:{e.get("font_family") or "DejaVu Sans"};font-size:{fs}px;'
                     f'font-weight:{"bold" if e.get("bold") else "normal"};'
                     f'font-style:{"italic" if e.get("italic") else "normal"};fill:{e.get("color") or "#000000"};'
                     f'text-anchor:{anchor};letter-spacing:{float(e.get("letter_spacing", 0))}px;'
                     f'line-height:{step / fs if fs else 1.0};'
                     f'text-transform:{"uppercase" if e.get("uppercase") else "none"}')
            spans = "".join(f'<tspan x="{ax}" y="{base + k * step}" sodipodi:role="line">{_line_runs(line)}</tspan>'
                            for k, line in enumerate(_lines_of(e)))
            parts.append(f'<text xml:space="preserve" style="{style}"{tr}{op}>{spans}</text>')
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
            f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{width}px" height="{height}px" '
            f'viewBox="0 0 {width} {height}"><sodipodi:namedview pagecolor="{background}" inkscape:pageopacity="1"/>'
            f'<rect id="canvas-background" x="0" y="0" width="{width}" height="{height}" fill="{background}" '
            f'sodipodi:insensitive="true"/><g inkscape:groupmode="layer" inkscape:label="Design" id="design">'
            + "".join(parts) + "</g></svg>")


def _lines_of(e: dict) -> list[list[dict]]:
    """A text element as lines of styled pieces (its runs, or one piece per line)."""
    runs = e.get("runs") or [{"text": e["text"]}]
    lines: list[list[dict]] = [[]]
    for r in runs:
        for k, part in enumerate(r["text"].split("\n")):
            if k:
                lines.append([])
            if part:
                lines[-1].append({**r, "text": part})
    return lines


def _line_runs(pieces: list[dict]) -> str:
    out = []
    for p in pieces:
        st = []
        if p.get("color"):
            st.append(f"fill:{p['color']}")
        if p.get("bold") is not None:
            st.append(f"font-weight:{'bold' if p['bold'] else 'normal'}")
        if p.get("italic") is not None:
            st.append(f"font-style:{'italic' if p['italic'] else 'normal'}")
        out.append(f'<tspan style="{";".join(st)}">{_esc(p["text"])}</tspan>' if st else _esc(p["text"]))
    return "".join(out)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _matrix(tr: str | None):
    """An SVG transform attribute as a 2x3 affine (a, b, c, d, e, f)."""
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", tr or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix":
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            r = math.radians(v[0])
            c, s = math.cos(r), math.sin(r)
            t = (c, s, -s, c, 0, 0)
            if len(v) == 3:
                t = _mul(_mul((1, 0, 0, 1, v[1], v[2]), t), (1, 0, 0, 1, -v[1], -v[2]))
        else:
            continue
        m = _mul(m, t)
    return m


def _mul(m, n):
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2, a * e2 + c * f2 + e,
            b * e2 + d * f2 + f)


def _place(m, x, y, w, h):
    """Box (x, y, w, h) under affine m -> DesignDoc geometry (unrotated box + angle about its centre)."""
    a, b, c, d, e, f = m
    sx, sy = math.hypot(a, b), math.hypot(c, d)
    angle = math.degrees(math.atan2(b, a))
    cx, cy = x + w / 2, y + h / 2
    ccx, ccy = a * cx + c * cy + e, b * cx + d * cy + f
    W2, H2 = w * sx, h * sy
    return {"x": ccx - W2 / 2, "y": ccy - H2 / 2, "width": max(W2, 0.5), "height": max(H2, 0.5), "angle": angle}


def _style(el) -> dict:
    st = dict(kv.split(":", 1) for kv in (el.get("style") or "").split(";") if ":" in kv)
    for k in ("fill", "font-family", "font-size", "font-weight", "font-style", "text-anchor", "opacity"):
        if el.get(k) is not None and k not in st:
            st[k] = el.get(k)
    return {k.strip(): v.strip() for k, v in st.items()}


def _num(v, default=0.0) -> float:
    m = re.match(r"-?[\d.]+", str(v or ""))
    return float(m.group()) if m else default


def parse_svg(text: str, size, matcher: Matcher) -> dict:
    from PIL import Image

    root = ET.fromstring(text)
    w, h = size
    els, skipped = [], 0
    bg = "#ffffff"
    nv = root.find("{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}namedview")
    if nv is not None and nv.get("pagecolor"):
        bg = nv.get("pagecolor")

    def walk(node, m, inherited):
        nonlocal skipped, bg
        for el in node:
            tag = el.tag.replace(SVG, "")
            mm = _mul(m, _matrix(el.get("transform")))
            st = {**inherited, **_style(el)}
            op = float(st.get("opacity", 1) or 1)
            if tag == "g":
                walk(el, mm, st)
            elif el.get("id") == "canvas-background":
                bg = st.get("fill", bg)
            elif tag == "image":
                href = el.get(XLINK + "href") or el.get("href") or ""
                aid = None
                if href.startswith("data:"):
                    aid = matcher.best(Image.open(io.BytesIO(base64.b64decode(href.split(",", 1)[1]))))
                else:
                    aid = asset_by_name(href, matcher.sig)
                if aid is None:
                    skipped += 1
                    continue
                els.append({"kind": "image", "asset_id": aid, "opacity": op,
                            **_place(mm, _num(el.get("x")), _num(el.get("y")), _num(el.get("width")),
                                     _num(el.get("height")))})
            elif tag in ("rect", "ellipse", "circle"):
                if tag == "rect":
                    geo = (_num(el.get("x")), _num(el.get("y")), _num(el.get("width")), _num(el.get("height")))
                else:
                    rx = _num(el.get("rx") or el.get("r"))
                    ry = _num(el.get("ry") or el.get("r"))
                    geo = (_num(el.get("cx")) - rx, _num(el.get("cy")) - ry, 2 * rx, 2 * ry)
                fill = st.get("fill", "#000000")
                if not fill.startswith("#"):
                    skipped += 1
                    continue
                els.append({"kind": "shape", "shape": "rect" if tag == "rect" else "ellipse", "fill": fill,
                            "opacity": op, **_place(mm, *geo)})
            elif tag == "text":
                spans = [t for t in el if t.tag.replace(SVG, "") == "tspan"] or [el]
                s0 = {**st, **_style(spans[0])}
                fs = _num(s0.get("font-size"), 16)
                runs = []
                for k, line in enumerate(spans):
                    if k and runs:
                        runs.append({**{q: runs[-1][q] for q in ("color", "bold", "italic")}, "text": "\n"})
                    lst = {**st, **_style(line)}  # a line inherits from the text, not from the line before it
                    pieces = [(line.text or "", lst)] + [
                        p for c in line for p in (("".join(c.itertext()), {**lst, **_style(c)}), (c.tail or "", lst))]
                    for txt, pst in pieces:
                        if txt:
                            runs.append({"text": txt, "color": pst.get("fill", "#000000"),
                                         "bold": pst.get("font-weight", "normal") in ("bold", "600", "700", "800", "900"),
                                         "italic": pst.get("font-style") == "italic"})
                text = "".join(r["text"] for r in runs)
                if not text.strip():
                    continue
                a, b, c, d, e2, f2 = mm
                xs, ys = _num(spans[0].get("x") or el.get("x")), _num(spans[0].get("y") or el.get("y"))
                lh_style = s0.get("line-height", "")
                lh_style = (_num(lh_style) / 100 if lh_style.endswith("%") else _num(lh_style) / fs
                            if lh_style.endswith("px") else _num(lh_style, 1.25)) if lh_style else 1.25
                step = (_num(spans[1].get("y")) - ys) if len(spans) > 1 and spans[1].get("y") else fs * lh_style
                anchor = s0.get("text-anchor", "start")
                styles = {(r["color"], r["bold"], r["italic"]) for r in runs if r["text"].strip()}
                els.append({"kind": "text", "text": text, "font_family": s0.get("font-family", "").strip("'\""),
                            "font_size": fs * math.hypot(a, b), "color": runs[0]["color"] if runs else "#000000",
                            "bold": runs[0]["bold"] if runs else False, "italic": runs[0]["italic"] if runs else False,
                            "align": {"start": "left", "middle": "center", "end": "right"}.get(anchor, "left"),
                            "line_height": step / fs if fs else 1.0, "opacity": op,
                            "letter_spacing": _num(s0.get("letter-spacing"), 0.0),
                            "uppercase": s0.get("text-transform") == "uppercase",
                            "runs": runs if len(styles) > 1 else None,
                            "angle": math.degrees(math.atan2(b, a)),
                            **({"_anchor_x": xs, "_baseline": ys, "_matrix": [a, b, c, d, e2, f2]}
                               if abs(b) > 1e-6 or a < 0 else  # rotated: the box turns about its centre
                               {"_anchor_x": a * xs + c * ys + e2, "_baseline": b * xs + d * ys + f2})})
            elif tag in ("defs", "namedview", "metadata", "title", "desc", "style"):
                continue
            elif tag.startswith("{"):
                continue
            else:
                skipped += 1

    walk(root, (1, 0, 0, 1, 0, 0), {})
    return {"width": w, "height": h, "background": bg, "elements": els, "skipped": skipped}


APPS = {"impress": Impress, "draw": Draw, "inkscape": Inkscape}
ROUTES: dict = {}  # plugins' extra POST routes
STATE: dict = {"app": None}
APP_DIR = Path(os.environ.get("DESKTOP_AGENT_APPS", Path(__file__).resolve().parent / "apps"))


def load_app_modules(folder: Path = APP_DIR) -> list[str]:
    """Load the plugin modules in `folder` (see the module docstring); returns their names."""
    sys.modules.setdefault("agent", sys.modules[__name__])  # run as a script, this module is __main__
    names = []
    for path in sorted(p for p in folder.glob("*.py") if not p.name.startswith("_")) if folder.is_dir() else []:
        spec = importlib.util.spec_from_file_location(f"agent_apps.{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        APPS.update(getattr(mod, "APPS", {}))
        ROUTES.update(getattr(mod, "ROUTES", {}))
        KILL.extend(n for n in getattr(mod, "KILL", ()) if n not in KILL)
        if hasattr(mod, "kill"):
            KILL_HOOKS.append(mod.kill)
        if hasattr(mod, "fonts_changed"):
            FONT_HOOKS.append(mod.fonts_changed)
        names.append(path.stem)
    return names


# ---------------------------------------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/health":
                key = Path(os.environ.get("DESIGN_CANVAS_FONTS", "/fonts")) / ".key"
                self._send(200, {"ok": True, "app": type(STATE["app"]).__name__ if STATE["app"] else None,
                                 "fonts": key.read_text().strip() if key.exists() else None})
            elif self.path == "/fonts.tgz":  # this desktop's fonts, for another desktop of the pool to copy
                src = Path(os.environ.get("DESIGN_CANVAS_FONTS", "/fonts"))
                r = subprocess.run(["tar", "czf", "-", "-C", str(src), "."], capture_output=True, timeout=300)
                self._send(200 if r.returncode == 0 else 500, r.stdout if r.returncode == 0 else
                           {"error": r.stderr.decode(errors="replace")[-500:]}, "application/gzip")
            elif self.path.split("?")[0] == "/screenshot":
                fmt, quality = _shot_args(self.path)
                self._send(200, screenshot(fmt, quality), f"image/{fmt}")
            elif self.path == "/a11y":
                with LOCK:
                    self._send(200, a11y())
            elif self.path.split("?")[0] == "/observe":  # a recorder's turn in one round trip: the widgets, then the screen
                fmt, quality = _shot_args(self.path)
                with LOCK:
                    nodes = a11y()["nodes"]
                    shot = screenshot(fmt, quality)
                # ("png" keeps its name for older clients; the bytes are whatever fmt was asked for)
                self._send(200, {"nodes": nodes, "png": base64.b64encode(shot).decode(), "format": fmt})
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            with LOCK:
                if self.path == "/input":
                    out = do_input(body)
                elif self.path == "/reset":
                    kill_apps()
                    if _BOUND:  # back to the plain map, so characters bound for past episodes never fill it
                        run(["setxkbmap", "us"])
                        _BOUND.clear()
                    paths = write_assets(body.get("assets") or {}, body.get("workspace"))
                    app = APPS[body["app"]]()
                    app.reset(int(body["width"]), int(body["height"]), body.get("background", "#ffffff"), paths)
                    STATE["app"] = app
                    out = {"ok": True}
                elif self.path == "/read":
                    out = STATE["app"].read()
                elif self.path == "/expert":
                    STATE["app"].expert(body["doc"])
                    out = {"ok": True}
                elif self.path == "/control":  # an app's own control channel, if it has one: state for checks
                    app = STATE.get("app")
                    if not hasattr(app, "call"):
                        raise RuntimeError("this app has no control channel")
                    out = {"value": app.call(body["method"], body.get("params") or {})}
                elif self.path in ROUTES:  # a plugin's route
                    out = ROUTES[self.path](body)
                else:
                    self._send(404, {"error": "not found"})
                    return
            self._send(200, out)
        except Exception as exc:  # noqa: BLE001
            import traceback

            self._send(500, {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    a = ap.parse_args()
    loaded = load_app_modules()
    if loaded:
        print(f"app modules: {', '.join(loaded)}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    xdg_home()  # (before any app starts: GTK and Qt read the user dirs once)
    start_display()
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

