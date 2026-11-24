"""The env's side of the desktop agent (realapps/desktop/agent.py): screen, input, and the app's document.

`DesktopBackend` gives each desktop-app episode one desktop (one X display = one episode at a time), from the pool in
DESIGN_CANVAS_DESKTOP_URLS (comma list; default http://localhost:5005), or from Daytona sandboxes the env starts
(realapps/daytona.py). The GUI driver forwards that episode's
screenshots and input to the desktop instead of a browser page, so the GUI tools are the same.
"""

from __future__ import annotations

import base64
import errno
import fcntl
import hashlib
import json
import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ...core.doc import DesignDoc, Element
from ...core.text import layout
from .. import registry

LOCKS = Path(os.environ.get("DESIGN_CANVAS_LOCK_DIR", "/tmp/design-canvas-locks"))


# Connection-level failures on the way to a desktop (the request never reached it): DNS not answering, the TCP
# connect timing out, refused, host or network unreachable, reset before anything was sent. Over a weak link these
# come and go within seconds; a request is tried again a few times before it counts as an error. A request that
# was answered with an error, or timed out while being read, is not: for an input it may have been carried out.
_RETRY_ERRNO = {errno.ENETDOWN, errno.ENETUNREACH, errno.ETIMEDOUT, errno.ECONNREFUSED, errno.EHOSTDOWN,
                errno.EHOSTUNREACH}  # (by name: the numbers differ between macOS and Linux)
_RETRIES = int(os.environ.get("DESIGN_CANVAS_DESKTOP_RETRIES", "4"))


def _connection_error(e: BaseException, sent: bool) -> bool:
    """A failure before the request reached the desktop. A reset counts only for a request without a body (a
    read): for an /input the body may have gone out before the reset, and the click may have happened."""
    import socket

    if isinstance(e, urllib.error.HTTPError):
        return False
    r = getattr(e, "reason", e)
    if isinstance(r, socket.gaierror):  # (DNS not answering)
        return True
    code = getattr(r, "errno", None) if isinstance(r, OSError) else None
    return code in _RETRY_ERRNO or (code == errno.ECONNRESET and not sent)


def _open(req, timeout: float, retry_reads: bool = False):
    """urlopen with retries on connection-level failures (and, for reads, on a timed-out read as well)."""
    import socket
    import time

    sent = req.data is not None  # (a POST: its body may have reached the desktop before a failure)
    for k in range(_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError, socket.timeout) as e:
            read_timeout = isinstance(e, (socket.timeout, TimeoutError)) or "timed out" in str(getattr(e, "reason", e))
            if k == _RETRIES or not (_connection_error(e, sent) or (retry_reads and read_timeout)):
                raise
            time.sleep(2.0 * 2 ** k)  # 2, 4, 8, 16 s


# DESIGN_CANVAS_SCREENSHOT_FORMAT=webp asks the desktop agent for WebP screenshots (quality DESIGN_CANVAS_SCREENSHOT_QUALITY,
# default 90) instead of PNG: four to six times less traffic for a recorder on a slow or metered link. Unset: PNG.
_SHOT_QUERY = ""
if os.environ.get("DESIGN_CANVAS_SCREENSHOT_FORMAT", "").lower() == "webp":
    _SHOT_QUERY = f"?fmt=webp&q={int(os.environ.get('DESIGN_CANVAS_SCREENSHOT_QUALITY', '90'))}"


class DesktopClient:
    def __init__(self, base_url: str, headers: dict[str, str] | None = None):
        self.base = base_url.rstrip("/")
        self.headers = dict(headers or {})  # e.g. a Daytona sandbox's preview token

    def _post(self, path: str, body: dict, timeout: float = 300):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **self.headers})
        try:
            return json.loads(_open(req, timeout))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"desktop {path}: {e.read()[:1500].decode(errors='replace')}") from None

    def screenshot(self) -> bytes:
        req = urllib.request.Request(self.base + "/screenshot" + _SHOT_QUERY, headers=self.headers)
        return _open(req, 60, retry_reads=True)

    def a11y(self) -> list[dict]:
        """The on-screen widgets from the accessibility tree (agent.py GET /a11y): role, name, tooltip ("desc"),
        box [x, y, w, h] in screenshot pixels, states, text and value. For scripted recorders; agents see pixels."""
        req = urllib.request.Request(self.base + "/a11y", headers=self.headers)
        try:
            return json.loads(_open(req, 120, retry_reads=True))["nodes"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"desktop /a11y: {e.read()[:1500].decode(errors='replace')}") from None

    def observe(self) -> tuple[list[dict], bytes]:
        """The widgets (as a11y) and the screen taken right after them, in one round trip (agent.py GET /observe)."""
        import base64

        req = urllib.request.Request(self.base + "/observe" + _SHOT_QUERY, headers=self.headers)
        try:
            o = json.loads(_open(req, 120, retry_reads=True))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"desktop /observe: {e.read()[:1500].decode(errors='replace')}") from None
        return o["nodes"], base64.b64decode(o["png"])

    def input(self, **action) -> None:
        self._post("/input", action, timeout=120)

    def reset(self, app: str, width: int, height: int, background: str, assets: dict[str, Path],
              workspace: dict | None = None) -> None:
        body = {"app": app, "width": width, "height": height, "background": background,
                "assets": {a: base64.b64encode(Path(p).read_bytes()).decode() for a, p in assets.items()}}
        if workspace:  # the images under everyday names in an everyday folder, among other files (core.workspace)
            try:
                from ...core import workspace as ws
            except ImportError:  # flat layout inside the Docker image
                from core import workspace as ws
            body["workspace"] = {"folder": workspace["folder"], "files": workspace["files"],
                                 "others": [{"path": o["path"], "b64": base64.b64encode(
                                     ws.document_bytes(o["path"])).decode()} for o in workspace.get("others") or []]}
        self._post("/reset", body, timeout=120)

    def read(self) -> dict:
        return self._post("/read", {}, timeout=300)

    def expert(self, doc: DesignDoc) -> None:
        self._post("/expert", {"doc": doc.model_dump(mode="json")}, timeout=120)


@dataclass
class DesktopSession:
    app: str
    client: DesktopClient
    url: str
    skipped: int = 0
    fonts: object = None  # the pack's FontRegistry: SVG text is placed by baseline, converted with our own layout
    spec: object = None  # the app's registry entry (how it places text)


def text_anchor(e: Element, fonts) -> dict:
    """Where an app that places text by baseline (SVG) should put this element: first baseline, anchor x, line step."""
    lay = layout(e, fonts)
    return {"_baseline": e.y + lay.lines[0].baseline, "_step": lay.line_box,
            "_anchor_x": e.x + {"left": 0.0, "center": e.width / 2, "right": e.width}[e.align]}


def text_top(e: Element, fonts) -> dict:
    """For apps whose text layers start at the first line's top: our box top plus the half-leading."""
    lay = layout(e, fonts)
    ascent, descent = fonts.metrics(e.font_family or "", e.font_size, e.bold, e.italic)
    return {"_top": e.y + (lay.line_box - (ascent + descent)) / 2, "_step": lay.line_box,
            "_natural": ascent + descent}


def from_top(raw: dict, fonts) -> dict:
    top = raw.pop("_top")
    pivot = raw.pop("_pivot", None)  # a turned frame: x / _top are in its own frame, about this corner
    extra = raw.pop("_line_extra", None)
    if extra is not None:  # spacing given as pixels added to the font's own line height
        a, d = fonts.metrics(raw.get("font_family") or "", raw["font_size"], raw.get("bold", False),
                             raw.get("italic", False))
        raw["line_height"] = max(0.1, (a + d + extra) / raw["font_size"])
    el = Element.model_validate({"id": "t", "x": raw["x"], "y": 0, "width": raw["width"], "height": raw["height"],
                                 **{k: v for k, v in raw.items() if k not in ("x", "y", "width", "height")}})
    lay = layout(el, fonts)
    ascent, descent = fonts.metrics(el.font_family or "", el.font_size, el.bold, el.italic)
    y, h = top - (lay.line_box - (ascent + descent)) / 2, max(1.0, lay.height)
    if pivot is None:
        return {**raw, "y": y, "height": h}
    th = math.radians(float(raw.get("angle") or 0))
    lx, ly = raw["x"] + raw["width"] / 2, y + h / 2  # the box's centre in the frame's own coordinates
    cx = pivot[0] + lx * math.cos(th) - ly * math.sin(th)
    cy = pivot[1] + lx * math.sin(th) + ly * math.cos(th)
    return {**raw, "x": cx - raw["width"] / 2, "y": cy - h / 2, "height": h}


def from_anchor(raw: dict, fonts) -> dict:
    """The inverse: a text element read back by baseline and anchor -> DesignDoc box, through the same layout."""
    base, ax = raw.pop("_baseline"), raw.pop("_anchor_x")
    raw.pop("_step", None)
    m = raw.pop("_matrix", None)
    el = Element.model_validate({"id": "t", "x": 0, "y": 0, "width": 10, "height": 10, "auto_width": True, **raw})
    lay = layout(el, fonts)
    w, h = max(1.0, lay.width + 1.0), max(1.0, lay.height)
    k = {"left": 0.0, "center": 0.5, "right": 1.0}[el.align]
    if m is None:
        return {**raw, "auto_width": True, "width": w, "height": h, "x": ax - k * w, "y": base - lay.lines[0].baseline}
    # rotated text: the anchor is in the text's own frame; the box turns about its centre (DesignDoc's convention)
    a, b, c, d, e, f = m
    s = math.hypot(a, b) or 1.0
    cx, cy = ax - k * w / s + w / s / 2, base - lay.lines[0].baseline / s + h / s / 2
    return {**raw, "auto_width": True, "width": w, "height": h,
            "x": a * cx + c * cy + e - w / 2, "y": b * cx + d * cy + f - h / 2}


class DesktopBackend:
    """Real desktop apps (LibreOffice Impress and Draw, Inkscape, and plugins' apps) behind the GUI interface."""

    def __init__(self, urls: list[str] | None = None):
        urls = urls or os.environ.get("DESIGN_CANVAS_DESKTOP_URLS", "http://localhost:5005").split(",")
        self.free = [u.strip() for u in urls if u.strip()]
        self.held: dict[str, int] = {}  # desktop -> its lock file, held while an episode runs on it
        self.cv = threading.Condition()

    def _client(self, url: str) -> DesktopClient:
        return DesktopClient(url)

    @staticmethod
    def _lock(key: str) -> int | None:
        """An exclusive lock on one desktop shared by every process on this machine (the multi-worker server's
        workers all see the same desktops), or None if another process holds it."""
        LOCKS.mkdir(parents=True, exist_ok=True)
        fd = os.open(LOCKS / f"desktop-{hashlib.sha1(key.encode()).hexdigest()[:16]}.lock", os.O_CREAT | os.O_RDWR,
                     0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
            return None

    def new_episode(self, skin: str, width: int, height: int, assets: dict[str, Path],
                    background: str = "#ffffff", fonts=None, workspace: dict | None = None) -> DesktopSession:
        deadline = time.time() + 1800
        with self.cv:  # one episode per desktop: wait for one no process is using
            while True:
                url = next((u for u in self.free if (fd := self._lock(u)) is not None), None)
                if url is not None:
                    self.free.remove(url)
                    self.held[url] = fd
                    break
                if time.time() > deadline:
                    raise TimeoutError("no desktop became free in 30 minutes")
                self.cv.wait(timeout=1.0)  # a release here wakes it; a release in another process is polled
        spec = registry.get(skin)
        try:
            client = self._client(url)
            client.reset(spec.agent_app, width, height, background, assets, workspace=workspace)
        except Exception as e:
            if isinstance(e, OSError):  # (timeouts included) the desktop itself is unwell, not this design
                self.unwell(url)
            self.release(url)
            raise
        return DesktopSession(spec.agent_app, client, url, fonts=fonts, spec=spec)

    def unwell(self, url: str) -> None:
        """A desktop that could not even reset: nothing to do for a fixed URL (a pool may restart it)."""

    def release(self, url: str) -> None:
        with self.cv:
            fd = self.held.pop(url, None)
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            if url not in self.free:
                self.free.append(url)
            self.cv.notify()

    def read_doc(self, s: DesktopSession) -> DesignDoc:
        raw = s.client.read()
        doc = DesignDoc(width=raw["width"], height=raw["height"], background=raw.get("background") or "#ffffff")
        s.skipped = int(raw.get("skipped", 0))
        for n, e in enumerate(raw["elements"], 1):
            e = {k: v for k, v in e.items() if v is not None}
            if "_baseline" in e or "_top" in e:
                if s.fonts is None:
                    raise RuntimeError("this app places text by its own metrics: pass the pack's fonts to new_episode")
                e = from_anchor(e, s.fonts) if "_baseline" in e else from_top(e, s.fonts)
            e["opacity"] = min(1.0, max(0.0, float(e.get("opacity", 1.0))))
            try:
                doc.elements.append(Element.model_validate({"id": f"e{n}", **e}))
            except ValueError:
                s.skipped += 1
        doc.next_id = len(doc.elements) + 1
        return doc

    def write_doc(self, s: DesktopSession, doc: DesignDoc) -> None:
        d = doc.model_dump(mode="json")
        if s.fonts is not None:
            for e, el in zip(d["elements"], doc.elements, strict=True):
                if el.kind == "text":
                    e.update(text_anchor(el, s.fonts) if s.spec.text == "baseline" else text_top(el, s.fonts))
                    if s.spec.natural_width:  # a text frame clips what does not fit: at least the text's own width
                        e["_natural_w"] = layout(el, s.fonts).width
        s.client._post("/expert", {"doc": d}, timeout=120)
