"""Episodes: one design being worked on, reachable from every surface at once.

OpenEnv gives each WebSocket session its own environment object. The episode
lives outside that object, in a process-wide store keyed by `episode_id`, so a
browser tab can open `/editor?episode=<id>` and watch (or take over) the same
document an agent is editing over MCP. Every change is broadcast to subscribed
editors as a full document; documents are a few kilobytes, so there is no diff
protocol to get wrong.

State per episode is a document, its undo history, and a trajectory log. The
log records every operation from every source in one format, which is what the
SFT export and the rollout traces read.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    from ..core import ops
    from ..core import workspace as ws
    from ..core.doc import DesignDoc
    from ..core.html_profile import doc_to_html
    from ..core.pack import TaskPack
    from ..core.score import Grade
    from ..realapps.registry import RealSkins
except ImportError:  # flat layout inside the Docker image
    from core import ops
    from core import workspace as ws
    from core.doc import DesignDoc
    from core.html_profile import doc_to_html
    from core.pack import TaskPack
    from core.score import Grade
    from realapps.registry import RealSkins

MODES = ("reference", "description")
SURFACES = ("tools", "html", "gui")
# Editor layouts: id -> (display name, the kind of app it is modelled on). Neutral names and
# brand marks only; the ids are what `reset(skin=...)` and the editor's `?skin=` accept.
SKIN_INFO = {
    "figma": ("Studio", "Figma-like"),
    "canva": ("Template", "Canva-like"),
    "photoshop": ("Pixel", "Photoshop-like"),
    "slides": ("Deck", "Google Slides-like"),
    "powerpoint": ("Present", "PowerPoint-like"),
    "inkscape": ("Vector", "Inkscape-like"),
}
SKINS = tuple(SKIN_INFO)
# Real apps behind the same GUI interface (designgym.realapps): never drawn at random, only by name. Read live from
# the registry: LibreOffice Impress ("impress") and Draw ("draw"), Inkscape ("inkscape-desktop"), plus any plugin's.
REAL_SKINS = RealSkins()
HISTORY_LIMIT = 200
IDLE_TTL_S = 3 * 3600


@dataclass
class Listener:
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop


@dataclass
class Episode:
    pack: TaskPack
    task: dict
    mode: str = "reference"
    surface: str = "tools"
    skin: str = "figma"
    theme: str = "light"
    max_steps: int = 60
    # behind the multi-worker router an id names its worker (w2-...), so requests for it reach that process
    episode_id: str = field(default_factory=lambda: (f"w{os.environ['DESIGN_CANVAS_WORKER_ID']}-"
                                                     if os.environ.get("DESIGN_CANVAS_WORKER_ID") else "")
                            + uuid.uuid4().hex[:12])
    doc: DesignDoc | None = None
    step_count: int = 0          # actions; looking (screenshots, views, lists) is counted separately
    observation_count: int = 0
    done: bool = False
    result: Grade | None = None
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    version: int = 0
    log: list[dict] = field(default_factory=list)
    _undo: list[DesignDoc] = field(default_factory=list)
    _redo: list[DesignDoc] = field(default_factory=list)
    _listeners: list[Listener] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    # (backend, session) when the editor is a real app: its own document is read back at submit
    realapp: Any = None
    # where the task's images live on "this computer" (core.workspace); None: a1.png, ... in Pictures
    workspace: dict | None = None
    # how the prompt speaks of the action budget: "count" names it ("at most N actions"), "efficient" asks for few
    # actions without a number (evals: the limit is still enforced, only not announced)
    budget_note: str = "count"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}")
        if self.skin not in SKINS and self.skin not in REAL_SKINS:
            raise ValueError(f"skin must be one of {SKINS + tuple(REAL_SKINS)}")
        gt = self.pack.gt_doc(self.task)
        if self.doc is None:
            self.doc = DesignDoc(width=gt.width, height=gt.height)
        self.ctx = ops.OpContext(self.pack.fonts, self.pack.asset_ctx(self.task))
        self.assets = self.pack.asset_paths(self.task)
        if self.workspace:  # decoy images are assets of the episode too: opening one inserts it, as a mistake
            self.workspace = ws.validate(self.workspace, self.task)
            paths, ctx = ws.decoy_assets(self.workspace, self.pack)
            self.assets = {**self.assets, **paths}
            self.ctx.assets.update(ctx)

    # ------------------------------------------------------------ editing

    def _record(self, source: str, op: str, args: dict | None, ok: bool, error: str | None = None,
                result: Any = None):
        entry = {"t": round(time.time() - self.created, 3), "source": source, "op": op,
                 "args": args or {}, "ok": ok}
        if error:
            entry["error"] = error
        if result is not None and op in ("add_image", "add_text", "add_shape",
                                         "duplicate_element"):
            entry["element_id"] = result.get("element_id")
        self.log.append(entry)

    def _check_open(self):
        if self.done:
            raise ops.OpError("this episode is finished; reset to start another")

    def apply(self, op: str, args: dict | None = None, source: str = "agent") -> dict:
        with self._lock:
            self._check_open()
            try:
                new, result = ops.apply(self.doc, op, args or {}, self.ctx)
            except ops.OpError as exc:
                self._record(source, op, args, False, str(exc))
                raise
            self._undo.append(self.doc)
            if len(self._undo) > HISTORY_LIMIT:
                self._undo.pop(0)
            self._redo.clear()
            self.doc = new
            self._record(source, op, args, True, result=result)
            self._changed(source)
            return result

    def undo(self, source: str = "agent") -> bool:
        with self._lock:
            self._check_open()
            if not self._undo:
                self._record(source, "undo", None, False, "nothing to undo")
                return False
            self._redo.append(self.doc)
            self.doc = self._undo.pop()
            self._record(source, "undo", None, True)
            self._changed(source)
            return True

    def redo(self, source: str = "agent") -> bool:
        with self._lock:
            self._check_open()
            if not self._redo:
                self._record(source, "redo", None, False, "nothing to redo")
                return False
            self._undo.append(self.doc)
            self.doc = self._redo.pop()
            self._record(source, "redo", None, True)
            self._changed(source)
            return True

    def submit(self, source: str = "agent", reason: str = "submitted") -> Grade:
        with self._lock:
            if self.done and self.result is not None:
                return self.result
            if self.realapp is not None:  # the app's own document is what gets scored
                backend, session = self.realapp
                self.doc = backend.read_doc(session)
            self.result = self.score()
            self.done = True
            self._record(source, "submit", {"reason": reason}, True)
            self._changed(source)
            return self.result

    def score(self) -> Grade:
        return self.pack.grade(self.task, self.doc, self.mode, assets=self.assets if self.workspace else None)

    def read_app(self) -> bool:
        """Bring a real app's document into `doc` as it stands now (what submit would score), for the playground
        and the editor API; the agent's tools never call this. False for the built-in editor."""
        if self.realapp is None or self.done:
            return False
        with self._lock:
            backend, session = self.realapp
            self.doc = backend.read_doc(session)
            self._changed("app")
        return True

    # ------------------------------------------------------------ views

    def render_png(self, max_size: int = 768) -> bytes:
        return self.pack.renderer.render_png(self.doc, self.assets, max_size)

    def render_jpeg(self, max_size: int = 768) -> bytes:
        return self.pack.renderer.render_jpeg(self.doc, self.assets, max_size)

    def html(self) -> str:
        return doc_to_html(self.doc)

    def info(self) -> dict:
        """Task info a model is allowed to see."""
        t = self.task
        info = {
            "episode_id": self.episode_id,
            "task_id": t["task_id"],
            "split": t["split"],
            "mode": self.mode,
            "surface": self.surface,
            "title": t["title"],
            "format": t["format"],
            "canvas": t["canvas"],
            "assets": [
                {"id": a["id"], "kind": a["kind"], "aspect_ratio": a["aspect"]}
                for a in t["assets"]
            ],
            "fonts_used": t["fonts"],
            "max_steps": self.max_steps,
            "budget_note": self.budget_note,
        }
        if self.workspace:  # the images are among other files, under everyday names: the prompt names the files
            info["workspace"] = True
            names = ws.file_names(self.workspace)
            for a in info["assets"]:
                a["file"] = names[a["id"]]
        if self.mode == "description":
            info["description"] = t["description"]
        return info

    def editor_state(self, base: str = "") -> dict:
        """Everything the browser editor needs to draw this episode."""
        t = self.task
        return {
            "type": "state",
            "episode_id": self.episode_id,
            "version": self.version,
            "doc": self.doc.model_dump(mode="json"),
            "task": {
                "task_id": t["task_id"],
                "title": t["title"],
                "format": t["format"],
                "canvas": t["canvas"],
                "mode": self.mode,
                "surface": self.surface,
                "description": t["description"] if self.mode == "description" else None,
                # Only after export (the result screen): the editor never shows the target.
                "reference_url": f"{base}/api/episodes/{self.episode_id}/reference.jpg"
                if self.mode == "reference" and self.done else None,
                "fonts_used": t["fonts"],
            },
            "assets": [
                {"id": a["id"], "kind": a["kind"], "aspect": a["aspect"],
                 "url": f"{base}/api/episodes/{self.episode_id}/assets/{a['id']}"}
                for a in t["assets"]
            ] + [
                {"id": did, "kind": c.get("kind", "graphic"), "aspect": c.get("aspect", 1.0),
                 "url": f"{base}/api/episodes/{self.episode_id}/assets/{did}", "decoy": True}
                for did, c in self.ctx.assets.items() if did not in {a["id"] for a in t["assets"]}
            ],
            "workspace": ws.layout(self.workspace) if self.workspace else None,
            # (a lived-in computer: each picture's pixel size and bytes, for the file window's columns and preview)
            "facts": {aid: ws.file_facts(p) for aid, p in self.assets.items()} if self.workspace else None,
            "can_undo": bool(self._undo),
            "can_redo": bool(self._redo),
            "done": self.done,
            "result": self.result.as_dict() if self.result else None,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

    # ------------------------------------------------------------ pub/sub

    def subscribe(self) -> Listener:
        listener = Listener(asyncio.Queue(maxsize=256), asyncio.get_running_loop())
        with self._lock:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _changed(self, source: str) -> None:
        self.version += 1
        self.touched = time.time()
        payload = {"type": "changed", "version": self.version, "source": source}
        for listener in list(self._listeners):
            try:
                listener.loop.call_soon_threadsafe(_offer, listener.queue, payload)
            except RuntimeError:  # loop closed: the tab went away
                self.unsubscribe(listener)

    @property
    def watchers(self) -> int:
        return len(self._listeners)


def _offer(queue: asyncio.Queue, payload: dict) -> None:
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(payload)


class EpisodeStore:
    def __init__(self, limit: int = 4000):
        self._items: dict[str, Episode] = {}
        self._lock = threading.Lock()
        self.limit = limit

    def add(self, ep: Episode) -> Episode:
        with self._lock:
            self._evict()
            self._items[ep.episode_id] = ep
        return ep

    def get(self, episode_id: str) -> Episode:
        with self._lock:
            ep = self._items.get(episode_id)
        if ep is None:
            raise KeyError(f"unknown or expired episode {episode_id!r}")
        return ep

    def maybe(self, episode_id: str | None) -> Episode | None:
        if not episode_id:
            return None
        with self._lock:
            return self._items.get(episode_id)

    def recent(self, n: int = 20) -> list[Episode]:
        with self._lock:
            items = sorted(self._items.values(), key=lambda e: e.touched, reverse=True)
        return items[:n]

    def _evict(self) -> None:
        now = time.time()
        stale = [k for k, e in self._items.items()
                 if now - e.touched > IDLE_TTL_S and not e.watchers]
        for k in stale:
            del self._items[k]
        if len(self._items) >= self.limit:
            for k, _ in sorted(self._items.items(), key=lambda kv: kv[1].touched)[
                : len(self._items) - self.limit + 1
            ]:
                del self._items[k]


STORE = EpisodeStore()
