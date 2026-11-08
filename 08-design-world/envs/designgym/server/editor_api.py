"""Routes for the browser editor, mounted beside OpenEnv's own.

    GET  /editor/                          the editor app (static)
    POST /api/episodes                     start an episode from the browser
    GET  /api/episodes                     recent episodes (live sessions to watch)
    GET  /api/episodes/{id}                full editor state
    WS   /api/episodes/{id}/ws             live sync: ops in, state out
    GET  /api/episodes/{id}/assets/{aid}   asset file
    GET  /api/episodes/{id}/reference.jpg  target render, once the episode is exported (for the result screen)
    GET  /api/episodes/{id}/render.png     current render, as scored
    GET  /api/episodes/{id}/screen.png     what the agent sees now (computer-use episodes, real apps included)
    POST /api/episodes/{id}/app/read       a real app's document as it stands, into the render (not scored)
    GET  /api/apps                         the real apps this server offers and where they run (?check=1: up?)
    GET  /api/episodes/{id}/html           current page in the HTML profile
    POST /api/episodes/{id}/html           replace the canvas from HTML
    GET  /api/episodes/{id}/log            trajectory so far
    POST /api/episodes/{id}/submit         score and finish
    GET  /api/fonts                        families and their faces
    GET  /fonts/{slug}/{file}              font files, for @font-face
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from ..core import ops
    from ..core.fonts import slug
    from ..core.html_profile import HTMLProfileError, html_to_doc
    from ..core.pack import TaskPack
    from .episodes import STORE, Episode
except ImportError:
    from core import ops
    from core.fonts import slug
    from core.html_profile import HTMLProfileError, html_to_doc
    from core.pack import TaskPack
    from server.episodes import STORE, Episode

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
mimetypes.add_type("font/ttf", ".ttf")
mimetypes.add_type("font/otf", ".otf")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("image/webp", ".webp")


class NewEpisode(BaseModel):
    split: str | None = None
    index: int | None = None
    task_id: str | None = None
    mode: str = "reference"
    skin: str = "figma"
    theme: str = "light"


class _RevalidatedStatic(StaticFiles):
    """Editor files are revalidated on every load, so an edited stylesheet is never served stale."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _ep(episode_id: str) -> Episode:
    try:
        return STORE.get(episode_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


def mount_editor(app: FastAPI, pack: TaskPack) -> None:
    router = APIRouter()
    fonts = pack.fonts
    font_index = {slug(f): f for f in fonts.families()}

    @router.post("/api/episodes")
    def new_episode(req: NewEpisode):
        if req.task_id:
            task = pack.get(req.task_id)
        else:
            split = req.split or pack.splits()[0]
            index = req.index if req.index is not None else 0
            task = pack.at(split, index)
        try:
            ep = Episode(pack=pack, task=task, mode=req.mode, surface="tools", skin=req.skin,
                         theme=req.theme, max_steps=10_000)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        STORE.add(ep)
        return {"episode_id": ep.episode_id,
                "editor_url": f"/editor/?episode={ep.episode_id}&skin={req.skin}"}

    @router.get("/api/episodes")
    def recent():
        return [{"episode_id": e.episode_id, "task_id": e.task["task_id"],
                 "title": e.task["title"], "mode": e.mode, "surface": e.surface,
                 "elements": len(e.doc.elements), "done": e.done, "watchers": e.watchers,
                 "reward": e.result.total if e.result else None}
                for e in STORE.recent(30)]

    @router.get("/api/episodes/{episode_id}")
    def state(episode_id: str):
        return _ep(episode_id).editor_state()

    @router.get("/api/episodes/{episode_id}/assets/{asset_id}")
    def asset(episode_id: str, asset_id: str):
        ep = _ep(episode_id)
        path = ep.assets.get(asset_id)
        if path is None:
            raise HTTPException(404, f"no asset {asset_id}")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    @router.get("/api/episodes/{episode_id}/reference.jpg")
    def reference(episode_id: str, size: int = 1024):
        ep = _ep(episode_id)
        # The editor never shows the target while a design is being made; after export
        # the result screen compares it with the design.
        if not ep.done:
            raise HTTPException(403, "the target is shown after export")
        if ep.mode != "reference":
            raise HTTPException(404, "description-mode episodes have no reference image")
        return Response(pack.target_jpeg(ep.task, min(2048, max(128, size))),
                        media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @router.get("/api/episodes/{episode_id}/render.png")
    def render(episode_id: str, size: int = 768):
        ep = _ep(episode_id)
        return Response(ep.render_png(min(2048, max(64, size))), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @router.get("/api/episodes/{episode_id}/screen.png")
    async def screen(episode_id: str):
        ep = _ep(episode_id)
        if ep.surface != "gui":
            raise HTTPException(404, "only computer-use episodes have a screen")
        from .gui import GuiDriver

        try:
            png = await GuiDriver.get().screenshot(ep.episode_id)
        except RuntimeError:  # no page (closed, or never opened in this process)
            raise HTTPException(410, "this episode's screen is closed") from None
        return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})

    @router.post("/api/episodes/{episode_id}/app/read")
    def read_app(episode_id: str):
        ep = _ep(episode_id)
        if not ep.read_app():
            raise HTTPException(409, "not a running real-app episode")
        return {"elements": len(ep.doc.elements), "version": ep.version}

    @router.get("/api/apps")
    def apps(check: bool = False):
        try:
            from ..realapps import status
        except ImportError:
            from realapps import status
        return status(check=check)

    @router.get("/api/episodes/{episode_id}/html", response_class=PlainTextResponse)
    def get_html(episode_id: str):
        return _ep(episode_id).html()

    @router.post("/api/episodes/{episode_id}/html")
    def put_html(episode_id: str, body: str = Body(..., media_type="text/plain")):
        ep = _ep(episode_id)
        try:
            doc, warnings = html_to_doc(body, ep.doc.width, ep.doc.height)
            ep.apply("replace_document", {"document": doc}, "editor")
        except (HTMLProfileError, ops.OpError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from None
        return {"elements": len(ep.doc.elements), "warnings": sorted(set(warnings))}

    @router.get("/api/episodes/{episode_id}/log")
    def log(episode_id: str):
        ep = _ep(episode_id)
        return {"episode_id": ep.episode_id, "task_id": ep.task["task_id"], "log": ep.log}

    @router.post("/api/episodes/{episode_id}/submit")
    def submit(episode_id: str):
        ep = _ep(episode_id)
        return ep.submit("editor").as_dict()

    @router.websocket("/api/episodes/{episode_id}/ws")
    async def live(ws: WebSocket, episode_id: str):
        await ws.accept()
        ep = STORE.maybe(episode_id)
        if ep is None:
            await ws.send_json({"type": "error", "message": f"unknown episode {episode_id}"})
            await ws.close()
            return
        listener = ep.subscribe()
        await ws.send_json(ep.editor_state())
        source = ws.query_params.get("source", "editor")

        async def pump():
            while True:
                msg = await listener.queue.get()
                # Coalesce bursts (an agent firing ten calls) into one state push.
                while not listener.queue.empty():
                    msg = listener.queue.get_nowait()
                await ws.send_json({**ep.editor_state(), "source": msg.get("source")})

        pusher = asyncio.create_task(pump())
        try:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("type")
                reply = {"type": "ack", "id": msg.get("id"), "ok": True}
                try:
                    if kind == "op":
                        reply["result"] = await asyncio.to_thread(
                            ep.apply, msg["op"], msg.get("args") or {}, source)
                    elif kind == "undo":
                        await asyncio.to_thread(ep.undo, source)
                    elif kind == "redo":
                        await asyncio.to_thread(ep.redo, source)
                    elif kind == "submit":
                        grade = await asyncio.to_thread(ep.submit, source)
                        reply["result"] = grade.as_dict()
                    elif kind == "ping":
                        reply["type"] = "pong"
                    else:
                        raise ops.OpError(f"unknown message type {kind!r}")
                except (ops.OpError, KeyError, ValueError) as exc:
                    reply.update(ok=False, error=str(exc).strip("'\""))
                await ws.send_json(reply)
        except WebSocketDisconnect:
            pass
        finally:
            pusher.cancel()
            ep.unsubscribe(listener)

    @router.get("/api/fonts")
    def font_list():
        return {"families": [{"family": f, "slug": slug(f), "styles": fonts.styles(f)}
                             for f in fonts.families()]}

    @router.get("/fonts/{family_slug}/{filename}")
    def font_file(family_slug: str, filename: str):
        family = font_index.get(family_slug)
        style = filename.rsplit(".", 1)[0]
        path = fonts.file_path(family, style) if family else None
        if path is None or not path.exists():
            raise HTTPException(404, "no such font")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=604800",
                                           "Access-Control-Allow-Origin": "*"})

    @router.get("/fonts.css")
    def fonts_css(families: str = ""):
        wanted = [f for f in families.split(",") if f] or fonts.families()
        rules = []
        for fam in wanted:
            fam = fonts.resolve(fam)
            for style in fonts.styles(fam):
                path = fonts.file_path(fam, style)
                weight = 700 if "bold" in style else 400
                fstyle = "italic" if "italic" in style else "normal"
                rules.append(
                    f"@font-face{{font-family:'{fam}';src:url('/fonts/{slug(fam)}/{style}"
                    f"{path.suffix}');font-weight:{weight};font-style:{fstyle};"
                    f"font-display:block}}"
                )
        return Response("\n".join(rules), media_type="text/css",
                        headers={"Cache-Control": "public, max-age=3600"})

    if os.environ.get("DESIGN_CANVAS_DEBUG") == "1":
        # Debug only: lets a scripted GUI expert find controls on the agent's
        # own page. Never offered to agents, and off unless explicitly enabled.
        from .gui import GuiDriver

        @router.post("/api/debug/gui/{episode_id}/locate")
        async def locate(episode_id: str, body: dict = Body(...)):
            driver = GuiDriver.get()

            async def run():
                page = driver._page(episode_id)
                return await page.evaluate(
                    """({selector, text}) => {
                        const vis = (e) => e.offsetParent !== null || e.getClientRects().length;
                        let els = [...document.querySelectorAll(selector)].filter(vis);
                        if (text !== null && text !== undefined)
                            els = els.filter((e) => e.textContent.trim() === text);
                        // What a person would see: the part of the control not clipped by a
                        // scrolling panel, whether the pointer there actually hits it, and
                        // which panel to scroll (and which way) when it is out of view.
                        const W = innerWidth, H = innerHeight;
                        return els.slice(0, 8).map((e) => {
                            const r = e.getBoundingClientRect();
                            let t = Math.max(r.top, 0), b = Math.min(r.bottom, H);
                            let l = Math.max(r.left, 0), rt = Math.min(r.right, W), scroller = null;
                            for (let p = e.parentElement; p && p !== document.body; p = p.parentElement) {
                                const cs = getComputedStyle(p);
                                if (!/(auto|scroll|hidden|clip)/.test(cs.overflowX + cs.overflowY)) continue;
                                const pr = p.getBoundingClientRect();
                                t = Math.max(t, pr.top); b = Math.min(b, pr.bottom);
                                l = Math.max(l, pr.left); rt = Math.min(rt, pr.right);
                                if (!scroller && /(auto|scroll)/.test(cs.overflowY) && p.scrollHeight > p.clientHeight + 1)
                                    scroller = {x: pr.left + pr.width / 2, y: pr.top + pr.height / 2,
                                                dir: r.top < pr.top ? "up" : "down"};
                            }
                            const cx = (l + rt) / 2, cy = (t + b) / 2;
                            const open = b - t >= Math.min(8, r.height) - 0.5 && rt - l >= Math.min(8, r.width) - 0.5;
                            const at = open ? document.elementFromPoint(cx, cy) : null;
                            const hit = !!at && (at === e || e.contains(at));
                            return {x: open ? cx : r.left + r.width / 2, y: open ? cy : r.top + r.height / 2,
                                    w: r.width, h: r.height, top: r.top, bottom: r.bottom,
                                    visible: open && hit, covered: open && !hit, scroller};
                        });
                    }""",
                    {"selector": body["selector"], "text": body.get("text")},
                )

            return await driver.call(run())

        @router.post("/api/debug/gui/{episode_id}/eval")
        async def evaluate(episode_id: str, body: dict = Body(...)):
            driver = GuiDriver.get()

            async def run():
                return await driver._page(episode_id).evaluate(body["js"])

            return await driver.call(run())

    app.include_router(router)
    if WEB_DIR.exists():
        app.mount("/editor", _RevalidatedStatic(directory=WEB_DIR, html=True), name="editor")

    @app.get("/editor", include_in_schema=False)
    def editor_redirect():
        return HTMLResponse('<meta http-equiv="refresh" content="0; url=/editor/">')

    @app.exception_handler(KeyError)
    async def key_error(_, exc: KeyError):
        return JSONResponse({"detail": str(exc).strip("'\"")}, status_code=404)
