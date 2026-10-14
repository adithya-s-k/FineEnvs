"""The computer-use surface: a real browser driving the real editor.

For `surface="gui"` the environment opens the editor for the episode in a
headless Chromium page and the agent acts on it with mouse and keyboard, seeing
only screenshots. Nothing is simulated: a click lands on the same DOM a person
would click, the editor turns it into the same operation an MCP call would
make, and the document is scored like any other.

The browser lives on a private event loop in a background thread. A Playwright
browser is bound to the loop that launched it, and the editor page talks back
to this very server over a WebSocket, so driving it from the server's own loop
would deadlock: the tool would wait on the page, and the page would wait on a
server that is blocked waiting for the tool. Tools therefore `await` futures
from this thread and keep the server loop free.

One browser per process, one context per episode (isolated storage), which is
a few tens of megabytes per session rather than a VM.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading

VIEWPORT = {"width": 1280, "height": 800}

# "ctrl" means the platform's shortcut modifier, so an agent's ctrl+a selects
# all and ctrl+z undoes whether the server runs on Linux or macOS.
_KEY_ALIASES = {
    "ctrl": "ControlOrMeta", "control": "ControlOrMeta", "cmd": "Meta", "command": "Meta", "meta": "Meta",
    "super": "Meta", "win": "Meta", "alt": "Alt", "option": "Alt", "shift": "Shift",
    "enter": "Enter", "return": "Enter", "esc": "Escape", "escape": "Escape",
    "backspace": "Backspace", "delete": "Delete", "del": "Delete", "tab": "Tab",
    "space": " ", "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft",
    "right": "ArrowRight", "home": "Home", "end": "End", "pageup": "PageUp",
    "pagedown": "PageDown", "plus": "+", "minus": "-",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
}


def normalize_keys(combo: str) -> str:
    """'ctrl+shift+z' -> 'Control+Shift+z' (Playwright's spelling)."""
    parts = [p for p in combo.replace(" ", "").split("+") if p] or [combo]
    out = []
    for p in parts:
        low = p.lower()
        if low in _KEY_ALIASES:
            out.append(_KEY_ALIASES[low])
        elif len(p) == 1:
            out.append(p)
        elif low.startswith("f") and low[1:].isdigit():
            out.append(p.upper())
        else:
            out.append(p[:1].upper() + p[1:])
    return "+".join(out)


def self_url() -> str:
    url = os.environ.get("DESIGN_CANVAS_SELF_URL")
    if url:
        return url.rstrip("/")
    return f"http://127.0.0.1:{os.environ.get('PORT', '8000')}"


class GuiDriver:
    _instance: GuiDriver | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> GuiDriver:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="gui-driver",
                                        daemon=True)
        self._thread.start()
        self._pw = None
        self._browser = None
        self._pages: dict[str, tuple] = {}  # episode_id -> (context, page)
        self._remote: dict[str, object] = {}  # episode_id -> remote screen (desktop agent), instead of a page
        self._real: set[str] = set()  # episodes whose page is a real web app (a plugin's), not the built-in editor
        self._start_lock = None

    # ------------------------------------------------------------ plumbing

    def submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def call(self, coro, timeout: float = 30.0):
        """Await `coro` on the driver loop from any other loop."""
        fut = self.submit(coro)
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout)

    def call_sync(self, coro, timeout: float = 30.0):
        return self.submit(coro).result(timeout)

    async def _browser_ready(self):
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._browser is None or not self._browser.is_connected():
                try:
                    from playwright.async_api import async_playwright
                except ImportError as exc:  # pragma: no cover - depends on extras
                    raise RuntimeError(
                        "the gui surface needs Playwright: pip install designgym"
                        " && playwright install chromium"
                    ) from exc
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(
                    args=["--disable-dev-shm-usage", "--font-render-hinting=none",
                          "--disable-lcd-text"]
                )
        return self._browser

    # ------------------------------------------------------------ pages

    async def _open(self, episode_id: str, url: str, cookies=None, ready=None, headers=None, init_script=None,
                    file_window=None):
        await self._close(episode_id)
        browser = await self._browser_ready()
        context = await browser.new_context(viewport=VIEWPORT, device_scale_factor=1, extra_http_headers=headers)
        if init_script:  # e.g. a web app that keeps its document in the page: a handle for setup and read-back
            await context.add_init_script(init_script)
        if file_window:  # a lived-in computer: the app's file input opens the episode's file window (file_window.py)
            await self._file_window(context, episode_id, file_window)
        if headers:
            # the headers carry a sandbox's access token: the page may reach that app and nothing else
            origin = "/".join(url.split("/")[:3])

            async def only_app(route):
                url = route.request.url
                await (route.continue_() if url.startswith(origin) or url.startswith(("data:", "blob:"))
                       else route.abort("blockedbyclient"))

            await context.route("**/*", only_app)
        if cookies:
            await context.add_cookies(cookies)
        page = await context.new_page()
        await page.goto(url, wait_until="load")
        if ready:
            await page.wait_for_selector(ready, timeout=120000)
            await page.wait_for_timeout(2500)  # a real app keeps loading fonts and its renderer after first paint
        else:
            await page.wait_for_function("window.designEditor && window.designEditor.ready === true",
                                         timeout=20000, polling=100)
        await self._settle(page)
        self._pages[episode_id] = (context, page)
        if ready:
            self._real.add(episode_id)
        else:
            self._real.discard(episode_id)
        return True

    async def _file_window(self, context, episode_id: str, cfg: dict):
        """When the page asks for a file, show the file window; what is opened in it goes to the page's input."""
        await context.add_init_script(cfg["script"])
        pending: dict = {}

        async def picked(source, ids):
            chooser = pending.pop("chooser", None)
            if chooser is not None and ids:
                try:
                    await chooser.set_files([cfg["files"][i] for i in ids if i in cfg["files"]])
                except Exception as exc:  # noqa: BLE001 - logged: the app simply gets no file
                    print(f"[gui] file window: handing the file to the page failed: {exc}", flush=True)

        async def closed(source):
            pending.pop("chooser", None)

        await context.expose_binding("__fileWindowPicked", picked)
        await context.expose_binding("__fileWindowClosed", closed)

        def on_page(page):
            async def on_chooser(chooser):
                pending["chooser"] = chooser
                await page.evaluate("cfg => window.__fileWindow.open(cfg)", cfg["page"])

            page.on("filechooser", lambda c: asyncio.ensure_future(on_chooser(c)))

        context.on("page", on_page)

    async def _close(self, episode_id: str):
        entry = self._pages.pop(episode_id, None)
        if entry:
            try:
                await entry[0].close()
            except Exception:
                pass

    def _page(self, episode_id: str):
        entry = self._pages.get(episode_id)
        if entry is None:
            raise RuntimeError("no editor page for this episode; call reset(surface='gui')")
        return entry[1]

    async def _settle(self, page, ms: int = 60):
        # Let the editor finish pending operations and repaint before the next
        # screenshot, so the agent never acts on a half-updated frame.
        try:
            await page.evaluate(
                "() => (window.designEditor && window.designEditor.whenIdle)"
                " ? window.designEditor.whenIdle(2000) : null"
            )
        except Exception:
            pass
        await page.wait_for_timeout(ms)

    # ------------------------------------------------------------ actions

    def attach_remote(self, episode_id: str, client) -> None:
        """Drive this episode through a remote screen (a desktop agent: realapps.desktop.client.DesktopClient)."""
        self._remote[episode_id] = client

    def detach_remote(self, episode_id: str) -> None:
        self._remote.pop(episode_id, None)

    def open(self, episode_id: str, url: str, cookies: list[dict] | None = None, ready: str | None = None,
             headers: dict[str, str] | None = None, init_script: str | None = None, file_window: dict | None = None):
        """Open the editor for an episode. A real app (realapps) passes its session cookies and a CSS selector
        that appears once its editor has loaded, instead of the built-in editor's ready flag, and, when it runs in
        a sandbox, the headers that admit the browser (requests to any other origin are then blocked). With a
        lived-in computer, `file_window` (server.file_window.config) answers the app's requests for a file."""
        return self.submit(self._open(episode_id, url, cookies, ready, headers, init_script, file_window))

    def close(self, episode_id: str):
        return self.submit(self._close(episode_id))

    async def screenshot(self, episode_id: str) -> bytes:
        if (r := self._remote.get(episode_id)) is not None:
            return await asyncio.to_thread(r.screenshot)
        async def run():
            page = self._page(episode_id)
            await self._settle(page, 30)
            return await page.screenshot(type="png")

        return await self.call(run())

    async def click(self, episode_id, x, y, button="left", clicks=1, modifiers=None):
        if (r := self._remote.get(episode_id)) is not None:
            return await asyncio.to_thread(r.input, type={1: "click", 2: "double", 3: "triple"}.get(clicks, "click") if button != "right" else "right", x=x, y=y, modifiers=[normalize_keys(m) for m in (modifiers or [])])
        async def run():
            page = self._page(episode_id)
            mods = [normalize_keys(m) for m in (modifiers or [])]
            for m in mods:
                await page.keyboard.down(m)
            try:
                await page.mouse.click(x, y, button=button, click_count=clicks, delay=20)
            finally:
                for m in reversed(mods):
                    await page.keyboard.up(m)
            await self._settle(page)

        return await self.call(run())

    async def move(self, episode_id, x, y):
        if (r := self._remote.get(episode_id)) is not None:
            return await asyncio.to_thread(r.input, type="move", x=x, y=y)
        async def run():
            page = self._page(episode_id)
            await page.mouse.move(x, y, steps=4)
            await self._settle(page, 30)

        return await self.call(run())

    async def drag(self, episode_id, sx, sy, ex, ey, modifiers=None):
        if (r := self._remote.get(episode_id)) is not None:
            return await asyncio.to_thread(r.input, type="drag", **{"from": [sx, sy], "to": [ex, ey]})
        async def run():
            page = self._page(episode_id)
            mods = [normalize_keys(m) for m in (modifiers or [])]
            await page.mouse.move(sx, sy)
            for m in mods:
                await page.keyboard.down(m)
            await page.mouse.down()
            steps = max(6, int(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5 / 25))
            await page.mouse.move(ex, ey, steps=steps)
            if episode_id in self._real:
                # a hand rests at the drop point before letting go; a real app that takes the drop position from
                # its last (throttled) dragover would otherwise drop at a point midway along the path (seen in web apps)
                await page.wait_for_timeout(150)
                await page.mouse.move(ex, ey)
                await page.wait_for_timeout(50)
            await page.mouse.up()
            for m in reversed(mods):
                await page.keyboard.up(m)
            await self._settle(page, 90)

        return await self.call(run())

    async def scroll(self, episode_id, x, y, dx, dy, modifiers=None):
        if (r := self._remote.get(episode_id)) is not None:
            return await asyncio.to_thread(r.input, type="scroll", x=x, y=y, dx=round(dx / 100), dy=round(dy / 100))
        async def run():
            page = self._page(episode_id)
            await page.mouse.move(x, y)
            mods = [normalize_keys(m) for m in (modifiers or [])]
            for m in mods:
                await page.keyboard.down(m)
            await page.mouse.wheel(dx, dy)
            for m in reversed(mods):
                await page.keyboard.up(m)
            await self._settle(page, 80)

        return await self.call(run())

    async def type_text(self, episode_id, text: str):
        if (r := self._remote.get(episode_id)) is not None:
            return await asyncio.to_thread(r.input, type="type", text=text)
        async def run():
            page = self._page(episode_id)
            await page.keyboard.type(text, delay=8)
            await self._settle(page)

        return await self.call(run(), timeout=60)

    async def key(self, episode_id, combo: str):
        if (r := self._remote.get(episode_id)) is not None:
            return await asyncio.to_thread(r.input, type="key", keys=[normalize_keys(p) for p in combo.split()])
        async def run():
            page = self._page(episode_id)
            for part in combo.split():
                await page.keyboard.press(normalize_keys(part))
            await self._settle(page)

        return await self.call(run())

    async def wait(self, episode_id, seconds: float):
        if episode_id in self._remote:
            return await asyncio.sleep(max(0.0, min(seconds, 10.0)))
        async def run():
            page = self._page(episode_id)
            await page.wait_for_timeout(int(max(0.0, min(seconds, 10.0)) * 1000))

        return await self.call(run(), timeout=15)
