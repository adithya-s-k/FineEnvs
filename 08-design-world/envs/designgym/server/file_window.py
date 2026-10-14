"""The computer's file window for a real web app (Penpot, draw.io) on a lived-in computer.

A real web app brings images in through the browser: its upload button or image tool opens the operating system's
file window. A headless browser has none to show, so when such an app asks for a file (Playwright's `filechooser`
event) the page gets the same file window the built-in editors draw (web/js/files.js), in the look of the desktop the
browser runs on (GNOME), holding the episode's workspace (core.workspace): everyday names in everyday folders, the
design's pictures among other pictures and documents. What is opened in it goes to the app's own file input, under
the file's name, exactly as a picked file would; nothing else about the app changes.

    cfg = config(ep)                      # what the window shows, for this episode
    GuiDriver.get().open(..., file_window=cfg)
"""

from __future__ import annotations

import base64
import io
import re
from functools import lru_cache
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}


@lru_cache(maxsize=1)
def bundle() -> str:
    """files.js and what it imports (util.js, icons.js) as one classic script, plus its stylesheet, exposing
    `window.__fileWindow.open(cfg)`."""
    parts = []
    for name in ("util.js", "icons.js", "files.js"):
        src = (WEB / "js" / name).read_text()
        src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M)
        src = re.sub(r"^export (default )?", "", src, flags=re.M)
        parts.append(src)
    css = (WEB / "css" / "files.css").read_text() + (
        "\n.fc { font-family: Cantarell, Ubuntu, 'Noto Sans', 'DejaVu Sans', system-ui, sans-serif; font-size: 13px;"
        " line-height: normal; letter-spacing: normal; text-transform: none; }\n"
        ".fc * { box-sizing: border-box; }\n")
    glue = r"""
window.__fileWindow = {
  app: null,
  open(cfg) {
    let app = this.app;
    if (!app) {
      app = this.app = { params: { os: cfg.os || "linux", fs: cfg.episodeId }, lastPlace: null, fileChooser: null,
        stage: { dropPoint: () => null }, refreshAll() {},
        store: { episodeId: cfg.episodeId, workspace: cfg.workspace, assets: cfg.assets } };
      const style = document.createElement("style");
      style.textContent = cfg.css;
      document.head.append(style);
    }
    let picked = false;  // (Open closes the window first, then hands over the file: a close is a cancel only after)
    const done = (assets) => { picked = true; return window.__fileWindowPicked(assets.map((a) => a.id)); };
    const w = openFileChooser(app, { title: cfg.title || "Open File", action: "Open", multiple: false,
      onOpen: done, onDrop: null });
    const close = w.close.bind(w);
    w.close = () => { close(); setTimeout(() => { if (!picked) window.__fileWindowClosed(); }, 0); };
    return true;
  },
  isOpen() { return !!(this.app && this.app.fileChooser && this.app.fileChooser.el.isConnected); },
};
"""
    return "(() => {\n" + "\n".join(parts) + glue + "\n})();", css


def _thumb(path: Path) -> str:
    """A picture as a data URL the page can show (at most 640 px: the window's icons and preview)."""
    from PIL import Image

    img = Image.open(path)
    img.thumbnail((640, 640))
    buf = io.BytesIO()
    if img.mode in ("RGBA", "LA", "P"):
        img.convert("RGBA").save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    img.convert("RGB").save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def config(ep, os: str = "linux", title: str = "Open File") -> dict | None:
    """What the episode's file window shows (None without a workspace), and the files it hands to the app."""
    if not ep.workspace:
        return None
    try:
        from ..core import workspace as ws
    except ImportError:  # flat layout inside the Docker image
        from core import workspace as ws
    layout = ws.layout(ep.workspace)
    kinds = {a["id"]: a["kind"] for a in ep.task["assets"]}
    kinds.update({k: c.get("kind", "graphic") for k, c in ep.ctx.assets.items() if k not in kinds})
    assets = [{"id": aid, "kind": kinds.get(aid, "graphic"), "url": _thumb(Path(p)), **ws.file_facts(p)}
              for aid, p in ep.assets.items()]
    files = {}
    for f in layout["files"]:  # the bytes the app's file input gets: the file as it is on disk, under its name
        p = Path(ep.assets[f["asset"]])
        ext = f["name"].rsplit(".", 1)[-1].lower()
        data = p.read_bytes()
        # the bytes are what the name says (an app checks): a .jpg is a JPEG (on white: no alpha), a .png a PNG
        if ext in ("jpg", "jpeg") and not data[:3] == b"\xff\xd8\xff":
            from PIL import Image

            img = Image.open(io.BytesIO(data)).convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.getchannel("A"))
            buf = io.BytesIO()
            bg.save(buf, "JPEG", quality=92)
            data = buf.getvalue()
        elif ext == "png" and not data[:8] == b"\x89PNG\r\n\x1a\n":
            from PIL import Image

            buf = io.BytesIO()
            Image.open(io.BytesIO(data)).save(buf, "PNG")
            data = buf.getvalue()
        files[f["asset"]] = {"name": f["name"], "mimeType": _MIME.get(ext, "application/octet-stream"),
                             "buffer": data}
    script, css = bundle()
    return {"script": script, "page": {"episodeId": ep.episode_id, "workspace": layout, "assets": assets, "os": os,
                                       "title": title, "css": css}, "files": files}
