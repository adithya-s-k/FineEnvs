"""Real apps on Daytona, run by the env server itself.

Clients connect to one DesignGym server, as always. When an episode needs a real app, the server takes a sandbox
from a pool it owns and talks to it over the sandbox's preview URL; the access token travels in a header, never in a
URL, and the env's browser may reach that app and nothing else (server/gui.py):

    desktop   one sandbox per concurrent episode (one X display each), from a snapshot of realapps/desktop/Dockerfile
              (or the image a plugin registers: registry.register_desktop); the agent (agent.py, plus any plugin's
              agent modules) and the task pack's fonts are uploaded the first time a sandbox is taken

A plugin's web app can keep its own shared sandbox here too: `DaytonaApps.sandboxes(role)`, `link(sb, port)`,
`claim`/`release` (kept awake while an episode runs) and `sh(sb, cmd)` are what it needs.

Sandboxes carry the labels {"project": "design-canvas", "role": ...} (plus "pool" when DESIGN_CANVAS_DAYTONA_POOL
names a separate pool) and outlive the server: the next server of the same pool reuses
them (starting stopped ones). While an episode runs (up to DESIGN_CANVAS_DAYTONA_MAX_EPISODE_MIN, default 120) the
server keeps its sandbox awake; idle ones stop by themselves after DESIGN_CANVAS_DAYTONA_AUTO_STOP minutes (default
15) and are started again when needed.

Turn it on for a server with DESIGN_CANVAS_APPS=daytona (and DAYTONA_API_KEY); DESIGN_CANVAS_DAYTONA_DESKTOPS sets the
number of desktops (default 2). To start them ahead of a run, check on them, or delete them:

    python -m designgym.realapps.daytona up --desktops 4 [--apps]
    python -m designgym.realapps.daytona status
    python -m designgym.realapps.daytona down
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import registry
from .desktop.client import DesktopBackend, DesktopClient

HERE = Path(__file__).parent
PROJECT = {"project": "design-canvas"}


def _pool() -> str | None:
    """DESIGN_CANVAS_DAYTONA_POOL names a separate pool (e.g. "space" for the public Space): its sandboxes carry
    that name as a label, so two servers on different machines never take each other's desktops. Unset = the
    default pool, the sandboxes with no pool label."""
    return os.environ.get("DESIGN_CANVAS_DAYTONA_POOL") or None


def _labels(**extra) -> dict:
    return {**PROJECT, **({"pool": _pool()} if _pool() else {}), **extra}
AGENT_PORT = 5005
HOME = "/home/user"  # the desktop image's user (realapps/desktop/Dockerfile)


def _auto_stop() -> int:
    return int(os.environ.get("DESIGN_CANVAS_DAYTONA_AUTO_STOP", "15"))


def _log(msg: str) -> None:
    print(f"[daytona] {msg}", file=sys.stderr, flush=True)


def _fingerprint(*parts: bytes) -> str:
    return hashlib.sha256(b"".join(parts)).hexdigest()[:10]


def sh(sb, cmd: str, timeout: int = 600, check: bool = True) -> str:
    # exec does not go through a shell: ship the command as a script and run that
    path = f"/tmp/dc-{secrets.token_hex(4)}.sh"
    sb.fs.upload_file(("set -e\n" + cmd + "\n").encode(), path)
    r = sb.process.exec(f"sh {path}", timeout=timeout)
    if check and r.exit_code != 0:
        raise RuntimeError(f"sandbox {sb.id}: command failed ({r.exit_code}): {cmd[:200]}\n{r.result[-2000:]}")
    return r.result


class DaytonaApps:
    """The server's Daytona sandboxes: finding, creating, starting and preparing them."""

    def __init__(self):
        from daytona import Daytona

        self.d = Daytona()
        self.busy: dict[str, int] = {}  # sandbox id -> episodes running on it (kept awake)
        self.claimed: dict[str, float] = {}  # sandbox id -> when its latest episode started
        self.lock = threading.Lock()
        threading.Thread(target=self._keep_awake, daemon=True).start()

    # ------------------------------------------------------------------ bookkeeping

    def sandboxes(self, role: str | None = None) -> list:
        from daytona import ListSandboxesQuery

        labels = _labels(**({"role": role} if role else {}))
        return [s for s in self.d.list(ListSandboxesQuery(labels=labels))
                if s.labels.get("pool") == _pool()  # a label query also matches other pools' sandboxes
                and str(s.state).split(".")[-1].lower() not in ("destroyed", "destroying", "error")]

    def _state(self, sb) -> str:
        sb.refresh_data()
        return str(sb.state).split(".")[-1].lower()

    def _started(self, sb) -> None:
        if self._state(sb) != "started":
            _log(f"starting {sb.labels.get('role')} sandbox {sb.id}")
            sb.start(timeout=300)

    def claim(self, sb) -> None:
        with self.lock:
            self.busy[sb.id] = self.busy.get(sb.id, 0) + 1
            self.claimed[sb.id] = time.time()
        sb.refresh_activity()

    def release(self, sb) -> None:
        with self.lock:
            self.busy[sb.id] = max(0, self.busy.get(sb.id, 0) - 1)

    def _keep_awake(self) -> None:
        # an episode nobody closes (a playground tab left open) stops being kept awake after the longest real
        # episode, DESIGN_CANVAS_DAYTONA_MAX_EPISODE_MIN (default 120); its sandbox then stops when idle
        max_age = 60 * int(os.environ.get("DESIGN_CANVAS_DAYTONA_MAX_EPISODE_MIN", "120"))
        while True:
            time.sleep(240)
            with self.lock:
                ids = [i for i, n in self.busy.items() if n > 0 and time.time() - self.claimed.get(i, 0) < max_age]
            for i in ids:
                try:
                    self.d.get(i).refresh_activity()
                except Exception as e:  # noqa: BLE001 - the next tick tries again
                    _log(f"keep-awake {i}: {e}")

    @staticmethod
    def link(sb, port: int) -> tuple[str, dict[str, str]]:
        p = sb.get_preview_link(port)
        return p.url.rstrip("/"), {"x-daytona-preview-token": p.token, "X-Daytona-Skip-Preview-Warning": "true"}

    # ------------------------------------------------------------------ desktops

    def desktop_snapshot(self) -> str:
        """The snapshot desktops start from: the desktop image, built by Daytona once per Dockerfile version."""
        from daytona import CreateSnapshotParams, Image, Resources

        dockerfile = registry.desktop().dockerfile
        # DESIGN_CANVAS_DAYTONA_DESKTOP_MEM (GB, default 4): a desktop uses under 1 GB recording (LibreOffice,
        # ONLYOFFICE, Inkscape, Scribus measured 2026-10-13), so a big run can take smaller ones: its own snapshot
        mem = int(os.environ.get("DESIGN_CANVAS_DAYTONA_DESKTOP_MEM", "4"))
        name = "design-canvas-desktop-" + _fingerprint(dockerfile.read_bytes()) + (f"-m{mem}" if mem != 4 else "")
        try:
            snap = self.d.snapshot.get(name)
            if str(snap.state).split(".")[-1].lower() == "active":
                return name
            if str(snap.state).split(".")[-1].lower() == "error":
                self.d.snapshot.delete(snap)
            else:  # building elsewhere: wait for it
                for _ in range(360):
                    time.sleep(10)
                    if str(self.d.snapshot.get(name).state).split(".")[-1].lower() == "active":
                        return name
        except Exception:  # noqa: BLE001 - not there yet
            pass
        _log(f"building snapshot {name} from {dockerfile} (once; several minutes)")
        self.d.snapshot.create(
            CreateSnapshotParams(name=name, image=Image.from_dockerfile(dockerfile),
                                 resources=Resources(cpu=2, memory=mem, disk=8),
                                 # the agent is uploaded and started by the server, not the image
                                 entrypoint=["sh", "-c", "sleep infinity"]),
            on_logs=lambda line: None)
        return name

    def desktops(self, n: int, fonts_dir: Path | None) -> list:
        """n desktop sandboxes, started and running the agent (created as needed, in parallel)."""
        from daytona import CreateSandboxFromSnapshotParams

        snap = self.desktop_snapshot()
        have = [s for s in self.sandboxes("desktop") if s.labels.get("snapshot") == snap][:n]

        def create(_):
            return self.d.create(CreateSandboxFromSnapshotParams(
                snapshot=snap, labels=_labels(role="desktop", snapshot=snap),
                auto_stop_interval=_auto_stop(), public=False), timeout=600)

        def prepare(sb):
            # a desktop another process has an episode on (its lock is held) must not have its agent restarted
            # under it: only make sure it answers; it gets this agent.py the next time it is free at a start
            fd = DesktopBackend._lock(sb.id)
            try:
                self.prepare_desktop(sb, fonts, check_agent=fd is not None)
            finally:
                if fd is not None:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)

        with ThreadPoolExecutor(max_workers=8) as pool:
            have += list(pool.map(create, range(n - len(have))))
            fonts = _fonts_bundle(fonts_dir)
            if have:  # the first one first: the rest copy its fonts rather than all uploading them at once
                prepare(have[0])
            list(pool.map(prepare, have[1:]))
        return have

    def prepare_desktop(self, sb, fonts: tuple[str, bytes] | None, check_agent: bool = False) -> None:
        """Make sure the sandbox is started and its agent answers (after a stop, processes are gone); with
        check_agent, also that it runs this agent.py."""
        import fcntl

        # one process at a time prepares a given desktop (several servers start together: their uploads and agent
        # restarts would otherwise land on top of each other)
        lock_dir = Path(os.environ.get("DESIGN_CANVAS_LOCK_DIR", "/tmp/design-canvas-locks"))
        lock_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_dir / f"prepare-{sb.id}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._prepare_desktop(sb, fonts, check_agent)

    def _prepare_desktop(self, sb, fonts, check_agent: bool) -> None:
        self._started(sb)
        url, headers = self.link(sb, AGENT_PORT)
        agent = (HERE / "desktop" / "agent.py").read_bytes()
        modules = {p.name: p.read_bytes() for p in registry.desktop().agent_modules}  # plugins' agent-side apps
        key = _fingerprint(agent, *(k.encode() + v for k, v in sorted(modules.items())))
        if _healthy(url, headers) and (not check_agent or sh(
                sb, f"cat {HOME}/agent.key 2>/dev/null || true", check=False).strip() == key):
            return
        sb.fs.upload_file(agent, f"{HOME}/agent.py")
        # exactly this server's app modules: none left from a server with other plugins
        sh(sb, f"rm -rf {HOME}/apps && mkdir -p {HOME}/apps", check=False)
        for name, data in modules.items():
            sb.fs.upload_file(data, f"{HOME}/apps/{name}")
        sb.fs.upload_file(key.encode(), f"{HOME}/agent.key")
        if fonts and sh(sb, f"cat {HOME}/fonts/.key 2>/dev/null || true", check=False).strip() != fonts[0]:
            if not self._fonts_from_donor(sb, fonts[0]):  # one upload per pool; the others copy a desktop's
                tgz = f"{HOME}/fonts-{os.getpid()}.tgz"
                sb.fs.upload_file(fonts[1], tgz)
                sh(sb, f"rm -rf {HOME}/fonts && mkdir -p {HOME}/fonts && tar xzf {tgz} -C {HOME}/fonts && "
                       f"rm {tgz} && echo {fonts[0]} > {HOME}/fonts/.key", timeout=600)
        sh(sb, f"pkill -f 'agent.py --port' || true\n"
               f"cd {HOME} && HOME={HOME} DESIGN_CANVAS_FONTS={HOME}/fonts nohup python3 {HOME}/agent.py "
               f"--port {AGENT_PORT} > /tmp/agent.log 2>&1 &", timeout=60)
        for _ in range(120):
            if _healthy(url, headers):
                _log(f"desktop {sb.id} ready")
                return
            time.sleep(2)
        raise RuntimeError(f"desktop agent in {sb.id} did not start: "
                           f"{sh(sb, 'tail -20 /tmp/agent.log', check=False)}")

    def _donors(self) -> list:
        """Desktops to copy fonts from: this pool's first, then those of the project's other pools (a new pool,
        e.g. an eval server's, starts with none; another pool's desktop with the same fonts serves them as well)."""
        from daytona import ListSandboxesQuery

        own = self.sandboxes("desktop")
        others = [s for s in self.d.list(ListSandboxesQuery(labels={**PROJECT, "role": "desktop"}))
                  if s.labels.get("pool") != _pool() and str(s.state).split(".")[-1].lower() == "started"]
        return own + others

    def _fonts_from_donor(self, sb, key: str) -> bool:
        """Copy the fonts from a desktop of the pool that has this version (its agent serves them), instead of
        uploading them from here: the bundle is tens of MB, and uploads from a laptop are slow."""
        for donor in self._donors():
            if donor.id == sb.id or str(donor.state).split(".")[-1].lower() != "started":
                continue
            url, headers = self.link(donor, AGENT_PORT)
            try:
                with urllib.request.urlopen(urllib.request.Request(url + "/health", headers=headers), timeout=15) as r:
                    if json.loads(r.read()).get("fonts") != key:
                        continue
            except Exception:  # noqa: BLE001
                continue
            fetch = ("import sys, urllib.request\n"
                     f"req = urllib.request.Request({url + '/fonts.tgz'!r}, headers={headers!r})\n"
                     "sys.stdout.buffer.write(urllib.request.urlopen(req, timeout=600).read())\n")
            sb.fs.upload_file(fetch.encode(), "/tmp/dc-fetch-fonts.py")
            try:
                sh(sb, f"rm -rf {HOME}/fonts && mkdir -p {HOME}/fonts && python3 /tmp/dc-fetch-fonts.py | "
                       f"tar xz -C {HOME}/fonts && rm -f /tmp/dc-fetch-fonts.py && test \"$(cat {HOME}/fonts/.key)\" = {key}",
                   timeout=700)
                _log(f"desktop {sb.id}: fonts copied from {donor.id}")
                return True
            except RuntimeError:
                continue
        return False

    def down(self) -> list[str]:
        gone = []
        for sb in self.sandboxes():
            sb.delete()
            gone.append(sb.id)
        return gone


def _healthy(url: str, headers: dict[str, str]) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url if "/api/" in url else url + "/health",
                                                           headers=headers), timeout=15) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _ok(url: str, headers: dict[str, str]) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


_FONTS: dict = {}


def _fonts_bundle(fonts_dir: Path | None) -> tuple[str, bytes] | None:
    """The pack's fonts as a tar.gz and a key naming that version (uploaded once per desktop)."""
    if fonts_dir is None or not Path(fonts_dir).is_dir():
        return None
    from .fonts import app_fonts

    fonts_dir = app_fonts(Path(fonts_dir))  # each file named for the family the designs use
    if fonts_dir not in _FONTS:
        manifest = fonts_dir / "manifest.json"
        key = _fingerprint(manifest.read_bytes() if manifest.exists() else str(sorted(os.listdir(fonts_dir))).encode())
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in sorted(fonts_dir.iterdir()):
                tar.add(p, arcname=p.name)
        _FONTS[fonts_dir] = (key, buf.getvalue())
    return _FONTS[fonts_dir]


# ---------------------------------------------------------------------- backends the env uses


class DaytonaDesktopBackend(DesktopBackend):
    """DesktopBackend over desktop sandboxes: each episode takes a free one, started and awake while it runs."""

    def __init__(self, apps: DaytonaApps, n: int, fonts_dir: Path | None):
        self.apps, self.fonts_dir = apps, fonts_dir
        pinned = [i for i in os.environ.get("DESIGN_CANVAS_DAYTONA_DESKTOP_IDS", "").split(",") if i]
        if pinned:  # development: these sandboxes only (e.g. a test desktop beside a running pool)
            boxes = [apps.d.get(i) for i in pinned]
            for sb in boxes:
                apps.prepare_desktop(sb, _fonts_bundle(fonts_dir), check_agent=True)
        else:
            boxes = apps.desktops(n, fonts_dir)
        self.boxes = {sb.id: sb for sb in boxes}
        super().__init__(urls=list(self.boxes))  # the pool is keyed by sandbox id

    def _client(self, key: str) -> DesktopClient:
        sb = self.boxes[key]
        self.apps.prepare_desktop(sb, _fonts_bundle(self.fonts_dir))  # it may have stopped while idle
        self.apps.claim(sb)
        return DesktopClient(*self.apps.link(sb, AGENT_PORT))

    def unwell(self, url: str) -> None:
        # a desktop whose reset hangs (a wedged app or agent) would take every next design and fail it the same
        # way: stop it, so its next use starts it again with a fresh agent
        if url in self.boxes:
            _log(f"desktop {url} did not reset: stopping it (its next use starts it again)")
            try:
                self.boxes[url].stop(timeout=300)
            except Exception as e:  # noqa: BLE001 - its next use tries again
                _log(f"stopping {url}: {e}")

    def release(self, url: str) -> None:
        if url in self.boxes:
            self.apps.release(self.boxes[url])
        super().release(url)


def main() -> int:
    ap = argparse.ArgumentParser(description="The env server's Daytona sandboxes for the real apps.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="create/start them ahead of a run (the server would do it on first use)")
    u.add_argument("--desktops", type=int, default=0)
    u.add_argument("--apps", action="store_true", help="also start plugins' shared sandboxes (their web apps)")
    u.add_argument("--fonts", default=None, help="fonts dir to upload (default: the task pack's)")
    sub.add_parser("status")
    sub.add_parser("down", help="delete every DesignGym sandbox of this pool")
    a = ap.parse_args()
    apps = DaytonaApps()
    if a.cmd == "up":
        fonts = a.fonts
        if fonts is None and a.desktops:
            from ..core.pack import load_pack

            fonts = load_pack().root / "fonts"
        if a.desktops:
            apps.desktops(a.desktops, Path(fonts) if fonts else None)
        if a.apps:
            for app in registry.apps().values():
                if app.prepare_daytona is not None:
                    app.prepare_daytona(apps)
    if a.cmd == "down":
        print(json.dumps({"deleted": apps.down()}))
        return 0
    for sb in apps.sandboxes():
        print(f"{sb.labels.get('role', '?'):8} {sb.id}  {str(sb.state).split('.')[-1].lower():9} "
              f"cpu {sb.cpu} mem {sb.memory}G  busy {apps.busy.get(sb.id, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
