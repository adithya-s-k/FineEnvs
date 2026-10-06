"""Fixture server for working on the viewer without the env server: same routes as server/api.py.

    python dev_server.py 8111     ->  http://127.0.0.1:8111/viewer/
"""

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB = Path(__file__).resolve().parent
FIX = WEB / "fixtures"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(WEB), **k)

    def _json(self, path: Path):
        if not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p == "/":
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        if p.startswith("/viewer/"):
            self.path = p[len("/viewer"):] + (("?" + u.query) if u.query else "")
            return super().do_GET()
        parts = [x for x in p.split("/") if x]
        if parts[:2] == ["api", "tasks"]:
            if len(parts) == 2:
                return self._json(FIX / "tasks.json")
            if len(parts) == 3:
                return self._json(FIX / f"task_{parts[2]}.json")
            if len(parts) == 4 and parts[3] == "reference":
                return self._json(FIX / f"reference_{parts[2]}.json")
        if parts[:2] == ["api", "episodes"]:
            if len(parts) == 2:
                return self._json(FIX / "episodes.json")
            if len(parts) == 3:
                return self._json(FIX / f"episode_{parts[2]}.json")
        if parts[:2] == ["api", "runs"]:
            if len(parts) == 2:
                return self._json(FIX / "runs.json")
            if len(parts) == 3:
                return self._json(FIX / f"run_{parts[2]}.json")
            if len(parts) == 4 and parts[3] == "episode":
                q = parse_qs(u.query)  # ?model=...&task_id=... (fixtures hold one episode)
                return self._json(FIX / "rollout_fixture.json")
        self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8111
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
