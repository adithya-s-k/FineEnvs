"""Native HTTP regression tests for the article's shipped nginx configuration.

Run with NGINX_BINARY=/path/to/nginx python -m unittest discover -s review/tests
-t review/tests -p test_nginx_cache.py. No Astro build or HF credentials are needed.
"""
import contextlib
import http.server
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Backend(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"review fixture")

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def article_server():
    binary = os.environ.get("NGINX_BINARY") or shutil.which("nginx")
    if not binary:
        raise unittest.SkipTest("set NGINX_BINARY or install nginx to run native HTTP tests")
    article = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="article-cache-") as scratch:
        root = Path(scratch)
        (root / "logs").mkdir()
        dist = root / "dist"
        (dist / "data").mkdir(parents=True)
        (dist / "_astro").mkdir()
        (dist / "index.html").write_text("<!doctype html><title>article fixture</title>")
        (dist / "data" / "training-results.json").write_text('{"revision":1}')
        (dist / "llms.txt").write_text("revision 1")
        (dist / "_astro" / "app.hash.js").write_text("console.log(1)")
        # Give browsers a meaningful Last-Modified age for heuristic caching.
        for path in dist.rglob("*"):
            if path.is_file():
                old = time.time() - 30 * 86400
                os.utime(path, (old, old))
        backend = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        worker = threading.Thread(target=backend.serve_forever, daemon=True)
        worker.start()
        port = unused_port()
        conf = (article / "nginx.conf").read_text()
        # Preserve all HTTP/cache directives; adapt only OS, paths and ports.
        conf = conf.replace("worker_processes auto;", "worker_processes 1;")
        if sys.platform == "darwin":
            conf = conf.replace("use epoll;", "use kqueue;")
        conf = conf.replace("pid /tmp/nginx.pid;", f"pid {root}/nginx.pid;")
        mime_paths = (Path("/etc/nginx/mime.types"),
                      Path(binary).resolve().parent.parent / "conf" / "mime.types")
        mime = next((path for path in mime_paths if path.is_file()), None)
        if mime is None:
            raise RuntimeError("cannot locate nginx's installed mime.types")
        conf = conf.replace("include /etc/nginx/mime.types;", f"include {mime};")
        conf = conf.replace("/tmp/access.log", str(root / "access.log"))
        conf = conf.replace("/tmp/error.log", str(root / "error.log"))
        conf = conf.replace("root /app/dist;", f"root {dist};")
        conf = conf.replace("listen 8080;", f"listen 127.0.0.1:{port};")
        conf = conf.replace("127.0.0.1:7861", f"127.0.0.1:{backend.server_port}")
        config = root / "nginx.conf"
        config.write_text(conf)
        command = [binary, "-p", str(root), "-c", str(config)]
        validation = subprocess.run(command + ["-t"], capture_output=True, text=True)
        if validation.returncode:
            backend.shutdown()
            backend.server_close()
            worker.join(timeout=5)
            raise RuntimeError(validation.stderr)
        process = subprocess.Popen(command + ["-g", "daemon off;"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(url + "/health", timeout=0.2).close()
                    break
                except (OSError, urllib.error.URLError):
                    if process.poll() is not None:
                        raise RuntimeError(process.communicate()[1].decode())
                    time.sleep(0.02)
            else:
                raise RuntimeError("nginx did not become ready")
            yield url, dist, root / "access.log"
        finally:
            process.terminate()
            process.communicate(timeout=5)
            backend.shutdown()
            backend.server_close()
            worker.join(timeout=5)


def get(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


class ArticleCacheTest(unittest.TestCase):
    def test_pages_data_and_llms_require_revalidation(self):
        with article_server() as (url, _, _):
            for path in ("/", "/data/training-results.json", "/llms.txt", "/chapter"):
                with self.subTest(path=path):
                    status, headers, _ = get(url + path)
                    self.assertEqual(status, 200)
                    self.assertEqual(headers.get("Cache-Control"), "no-cache")

    def test_unchanged_content_revalidates_with_304(self):
        with article_server() as (url, _, _):
            for path in ("/", "/data/training-results.json", "/llms.txt"):
                with self.subTest(path=path):
                    _, headers, _ = get(url + path)
                    status, cached_headers, body = get(
                        url + path, {"If-None-Match": headers["ETag"]})
                    self.assertEqual(status, 304)
                    self.assertEqual(body, b"")
                    self.assertEqual(cached_headers.get("Cache-Control"), "no-cache")

    def test_fingerprinted_assets_keep_immutable_policy(self):
        with article_server() as (url, _, _):
            status, headers, _ = get(url + "/_astro/app.hash.js")
            self.assertEqual(status, 200)
            self.assertIn("immutable", ",".join(headers.get_all("Cache-Control")))
            self.assertIn("max-age=2592000", ",".join(headers.get_all("Cache-Control")))

    def test_review_and_oauth_responses_remain_no_store(self):
        with article_server() as (url, _, _):
            for path in ("/api/review/threads", "/oauth/userinfo"):
                with self.subTest(path=path):
                    status, headers, body = get(url + path)
                    self.assertEqual(status, 200)
                    self.assertEqual(body, b"review fixture")
                    self.assertEqual(headers.get("Cache-Control"), "no-store")


if __name__ == "__main__":
    unittest.main()
