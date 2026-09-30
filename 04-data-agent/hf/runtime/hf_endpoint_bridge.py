"""Sandbox-local bridge for authenticated HF Jobs capture endpoints."""
import hashlib
import http.client
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "host"}


def transport_key(token):
    return hashlib.sha256(("openenv-hf-transport-v1:" + token).encode()).hexdigest()


class Bridge(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        # Never retry a model request: replaying it can duplicate captured turns.
        target = urlsplit(os.environ["HF_CAPTURE_URL"])
        if target.scheme != "https" or not target.hostname.endswith(".hf.jobs"):
            self.send_error(502, "Invalid capture endpoint")
            return
        if self.headers.get("Transfer-Encoding"):
            self.send_error(400, "Request body must have Content-Length")
            return
        conn = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=620)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_HEADERS and not k.lower().startswith("x-openenv-forwarded-")
                   and k.lower() != "x-openenv-transport-key"}
        for key in ("authorization", "x-api-key", "x-goog-api-key"):
            value = self.headers.get(key)
            if value:
                headers["x-openenv-forwarded-" + key] = value
            headers = {k: v for k, v in headers.items() if k.lower() != key}
        token = os.environ["HF_TRANSPORT_TOKEN"]
        headers["Authorization"] = "Bearer " + token
        headers["X-OpenEnv-Transport-Key"] = transport_key(token)
        headers["Connection"] = "close"
        started = False
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            conn.request(self.command, target.path.rstrip("/") + self.path, body=body, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            started = True
            while data := response.read1(65536):
                self.wfile.write(data)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not started:
                self.send_error(502, "Capture transport failed")
        finally:
            conn.close()
            self.close_connection = True


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 12121), Bridge).serve_forever()
