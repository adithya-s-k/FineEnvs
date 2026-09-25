"""Real HTTP response-contract checks without importing the GPU inference stack."""
import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import unittest

import requests

WORKSPACE = Path(__file__).resolve().parents[4]


class ControlHTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        source = Path(importlib.util.find_spec("trl").origin).parent / "generation/vllm_client.py"
        tree = ast.parse(source.read_text())
        client = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VLLMClient")
        # Execute the production HTTP methods verbatim; constructors require GPUs.
        client.body = [node for node in client.body if isinstance(node, ast.FunctionDef)
                       and node.name in {"_post", "reset_prefix_cache"}]
        namespace = {}
        exec(compile(ast.Module(body=[client], type_ignores=[]), str(source), "exec"), namespace)
        cls.client_type = namespace["VLLMClient"]

    def setUp(self):
        self.status, self.body, self.paths = 200, b"", []
        case = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                case.paths.append(self.path)
                self.send_response(case.status)
                self.send_header("Content-Length", str(len(case.body)))
                self.end_headers()
                self.wfile.write(case.body)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = self.client_type()
        self.client.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.client.session = requests.Session()

    def tearDown(self):
        self.client.session.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_empty_reset_success(self):
        self.assertIsNone(self.client.reset_prefix_cache())
        self.assertEqual(self.paths, ["/reset_prefix_cache"])

    def test_reset_failure_is_not_suppressed(self):
        self.status, self.body = 500, b"cache reset failed"
        with self.assertRaisesRegex(Exception, "500, cache reset failed"):
            self.client.reset_prefix_cache()

    def test_structured_response_still_requires_json(self):
        with self.assertRaises(requests.exceptions.JSONDecodeError):
            self.client._post(self.client.base_url + "/v1/completions")
        self.body = b'{"choices": []}'
        self.assertEqual(self.client._post(self.client.base_url + "/v1/completions"), {"choices": []})


if __name__ == "__main__":
    unittest.main()
