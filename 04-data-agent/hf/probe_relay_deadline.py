"""Measure a share relay's delayed-header/body behavior without model requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
import time
import uuid

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forwarding-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--delay', type=float, default=75)
    args = parser.parse_args()
    assert 1 <= args.delay <= 120
    args.out.mkdir(parents=True, exist_ok=True)
    instance = uuid.uuid4().hex
    events, lock = [], threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_GET(self):
            data = json.dumps({'instance': instance}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            path = self.path.removeprefix('/' + instance)
            if path not in ('/delayed', '/headers', '/heartbeats') or not self.path.startswith('/' + instance):
                self.send_error(404)
                return
            began = time.monotonic()
            try:
                if path == '/delayed':
                    time.sleep(args.delay)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Connection', 'close')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                self.wfile.flush()
                if path == '/headers':
                    time.sleep(args.delay)
                if path == '/heartbeats':
                    while time.monotonic() - began < args.delay:
                        self.wfile.write(b'\n')
                        self.wfile.flush()
                        time.sleep(min(5, args.delay - (time.monotonic() - began)))
                self.wfile.write(json.dumps({'instance': instance, 'mode': path}).encode())
                self.wfile.flush()
                error = None
            except (BrokenPipeError, ConnectionResetError) as exc:
                error = type(exc).__name__
            finally:
                self.close_connection = True
                with lock:
                    events.append({'mode': path, 'elapsed': time.monotonic() - began, 'error': error})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    spec = importlib.util.spec_from_file_location('probe_forwarder', args.forwarding_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forwarder = module.GradioForwarder()
    try:
        url = forwarder.start(server.server_port)
        def health():
            response = httpx.get(url + '/health', timeout=15)
            return {'status': response.status_code, 'same_instance': response.status_code == 200 and response.json().get('instance') == instance}
        before = health()
        def request(mode):
            start = time.monotonic()
            try:
                response = httpx.post(url + '/' + instance + '/' + mode, content=b'', timeout=args.delay + 30)
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                return {'mode': mode, 'seconds': time.monotonic() - start, 'status': response.status_code,
                        'same_instance': payload.get('instance') == instance,
                        'no_interface_html': 'No interface is running' in response.text,
                        'response_bytes': len(response.content)}
            except httpx.RequestError as exc:
                return {'mode': mode, 'seconds': time.monotonic() - start, 'error': type(exc).__name__}
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(request, ['delayed', 'headers', 'heartbeats']))
        report = {'delay_seconds': args.delay, 'forwarding_sha256': hashlib.sha256(args.forwarding_source.read_bytes()).hexdigest(),
                  'health_before': before, 'results': results, 'health_after': health(), 'server_events': events}
        (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        forwarder.stop()
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
