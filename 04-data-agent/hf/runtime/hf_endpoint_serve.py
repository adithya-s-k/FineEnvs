"""Isolated qualification adapter; the frozen production bundle stays unchanged."""
import os
from pathlib import Path
import runpy

from hf_endpoint_auth import CaptureTransportAuth


def install():
    from openenv.core.harness.capture import forwarding, runner
    from openenv.harbor.seams import Seam
    from openenv.harbor.e2b_stream import E2BStreamingEnvironment

    endpoint = f"https://{os.environ['JOB_ID']}--8201.hf.jobs"
    token = os.environ["HF_TOKEN"]
    original_init = runner.CaptureServer.__init__

    def init(self, **kwargs):
        original_init(self, **kwargs)
        self.app.add_middleware(CaptureTransportAuth, token=token)

    runner.CaptureServer.__init__ = init

    class Forwarder:
        def start(self, port):
            if port != 8201:
                raise ValueError("Qualification exposes only capture port 8201")
            return endpoint

        def stop(self):
            pass

    forwarding.make_forwarder = lambda kind: Forwarder()
    original_resolve = Seam.resolve

    def resolve(self, **kwargs):
        kwargs["base_url"] = "http://127.0.0.1:12121"
        return original_resolve(self, **kwargs)

    Seam.resolve = resolve
    original_start = E2BStreamingEnvironment.start
    bridge = Path(__file__).with_name("hf_endpoint_bridge.py").read_text()

    async def start(self, force_build):
        await original_start(self, force_build)
        await self._sandbox.files.write("/tmp/openenv_hf_bridge.py", bridge)
        await self._sandbox.commands.run("python /tmp/openenv_hf_bridge.py", background=True,
            envs={"HF_TRANSPORT_TOKEN": token, "HF_CAPTURE_URL": endpoint}, timeout=0)
        health = """import time, urllib.request
deadline = time.monotonic() + 30
while True:
    try:
        urllib.request.urlopen('http://127.0.0.1:12121/health', timeout=5).read()
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise RuntimeError('Bridge readiness deadline exceeded') from None
        time.sleep(0.2)
"""
        await self._sandbox.files.write("/tmp/openenv_hf_health.py", health)
        result = await self._sandbox.commands.run("python /tmp/openenv_hf_health.py", timeout=40)
        if result.exit_code:
            raise RuntimeError("Sandbox capture bridge failed its health check")

    E2BStreamingEnvironment.start = start


if __name__ == "__main__":
    install()
    runpy.run_module("openenv.cli", run_name="__main__")
