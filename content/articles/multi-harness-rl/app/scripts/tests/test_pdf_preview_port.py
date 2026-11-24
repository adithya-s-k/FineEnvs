"""Exercise each export CLI through a real npm/Astro preview, stopping at readiness.

Run with PDF_PREVIEW_NODE_MODULES pointing to installed Astro and Playwright,
and optionally PDF_NODE_BINARY pointing to Node. No article build or PDF needed.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
MODULES = os.environ.get("PDF_PREVIEW_NODE_MODULES")
NODE = os.environ.get("PDF_NODE_BINARY") or shutil.which("node")


def fetch(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.2) as response:
            return response.read().decode()
    except (OSError, urllib.error.URLError):
        return None


@pytest.mark.parametrize("script", ["export-pdf.mjs", "export-pdf-book.mjs", "export-pdf-book-simple.mjs"])
@pytest.mark.parametrize("port", [None, 8081], ids=["default-port", "documented-custom-port"])
def test_export_cli_reaches_its_selected_preview(tmp_path, script, port):
    if not NODE or not MODULES:
        pytest.skip("Set PDF_PREVIEW_NODE_MODULES to real Astro/Playwright installation")
    modules = Path(MODULES).resolve()
    for name in ("astro", "playwright"):
        if not (modules / name / "package.json").exists():
            pytest.skip(f"Missing native dependency: {name}")
    for number in (8080, 8081):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", number))
            except OSError:
                pytest.skip(f"Native preview control requires free port {number}")

    app = tmp_path / "app"
    app.mkdir()
    (app / "node_modules").symlink_to(modules, target_is_directory=True)
    package = json.loads((SCRIPTS.parent / "package.json").read_text())
    (app / "package.json").write_text(json.dumps({
        "type": "module", "scripts": {"preview": package["scripts"]["preview"]},
    }))
    (app / "astro.config.mjs").write_text("export default { output: 'static' };\n")
    (app / "dist").mkdir()
    marker = "Owned native article preview fixture"
    (app / "dist" / "index.html").write_text(f"<!doctype html><title>Article</title><main>{marker}</main>")
    shutil.copyfile(SCRIPTS / script, app / script)

    # Record the owned detached preview group while forwarding every argument
    # unchanged to the installed npm CLI, which runs the real Astro producer.
    npm_cli = Path(NODE).resolve().parent.parent / "lib/node_modules/npm/bin/npm-cli.js"
    if not npm_cli.exists():
        pytest.skip("Node installation must include npm")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    receipt = tmp_path / "preview.pid"
    argv_receipt = tmp_path / "preview.argv"
    wrapper = bin_dir / "npm"
    wrapper.write_text("#!/bin/sh\n"
                       + f"printf '%s\\n' \"$$\" > {shlex.quote(str(receipt))}\n"
                       + f"printf '%s\\n' \"$@\" > {shlex.quote(str(argv_receipt))}\n"
                       + f"exec {shlex.quote(NODE)} {shlex.quote(str(npm_cli))} \"$@\"\n")
    wrapper.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{Path(NODE).parent}:{os.environ.get('PATH', '')}", ASTRO_TELEMETRY_DISABLED="1")
    env.pop("PREVIEW_PORT", None)
    if port is not None:
        env["PREVIEW_PORT"] = str(port)
    process = subprocess.Popen([NODE, str(app / script)], cwd=app, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    reader = threading.Thread(target=lambda: output.extend(process.stdout), daemon=True)
    reader.start()
    expected = port or 8080
    served = False
    wrong_port_served = False
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            served |= marker in (fetch(expected) or "")
            if port is not None:
                wrong_port_served |= marker in (fetch(8080) or "")
            if served and "Server ready" in "".join(output):
                break
            if process.poll() is not None:
                reader.join(timeout=1)
                break
            # Wrong-port service is the positive producer control. Give the
            # selected endpoint several additional polls before reporting it.
            if wrong_port_served and time.monotonic() > deadline - 8:
                break
            time.sleep(0.05)
        arguments = argv_receipt.read_text().splitlines() if argv_receipt.exists() else []
        assert served and "Server ready" in "".join(output), (
            f"{script}: requested={expected}, selected_served={served}, "
            f"8080_served={wrong_port_served}, preview_argv={arguments}\n{''.join(output)}"
        )
        assert "preview" in arguments
    finally:
        if receipt.exists():
            try:
                os.killpg(int(receipt.read_text()), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # The exporter may already have reaped its preview group.
                pass
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=1)
        process.stdout.close()
        for number in (8080, expected):
            assert marker not in (fetch(number) or ""), "Owned preview process leaked after cleanup"
