"""Time the real export CLI optional-library phase with native Astro/Chromium.

Set PDF_PREVIEW_NODE_MODULES, PDF_NODE_BINARY, PLAYWRIGHT_CHROME and
PDF_LIBRARY_ASSETS (d3.min.js 7.9.0 and plotly.min.js 2.35.2). The assets are
real library distributions served locally; no wait/library result is mocked.
Assets used: https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js and
https://cdn.plot.ly/plotly-basic-2.35.2.min.js. No Astro build is performed.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["export-pdf.mjs", "export-pdf-book.mjs", "export-pdf-book-simple.mjs"])
@pytest.mark.parametrize("plotly_ready", [False, True], ids=["article-without-plotly", "both-libraries-ready"])
def test_export_cli_respects_library_timeout(tmp_path, script, plotly_ready):
    node = os.environ.get("PDF_NODE_BINARY") or shutil.which("node")
    modules = os.environ.get("PDF_PREVIEW_NODE_MODULES")
    assets = os.environ.get("PDF_LIBRARY_ASSETS")
    chrome = os.environ.get("PLAYWRIGHT_CHROME")
    if not all((node, modules, assets, chrome)):
        pytest.skip("Provide native Node/Astro/Playwright/Chromium and real library assets")
    modules, assets = Path(modules).resolve(), Path(assets).resolve()
    for name in ("astro", "playwright", "pagedjs"):
        if not (modules / name / "package.json").exists():
            pytest.skip(f"Missing native dependency: {name}")
    for name in ("d3.min.js", "plotly.min.js"):
        if not (assets / name).exists():
            pytest.skip(f"Missing real library asset: {name}")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", 8080))
        except OSError:
            pytest.skip("Native default preview control requires free port 8080")

    app = tmp_path / "app"
    (app / "scripts").mkdir(parents=True)
    (app / "dist").mkdir()
    (app / "src/styles").mkdir(parents=True)
    (app / "node_modules").symlink_to(modules, target_is_directory=True)
    shutil.copyfile(SCRIPTS / script, app / "scripts" / script)
    shutil.copyfile(SCRIPTS.parent / "src/styles/_print-book.css", app / "src/styles/_print-book.css")
    package = json.loads((SCRIPTS.parent / "package.json").read_text())
    (app / "package.json").write_text(json.dumps({"type": "module", "scripts": {"preview": package["scripts"]["preview"]}}))
    (app / "astro.config.mjs").write_text("export default {output:'static'};\n")
    for name in ("d3.min.js", "plotly.min.js"):
        shutil.copyfile(assets / name, app / "dist" / name)
    (app / "dist/index.html").write_text(
        '<!doctype html><title>Owned article</title><script src="/d3.min.js"></script>'
        + ('<script src="/plotly.min.js"></script>' if plotly_ready else '')
        + '<h1 class="hero-title">Owned article</h1><main><h2>Article section</h2><p>Native export fixture.</p></main>'
    )
    # Only select the already installed host executable and observe native
    # library values. The CLI, browser implementation, pages and waits are real.
    bootstrap = app / "native-chrome.mjs"
    bootstrap.write_text('''import { chromium } from 'playwright';
const launch = chromium.launch.bind(chromium);
chromium.launch = async options => {
  const browser = await launch({...options, executablePath: process.env.PLAYWRIGHT_CHROME});
  let checking = false;
  const observer = setInterval(async () => {
    if (checking) return;
    const page = browser.contexts().flatMap(context => context.pages()).find(page => page.url().startsWith('http:'));
    if (!page) return;
    checking = true;
    try {
      const values = await page.evaluate(() => document.readyState === 'complete' ? {d3:window.d3?.version || null,Plotly:window.Plotly?.version || null} : null);
      if (values) {console.log('Native library values:', JSON.stringify(values));clearInterval(observer);}
    } catch {} finally {checking = false;}
  }, 50);
  browser.on('disconnected', () => clearInterval(observer));
  return browser;
};
''')
    env = dict(os.environ, PATH=f"{Path(node).parent}:{os.environ.get('PATH', '')}", ASTRO_TELEMETRY_DISABLED="1")
    env.pop("PREVIEW_PORT", None)
    process = subprocess.Popen([node, "--import", str(bootstrap), str(app / "scripts" / script), "--wait=images", "--filename=native-fixture"],
                               cwd=app, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ready, after_wait = None, None
    output = []
    for line in process.stdout:
        now = time.monotonic()
        output.append(line)
        if "Server ready" in line:
            ready = now
        if ready and after_wait is None and any(marker in line for marker in (
            "Waiting for content readiness", "Waiting for images", "Scrolling page")):
            after_wait = now
    code = process.wait(timeout=10)
    process.stdout.close()
    log = "".join(output)
    assert code == 0, log
    assert ready is not None and after_wait is not None, log
    expected_values = {"d3": "7.9.0", "Plotly": "2.35.2" if plotly_ready else None}
    values_line = next((line for line in output if line.startswith("Native library values:")), "")
    assert json.loads(values_line.split(":", 1)[1]) == expected_values, log
    elapsed = after_wait - ready
    # Include native browser startup/load time. Both-ready controls must bypass
    # the optional wait, while absent Plotly must honor the 5/8 second budget.
    limit = 4 if plotly_ready else 20
    print(f"{script} ready={plotly_ready}: library phase {elapsed:.3f}s (limit {limit}s)")
    assert elapsed < limit, f"Library phase exceeded {limit}s: {elapsed:.3f}s\n{log}"
    if not plotly_ready:
        assert elapsed >= (8 if script == "export-pdf.mjs" else 5), log
    assert (app / "dist/native-fixture.pdf").read_bytes().startswith(b"%PDF-"), log
