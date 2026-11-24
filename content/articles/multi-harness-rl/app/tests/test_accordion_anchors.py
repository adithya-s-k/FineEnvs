"""Native fragment navigation with the shipped accordion script and styles.

Requires pytest, Playwright/Chromium, and Node >=22.6. NODE_BINARY and
PLAYWRIGHT_CHROME can select already installed runtimes. No Astro build needed.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import subprocess
import threading

import pytest

playwright = pytest.importorskip("playwright.sync_api")
COMPONENT = Path(__file__).resolve().parents[1] / "src/components/Accordion.astro"


@pytest.fixture(scope="module")
def article_url():
    source = COMPONENT.read_text()
    script = re.search(r"<script>([\s\S]*?)</script>", source).group(1)
    emitted = subprocess.run(
        [os.environ.get("NODE_BINARY", "node"), "--input-type=module", "-e",
         "import {stripTypeScriptTypes} from 'node:module'; "
         "let s='';for await(const c of process.stdin)s+=c; "
         "process.stdout.write(stripTypeScriptTypes(s));"],
        input=script, text=True, capture_output=True, check=True,
    ).stdout
    template = source.split("---", 2)[2].split("<script>", 1)[0]
    template = template.replace('{wrapperClass}', '"accordion"')
    template = template.replace('open={open} {...props}', '')
    template = template.replace('{title}', 'Serving Harbor through OpenEnv')
    template = template.replace('<slot />', '<h4 id="wiring-each-harness">Wiring each harness</h4><p>Harness connection details.</p>')
    style = re.search(r"<style>([\s\S]*?)</style>", source).group(1)
    # These existing heading/link values reproduce the public article's native
    # fragment producer without replacing the component's navigation behavior.
    html = ('<!doctype html><a id="section-link" href="#wiring-each-harness">Wiring each harness</a>'
            '<a id="prose-link" href="#prose">Article prose</a>' + template
            + '<h2 id="prose">Article prose</h2><p>Ordinary article content.</p>'
            + f'<style>{style}</style><script>{emitted}</script>').encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def page():
    with playwright.sync_playwright() as runtime:
        chrome = os.environ.get("PLAYWRIGHT_CHROME")
        options = {"executable_path": chrome} if chrome else {}
        with runtime.chromium.launch(**options) as browser:
            with browser.new_page(reduced_motion="no-preference") as page:
                page.set_default_timeout(2000)
                yield page


@pytest.mark.parametrize("navigation", ["initial-fragment", "section-link"])
def test_native_fragment_reveals_accordion_content(page, article_url, navigation):
    page.goto(article_url + ("#wiring-each-harness" if navigation == "initial-fragment" else ""))
    if navigation == "section-link":
        page.locator('#section-link').click()
    page.wait_for_function("document.querySelector('details.accordion').open")
    # The native details reveal alone is insufficient: the animated wrapper
    # must stop clipping the linked heading and following prose.
    page.wait_for_function("document.querySelector('.accordion__content-wrapper').getBoundingClientRect().height > 0")
    assert page.locator('.accordion__content-wrapper').evaluate("e => e.style.height") == "auto"
    assert page.locator('#wiring-each-harness').evaluate("e => e.closest('.accordion__content-wrapper').getBoundingClientRect().height >= e.getBoundingClientRect().height")


def test_summary_button_still_animates_and_closes(page, article_url):
    page.goto(article_url)
    page.locator('summary').click()
    page.wait_for_function("document.querySelector('.accordion__content-wrapper').style.height === 'auto'")
    page.locator('summary').click()
    page.wait_for_function("!document.querySelector('details.accordion').open")
    assert page.locator('.accordion__content-wrapper').evaluate("e => e.getBoundingClientRect().height") == 0


def test_prose_fragment_does_not_open_unrelated_accordion(page, article_url):
    page.goto(article_url)
    page.locator('#prose-link').click()
    assert page.url.endswith('#prose')
    assert not page.locator('details.accordion').evaluate("e => e.open")
