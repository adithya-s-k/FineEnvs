"""Native browser controls for the actual desktop and mobile theme scripts.

Requires pytest and Playwright/Chromium. Set PLAYWRIGHT_CHROME to reuse a browser.
The owned HTTP article uses unchanged component scripts/markup and minimal layout
styles; it does not require an Astro build or change the system's actual settings.
"""
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
COMPONENTS = Path(__file__).resolve().parents[1] / "src/components"


@pytest.fixture(scope="module")
def article():
    theme = (COMPONENTS / "ThemeToggle.astro").read_text()
    toc = (COMPONENTS / "TableOfContents.astro").read_text().split('---', 2)[2]
    toc = toc.replace('{tableOfContentAutoCollapse ? "1" : "0"}', '0')
    html = ('<!doctype html><style>:root{--bp-content-collapse:1100px;--page-bg:white;}'
            '.toc-mobile-toggle,.toc-mobile-sidebar{display:flex!important}'
            '.toc-mobile-toggle{top:50px!important;left:10px!important}</style>' + theme + toc +
            '<section class="content-grid"><main><h2 id="introduction">Introduction</h2>'
            '<p>Owned article prose.</p></main></section>').encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/'
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.fixture
def browser():
    with playwright.sync_playwright() as runtime:
        chrome = os.environ.get('PLAYWRIGHT_CHROME')
        with runtime.chromium.launch(**({'executable_path': chrome} if chrome else {})) as browser:
            yield browser


def load(browser, article, saved=None, mobile=False):
    page = browser.new_page(color_scheme='light', viewport={'width': 480 if mobile else 1280, 'height': 900})
    page.set_default_timeout(5000)
    page.add_init_script('window.__schemeEvents=[];matchMedia("(prefers-color-scheme: dark)").addEventListener("change",e=>window.__schemeEvents.push(e.matches));')
    if saved:
        page.add_init_script(f'localStorage.setItem("theme", "{saved}")')
    page.goto(article, wait_until='domcontentloaded')
    return page


def change_system(page, scheme):
    count = page.evaluate('window.__schemeEvents.length')
    page.emulate_media(color_scheme=scheme)
    page.wait_for_function(f'window.__schemeEvents.length > {count}')


def state(page):
    return page.evaluate('({theme:document.documentElement.dataset.theme,saved:localStorage.getItem("theme")})')


def test_unselected_theme_follows_system(browser, article):
    page = load(browser, article)
    assert state(page) == {'theme': 'light', 'saved': None}
    change_system(page, 'dark')
    assert state(page) == {'theme': 'dark', 'saved': None}
    change_system(page, 'light')
    assert state(page) == {'theme': 'light', 'saved': None}


@pytest.mark.parametrize('mobile', [False, True])
def test_explicit_choice_survives_system_changes(browser, article, mobile):
    page = load(browser, article, mobile=mobile)
    if mobile:
        page.locator('.toc-mobile-toggle').click()
        page.locator('.toc-mobile-sidebar__theme').click()
    else:
        page.locator('#theme-toggle').click()
    assert state(page) == {'theme': 'dark', 'saved': 'dark'}
    change_system(page, 'dark')
    change_system(page, 'light')
    assert state(page) == {'theme': 'dark', 'saved': 'dark'}
    page.reload(wait_until='domcontentloaded')
    assert state(page) == {'theme': 'dark', 'saved': 'dark'}


def test_previously_saved_choice_survives_system_changes(browser, article):
    page = load(browser, article, saved='dark')
    assert state(page) == {'theme': 'dark', 'saved': 'dark'}
    change_system(page, 'dark')
    change_system(page, 'light')
    assert state(page) == {'theme': 'dark', 'saved': 'dark'}
