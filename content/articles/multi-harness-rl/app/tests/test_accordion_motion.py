"""Native accordion controls using the component's actual script and styles.

Requires pytest, Playwright with Chromium, and Node >=22.6 for TypeScript stripping.
Set NODE_BINARY and PLAYWRIGHT_CHROME to reuse existing installed runtimes.
The fixture renders the component's template with owned title/slot values; no Astro
build, live Space mutation, or replacement of the animation code is involved.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
COMPONENT = Path(__file__).resolve().parents[1] / "src/components/Accordion.astro"


@pytest.fixture(scope="module")
def component_html():
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
    template = template.replace('{title}', 'Wiring each harness')
    template = template.replace('<slot />', '<p>Owned article detail.</p>')
    style = re.search(r"<style>([\s\S]*?)</style>", source).group(1)
    return f'<!doctype html>{template}<style>{style}</style><script>{emitted}</script>'


@pytest.fixture
def browser():
    with playwright.sync_playwright() as runtime:
        chrome = os.environ.get("PLAYWRIGHT_CHROME")
        options = {"executable_path": chrome} if chrome else {}
        with runtime.chromium.launch(**options) as browser:
            yield browser


@pytest.mark.parametrize("motion", ["reduce", "no-preference"])
@pytest.mark.parametrize("initially_open", [False, True])
def test_accordion_reopens_and_expands_with_content(browser, component_html, motion, initially_open):
    page = browser.new_page(reduced_motion=motion)
    page.set_default_timeout(2000)
    html = component_html.replace('class="accordion"', 'class="accordion" open', 1) if initially_open else component_html
    page.set_content(html)
    accordion = page.locator('details.accordion')
    summary = accordion.locator('summary')
    wrapper = accordion.locator('.accordion__content-wrapper')
    if not initially_open:
        summary.click()
    # Both the no-animation and animated paths must complete, leaving height auto.
    page.wait_for_function("document.querySelector('.accordion__content-wrapper').style.height === 'auto'")
    before = wrapper.evaluate('(e) => e.getBoundingClientRect().height')
    accordion.locator('.accordion__content').evaluate("e => { const p = document.createElement('p'); p.textContent = 'Additional loaded article content'; e.append(p); }")
    assert wrapper.evaluate('(e) => e.getBoundingClientRect().height') > before
    # Keyboard activation uses the same real summary click handler.
    summary.focus()
    page.keyboard.press('Enter')
    page.wait_for_function("!document.querySelector('details.accordion').open")
    assert wrapper.evaluate('(e) => e.getBoundingClientRect().height') == 0
    summary.click()
    page.wait_for_function("document.querySelector('.accordion__content-wrapper').style.height === 'auto'")
    assert accordion.evaluate('(e) => e.open')
    assert wrapper.evaluate('(e) => e.getBoundingClientRect().height') > before
    page.close()
