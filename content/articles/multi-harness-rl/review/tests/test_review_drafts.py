"""Native review.js/browser tests against real HTTP, owned storage and fake OAuth.

Install pytest/httpx/Playwright alongside review requirements. Optionally set
PLAYWRIGHT_PYTHON to that Python and PLAYWRIGHT_CHROME to an existing browser.
"""
import asyncio
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import Request

SERVER = Path(__file__).resolve().parents[1] / "server.py"


@pytest.fixture
def api(tmp_path, monkeypatch):
    import huggingface_hub._oauth as oauth

    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("REVIEW_MODE", "on")
    monkeypatch.setenv("REVIEW_OWNER", "owner")
    monkeypatch.setenv("REVIEWERS", "reviewer")
    monkeypatch.setenv("COMMENTS_DIR", str(tmp_path))
    # Only replace the external identity provider's local fixture. Actual session
    # middleware, signed cookies, OAuth parsing and all role checks remain active.
    monkeypatch.setattr(oauth, "_get_mocked_oauth_info", lambda: {
        "userinfo": {"preferred_username": "owner", "name": "Owner"},
        "expires_at": time.time() + 3600,
    })
    spec = importlib.util.spec_from_file_location("article_review_test", SERVER)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    @server.app.get("/__test_login/{username}")
    async def login(username: str, request: Request):
        request.session["oauth_info"] = {
            "userinfo": {"preferred_username": username, "name": username},
            "expires_at": time.time() + 3600,
        }
        return {"username": username}

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(server.app, log_level="error", lifespan="on")
    runner = uvicorn.Server(config)
    thread = threading.Thread(target=runner.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    end = time.monotonic() + 5
    while not runner.started and thread.is_alive() and time.monotonic() < end:
        time.sleep(0.01)
    assert runner.started
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    yield server, url, tmp_path
    runner.should_exit = True
    thread.join(timeout=5)
    sock.close()
    assert not thread.is_alive()


BROWSER = r'''
import json, sys
from playwright.sync_api import sync_playwright
args=json.load(sys.stdin)
with sync_playwright() as playwright:
    options={"executable_path":args["chrome"]} if args["chrome"] else {}
    browser=playwright.chromium.launch(**options)
    page=browser.new_page(viewport={"width":480,"height":900})
    page.goto(args["url"]+"/__test_login/owner")
    seeded=[]
    if args["scenario"] == "errors":
        seeded=page.evaluate("""async () => {
            const ids=[];
            for(let i=0;i<30;i++){
                const response=await fetch('/api/review/threads',{method:'POST',headers:{'content-type':'application/json','x-review':'1'},body:JSON.stringify({anchor:{type:'text',quote:'Existing note'},body:'Existing comment '+i})});
                if(response.status!==200)throw new Error('unexpected seed '+response.status);
                ids.push((await response.json()).id);
            }
            return ids;
        }""")

    def load():
        page.goto(args["url"]+"/__test_article")
        page.wait_for_function("window.__rvState && document.querySelector('.rv-user img')")
        page.wait_for_function("window.__rvState().threads === " + str(len(seeded)))

    def compose(kind):
        page.evaluate("""() => {
            const range=document.createRange(), walker=document.createTreeWalker(document.querySelector('#prose'),NodeFilter.SHOW_TEXT), nodes=[];
            for(let n=walker.nextNode();n;n=walker.nextNode())if(n.nodeValue)nodes.push(n);
            range.setStart(nodes[0],0);range.setEnd(nodes[nodes.length-1],nodes[nodes.length-1].nodeValue.length);
            const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
            document.dispatchEvent(new MouseEvent('mouseup',{bubbles:true}));
        }""")
        page.locator('.rv-add [data-kind="'+kind+'"]').click()

    def submit(selector, suffix):
        with page.expect_response(lambda r:r.url.endswith(suffix) and r.request.method!='GET') as response:
            page.locator(selector).click()
        return response.value.status

    results={"browser":browser.version}
    if args["scenario"] == "errors":
        for mode in ['comment','replace','reply','edit']:
            load()
            if mode in ['comment','replace']:
                compose(mode)
                selectors=['textarea[data-role="new"]'] if mode=='comment' else ['textarea[data-role="replace"]','textarea[data-role="note"]']
                button='[data-act="post"]'; suffix='/api/review/threads'
            else:
                page.locator('[data-act="history"]').click()
                card=page.locator('.rv-panel .rv-card[data-thread="'+seeded[0]+'"]')
                card.click()
                if mode=='edit':
                    card.locator('[data-act="menu"]').first.click()
                    page.locator('.rv-menu [data-act="edit"]').click()
                    page.set_viewport_size({"width":1280,"height":900})
                    page.wait_for_function('window.__rvState().geo.margin === true')
                    page.wait_for_timeout(150)
                    selectors=['.rv-panel textarea.rv-edit']; button='.rv-panel [data-act="save-edit"]'
                    suffix='/messages/'+card.locator('.rv-msg').first.get_attribute('data-mid')
                else:
                    selectors=['.rv-card[data-thread="'+seeded[0]+'"] .rv-reply textarea']
                    button='.rv-card[data-thread="'+seeded[0]+'"] [data-act="send-reply"]'; suffix='/replies'
            expected=[]
            for index, selector in enumerate(selectors):
                value='Unsent '+mode+' field '+str(index)+' that must survive.'
                page.locator(selector).fill(value); expected.append(value)
            status=submit(button,suffix)
            page.locator('.rv-error').first.wait_for()
            results[mode]={"status":status,"expected":expected,"actual":[page.locator(selector).input_value() for selector in selectors]}
            if mode=='edit':
                results['mirroredEdit']=[page.locator('.rv-margin textarea.rv-edit').input_value(),page.locator('.rv-panel textarea.rv-edit').input_value()]
    else:
        load(); compose('comment')
        page.locator('textarea[data-role="new"]').fill('A successful comment')
        results['postStatus']=submit('[data-act="post"]','/api/review/threads')
        page.wait_for_function('window.__rvState().threads === 1')
        seeded=page.evaluate("async () => (await (await fetch('/api/review/threads')).json()).map(t=>t.id)")
        page.set_viewport_size({"width":1280,"height":900})
        page.wait_for_function('window.__rvState().geo.margin === true')
        page.locator('[data-act="history"]').click()
        reply='.rv-margin .rv-card[data-thread="'+seeded[0]+'"] .rv-reply textarea'
        page.locator(reply).fill('A successful reply')
        results['replyStatus']=submit('.rv-margin [data-act="send-reply"]','/replies')
        page.wait_for_function("document.querySelectorAll('.rv-margin .rv-msg').length === 2")
        results['sentReplyDrafts']=page.locator('.rv-reply textarea').evaluate_all('(fields)=>fields.map(ta=>ta.value)')
        page.locator(reply).fill('A reply I will cancel')
        page.locator('.rv-margin [data-act="cancel-reply"]').click()
        page.set_viewport_size({"width":1250,"height":900})
        page.wait_for_timeout(150)
        results['cancelledReplyDrafts']=page.locator('.rv-reply textarea').evaluate_all('(fields)=>fields.map(ta=>ta.value)')
        page.locator('[data-act="history"]').click()
        page.set_viewport_size({"width":480,"height":900})
        page.wait_for_function('window.__rvState().geo.margin === false')
        # New drafts survive an actual mobile->margin->mobile resize, including
        # the selection, both suggestion fields and their respective values.
        compose('replace')
        expected=['New wording that has not been sent','Optional explanation draft']
        selectors=['textarea[data-role="replace"]','textarea[data-role="note"]']
        for selector,value in zip(selectors,expected):page.locator(selector).fill(value)
        page.locator(selectors[1]).evaluate('(ta)=>ta.setSelectionRange(2,7)')
        page.set_viewport_size({"width":1280,"height":900})
        page.wait_for_function('window.__rvState().geo.margin === true')
        page.wait_for_timeout(150)
        results['desktopDrafts']=[page.locator('.rv-margin '+selector).input_value() for selector in selectors]
        page.set_viewport_size({"width":480,"height":900})
        page.wait_for_function('window.__rvState().geo.margin === false')
        page.wait_for_timeout(150)
        results['mobileDrafts']=[page.locator('.rv-panel '+selector).input_value() for selector in selectors]
        results['expectedResizeDrafts']=expected
        results['selection']=page.locator('.rv-panel '+selectors[1]).evaluate('(ta)=>[ta.selectionStart,ta.selectionEnd]')
        page.locator('[data-act="cancel"]').click()
        compose('comment')
        results['cancelledDraft']=page.locator('textarea[data-role="new"]').input_value()
        results['storedMessages']=page.evaluate("async () => (await (await fetch('/api/review/threads')).json())[0].messages.length")
    print(json.dumps(results))
    browser.close()
'''


def run_browser(api, scenario):
    import os
    import subprocess
    import sys
    from fastapi.responses import HTMLResponse, Response

    server, url, _ = api
    executable = os.environ.get("PLAYWRIGHT_PYTHON", sys.executable)
    available = subprocess.run([executable, "-c", "import playwright"], capture_output=True)
    if available.returncode:
        pytest.skip("install Playwright or set PLAYWRIGHT_PYTHON to a Python with it installed")
    javascript = (SERVER.parents[1] / "app/public/review/review.js").read_text()

    @server.app.get("/__test_article")
    async def article():
        return HTMLResponse('<!doctype html><style>main{max-width:600px}</style><main><h2>Review fixture</h2><p>Existing note quoted by seeded comments.</p><p id="prose">Selected article prose for a legitimate reviewer comment.</p></main><script src="/__test_review_js"></script>')

    @server.app.get("/__test_review_js")
    async def script():
        return Response(javascript, media_type="application/javascript")

    args = {"url": url, "scenario": scenario, "chrome": os.environ.get("PLAYWRIGHT_CHROME")}
    result = subprocess.run([executable, "-c", BROWSER], input=json.dumps(args),
                            text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    print(json.dumps({"scenario": scenario, **data}))
    return data


def test_failed_writes_preserve_all_review_drafts(api):
    result = run_browser(api, "errors")
    for mode in ("comment", "replace", "reply", "edit"):
        assert result[mode]["status"] == 429
        assert result[mode]["actual"] == result[mode]["expected"], result
    assert result["mirroredEdit"] == result["edit"]["expected"] * 2


def test_resizes_preserve_drafts_and_successful_sends_clear_them(api):
    result = run_browser(api, "success")
    assert result["postStatus"] == result["replyStatus"] == 200
    assert result["storedMessages"] == 2
    assert result["sentReplyDrafts"] == result["cancelledReplyDrafts"] == [""]
    assert result["cancelledDraft"] == ""
    assert result["desktopDrafts"] == result["expectedResizeDrafts"]
    assert result["mobileDrafts"] == result["expectedResizeDrafts"]
    assert result["selection"] == [2, 7]
