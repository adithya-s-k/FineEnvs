"""Grade a Webdev task as the explorer does (verl recipes/design webdev-eval): render, shrink, judge.

No index.html in dist/ scores 0. A render that fails, or a judge that never answers, writes no reward (not scored):
neither says anything about the page."""
import base64
import io
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/tests")
os.environ["WEBDEV_GRADE_HTTP"] = "1"   # serve over loopback http so React/Vue/ES-module sites actually run
from webdev.eval_rubric import build_prompt  # noqa: E402
from webdev.shot import _build_shot_cmd, _parse_render_env, _parse_shot_b64  # noqa: E402
from webdev.verdict import parse_verdict  # noqa: E402

cfg = json.loads(Path("/tests/grade.json").read_text())
V = Path("/logs/verifier")
V.mkdir(parents=True, exist_ok=True)
dist = Path(cfg["cwd"]) / "dist"


def done(reward: float | None, why: str, **extra) -> None:
    (V / "result.json").write_text(json.dumps({"reward": reward, "summary": why, **extra}, indent=1))
    print(why)
    if reward is None:
        print("not scored", file=sys.stderr)
        sys.exit(1)
    (V / "reward.txt").write_text(f"{reward}\n")
    sys.exit(0)


files = sorted(str(p.relative_to(dist)) for p in dist.rglob("*") if p.is_file())[:200] if dist.is_dir() else []
if "index.html" not in files:
    done(0.0, "The agent delivered nothing to dist/." if not files else "dist/ has no index.html, so there is no page to open.",
         delivered=files)
res = subprocess.run(["sh", "-c", _build_shot_cmd(f"file://{dist}/index.html")], capture_output=True, text=True,
                     timeout=150)   # SHOT_TIMEOUT_S
shot = {"output": (res.stdout or "") + "\n" + (res.stderr or "")}
shot_b64, console_errors = _parse_shot_b64(shot)
render_env = _parse_render_env(shot)
if render_env.get("proxy_failed") or not shot_b64:
    print(shot["output"][-2000:], file=sys.stderr)
    done(None, "external assets were unreachable on every route" if render_env.get("proxy_failed")
         else "the screenshot produced no image")
jpg = base64.b64decode(shot_b64)
try:   # eval_mode._shrink: the judge refuses shots over 20 MP; proportional, quality 75
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(io.BytesIO(jpg))
    size = im.size
    if im.width * im.height > 20e6:
        k = (20e6 / (im.width * im.height)) ** 0.5
        im = im.convert("RGB").resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "JPEG", quality=75); jpg = b.getvalue(); size = im.size
except ImportError:
    size = None
(V / "screenshot.jpg").write_bytes(jpg)
content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
           {"type": "text", "text": build_prompt().format(query=cfg["query"][:1500])}]   # QUERY_CAP
url = os.environ.get("WEBDEV_JUDGE_URL", "https://router.huggingface.co/v1").rstrip("/") + "/chat/completions"
body = json.dumps({"model": os.environ["WEBDEV_JUDGE_MODEL"], "temperature": 1.0,
                   "messages": [{"role": "user", "content": content}]}).encode()
verdict, last = None, ""
for attempt in range(4):   # JUDGE_RETRIES, with eval_mode's backoff
    if attempt:
        time.sleep(min(60.0, 3.0 * (3 ** (attempt - 1))) * (0.5 + random.random()))
    try:
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('WEBDEV_JUDGE_KEY', '')}"})
        with urllib.request.urlopen(req, timeout=180) as r:   # JUDGE_TIMEOUT_S
            msg = json.loads(r.read())["choices"][0]["message"]
        v = parse_verdict(msg.get("content") or msg.get("reasoning_content") or "")
        if "_failed" not in v:
            verdict = v
            break
        last = v["_failed"]
    except Exception as e:  # noqa: BLE001
        last = f"{type(e).__name__}: {e}"[:200]
if not verdict:
    done(None, f"judge unavailable: {last}")
done(verdict["score"], verdict.get("reason", ""), dims=verdict["dims"], visual=verdict["visual"], screenshot_size=size,
     console_errors=console_errors[:1500] if console_errors not in ("", "[]") else "")
