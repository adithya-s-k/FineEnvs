"""Health checks for environments, with no model in the loop.

Runs the real adapters (setup, isolation, grading: the same code a rollout uses) with a fixed probe in place of
the agent, then checks that the environment behaved as intended:

  code      hidden tests apply and FAIL on the untouched repo (if they pass, the task gives reward for nothing)
  cyber     server up, answer hidden from the agent, a harmless submission is recorded as "no crash"
  general   every MCP system answers the agent user, the databases are hidden, the grader returns a valid score
  terminal  the tests run cleanly and fail on the untouched environment, and the anti-hack guard stays quiet
  webdev    a fixture page renders (reveal-on-scroll content included) and the judge returns a verdict
  music     a valid tune passes the validity gate and scores above 0

    uv run python -m app.validate --per-domain 12 --workers 8        # a stratified sample
    uv run python -m app.validate --ids format-code-task-000001 ...  # specific tasks

Runs on your HF account (sandbox time, plus a few cheap judge calls for General and Webdev). Results go to a
separate store, never into anyone's rollouts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT = Path(os.environ.get("VALIDATE_DIR") or Path(__file__).resolve().parents[1] / ".validate")
os.environ["STORAGE_DIR"] = str(OUT / "store")   # before the app's config is imported

from . import catalog, store  # noqa: E402
from .runner import core  # noqa: E402

ANSWER_PROBE = "I was unable to complete this task, so I have no findings to report."

# 24 bars of 4/4 at L:1/8 (8 eighths a bar), no blank lines, one voice: passes the scorer's validity gate
TUNE = """X:1
T:Validation tune
M:4/4
L:1/8
Q:1/4=96
K:G
|:GABc dedB|dedB dedB|c2ec B2dB|c2A2 A2BA|GABc dedB|dedB dedB|c2ec B2dB|A2F2 G4:|
|:g2gf gdBd|g2f2 e2d2|c2ec B2dB|c2A2 A2BA|g2gf gdBd|g2f2 e2d2|c2ec B2dB|A2F2 G4:|
|:B2dB c2ec|B2dB A2BA|G2BG A2cA|B2G2 A4|B2dB c2ec|B2dB A2BA|G2BG A2FA|G2G2 G4:|
"""

FIXTURE = """mkdir -p dist && cat > dist/index.html <<'HTML'
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Fixture</title>
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
<style>.reveal{opacity:0;transform:translateY(24px);transition:all .6s}.reveal.on{opacity:1;transform:none}</style></head>
<body class="bg-slate-50 text-slate-900">
<header class="p-10 bg-indigo-600 text-white"><h1 class="text-4xl font-bold">Validation fixture</h1><p>Hero section</p></header>
<main class="max-w-4xl mx-auto p-8 space-y-24">
<section class="reveal p-8 bg-white rounded-xl shadow"><h2 class="text-2xl font-semibold">Revealed on scroll 1</h2><p>This only becomes visible when scrolled into view.</p></section>
<section class="reveal p-8 bg-white rounded-xl shadow"><h2 class="text-2xl font-semibold">Revealed on scroll 2</h2><p>Second section.</p></section>
<section class="reveal p-8 bg-white rounded-xl shadow"><h2 class="text-2xl font-semibold">Revealed on scroll 3</h2><p>Third section.</p></section>
</main>
<script>const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)e.target.classList.add('on')}));document.querySelectorAll('.reveal').forEach(s=>io.observe(s));</script>
</body></html>
HTML
ls -la dist"""


def mcp_probe(servers: list[dict]) -> str:
    """As the agent user: initialize + tools/list on every MCP server over streamable HTTP, and try the databases."""
    return f"""echo "== databases hidden from the agent:"; ls /work/system >/dev/null 2>&1 && echo LEAK || echo hidden
python3 - <<'PY'
import json, urllib.request
servers = {json.dumps([{"name": s["name"], "url": s["url"]} for s in servers])}
def call(url, body, sid=None):
    h = {{"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}}
    if sid: h["Mcp-Session-Id"] = sid
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        raw, sid2 = r.read().decode(), r.headers.get("Mcp-Session-Id") or sid
    for line in raw.splitlines():
        if line.startswith("data:"):
            raw = line[5:].strip()
    return (json.loads(raw) if raw.strip() else {{}}), sid2
out = {{}}
for s in servers:
    try:
        init, sid = call(s["url"], {{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {{"protocolVersion": "2025-03-26",
                     "capabilities": {{}}, "clientInfo": {{"name": "probe", "version": "1"}}}}}})
        try: call(s["url"], {{"jsonrpc": "2.0", "method": "notifications/initialized"}}, sid)
        except Exception: pass
        tl, _ = call(s["url"], {{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {{}}}}, sid)
        out[s["name"]] = len(((tl.get("result") or {{}}).get("tools")) or [])
    except Exception as e:
        out[s["name"]] = "ERR " + type(e).__name__ + ": " + str(e)[:80]
print("MCP_PROBE:" + json.dumps(out))
PY"""


CYBER_PROBE = """echo "== agent: $(id -un)"
cat /root/expected_func.json >/dev/null 2>&1 && echo "EXPECTED:LEAK" || echo "EXPECTED:hidden"
cat /root/last_result.json >/dev/null 2>&1 && echo "RESULT:readable" || echo "RESULT:hidden"
printf 'mimo-explorer harmless probe input' > /tmp/probe_poc
bash submit.sh /tmp/probe_poc 2>&1 | tail -8"""


def plan(domain: str, task_id: str) -> dict:
    if domain == "cyber":
        return {"scripted": CYBER_PROBE}
    if domain == "general":
        inst = catalog.rows()[task_id]["instance"]
        if inst.get("dataset_type") == "terminal_bench":
            return {"scripted": ""}
        man = json.loads((catalog.env_dir(task_id) / "manifest.json").read_text())
        return {"scripted": mcp_probe(man["mcp_servers"]), "scripted_final": ANSWER_PROBE, "judge": "thinkingmachines/Inkling"}
    if domain == "webdev":
        return {"scripted": FIXTURE, "judge": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"}
    if domain == "music":
        return {"scripted": "", "scripted_final": TUNE}
    return {"scripted": ""}   # code: do nothing


def verdict(run: dict, events: list[dict], task_id: str) -> tuple[str, str]:
    """(status, why): ok, or a named problem."""
    d, rw = run["domain"], run.get("reward")
    if run["status"] != "done":
        return "BROKEN", (run.get("error") or run["status"])[:240]
    checks = next((e for e in reversed(events) if e["kind"] == "checks"), {})
    probe = "\n".join(e.get("output", "") for e in events if e["kind"] == "tool")
    logs = "\n".join(e.get("text", "") for e in events if e["kind"] == "log")
    inst = catalog.rows()[task_id]["instance"]
    if d == "code":
        if rw is None:
            return "BROKEN", checks.get("summary", "not scored")[:240]
        if rw >= 1:
            return "BROKEN", "hidden tests PASS on the untouched repo: reward for doing nothing"
        note = " (.git hidden: history not truncated)" if "not truncated" in logs else ""
        return "ok", "tests fail on the untouched repo" + note
    if d == "cyber":
        if "EXPECTED:LEAK" in probe or "RESULT:readable" in probe:
            return "BROKEN", "the agent can read the expected answer or the verdict"
        subs = next((c for c in checks.get("checks", []) if c["id"] == "submitted"), {})
        if not subs.get("passed"):
            return "BROKEN", "a submission through submit.sh was not recorded"
        return ("ok", "server up, answer hidden, harmless input recorded as no crash") if rw == 0 else \
               ("ok", "harmless input crashed in the expected function (task is easy, not broken)")
    if d == "general" and inst.get("dataset_type") == "terminal_bench":
        if rw is None:
            return "BROKEN", checks.get("summary", "not scored")[:240]
        if any(c["id"] == "anti_hack_guard" for c in checks.get("checks", [])):
            return "BROKEN", "the anti-hack guard rejected the untouched environment: " + checks.get("summary", "")[:160]
        if rw >= 1:
            return "BROKEN", "tests PASS on the untouched environment: reward for doing nothing"
        return "ok", f"tests run and fail on the untouched environment ({len(checks.get('checks', []))} results)"
    if d == "general":
        m = next((l for l in probe.splitlines() if l.startswith("MCP_PROBE:")), "")
        tools = json.loads(m[len("MCP_PROBE:"):]) if m else {}
        bad = {k: v for k, v in tools.items() if not isinstance(v, int) or v < 1}
        if "LEAK" in probe:
            return "BROKEN", "the agent can read the systems' databases directly"
        if not tools or bad:
            return "BROKEN", f"MCP systems not usable by the agent: {bad or 'no probe output'}"
        if rw is None:
            return "BROKEN", "grader did not score: " + (checks.get("error") or checks.get("summary", ""))[:200]
        return "ok", f"{len(tools)} MCP systems answer the agent ({sum(tools.values())} tools), grader scored a non-answer {rw:.2f}"
    if d == "webdev":
        if rw is None:
            return "BROKEN", checks.get("summary", "not scored")[:240]
        return "ok", f"fixture rendered and judged {rw:.2f}"
    if d == "music":
        gate = [c for c in checks.get("checks", []) if c.get("tier") == "gate" or c.get("id") in ("errors", "bars", "blank_lines", "channels")]
        if not rw:
            return "BROKEN", "a valid tune was rejected: " + checks.get("summary", "")[:200]
        return "ok", f"valid tune passed the gate and scored {rw:.3f}"
    return "ok", ""


def sample(per_domain: int, seed: int) -> list[str]:
    rnd = random.Random(seed)
    idx = catalog.index()
    by: dict[str, dict[str, list[str]]] = {}
    for e in idx["envs"]:
        dom = next(x for x in idx["domains"] if x["id"] == e["d"])
        key = str((e["f"].get(dom["main"]) or ["?"])[0] if isinstance(e["f"].get(dom["main"]), list) else e["f"].get(dom["main"]))
        by.setdefault(e["d"], {}).setdefault(key, []).append(e["id"])
    picks: list[str] = []
    for d, groups in by.items():
        n = per_domain if d not in ("webdev", "music") else max(2, per_domain // 4)
        keys = sorted(groups, key=lambda k: -len(groups[k]))
        chosen: list[str] = []
        while len(chosen) < n and any(groups[k] for k in keys):   # round-robin over categories: coverage first
            for k in keys:
                if groups[k] and len(chosen) < n:
                    chosen.append(groups[k].pop(rnd.randrange(len(groups[k]))))
        picks += chosen
    # General: make sure Terminal-bench tasks are in the sample too
    tb = [t for t, r in catalog.rows().items() if r["instance"].get("dataset_type") == "terminal_bench"]
    picks += rnd.sample(tb, min(len(tb), max(2, per_domain // 2)))
    return list(dict.fromkeys(picks))


def run_one(task_id: str, token: str, user: str) -> dict:
    raw = catalog.rows()[task_id]
    domain = raw["domain"]
    t0 = time.time()
    run = store.create({"id": time.strftime("%Y%m%d-%H%M%S-") + os.urandom(3).hex(), "user": user, "task_id": task_id,
                        "domain": domain, "title": task_id, "model": "validation-probe", "provider": None,
                        "status": "queued", "harness": "probe", **plan(domain, task_id)})
    r = core.Rollout(run, token)
    r.execute()
    run = store.get(run["id"])
    status, why = verdict(run, store.read_events(run["id"]), task_id)
    return {"task_id": task_id, "domain": domain, "kind": raw["instance"].get("dataset_type"), "status": status, "why": why,
            "reward": run.get("reward"), "run_id": run["id"], "seconds": round(time.time() - t0),
            "sandbox_cost": (run.get("cost") or {}).get("sandbox")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    from huggingface_hub import get_token, whoami

    token = get_token()
    user = whoami(token=token)["name"]
    ids = a.ids or sample(a.per_domain, a.seed)
    print(f"validating {len(ids)} environments with {a.workers} at a time; store: {OUT / 'store'}", flush=True)
    results, report = [], OUT / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_one, t, token, user): t for t in ids}
        for f in as_completed(futs):
            try:
                res = f.result()
            except Exception as e:  # noqa: BLE001
                res = {"task_id": futs[f], "status": "BROKEN", "why": f"harness error: {type(e).__name__}: {e}"[:240]}
            results.append(res)
            print(f"[{len(results)}/{len(ids)}] {res['status']:6} {res.get('domain', '?'):8} {res['task_id']:55} {res['why']}", flush=True)
            report.write_text(json.dumps(results, indent=1))
    ok = sum(r["status"] == "ok" for r in results)
    print(f"\n{ok}/{len(results)} healthy · report: {report}")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
