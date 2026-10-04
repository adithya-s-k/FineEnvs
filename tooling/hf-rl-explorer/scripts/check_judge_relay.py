"""The judge relay end to end, minus the sandbox's network hop: a judge session made exactly as
app/runner.Rollout._judge_session makes it, then the call a task's grader makes (its variables bound by app/judge.py: the
per-rollout key, the relay URL, the judge model) to the capture proxy. Then a forged key, and the key after the session ends,
both refused. Uses your HF token (`hf auth login`) for one judge call; prints no token.

    uv run python scripts/check_judge_relay.py path/to/task.toml

Last run 2026-10-03 on a MiMo General task: 200 with a correct grade in 49 s (Inkling is slow; the local gradio.live
tunnel cuts requests at 60 s, so long judge prompts need the Space's own /capture), forged 401, 1 call, deleted 401."""
import asyncio, os, secrets, sys, time
from dataclasses import replace

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RLX_WARM", "0")
import httpx
from huggingface_hub import get_token

from app import config, judge, runner


async def main():
    runner._proxy_url()   # starts the capture server (and, locally, its tunnel)
    svc = runner.service()
    port = svc.capture.port
    pool = svc.capture.app.state.upstreams
    from openenv.core.harness.capture.sessions import Upstream

    jup = Upstream(llm_url=config.ROUTER, model="thinkingmachines/Inkling", api_key=get_token(), provider="hf")
    jclient, jlevel = await pool.resolve(jup)
    sid = "j" + secrets.token_hex(16)
    svc.capture.registry.create(sid, upstream=replace(jup, model=jclient.served_model or jup.model), capture_level=jlevel,
                                purpose="eval", max_model_calls=runner.JUDGE_MAX_CALLS, role="judge", run="local-test")
    toml = open(sys.argv[1]).read()
    b = judge.bindings(judge.plan(toml), relay=f"http://127.0.0.1:{port}", capability=sid, judge="thinkingmachines/Inkling")["verifier"]
    url_key = next(k for k in b if k.endswith("_URL")); key_key = next(k for k in b if k.endswith("_KEY") and b[k] == sid)
    model = next(v for k, v in b.items() if "MODEL" in k)
    print("grader gets", {k: ("<capability>" if v == sid else v) for k, v in b.items()})
    t = time.time()
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(b[url_key] + "/chat/completions", headers={"Authorization": f"Bearer {b[key_key]}"},
                         json={"model": model, "temperature": 0, "max_tokens": 200,
                               "messages": [{"role": "system", "content": "You grade answers. Reply with JSON {\"score\": 0 or 1, \"why\": str}."},
                                            {"role": "user", "content": "Rubric: the answer states that 7*8=56. Answer: 7 times 8 is 56."}]})
        # a forged key and a model the session doesn't serve
        bad = await c.post(b[url_key] + "/chat/completions", headers={"Authorization": "Bearer jforged"}, json={"model": model, "messages": [{"role": "user", "content": "hi"}]})
    print("judge reply", r.status_code, round(time.time() - t, 1), "s:", (r.json().get("choices") or [{}])[0].get("message", {}).get("content", r.text)[:200])
    print("forged key ->", bad.status_code)
    sess = svc.capture.registry.get(sid)
    print("session calls:", sess.graph.stats()["n_turns"])
    svc.capture.registry.delete(sid)
    print("after delete, same key ->", httpx.post(b[url_key] + "/chat/completions", headers={"Authorization": f"Bearer {sid}"},
                                               json={"model": model, "messages": [{"role": "user", "content": "hi"}]}, timeout=30).status_code)


asyncio.run(main())
for t in __import__("gradio.tunneling", fromlist=["x"]).CURRENT_TUNNELS:
    t.kill()
