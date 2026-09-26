"""Per-domain setup and grading, each mirroring Xiaomi's reference harness (XiaomiMiMo/verl + mimoagent).

A rule every adapter follows: nothing that grades the task (hidden tests, rubric answers, the
expected crash) is in the sandbox while the agent runs. It is uploaded after the agent finishes.
"""

from __future__ import annotations

import base64
import io
import json
import re
import secrets
import shlex
import tarfile
import time
from pathlib import Path

from .. import catalog, config
from . import opencode

VENDOR = Path(__file__).resolve().parents[1] / "vendor"


def _msg(m: str) -> str:
    """The General verifier labels model-judged checks in Chinese ("llm 3票" = 3 votes)."""
    v = re.fullmatch(r"llm (\d+)票", (m or "").strip())
    return (f"judged by the model ({v.group(1)} vote{'s' if v.group(1) != '1' else ''})" if v else m)


def _tail(res, n: int = 4000) -> str:
    return ((res.stdout or "") + ("\n" + res.stderr if res.stderr else "")).strip()[-n:]


# ── Code: fix a real issue; graded by hidden tests ───────────────────────────
class Code:
    steps, timeout = 250, 1800
    prompt = ("You are working in the git repository at {cwd}. Resolve the issue below by editing the source code. "
              "Hidden tests will check your change; do not add or modify tests.\n\n<issue>\n{task}\n</issue>")

    def run(self, r) -> dict:
        raw = catalog.rows()[r.run["task_id"]]
        inst = raw["instance"]
        cwd = inst["cwd"]
        r.start_sandbox(catalog.image_for(r.run["task_id"]), config.FLAVORS["code"])

        r.phase("setup", detail="recording the base commit, hiding git history")
        base = (r.sh("git rev-parse HEAD", cwd=cwd).stdout or "").strip().split()[-1]
        stash = f"/root/.g{secrets.token_hex(6)}"
        r.sh(f"rm -f /tmp/patch.diff /tmp/test_patch.diff /tmp/test_files.json; mv {cwd}/.git {stash}")
        version = opencode.install(r)
        r.phase("setup", "done", f"base {base[:10]} · OpenCode {version}")

        r.phase("agent", detail=r.run["model"])
        opencode.run(r, self.prompt.format(cwd=cwd, task=raw["prompt"]), cwd, steps=self.steps, timeout=self.timeout)
        r.phase("agent", "done")

        r.phase("verify", detail="hidden tests")
        r.sh(f"rm -rf {cwd}/.git && mv {stash} {cwd}/.git")
        diff = r.sh("git add -A >/dev/null 2>&1 && git -c core.fileMode=false diff --cached HEAD", cwd=cwd).stdout or ""
        r.emit("diff", text=diff[:200_000], files=[f["path"] for f in catalog.patch_files(diff)])
        r.sh("git reset -q", cwd=cwd)
        patch = inst["test_patch"]
        touched = sorted({f for f in re.findall(r"^(?:\+\+\+|---) (?:[ab]/)?(\S+)", patch, re.M) if f != "/dev/null"})
        blob = "\n".join(touched)
        r.sh(f"""while IFS= read -r tf; do [ -z "$tf" ] && continue
  if git cat-file -e {base}:"$tf" 2>/dev/null; then git checkout {base} -- "$tf" 2>/dev/null || true
  else git rm -f --cached "$tf" >/dev/null 2>&1 || true; rm -f "$tf"; fi
done <<'EOF_RESET'
{blob}
EOF_RESET""", cwd=cwd)
        r.sandbox.files.write("/tmp/.hidden_tests.patch", patch)
        ap = r.sh("git apply --verbose /tmp/.hidden_tests.patch", cwd=cwd)
        if ap.exit_code != 0:
            r.log(_tail(ap))
            r.emit("checks", reward=None, summary="The hidden tests could not be applied (testbed problem, not the model's).",
                   checks=[{"id": "apply_tests", "passed": False, "message": _tail(ap, 600)}])
            return {"reward": None, "error": "test patch did not apply"}
        t = time.time()
        res = r.sh(inst["test_command"], cwd=cwd, timeout=inst.get("verifier_timeout_sec") or 1800)
        out = _tail(res, 12000)
        r.log(out)
        reward = 1.0 if res.exit_code == 0 else 0.0
        summ = (re.findall(r"=+ (.*(?:passed|failed|error).*) =+", out) or [""])[-1]
        r.emit("checks", reward=reward, summary=summ or f"test command exited {res.exit_code}",
               checks=[{"id": "hidden_tests", "passed": reward == 1.0, "method": "tests",
                        "message": f"exit {res.exit_code} in {time.time() - t:.0f}s" + (f" · {summ}" if summ else "")}])
        r.phase("verify", "done", f"reward {reward}")
        return {"reward": reward}


# ── Cyber: reproduce a crash; graded by a root-owned server ──────────────────
class Cyber:
    steps, timeout = 300, 2700
    prompt = ("You are a security researcher. Your working directory is /home/agent.\n"
              "You can use bash, read, write, and edit tools to interact with the system.\n\n{task}")

    def run(self, r) -> dict:
        raw = catalog.rows()[r.run["task_id"]]
        exp = catalog.cyber_expected(raw["instance"].get("description") or raw["prompt"])
        r.start_sandbox(catalog.image_for(r.run["task_id"]), config.FLAVORS["cyber"])

        r.phase("setup", detail="starting the PoC verification server")
        r.sandbox.files.write("/root/expected_func.json", json.dumps(exp))
        r.sandbox.files.write("/root/server.py", (VENDOR / "server_arvo.py").read_text())
        r.sh("chmod 700 /root/server.py /root/expected_func.json; (python3 /root/server.py </dev/null &>/tmp/.srv.log &); "
             "for i in $(seq 40); do curl -s -o /dev/null http://127.0.0.1:8666/ && break; sleep 0.5; done")
        version = opencode.install(r)
        r.phase("setup", "done", f"expecting {exp['error_type']} in {exp['function']} · OpenCode {version}")

        r.phase("agent", detail=r.run["model"])
        opencode.run(r, self.prompt.format(task=raw["prompt"]), "/home/agent", steps=self.steps, timeout=self.timeout,
                     user="agent", home="/home/agent")
        r.phase("agent", "done")

        r.phase("verify", detail="reading the verification server's last result")
        res = r.sh("cat /root/last_result.json 2>/dev/null")
        count = (r.sh("cat /root/submit_count 2>/dev/null").stdout or "0").strip()
        try:
            last = json.loads(res.stdout)
        except (ValueError, TypeError):
            last = None
        if not last:
            checks = [{"id": "submitted", "passed": False, "message": "No PoC was submitted with submit.sh."}]
            reward = 0.0
        else:
            crash, match = bool(last.get("crash")), bool(last.get("match"))
            reward = 1.0 if crash and match else 0.0
            checks = [
                {"id": "submitted", "passed": True, "message": f"{count or last.get('submit_count', 1)} submission(s)"},
                {"id": "crashes", "passed": crash, "message": last.get("actual_error_type") or f"exit {last.get('exit_code')}, no crash"},
                {"id": "right_function", "passed": match,
                 "message": f"crashed in `{last.get('actual_func') or '-'}`, expected `{exp['function']}`"},
            ]
            r.log((last.get("stderr") or "")[-6000:])
        r.emit("checks", reward=reward, checks=checks,
               summary="Crash reproduced in the expected function." if reward else "Not reproduced.")
        r.phase("verify", "done", f"reward {reward}")
        return {"reward": reward}


# ── General (simulated workplace): MCP systems; graded by the task's own verify.py ──
class General:
    steps, timeout = 500, 2400
    prompt = ("You are an agent, your current working directory is {cwd}.\n\n"
              "You can use the tools available to you to interact with the computer to assist the user in completing tasks.\n\n{task}")

    def _push(self, r, root: Path, pairs: list[tuple[str, str]], name: str) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for src, dst in pairs:
                p = root / src
                if p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file():
                            t.add(f.resolve(), arcname=dst.lstrip("/") + "/" + str(f.relative_to(p)))
                elif p.exists():
                    t.add(p.resolve(), arcname=dst.lstrip("/"))
        r.sandbox.files.write(f"/tmp/.{name}.tgz", buf.getvalue())
        r.sh(f"tar -xzf /tmp/.{name}.tgz -C / && rm -f /tmp/.{name}.tgz")

    def run(self, r) -> dict:
        tid = r.run["task_id"]
        raw = catalog.rows()[tid]
        if raw["instance"].get("dataset_type") == "terminal_bench":
            return Terminal().run(r)
        root = catalog.env_dir(tid)
        man = json.loads((root / "manifest.json").read_text())
        r.start_sandbox(catalog.image_for(tid), config.FLAVORS["general"])

        r.phase("setup", detail=f"{len(man['mcp_servers'])} systems")
        self._push(r, root, [(u["source"], u["target"]) for u in man["uploads"]], "env")   # not the verifier
        # the sidecar expects a venv with mcp<2 (the image's own python has mcp 2.x, which renamed FastMCP)
        v = r.sh("[ -x /opt/openai-agents-venv/bin/python ] || { python3 -m venv /opt/openai-agents-venv && "
                 "/opt/openai-agents-venv/bin/pip install -q 'mcp<2'; }", timeout=300)
        if v.exit_code != 0:
            raise RuntimeError("could not prepare the MCP sidecar: " + _tail(v, 400))
        s = r.sh(man["setup"]["command"], timeout=man["setup"].get("timeout_sec") or 300)
        ports = " ".join(map(str, man.get("wait_ports") or []))
        w = r.sh(f"bash -c 'for p in {ports}; do for i in $(seq 60); do (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null && break; "
                 f"sleep 1; done; (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null || echo DOWN $p; done'", timeout=180)
        if "DOWN" in (w.stdout or "") or s.exit_code != 0:
            r.log(_tail(s, 1500))
            raise RuntimeError("the task's systems did not start: " + (w.stdout or "").strip())
        # The upstream harness runs these systems in a separate sidecar pod. Here they share the sandbox, so
        # the agent is an unprivileged user and the systems' data and code are root-only: MCP is the way in.
        version = opencode.install(r)
        iso = r.sh("id -u agent >/dev/null 2>&1 || useradd -u 500 -m -s /bin/bash agent; chown -R agent:agent /work/workspace; "
                   "chmod 700 /work/system /work/tools /installed-agent /work/_setup 2>/dev/null; "
                   "runuser -u agent -- ls /work/system >/dev/null 2>&1 && echo LEAK || echo ok")
        if "LEAK" in (iso.stdout or ""):
            raise RuntimeError("could not isolate the systems' data from the agent")
        r.phase("setup", "done", f"{len(man['mcp_servers'])} MCP systems up · OpenCode {version}")

        r.phase("agent", detail=r.run["model"])
        final = opencode.run(r, self.prompt.format(cwd=man["cwd"], task=(root / "instruction.md").read_text()), man["cwd"],
                             steps=self.steps, timeout=self.timeout, user="agent", home="/home/agent", mcp=man["mcp_servers"])
        r.phase("agent", "done")

        r.phase("verify", detail=f"rubric · judge {r.run.get('judge')}")
        line = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": final}]}})
        r.sandbox.files.write("/tmp/agent_output/sessions/opencode.jsonl", line + "\n")
        self._push(r, root, [(u["source"], u["target"]) for u in man["verifier"]["uploads"]], "verifier")
        judge = r.run.get("judge") or "openai/gpt-oss-120b"
        env = {"GA_JUDGE_KEY": "$HF_TOKEN", "GA_JUDGE_MODEL": judge, "GA_JUDGE_API": "chat",
               "GA_JUDGE_URL": config.ROUTER, "VERIFY_DETERMINISTIC": "1", "VERIFY_AGENT_JUDGE": "1"}
        exports = " ".join(f'{k}="{v}"' for k, v in env.items())
        res = r.sh(f"mkdir -p /logs/verifier && {exports} {man['verifier']['command']}",
                   timeout=man["verifier"].get("timeout_sec") or 1200)
        r.log(_tail(res, 6000))
        detail = {}
        rd = man["verifier"].get("reward_detail_file")
        if rd and r.sandbox.files.exists(rd):
            detail = json.loads(r.sandbox.files.read_text(rd))
        meta = {i["id"]: i for i in json.loads((root / "verifier_meta.json").read_text()).get("items", [])}
        results = {x.get("id"): x for x in detail.get("results", [])}
        items = (detail.get("detail") or {}).get("items") or {}
        checks = []
        for cid, m in meta.items():
            got = results.get(cid) or {}
            sc = (items.get(cid) or {}).get("score", got.get("score"))
            checks.append({"id": cid, "passed": bool(got.get("passed")) if got else None, "score": sc,
                           "method": m.get("method"), "tier": m.get("tier"), "weight": m.get("weight"),
                           "question": m.get("question"), "message": _msg(got.get("message") or (items.get(cid) or {}).get("detail", ""))})
        reward = detail.get("reward", detail.get("score"))
        err = detail.get("reward_error")
        r.emit("checks", reward=None if err else reward, checks=checks, error=err,
               summary=f"judge unavailable ({err}): not scored" if err else f"{sum(1 for c in checks if c['passed'])}/{len(checks)} checks passed")
        r.phase("verify", "done", f"reward {reward}" if not err else f"not scored: {err}")
        return {"reward": None if err else reward, "error": err}


# ── Terminal-bench style tasks (in the General domain) ───────────────────────
class Terminal:
    steps = 300

    def run(self, r) -> dict:
        tid = r.run["task_id"]
        inst = catalog.rows()[tid]["instance"]
        cwd = inst.get("cwd") or "/app"
        r.start_sandbox(catalog.image_for(tid), config.FLAVORS["general"])
        r.phase("setup", detail="scrubbing test leftovers")
        r.sh("rm -rf /tests /logs/verifier")
        version = opencode.install(r)
        r.phase("setup", "done", f"OpenCode {version}")

        r.phase("agent", detail=r.run["model"])
        prompt = General.prompt.format(cwd=cwd, task=inst["problem_statement"])
        opencode.run(r, prompt, cwd, steps=self.steps, timeout=float(inst.get("agent_timeout_sec") or 900) + 300)
        r.phase("agent", "done")

        r.phase("verify", detail="hidden tests")
        files = json.loads(inst.get("tests_files") or "{}")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for name, b64 in files.items():
                data = base64.b64decode(b64)
                info = tarfile.TarInfo("tests/" + name)
                info.size, info.mode = len(data), 0o755
                t.addfile(info, io.BytesIO(data))
        r.sandbox.files.write("/tmp/.tests.tgz", buf.getvalue())
        r.sh("tar -xzf /tmp/.tests.tgz -C / && rm -f /tmp/.tests.tgz && mkdir -p /logs/verifier")
        res = r.sh(f"cd {shlex.quote(cwd)} && sh /tests/test.sh", timeout=float(inst.get("verifier_timeout_sec") or 300) + 120)
        r.log(_tail(res, 8000))
        rew = (r.sh("cat /logs/verifier/reward.txt 2>/dev/null").stdout or "").strip()
        reward = float(rew) if re.fullmatch(r"[01](\.0+)?", rew) else 0.0
        checks = []
        ctrf = r.sh("cat /logs/verifier/ctrf.json 2>/dev/null").stdout
        try:
            for t in json.loads(ctrf)["results"]["tests"]:
                checks.append({"id": t.get("name"), "passed": t.get("status") == "passed", "method": "tests",
                               "message": (t.get("message") or t.get("status") or "")[:300]})
        except (ValueError, KeyError, TypeError):
            checks.append({"id": "tests", "passed": reward == 1.0, "method": "tests", "message": f"reward.txt = {rew or 'missing'}"})
        r.emit("checks", reward=reward, checks=checks, summary=f"{sum(c['passed'] for c in checks)}/{len(checks)} tests passed")
        r.phase("verify", "done", f"reward {reward}")
        return {"reward": reward}


# ── Webdev: build a site; graded by a vision model on a full-page render ─────
SHOT = r'''
import base64, sys
from playwright.sync_api import sync_playwright
url = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto(url, wait_until="networkidle", timeout=60000)
    pg.wait_for_timeout(1500)
    img = pg.screenshot(full_page=True, type="jpeg", quality=80)
    b.close()
print("SHOT_B64:" + base64.b64encode(img).decode())
print("CONSOLE_ERRORS:" + str(len(errors)))
'''


class Webdev:
    steps, timeout = 64, 1800

    def run(self, r) -> dict:
        from ..vendor.webdev.eval_rubric import build_prompt
        from ..vendor.webdev.verdict import parse_verdict
        import httpx
        from PIL import Image

        raw = catalog.rows()[r.run["task_id"]]
        cwd = raw["instance"].get("cwd") or "/workspace"
        r.start_sandbox(catalog.image_for(r.run["task_id"]), config.FLAVORS["webdev"])
        r.phase("setup")
        r.sh(f"mkdir -p {cwd}/dist")
        version = opencode.install(r)
        r.phase("setup", "done", f"deliver to {cwd}/dist · OpenCode {version}")

        tmpl = (VENDOR / "webdev" / "agent_prompt.txt").read_text()
        prompt = tmpl.replace("{{cwd}}", cwd) + "\n\nBuild a website for the following request:\n\n" + raw["prompt"]
        r.phase("agent", detail=r.run["model"])
        opencode.run(r, prompt, cwd, steps=self.steps, timeout=self.timeout)
        r.phase("agent", "done")

        r.phase("verify", detail=f"render + vision judge {r.run.get('judge')}")
        listing = r.sh(f"cd {cwd}/dist 2>/dev/null && find . -type f | head -200").stdout or ""
        delivered = [l[2:] for l in listing.splitlines() if l.startswith("./")]
        r.emit("files", title="Delivered in dist/", files=delivered)
        if "index.html" not in delivered:
            msg = ("The agent delivered nothing to dist/." if not delivered
                   else "dist/ has no index.html, so there is no page to open.")
            r.emit("checks", reward=0.0, checks=[{"id": "delivered", "passed": False, "message": msg}], summary=msg)
            r.phase("verify", "done", "reward 0.0")
            return {"reward": 0.0}
        r.sandbox.files.write("/tmp/.shot.py", SHOT)
        res = r.sh(f"python3 /tmp/.shot.py file://{cwd}/dist/index.html", timeout=240)
        m = re.search(r"SHOT_B64:(\S+)", res.stdout or "")
        if not m:
            r.log(_tail(res, 2000))
            r.emit("checks", reward=None, checks=[{"id": "render", "passed": False, "message": "dist/index.html did not render"}],
                   summary="Nothing to grade: the page did not render.")
            return {"reward": None, "error": "render failed"}
        jpg = base64.b64decode(m.group(1))
        im = Image.open(io.BytesIO(jpg))
        if im.width * im.height > 20e6:                     # the judge refuses oversized shots (Xiaomi: MAX_MEGAPIXELS 20)
            k = (20e6 / (im.width * im.height)) ** 0.5
            im = im.convert("RGB").resize((int(im.width * k), int(im.height * k)))
            b = io.BytesIO(); im.save(b, "JPEG", quality=80); jpg = b.getvalue()
        from .. import store
        store.write_artifact(r.id, "screenshot.jpg", jpg)
        r.emit("image", name="screenshot.jpg", width=im.width, height=im.height)

        judge = r.run.get("judge") or "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
                   {"type": "text", "text": build_prompt().format(query=raw["prompt"][:4000])}]
        verdict, last = None, ""
        for attempt in range(3):
            try:
                resp = httpx.post(f"{config.ROUTER}/chat/completions", timeout=300,
                                  headers={"Authorization": f"Bearer {r.token}"},
                                  json={"model": judge, "temperature": 1.0, "max_tokens": 4000,
                                        "messages": [{"role": "user", "content": content}]})
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                v = parse_verdict(msg.get("content") or msg.get("reasoning_content") or "")
                if "_failed" not in v:
                    verdict = v
                    break
                last = v["_failed"]
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"[:200]
        if not verdict:
            r.emit("checks", reward=None, checks=[], error=last, summary=f"judge unavailable: {last}")
            return {"reward": None, "error": f"judge: {last}"}
        labels = {k: l for k, l, _ in catalog.WEBDEV_DIMS}
        checks = [{"id": labels.get(k, k), "score": round(v, 3), "passed": v >= 0.5, "method": "vision",
                   "tier": "visual" if k not in ("query_fulfillment", "premium_assets") else k}
                  for k, v in verdict["dims"].items()]
        r.emit("checks", reward=verdict["score"], checks=checks, summary=verdict.get("reason", ""),
               formula=f"mean(visual {verdict['visual']:.2f}, brief {verdict['dims']['query_fulfillment']:.2f}, "
                       f"assets {verdict['dims']['premium_assets']:.2f})")
        r.phase("verify", "done", f"reward {verdict['score']}")
        return {"reward": verdict["score"]}


# ── Music: one completion, no sandbox; graded by feature distance to human music ──
class Music:
    def run(self, r) -> dict:
        import httpx
        from ..vendor.music_scorer.pipeline import do, extract_abc

        raw = catalog.rows()[r.run["task_id"]]
        r.update(flavor=None)
        r.phase("agent", detail=r.run["model"])
        base, key, mid = r.agent_api()
        params = r.run.get("params") or {}
        if r.run.get("endpoint"):
            from .. import endpoints
            base = endpoints.check_url(base)   # re-checked at call time: DNS can change after the test
        text, thinking, usage = "", "", {}
        with httpx.stream("POST", f"{base}/chat/completions", timeout=600,
                          headers={"Authorization": f"Bearer {key}"} if key else {},
                          json={"model": mid, "stream": True, "stream_options": {"include_usage": True},
                                "max_tokens": params.get("max_tokens") or 32000,
                                **({"temperature": params["temperature"]} if params.get("temperature") is not None else {}),
                                **({"reasoning_effort": params["thinking"]} if params.get("thinking") not in (None, "default") else {}),
                                "messages": [{"role": "user", "content": raw["prompt"]}]}) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"model call failed: HTTP {resp.status_code} {resp.read()[:300]!r}")
            last_emit = time.time()
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                for c in chunk.get("choices") or []:
                    delta = c.get("delta") or {}
                    text += delta.get("content") or ""
                    thinking += delta.get("reasoning_content") or delta.get("reasoning") or ""
                if time.time() - last_emit > 2:
                    r.emit("partial", text=text[-4000:], chars=len(text), thinking_chars=len(thinking))
                    last_emit = time.time()
                r.check_cancel()
        details = usage.get("completion_tokens_details") or {}
        reasoning = int(details.get("reasoning_tokens") or 0)
        r.add_tokens({"input": usage.get("prompt_tokens"), "output": (usage.get("completion_tokens") or 0) - reasoning,
                      "reasoning": reasoning})
        if thinking:
            r.emit("thinking", text=thinking[-20000:], chars=len(thinking))
        r.emit("text", text=text)
        r.phase("agent", "done")
        if not text.strip():
            msg = ("The model spent its whole token budget thinking and never wrote the piece."
                   if thinking or reasoning else "The model returned an empty answer.")
            r.emit("checks", reward=0.0, checks=[{"id": "answer", "passed": False, "message": msg}], summary=msg)
            return {"reward": 0.0}

        r.phase("verify", detail="abc2midi + 18 human-likeness features")
        abc = extract_abc(text)
        if not abc:
            r.emit("checks", reward=0.0, checks=[{"id": "abc", "passed": False, "message": "No ABC notation found."}], summary="No ABC.")
            return {"reward": 0.0}
        res = do({"key": "rollout", "id": 0, "rep": 0, "abc": abc, "tag": None, "lang": None,
                  "abc_len": len(abc), "nvoice": 0, "latency": None})
        store_abc = abc
        from .. import store
        store.write_artifact(r.id, "piece.abc", store_abc)
        if res.get("skip"):
            msg = {"no_midi": "abc2midi could not turn the notation into MIDI."}.get(res["skip"], str(res["skip"]))
            r.emit("checks", reward=0.0, checks=[{"id": "renders", "passed": False, "message": msg}], summary=msg)
            return {"reward": 0.0}
        # Xiaomi's validity gate: any of these zeroes the piece before its quality counts
        gate = [
            {"id": "No notation errors", "passed": res.get("err", 0) == 0, "method": "gate",
             "message": f"abc2midi reported {res.get('err', 0)} error(s)"},
            {"id": "Bars add up", "passed": res.get("bar", 0) < 10, "method": "gate",
             "message": f"{res.get('bar', 0)} bar(s) with the wrong length (10 or more rejects)"},
            {"id": "No blank lines in the tune", "passed": not res.get("blank"), "method": "gate",
             "message": "a blank line ends an ABC tune early" if res.get("blank") else "ok"},
            {"id": "One instrument per MIDI channel", "passed": not res.get("ch_conflict"), "method": "gate",
             "message": f"{res.get('ch_conflict', 0)} channel conflict(s)"},
        ]
        groups = res.get("groups") or {}
        quality = [{"id": f"{g.capitalize()}", "score": round(float(v) / 100, 3) if v is not None else None, "method": "feature",
                    "tier": "quality", "passed": None, "message": f"{v:.0f}/100" if isinstance(v, (int, float)) else ""}
                   for g, v in groups.items()]
        rejected = bool(res.get("reject"))
        total = res.get("total")
        reward = 0.0 if rejected else max(0.0, min(1.0, float(total or 0) / 100.0))
        summary = ("Rejected by the validity gate: " + "; ".join(c["id"].lower() for c in gate if not c["passed"])
                   if rejected else f"human-likeness {total:.1f}/100")
        r.emit("checks", reward=reward, checks=gate + quality, summary=summary,
               formula=f"quality {total:.1f}/100 would score {float(total) / 100:.3f}" if rejected and total is not None else None)
        r.phase("verify", "done", f"reward {reward:.3f}")
        return {"reward": reward}


ADAPTERS = {"code": Code(), "cyber": Cyber(), "general": General(), "webdev": Webdev(), "music": Music()}


def run_defaults(task_id: str, domain: str) -> dict:
    """The step cap and time limit a rollout gets unless the user changes them (the training harness's values)."""
    if domain == "music":
        return {"steps": None, "timeout_min": None, "max_tokens": 32000}
    if domain == "general":
        inst = (catalog.rows().get(task_id) or {}).get("instance") or {}
        if inst.get("dataset_type") == "terminal_bench":
            return {"steps": Terminal.steps, "timeout_min": round((float(inst.get("agent_timeout_sec") or 900) + 300) / 60), "max_tokens": None}
    a = ADAPTERS[domain]
    return {"steps": a.steps, "timeout_min": round(a.timeout / 60), "max_tokens": None}
