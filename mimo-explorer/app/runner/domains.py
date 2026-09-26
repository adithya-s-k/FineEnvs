"""Per-domain setup and grading, each mirroring Xiaomi's reference harness (XiaomiMiMo/verl + mimoagent).

A rule every adapter follows: nothing that grades the task (hidden tests, rubric answers, the
expected crash) is in the sandbox while the agent runs. It is uploaded after the agent finishes.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
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
# mimoagent environments/datasets/opensource_code.py + base.py, with verl's code configs (config/agent/code/*.yaml):
# step_limit 500, anti_hack_cleanup on, "Fix the following issue:" as the instruction.
RESIDUE_SCRUB = (   # base.py _RESIDUE_SCRUB_GLOBAL, verbatim
    "rm -f /tmp/fail.log /tmp/pass.log /tmp/patch.diff /tmp/test_patch.diff /tmp/*.log 2>/dev/null; "
    "rm -rf /tmp/claude-0 /tmp/claude-* /tmp/testem-* /tmp/puppeteer_dev_chrome_profile-* 2>/dev/null; "
    "rm -f /tmp/test_files.json 2>/dev/null; "
    "rm -rf /tmp/jest_* /tmp/jest-* /tmp/build /tmp/build_env 2>/dev/null; "
    "rm -rf /tmp/pytest-of-root /tmp/pytest-* /tmp/.pytest_cache /tmp/__pycache__ 2>/dev/null; "
    "rm -rf /tmp/go-build* /tmp/standards /tmp/wordpress /tmp/zig-* /tmp/testbase /tmp/testbed 2>/dev/null; "
    "rm -f /tmp/tmp*.tmp /tmp/*.bak /tmp/expect* /tmp/butwas* 2>/dev/null; "
    "rm -rf /root/.cache/go-build 2>/dev/null; "
    "rm -rf /tests /logs 2>/dev/null; "
    "find /var/log -type f -delete 2>/dev/null || true; "
    "rm -rf /var/lib/postgresql/*/*/log /var/lib/postgresql/*/*/pg_log 2>/dev/null || true; "
    "find /var/lib/mysql /var/lib/mongodb -type f -name '*.log' -delete 2>/dev/null || true; "
    "rm -rf /root/.npm/_logs /root/.babel.json /root/.pytest_cache 2>/dev/null || true; "
    "find / -maxdepth 4 -xdev -name task_description.md -path '*/.build_env/*' -delete 2>/dev/null || true; "
    "true"
)
GLOBAL_CACHE_SCRUB = (   # base.py _GLOBAL_CACHE_SCRUB, verbatim
    "find /root/.m2 -type f \\( -name '*-SNAPSHOT.jar' -o -name '*-SNAPSHOT-sources.jar' "
    "-o -name '*-SNAPSHOT-tests.jar' -o -name '*-SNAPSHOT-test-sources.jar' \\) -delete 2>/dev/null; "
    "rm -rf /root/.julia/compiled 2>/dev/null; "
    "rm -rf /root/.gradle/caches/build-cache-* /root/.gradle/daemon 2>/dev/null; "
    "rm -rf /root/.gradle/caches/*/scripts /root/.gradle/caches/jars-* 2>/dev/null; "
    "rm -rf /root/.cache/bazel 2>/dev/null; "
    "true"
)
# base.py _CLEAN_KEEP_COMMON + _CLEAN_KEEP_UNKNOWN: these rows carry no `language`, so mimoagent keeps the union
CLEAN_KEEP = ["node_modules", "bower_components", ".husky", "vendor", "third_party", "_deps", "vcpkg_installed",
              ".venv", "venv", ".gradle", "target", ".build", "lib", ".bundle", "Manifest.toml", "_build", ".stack-work",
              "dist-newstyle"]


def touched_files(patch: str) -> list[str]:
    """base.py get_patch_touched_files: both sides of every `diff --git`, so rename sources are reset too."""
    out: list[str] = []
    for line in patch.split("\n"):
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            for fp in (m.group(1), m.group(2)):
                if fp not in out:
                    out.append(fp)
    return out


def reset_cmd(patch: str, base: str) -> str | None:
    """base.py build_reset_test_files_cmd, verbatim: restore paths that exist at base, remove the rest."""
    files = touched_files(patch)
    if not (files and base):
        return None
    blob = "\n".join(files)
    return ("while IFS= read -r tf; do\n"
            '  [ -z "$tf" ] && continue\n'
            f'  if git cat-file -e {base}:"$tf" 2>/dev/null; then\n'
            f'    git checkout {base} -- "$tf" 2>/dev/null || true\n'
            "  else\n"
            '    git rm -f --cached "$tf" >/dev/null 2>&1 || true\n'
            '    rm -f "$tf"\n'
            "  fi\n"
            "done <<'EOF_RESET_TEST_FILES'\n"
            f"{blob}\n"
            "EOF_RESET_TEST_FILES\n")


class Code:
    steps, timeout = 500, 3600
    prompt = "Fix the following issue:\n\n{task}"

    def run(self, r) -> dict:
        raw = catalog.rows()[r.run["task_id"]]
        inst = raw["instance"]
        cwd = inst["cwd"]
        r.start_sandbox(catalog.image_for(r.run["task_id"]), config.FLAVORS["code"])

        r.phase("setup", detail="recording the base commit, checking history, anti-hack cleanup")
        r.sh(f"git config --global --add safe.directory {shlex.quote(cwd)}")
        if r.sh("git rev-parse --git-dir", cwd=cwd).exit_code != 0:   # a bare source tree: give it a baseline commit
            r.sh("git init -q && git add -A && git commit -q -m baseline --allow-empty", cwd=cwd)
        base = (r.sh("git rev-parse HEAD", cwd=cwd).stdout or "").strip().split()[-1]
        if len(base) != 40:
            raise RuntimeError(f"could not resolve the base commit in {cwd}")
        # Images are built with history truncated at the base, so .git stays visible (git diff/status work as
        # usual). If an image is not, the fix could be read out of git log: fall back to hiding .git.
        extra = [x for x in (r.sh(f"git rev-list --all --not {base} | head -n 5", cwd=cwd).stdout or "").split() if len(x) == 40]
        stash = None
        r.sh(RESIDUE_SCRUB, timeout=120)
        r.sh("git clean -fdx " + " ".join(f"--exclude={d}" for d in CLEAN_KEEP), cwd=cwd, timeout=300)
        r.sh(GLOBAL_CACHE_SCRUB, timeout=180)
        if extra:
            stash = f"/usr/lib/.{secrets.token_hex(12)}"
            r.sh(f"mv {shlex.quote(cwd)}/.git {stash}")
            r.log(f"This image's git history is not truncated at {base[:12]} ({len(extra)}+ later commits reachable), "
                  "so .git is hidden while the agent works.")
        version = opencode.install(r)   # before the blocklist: the release is fetched from github.com
        opencode.block_answer_hosts(r)
        r.phase("setup", "done", f"base {base[:10]}{' · .git hidden' if stash else ''} · OpenCode {version}")

        r.phase("agent", detail=r.run["model"])
        opencode.run(r, self.prompt.format(task=raw["prompt"]), cwd, steps=self.steps, timeout=self.timeout)
        r.phase("agent", "done")

        r.phase("verify", detail="hidden tests")
        if stash:
            r.sh(f"rm -rf {shlex.quote(cwd)}/.git && mv {stash} {shlex.quote(cwd)}/.git")
        # the agent's change, against the base (so its own commits count too)
        diff = r.sh(f"git add -A >/dev/null 2>&1 && git -c core.fileMode=false diff --cached {base}", cwd=cwd).stdout or ""
        r.emit("diff", text=diff[:200_000], files=[f["path"] for f in catalog.patch_files(diff)])
        r.sh("git reset -q", cwd=cwd)
        patch = inst["test_patch"]
        rc = reset_cmd(patch, base)
        if rc and r.sh(rc, cwd=cwd).exit_code != 0:
            r.emit("checks", reward=None, summary="Couldn't reset the test files (testbed problem, not the model's).",
                   checks=[{"id": "reset_tests", "passed": False, "message": "reset of test-patch paths failed"}])
            return {"reward": None, "error": "reset_tests_failed"}
        r.sandbox.files.write("/tmp/_opensource_code_test.patch", patch)
        ap = r.sh("git apply --verbose /tmp/_opensource_code_test.patch", cwd=cwd)
        r.sh("rm -f /tmp/_opensource_code_test.patch")
        if ap.exit_code != 0:
            r.log(_tail(ap))
            r.emit("checks", reward=None, summary="The hidden tests could not be applied (testbed problem, not the model's).",
                   checks=[{"id": "apply_tests", "passed": False, "message": _tail(ap, 600)}])
            return {"reward": None, "error": "apply_test_patch_failed"}
        t = time.time()
        res = r.sh(inst["test_command"], cwd=cwd, timeout=int(inst.get("verifier_timeout_sec") or 1800))
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
    steps, timeout = 300, 3600   # verl config/agent/arvo/arvo.yaml: step_limit 300
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
        version = opencode.install(r)   # before the blocklist: the release is fetched from github.com
        opencode.block_answer_hosts(r)
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
    steps, timeout = 500, 3600   # verl config/agent/general/s3k.yaml: step_limit 500
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
        inst = raw["instance"]
        if inst.get("dataset_type") == "terminal_bench":
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
        opencode.block_answer_hosts(r)
        r.phase("setup", "done", f"{len(man['mcp_servers'])} MCP systems up · OpenCode {version}")

        r.phase("agent", detail=r.run["model"])
        final = opencode.run(r, self.prompt.format(cwd=man["cwd"], task=(root / "instruction.md").read_text()), man["cwd"],
                             steps=self.steps, timeout=self.timeout, user="agent", home="/home/agent", mcp=man["mcp_servers"])
        r.phase("agent", "done")

        r.phase("verify", detail=f"rubric · judge {r.run.get('judge')}")
        # verl general_agent/environment.py _do_calculate_reward, step by step
        dead = [p for p in (man.get("wait_ports") or [])
                if r.sh(f"bash -c '(echo > /dev/tcp/127.0.0.1/{int(p)}) 2>/dev/null'").exit_code != 0]
        if dead:
            r.emit("checks", reward=None, checks=[], error="mcp_backend_down",
                   summary=f"A system stopped responding (port {dead[0]}): testbed problem, not scored.")
            return {"reward": None, "error": "mcp_backend_down"}
        line = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": final}]}})
        r.sandbox.files.write("/tmp/agent_output/sessions/opencode.jsonl", line + "\n")   # run_verify reads this shape
        if final and r.sh(f"test -s {shlex.quote(man['cwd'])}/answer.md").exit_code != 0:   # _write_answer_md
            r.sandbox.files.write(f"{man['cwd']}/answer.md", final)
            r.sh(f"chown agent:agent {shlex.quote(man['cwd'])}/answer.md")
        self._push(r, root, [(u["source"], u["target"]) for u in man["verifier"]["uploads"]], "verifier")
        judge = r.run.get("judge") or "thinkingmachines/Inkling"
        env = {"GA_JUDGE_KEY": "$HF_TOKEN", "GA_JUDGE_MODEL": judge, "GA_JUDGE_API": "chat",
               "GA_JUDGE_URL": config.ROUTER, "VERIFY_DETERMINISTIC": "1", "VERIFY_AGENT_JUDGE": "1"}
        exports = " ".join(f'{k}="{v}"' for k, v in env.items())
        rf = man["verifier"].get("reward_file") or "/logs/verifier/reward.json"
        rd = man["verifier"].get("reward_detail_file") or "/logs/verifier/reward_detail.json"
        r.sh(f"mkdir -p {os.path.dirname(rf)} /tmp/agent_output && rm -f {rf} {rd}")
        res = r.sh(f"cd /work && {exports} {man['verifier']['command']}",
                   timeout=int(man["verifier"].get("timeout_sec") or inst.get("verifier_timeout_sec") or 900))
        r.log(_tail(res, 6000))
        reward_raw = (r.sh(f"cat {rf} 2>/dev/null").stdout or "").strip()
        detail = {}
        if r.sandbox.files.exists(rd):
            try:
                detail = json.loads(r.sandbox.files.read_text(rd))
            except ValueError:
                detail = {}
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
        if "judge_crashed" in reward_raw or detail.get("reward_error") == "judge_crashed":
            # run_verify.py drops verify()'s judge_error; ask for it once, so the reason is on the page
            diag = r.sh(f"cd /work && {exports} python3 -c \"import sys; sys.path.insert(0, '/work'); import verify as v; "
                        f"from pathlib import Path; g = v._grade_sc(Path('{man['cwd']}')); print('JUDGE_ERR:' + str(g.get('judge_error', '')))\"",
                        timeout=900)
            why = next((l[10:] for l in (diag.stdout or "").splitlines() if l.startswith("JUDGE_ERR:")), "") or _tail(diag, 300)
            r.log("Judge error: " + why)
            detail["reward_error"] = f"judge_crashed: {why[:200]}"
        # masking contract: reward.json without a numeric reward in [0, 1] means the testbed failed, not the model
        try:
            reward = float(json.loads(reward_raw)["reward"])
            err = None if math.isfinite(reward) and 0.0 <= reward <= 1.0 else "reward_out_of_range"
        except (ValueError, KeyError, TypeError):
            reward, err = None, detail.get("reward_error") or "missing_or_invalid_reward_json"
        r.emit("checks", reward=None if err else reward, checks=checks, error=err,
               summary=f"judge unavailable ({err}): not scored" if err else f"{sum(1 for c in checks if c['passed'])}/{len(checks)} checks passed")
        r.phase("verify", "done", f"reward {reward}" if not err else f"not scored: {err}")
        return {"reward": None if err else reward, "error": err}


# ── Terminal-bench style tasks (in the General domain) ───────────────────────
class Terminal:
    """Harbor/Terminal-Bench convention (no Terminal-Bench environment ships in XiaomiMiMo/verl or mimoagent; this
    follows the same protocol as mimoagent's deepswe.py): tests arrive in /tests after the agent exits,
    `bash /tests/test.sh` writes /logs/verifier/reward.txt, and "-1" there is the verifier's crash sentinel."""
    steps = 500

    def run(self, r) -> dict:
        tid = r.run["task_id"]
        inst = catalog.rows()[tid]["instance"]
        cwd = inst.get("cwd") or "/app"
        r.start_sandbox(catalog.image_for(tid), config.FLAVORS["general"])
        r.phase("setup", detail="scrubbing test leftovers")
        r.sh("rm -rf /tests /logs/verifier")
        before = set((r.sh(f"ls -A {shlex.quote(cwd)} 2>/dev/null").stdout or "").split())
        version = opencode.install(r)   # before the blocklist: the release is fetched from github.com
        opencode.block_answer_hosts(r)
        r.phase("setup", "done", f"OpenCode {version}")

        r.phase("agent", detail=r.run["model"])
        prompt = General.prompt.format(cwd=cwd, task=inst["problem_statement"])
        opencode.run(r, prompt, cwd, steps=self.steps, timeout=float(inst.get("agent_timeout_sec") or 900) + 300)
        r.phase("agent", "done")

        r.phase("verify", detail="hidden tests")
        # the tests' anti-hack guard rejects files planted in the workspace: drop what the harness itself left there
        for name in (".opencode",):
            if name not in before:
                r.sh(f"rm -rf {shlex.quote(cwd)}/{name}")
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
        res = r.sh(f"cd {shlex.quote(cwd)} && bash /tests/test.sh", timeout=float(inst.get("verifier_timeout_sec") or 300) + 120)
        out = _tail(res, 8000)
        r.log(out)
        rew = (r.sh("cat /logs/verifier/reward.txt 2>/dev/null").stdout or "").strip()
        guard = re.search(r"anti_hack_guard: REJECT (\S+)", out)
        if rew == "-1" or not rew:
            r.emit("checks", reward=None, checks=[], error="verifier_crashed",
                   summary="The tests crashed before grading (testbed problem): not scored." if rew == "-1" else "The tests wrote no reward: not scored.")
            return {"reward": None, "error": "verifier_crashed"}
        try:
            reward = max(0.0, min(1.0, float(rew)))
        except ValueError:
            r.emit("checks", reward=None, checks=[], error="bad_reward", summary=f"reward.txt held {rew[:40]!r}: not scored.")
            return {"reward": None, "error": "bad_reward"}
        checks = []
        if guard:
            checks.append({"id": "anti_hack_guard", "passed": False, "method": "tests", "message": f"rejected: {guard.group(1)}"})
        ctrf = r.sh("cat /logs/verifier/ctrf.json 2>/dev/null").stdout
        try:
            for t in json.loads(ctrf)["results"]["tests"]:
                checks.append({"id": t.get("name"), "passed": t.get("status") == "passed", "method": "tests",
                               "message": (t.get("message") or t.get("status") or "")[:300]})
        except (ValueError, KeyError, TypeError):
            if not guard:
                checks.append({"id": "tests", "passed": reward == 1.0, "method": "tests", "message": f"reward.txt = {rew}"})
        summary = (f"Rejected by the task's anti-hack guard ({guard.group(1)})" if guard
                   else f"{sum(bool(c['passed']) for c in checks)}/{len(checks)} tests passed")
        r.emit("checks", reward=reward, checks=checks, summary=summary)
        r.phase("verify", "done", f"reward {reward}")
        return {"reward": reward}


# ── Webdev: build a site; graded by a vision model on a full-page render ─────



class Webdev:
    steps, timeout = 64, 1800   # verl config/agent/design/webdev-eval.yaml: step_limit 64

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
        version = opencode.install(r)   # no answer-leak blocklist here: there is no answer to leak, and npm/CDNs are fair game
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
        # verl recipes/design/webdev: shot.py (vendored byte for byte) with WEBDEV_GRADE_HTTP=1, i.e. served over a
        # loopback http server, so React/Vue/ES-module sites actually run instead of rendering blank from file://
        from ..vendor.webdev.shot import _build_shot_cmd, _parse_render_env, _parse_shot_b64
        import os as _os
        _os.environ["WEBDEV_GRADE_HTTP"] = "1"
        res = r.sh(_build_shot_cmd(f"file://{cwd}/dist/index.html"), timeout=150)   # SHOT_TIMEOUT_S
        shot = {"output": (res.stdout or "") + "\n" + (res.stderr or "")}
        shot_b64, console_errors = _parse_shot_b64(shot)
        render_env = _parse_render_env(shot)
        if render_env.get("proxy_failed") or not shot_b64:
            r.log(_tail(res, 2000))
            why = ("external assets were unreachable on every route" if render_env.get("proxy_failed")
                   else "the screenshot produced no image")
            r.emit("checks", reward=None, checks=[{"id": "render", "passed": False, "message": why}],
                   summary=f"Not scored: {why} (a render failure is not a bad page).")
            return {"reward": None, "error": "render failed"}
        jpg = base64.b64decode(shot_b64)
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(io.BytesIO(jpg))
        if im.width * im.height > 20e6:   # eval_mode._shrink: the judge refuses oversized shots; proportional, quality 75
            k = (20e6 / (im.width * im.height)) ** 0.5
            im = im.convert("RGB").resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
            b = io.BytesIO(); im.save(b, "JPEG", quality=75); jpg = b.getvalue()
        from .. import store
        store.write_artifact(r.id, "screenshot.jpg", jpg)
        r.emit("image", name="screenshot.jpg", width=im.width, height=im.height)
        if console_errors not in ("", "[]"):
            r.log("Console errors while rendering: " + console_errors[:1500])

        judge = r.run.get("judge") or "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
                   {"type": "text", "text": build_prompt().format(query=raw["prompt"][:1500])}]   # QUERY_CAP
        verdict, last = None, ""
        for attempt in range(4):   # JUDGE_RETRIES, with eval_mode's backoff
            if attempt:
                time.sleep(min(60.0, 3.0 * (3 ** (attempt - 1))) * (0.5 + secrets.randbelow(1000) / 1000))
            r.check_cancel()
            try:
                resp = httpx.post(f"{config.ROUTER}/chat/completions", timeout=180,   # JUDGE_TIMEOUT_S
                                  headers={"Authorization": f"Bearer {r.token}"},
                                  json={"model": judge, "temperature": 1.0,
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
        if r.run.get("scripted") is not None:   # validation: score a fixed piece, no model call
            return self._score(r, raw, r.run.get("scripted_final", ""), "", {}, params)
        if r.run.get("endpoint"):
            from .. import endpoints
            base = endpoints.check_url(base)   # re-checked at call time: DNS can change after the test
        text, thinking, usage = "", "", {}
        # verl recipes/design/config/music.yaml: response_length 100000. Providers that cap output lower refuse
        # such a request outright, so step down rather than fail the rollout on a limit.
        budgets = [params["max_tokens"]] if params.get("max_tokens") else [100000, 32000]
        for n, budget in enumerate(budgets):
            body = {"model": mid, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": budget,
                    **({"temperature": params["temperature"]} if params.get("temperature") is not None else {}),
                    **({"reasoning_effort": params["thinking"]} if params.get("thinking") not in (None, "default") else {}),
                    "messages": [{"role": "user", "content": raw["prompt"]}]}
            with httpx.stream("POST", f"{base}/chat/completions", timeout=900, json=body,
                              headers={"Authorization": f"Bearer {key}"} if key else {}) as resp:
                if resp.status_code != 200:
                    err = resp.read()[:400].decode(errors="replace")
                    if resp.status_code in (400, 422) and n + 1 < len(budgets):
                        r.log(f"The provider refused max_tokens={budget} ({err[:160]}); retrying with {budgets[n + 1]}.")
                        continue
                    raise RuntimeError(f"model call failed: HTTP {resp.status_code} {err[:300]}")
                r.update(params={**params, "max_tokens_used": budget})
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
                break
        return self._score(r, raw, text, thinking, usage, params)

    def _score(self, r, raw, text, thinking, usage, params) -> dict:
        from ..vendor.music_scorer.pipeline import do, extract_abc
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
        return {"steps": None, "timeout_min": None, "max_tokens": 100000}
    if domain == "general":
        inst = (catalog.rows().get(task_id) or {}).get("instance") or {}
        if inst.get("dataset_type") == "terminal_bench":
            return {"steps": Terminal.steps, "timeout_min": round((float(inst.get("agent_timeout_sec") or 900) + 300) / 60), "max_tokens": None}
    a = ADAPTERS[domain]
    return {"steps": a.steps, "timeout_min": round(a.timeout / 60), "max_tokens": None}
