"""The agent harness: OpenCode, run headless inside the task's sandbox, calling HF Inference Providers.

`opencode run --format json --auto` prints one JSON event per line. Those stream back through
`Sandbox.run(on_stdout=...)` and are turned into the rollout's own events as they arrive.

Rules every rollout gets, whatever the task:
  * no web tools, and no curl / wget / git fetch in bash: the model is remote, so the network cannot
    be cut entirely, but the agent should not look the answer up. (A first test run on a Code task
    found the upstream fix commit with `webfetch` and copied it.)
  * a step cap per domain, the same as Xiaomi's training harness uses.
"""

from __future__ import annotations

import json
import shlex

from .. import config

VERSION = "1.18.32"   # pinned: a rollout should not change because OpenCode shipped overnight
# The release binary, fetched with whatever the image has (some task images have no curl), and the
# same build variants the official installer picks: -baseline without AVX2, -musl on musl libc.
INSTALL = r"""command -v opencode >/dev/null 2>&1 || {
  t=linux-x64; grep -qw avx2 /proc/cpuinfo || t=$t-baseline; (ldd --version 2>&1 | grep -qi musl) && t=$t-musl
  u=https://github.com/anomalyco/opencode/releases/download/v%s/opencode-$t.tar.gz
  { curl -fsSL "$u" -o /tmp/.oc.tgz || wget -qO /tmp/.oc.tgz "$u" ||
    python3 -c "import sys,urllib.request; urllib.request.urlretrieve(sys.argv[1], '/tmp/.oc.tgz')" "$u"; } &&
  tar -xzf /tmp/.oc.tgz -C /tmp opencode && install -m 755 /tmp/opencode /usr/local/bin/opencode && rm -f /tmp/.oc.tgz /tmp/opencode
}; opencode --version""" % VERSION

DENY_BASH = ["curl *", "wget *", "git fetch*", "git clone*", "git pull*", "git remote*", "pip download*"]


def install(r) -> str:
    res = r.sh(INSTALL, timeout=300)
    version = (res.stdout or "").strip().splitlines()[-1:] or ["?"]
    if res.exit_code != 0:
        raise RuntimeError("could not install OpenCode in the sandbox: " + (res.stderr or res.stdout or "")[-300:])
    return version[0]


def config_for(model: str, provider: str | None, steps: int, mcp: list[dict] | None = None,
               endpoint: dict | None = None, params: dict | None = None) -> dict:
    params = params or {}
    if endpoint:   # your own OpenAI-compatible endpoint; the key arrives as AGENT_API_KEY
        mid, prov = model, {"byo": {"npm": "@ai-sdk/openai-compatible", "name": "Your endpoint",
                                    "options": {"baseURL": endpoint["base_url"], "apiKey": "{env:AGENT_API_KEY}"},
                                    "models": {model: {"name": model, "tool_call": True}}}}
        ref = f"byo/{model}"
    else:
        mid = f"{model}:{provider}" if provider else model
        prov = {"hf": {"npm": "@ai-sdk/openai-compatible", "name": "Hugging Face Inference Providers",
                       "options": {"baseURL": config.ROUTER, "apiKey": "{env:HF_TOKEN}"},
                       "models": {mid: {"name": model, "tool_call": True}}}}
        ref = f"hf/{mid}"
    entry = next(iter(next(iter(prov.values()))["models"].values()))
    if params.get("thinking") and params["thinking"] != "default":
        entry["options"] = {"reasoningEffort": params["thinking"]}   # sent as reasoning_effort ("none" turns it off)
    if params.get("max_tokens"):
        entry["limit"] = {"context": 1_000_000, "output": params["max_tokens"]}
    build = {"steps": steps}
    if params.get("temperature") is not None:
        build["temperature"] = params["temperature"]
    return {
        "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
        "model": ref,
        "provider": prov,
        "agent": {"build": build},
        "mcp": {s["name"]: {"type": "remote", "url": s["url"], "enabled": True} for s in (mcp or [])},
        "permission": {"webfetch": "deny", "websearch": "deny",
                       "bash": {"*": "allow", **{p: "deny" for p in DENY_BASH}}},
    }


def run(r, prompt: str, cwd: str, *, steps: int, timeout: float, user: str | None = None,
        home: str | None = None, mcp: list[dict] | None = None) -> str:
    """Run the agent to completion. Returns its final message."""
    ep, params = r.run.get("endpoint"), r.run.get("params") or {}
    steps = params.get("steps") or steps
    timeout = params["timeout_min"] * 60 if params.get("timeout_min") else timeout
    cfg = config_for(r.run["model"], r.run.get("provider"), steps, mcp, ep, params)
    r.sandbox.files.write("/tmp/opencode.json", json.dumps(cfg))
    r.sandbox.files.write("/tmp/prompt.txt", prompt)
    r.sh("chmod 644 /tmp/opencode.json /tmp/prompt.txt")
    key_env = ""
    if ep:
        # a root-only file, read by the root shell when it builds the command: the key is never on a command line
        r.sandbox.files.write("/root/.agent_key", r.agent_key or "")
        r.sh("chmod 600 /root/.agent_key")
        key_env = 'AGENT_API_KEY="$(cat /root/.agent_key)" '
    env = f"env {key_env}HF_TOKEN=\"$HF_TOKEN\" OPENCODE_CONFIG=/tmp/opencode.json HOME={home or '$HOME'}"
    # --thinking streams the model's reasoning as its own events, so the trace shows why, not only what
    inner = f'{env} opencode run --format json --thinking --auto "$(cat /tmp/prompt.txt)"'
    cmd = f"cd {shlex.quote(cwd)} && " + (f"runuser -u {user} -- {inner}" if user else inner)

    buf = [""]
    texts: list[str] = []

    def on_out(chunk: str) -> None:
        buf[0] += chunk
        *lines, buf[0] = buf[0].split("\n")
        for line in lines:
            if line.strip():
                _event(r, line, texts)

    def on_err(chunk: str) -> None:
        s = chunk.strip()
        if s and "DeprecationWarning" not in s:
            r.log(s[-1500:])

    try:
        res = r.sh(cmd, timeout=timeout, on_stdout=on_out, on_stderr=on_err)
    finally:
        if ep:
            r.sandbox.run("rm -f /root/.agent_key", shell=True, check=False, timeout=30)
    if buf[0].strip():
        _event(r, buf[0], texts)
    if res.timed_out:
        r.emit("error", text=f"The agent hit the {int(timeout // 60)}-minute limit; grading what it left.")
    return texts[-1] if texts else ""


def _event(r, line: str, texts: list[str]) -> None:
    try:
        ev = json.loads(line)
    except ValueError:
        r.log(line[:1500])
        return
    typ, part = ev.get("type"), ev.get("part") or {}
    if typ == "text" and part.get("text"):
        texts.append(part["text"])
        r.emit("text", text=part["text"])
    elif typ == "reasoning" and (part.get("text") or "").strip():
        t = part["text"].strip()
        r.emit("thinking", text=t[:20000], chars=len(t))
    elif typ == "tool_use":
        st = part.get("state") or {}
        tm = st.get("time") or {}
        out = st.get("output")
        if not isinstance(out, str):
            out = json.dumps(out, ensure_ascii=False) if out is not None else ""
        r.emit("tool", tool=part.get("tool"), title=st.get("title") or "", input=st.get("input"),
               output=out[:8000], truncated=len(out) > 8000, status=st.get("status"),
               ms=(tm.get("end") - tm.get("start")) if tm.get("end") and tm.get("start") else None)
    elif typ == "step_finish":
        t = part.get("tokens") or {}
        cache = t.get("cache") or {}
        tok = {"input": t.get("input"), "output": t.get("output"), "reasoning": t.get("reasoning"),
               "cache_read": cache.get("read"), "cache_write": cache.get("write")}
        r.add_tokens(tok)
        r.emit("step", tokens=tok, cost=r.cost_now()["model"])
    elif typ == "error":
        err = ev.get("error") or part
        r.emit("error", text=(err.get("data") or {}).get("message") or json.dumps(err)[:800]
               if isinstance(err, dict) else str(err)[:800])
