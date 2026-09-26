"""The agent harness: OpenCode, run headless inside the task's sandbox.

`opencode run --format json` prints one JSON event per line. Those stream back through
`Sandbox.run(on_stdout=...)` and become the rollout's own events as they arrive.

Configured the way mimoagent's OpenCode adapter runs it (agents/blackbox/opencode.py): the injected
config is the only config (a repo's own opencode.json is ignored), no auto-update, no share, no LSP
download, no title-generation call, OpenCode's own state kept outside the workspace, and the output
ceiling raised past 32k tokens when asked.

Where this differs, and why: Xiaomi's pods had no route to the internet, so their agents could not look
answers up. An HF Sandbox does, so web search/fetch tools are off and the hosts where answers live (code
hosting, bug trackers, source-serving package proxies, search engines) are blackholed in /etc/hosts, the
mechanism mimoagent itself uses (``answer_leak_blocklist``). Everything else, including curl to local
services, works as usual. (A first test run on a Code task found the upstream fix with ``webfetch``.)
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

# Blackholed for the agent (mimoagent's answer_leak_blocklist mechanism). The model endpoint and HF are untouched.
ANSWER_HOSTS = [
    "github.com", "www.github.com", "api.github.com", "raw.githubusercontent.com", "codeload.github.com",
    "objects.githubusercontent.com", "gist.github.com", "gist.githubusercontent.com", "gitlab.com", "bitbucket.org",
    "sourceforge.net", "git.kernel.org", "chromium.googlesource.com", "android.googlesource.com", "sourcegraph.com",
    "bugs.chromium.org", "issues.oss-fuzz.com", "oss-fuzz.com", "osv.dev", "api.osv.dev", "bugzilla.mozilla.org",
    "proxy.golang.org", "sum.golang.org", "pkg.go.dev", "google.com", "www.google.com", "bing.com", "www.bing.com",
    "duckduckgo.com", "html.duckduckgo.com", "search.brave.com", "stackoverflow.com",
]


def block_answer_hosts(r) -> None:
    lines = "\n".join(f"0.0.0.0 {h}" for h in ANSWER_HOSTS)
    res = r.sh(f"printf '%b\\n' '# mimo-explorer answer-leak blocklist\\n{lines}' >> /etc/hosts && grep -c '^0.0.0.0' /etc/hosts")
    if res.exit_code != 0:   # fail closed, as mimoagent does
        raise RuntimeError("could not install the answer-leak blocklist in /etc/hosts: " + (res.stderr or res.stdout or "")[-300:])


def install(r) -> str:
    res = r.sh(INSTALL, timeout=300)
    version = (res.stdout or "").strip().splitlines()[-1:] or ["?"]
    if res.exit_code != 0:
        raise RuntimeError("could not install OpenCode in the sandbox: " + (res.stderr or res.stdout or "")[-300:])
    prov = r.run.get("provenance") or {}
    if prov:   # what actually ran, not only what was pinned
        r.update(provenance={**prov, "harness": {**prov.get("harness", {}), "installed": version[0]}})
    return version[0]


def config_for(model: str, provider: str | None, steps: int, mcp: list[dict] | None = None,
               endpoint: dict | None = None, params: dict | None = None, proxy: str | None = None) -> dict:
    """With `proxy` (the Space), every model call goes to this server's per-rollout proxy and the sandbox holds no
    credential at all; the capability in the URL is the only secret, and it dies with the rollout."""
    params = params or {}
    if endpoint:   # your own OpenAI-compatible endpoint; without the proxy, the key arrives as AGENT_API_KEY
        mid, prov = model, {"byo": {"npm": "@ai-sdk/openai-compatible", "name": "Your endpoint",
                                    "options": {"baseURL": proxy or endpoint["base_url"],
                                                "apiKey": "proxied" if proxy else "{env:AGENT_API_KEY}"},
                                    "models": {model: {"name": model, "tool_call": True}}}}
        ref = f"byo/{model}"
    else:
        mid = f"{model}:{provider}" if provider else model
        prov = {"hf": {"npm": "@ai-sdk/openai-compatible", "name": "Hugging Face Inference Providers",
                       "options": {"baseURL": proxy or config.ROUTER, "apiKey": "proxied" if proxy else "{env:HF_TOKEN}"},
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
        # mimoagent: "lsp/formatter are coding capability, and snapshot is opencode's own undo history"
        "snapshot": True, "lsp": True, "formatter": True,
        "permission": {"webfetch": "deny", "websearch": "deny", "bash": "allow", "edit": "allow"},
    }


def run(r, prompt: str, cwd: str, *, steps: int, timeout: float, user: str | None = None,
        home: str | None = None, mcp: list[dict] | None = None) -> str:
    """Run the agent to completion. Returns its final message."""
    # the exact user message the agent starts from; OpenCode adds its own system prompt and tool definitions
    r.emit("prompt", text=prompt, harness="opencode")
    if r.run.get("scripted") is not None:   # validation only (app/validate.py): a fixed probe instead of a model
        return _scripted(r, cwd, user)
    ep, params = r.run.get("endpoint"), r.run.get("params") or {}
    steps = params.get("steps") or steps
    timeout = params["timeout_min"] * 60 if params.get("timeout_min") else timeout
    proxy = r.llm_base()
    cfg = config_for(r.run["model"], r.run.get("provider"), steps, mcp, ep, params, proxy)
    r.sandbox.files.write("/tmp/opencode.json", json.dumps(cfg))
    r.sandbox.files.write("/tmp/prompt.txt", prompt)
    r.sh("chmod 644 /tmp/opencode.json /tmp/prompt.txt")
    key_env = ""
    if ep and not proxy:
        # a root-only file, read by the root shell when it builds the command: the key is never on a command line
        r.sandbox.files.write("/root/.agent_key", r.agent_key or "")
        r.sh("chmod 600 /root/.agent_key")
        key_env = 'AGENT_API_KEY="$(cat /root/.agent_key)" '
    xdg = "/tmp/.opencode-state"   # OpenCode's own sessions/caches: outside the workspace, which is graded state
    r.sh(f"mkdir -p {xdg} && chmod 777 {xdg}")
    flags = {"OPENCODE_CONFIG": "/tmp/opencode.json", "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
             "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
             "OPENCODE_DISABLE_SHARE": "1", "XDG_DATA_HOME": f"{xdg}/data", "XDG_CONFIG_HOME": f"{xdg}/config",
             "XDG_CACHE_HOME": f"{xdg}/cache", "XDG_STATE_HOME": f"{xdg}/state"}
    if (params.get("max_tokens") or 0) > 32000:   # upstream silently caps output at 32k otherwise
        flags["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(params["max_tokens"])
    token_env = "" if proxy else 'HF_TOKEN="$HF_TOKEN" '
    env = f"env {key_env}{token_env}HOME={home or '$HOME'} " + " ".join(f"{k}={v}" for k, v in flags.items())
    # --thinking streams the model's reasoning as its own events, so the trace shows why, not only what;
    # --title skips the extra model call that would otherwise name the session
    inner = f'{env} opencode run --format json --thinking --auto --title mimo-rollout "$(cat /tmp/prompt.txt)"'
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
        if ep and not proxy:
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


def _scripted(r, cwd: str, user: str | None) -> str:
    """Stand-in for the model in health checks: run a probe as the agent user, in the agent's directory."""
    probe = r.run["scripted"]
    if probe:
        r.sandbox.files.write("/tmp/.probe.sh", probe)
        r.sh("chmod 755 /tmp/.probe.sh")
        cmd = f"cd {shlex.quote(cwd)} && " + (f"runuser -u {user} -- bash /tmp/.probe.sh" if user else "bash /tmp/.probe.sh")
        res = r.sh(cmd, timeout=600)
        r.emit("tool", tool="bash", title="probe", input={"command": probe[:2000]}, output=((res.stdout or "") + (res.stderr or ""))[:8000],
               truncated=False, status="completed" if res.exit_code == 0 else "error", ms=None)
    final = r.run.get("scripted_final", "")
    if final:
        r.emit("text", text=final)
    return final
