"""Harbor task folders (`task.toml`, `instruction.md`, `tests/`, `environment/`): indexed once (app/catalog.py), then
every task is a folder that runs as it is on an HF Sandbox through OpenEnv's Harbor runner."""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any

from .. import catalog
from . import contract as c

KIND = {"json": "Named scores", "txt": "One score"}
LEVELS = ("difficulty",)


def _kind(r: dict[str, Any]) -> str:
    return KIND.get(r["verifier"]["kind"], "Not found")


def _runs_in(r: dict[str, Any]) -> str:
    e = r["env"]
    return ("Prebuilt image" if e["image"] else "Dockerfile, replayed" if r.get("run") == "dockerfile" else "Compose, can't run here"
            if e["compose"] else "Dockerfile, can't run here" if e["dockerfile"] else "Not given")


def _secs(s: Any) -> str | None:
    if not isinstance(s, (int, float)):
        return None
    return f"{s / 3600:.1f} h".replace(".0 h", " h") if s >= 3600 else f"{round(s / 60)} min" if s >= 60 else f"{int(s)} s"


def _median(xs: list[Any]) -> Any:
    v = sorted(x for x in xs if isinstance(x, (int, float)))
    return v[len(v) // 2] if v else None


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


class HarborAdapter(c.Adapter):
    id, name, framework = "harbor", "Harbor task folders", "Harbor"
    about = ("Each task is a folder: task.toml (settings), instruction.md (what the agent is asked), tests/ (the grader, "
             "which writes the reward) and environment/ (what it runs in).")
    kinds = ("dataset",)

    def detect(self, kind: str, meta: dict[str, Any]) -> float:
        if kind != "dataset":
            return 0.0
        if meta.get("framework") == "harbor":
            return 0.9          # tagged: confirmed by its index (a Harbor tag on packed rows hands it to the rows reader)
        return 0.85 if catalog.looks_harbor(meta["id"], meta["sha"], None) else 0.0

    def confirm(self, env: c.Env) -> bool:
        st = catalog.index_status(env.id, env.read_token)
        return not (st.get("state") == "done" and not st["index"]["tasks"])

    def settled(self, env: c.Env) -> bool:
        """Whether the choice of this adapter is final: once the index is done (until then it may hand over)."""
        return catalog.index_status(env.id, env.read_token).get("state") == "done"

    # ── tasks ──
    def summary(self, env: c.Env, subset: str | None = None) -> dict[str, Any]:
        st = catalog.index_status(env.id, env.read_token)
        if st.get("state") != "done":
            return {"state": "indexing" if st.get("state") != "error" else "error", "progress": st, "total": None, "inline": True,
                    "how": self._how(), "overview": [], "facets": [], "subsets": [], "search": True}
        idx = st["index"]
        if not idx["tasks"]:
            return {"state": "handoff"}   # no task folders after all (tasks packed into rows): the registry asks the next adapter
        rows, s = idx["tasks"], idx["summary"]
        facets = [c.facet("graded", "Graded by"), c.facet("runs", "Runs in")]
        if any(_setup(r) for r in rows):
            facets.append(c.facet("setup", "Setup"))
        names = {"difficulty": "Difficulty", "category": "Category", "group": "Group", "tags": "Tags"}
        for key in (s.get("facets") or {}):
            label = key[5:].replace("_", " ") if key.startswith("meta:") else names.get(key, key)
            facets.append(c.facet(key, label, "level" if key == "difficulty" or "level" in key or "tier" in key else "count"))
        return {"state": "ready", "progress": st.get("progress"), "total": len(rows), "inline": True, "search": True,
                "how": self._how(), "facets": facets, "subsets": [], "tile": _pick_tile(rows), "map": None,
                "overview": [self._graded(rows, s), self._runs(rows, s)]}

    def _how(self) -> dict[str, Any]:
        return {"framework": self.framework, "name": self.name, "about": self.about}

    def _graded(self, rows: list[dict[str, Any]], s: dict[str, Any]) -> dict[str, Any]:
        n = s["tasks"]
        kinds = sorted((s.get("verifier") or {}).items(), key=lambda kv: -kv[1])
        label = {"json": "Named scores in `reward.json`", "txt": "One score in `reward.txt`"}
        shares = c.shares([(label.get(k, "Not found in `tests/test.sh`"), v) for k, v in kinds], n)
        checks = [f"pytest in {_plural(s['pytest'], 'task')}" if s.get("pytest") else "", f"a model judge in {s['judge']:,}" if s.get("judge") else ""]
        vt = _median([r["verifier"]["timeout"] for r in rows])
        sep = [r["verifier"]["separate"] for r in rows if r["verifier"].get("separate")]
        built = sum(1 for x in sep if not x.get("image") and x.get("dockerfile"))
        return c.section("graded", "How it's graded", [
            {"type": "kv", "rows": [["Reward", shares]]},
            c.kv([("Score names", " ".join(f"`{k}` {v:,}" for k, v in s.get("reward_keys") or []) or None),
                  ("Checks", " · ".join(x for x in checks if x) or "the task's own script"),
                  ("Time limit", f"{_secs(vt)} (median)" if vt else None),
                  ("Grader's own container", f"{_plural(len(sep), 'task')} grade in a container of their own, not the agent's sandbox"
                   + (f" ({built:,} built from `tests/Dockerfile`)" if built else "") if sep else None),
                  ("Reference solution", f"in {_plural(s['solution'], 'task')}, listed, never shown" if s.get("solution") else "none"),
                  ("Withheld", (", ".join(f"`{k}`" for k in s["withheld"]) + " (metadata that holds answers)") if s.get("withheld") else None)]),
        ], icon="target", note="read from each task's tests")

    def _runs(self, rows: list[dict[str, Any]], s: dict[str, Any]) -> dict[str, Any]:
        n = s["tasks"]
        where = [(w, k) for k, w in ((s["image"], "Prebuilt image"), (s["dockerfile_only"], "Dockerfile"), (n - s["image"] - s["dockerfile_only"], "Not given")) if k > 0]
        cpus, mem, at = _median([r["env"]["cpus"] for r in rows]), _median([r["env"]["memory_mb"] for r in rows]), _median([r["agent_timeout"] for r in rows])
        net = sum(1 for r in rows if r["env"]["internet"] is False)
        phased = sum(1 for r in rows if any(m != "public" for m in (r["env"].get("phases") or {}).values()))
        mcp, hc = sum(1 for r in rows if r["env"].get("mcp")), sum(1 for r in rows if r["env"].get("healthcheck"))
        win, bad = sum(1 for r in rows if r["env"].get("os") == "windows"), sum(1 for r in rows if (r.get("spec") or {}).get("harbor_error"))
        return c.section("runs", "What the tasks run in", [
            {"type": "kv", "rows": [["Environment", c.shares(where, n)],
                                    *([["Runs here", c.shares([("on an HF Sandbox", s["runnable"])], n,
                                                              f"{s['replayed']:,} by replaying their Dockerfile on its base image" if s.get("replayed") else "")]]
                                      if s.get("runnable") is not None else [])]},
            c.kv([("Resources", ", ".join(x for x in (f"{cpus:g} CPU{'' if cpus == 1 else 's'}" if cpus else "", f"{_mb(mem)} memory" if _mb(mem) else "") if x) + " (median)" if cpus or mem else None),
                  ("Agent time limit", f"{_secs(at)} (median)" if at else None),
                  ("Internet", f"blocked in {_plural(net, 'task')}" if net else None),
                  ("Per phase", f"{_plural(phased, 'task')} limit the agent's or the grader's network on their own" if phased else None),
                  ("MCP servers", f"given to the agent in {_plural(mcp, 'task')}" if mcp else None),
                  ("Healthcheck", f"{_plural(hc, 'task')} wait for one before the agent starts" if hc else None),
                  ("Windows", f"{_plural(win, 'task')} target Windows containers" if win else None),
                  ("Several containers", f"{_plural(s['compose'], 'task')}, from a compose file" if s.get("compose") else None),
                  ("GPU", _plural(s["gpu"], "task") if s.get("gpu") else None),
                  ("Steps", f"{_plural(s['multi_step'], 'task')} give the instruction in steps" if s.get("multi_step") else None),
                  ("Runner's variables", f"{_plural(s['reads_env'], 'task')} pass `${{VAR}}` from whoever runs them" if s.get("reads_env") else None),
                  ("Unreadable", f"{_plural(s['invalid'], 'task.toml file')}" if s.get("invalid") else None),
                  ("Rejected by Harbor", f"{_plural(bad, 'task.toml file')} fail Harbor's own checks" if bad else None)]),
        ], icon="box", note="read from each task.toml")

    def tasks(self, env: c.Env, *, subset=None, q="", filters=None, offset=0, everything=False) -> dict[str, Any]:
        st = catalog.index_status(env.id, env.read_token)
        if st.get("state") != "done":
            return {"cards": [], "total": 0, "offset": 0, "page": 0, "note": "still indexing"}
        rows = st["index"]["tasks"]
        n = len(rows)
        shared_tags = {t for t in set().union(*(set(r["tags"]) for r in rows)) if all(t in r["tags"] for r in rows)} if rows else set()
        one_kind = len({_kind(r) for r in rows}) == 1
        one_runs = len({_runs_in(r) for r in rows}) == 1
        cards = []
        for r in rows:
            lead = r.get("category") or r.get("group") or r.get("difficulty")
            sub = [x for x in (r.get("group") if r.get("category") and r.get("group") else None,
                               r.get("difficulty") if r.get("difficulty") and r.get("difficulty") != lead else None) if x]
            chips = [
                (f"{_kind(r).lower()}{': ' + ', '.join(r['verifier']['keys'][:3]) if r['verifier']['keys'] else ''}") if not one_kind or r["verifier"]["keys"] else "",
                "" if one_runs else _runs_in(r).lower(),
                "model judge" if r["verifier"]["judge"] else "",
                f"{r['steps']} steps" if r.get("steps") else "",
                *[t for t in r["tags"] if t != lead and t not in shared_tags][:3],
            ]
            facets = {"graded": [_kind(r)], "runs": [_runs_in(r)], "tags": list(r["tags"]), "setup": _setup(r)}
            for k in ("difficulty", "category", "group"):
                if r.get(k):
                    facets[k] = [str(r[k])]
            for k, v in (r.get("meta") or {}).items():
                facets[f"meta:{k}"] = [str(v)]
            cards.append(c.card(r["path"], r["title"], id=r["path"].rsplit("/", 1)[-1], brief=r.get("brief") or "", lead=lead, sub=sub,
                                chips=[x for x in chips if x], facets=facets,
                                text=f"{r['path']} {r.get('brief') or ''} {' '.join(r['tags'])} {r.get('category') or ''} {r.get('group') or ''}"))
        return {"cards": cards, "total": n, "offset": 0, "page": n}

    def task(self, env: c.Env, ref: str) -> dict[str, Any]:
        t = catalog.task(env.id, ref, env.read_token)
        k, cfg = t["verifier"], t.get("config") or {}
        steps = t.get("steps_detail") or []
        shape = (f"Named scores in `/logs/verifier/reward.json`{': ' + ', '.join(f'`{x}`' for x in k['keys']) if k['keys'] else ' (keys set at run time)'}"
                 if k["kind"] == "json" else "One score in `/logs/verifier/reward.txt`" if k["kind"] == "txt"
                 else "Not found in `tests/test.sh`: the score is written elsewhere")
        tests = dict(t.get("tests") or {})
        sol = [f for f in t["tree"] if f.get("withheld") and f["path"].startswith("solution/")]
        step_sol = [s["name"] for s in steps if s.get("solution")]
        other = [f for f in k.get("files") or [] if f not in tests]
        ver = cfg.get("verifier") or {}
        grading = [
            c.kv([("Writes", shape),
                  ("Checks", ", ".join(x for x in ("tests run with pytest" if k.get("pytest") else "", "a model as judge" if k.get("judge") else "") if x) or "its own script"),
                  ("Reward", _strategy(cfg, len(steps)) if steps else None),
                  ("Time limit", _secs(k.get("timeout"))),
                  ("Gets", (", ".join(f"`{x}`" for x in k["env"]) + " (environment variables)") if k.get("env") else None),
                  ("Runs", _grader_runs(k.get("separate"), bool(steps), bool(cfg.get("artifacts")))),
                  ("Runs as", f"`{ver['user']}`" if ver.get("user") not in (None, "") else None)]),
            c.code(tests.pop("tests/test.sh"), "tests/test.sh", "tests/test.sh") if "tests/test.sh" in tests else None,
            *[c.disclose(f"{name} · {len(text.splitlines()):,} lines", [c.code(text, name)]) for name, text in tests.items()],
            c.note("Also in `tests/`: " + ", ".join(f"`{f[6:]}`" for f in other[:12]) + (f" and {len(other) - 12} more" if len(other) > 12 else "") + ".", "folder") if other else None,
            c.note("Each step is graded by its own `tests/` where it has one (below, with the step), else by these.", "list") if steps and tests else None,
            c.note("A reference solution exists (" + ", ".join(x for x in (f"{_plural(len(sol), 'file')} in `solution/`" if sol else "",
                                                                       f"one for {_plural(len(step_sol), 'step')} in its own `solution/`" if step_sol else "") if x)
                   + "). It is not shown here.", "shield") if sol or step_sol else None,
        ]
        from .. import judge as judges

        plan = judges.plan(t.get("toml") or "")
        if plan.get("judge"):
            j = plan["judge"]
            grading.insert(1, c.note(f"**Graded by a model**{' (it asks for `' + j['requested'] + '`)' if j['requested'] else ''}: "
                                     f"{', '.join(f'`{k}`' for k in j['keys'])} reach it. In a rollout here its calls go through a relay to the judge you "
                                     "pick on HF Inference Providers; the sandbox gets a key that works for this rollout's grading only.", "scale"))
        e = t["env"]
        envb = [
            c.kv([("Image", f"`{e['image']}` (prebuilt)" if e.get("image") else "built from `environment/Dockerfile`" if e.get("dockerfile") else "none given"),
                  ("Containers", "several, from a compose file" if e.get("compose") else None),
                  ("Operating system", "Windows" if e.get("os") == "windows" else None),
                  ("On an HF Sandbox", "its Dockerfile is replayed on its base image" if t["runnable"].get("how") == "dockerfile" else None)]),
            c.note("Its task.toml passes environment variables from whoever runs it (`${VAR}`) into the sandbox.", "key") if t.get("reads_env") else None,
            c.code(t["dockerfile"], "Dockerfile", "environment/Dockerfile") if t.get("dockerfile") else None,
            c.disclose("Compose file", [c.code(t["compose"], "compose.yaml")]) if t.get("compose") else None,
            c.disclose(f"Variables ({len(plan['vars'])}) and how a rollout here fills them", [c.kv(judges.table(plan))]) if plan["vars"] else None,
        ]
        meta_blocks = _metadata(t)
        if steps:
            task_blocks = [c.steps([(f"{s['name']}", _gist(s) or ("withheld" if s.get("withheld") else "")) for s in steps]),
                           c.note("Harbor gives the agent one step at a time, each with its own instruction, and grades it before the next.", "list"),
                           c.disclose("instruction.md (not sent: a multi-step task's instructions are its steps')", [c.markdown(t["instruction"])])
                           if (t.get("instruction") or "").strip() else None]
        else:
            task_blocks = [c.markdown(t.get("instruction") or "This task has no instruction.md.")]
        if (t.get("spec") or {}).get("trajectory"):
            task_blocks.append(c.note("It starts from a prior conversation: Harbor seeds the agent with `trajectory.json` first.", "message"))
        sections = [
            c.section("task", f"The task, in {len(steps)} steps" if steps else "The task", task_blocks, icon="message",
                      note="each step's instruction is below" if steps else "word for word, as the agent gets it"),
            *([c.section("steps", "Steps", _step_blocks(steps, cfg), icon="list", note="word for word, as the agent gets each")] if steps else []),
            c.section("grading", "How it's graded", grading, icon="target"),
            c.section("environment", "What it runs in", envb, icon="box"),
            c.section("config", "Configuration", _configuration(t, cfg), icon="wrench", note="task.toml, as Harbor reads it"),
            *([c.section("metadata", "Metadata", meta_blocks, icon="list")] if meta_blocks else []),
            c.section("files", "Files", [c.files(t["tree"], t.get("tree_truncated"), c.hub_folder(env.id, t["sha"], ref), not t.get("restricted"))], icon="folder",
                      note=f"{_plural(len([f for f in t['tree'] if not f.get('dir')]), 'file')} · {t['bytes']:,} bytes" + (" · the first 2,000 listed" if t.get("tree_truncated") else "")),
        ]
        glance = [
            ["Reward", "named scores" if k["kind"] == "json" else "one score" if k["kind"] == "txt" else "not found in test.sh"],
            ["Checks", ", ".join(x for x in ("pytest" if k.get("pytest") else "", "a model judge" if k.get("judge") else "") if x) or "its own script"],
            ["Runs in", "a prebuilt image" if e.get("image") else "a Dockerfile and compose" if e.get("compose") else "a Dockerfile, built first" if e.get("dockerfile") else "–"],
            *([["Steps", f"{len(steps)}"]] if steps else []),
            *([["Grader", "in its own container"]] if k.get("separate") else []),
            ["Agent time limit", _secs(t.get("agent_timeout")) or "–"],
            ["Reference solution", "yes, not shown" if t.get("solution") else "none"],
            ["Files", f"{t['files']:,}"],
        ]
        coll = t.get("collection")
        return {"ref": ref, "title": t["title"], "id": posixpath.basename(ref) or ref, "chips": [x for x in (t.get("difficulty"), t.get("category"), t.get("group") if t.get("group") != t.get("category") else None) if x],
                "collection": coll, "sections": sections, "glance": glance, "withheld": list(t.get("withheld") or []),
                "hub": f"https://huggingface.co/datasets/{env.id}/tree/{t['sha']}/{ref}", "sha": t["sha"], "restricted": t.get("restricted"),
                "summary": f"{'named scores' if k['kind'] == 'json' else 'one score' if k['kind'] == 'txt' else 'score not found'}",
                "_runnable": t["runnable"], "_toml": t.get("toml") or "", "_image": e.get("image"), "links": []}

    def file(self, env: c.Env, ref: str, path: str) -> dict[str, Any]:
        return catalog.task_file(env.id, ref, path, env.read_token)

    def folder(self, env: c.Env, ref: str, path: str) -> dict[str, Any]:
        return catalog.task_folder(env.id, ref, path, env.read_token)

    def random(self, env: c.Env, subset=None) -> str:
        return catalog.random_task(env.id, env.read_token)["path"]

    # ── running ──
    def run_options(self, env: c.Env, ref: str, view: dict[str, Any]) -> list[dict[str, Any]]:
        return [harbor_option(view["_runnable"], view.get("_toml") or "")]

    def materialize(self, env: c.Env, ref: str) -> Path:
        return catalog.task_dir(env.id, ref, env.read_token)

    def run_task(self, env: c.Env, ref: str) -> dict[str, Any]:
        t = catalog.task(env.id, ref, env.read_token)
        return {"title": t["title"], "restricted": bool(t.get("restricted")), "sha": t["sha"], "bytes": t.get("bytes") or 0,
                "runnable": t["runnable"], "image": t["env"]["image"] or t["runnable"].get("base"),
                "collection": (t.get("collection") or {}).get("id"), "difficulty": t.get("difficulty"), "category": t.get("category")}


# ── a task's settings, as Harbor's own viewer lays them out ─────────────────
NET = {"public": "internet", "no-network": "no internet", "allowlist": "allowlisted hosts only"}
PHASE = {"agent": "agent", "verifier": "grader", "verifier.environment": "grader's container"}


def _harbor_version() -> str:
    try:
        from importlib.metadata import version

        return version("harbor")
    except Exception:  # noqa: BLE001
        return ""


def _scalarish(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _fmt(v: Any, limit: int = 400) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return ", ".join(_fmt(x, 80) for x in v[:30]) + (f" and {len(v) - 30} more" if len(v) > 30 else "")
    s = "" if v is None else str(v)
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


def _code(s: Any, limit: int = 100) -> str:
    """A value as inline code on one line, cut at `limit`."""
    t = " ".join(str(s).split()).replace("`", "'")
    return f"`{t if len(t) <= limit else t[:limit - 1].rstrip() + '…'}`"


def _mb(v: Any) -> str | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        return None
    return f"{v / 1024:g} GB" if v >= 1024 else f"{v:g} MB"


def _resources(e: dict[str, Any] | None) -> str | None:
    """CPUs, memory, storage, GPUs and a TPU, as Harbor reads them (legacy `memory = "4G"` converted)."""
    e = e or {}
    out = []
    if isinstance(e.get("cpus"), (int, float)) and e["cpus"]:
        out.append(f"{e['cpus']:g} CPU{'' if e['cpus'] == 1 else 's'}")
    for key, word in (("memory_mb", "memory"), ("storage_mb", "storage")):
        if _mb(e.get(key)):
            out.append(f"{_mb(e[key])} {word}")
    if isinstance(e.get("gpus"), int) and e["gpus"] > 0:
        out.append(_plural(e["gpus"], "GPU") + (f" ({', '.join(map(str, e['gpu_types']))})" if e.get("gpu_types") else ""))
    tpu = e.get("tpu")
    if isinstance(tpu, dict) and tpu.get("type"):
        out.append(f"a TPU slice, {tpu['type']} {tpu.get('topology') or ''}".strip())
    return " · ".join(out) or None


def _phase(key: str) -> str:
    """A phase key (catalog._phases) in a few words: "steps.s1.agent" is "step `s1`, agent"."""
    if key.startswith("steps."):
        for own, word in sorted(PHASE.items(), key=lambda kv: -len(kv[0])):
            if key.endswith("." + own):
                return f"step `{key[6:-len(own) - 1]}`, {word}"
    return PHASE.get(key, key)


def _network(e: dict[str, Any]) -> str:
    """The sandbox's network, then each phase that sets its own."""
    base = e.get("network") or "public"
    return " · ".join([f"sandbox: {NET.get(base, base)}"] + [f"{_phase(k)}: {NET.get(m, m)}" for k, m in (e.get("phases") or {}).items()])


def _strategy(cfg: dict[str, Any], n: int) -> str:
    s = cfg.get("multi_step_reward_strategy")
    how = "the last step's reward" if s == "final" else "the mean of the steps' rewards, key by key"
    return f"{how} (`multi_step_reward_strategy = \"{s}\"`)" if s else f"{how} (Harbor's default for a task in steps)"


def _grader_runs(sep: dict[str, Any] | None, multi: bool, artifacts: bool) -> str:
    """Where the grader runs: in the agent's sandbox, or (`environment_mode = "separate"`) in a container of its own."""
    when = "after each step" if multi else "after the agent finishes"
    if not sep:
        return f"{when}, in the same sandbox, with `tests/` copied in"
    src = (f"started from `{sep['image']}`" if sep.get("image") else "built from `tests/Dockerfile`" if sep.get("dockerfile")
           else "a fresh copy of the task's environment")
    some = f" (for {_plural(sep['steps'], 'step')})" if multi and sep.get("steps") else ""
    return (f"{when}, in a separate container{some}, {src}, not in the agent's sandbox"
            + ("; it gets the files the task collects as artifacts" if artifacts else ""))


def _min_reward(v: Any) -> str | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"reward ≥ {v:g}, else the steps after it are skipped"
    if isinstance(v, dict) and v:
        return ", ".join(f"`{k}` ≥ {x:g}" for k, x in v.items() if isinstance(x, (int, float))) + ", else the steps after it are skipped"
    return None


def _healthcheck(hc: dict[str, Any] | None) -> str | None:
    if not isinstance(hc, dict) or not hc.get("command"):
        return None
    n = hc.get("retries")
    how = [f"every {_secs(hc.get('interval_sec'))}" if _secs(hc.get("interval_sec")) else "",
           f"up to {_secs(hc.get('timeout_sec'))} a try" if _secs(hc.get("timeout_sec")) else "",
           f"giving up after {'1 failed try' if n == 1 else f'{n} failed tries in a row'}" if isinstance(n, int) and n > 0 else "",
           f"after a {_secs(hc['start_period_sec'])} grace period" if (hc.get("start_period_sec") or 0) > 0 else ""]
    return f"{_code(hc['command'], 80)}, until it passes, before the agent starts: " + ", ".join(x for x in how if x)


def _mcp(m: Any) -> str | None:
    if not isinstance(m, dict) or not m.get("name"):
        return None
    where = m.get("url") or " ".join([str(m.get("command") or "")] + [str(a) for a in m.get("args") or []]).strip()
    return f"`{m['name']}` · {m.get('transport') or 'sse'}" + (f" · {_code(where, 90)}" if where else "")


def _artifacts(items: Any) -> str | None:
    out = []
    for a in items or []:
        if isinstance(a, str):
            out.append(_code(a, 80))
        elif isinstance(a, dict) and a.get("source"):
            out.append(_code(a["source"], 80) + (f" → `{a['destination']}`" if a.get("destination") else "")
                       + (f" from `{a['service']}`" if a.get("service") not in (None, "", "main") else "")
                       + (f" (not {', '.join(_code(x, 30) for x in a['exclude'][:4])})" if a.get("exclude") else ""))
    return (", ".join(out[:8]) + (f" and {len(out) - 8} more" if len(out) > 8 else "")) if out else None


def _collect(hooks: Any) -> str | None:
    hooks = [h for h in hooks or [] if isinstance(h, dict) and h.get("command")]
    if not hooks:
        return None
    return "; ".join(f"{_code(h['command'], 80)} in `{h.get('service') or 'main'}`" for h in hooks[:4]) + (f" and {len(hooks) - 4} more" if len(hooks) > 4 else "")


def _users(agent: dict[str, Any], ver: dict[str, Any]) -> str | None:
    out = [f"{who} as `{sec['user']}`" for who, sec in (("agent", agent), ("grader", ver)) if sec.get("user") not in (None, "")]
    return " · ".join(out) or None


def _configuration(t: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The task's settings as Harbor reads them (Harbor's viewer: timeouts, resources, users, network per phase,
    healthcheck, MCP servers, skills, workdir, artifacts, collect hooks, steps), then task.toml itself."""
    env, agent, ver = cfg.get("environment") or {}, cfg.get("agent") or {}, cfg.get("verifier") or {}
    pkg, spec, e = cfg.get("task") or {}, t.get("spec") or {}, t["env"]
    venv = ver.get("environment") if isinstance(ver.get("environment"), dict) else None
    sep = t["verifier"].get("separate") or {}
    hc = env.get("healthcheck") if isinstance(env.get("healthcheck"), dict) else None
    steps = cfg.get("steps") or []
    mins = [(s.get("name"), _min_reward(s.get("min_reward"))) for s in steps if isinstance(s, dict) and s.get("min_reward") is not None]
    limits = [(w, _secs(v)) for w, v in (("agent", agent.get("timeout_sec")), ("grader", ver.get("timeout_sec")), ("build", env.get("build_timeout_sec")))]
    rows = [
        ("Name", " · ".join(x for x in (f"`{pkg['name']}`" if pkg.get("name") else "", f"version {pkg['version']}" if pkg.get("version") else "") if x) or None),
        ("Authors", ", ".join(str(a["name"]) for a in pkg.get("authors") or [] if isinstance(a, dict) and a.get("name")) or None),
        ("Schema", f"`{cfg['schema_version']}`" if cfg.get("schema_version") else None),
        ("Source", _fmt(cfg.get("source"), 200) if cfg.get("source") else None),
        ("Time limits", " · ".join(f"{w} {s}" for w, s in limits if s) or None),
        ("Resources", _resources(env)),
        ("Grader's container", " · ".join(x for x in (f"`{sep['image']}`" if sep.get("image") else "built from `tests/Dockerfile`" if sep.get("dockerfile") else "",
                                                       (_resources(venv) or "") if venv else "a copy of the task's environment") if x) if sep else None),
        ("Operating system", "Windows" if env.get("os") == "windows" else None),
        ("Users", _users(agent, ver)),
        ("Network", _network(e)),
        ("Allowed hosts", ", ".join(f"`{h}`" for h in (e.get("hosts") or [])[:20]) or None),
        ("Working directory", f"`{env['workdir']}`" if env.get("workdir") else None),
        ("Healthcheck", _healthcheck(hc)),
        *[("MCP server", _mcp(m)) for m in (env.get("mcp_servers") or [])[:8]],
        ("Skills", f"`{env['skills_dir']}`, copied into the agent's skills folder" if env.get("skills_dir") else None),
        ("Artifacts", _artifacts(cfg.get("artifacts"))),
        ("Collect hooks", (_collect(ver.get("collect")) or "") + ", after the agent and before the artifacts are collected" if _collect(ver.get("collect")) else None),
        ("Steps", f"{len(steps)}; the task's reward is {_strategy(cfg, len(steps))}" if steps else None),
        ("Min reward", "; ".join(f"`{n}`: {m}" for n, m in mins) or None),
        ("Prior context", "`trajectory.json`, seeded before the agent starts" if spec.get("trajectory") else None),
    ]
    err = spec.get("harbor_error")
    toml = t.get("toml") or ""
    return [
        c.note(f"Harbor {_harbor_version()} rejects this task.toml ({err}), so it can't run anywhere as it is. Read as written below.".replace("Harbor  ", "Harbor "), "alert") if err else None,
        c.kv(rows),
        c.disclose("Healthcheck command, in full", [c.code(hc["command"], "healthcheck.sh")]) if hc and len(str(hc["command"])) > 80 else None,
        c.disclose(f"task.toml · {len(toml.splitlines()):,} lines", [c.code(toml, "task.toml", "task.toml")]) if toml else None,
    ]


def _gist(s: dict[str, Any], limit: int = 140) -> str:
    """A step's instruction in a line: its first sentence of prose, cut at a word."""
    t = catalog._first_line(s.get("instruction") or s.get("inline") or "", 400)
    return t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _step_blocks(steps: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Each step of a multi-step task: its instruction word for word, its tests, what it stages and its settings."""
    task_mode = catalog._verifier_mode(cfg.get("verifier")) or "shared"
    out: list[dict[str, Any]] = []
    for i, s in enumerate(steps, 1):
        sc = s.get("config") or {}
        sa, sv = sc.get("agent") or {}, sc.get("verifier") or {}
        own = [x[6:] for x in s.get("tests") or []]
        work = s.get("workdir") or []
        text = s.get("instruction")
        missing = ("Its instruction is withheld: the step's name looks like an answer's." if s.get("withheld")
                   else "Its `instruction.md` wasn't read here: open it in Files." if s.get("found")
                   else "Its instruction is written in task.toml, where Harbor no longer reads it: Harbor looks for "
                        f"`steps/{s['name']}/instruction.md`, so this step can't run." if s.get("inline")
                   else f"It has no `steps/{s['name']}/instruction.md`, so Harbor can't run it.")
        nets = [f"{w}: {NET.get(sec['network_mode'], sec['network_mode'])}" for w, sec in (("agent", sa), ("grader", sv)) if sec.get("network_mode")]
        out += [
            c.kv([(f"Step {i}", f"`{s['name']}`" + (f" · {_secs(sa.get('timeout_sec'))} for the agent" if _secs(sa.get("timeout_sec")) else ""))]),
            c.markdown(text) if text is not None else c.note(missing, "alert"),
            c.markdown(s["inline"]) if text is None and s.get("inline") and not s.get("withheld") else None,
            c.kv([("Graded by", (", ".join(f"`tests/{x}`" for x in own[:10]) + (f" and {len(own) - 10} more" if len(own) > 10 else "")) if own
                   else "the task's `tests/`"),
                  ("Grader", "in a container of its own" if (catalog._verifier_mode(sv) or task_mode) == "separate" else None),
                  ("Grader time limit", _secs(sv.get("timeout_sec"))),
                  ("Staged first", (", ".join(f"`{x}`" for x in work[:8]) + (f" and {len(work) - 8} more" if len(work) > 8 else "")
                                    + (" (`setup.sh` runs before the agent)" if "setup.sh" in work else "")) if work else None),
                  ("Passes on", _min_reward(sc.get("min_reward"))),
                  ("Network", " · ".join(nets) or None),
                  ("Healthcheck", _healthcheck(sc.get("healthcheck"))),
                  ("Artifacts", _artifacts(sc.get("artifacts"))),
                  ("Reference solution", "yes, not shown" if s.get("solution") else None)]),
            *[c.disclose(f"steps/{s['name']}/{name} · {len(body.splitlines()):,} lines", [c.code(body, name)]) for name, body in (s.get("texts") or {}).items()],
        ]
    return out


def _metadata(t: dict[str, Any]) -> list[dict[str, Any]]:
    """`[metadata]` in full (nested tables too) and the dataset's own top-level tables (`[scoring]`), answers and
    contact details left out (catalog._clean)."""
    meta, tables = t.get("metadata") or {}, t.get("tables") or {}
    flat: list[tuple[str, str]] = []
    deep: dict[str, Any] = {}

    def add(name: str, v: Any) -> None:
        if _scalarish(v) or (isinstance(v, list) and all(_scalarish(x) for x in v)):
            flat.append((name, _fmt(v)))
        elif isinstance(v, dict) and len(v) <= 12 and all(_scalarish(x) or (isinstance(x, list) and all(_scalarish(y) for y in x)) for x in v.values()):
            flat.extend((f"{name}.{k}", _fmt(x)) for k, x in v.items())
        else:
            deep[name] = v

    for key, v in meta.items():
        if key not in ("tags", "keywords"):
            add(key, v)
    for key, v in tables.items():
        add(key, v)
    blocks = [c.kv(flat), c.kv([("Tags", " ".join(f"`{x}`" for x in t["tags"]))]) if t.get("tags") else None,
              c.value(deep) if deep else None,
              c.note("Withheld, as they hold the answer: " + ", ".join(f"`{x}`" for x in t["withheld"]) + ".", "shield") if t.get("withheld") else None]
    return [b for b in blocks if b and b.get("rows") != []]


def _setup(r: dict[str, Any]) -> list[str]:
    """What sets a task apart, for the Setup facet."""
    e, v = r["env"], r["verifier"]
    return [x for x, on in (("Multi-step", r.get("steps")), ("Separate grader", v.get("separate")), ("MCP servers", e.get("mcp")),
                            ("Model judge", v.get("judge")), ("GPU", e.get("gpus"))) if on]


def harbor_option(runnable: dict[str, Any], toml_text: str, *, label: str = "Harbor, on an HF Sandbox") -> dict[str, Any]:
    """The Harbor harness's run option for a task: from catalog.runnable() and the task's own variables (app/judge.py:
    a grader that calls a model gets a judge relay and a judge field; a variable nothing here can fill stops it)."""
    from .. import judge as judges

    p = judges.plan(toml_text)
    notes, warnings, fields = [], [], list(LIMITS)
    ok, why = bool(runnable.get("ok")), runnable.get("why") or ""
    if runnable.get("how") == "dockerfile":
        notes.append(f"No prebuilt image: the sandbox starts from `{runnable.get('base')}` and replays the task's Dockerfile first, so it takes longer to start.")
    if p["missing"] and ok:
        ok, why = False, f"it needs {', '.join(p['missing'])}, which rollouts here don't provide"
    j = p.get("judge")
    if j:
        pool = "vision_judges" if j["vision"] else "text_judges"
        fields.insert(0, c.field("judge", "Judge model", "model", pool=pool, required=True, default=j["requested"],
                                 help="the grader calls it, through a relay on your HF account"))
        notes.append("Its grader calls a model. Here it goes through a relay to the judge you pick on HF Inference Providers, billed to you; "
                     "the sandbox gets a key that only works for this rollout's grading."
                     + (f" The task asks for `{j['requested']}`: pick it to grade as the task's authors did." if j["requested"] else ""))
    if p["agent_keys"]:
        notes.append(f"{', '.join(f'`{k}`' for k in p['agent_keys'])}: left empty, so the agent gets no key of ours.")
    return c.run_option("harbor", label, ok=ok, why=why, default=True, notes=notes, warnings=warnings,
                        about=("A fresh HF Sandbox from the task's image, the agent you pick, then the task's own tests. The model is "
                               "called through a recording proxy, so the sandbox never holds a key."),
                        fields=fields, estimate={"tokens_in": 1_500_000, "tokens_out": 40_000, "minutes": 15, "flavor": "cpu-basic"})


LIMITS = [c.field("steps", "Model calls", "number", advanced=True, placeholder="no cap", min=1, max=1000, help="max"),
          c.field("timeout_min", "Time limit", "number", advanced=True, default=30, min=2, max=120, help="minutes")]


def _pick_tile(rows: list[dict[str, Any]]) -> str | None:
    for key in ("group", "difficulty", "category"):
        vals = {r.get(key) for r in rows if r.get(key)}
        if 2 <= len(vals) <= 6:
            return key
    return None

