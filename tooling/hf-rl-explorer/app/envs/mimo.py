"""Xiaomi's MiMo-V2.6 RL release (XiaomiMiMo/MiMo-V2.6-RL-oss), read the way the MiMo RL Environment Explorer reads it
(app/mimo): 7,780 tasks in five domains, each shown in full, and run two ways: as its Harbor conversion (the default,
FineEnvs/MiMo-V2.6-RL-harbor-*) or on the release's own harness (OpenCode on an HF Sandbox, Xiaomi's graders).

A task's ref is its id (`format-code-task-000720`, `s3k_…`, `music-12`); a row ref of the raw release
(`code/train/6`, what the generic rows reader used) opens the same task.
"""

from __future__ import annotations

import gzip
import json
import mimetypes
import re
from functools import lru_cache
from typing import Any

from . import contract as c

SPEC = "XiaomiMiMo/MiMo-V2.6-RL-oss"
ICON = {"code": "code", "webdev": "layout", "cyber": "bug", "music": "music", "general": "briefcase"}
LEFTOVER = re.compile(r"^(Other|Unknown|unknown|None named|Unspecified|Unrated)$")
MAP_TOP = {"code": 7, "webdev": 9, "cyber": 7, "music": 9, "general": 9}
ORDINAL = {"tier", "voices", "tempo", "meter"}
ROW_REF = re.compile(r"^(code|cyber|general|music|webdev)/train/(\d+)$")
# typical tokens in, out and minutes of a rollout, per domain (from MiMo explorer rollouts), for the cost shown
TYPICAL = {"code": (100e3, 3e3, 8), "cyber": (1.2e6, 30e3, 12), "general": (1.2e6, 20e3, 15), "webdev": (60e3, 20e3, 10), "music": (500, 6e3, 0)}
WITHHELD = {"general": ["pass_anchor", "gold_answer", "check_code"], "cyber": ["the proof of concept"], "code": ["the fix (patch)"]}
SETUP = {
    "code": "The repository is at the task's base commit with its git history truncated there (if an image's history isn't, .git is hidden while the agent works), and build leftovers that could leak the fix are cleaned up first.",
    "cyber": "The agent runs as an unprivileged user. It gets the project source, a prebuilt fuzz binary and submit.sh; the verifier runs as root.",
    "general": "The agent is an unprivileged user in the workspace. The systems' databases and code are root-only, so MCP is the only way in.",
    "webdev": "Node, pnpm, Playwright and Chromium are in the image; the agent builds however it likes and delivers to dist/.",
}


def _cat():
    from ..mimo import catalog

    return catalog


@lru_cache(maxsize=1)
def _domains() -> dict[str, dict[str, Any]]:
    return {d["id"]: d for d in _cat().index()["domains"]}


@lru_cache(maxsize=1)
def _order() -> dict[str, list[str]]:
    """Each domain's task ids in the release's row order (row n of `<domain>/train` is the n-th)."""
    out: dict[str, list[str]] = {}
    for tid, r in _cat().rows().items():
        out.setdefault(r["domain"], []).append(tid)
    return out


@lru_cache(maxsize=1)
def _position() -> dict[str, tuple[str, int]]:
    return {tid: (d, i) for d, ids in _order().items() for i, tid in enumerate(ids)}


def _vals(v: Any) -> list[str]:
    return [str(x) for x in (v if isinstance(v, list) else [v] if v not in (None, "") else [])]


class MiMoAdapter(c.Adapter):
    id, name, framework = "mimo", "MiMo-V2.6 RL", "MiMo"
    about = ("Xiaomi's MiMo-V2.6 RL environments as released, read the way the MiMo RL Environment Explorer reads them: "
             "five domains (Code, Webdev, Cyber, Music, General), each task with its brief, the agent's prompt, what it runs "
             "in and its grader in full. Runs as its Harbor conversion, or on the release's own harness.")
    kinds = ("dataset",)

    def detect(self, kind: str, meta: dict[str, Any]) -> float:
        return 1.0 if kind == "dataset" and meta.get("id") == SPEC else 0.0

    # ── the environment ──
    def summary(self, env: c.Env, subset: str | None = None) -> dict[str, Any]:
        idx = _cat().index()
        doms = idx["domains"]
        # with no domain picked: the domain and the brief's language; with one: that domain's own facets
        lang_scope = ["*"] + [d["name"] for d in doms if any(k == "language" for k, _ in d["facets"])]
        facets = [c.facet("domain", "Domain", scope=["*"]), c.facet("language", "Brief language", scope=lang_scope)]
        for d in doms:
            facets += [c.facet(f"{d['id']}.{key}", label, "level" if key in ORDINAL else "count", scope=[d["name"]])
                       for key, label in d["facets"] if key != "language"]
        tiles = c.tiles("domain", [{"value": d["name"], "note": f"Graded by {d['verifier']}", "icon": ICON[d["id"]], "color": d["id"],
                                    "title": d["task"]} for d in doms], all_label="All domains", all_note=f"{len(doms)} kinds of grader")
        return {"state": "ready", "total": idx["total"], "inline": True, "search": True, "subsets": [], "facets": facets, "tiles": tiles,
                "map": c.treemap(by_tile={d["name"]: f"{d['id']}.{d['main']}" for d in doms}, top={d["name"]: MAP_TOP[d["id"]] for d in doms}),
                "order": "shuffle", "noun": ["environment", "environments"],
                "how": {"framework": self.framework, "name": self.name, "about": self.about},
                "overview": [self._rewards(idx)]}

    def _rewards(self, idx: dict[str, Any]) -> dict[str, Any]:
        from ..mimo import config

        stats = json.loads(gzip.decompress((config.DATA_DIR / "rewards.json.gz").read_bytes()))
        kinds = {}
        for e in idx["envs"]:
            k = e["d"] if e["d"] != "general" else ("general-workplace" if e["id"].startswith("s3k_") else "general-terminal")
            kinds.setdefault(k, {"count": 0, "example": e["id"]})["count"] += 1
        text = ("Code, Cyber and General terminal tasks score 0 or 1 (hidden tests, a crash in the expected function, a test suite); "
                "General workplace tasks score the weighted share of rubric checks passed (code checks and a text judge); Webdev a "
                "vision judge's mean of visual quality, brief fulfilment and asset quality; Music 18 features of the piece against human music.")
        return c.section("rewards", "Reward design", [c.custom("mimo", "rewards", {"kinds": kinds, "stats": stats}, text)], icon="scale",
                         note="how each kind of task turns a rollout into a reward", collapsed=True)

    def tasks(self, env: c.Env, *, subset=None, q="", filters=None, offset=0, everything=False) -> dict[str, Any]:
        idx = _cat().index()
        doms = _domains()
        cards = []
        for e in idx["envs"]:
            d = doms[e["d"]]
            fac = {"domain": [d["name"]]}
            for key, v in e["f"].items():
                fac["language" if key == "language" else f"{e['d']}.{key}"] = _vals(v)
            main = _vals(e["f"].get(d["main"]))
            chips = [v for key, _ in d["facets"] if key not in (d["main"], "source") for v in _vals(e["f"].get(key)) if not LEFTOVER.match(v)][:4]
            n = e.get("n") or {}
            stats = [(n["systems"], "systems"), (n["files"], "files"), (n["checks"], "checks")] if n else []
            cards.append(c.card(e["id"], e["t"], brief=e["s"], lead=d["name"], sub=[", ".join(main)] if main else [], chips=chips, facets=fac,
                                text=" ".join(" ".join(_vals(v)) for v in e["f"].values()) + " " + d["name"] + " " + e["id"],
                                icon=ICON[e["d"]], color=e["d"], stats=stats))
        return {"cards": cards, "total": len(cards), "offset": 0, "page": len(cards)}

    def random(self, env: c.Env, subset=None) -> str:
        import random

        return random.choice(_cat().index()["envs"])["id"]

    # ── one task ──
    def _tid(self, ref: str) -> str:
        m = ROW_REF.match(ref.strip("/"))
        if m:   # a row of the raw release
            ids = _order().get(m.group(1)) or []
            i = int(m.group(2))
            if i >= len(ids):
                raise LookupError("no such row")
            return ids[i]
        if not _cat().record(ref):
            raise LookupError("no such task")
        return ref

    def _view(self, tid: str) -> dict[str, Any]:
        try:
            v = _cat().view(tid)
        except FileNotFoundError as e:
            raise LookupError(f"couldn't fetch this environment's files: {e}")
        if not v:
            raise LookupError("no such task")
        return v

    def task(self, env: c.Env, ref: str) -> dict[str, Any]:
        from ..mimo.runner.domains import run_defaults

        tid = self._tid(ref)
        v = self._view(tid)
        d = v["domain"]
        dom = _domains()[d]
        vf = v.get("verify") or {}
        prompt = v.get("prompt")
        spec = vf.get("spec") if vf.get("kind") == "music" else None
        meta = [(k, str(x)) for k, x in v.get("meta") or [] if x and len(str(x)) < 200
                and not re.match(r"^(none named|unspecified|unknown|n/a|none)$", str(x).strip(), re.I)]
        sections = [c.section("task", "The task", [
            c.stats([("Style", spec.get("style")), ("Tempo", spec.get("bpm") and f"{spec['bpm']} BPM"), ("Meter", spec.get("meter")),
                     ("Length", spec.get("bars") and f"{spec['bars']} bars"), ("Voices", spec.get("voices"))]) if spec else None,
            c.markdown(v.get("brief") or ""),
            c.kv(meta) if meta else None,
            c.links([(link["label"], link["url"]) for link in v.get("links") or []]) if v.get("links") else None,
        ], icon="message", note="word for word, inside the agent prompt below" if (prompt or {}).get("task_is_brief") else "")]
        if prompt and prompt.get("parts"):
            sections.append(c.section("prompt", "Agent prompt", [c.custom("mimo", "prompt", {**prompt, "systems": len(v.get("systems") or [])},
                                                                         "".join(t for _, t in prompt["parts"]))],
                                      icon="doc", note="the first message, exactly as sent"))
        if v.get("systems"):
            text = "\n".join(f"- {s['label']}: tools {', '.join(t['name'] for t in s['tools'])}; tables {', '.join(t['table'] for t in s['tables'])}" for s in v["systems"])
            sections.append(c.section("systems", "Systems the agent works through", [c.custom("mimo", "systems", v["systems"], text)],
                                      icon="plug", note="MCP servers, each backed by its own database"))
        if v.get("files"):
            kinds: dict[str, int] = {}
            for f in v["files"]:
                kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
            sections.append(c.section("workspace", "Workspace files", [c.custom("mimo", "workspace", v["files"], "\n".join(f"- {f['path']}" for f in v["files"]))],
                                      icon="folder", note=" · ".join(f"{n} {k}" for k, n in kinds.items())))
        if d == "code":
            sections.append(self._repo(tid, v))
        setup = self._setup(v)
        if setup:
            sections.append(setup)
        if vf:
            text = _grading_text(vf)
            sections.append(c.section("grading", "How it's graded", [c.custom("mimo", "grading", {**vf, "domain": d}, text)], icon="scale",
                                      note=f"needs a {vf['needs_judge']} judge model" if vf.get("needs_judge") else "no model in the loop"))
        dom_i = _position().get(tid)
        ids = _order().get(dom_i[0]) if dom_i else None
        nav = {"prev": ids[dom_i[1] - 1] if dom_i and dom_i[1] > 0 else None, "next": ids[dom_i[1] + 1] if dom_i and dom_i[1] + 1 < len(ids) else None,
               "index": dom_i[1], "total": len(ids), "subset": None, "label": dom["name"]} if dom_i else None
        twin = self._twin(tid, d)
        links = [c.link(f"In Harbor's format: {twin['dataset']}", f"/t/{twin['dataset']}/{twin['path']}",
                        "the same task, converted; its rollouts are this task's too", rel="same")] if twin else []
        chips = [x for vals in (v.get("facets") or {}).values() for x in _vals(vals) if not LEFTOVER.match(x)][:6]
        return {"ref": tid, "title": v["title"], "id": tid, "chips": chips, "sections": sections, "glance": self._glance(v),
                "withheld": WITHHELD.get(d, []) if d != "general" or vf.get("kind") == "rubric" else [], "links": links,
                "summary": f"graded by {dom['verifier']}", "icon": ICON[d], "color": d, "framework": f"MiMo · {dom['name']}",
                "hub": f"https://huggingface.co/datasets/{SPEC}/viewer/{d}/train?row={dom_i[1]}" if dom_i else f"https://huggingface.co/datasets/{SPEC}",
                "nav": nav, "_v": v, "_twin": twin, "_defaults": run_defaults(tid, d)}

    def _twin(self, tid: str, domain: str) -> dict[str, Any] | None:
        from .processors import twin

        try:
            return twin(domain, tid)
        except Exception:  # noqa: BLE001 - the twin's index isn't here: no link
            return None

    def _repo(self, tid: str, v: dict[str, Any]) -> dict[str, Any]:
        snap = _snapshot(tid)
        if not snap:
            return c.section("repo", "The repository", [c.note(f"This repository hasn't been indexed yet. It lives inside the task image "
                                                               f"(`{(v.get('environment') or {}).get('image') or ''}`); snapshots are added in batches.", "folder")],
                             icon="folder", note="as the agent finds it, at the task's base commit")
        files = [{"path": f["path"], "size": f["size"], **({} if f.get("preview") else {"note": "no preview: binary or larger than 200 KB"})} for f in snap["files"]]
        return c.section("repo", "The repository", [
            c.stats([("Files", f"{len(files):,}"), ("Base commit", f"`{(snap.get('base') or '')[:10]}`"),
                     ("Upstream", re.sub(r"^https?://|\.git$", "", snap["remote"]) if snap.get("remote") else None),
                     ("History", "truncated at base" if snap.get("history_truncated") else "not truncated, hidden from the agent")]),
            c.files(files),
            c.note(f"`{snap.get('cwd') or ''}` in the task image, as the agent finds it. Text files up to 200 KB open here; the hidden tests only arrive at grading.", "info"),
        ], icon="folder", note="as the agent finds it, at the task's base commit")

    def _setup(self, v: dict[str, Any]) -> dict[str, Any] | None:
        e, d = v.get("environment") or {}, v["domain"]
        rows = []
        if e.get("sandbox") is False:
            rows.append(("Where it runs", "No sandbox: one model call, then the scorer"))
        if e.get("image"):
            rows.append(("Sandbox image", f"`{e['image'].replace('docker.io/', '')}`"))
        if e.get("cwd"):
            rows.append(("Working directory", f"`{e['cwd']}`"))
        if e.get("deliver"):
            rows.append(("Deliverable", f"`{e['deliver']}/index.html` (and its assets)"))
        if e.get("ports"):
            rows.append(("MCP ports", ", ".join(str(p) for p in e["ports"])))
        if e.get("cpus"):
            rows.append(("Resources", f"{e['cpus']} CPU · {e.get('memory_mb')} MB · internet {'on' if e.get('internet') else 'off'}"))
        if e.get("tags"):
            rows.append(("Tags", ", ".join(e["tags"])))
        how = SETUP.get(d, "")
        if not rows and not how:
            return None
        net = ("Web search and fetch tools are off." if d == "webdev" else "Web search and fetch tools are off, and the sites where answers live "
               "(code hosting, bug trackers, search engines) are unreachable from the sandbox; everything else, including local services, works as usual.")
        return c.section("setup", "How the sandbox is set up", [c.kv(rows), c.note(f"{how} {net}", "shield") if how else None], icon="box")

    def _glance(self, v: dict[str, Any]) -> list[list[str]]:
        vf, d = v.get("verify") or {}, _domains()[v["domain"]]
        rows = [["Domain", d["name"]], ["Graded by", d["verifier"]]]
        if v.get("systems"):
            rows += [["Systems", str(len(v["systems"]))], ["Tools", str(sum(len(s["tools"]) for s in v["systems"]))], ["Workspace files", str(len(v.get("files") or []))]]
        if vf.get("kind") == "rubric":
            rows.append(["Rubric checks", str(len(vf.get("checks") or []))])
        if vf.get("kind") in ("tests", "terminal"):
            rows.append(["Hidden test files", str(len(vf.get("files") or []))])
        if vf.get("kind") == "crash" and (vf.get("expected") or {}).get("function"):
            rows.append(["Must crash in", f"`{vf['expected']['function']}`"])
        if vf.get("kind") == "visual":
            rows.append(["Graded from", "a full-page screenshot"])
        if vf.get("kind") == "music":
            rows.append(["Scored by", "code, no model"])
        if vf.get("needs_judge"):
            rows.append(["Judge", f"a {vf['needs_judge']} model you pick"])
        return rows

    # ── its files and data ──
    def file(self, env: c.Env, ref: str, path: str) -> dict[str, Any]:
        tid = self._tid(ref)
        snap = _snapshot(tid)
        if not snap:
            return {"path": path, "error": "this task has no files here"}
        meta = next((f for f in snap["files"] if f["path"] == path), None)
        if meta is None:
            raise LookupError("no such file")
        text = (snap.get("texts") or {}).get(path)
        if text is None:
            return {"path": path, "size": meta["size"], "error": "No preview: this file is binary or larger than 200 KB."}
        return {"path": path, "size": meta["size"], "text": text}

    def data(self, env: c.Env, ref: str, name: str, params: dict[str, str]) -> Any:
        cat = _cat()
        tid = self._tid(ref)
        if name == "table":   # a system's database table, as the agent finds it
            try:
                return cat.db_rows(cat.system_db(tid, params.get("system", "")), params.get("table", ""), min(int(params.get("limit") or 100), 500))
            except (PermissionError, KeyError, ValueError):
                raise LookupError("no such table")
        if name == "preview":   # a workspace file, as text and tables
            from ..mimo import previews

            p = self._workspace(tid, params.get("path", ""))
            return previews.preview(p)
        raise LookupError(f"nothing named {name!r} here")

    def _workspace(self, tid: str, rel: str):
        try:
            p = _cat().workspace_path(tid, rel)
        except (PermissionError, TypeError):
            raise LookupError("outside the workspace")
        if not p.is_file():
            raise LookupError("no such file")
        return p

    def raw(self, env: c.Env, ref: str, path: str) -> tuple[bytes, str]:
        p = self._workspace(self._tid(ref), path)
        return p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream"

    # ── running ──
    def run_options(self, env: c.Env, ref: str, view: dict[str, Any]) -> list[dict[str, Any]]:
        from .harbor import harbor_option

        v, twin, d = view["_v"], view.get("_twin"), view["_v"]["domain"]
        opts = []
        if twin:   # Harbor first: the default way to run any task here
            try:
                from .. import catalog

                t = catalog.task(twin["dataset"], twin["path"])
                opt = harbor_option(t["runnable"], t.get("toml") or "", label="Harbor, on an HF Sandbox")
                opt["notes"] = [f"Runs `{twin['dataset']}` · `{twin['path']}`: this task in Harbor's format."] + opt["notes"]
                if not opt["ok"]:   # it can't run as Harbor here: the release's own harness is the default
                    opt["default"] = False
                opts.append(opt)
            except Exception:  # noqa: BLE001 - its twin isn't readable now: the native harness still is
                pass
        opts.append(self._native(v, view["_defaults"], default=not any(o["default"] and o["ok"] for o in opts)))
        return opts

    def _native(self, v: dict[str, Any], defaults: dict[str, Any], default: bool) -> dict[str, Any]:
        d = v["domain"]
        music = d == "music"
        need = (v.get("verify") or {}).get("needs_judge")
        ti, to, minutes = TYPICAL[d]
        fields = []
        if need:
            fields.append(c.field("judge", "Judge model", "model", pool="vision_judges" if need == "vision" else "text_judges", required=True,
                                  help="scores a screenshot of the page" if need == "vision" else "answers each rubric check"))
        fields += [
            c.field("thinking", "Thinking", "select", default="low", advanced=True,
                    options=[{"value": k, "label": l} for k, l in (("default", "Model default"), ("none", "Off"), ("low", "Low"), ("medium", "Medium"), ("high", "High"))],
                    help="sent as reasoning_effort"),
            c.field("temperature", "Temperature", "number", advanced=True, placeholder="default", min=0, max=2, step=0.1),
            *([] if music else [c.field("steps", "Step cap", "number", advanced=True, placeholder=str(defaults.get("steps") or ""), min=1, max=1000),
                                c.field("timeout_min", "Time limit", "number", advanced=True, placeholder=str(defaults.get("timeout_min") or ""), min=2, max=120, help="minutes")]),
            c.field("max_tokens", "Max tokens" + ("" if music else " / step"), "number", advanced=True, placeholder=str(defaults.get("max_tokens") or "default"),
                    min=256, max=128000, step=256),
        ]
        about = ("One model call, then Xiaomi's scorer. No sandbox." if music
                 else "The release's own harness: a fresh HF Sandbox from this task's image, OpenCode as the agent, then Xiaomi's grader, as the MiMo explorer runs it.")
        notes = [] if music else ["The step cap and time limit default to the values Xiaomi's training harness uses for this domain."]
        if need == "vision":
            notes.append("Judges disagree (0.34–0.87 on the same page), so compare webdev scores only between runs graded by the same judge.")
        return c.run_option("mimo", "MiMo harness" if not music else "MiMo scorer", ok=bool(v.get("runnable")), why="this task can't be run here yet",
                            default=default, about=about, notes=notes, fields=fields, harnesses=["opencode"], sandbox=not music,
                            estimate={"tokens_in": ti, "tokens_out": to, "minutes": minutes, "flavor": "cpu-upgrade" if d == "webdev" else "cpu-basic"})

    def materialize(self, env: c.Env, ref: str):
        from .. import catalog

        twin = self._twin(self._tid(ref), self._view(self._tid(ref))["domain"])
        if not twin:
            raise LookupError("this task has no Harbor form here")
        return catalog.task_dir(twin["dataset"], twin["path"])

    def run_task(self, env: c.Env, ref: str) -> dict[str, Any]:
        from .. import catalog

        tid = self._tid(ref)
        v = self._view(tid)
        out = {"title": v["short_title"], "restricted": False, "sha": env.meta.get("sha"), "mimo": {"id": tid, "domain": v["domain"], "title": v["short_title"],
               "facets": v.get("facets"), "needs_judge": (v.get("verify") or {}).get("needs_judge")}, "domain": v["domain"], "ref": tid}
        twin = self._twin(tid, v["domain"])
        if twin:
            try:
                t = catalog.task(twin["dataset"], twin["path"])
                out.update(bytes=t.get("bytes") or 0, runnable=t["runnable"], image=t["env"]["image"] or t["runnable"].get("base"))
            except Exception:  # noqa: BLE001
                pass
        return out

    def aliases(self, env: c.Env, ref: str) -> list[tuple[str, str]]:
        try:
            tid = self._tid(ref)
            twin = self._twin(tid, _cat().record(tid)["d"])
        except Exception:  # noqa: BLE001
            return []
        out = [(SPEC, tid)] + ([(SPEC, row_ref(tid))] if row_ref(tid) else [])   # and rollouts recorded against its row
        return out + ([(twin["dataset"], twin["path"])] if twin else [])


def _grading_text(vf: dict[str, Any]) -> str:
    """The grader, as text for agents (MCP): what the page's grading view shows, answers left out as there."""
    out = [vf.get("summary") or "", *[f"- {x}" for x in vf.get("steps") or []], vf.get("formula") or ""]
    k = vf.get("kind")
    if k in ("tests", "terminal") and vf.get("files"):
        out.append("Hidden test files: " + ", ".join(f["path"] for f in vf["files"]))
    if k == "crash" and vf.get("expected"):
        x = vf["expected"]
        out.append(f"Must crash in `{x.get('function')}` ({x.get('file')}), {x.get('sanitizer')} {x.get('error_type')}")
    if k == "rubric":
        wt = [1 if ch.get("weight") is None else float(ch["weight"]) for ch in vf.get("checks") or []]
        tot = sum(wt) or 1
        out.append("Checks (the answer each expects is withheld):")
        out += [f"{i}. [{ch.get('tier') or ''}, {'judged by a model' if ch.get('method') == 'llm' else 'checked by code'}, {round(100 * w / tot)}% of the reward] "
                f"{ch.get('question') or ch.get('id')}" for i, (ch, w) in enumerate(zip(vf.get("checks") or [], wt), 1)]
    if k == "visual":
        out += [f"- {d['label']}: {d['desc']}" for d in vf.get("dims") or []]
    if k == "music":
        out += [f"- {f.get('label') or f['name']} ({f['group']}): {f['rule']}" + (f", human range {f['band'][0]}–{f['band'][1]}" if f.get("band") else "")
                for f in vf.get("features") or []]
    if vf.get("not_scored"):
        out.append(f"Not scored: {vf['not_scored']}")
    return "\n".join(x for x in out if x)


def row_ref(tid: str) -> str | None:
    pos = _position().get(tid)
    return f"{pos[0]}/train/{pos[1]}" if pos else None


def _snapshot(tid: str) -> dict[str, Any] | None:
    """A Code task's repository, as snapshotted from its image (the MiMo explorer's snapshot_repos.py)."""
    from ..mimo import config

    if "/" in tid or ".." in tid:
        return None
    p = config.SNAPSHOT_DIR / f"{tid}.json.gz"
    if not p.is_file():
        return None
    return _read_snapshot(str(p), p.stat().st_mtime)


@lru_cache(maxsize=16)
def _read_snapshot(path: str, mtime: float) -> dict[str, Any]:
    return json.loads(gzip.decompress(open(path, "rb").read()))


# ── the Harbor conversion, back to the release ───────────────────────────────
TWIN_DATASET = re.compile(r"^FineEnvs/MiMo-V2\.6-RL-harbor-(code|cyber|general|terminal|webdev|music)$")


def harbor_to_mimo(env: c.Env, ref: str, view: dict[str, Any]) -> list[dict[str, Any]]:
    """A linker: a task of one of FineEnvs' Harbor conversions links to the same task in the release."""
    if not TWIN_DATASET.match(env.id) or not ref.startswith("tasks/"):
        return []
    tid = _by_slug().get(ref[6:])
    if not tid:
        return []
    return [c.link(f"In the release: {SPEC}", f"/t/{SPEC}/{tid}", "the task as Xiaomi released it, with its own harness; its rollouts are this task's too", rel="same")]


@lru_cache(maxsize=1)
def _by_slug() -> dict[str, str]:
    from .processors import _slug

    return {_slug(e["id"]): e["id"] for e in _cat().index()["envs"]}


def harbor_aliases(env: c.Env, ref: str) -> list[tuple[str, str]]:
    if not TWIN_DATASET.match(env.id) or not ref.startswith("tasks/"):
        return []
    tid = _by_slug().get(ref[6:])
    return [(SPEC, tid)] + ([(SPEC, row_ref(tid))] if row_ref(tid) else []) if tid else []
