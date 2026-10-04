"""Data for Verifiers environments (PrimeIntellect-ai/verifiers; Prime Intellect's Environments Hub), read as the
environments that load it read it.

A dataset tagged `verifiers` holds rows an environment turns into tasks. There are two APIs (verifiers 0.3.1,
2026-08-24, the latest release; its `main` drops v0 entirely):

- **v0** (`load_environment()` → SingleTurnEnv / MultiTurnEnv / ToolEnv / ..., a Rubric of reward functions): rows are
  `prompt` (chat messages) or `question` (turned into one, with the environment's system prompt and few-shot examples),
  `answer` (the ground truth, passed to the reward functions: withheld), `info` (per-row data the rewards also get),
  and, before 0.1.14, `task` (EnvGroup's routing key). Run with `vf-eval <env>` (0.3.1: the legacy runner).
- **v1** (Taskset / Task / TaskData + Harness + Runtime): each taskset's `load()` maps the columns into its own TaskData,
  so column meanings are the taskset's. Prime's SWE tasksets (research-environments) read a problem statement, a Docker
  image and hidden tests; their gold patches and test lists are withheld here (SWE_HIDE). Run with `eval <taskset>`
  (0.3.1; `vf-eval` on main).

Which Hub environment loads which dataset: KNOWN (Prime's own datasets, from research-environments d4fb80a), else the
dataset card's own mention (an Environments Hub link or `prime env install owner/name`). The Hub says which API an
environment uses (`runtime`); asked once, cached, never waited on for long.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from . import contract as c
from . import convo
from .base import Dataset, Processor, _common_prefix, parse_json, section, withhold

VF = "https://github.com/PrimeIntellect-ai/verifiers"
VF_DOCS = "https://docs.primeintellect.ai/verifiers/overview"
HUB = "https://app.primeintellect.ai/dashboard/environments"
RE_SWE = "https://github.com/PrimeIntellect-ai/research-environments/tree/main/environments/swe"
PRIME_RL = "https://github.com/PrimeIntellect-ai/prime-rl"

# Prime's datasets and the Hub environment that loads each (research-environments README, the tasksets' DATASET)
KNOWN: dict[str, dict[str, Any]] = {
    "primeintellect/reverse-text-rl": {"env": "primeintellect/reverse-text", "kind": "text", "runtime": "v0",
                                       "reward": "LCS ratio (difflib) between the reply's `<reversed_text>` and the prompt reversed (computed, not stored)",
                                       "system": "Reverse the text character-by-character. Put your answer in <reversed_text> tags."},
    "primeintellect/r2e-gym-subset-verified": {"env": "primeintellect/r2e-gym", "kind": "swe", "src": "r2e_gym",
                                               "reward": "the hidden R2E tests, run in the task's container after the agent: each test's result must match `expected_output_json` (withheld)"},
    "primeintellect/multi-swe-rl-verified": {"env": "primeintellect/multiswe", "kind": "swe", "src": "multiswe"},
    "primeintellect/multi-swe-rl": {"env": "primeintellect/multiswe", "kind": "swe", "src": "multiswe"},
    "primeintellect/multi-swe-bench": {"env": "primeintellect/multiswe", "kind": "swe", "src": "multiswe"},
    "primeintellect/scale-swe-verified": {"env": "primeintellect/scaleswe", "kind": "swe", "src": "scaleswe"},
    "primeintellect/swe-lego-real-data-verified": {"env": "primeintellect/swelego", "kind": "swe", "src": "swelego"},
    "primeintellect/swe-lego-real-data": {"env": "primeintellect/swelego", "kind": "swe", "src": "swelego"},
    "primeintellect/swe-rebench-v2-filtered-verified": {"env": "primeintellect/swerebench-v2", "kind": "swe", "src": "swerebench_v2"},
    "primeintellect/swe-rebench-v2-filtered-easy-verified": {"env": "primeintellect/swerebench-v2", "kind": "swe", "src": "swerebench_v2"},
}
# each SWE taskset's own default (dataset, split): anything else is set in the eval config
DEFAULTS = {"primeintellect/r2e-gym": ("PrimeIntellect/R2E-Gym-Subset-Verified", "train"), "primeintellect/multiswe": ("PrimeIntellect/Multi-SWE-RL-Verified", "train"),
            "primeintellect/scaleswe": ("PrimeIntellect/Scale-SWE-Verified", "train"), "primeintellect/swelego": ("PrimeIntellect/SWE-Lego-Real-Data-Verified", "resolved"),
            "primeintellect/swerebench-v2": ("PrimeIntellect/SWE-rebench-V2-Filtered-Verified", "train")}
SWE_REWARD = "the hidden tests, run in the task's container after the agent: the failing tests must pass and the passing ones still pass (1, else 0)"
# what SWE rows hold that the agent must not see (research-environments' grading-material table): gold patches, the
# tests and their expected results, the files a fix touches, and a solved trajectory
SWE_HIDE = frozenset({
    "patch", "test_patch", "fix_patch", "gold_patch", "f2p_patch", "f2p_script", "FAIL_TO_PASS", "PASS_TO_PASS", "FAIL_TO_FAIL", "PASS_TO_FAIL",
    "f2p", "p2p", "s2p", "n2p", "fixed_tests", "run_result", "test_patch_result", "fix_patch_result", "expected_output_json",
    "parsed_commit_content", "execution_result_content", "modified_files", "modified_entity_summaries", "relevant_files", "pr_commit",
    "eval_cmd", "eval_script", "resolved", "meta.llm_metadata", "pr_description", "hints_text", "hints",
    "p2p_tests", "f2p_tests", "s2p_tests", "n2p_tests",
})
# community datasets whose answers have names no rule could know: the hidden rule (code and its description) of a
# rule-induction game
DATASET_HIDE = {"nph4rd/eleusis-simple-rules": frozenset({"code", "label", "rule_id"}), "nph4rd/eleusis-scaled": frozenset({"code", "label", "rule_id"})}
_GRADING_COLS = ("patch", "fix_patch", "test_patch", "FAIL_TO_PASS", "f2p_patch", "parsed_commit_content")
IMAGE_COLS = ("docker_image", "image_name", "image_url", "image")
_ENV_LINK = re.compile(r"app\.primeintellect\.ai/dashboard/environments/([\w.-]+)/([\w.-]+)")
_ENV_CMD = re.compile(r"prime env install\s+([\w.-]+/[\w.-]+)")


def is_swe(cols: list[str]) -> bool:
    """An issue to resolve, and the material that grades it."""
    return any(x in cols for x in ("problem_statement", "resolved_issues")) and any(x in cols for x in _GRADING_COLS)


def hide_for(cols: list[str], spec: str = "") -> frozenset[str]:
    """SWE rows' grading material. In them `messages` is a solved trajectory, R2E's `prompt` the commit's own
    issue-writing prompt, and Multi-SWE-bench's `title`/`body` the fixing pull request's: all withheld."""
    own = DATASET_HIDE.get(spec.lower(), frozenset())
    if not is_swe(cols):
        return own
    extra = {"messages"} if "messages" in cols else set()
    if "parsed_commit_content" in cols and "prompt" in cols:
        extra.add("prompt")
    if "resolved_issues" in cols and "fix_patch" in cols:
        extra |= {"title", "body"}
    return SWE_HIDE | own | extra


def issue_text(row: dict[str, Any]) -> str:
    """What a SWE row's agent is asked: its problem statement, or the issues the fix resolved (Multi-SWE-bench)."""
    ps = row.get("problem_statement")
    if isinstance(ps, str) and ps.strip():
        return ps
    ri = parse_json(row.get("resolved_issues"))
    if isinstance(ri, dict) and isinstance(ri.get("title"), list):   # columnar: {title: [...], body: [...]}
        ri = [{"title": t, "body": (ri.get("body") or [""] * len(ri["title"]))[j] if j < len(ri.get("body") or []) else ""} for j, t in enumerate(ri["title"])]
    if isinstance(ri, list):
        return "\n\n".join(f"# {x.get('title') or ''}\n\n{x.get('body') or ''}".strip() for x in ri if isinstance(x, dict))
    return ""


_seen: dict[tuple, tuple[float, Any]] = {}
_seen_lock = threading.Lock()


def _remember(key: tuple, fn, ttl: float = 6 * 3600, retry: float = 600) -> Any:
    """`fn()` kept `ttl` seconds; a failure (None) is kept `retry` seconds, so a slow or down service costs one wait,
    not one per page."""
    now = time.time()
    with _seen_lock:
        hit = _seen.get(key)
        if hit and hit[0] > now:
            return hit[1]
    try:
        value = fn()
    except Exception:  # noqa: BLE001 - a service that can't say quickly says nothing
        value = None
    with _seen_lock:
        _seen[key] = (now + (ttl if value is not None else retry), value)
        if len(_seen) > 4000:
            for k in sorted(_seen, key=lambda k: _seen[k][0])[:1000]:
                _seen.pop(k, None)
    return value


def _hub(env_id: str) -> dict[str, Any]:
    """What the Environments Hub says about an environment (runtime, description); {} when it can't say quickly."""
    def fetch():
        import httpx

        r = httpx.get(f"https://api.primeintellect.ai/api/v1/environmentshub/{env_id}/@latest", timeout=httpx.Timeout(4, connect=3),
                      headers={"User-Agent": "hf-rl-explorer"})
        if r.status_code != 200:
            return None
        d = r.json()
        d = d.get("data", d) if isinstance(d, dict) else {}
        return {k: d.get(k) for k in ("runtime", "description", "version", "name") if isinstance(d.get(k), str)}

    return _remember(("prime-hub", env_id), fetch) or {}


def _card_env(spec: str, meta: dict[str, Any]) -> str | None:
    """The environment a dataset's card names: an Environments Hub link, or a `prime env install owner/name`."""
    if meta.get("restricted") or not meta.get("sha"):
        return None

    def fetch():
        import httpx

        r = httpx.get(f"https://huggingface.co/datasets/{spec}/resolve/{meta['sha']}/README.md", timeout=httpx.Timeout(4, connect=3),
                      headers={"Range": "bytes=0-262143"}, follow_redirects=True)
        if r.status_code not in (200, 206):
            return "" if r.status_code == 404 else None
        m = _ENV_LINK.search(r.text)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        m = _ENV_CMD.search(r.text)
        return m.group(1) if m else ""

    return _remember(("vf-card-env", spec, meta["sha"]), fetch) or None


def env_of(ds: Dataset) -> dict[str, Any]:
    """{env, kind, reward?, system?, src?, runtime?, about?, from}: which Hub environment loads these rows."""
    k = KNOWN.get(ds.spec.lower())
    out: dict[str, Any] = dict(k, source="known") if k else {}
    if not out:
        env = _card_env(ds.spec, ds.card)
        if env:
            out = {"env": env, "kind": "swe" if is_swe(ds.columns) else "text", "source": "card"}
    if not out:
        out = {"env": None, "kind": "swe" if is_swe(ds.columns) else "text", "source": None}
    if out.get("env"):
        h = _hub(out["env"])
        # the Hub's word, else what's known (Prime's SWE tasksets are v1, reverse-text v0, at research-environments d4fb80a)
        out["runtime"] = {"VERIFIERS_V0": "v0", "VERIFIERS_V1": "v1"}.get(h.get("runtime", ""), None) or out.get("runtime") \
            or ("v1" if out.get("kind") == "swe" and out.get("source") == "known" else None)
        out["about"] = h.get("description")
    return out


def _prompt_turns(row: dict[str, Any], env: dict[str, Any]) -> list[dict[str, Any]]:
    p = parse_json(row.get("prompt"))
    if convo.is_conversation(p):
        ts = convo.turns(p)
    elif isinstance(p, str) and p.strip():
        ts = [{"kind": "message", "role": "user", "text": p}]
    elif isinstance(row.get("question"), str):
        ts = [{"kind": "message", "role": "user", "text": row["question"]}]
    else:
        ts = []
    if env.get("system") and not any(t["role"] == "system" for t in ts):
        ts.insert(0, {"kind": "message", "role": "system", "text": env["system"]})
    return ts


def _ident(row: dict[str, Any], roles: dict[str, Any]) -> str | None:
    """A row's id: a column named like one (never one that holds the answer)."""
    for k in ("instance_id", "id", "task_id", "example_id", *[x for x in row if x.endswith("_id")]):
        if isinstance(row.get(k), (str, int)) and str(row[k]) and k not in roles.get("answer", []):
            return str(row[k])
    return None


class Verifiers(Processor):
    id, name, framework = "verifiers", "Verifiers", "Verifiers"
    about = ("Data for a Verifiers environment (Prime Intellect's Environments Hub): each row becomes a task when the "
             "environment loads it. Its prompt or question is what the model gets; its answer (and, for SWE tasks, the gold "
             "patch and hidden tests) is what the environment's rewards check, withheld here.")

    def match(self, ds: Dataset) -> float:
        tagged = any(t in ds.tags for t in ("library:verifiers", "verifiers"))
        if ds.spec.lower() in KNOWN:
            return 0.96
        return 0.55 if tagged else 0

    def roles(self, ds: Dataset) -> dict[str, Any]:
        r = super().roles(ds)
        cols = ds.columns
        hide = hide_for(cols, ds.spec)
        swe = is_swe(cols)
        r["task"] = ("problem_statement" if "problem_statement" in cols else "resolved_issues") if swe else "prompt" if "prompt" in cols else "question" if "question" in cols else r["task"]
        r["answer"] = sorted({x for x in cols if x in hide} | {h for h in hide if "." in h and h.split(".")[0] in cols} | set(r["answer"]))
        r["environment"] = [x for x in cols if x in IMAGE_COLS or x in ("repo", "base_commit", "workdir", "test_cmd", "install_config", "language", "repo_language")]
        r["grading"] = [x for x in cols if x in ("info", "task") and x not in r["answer"]]
        texts = [convo.first_user(_prompt_turns(x, {})) for x in ds.sample[:20]] if not swe else []
        r["_prefix"] = _common_prefix(texts) if len([t for t in texts if t]) >= 3 else ""
        return r

    def _title(self, row: dict[str, Any], ts: list[dict[str, Any]], roles: dict[str, Any], i: int) -> tuple[str, str]:
        text = issue_text(row) if is_swe(list(row)) else convo.first_user(ts)
        p = roles.get("_prefix") or ""
        own = text[len(p):].lstrip() if p and text.startswith(p) else text
        title = convo.task_title(own) or (_ident(row, roles) or f"Row {i}")
        return title, convo.after_title(own, title)

    def card(self, row: dict[str, Any], i: int, roles: dict[str, Any]) -> dict[str, Any]:
        ts = _prompt_turns(row, {})
        title, brief = self._title(row, ts, roles, i)
        ident = _ident(row, roles)
        chips = [str(row[k]) for k in ("repo", "language", "repo_language", "task", "difficulty", "category") if isinstance(row.get(k), str) and 0 < len(row[k]) <= 40][:3]
        return {"i": i, "id": ident[:60] if ident else None, "title": title[:200], "snippet": brief[:260], "chips": chips}

    def view(self, row: dict[str, Any], i: int, ds: Dataset, roles: dict[str, Any]) -> dict[str, Any]:
        cols = list(row)
        env = env_of(ds)
        swe = is_swe(cols)
        hide = hide_for(cols, ds.spec)
        clean, gone = withhold({k: parse_json(v) for k, v in row.items()}, hide=hide,
                               keep=tuple(x for x in ("instance_id", "task_id") if x in row))
        title, _ = self._title(row, _prompt_turns(row, {}), roles, i)
        config, split = roles.get("_config", "default"), roles.get("_split", "train")
        sections = []
        if swe:
            ps = issue_text(row)
            sections.append(section("task", "The task", [
                c.note("The agent works in the repository inside the task's container and must resolve this issue; it never sees the fix or the tests.", "info"),
                c.markdown(ps) if ps.strip() else c.note("This row has no problem statement.", "alert")], kind="blocks",
                note="problem_statement, the issue the agent gets" if row.get("problem_statement") else "resolved_issues, the issues the agent gets"))
            where = [(k, f"`{clean[k]}`") for k in (*IMAGE_COLS, "repo", "base_commit", "parent_commit", "workdir", "language", "repo_language", "version")
                     if isinstance(clean.get(k), (str, int)) and str(clean.get(k))]
            ic = clean.get("install_config") if isinstance(clean.get("install_config"), dict) else {}
            tc = clean.get("test_cmd") or ic.get("test_cmd")
            if tc:
                where.append(("Test command", f"`{tc if isinstance(tc, str) else ' '.join(map(str, tc))}`"))
            if where:
                sections.append(section("environment", "What it runs in", [c.kv(where)], kind="blocks", note="the task's container"))
        else:
            ts = _prompt_turns(row, env)
            note = "" if env.get("system") else ("The environment may add its own system prompt and few-shot examples before this." if "question" in row and "prompt" not in row else "")
            nxt = {"label": "Model", "text": "Replies; the environment's rewards score the reply" + (" against the withheld answer." if "answer" in row or env.get("reward") else ".")}
            sections.append(section("task", "The task", [c.custom("rl", "transcript", {"turns": ts, "next": nxt, "note": note}, convo.transcript_text(ts))]
                                    if ts else [c.note("This row has no prompt or question.", "alert")], kind="blocks",
                                    note="prompt, as the model gets it" if "prompt" in row else "question" if "question" in row else ""))
            tools = convo.tools((parse_json(row.get("info")) or {}).get("tool_defs")) if isinstance(parse_json(row.get("info")), dict) else []
            if tools:
                sections.append(section("tools", f"Tools ({len(tools)})", [c.custom("rl", "tools", {"tools": tools}, convo.tools_text(tools))], kind="blocks",
                                        note="info.tool_defs"))

        g: list[tuple[str, Any]] = []
        if env.get("env"):
            g.append(("Environment", f"[{env['env']}]({HUB}/{env['env']})" + (f": {env['about']}" if env.get("about") else "")
                      + ("" if env.get("source") == "known" else " (named by the dataset's card)")))
            if env.get("runtime"):
                g.append(("Verifiers API", "v1: a Taskset of Tasks, run by a harness" if env["runtime"] == "v1" else "v0: `load_environment()` with a Rubric"))
        else:
            g.append(("Environment", "not named: the dataset's card or repository should say which environment loads it"))
        g.append(("Scored by", env.get("reward") or (SWE_REWARD if swe else "the environment's rewards (a Rubric's functions in v0, `@vf.reward` methods in v1), "
                                                     "which get the reply" + (", `answer`" if "answer" in row else "") + (" and `info`" if "info" in row else ""))))
        if isinstance(row.get("task"), str):
            g.append(("Task", f"`{row['task']}`: the row's sub-environment (EnvGroup's routing key in verifiers up to 0.1.12)"))
        if gone:
            g.append(("Against", ", ".join(f"`{x}`" for x in gone) + ": withheld"))
        if env.get("src"):
            g.append(("Source", f"[research-environments: {env['src']}]({RE_SWE}/{env['src']})"))
        gblocks: list[dict[str, Any]] = [c.kv(g)]
        info = clean.get("info") if isinstance(clean.get("info"), dict) else None
        if info:
            rows = convo.field_rows({k: v for k, v in info.items() if k != "tool_defs"})
            if rows:
                gblocks += [c.note("`info`, which the rewards also get:", "list"), c.custom("rl", "fields", {"rows": rows}, convo.fields_text(rows))]
        sections.append(section("grading", "How it's graded", gblocks, kind="blocks", note="the environment's rewards"))
        sections.append(section("run", "Run it with Verifiers", run_blocks(ds.spec, env, config, split, row=i), kind="blocks", note="it doesn't run in this app"))
        shown = {"problem_statement", "prompt", "question", "info", "task", *IMAGE_COLS, "repo", "base_commit", "parent_commit", "workdir", "language",
                 "repo_language", "version", "test_cmd", "install_config"}
        rest = {k: v for k, v in clean.items() if k not in shown and v not in (None, "", [], {})}
        if rest:
            rows = convo.field_rows(rest)
            sections.append(section("data", "The rest of the row", [c.custom("rl", "fields", {"rows": rows}, convo.fields_text(rows))], kind="blocks"))

        glance = [["Row", f"{i:,}"]]
        if env.get("env"):
            glance.append(["Environment", f"`{env['env']}`"])
        if env.get("runtime"):
            glance.append(["API", env["runtime"]])
        img = next((clean[k] for k in IMAGE_COLS if isinstance(clean.get(k), str)), None)
        if img:
            glance.append(["Image", f"`{img}`"])
        glance.append(["Scored by", "hidden tests" if swe else "the environment's rewards"])
        card = self.card(row, i, roles)
        return {"title": title[:200], "id": card["id"], "chips": card["chips"], "sections": sections, "withheld": gone, "glance": glance,
                "summary": "Scored by hidden tests in the task's container" if swe else f"Scored by {env['env'] if env.get('env') else 'its environment'}",
                "framework_run": c.run_option("verifiers", "Verifiers", ok=False, endpoint=False, sandbox=False,
                                              why="Verifiers environments run with verifiers' own runner" + (" and their task containers" if swe else "")
                                                  + ", which this app doesn't host",
                                              about="Runs with Verifiers: the commands are under **Run it with Verifiers** on this page.")}

    def overview(self, ds: Dataset, roles: dict[str, Any], stats: list[dict[str, Any]], config: str, split: str) -> list[dict[str, Any]]:
        env = env_of(ds)
        swe = is_swe(ds.columns)
        kv: list[tuple[str, Any]] = []
        if env.get("env"):
            kv.append(("Environment", f"[{env['env']}]({HUB}/{env['env']})" + (f": {env['about']}" if env.get("about") else "")
                       + ("" if env.get("source") == "known" else " (named by the card)")))
            if env.get("runtime"):
                kv.append(("Verifiers API", "v1 (Taskset, harness, runtime)" if env["runtime"] == "v1" else "v0 (`load_environment()`, Rubric)"))
        else:
            kv.append(("Environment", "not named by the card: find the environment that loads it on the Environments Hub"))
        if swe:
            kv.append(("Each row", "a GitHub issue: the agent fixes it in the repository inside the task's container"))
            imgs = [next((str(r[k]) for k in IMAGE_COLS if isinstance(r.get(k), str)), "") for r in ds.sample[:20]]
            if any(imgs):
                kv.append(("Runs in", f"a Docker image per task (`{next(x for x in imgs if x)}`, …)"))
        else:
            kv.append(("The model gets", f"`{roles.get('task')}`" if roles.get("task") else "the environment's prompt"))
        kv.append(("Scored by", env.get("reward") or (SWE_REWARD if swe else "the environment's rewards")))
        tasks = next(((st.get("column_statistics") or {}).get("frequencies") for st in stats or [] if st.get("column_name") == "task"), None)
        if isinstance(tasks, dict) and len(tasks) > 1:
            kv.append(("Tasks", c.shares(sorted(((f"`{k}`", int(v)) for k, v in tasks.items()), key=lambda r: -r[1])[:8], sum(tasks.values()), "the `task` column")))
        if roles.get("answer"):
            kv.append(("Withheld", ", ".join(f"`{x}`" for x in roles["answer"])))
        return [c.section("vf", "What a row asks", [c.kv(kv)], icon="target", note="from the dataset and the Environments Hub"),
                c.section("run", "Run it with Verifiers", run_blocks(ds.spec, env, config, split), icon="play", collapsed=True, note="verifiers 0.3.1")]


def run_blocks(spec: str, env: dict[str, Any], config: str, split: str, row: int | None = None) -> list[dict[str, Any]]:
    """Commands that evaluate these rows with Verifiers 0.3.1 (the current release) and the Prime CLI."""
    eid = env.get("env")
    head = ["# Verifiers (pip: verifiers 0.3.1) and the Prime CLI, in a new uv project",
            "uv init vf-eval-here && cd vf-eval-here && uv add verifiers", "uv tool install prime"]
    if not eid:
        lines = head + ["", "# find and install the environment that loads this dataset (its card or repository names it)",
                        "prime env list", "prime env install <owner>/<environment>", "",
                        "# a v1 environment (a Taskset): verifiers' `eval`; a v0 one (load_environment): `vf-eval`",
                        "uv run eval <owner>/<environment> -m deepseek/deepseek-v4-flash -n 5 -r 1"]
        return [c.code("\n".join(lines), "run.sh"), c.links([("Verifiers on GitHub", VF), ("Docs", VF_DOCS), ("Environments Hub", HUB)])]
    name = eid.split("/")[-1]
    lines = head + ["", "# the environment, from Prime's Environments Hub", f"prime env install {eid}", ""]
    override = env.get("kind") == "swe" and DEFAULTS.get(eid) != (spec, split)
    if env.get("runtime") == "v0":
        lines += ["# a v0 environment: verifiers 0.3.1 runs it with the legacy vf-eval (OpenAI by default: OPENAI_API_KEY)",
                  f"uv run vf-eval {name} -m gpt-4.1-mini -n 5 -r 3"]
    elif override:
        lines += ["# this dataset and split, not the taskset's default: set them in a config", "cat > eval.toml <<'EOF'", "[env.taskset]", f'id = "{eid}"',
                  f'dataset_name = "{spec}"', f'split = "{split}"', "EOF",
                  "# each task runs in a Prime sandbox (prime login), or add --env.agent.runtime.type docker",
                  "uv run eval @ eval.toml -m deepseek/deepseek-v4-flash -n 5 -r 1"]
    else:
        lines += ["# v1: a Taskset, played by a harness" + ("; each task runs in a Prime sandbox (prime login), or add --env.agent.runtime.type docker"
                                                          if env.get("kind") == "swe" else ""),
                  f"uv run eval {eid} -m deepseek/deepseek-v4-flash -n 5 -r 1"]
    lines += ["# (on verifiers' main branch, eval is renamed vf-eval)"] if env.get("runtime") != "v0" else []
    blocks = [c.code("\n".join(lines), "run.sh")]
    if row is not None:
        blocks.append(c.note("The runner evaluates the first `-n` tasks of the split (or a shuffled sample with `-s`): there is no flag for one row.", "info"))
    items = [("This environment on the Hub", f"{HUB}/{eid}"), ("Verifiers on GitHub", VF), ("Docs", VF_DOCS)]
    if env.get("src"):
        items.insert(1, ("Its taskset's source", f"{RE_SWE}/{env['src']}"))
    items.append(("Train on it with prime-rl", PRIME_RL))
    blocks.append(c.links(items))
    return blocks
