"""NVIDIA NeMo Gym rows (nvidia/Nemotron-RL-*, anything with `responses_create_params`), read as NeMo Gym reads them.

A row is one task: `responses_create_params` is the request the policy gets (an OpenAI Responses API request: `input`,
a list of message / reasoning / function_call / function_call_output items, and `tools`), `agent_ref` names the agent
server that runs it ({type: responses_api_agents, name}), and the other columns are task data for that agent's
**resources server**, which scores the policy's result (`expected_answer`, `ground_truth`, `verifier_metadata`, ...).

What NeMo Gym itself says about each agent and resources server comes from its configs and TaskData schemas, read
into nemo_gym_catalog.json by nemo_gym_build.py (`catalog()["commit"]` is the commit): which resources server an agent
uses, what it verifies, which row fields its verifier reads, which of those hold the answer, and the config to start
it with. Rows naming an agent no config defines (renamed since, or internal) are still shown, from their fields.

"Pivot" datasets (…-Pivot-v1) pose each assistant step of an expert trajectory as a task of its own: the conversation
so far is the input, and the policy's next action (a tool call or a reply) is compared with the expert's
(`expected_action`, withheld).
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

from . import contract as c
from . import convo, rowfiles
from .base import Dataset, Processor, _common_prefix, parse_json, section, withhold

GITHUB = "https://github.com/NVIDIA-NeMo/Gym"
DOCS_CLI = "https://docs.nvidia.com/nemo/gym/main/reference/cli-commands"
DOCS_START = "https://docs.nvidia.com/nemo/gym/main/get-started"
DOCS_TRAIN = "https://docs.nvidia.com/nemo/gym/tutorials/training-tutorials/nemo-rl-grpo"

# Fields that hold the answer though their names don't say so (base.ANSWER catches expected_*, ground_truth, answer,
# reference_*, ...), whatever the agent: an expert's next step and its patch (Pivot rows), the gold calendar, hidden
# unit tests, a tau-style task's gold actions and must-say facts, a citation task's patterns, the expected response
# policy, a worked solution, a pairwise preference, a rubric's pass criteria. Each resources server adds its own
# (catalog: the TaskData fields that hold its answer).
HIDE = frozenset({
    "expected_action", "ref_message", "ref_patch", "exp_cal_state", "verifier_metadata.unit_tests", "verifier_metadata.tests",
    "evaluation_criteria.actions", "evaluation_criteria.communicate_info", "verifier.patterns", "response_policy_mapped",
    "expanded_completion", "simplifications", "simplified_values", "preference_ranking", "score1", "score2", "chosen", "rejected",
    "rubric.pass_criteria", "customer_scenario",
})
# Agents rows still name that NeMo Gym has since renamed or removed: the config that runs those rows now
RENAMED = {"single_step_tool_use_with_argument_comparison_swe": "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
           "turing_vif_simple_agent": "verifif_simple_agent", "rdkit_chemistry_agent": "litmus_agent_agent"}
FRAMEWORK = {"responses_create_params", "agent_ref", "task_source", "_ng_task_index", "_ng_rollout_index"}
# without a catalog entry: columns that look like task data for a verifier
VERIFIER_HINT = re.compile(r"(instruction|kwargs|schema|template_metadata|options|verifier|llm_judge|rubric|judge|principle|injection|"
                           r"required_tools|user_scenario|initial_state|evaluation_criteria|context|grading|output_regex|answer_format)", re.IGNORECASE)
ROLE_LABEL = {"verify": "the verifier", "prompt": "the prompt", "metrics": "metrics", "provenance": "provenance"}


@functools.lru_cache(maxsize=1)
def catalog() -> dict[str, Any]:
    try:
        return json.loads((Path(__file__).with_name("nemo_gym_catalog.json")).read_text())
    except (OSError, ValueError):
        return {"agents": {}, "servers": {}, "datasets": {}, "agent_types": {}}


def agent_of(row: dict[str, Any], spec: str) -> dict[str, Any]:
    """The agent a row names and what NeMo Gym's configs say about it: {name, known, config_name, config,
    resources_server, agent_type, max_steps, server_about, verification, readme, fields, renamed}."""
    ref = parse_json(row.get("agent_ref"))
    name = str(ref.get("name") or "") if isinstance(ref, dict) else ""
    if not name and isinstance(row.get("task_source"), str):
        name = row["task_source"]
    cat = catalog()
    out: dict[str, Any] = {"name": name, "known": False, "fields": {}}
    key = name if name in cat["agents"] else RENAMED.get(name) if RENAMED.get(name) in cat["agents"] else None
    if key is None:   # no agent_ref, or one no config defines: the config that names this very dataset, if one does
        key = next((u["agent"] for u in cat["datasets"].get(spec.lower()) or [] if u["agent"] in cat["agents"]), None)
    if key is None and not name:   # no agent named at all: the resources server whose task data these columns are
        key = _by_fields(row)
        out["inferred"] = bool(key)
    if key:
        a = cat["agents"][key]
        srv = cat["servers"].get(a.get("resources_server") or "", {})
        out.update(a, known=True, config_name=key, renamed=bool(name) and key != name, name=name or key, verification=srv.get("verification"),
                   server_about=srv.get("description"), readme=srv.get("readme"), fields=srv.get("fields") or {})
    return out


@functools.lru_cache(maxsize=256)
def _server_by_fields(cols: frozenset[str]) -> str | None:
    """The agent of the resources server that declares the most of these columns as its task data (at least three,
    and most of its own fields), for rows that name no agent."""
    cat = catalog()
    best, score = None, 0.0
    for rs, srv in cat["servers"].items():
        own = {k for k in (srv.get("fields") or {}) if "." not in k}
        hit = len(own & cols)
        if hit >= 3 and hit / max(1, len(own)) >= 0.6 and hit + hit / len(own) > score:
            best, score = rs, hit + hit / len(own)
    if not best:
        return None
    agents = sorted(k for k, a in cat["agents"].items() if a.get("resources_server") == best)
    return next((k for k in agents if k.endswith("_simple_agent")), agents[0] if agents else None)


def _by_fields(row: dict[str, Any]) -> str | None:
    return _server_by_fields(frozenset(k for k in row if k not in FRAMEWORK))


def hidden_for(agent: dict[str, Any]) -> frozenset[str]:
    """Every field name withheld for rows of this agent: HIDE, and its resources server's answer fields, wherever the
    server's TaskData may sit (top level, the legacy verifier_metadata bucket, the newer task_data one)."""
    own = {k for k, v in agent.get("fields", {}).items() if v.get("hide")}
    return HIDE | own | {f"{b}.{k}" for k in own for b in ("verifier_metadata", "task_data")}


def _has(row: dict[str, Any], dotted: str) -> bool:
    """Whether a row holds a value at a dotted path (through lists, as withhold reads them)."""
    nodes: list[Any] = [row]
    for k in dotted.split("."):
        nxt: list[Any] = []
        for n in nodes:
            n = parse_json(n)
            for m in (n if isinstance(n, list) else [n])[:50]:
                m = parse_json(m)
                if isinstance(m, dict) and m.get(k) is not None:
                    nxt.append(m[k])
        if not nxt:
            return False
        nodes = nxt
    return True


def is_pivot(row: dict[str, Any]) -> bool:
    """A Pivot row: one step of an expert trajectory, the expert's next action to match (a blend's other rows carry
    the column empty)."""
    return row.get("expected_action") is not None or row.get("ref_message") is not None


def _placeholder(row: dict[str, Any]) -> dict[str, Any] | None:
    """A blend row whose question its dataset's script fills in from another dataset: where from."""
    for k in ("_hf_placeholder", "_hf_question_placeholder"):
        v = parse_json(row.get(k))
        if isinstance(v, dict):
            return v
    return None


def _rcp(row: dict[str, Any]) -> dict[str, Any]:
    v = parse_json(row.get("responses_create_params"))
    return v if isinstance(v, dict) else {}


def _turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    rcp = _rcp(row)
    items = rcp.get("input")
    if items is None:   # raw rows (a prepare script makes the request): their messages or question
        items = parse_json(row.get("messages")) or next((row[k] for k in ("prompt", "question", "problem") if isinstance(row.get(k), str) and row[k]), None)
    ts = convo.turns(items)
    if isinstance(rcp.get("instructions"), str) and rcp["instructions"].strip():
        ts.insert(0, {"kind": "message", "role": "system", "text": rcp["instructions"]})
    return ts


def _strip(text: str, prefix: str) -> str:
    return text[len(prefix):].lstrip() if prefix and text.startswith(prefix) else text


def _template(row: dict[str, Any]) -> str | None:
    tm = parse_json(row.get("template_metadata"))
    return tm.get("template_prompt") if isinstance(tm, dict) else None


def _title(row: dict[str, Any], ts: list[dict[str, Any]], prefix: str, i: int) -> tuple[str, str]:
    """A title and a line under it: a question column of its own, else what the user first asks (an issue's title,
    or its first line past the preamble every task shares); a Pivot row's line is the user's latest message."""
    for k in ("question", "problem", "title"):
        if isinstance(row.get(k), str) and convo.headline(row[k]):
            return convo.headline(row[k]), ""
    ph = _placeholder(row)
    if ph and not convo.first_user(ts).strip():
        src = ph.get("dataset") or "another dataset"
        return f"A question from {src}" + (f" ({ph['split']}, row {ph['row']})" if ph.get("split") and ph.get("row") is not None else ""), \
            "Filled in by the dataset's own script; the published row holds only where it comes from."
    first = _strip(convo.first_user(ts), prefix)
    if not first:   # the user is simulated (tau-style): its opening line, or its scenario, names the task
        sc = parse_json(row.get("user_scenario"))
        inst = sc.get("instructions") if isinstance(sc, dict) else None
        first = next((str(v) for v in (row.get("opening_message"), row.get("initial_user_message"),
                                       inst.get("reason_for_call") if isinstance(inst, dict) else None) if isinstance(v, str) and v.strip()), "")
    title = convo.task_title(first, _template(row)) or convo.headline(next((t["text"] for t in ts if t["kind"] == "message" and t["text"].strip()), "")) \
        or f"Row {i}"
    brief = convo.after_title(first, title)
    last = convo.last_user(ts)
    if is_pivot(row) and last and last != convo.first_user(ts):
        brief = "Latest: " + re.sub(r"\s+", " ", last).strip()
    return title, brief


def _short(s: Any) -> bool:
    return isinstance(s, str) and 0 < len(s) <= 40 and "\n" not in s


def _step(row: dict[str, Any]) -> str:
    info = parse_json(row.get("info")) or parse_json(row.get("meta_info")) or {}
    if not isinstance(info, dict) or info.get("turn") is None and info.get("step") is None:
        return ""
    bits = [f"turn {info['turn']}" if info.get("turn") is not None else "", f"step {info['step']}" if info.get("step") is not None else ""]
    return " · ".join(b for b in bits if b)


def _gh(path: str, tree: bool = False) -> str:
    return f"{GITHUB}/{'tree' if tree else 'blob'}/main/{path}"


def graded_on(agent: dict[str, Any], pivot: bool) -> str:
    if pivot:
        return "the policy's next action matching the expert's (withheld): the same tool with matching arguments, or a reply where the expert replied"
    return agent.get("verification") or "its resources server's verifier"


def _server_link(a: dict[str, Any]) -> str:
    rs = a["resources_server"]
    return f"[{rs}]({_gh('resources_servers/' + rs, tree=True)})" + (f": {a['server_about']}" if a.get("server_about") else "")


# ── how to run it ────────────────────────────────────────────────────────────
def _download(spec: str, config: str, split: str, local: str, artifact: str | None, files: list[str]) -> list[str]:
    """Commands that put the split's rows in `local` as JSON Lines: the file a NeMo Gym config names, the default
    config's split by name, or a subset's own files (concatenated, or converted from parquet)."""
    if artifact:
        return [f"gym dataset download --repo-id {spec} \\", f"    --artifact {artifact} --output {local}"]
    if config == "default":
        return [f"gym dataset download --repo-id {spec} \\", f"    --split {split} --output {local}"]
    if files and all(f.endswith((".jsonl", ".ndjson")) for f in files):
        return [rowfiles.download(spec, files), "cat " + " ".join(rowfiles.local_paths(spec, files, "")) + f" > {local}"]
    if files and all(f.endswith(".parquet") for f in files):
        paths = rowfiles.local_paths(spec, files, "")
        src = repr(paths) if "*" not in paths[0] else f"glob.glob('{paths[0]}')"
        return [rowfiles.download(spec, files),
                "python -c \"import glob, pandas as pd; pd.concat(pd.read_parquet(p) for p in " + src + f").to_json('{local}', orient='records', lines=True)\""]
    return [f"python -c \"from datasets import load_dataset; load_dataset('{spec}', '{config}', split='{split}').to_json('{local}')\""]


def run_blocks(spec: str, config: str, split: str, agents: list[dict[str, Any]], row: int | None = None,
               meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The commands that run these rows with NeMo Gym's `gym` CLI (start the servers, then collect rollouts), and
    links. `agents`: the agents the rows name (one for a task; a blend's several for its dataset page)."""
    cat = catalog()
    known = [a for a in agents if a.get("known")]
    local = f"data/{split if config == 'default' else config + '-' + split}.jsonl"
    artifact = next((u.get("artifact") for u in cat["datasets"].get(spec.lower()) or [] if u.get("type") == split and u.get("artifact")), None) \
        if config == "default" else None
    what = f"this dataset's {split} split" + ("" if config == "default" else f" of {config}")
    lines = ["# NeMo Gym, from the revision used by this explorer's catalog (Python 3.13+, uv)",
             "git clone https://github.com/NVIDIA-NeMo/Gym.git && cd Gym", f"git checkout {cat['commit']}",
             "uv venv --python 3.13.14 && source .venv/bin/activate && uv sync", "",
             f"# the rows: {what}, as JSON Lines" + (", and this row alone" if row is not None else ""), *_download(spec, config, split, local, artifact,
                                                                                                    rowfiles.files_of(spec, meta or {}, config, split) if config != "default" else [])]
    inp = local
    if row is not None:
        inp = f"data/row{row}.jsonl"
        lines.append(f"sed -n '{row + 1}p' {local} > {inp}")
    if known:
        servers = sorted({a["resources_server"] for a in known})
        lines += ["", f"# the servers: {', '.join(servers)}, {'its agent' if len(known) == 1 else 'their agents'}, and your model",
                  "# (any OpenAI-compatible endpoint, or policy_base_url / policy_api_key / policy_model_name in env.yaml)",
                  "gym env start \\", *[f"    --config {cfg} \\" for cfg in sorted({a["config"] for a in known})],
                  "    --model-type openai_model --model gpt-4.1-2025-04-14 \\",
                  "    --model-url https://api.openai.com/v1 --model-api-key \"$OPENAI_API_KEY\"", "",
                  "# in a second terminal: roll out" + (" the row" if row is not None else " the first 5 rows") + "; the resources server scores each rollout",
                  "gym eval run --no-serve \\"]
        if len({a["config_name"] for a in known}) == 1:
            lines.append(f"    --agent {known[0]['config_name']} \\")   # several: each row goes to the agent it names
        lines.append(f"    --input {inp} --output results/rollouts.jsonl" + ("" if row is not None else " --limit 5"))
    else:
        lines += ["", "# the servers: find the config that defines the agent the rows name, then start it",
                  "gym list agents", "gym env start --config <its config> --model-type openai_model", "",
                  "gym eval run --no-serve --agent <its agent> \\", f"    --input {inp} --output results/rollouts.jsonl"]
    blocks = [c.code("\n".join(lines), "run.sh")]
    for a in agents:
        if a.get("renamed"):
            blocks.append(c.note(f"The rows name `{a['name']}`; NeMo Gym's current config for them defines `{a['config_name']}`, so `--agent` takes that name.", "info"))
            break
    unknown = sorted({a["name"] for a in agents if not a.get("known") and a.get("name")})
    if unknown or not known:
        blocks.append(c.note((f"No config in NeMo Gym's main branch (at {cat.get('commit', '')[:8]}) defines " + ", ".join(f"`{n}`" for n in unknown)
                              if unknown else "These rows name no agent") + ": it may be internal to NVIDIA, or renamed.", "alert"))
    items = [("NeMo Gym on GitHub", GITHUB)]
    if len(known) == 1:
        items += [("Its resources server", _gh(f"resources_servers/{known[0]['resources_server']}", tree=True)),
                  ("Its config", _gh(known[0]["config"]))]
    items += [("Quick start", DOCS_START), ("CLI reference", DOCS_CLI), ("Training with NeMo RL", DOCS_TRAIN)]
    blocks.append(c.links(items))
    return blocks


def run_option(agent: dict[str, Any], row: dict | None = None) -> dict[str, Any]:
    from .. import nemo_runner
    if row is not None and nemo_runner.supported(agent):
        try:
            nemo_runner.validate(row, agent)
        except (ValueError, KeyError, TypeError) as error:
            return c.run_option("nemo-gym", "NeMo Gym prediction", ok=False, why=str(error), endpoint=False, sandbox=False)
        return c.run_option("nemo-gym", "NeMo Gym prediction", ok=True, endpoint=False, sandbox=False,
                            about="One model prediction, scored by NVIDIA's NeMo Gym argument comparator. Tool calls are predictions, not executed actions. No sandbox is needed.",
                            notes=[f"Verifier pinned to NVIDIA-NeMo/Gym `{nemo_runner.REVISION[:12]}` with its native 0.1 word-similarity threshold."] +
                                  (["Some source tools have invalid strict schemas. Their parameters stay unchanged; strict enforcement is disabled so HF providers accept them."] if nemo_runner.relaxed_tools(row) else []),
                            harnesses=["opencode"], fields=[
                                c.field("max_tokens", "Max output tokens", "number", default=2048, min=256, max=8192, advanced=True),
                                c.field("temperature", "Temperature", "number", min=0, max=2, step=0.1, advanced=True)],
                            estimate={"tokens_in": 4000, "tokens_out": 1000, "minutes": 0})
    rs = agent.get("resources_server")
    return c.run_option("nemo-gym", "NeMo Gym", ok=False,
                        why=f"it is scored by NeMo Gym's {rs + ' ' if rs else ''}resources server, which runs only inside NeMo Gym",
                        about="Runs with NeMo Gym: the commands are under **Run it with NeMo Gym** on this page.", endpoint=False, sandbox=False)


# ── the reader ───────────────────────────────────────────────────────────────
class NemoGym(Processor):
    id, name, framework = "nemo-gym", "NeMo Gym", "NeMo Gym"
    about = ("NVIDIA NeMo Gym tasks: each row is a request for the policy (responses_create_params: the conversation so "
             "far and the tools it may call), the agent that runs it (agent_ref), and the task data its resources server "
             "scores the result with.")

    def match(self, ds: Dataset) -> float:
        if "responses_create_params" in ds.columns:
            return 0.95
        return 0.6 if any(t in ds.tags for t in ("nemo-gym", "library:nemo-gym")) else 0

    def roles(self, ds: Dataset) -> dict[str, Any]:
        r = super().roles(ds)
        cols = ds.columns
        agents = [agent_of(x, ds.spec) for x in ds.sample[:20]] or [agent_of({}, ds.spec)]
        hide = frozenset().union(*(hidden_for(a) for a in agents))
        verify = {k for a in agents for k, v in a["fields"].items() if "verify" in v.get("by", [])}
        r["task"] = "responses_create_params" if "responses_create_params" in cols else r["task"]
        r["answer"] = sorted({col for col in cols if col in hide} | {h for h in hide if "." in h and any(_has(x, h) for x in ds.sample[:20])}
                             | {col for col in r["answer"] if col not in FRAMEWORK})
        r["grading"] = ["agent_ref"] * ("agent_ref" in cols) + [col for col in cols if col not in r["answer"] and col not in FRAMEWORK and col != r["task"]
                                                              and (col in verify or (not verify and VERIFIER_HINT.search(col)))]
        r["environment"] = []
        firsts = [convo.first_user(_turns(x)) for x in ds.sample[:20]]
        r["_prefix"] = _common_prefix(firsts) if len([f for f in firsts if f]) >= 3 else ""
        r["_mixed"] = len({a["name"] for a in agents}) > 1
        return r

    def card(self, row: dict[str, Any], i: int, roles: dict[str, Any]) -> dict[str, Any]:
        ts = _turns(row)
        title, brief = _title(row, ts, roles.get("_prefix") or "", i)
        chips = [str(row[k]) for k in ("category", "domain", "source", "schema_type", "attack_category", "question_type", "property_type", "language", "dataset")
                 if _short(row.get(k))][:3]
        step = _step(row)
        if step:
            chips.insert(0, step)
        ident = next((str(row[k]) for k in ("id", "uuid", "task_id", "trajectory_id") if isinstance(row.get(k), (str, int)) and str(row.get(k))), None)
        tools = convo.tools(_rcp(row).get("tools"))
        out: dict[str, Any] = {"i": i, "id": ident[:40] if ident else None, "title": title[:200], "snippet": brief[:260], "chips": chips,
                               "stats": [(len(tools), "tools")] if tools else []}
        if roles.get("_mixed"):
            ref = parse_json(row.get("agent_ref"))
            out["lead"] = str(ref.get("name")) if isinstance(ref, dict) and ref.get("name") else None
        return out

    def view(self, row: dict[str, Any], i: int, ds: Dataset, roles: dict[str, Any]) -> dict[str, Any]:
        rcp = _rcp(row)
        pivot = is_pivot(row)
        agent = agent_of(row, ds.spec)
        hide = hidden_for(agent)
        # what the policy sees (input, tools) is shown as it is: it can't be the hidden answer. The rest is withheld from.
        rest_rcp, gone_rcp = withhold({k: v for k, v in rcp.items() if k not in ("input", "tools")}, "responses_create_params")
        clean, gone = withhold({k: parse_json(v) for k, v in row.items() if k != "responses_create_params"}, hide=hide, keep=("agent_ref",))
        gone = sorted(set(gone + gone_rcp))
        ts = _turns(row)
        tools = convo.tools(rcp.get("tools"))
        title, _ = _title(row, ts, roles.get("_prefix") or "", i)
        step = _step(row)
        config, split = roles.get("_config", "default"), roles.get("_split", "train")

        # the task: the conversation, and what the policy does next
        if pivot:
            nxt = {"label": "Policy", "text": "Its next action, a tool call or a reply, is compared with the expert's (withheld)."}
            note = (f"One step of an expert trajectory{f' ({step})' if step else ''}: the policy gets the conversation so far, "
                    "including the expert's earlier tool calls and their outputs, and must take the expert's next action.")
        else:
            rs = agent.get("resources_server")
            nxt = {"label": "Policy", "text": "Replies" + (", calling the tools below as it goes" if tools else "") + ". "
                   + (f"The {rs} resources server then scores the result" if rs else "Its resources server then scores the result")
                   + (f": {agent['verification'].lower()}." if agent.get("verification") else ".")}
            note = ""
        ph = _placeholder(row)
        if ph and not convo.first_user(ts).strip():
            where = f"`{ph.get('dataset')}`" + (f", split `{ph['split']}`, row {ph['row']}" if ph.get("split") and ph.get("row") is not None else "")
            task_blocks = [c.note(f"The published row leaves its question out: the dataset's own script fills it in from {where}, "
                                  "so the request below is empty until then.", "info")]
            if ts:
                task_blocks.append(c.custom("rl", "transcript", {"turns": ts, "next": nxt, "note": note}, convo.transcript_text(ts)))
        elif ts:
            task_blocks = [c.custom("rl", "transcript", {"turns": ts, "next": nxt, "note": note}, convo.transcript_text(ts))]
        else:
            task_blocks = [c.note("This row has no request for the policy (`responses_create_params.input`).", "alert")]
        if not rcp and ts:
            task_blocks.insert(0, c.note("This row isn't a NeMo Gym task yet (it has no `responses_create_params`): it's source data that a "
                                         "prepare script turns into requests. Shown here from its own columns.", "info"))
        sections = [section("task", "The task", task_blocks, kind="blocks",
                            note="responses_create_params.input, as the policy gets it" if rcp.get("input") is not None else "")]
        if tools:
            sections.append(section("tools", f"Tools ({len(tools)})", [c.custom("rl", "tools", {"tools": tools}, convo.tools_text(tools))], kind="blocks",
                                    note="responses_create_params.tools"))

        # how it's graded: the agent, its resources server, what it checks, and the row's task data, by who reads it
        g: list[tuple[str, Any]] = []
        if agent.get("inferred"):
            g.append(("Agent", f"none named in the row; `{agent['config_name']}` runs rows like it (its resources server's task data matches these columns)"))
        elif agent.get("name"):
            g.append(("Agent", f"`{agent['name']}`" + (f", a `{agent['agent_type']}`" if agent.get("agent_type") else "")))
        if agent.get("renamed"):
            g.append(("Now", f"`{agent['config_name']}` (NeMo Gym renamed it)"))
        if agent.get("resources_server"):
            g.append(("Resources server", _server_link(agent)))
        g.append(("Scored on", graded_on(agent, pivot)))
        if agent.get("max_steps"):
            g.append(("Step limit", f"{agent['max_steps']} model calls"))
        if gone:
            g.append(("Against", ", ".join(f"`{x}`" for x in gone) + ": withheld"))
        p = _pass_rate(row)
        if p:
            g.append(("Pass rate", p))
        blocks: list[dict[str, Any]] = [c.kv(g)]
        if agent.get("readme"):
            blocks.append(c.disclose("What the resources server does", [c.markdown(agent["readme"])]))
        roles_of = agent.get("fields") or {}
        task_data = {k: v for k, v in clean.items() if k not in FRAMEWORK and v not in (None, "", [], {})
                     and ("verify" in roles_of.get(k, {}).get("by", []) or (not roles_of and VERIFIER_HINT.search(k)))}
        if task_data:
            rows = convo.field_rows(task_data)
            blocks += [c.note("What the verifier reads from the row" + ("" if roles_of else " (inferred from the column names)") + ":", "target"),
                       c.custom("rl", "fields", {"rows": rows}, convo.fields_text(rows))]
        if agent.get("name") and not agent.get("known"):
            blocks.append(c.note(f"No config in NeMo Gym's main branch defines `{agent['name']}`, so what it checks is read off the row's fields.", "alert"))
        sections.append(section("grading", "How it's graded", blocks, kind="blocks", note="agent_ref and the task data"))

        settings = [(k, v) for k, v in rest_rcp.items() if v is not None and not isinstance(v, (dict, list))]
        nested = {k: v for k, v in rest_rcp.items() if isinstance(v, (dict, list)) and v}
        if settings or nested:
            rows = convo.field_rows({**dict(settings), **nested})
            sections.append(section("request", "Request settings", [c.custom("rl", "fields", {"rows": rows}, convo.fields_text(rows))], kind="blocks",
                                    note="the rest of responses_create_params"))
        option = run_option(agent, row)
        sections.append(section("run", "Run it with NeMo Gym", run_blocks(ds.spec, config, split, [agent], row=i, meta=ds.card), kind="blocks",
                                note="Native CLI alternative; prediction runs are also available here" if option["ok"] else "This task needs the native runtime"))
        rest = {k: v for k, v in clean.items() if k not in task_data and k not in FRAMEWORK and v not in (None, "", [], {})}
        if rest:
            label = "provenance and metrics" if roles_of else "the rest of the row"
            rows = convo.field_rows(rest)
            sections.append(section("data", "The rest of the row", [c.note(f"Not read by the verifier: {label}.", "info"),
                                                                    c.custom("rl", "fields", {"rows": rows}, convo.fields_text(rows))], kind="blocks"))

        glance = [["Row", f"{i:,}"]]
        if agent.get("name"):
            glance.append(["Agent", f"`{agent['name']}`"])
        if agent.get("resources_server"):
            glance.append(["Resources server", f"`{agent['resources_server']}`"])
        glance.append(["Scored on", "the expert's next action" if pivot else (agent.get("verification") or ("its resources server" if agent.get("known") else "–"))])
        glance.append(["Tools", f"{len(tools):,}" if tools else "none"])
        if step:
            glance.append(["Step", step])
        if p:
            glance.append(["Pass rate", p])
        card = self.card(row, i, roles)
        return {"title": title[:200], "id": card["id"], "chips": card["chips"], "sections": sections, "withheld": gone, "glance": glance,
                "framework_run": option,
                "summary": "Scored on the next action matching the expert's" if pivot else
                           (f"Verifier: {agent['verification'].lower()}" if agent.get("verification") else "Scored by its resources server")}

    def overview(self, ds: Dataset, roles: dict[str, Any], stats: list[dict[str, Any]], config: str, split: str) -> list[dict[str, Any]]:
        """The dataset at a glance, from its first rows and the viewer's statistics: the agent(s) and resources
        server(s), what's scored and what's withheld, tools, pass rates; then how to run it."""
        rows = ds.sample[:20]
        agents: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = {}
        for x in rows:
            a = agent_of(x, ds.spec)
            k = a.get("name") or "(none)"
            agents.setdefault(k, a)
            counts[k] = counts.get(k, 0) + 1
        n_piv = sum(1 for x in rows if is_pivot(x))
        pivot = bool(rows) and n_piv == len(rows)
        ntools = [len(convo.tools(_rcp(x).get("tools"))) for x in rows]
        kv: list[tuple[str, Any]] = []
        if len(agents) == 1:
            a = next(iter(agents.values()))
            if a.get("inferred"):
                kv.append(("Agent", f"none named in the rows; `{a['config_name']}` runs rows like these (its resources server's task data matches the columns)"))
            elif a.get("name"):
                kv.append(("Agent", f"`{a['name']}`" + (f", a `{a['agent_type']}`" if a.get("agent_type") else "")))
            if a.get("renamed"):
                kv.append(("Now", f"`{a['config_name']}` (NeMo Gym renamed it)"))
            if a.get("resources_server"):
                kv.append(("Resources server", _server_link(a)))
            kv.append(("Scored on", graded_on(a, pivot)))
        else:
            kv.append(("Agents", c.shares(sorted(((f"`{k}`" + (f" ({v['resources_server']})" if v.get("resources_server") else ""), counts[k])
                                                  for k, v in agents.items()), key=lambda r: -r[1]), len(rows), "in the first rows; each row names its own")))
        if pivot:
            kv.append(("Each row", "one step of an expert trajectory: the conversation so far in, the expert's next action withheld"))
        elif n_piv:
            kv.append(("Pivot rows", f"{n_piv} of the first {len(rows)}: one step of an expert trajectory each, the expert's next action withheld"))
        if any(ntools):
            lo, hi = min(ntools), max(ntools)
            kv.append(("Tools", f"{lo} per task" if lo == hi else f"{lo} to {hi} per task"))
        else:
            kv.append(("Tools", "none: the policy answers in words"))
        if roles.get("answer"):
            kv.append(("Withheld", ", ".join(f"`{x}`" for x in roles["answer"])))
        out = [c.section("nemo", "What the policy does", [c.kv(kv)], icon="target", note="from the first rows")]
        hist = _histogram(stats, ("pass_rate", "task_difficulty_qwen3_32b_avg_8"))
        if hist:
            out.append(c.section("passrate", "How hard it is", hist, icon="gauge", note="the dataset viewer's statistics"))
        out.append(c.section("run", "Run it with NeMo Gym", run_blocks(ds.spec, config, split, list(agents.values()), meta=ds.card), icon="play", collapsed=True,
                             note="the gym CLI"))
        return out


def _pass_rate(row: dict[str, Any]) -> str:
    n, k = row.get("pass_rate_total"), row.get("pass_rate_passed")
    if isinstance(n, int) and isinstance(k, int) and n:
        return f"{k} of {n} reference rollouts passed ({k / n:.0%})"
    q = parse_json(row.get("qwen_235b_info"))
    if isinstance(q, dict) and isinstance(q.get("rewards"), list) and q["rewards"]:
        r = q["rewards"]
        return f"{sum(1 for x in r if x)} of {len(r)} Qwen3-235B rollouts passed"
    return ""


def _histogram(stats: list[dict[str, Any]], cols: tuple[str, ...]) -> list[dict[str, Any]]:
    """A numeric column's spread from the viewer's statistics, as shares: pass rates, difficulty."""
    for st in stats or []:
        if st.get("column_name") in cols and st.get("column_type") in ("float", "int"):
            s = st.get("column_statistics") or {}
            h = s.get("histogram") or {}
            hist, edges = h.get("hist") or [], h.get("bin_edges") or []
            if not hist or len(edges) != len(hist) + 1:
                continue
            rows = [(f"{edges[j]:.2g} to {edges[j + 1]:.2g}", n) for j, n in enumerate(hist)]
            return [c.kv([(st["column_name"].replace("_", " "), f"mean {s.get('mean') or 0:.2f}, median {s.get('median') or 0:.2f}")]),
                    c.shares(rows, sum(hist), f"rows by `{st['column_name']}`")]
    return []
