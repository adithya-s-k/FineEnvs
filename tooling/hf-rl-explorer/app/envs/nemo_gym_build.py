"""Rebuild nemo_gym_catalog.json from a NeMo Gym checkout (github.com/NVIDIA-NeMo/Gym), for nemogym.py:

    git clone --depth 1 https://github.com/NVIDIA-NeMo/Gym.git /tmp/nemo-gym
    uv run python -m app.envs.nemo_gym_build /tmp/nemo-gym

It records, at that commit: every agent a config defines (its harness, resources server, config file, step limit),
every resources server (what it verifies, from the docs' environment list and its README) with the row fields its
TaskData declares and what consumes each (verify, prompt, metrics, provenance), which of those hold the answer (by
their description, then reviewed by hand: EXTRA_HIDE, NOT_SECRET), and which HF datasets each config names.
Needs PyYAML (a maintainer's tool; the app only reads the JSON).
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def main(ROOT: Path) -> None:
    OUT = CAT = Path(__file__).with_name("nemo_gym_catalog.json")
    sha = subprocess.check_output(["git", "-C", str(ROOT), "log", "-1", "--format=%H %cs"], text=True).split()
    servers, agents, datasets = {}, {}, {}

    def readme_para(d: Path) -> str:
        p = d / "README.md"
        if not p.exists():
            return ""
        text = p.read_text(errors="replace")
        paras = [x.strip() for x in re.split(r"\n\s*\n", text)]
        for x in paras:
            x = "\n".join(l for l in x.splitlines() if not l.lstrip().startswith("#")).strip()
            if x.startswith("#") or x.startswith("```") or x.startswith("|") or x.startswith("<") or x.startswith("!") or len(x) < 60:
                continue
            x = re.sub(r"\s+", " ", x)
            x = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", x)
            return x[:700]
        return ""

    # the environment list: verification per resources server
    table = {}
    envlist = ROOT / "fern/versions/latest/pages/evaluation/environment-list.mdx"
    for ln in envlist.read_text().splitlines():
        m = re.match(r"^\| \[([\w\-/.]+)\]\([^)]*\) \| (.*?) \| (.*?) \|", ln)
        if m:
            table[m.group(1)] = {"description": re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", m.group(2)).strip(), "verification": m.group(3).strip()}

    def load(p):
        try:
            return yaml.safe_load(p.read_text()) or {}
        except Exception:
            return {}

    def scan(cfg: Path, server_dir: str | None, flavor: str | None, env_name: str | None):
        doc = load(cfg)
        if not isinstance(doc, dict):
            return
        rs_instances = {}
        for inst, body in doc.items():
            if isinstance(body, dict) and isinstance(body.get("resources_servers"), dict):
                for stype, s in body["resources_servers"].items():
                    if isinstance(s, dict):
                        rs_instances[inst] = {"dir": stype, "description": s.get("description"), "value": s.get("value"), "domain": s.get("domain")}
        for inst, body in doc.items():
            if not (isinstance(body, dict) and isinstance(body.get("responses_api_agents"), dict)):
                continue
            for atype, a in body["responses_api_agents"].items():
                if not isinstance(a, dict):
                    continue
                rs = (a.get("resources_server") or {}).get("name") if isinstance(a.get("resources_server"), dict) else None
                info = rs_instances.get(rs) or {}
                rec = {"agent_type": atype, "config": str(cfg.relative_to(ROOT)), "resources_server": info.get("dir") or server_dir,
                       "server_instance": rs, "flavor": flavor, "environment": env_name, "description": info.get("description"),
                       "value": info.get("value"), "domain": info.get("domain"), "max_steps": a.get("max_steps")}
                rec = {k: v for k, v in rec.items() if v is not None}
                prev = agents.get(inst)
                if not prev or (prev.get("environment") and not rec.get("environment")):   # a resources_servers config wins (works in released versions too)
                    agents[inst] = {**(prev or {}), **rec} if prev and not rec.get("environment") else rec
                    if prev and prev.get("environment") and not rec.get("environment"):
                        agents[inst]["environment"] = prev["environment"]
                elif rec.get("environment") and not prev.get("environment"):
                    prev["environment"] = rec["environment"]
                for d in a.get("datasets") or []:
                    hf = d.get("huggingface_identifier") if isinstance(d, dict) else None
                    if isinstance(hf, dict) and hf.get("repo_id"):
                        datasets.setdefault(hf["repo_id"].lower(), []).append({"agent": inst, "type": d.get("type"), "artifact": hf.get("artifact_fpath"),
                                                                              "config": str(cfg.relative_to(ROOT))})

    for d in sorted((ROOT / "resources_servers").iterdir()):
        if not (d / "configs").is_dir():
            continue
        t = table.get(d.name, {})
        servers[d.name] = {k: v for k, v in {"description": t.get("description"), "verification": t.get("verification"), "readme": readme_para(d)}.items() if v}
        for cfg in sorted((d / "configs").glob("*.yaml")):
            flavor = d.name if cfg.stem == d.name else f"{d.name}/{cfg.stem}"
            scan(cfg, d.name, flavor, None)
    envs_dir = ROOT / "environments"
    if envs_dir.is_dir():
        for d in sorted(envs_dir.iterdir()):
            if (d / "config.yaml").exists():
                scan(d / "config.yaml", None, None, d.name)
    # dedupe datasets
    for k, v in datasets.items():
        seen, out = set(), []
        for x in v:
            key = (x["agent"], x["type"], x.get("artifact"))
            if key not in seen:
                seen.add(key); out.append(x)
        datasets[k] = out
    agent_dirs = {d.name: readme_para(d) for d in sorted((ROOT / "responses_api_agents").iterdir()) if d.is_dir()}
    OUT.write_text(json.dumps({"source": "https://github.com/NVIDIA-NeMo/Gym", "commit": sha[0], "date": sha[1],
                               "servers": servers, "agents": dict(sorted(agents.items())), "datasets": dict(sorted(datasets.items())),
                               "agent_types": {k: v for k, v in agent_dirs.items() if v}}, indent=0, ensure_ascii=False, sort_keys=False) + "\n")
    print(len(servers), "servers", len(agents), "agents", len(datasets), "datasets", OUT.stat().st_size, "bytes")

    # ── the row fields each resources server's TaskData declares ──
    SECRET = re.compile(r"(correct|expected|gold|ground[ -_]?truth|reference (answer|solution|response|patch)|solution|hidden|answer key|"
                        r"golden|oracle|target (value|output|answer)|the answer\b|label for|true label|unit tests?|test cases?|test patch|fail_to_pass|pass_to_pass)", re.IGNORECASE)
    classes = {}   # (module, name) -> (bases[(module,name)], fields{name: {...}})

    def mod_of(p: Path) -> str:
        return ".".join(p.relative_to(ROOT).with_suffix("").parts)

    def lit(node):
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    for f in sorted((ROOT / "resources_servers").glob("*/task_data.py")):
        m = mod_of(f)
        tree = ast.parse(f.read_text())
        imports = {}
        for n in tree.body:
            if isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    imports[a.asname or a.name] = (n.module, a.name)
        for n in ast.walk(tree):
            if not isinstance(n, ast.ClassDef):
                continue
            bases = []
            for b in n.bases:
                nm = b.id if isinstance(b, ast.Name) else None
                if nm and nm != "BaseModel":
                    bases.append(imports.get(nm, (m, nm)))
            fields = {}
            for st in n.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    name = st.target.id
                    desc, cons, alias = "", [], None
                    if isinstance(st.value, ast.Call):
                        for kw in st.value.keywords:
                            if kw.arg == "description":
                                desc = lit(kw.value) or ""
                            elif kw.arg == "json_schema_extra":
                                d = lit(kw.value)
                                if isinstance(d, dict):
                                    cons = d.get("consumed_by") or []
                            elif kw.arg in ("alias", "validation_alias"):
                                alias = lit(kw.value)
                    fields[alias if isinstance(alias, str) else name] = {"consumed_by": cons, "description": desc if isinstance(desc, str) else ""}
            classes[(m, n.name)] = (bases, fields)

    def all_fields(key, depth=0):
        if key not in classes or depth > 6:
            return {}
        bases, fields = classes[key]
        out = {}
        for b in bases:
            out.update(all_fields(b, depth + 1))
        out.update(fields)
        return out

    cat = json.loads(CAT.read_text())
    n_hide = 0
    for f in sorted((ROOT / "resources_servers").glob("*/task_data.py")):
        rs = f.parent.name
        fields = all_fields((mod_of(f), "TaskData"))
        if not fields or rs not in cat["servers"]:
            continue
        out = {}
        for name, x in fields.items():
            cons = x["consumed_by"]
            secret = "verify" in cons and "prompt" not in cons and bool(SECRET.search(x["description"] or ""))
            out[name] = {"by": cons, **({"hide": True} if secret else {}), **({"about": x["description"][:300]} if x["description"] else {})}
            n_hide += secret
        cat["servers"][rs]["fields"] = out
    CAT.write_text(json.dumps(cat, indent=0, ensure_ascii=False) + "\n")
    print("servers with fields:", sum(1 for s in cat["servers"].values() if s.get("fields")), "hidden fields:", n_hide, CAT.stat().st_size)
    for rs in ["mcqa", "reasoning_gym", "workplace_assistant", "calendar", "code_gen", "format_verification", "indirect_prompt_injection", "verifif",
               "multichallenge", "single_step_tool_use_with_argument_comparison", "instruction_following", "conversational_tool_use_simulation", "genrm_compare", "structured_outputs", "math_with_judge", "litmus_agent", "jailbreak_detection", "ns_tools"]:
        fs = cat["servers"].get(rs, {}).get("fields", {})
        print(rs, {k: ("HIDE " if v.get("hide") else "") + ",".join(v["by"]) for k, v in fs.items()})

    # reviewed by hand (2026-10-03): fields the descriptions miss, and descriptions that over-match
    EXTRA_HIDE = {"single_step_tool_use_with_argument_comparison": ["expected_action", "ref_message", "ref_patch"], "swe_pivot": ["expected_action", "ref_message", "ref_patch"],
                  "format_verification": ["verifier"], "conversational_tool_use_simulation": ["customer_scenario", "profile"],
                  "jailbreak_detection": ["response_policy_mapped"], "indirect_prompt_injection": ["injection"], "multichallenge": ["rubric.pass_criteria"],
                  "inverse_if": ["rubric.pass_criteria", "reference_response"], "calendar": ["exp_cal_state"], "code_gen": ["verifier_metadata.unit_tests"]}
    NOT_SECRET = {"proof_judge": ["problem"], "rolemrc": ["dimension"], "scicode": ["problem_id"], "spider2_lite": ["ignore_order"], "structeval": ["output_type"],
                  "litmus_agent": ["use_box_format"], "terminal_bench_2_1": ["task_folder"], "terminus_judge": ["metadata"], "verifif": ["instructions"],
                  "moldetox": ["scoring_mode"], "swerl_llm_judge": ["grading_mode"], "fukuyamabench": ["lenient"]}
    cat = json.loads(CAT.read_text())
    for rs, names in EXTRA_HIDE.items():
        s = cat["servers"].setdefault(rs, {})
        for n in names:
            s.setdefault("fields", {}).setdefault(n, {"by": ["verify"]})["hide"] = True
    for rs, names in NOT_SECRET.items():
        for n in names:
            (cat["servers"].get(rs, {}).get("fields", {}).get(n) or {}).pop("hide", None)
    for s in cat["servers"].values():   # trim: only what the explorer shows
        for v in (s.get("fields") or {}).values():
            v.pop("about", None)
    CAT.write_text(json.dumps(cat, indent=0, ensure_ascii=False) + "\n")
    print("final", CAT.stat().st_size)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
