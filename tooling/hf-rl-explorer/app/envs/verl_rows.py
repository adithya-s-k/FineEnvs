"""verl and SkyRL RL rows, read as those trainers read them.

verl (github.com/verl-project/verl, formerly volcengine/verl; docs "Prepare Data for Post-Training"): a row is
`data_source` (picks the reward function), `prompt` (the chat the policy gets), `ability` (a label nothing reads),
`reward_model` {style, ground_truth} (only ground_truth is read: the answer, withheld), `extra_info` (passed whole to the
reward function: index, split, and often the question and its full answer, withheld), optionally `agent_name` (the
agent loop: single_turn_agent, tool_agent), `extra_info.tools_kwargs` ({tool: {create_kwargs}}, for tool rows) and
`images` / `videos` / `audios`. With no custom reward function, verl.utils.reward_score.default_compute_score routes
`data_source` to a built-in scorer (SCORERS below) and raises for anything else.

SkyRL (github.com/NovaSky-AI/SkyRL): the same shape with `env_class` (the skyrl-gym environment that runs and scores
the row; a row's own overrides the config's `environment.env_class`) and `reward_spec` {method, ground_truth}; every
other column reaches the environment as extras.

Sources read at verl 8718ca30 (2026-10-01; v0.9.1 is the latest release) and SkyRL ed34c9c (2026-10-03).
"""

from __future__ import annotations

import re
from typing import Any

from . import contract as c
from . import convo, rowfiles
from .base import Dataset, Processor, _common_prefix, parse_json, section, withhold

VERL = "https://github.com/verl-project/verl"
VERL_DATA_DOC = "https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"
VERL_REWARD_DOC = "https://verl.readthedocs.io/en/latest/preparation/reward_function.html"
VERL_QUICKSTART = "https://verl.readthedocs.io/en/latest/start/quickstart.html"
SKYRL = "https://github.com/NovaSky-AI/SkyRL"
SKYRL_DATA_DOC = "https://docs.skyrl.ai/docs/datasets/dataset-preparation"

# verl.utils.reward_score.default_compute_score: data_source -> (scorer module, what it checks)
SCORERS: list[tuple[Any, str, str]] = [
    (lambda d: d == "openai/gsm8k", "gsm8k", "the number after `####` in the reply, exactly"),
    (lambda d: d in ("lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"), "math_reward",
     "the last `\\boxed{}` answer, normalized and compared"),
    (lambda d: d in ("math_dapo", "math", "math_dapo_reasoning") or d.startswith("aime"), "math_dapo",
     "the final answer (Minerva-style `Answer:`, else `\\boxed{}`), normalized: +1 right, -1 wrong"),
    (lambda d: d in ("numina_aops_forum", "numina_synthetic_math", "numina_amc_aime", "numina_synthetic_amc", "numina_cn_k12", "numina_olympiads"),
     "prime_math", "math equivalence of the final answer (PRIME's grader)"),
    (lambda d: d in ("codecontests", "apps", "codeforces", "taco"), "sandbox_fusion / prime_code",
     "runs the reply's code against the test cases in the ground truth (in Sandbox Fusion when its URL is set)"),
    (lambda d: d == "hiyouga/geometry3k", "geo3k", "the `\\boxed{}` answer, plus a small format score"),
    (lambda d: d.startswith("searchR1_"), "search_r1_like_qa_em", "exact match of the `<answer>` with any target"),
]
SKYRL_ENVS = {"gsm8k": "the number after `####`, exactly", "gsm8k_multi_turn": "GSM8K over several turns", "aime": "the final math answer",
              "text2sql": "runs the reply's SQL on the task's database and compares the result", "search": "exact match of the answer, with a search tool",
              "lcb": "runs the reply's code against LiveCodeBench tests", "searchcode": "code with a search tool"}
# answers verl's names don't say: SkyRL-SQL's gold query and its full reference reasoning
HIDE = frozenset({"sql", "output_seq", "gold_sql", "extra_info.sql", "extra_info.gold_sql", "extra_info.output_seq"})
BLOB = re.compile(r"^H4sI[A-Za-z0-9+/=]{200,}")   # gzip, base64: opaque, and often the hidden tests


def scorer(data_source: str) -> tuple[str, str] | None:
    d = str(data_source or "")
    return next(((name, what) for test, name, what in SCORERS if test(d)), None)


def is_skyrl(cols: list[str]) -> bool:
    return "env_class" in cols or "reward_spec" in cols


def _blobs(v: Any, path: str = "", depth: int = 0) -> tuple[Any, list[str]]:
    """Compressed blobs left out (named), wherever they are: they're opaque here, and hold tests or answers."""
    if depth > 6:
        return v, []
    if isinstance(v, str) and BLOB.match(v):
        return None, [path]
    if isinstance(v, dict):
        out, gone = {}, []
        for k, x in v.items():
            y, g = _blobs(x, f"{path}.{k}" if path else str(k), depth + 1)
            gone += g
            if y is not None or not g:
                out[k] = y
        return out, gone
    return v, []


def _chat(row: dict[str, Any]) -> list[dict[str, Any]]:
    return convo.turns(row.get("prompt") if row.get("prompt") is not None else row.get("raw_prompt"))


def _images(row: dict[str, Any]) -> list[str]:
    """The row's images as URLs the page can show: the viewer's signed https links, or small inline bytes."""
    import base64

    out = []
    for im in (parse_json(row.get("images")) or [])[:12] if isinstance(parse_json(row.get("images")), list) else []:
        if isinstance(im, dict) and isinstance(im.get("src"), str) and im["src"].startswith("https://"):
            out.append(im["src"])
        elif isinstance(im, dict) and isinstance(im.get("bytes"), (bytes, bytearray)) and len(im["bytes"]) < 1_500_000:
            out.append("data:image/png;base64," + base64.b64encode(im["bytes"]).decode())
        elif isinstance(im, str) and im.startswith("https://"):
            out.append(im)
    return out


def _tool_names(extra: dict[str, Any]) -> list[str]:
    tk = parse_json(extra.get("tools_kwargs")) if isinstance(extra, dict) else None
    return sorted(tk) if isinstance(tk, dict) else []


class Verl(Processor):
    id, name, framework = "verl", "verl / SkyRL rows", "verl"
    about = ("verl's RL format (SkyRL reads it too): each row is a chat prompt for the policy, its data source (which picks "
             "the reward function), its ability, and a reward_model whose ground truth, the answer, is withheld; SkyRL rows add "
             "env_class, the environment that scores them.")

    def framework_for(self, ds: Dataset) -> str:
        return "SkyRL" if is_skyrl(ds.columns) else "verl"

    def match(self, ds: Dataset) -> float:
        if ds.has("prompt", "reward_model") or ds.has("prompt", "reward_spec") or ds.has("prompt", "env_class"):
            return 0.9
        return 0.5 if ds.has("data_source", "prompt", "ability") else 0.0

    def roles(self, ds: Dataset) -> dict[str, Any]:
        r = super().roles(ds)
        cols = ds.columns
        r["task"] = "prompt" if "prompt" in cols else r["task"]
        r["grading"] = [x for x in ("data_source", "reward_model", "reward_spec", "env_class", "ability", "agent_name") if x in cols]
        r["answer"] = sorted(set(r["answer"]) | {x for x in cols if x in HIDE} | ({"reward_model.ground_truth"} if "reward_model" in cols else set())
                             | ({"reward_spec.ground_truth"} if "reward_spec" in cols else set())
                             | ({"extra_info.answer"} if any(isinstance(parse_json(x.get("extra_info")), dict) and "answer" in parse_json(x.get("extra_info"))
                                                             for x in ds.sample[:20]) else set()))
        r["answer"] = [a for a in r["answer"] if a not in ("reward_model", "reward_spec")]
        r["environment"] = [x for x in ("images", "videos", "audios") if x in cols]
        firsts = [convo.first_user(_chat(x)) for x in ds.sample[:20]]
        r["_prefix"] = _common_prefix(firsts) if len([f for f in firsts if f]) >= 3 else ""
        r["_skyrl"] = is_skyrl(cols)
        return r

    def _title(self, row: dict[str, Any], ts: list[dict[str, Any]], roles: dict[str, Any], i: int) -> tuple[str, str]:
        extra = parse_json(row.get("extra_info"))
        q = extra.get("question") if isinstance(extra, dict) and isinstance(extra.get("question"), str) else row.get("question")
        first = convo.first_user(ts)
        p = roles.get("_prefix") or ""
        own = first[len(p):].lstrip() if p and first.startswith(p) else first
        if isinstance(q, str) and q.strip():
            title = convo.task_title(q)
        else:
            title = convo.task_title(own)
        title = title or convo.headline(next((t["text"] for t in ts if t["text"].strip()), "")) or f"Row {i}"
        return title, convo.after_title(own, title)

    def card(self, row: dict[str, Any], i: int, roles: dict[str, Any]) -> dict[str, Any]:
        ts = _chat(row)
        title, brief = self._title(row, ts, roles, i)
        chips = [str(row[k]) for k in ("data_source", "ability", "env_class") if isinstance(row.get(k), str) and 0 < len(row[k]) <= 40][:3]
        ident = next((str(row[k]) for k in ("id", "uid", "instance_id") if isinstance(row.get(k), (str, int))), None)
        extra = parse_json(row.get("extra_info"))
        if ident is None and isinstance(extra, dict) and isinstance(extra.get("index"), (int, str)):
            ident = str(extra["index"])
        n_img = len(parse_json(row.get("images")) or []) if isinstance(parse_json(row.get("images")), list) else 0
        return {"i": i, "id": ident[:40] if ident else None, "title": title[:200], "snippet": brief[:260], "chips": chips,
                "stats": [(n_img, "images")] if n_img else []}

    def view(self, row: dict[str, Any], i: int, ds: Dataset, roles: dict[str, Any]) -> dict[str, Any]:
        skyrl = is_skyrl(list(row))
        ts = _chat(row)
        imgs = _images(row)
        if imgs:   # the images go with the first user turn, where its <image> placeholders are
            first = next((t for t in ts if t["kind"] == "message" and t["role"] == "user"), None)
            if first is not None:
                first["images"] = (first.get("images") or []) + imgs
        clean, gone = withhold({k: parse_json(v) for k, v in row.items() if k not in ("prompt", "images")}, hide=HIDE)
        clean, blobs = _blobs(clean)
        gone = sorted(set(gone + blobs))
        title, _ = self._title(row, ts, roles, i)
        ds_name = str(row.get("data_source") or "")
        sc = scorer(ds_name)
        rm = parse_json(row.get("reward_spec" if skyrl and row.get("reward_spec") is not None else "reward_model"))
        rm = rm if isinstance(rm, dict) else {}
        style = rm.get("method") if skyrl else rm.get("style")
        env_class = row.get("env_class") if isinstance(row.get("env_class"), str) else None
        extra = clean.get("extra_info") if isinstance(clean.get("extra_info"), dict) else {}
        agent_name = row.get("agent_name") if isinstance(row.get("agent_name"), str) else None
        tool_names = _tool_names(parse_json(row.get("extra_info")) or {})
        tools = convo.tools(row.get("tools")) if row.get("tools") is not None else []
        config, split = roles.get("_config", "default"), roles.get("_split", "train")

        if skyrl:
            how = (f"the `{env_class}` environment: {SKYRL_ENVS[env_class]}" if env_class in SKYRL_ENVS else
                   f"the `{env_class}` environment (not one skyrl-gym registers: the training code must)" if env_class else
                   "the environment set by `environment.env_class`")
        else:
            how = (f"verl's `{sc[0]}` scorer: {sc[1]}" if sc else
                   f"a custom reward function: `{ds_name or 'this data_source'}` isn't one of verl's built-in scorers")
        nxt = {"label": "Policy", "text": ("Replies" + (", calling tools" if tool_names or tools or agent_name == "tool_agent" else "")
                                           + f"; scored by {re.sub(r'`', '', how)}, against the withheld ground truth.")}
        blocks = [c.custom("rl", "transcript", {"turns": ts, "next": nxt}, convo.transcript_text(ts))] if ts else \
            [c.note("This row has no prompt.", "alert")]
        sections = [section("task", "The task", blocks, kind="blocks", note="prompt, as the policy gets it")]
        if tools:
            sections.append(section("tools", f"Tools ({len(tools)})", [c.custom("rl", "tools", {"tools": tools}, convo.tools_text(tools))], kind="blocks", note="tools"))
        elif tool_names:
            sections.append(section("tools", f"Tools ({len(tool_names)})", [
                c.kv([("Configured", ", ".join(f"`{t}`" for t in tool_names))]),
                c.note("verl defines a row's tools in the trainer's tool config (`actor_rollout_ref.rollout.multi_turn.tool_config_path`); the row "
                       "only passes each tool its `create_kwargs` (`extra_info.tools_kwargs`), whose ground truth is withheld.", "info")], kind="blocks",
                note="extra_info.tools_kwargs"))

        g: list[tuple[str, Any]] = []
        if ds_name:
            g.append(("Data source", f"`{ds_name}`"))
        if skyrl:
            g.append(("Environment", f"`{env_class}`" if env_class else "set by `environment.env_class`"))
        g.append(("Scored by", how))
        if style:
            g.append(("Reward style" if not skyrl else "Reward method", f"`{style}`"))
        if row.get("ability"):
            g.append(("Ability", f"`{row['ability']}`"))
        if agent_name:
            g.append(("Agent loop", f"`{agent_name}`" + (": multi-turn, with tools" if agent_name == "tool_agent" else "")))
        if gone:
            g.append(("Against", ", ".join(f"`{x}`" for x in gone) + ": withheld"))
        gblocks: list[dict[str, Any]] = [c.kv(g)]
        if not skyrl and not sc:
            gblocks.append(c.note("Training on these rows needs `reward.custom_reward_function.path` (a file with `compute_score(data_source, "
                                  "solution_str, ground_truth, extra_info)`): the dataset's own repository or card usually has it.", "info"))
        ex_rows = convo.field_rows({k: v for k, v in extra.items() if k not in ("tools_kwargs",)})
        if ex_rows:
            gblocks += [c.note("`extra_info`, passed whole to the reward function:", "list"), c.custom("rl", "fields", {"rows": ex_rows}, convo.fields_text(ex_rows))]
        sections.append(section("grading", "How it's graded", gblocks, kind="blocks", note="data_source and reward_model" if not skyrl else "env_class and reward_spec"))
        sections.append(section("run", "Run it with SkyRL" if skyrl else "Run it with verl",
                                run_blocks(ds, config, split, skyrl=skyrl, env_class=env_class, custom=not sc, tools=bool(tool_names or agent_name == "tool_agent")),
                                kind="blocks", note="it doesn't run in this app"))
        shown = {"data_source", "reward_model", "reward_spec", "env_class", "ability", "agent_name", "extra_info", "tools"}
        rest = {k: v for k, v in clean.items() if k not in shown and v not in (None, "", [], {})}
        if rest:
            rows = convo.field_rows(rest)
            sections.append(section("data", "The rest of the row", [c.custom("rl", "fields", {"rows": rows}, convo.fields_text(rows))], kind="blocks"))

        glance = [["Row", f"{i:,}"]]
        if ds_name:
            glance.append(["Data source", f"`{ds_name}`"])
        if row.get("ability"):
            glance.append(["Ability", str(row["ability"])])
        if skyrl:
            glance.append(["Environment", f"`{env_class}`" if env_class else "from the config"])
        glance.append(["Scorer", f"`{sc[0]}`" if sc and not skyrl else "the environment" if skyrl else "custom"])
        if agent_name:
            glance.append(["Agent loop", f"`{agent_name}`"])
        if imgs:
            glance.append(["Images", f"{len(imgs)}"])
        card = self.card(row, i, roles)
        fw = "SkyRL" if skyrl else "verl"
        return {"title": title[:200], "id": card["id"], "chips": card["chips"], "sections": sections, "withheld": gone, "glance": glance,
                "summary": f"Scored by {re.sub(r'`', '', how.split(':')[0])}",
                "framework_run": c.run_option(fw.lower(), fw, ok=False, endpoint=False, sandbox=False,
                                              why=f"{fw} trains on these rows on GPUs with its own reward functions; this app runs agents on single tasks, not training jobs",
                                              about=f"Train on it with {fw}: the command is under **Run it with {fw}** on this page."),
                "framework": fw}

    def overview(self, ds: Dataset, roles: dict[str, Any], stats: list[dict[str, Any]], config: str, split: str) -> list[dict[str, Any]]:
        rows = ds.sample[:20]
        skyrl = is_skyrl(ds.columns)
        freq = {st.get("column_name"): (st.get("column_statistics") or {}).get("frequencies") for st in stats or []}
        def dist(col: str) -> tuple[list[tuple[str, int]], int, bool]:
            f = freq.get(col)
            if isinstance(f, dict) and f:
                return sorted(((str(k), int(v)) for k, v in f.items()), key=lambda r: -r[1]), sum(f.values()), False
            vals = [str(r.get(col)) for r in rows if isinstance(r.get(col), str)]
            out: dict[str, int] = {}
            for v in vals:
                out[v] = out.get(v, 0) + 1
            return sorted(out.items(), key=lambda r: -r[1]), len(vals), True
        kv: list[tuple[str, Any]] = []
        sources, n, sampled = dist("data_source")
        if sources:
            routed = []
            for name, cnt in sources[:8]:
                sc = scorer(name)
                routed.append((f"`{name}`" + (f" → `{sc[0]}`" if sc and not skyrl else " → custom reward" if not skyrl else ""), cnt))
            kv.append(("Data sources", c.shares(routed, n, ("in the first rows" if sampled else "all rows") + ("" if skyrl else "; → the verl scorer each one gets"))))
        abil, n2, sampled2 = dist("ability")
        if abil and len(abil) > 1:
            kv.append(("Abilities", c.shares(abil[:8], n2, "in the first rows" if sampled2 else "all rows")))
        elif abil:
            kv.append(("Ability", f"`{abil[0][0]}`"))
        if skyrl:
            envs, n3, _ = dist("env_class")
            kv.append(("Environments", ", ".join(f"`{e}`" + ("" if e in SKYRL_ENVS else " (not in skyrl-gym)") for e, _ in envs[:6]) or "from `environment.env_class`"))
        styles = sorted({str((parse_json(r.get("reward_model")) or {}).get("style")) for r in rows if isinstance(parse_json(r.get("reward_model")), dict)} - {"None"})
        if styles:
            kv.append(("Reward style", ", ".join(f"`{s}`" for s in styles)))
        loops = sorted({r["agent_name"] for r in rows if isinstance(r.get("agent_name"), str)})
        if loops:
            kv.append(("Agent loop", ", ".join(f"`{x}`" for x in loops)))
        if "images" in ds.columns:
            kv.append(("Multimodal", "images with the prompt (`images`)"))
        if roles.get("answer"):
            kv.append(("Withheld", ", ".join(f"`{x}`" for x in roles["answer"])))
        out = [c.section("verl", "What a row asks", [c.kv(kv)], icon="target", note="the dataset viewer's statistics" if not sampled else "from the first rows")]
        num = [st for st in stats or [] if st.get("column_type") in ("float", "int") and re.search(r"pass_rate|difficulty|level|accuracy", str(st.get("column_name")), re.IGNORECASE)]
        if num:
            st = num[0]
            s = st.get("column_statistics") or {}
            h = s.get("histogram") or {}
            if h.get("hist") and len(h.get("bin_edges") or []) == len(h["hist"]) + 1:
                e = h["bin_edges"]
                out.append(c.section("difficulty", "How hard it is", [
                    c.kv([(st["column_name"].replace("_", " "), f"mean {s.get('mean') or 0:.2f}, median {s.get('median') or 0:.2f}")]),
                    c.shares([(f"{e[j]:.2g} to {e[j + 1]:.2g}", n) for j, n in enumerate(h["hist"])], sum(h["hist"]), f"rows by `{st['column_name']}`")],
                    icon="gauge", note="the dataset viewer's statistics"))
        custom = not skyrl and any(not scorer(name) for name, _ in sources[:8])
        tools = any(isinstance(parse_json(r.get("extra_info")), dict) and parse_json(r.get("extra_info")).get("tools_kwargs") for r in rows) or "tool_agent" in loops
        envs = sorted({r["env_class"] for r in rows if isinstance(r.get("env_class"), str)})
        out.append(c.section("run", "Run it with SkyRL" if skyrl else "Run it with verl",
                             run_blocks(ds, config, split, skyrl=skyrl, env_class=envs[0] if len(envs) == 1 else None, custom=custom, tools=tools),
                             icon="play", collapsed=True, note="a GRPO run"))
        return out


def run_blocks(ds: Dataset, config: str, split: str, *, skyrl: bool, env_class: str | None, custom: bool, tools: bool) -> list[dict[str, Any]]:
    """The command that trains on this dataset with verl (or SkyRL), on its own split's files."""
    spec = ds.spec
    splits = [s["split"] for s in ds.splits if s["config"] == config]
    val = next((s for s in ("validation", "test", "eval", "val") if s in splits and s != split), None)
    train_files = rowfiles.files_of(spec, ds.card, config, split)
    val_files = rowfiles.files_of(spec, ds.card, config, val) if val else []
    tr = rowfiles.local_paths(spec, train_files, f"**/*{split}*.parquet")
    va = rowfiles.local_paths(spec, val_files, f"**/*{val}*.parquet") if val else tr
    known = bool(train_files)
    fmt = (lambda xs: xs[0] if len(xs) == 1 else "[" + ",".join(xs) + "]")
    lines: list[str] = []
    if skyrl:
        lines += ["# SkyRL, from source (uv); it reads parquet or JSON Lines, or a Hub dataset as \"org/name:split\"",
                  "git clone https://github.com/NovaSky-AI/SkyRL.git && cd SkyRL", ""]
        train_arg = f"['{spec}:{split}']" if config == "default" else "[" + ",".join(f"'{p}'" for p in tr) + "]"
        val_arg = f"['{spec}:{val}']" if config == "default" and val else train_arg if not val else "[" + ",".join(f"'{p}'" for p in va) + "]"
        if config != "default":
            lines += [f"# the rows: the {config} subset's files", rowfiles.download(spec, train_files + val_files), ""]
        lines += ["# GRPO on it" + (f"; rows name their environment ({env_class})" if env_class else ""),
                  "uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \\",
                  f"    data.train_data=\"{train_arg}\" data.val_data=\"{val_arg}\" \\",
                  "    trainer.algorithm.advantage_estimator=grpo trainer.policy.model.path=Qwen/Qwen2.5-1.5B-Instruct \\",
                  "    trainer.strategy=fsdp trainer.placement.colocate_all=true \\",
                  "    generator.inference_engine.backend=vllm generator.n_samples_per_prompt=5 \\",
                  f"    environment.env_class={env_class or 'gsm8k'}"]
        blocks = [c.code("\n".join(lines), "run.sh")]
        if not env_class:
            blocks.append(c.note("`environment.env_class` is the skyrl-gym environment that scores the rows: pick the one they were written for.", "info"))
        blocks.append(c.links([("SkyRL on GitHub", SKYRL), ("Dataset format", SKYRL_DATA_DOC)]))
        return blocks
    lines += ["# verl, from source with uv (or pip install \"verl[vllm]\" and drop $UV_RUN)",
              "git clone https://github.com/verl-project/verl.git && cd verl",
              'UV_RUN="uv run --frozen --all-packages --extra vllm --extra fsdp"', "",
              "# the rows: verl reads local parquet / JSON / JSON Lines, not Hub ids" + ("" if known else " (the split's own files: check the dataset's file list)"),
              rowfiles.download(spec, train_files + val_files), "",
              "# GRPO on it" + ("; these data sources need your reward function (verl has no built-in scorer for them)" if custom else ""),
              "$UV_RUN python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo \\",
              f"    data.train_files={fmt(tr)} \\", f"    data.val_files={fmt(va)} \\",
              "    data.prompt_key=prompt data.max_prompt_length=1024 data.max_response_length=2048 \\",
              "    actor_rollout_ref.model.path=Qwen/Qwen3-8B actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=5 \\"]
    if tools:
        lines.append("    actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.tool_config_path=<your tool config> \\")
    if custom:
        lines.append("    reward.custom_reward_function.path=<your reward file> reward.custom_reward_function.name=compute_score \\")
    lines.append("    trainer.n_gpus_per_node=8 trainer.nnodes=1")
    blocks = [c.code("\n".join(lines), "run.sh")]
    blocks.append(c.links([("verl on GitHub", VERL), ("Quick start", VERL_QUICKSTART), ("Data format", VERL_DATA_DOC), ("Reward functions", VERL_REWARD_DOC)]))
    return blocks
