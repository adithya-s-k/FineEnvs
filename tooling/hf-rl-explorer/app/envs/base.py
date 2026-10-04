"""How a dataset that isn't a folder of Harbor tasks is read: a *processor* turns its rows into tasks people can browse.

A processor says whether it can read a dataset (`match`), how to show one row in a list (`card`), and how to show it
in full (`view`). The explorer picks the processor with the highest match, so a specific one (MiMo, NeMo Gym, verl,
packed Harbor tasks) wins over the generic reader, which infers each column's role from its name and its values.

Answers never leave the server: a row's answer-like fields (by name: answer, solution, ground_truth, expected, ...)
are left out of everything a processor returns, and named in `withheld` instead. Write a new processor in
`app/envs/processors.py` (see CUSTOM_ENVS.md for a walk-through).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

ANSWER = re.compile(r"(^|[_.\-])(answers?|solutions?|ground_?truths?|gold\w*|targets?|expected\w*|references?|oracle|correct\w*|"
                    r"canonical\w*|labels?|gt|pass_anchor|final_answer)($|[_.\-])", re.I)
GRADING = re.compile(r"(reward|verif|rubric|grade|grader|scor|metric|atol|rtol|toleran|tests?$|test_|check|judge|instruction_id|"
                     r"^kwargs$|style|eval)", re.I)
ENVIRON = re.compile(r"(docker|image|environment|^env$|setup|^files$|repo|commit|cpu|mem|disk|gpu|timeout|sandbox|tools?$|bucket)", re.I)
TASK_KEYS = ("instruction", "prompt", "question", "problem", "task", "query", "input", "description", "statement", "messages",
             "conversation", "conversations", "text", "content")
ID_KEYS = ("task_id", "instance_id", "id", "uid", "uuid", "problem_id", "question_id", "label", "name", "key", "slug", "path", "idx", "index")
TITLE_KEYS = ("title", "task_name", "name", "summary", "instruction_summary", "short_description", "heading")
FACET_SKIP = re.compile(r"(^id$|_id$|uuid|path|url|license|hash|sha|date|time)", re.I)


@dataclass
class Dataset:
    """What a processor sees of a dataset: its id, tags and card, its configs and splits, and a few rows."""
    spec: str
    tags: list[str]
    card: dict[str, Any]
    splits: list[dict[str, str]]
    features: list[dict[str, Any]] = field(default_factory=list)   # the viewer's: [{name, type}]
    sample: list[dict[str, Any]] = field(default_factory=list)      # the first rows of the first split

    @property
    def columns(self) -> list[str]:
        return [f["name"] for f in self.features]

    def has(self, *names: str) -> bool:
        cols = {c.lower() for c in self.columns}
        return all(n.lower() in cols for n in names)


def is_messages(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in v[:20])


def parse_json(v: Any) -> Any:
    """The viewer gives JSON columns as strings sometimes: a string that is a JSON object or list, decoded."""
    if isinstance(v, str) and v[:1] in "{[" and len(v) < 4_000_000:
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def withhold(v: Any, path: str = "", depth: int = 0, keep: tuple[str, ...] = (), hide: frozenset[str] | set[str] = frozenset()) -> tuple[Any, list[str]]:
    """A value with its answer-like fields left out (recursively), and their dotted names. Fields are answer-like by
    name (ANSWER), or because a format says so: `hide` holds dotted names ("expected_action", "verifier.patterns")
    that a reader knows hold its answers though their names don't say it."""
    gone: list[str] = []
    if depth > 8:
        return v, gone
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            name = f"{path}.{k}" if path else str(k)
            if name in hide or (isinstance(k, str) and ANSWER.search(k) and name not in keep):
                gone.append(name)
                continue
            out[k], g = withhold(parse_json(x), name, depth + 1, keep, hide)
            gone += g
        return out, gone
    if isinstance(v, list):
        out = []
        for x in v[:2000]:
            y, g = withhold(x, path, depth + 1, keep, hide)
            out.append(y)
            gone += g
        return out, sorted(set(gone))
    return v, gone


def retitle(cards: list[dict[str, Any]]) -> None:
    """When a page of tasks all open the same way (a shared preamble), title each by what comes after it."""
    texts = [c.get("_text") or "" for c in cards]
    if len(cards) >= 3 and all(texts) and len({c["title"] for c in cards}) < len(cards) * 0.6:
        p = _common_prefix(texts) or os.path.commonprefix(texts)
        cut = p.rfind("\n") + 1 if "\n" in p else 0
        for c, t in zip(cards, texts):
            rest = t[cut:].lstrip() if cut else t
            title = first_line(rest)
            if title:
                c["title"] = title[:200]
                body = re.sub(r"\s+", " ", rest).strip()
                c["snippet"] = body[body.find(title.rstrip("…")[:40]) + len(title):][:260].strip() if title.rstrip("…")[:40] in body else body[:260]
    for c in cards:
        c.pop("_text", None)


def _common_prefix(texts: list[str]) -> str:
    """The opening all these texts share, cut back to a paragraph or line break; "" when it's short."""
    import os

    p = os.path.commonprefix([t for t in texts if t])
    if len(p) < 40:
        return ""
    cut = max(p.rfind("\n\n"), p.rfind("\n"))
    return p[:cut + 1] if cut > 30 else ""


def first_line(text: str, n: int = 140) -> str:
    for line in str(text or "").splitlines():
        line = re.sub(r"^[#>*\-\s`]+", "", line).strip()
        if len(line) > 3:
            return line[:n] + ("…" if len(line) > n else "")
    return ""


def text_of(v: Any) -> str:
    """The readable text of a task field: a string, or a chat's user turns."""
    v = parse_json(v)
    if isinstance(v, str):
        return v
    if is_messages(v):
        return "\n\n".join(m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"))[:4000]
                           for m in v if m.get("role") in ("user", "human") or len(v) == 1)
    if isinstance(v, dict):
        for k in ("input", "messages", "prompt", "text", "content"):
            if k in v:
                return text_of(v[k])
    return ""


class Processor:
    id = "generic"
    name = "Rows"
    about = "Each row is a task. Which column holds the task, its grading and its answer is inferred from names and values."
    framework = "Rows"          # the label on cards and the page header

    def match(self, ds: Dataset) -> float:
        return 0.1

    # the columns' roles, worked out once per dataset from its features and first rows
    def roles(self, ds: Dataset) -> dict[str, Any]:
        cols = ds.columns
        low = {c.lower(): c for c in cols}
        sample = ds.sample[:20]

        def filled(c: str) -> bool:
            return any(parse_json(r.get(c)) not in (None, "", [], {}) for r in sample)

        task = next((low[k] for k in TASK_KEYS if k in low and filled(low[k])), None)
        if task is None:   # the longest text column
            texts = sorted(((sum(len(str(r.get(c) or "")) for r in sample), c) for c in cols
                            if any(isinstance(r.get(c), str) for r in sample) and not ANSWER.search(c)), reverse=True)
            task = texts[0][1] if texts and texts[0][0] > 40 else None
        ident = next((low[k] for k in ID_KEYS if k in low and low[k] != task and all(isinstance(r.get(low[k]), (str, int)) for r in sample if r.get(low[k]) is not None)
                      and len({str(r.get(low[k])) for r in sample}) == len(sample)), None)
        title = next((low[k] for k in TITLE_KEYS if k in low and low[k] not in (task, ident) and filled(low[k])), None)
        # every task opening with the same preamble makes a useless title: then a column that varies, and short, names it
        texts = [text_of(r.get(task)) for r in sample] if task else []
        prefix = _common_prefix(texts) if len(texts) >= 3 else ""
        if not title and texts and len({first_line(t) for t in texts}) < max(2, len(texts) * 0.6):
            for k in TASK_KEYS + TITLE_KEYS:
                c = low.get(k)
                if not c or c == task or ANSWER.search(c):
                    continue
                vals = [str(r.get(c) or "") for r in sample]
                if all(isinstance(r.get(c), str) for r in sample if r.get(c) is not None) and len(set(vals)) >= len(vals) * 0.6 \
                        and sum(map(len, vals)) / max(1, len(vals)) < 400:
                    title = c
                    break
        answer = [c for c in cols if ANSWER.search(c) and c != ident]
        grading = [c for c in cols if c not in answer and c not in (task, ident, title) and GRADING.search(c)]
        environ = [c for c in cols if c not in answer + grading and c not in (task, ident, title) and ENVIRON.search(c)]
        return {"task": task, "id": ident, "title": title, "answer": answer, "grading": grading, "environment": environ,
                "_prefix": prefix}

    def overview(self, ds: Dataset, roles: dict[str, Any], stats: list[dict[str, Any]], config: str, split: str) -> list[dict[str, Any]]:
        """Sections for the dataset's own page (contract.section): what a row asks, how it's graded, how to run it with
        its framework. Built from the first rows (`ds.sample`) and the viewer's column statistics (`stats`, [] when the
        rows come from files): no more reading than the page already did."""
        return []

    def card(self, row: dict[str, Any], i: int, roles: dict[str, Any]) -> dict[str, Any]:
        """One row in a list: its id, a title, a line of its task, a few short facts."""
        task = text_of(row.get(roles["task"])) if roles.get("task") else ""
        if roles.get("_prefix") and task.startswith(roles["_prefix"]):   # the preamble every task shares says nothing here
            task = task[len(roles["_prefix"]):].lstrip()
        ident = row.get(roles["id"]) if roles.get("id") else None
        title = str(row.get(roles["title"]) or "") if roles.get("title") else ""
        title = title or first_line(task) or (str(ident) if ident is not None else f"Row {i}")
        chips = []
        for c, v in row.items():
            if c in (roles.get("task"), roles.get("id"), roles.get("title")) or c in roles.get("answer", []) or FACET_SKIP.search(c):
                continue
            if isinstance(v, str) and 0 < len(v) <= 24 and "\n" not in v:
                chips.append(f"{v}")
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and c.lower() in ("difficulty", "level", "difficulty_level"):
                chips.append(f"{c} {v}")
            if len(chips) >= 4:
                break
        # titled by a column of its own (a question beside a long instruction): the instruction adds only boilerplate
        snippet = "" if roles.get("title") else task if task and first_line(task) != title else ""
        return {"i": i, "id": str(ident) if ident is not None else None, "title": title[:200],
                "snippet": re.sub(r"\s+", " ", snippet)[:260], "chips": chips, "_text": task if not roles.get("title") else ""}

    def view(self, row: dict[str, Any], i: int, ds: Dataset, roles: dict[str, Any]) -> dict[str, Any]:
        """One row in full, as sections the page renders: the task, how it's graded, what it runs in, the rest."""
        keep = tuple(c for c in (roles.get("id"), roles.get("title")) if c)
        clean, gone = withhold({k: parse_json(v) for k, v in row.items()}, keep=keep)
        sections = []
        t = roles.get("task")
        if t and t in clean:
            v = clean[t]
            sections.append(section("task", "The task", v, note=f"from {t}"))
        grading = {c: clean[c] for c in roles.get("grading", []) if c in clean}
        if grading:
            sections.append(section("grading", "How it's graded", grading, kind="value", note="the row's grading fields"))
        env = {c: clean[c] for c in roles.get("environment", []) if c in clean}
        if env:
            sections.append(section("environment", "What it runs in", env, kind="value"))
        rest = {k: v for k, v in clean.items() if k not in (t, *grading, *env)}
        if rest:
            sections.append(section("data", "Everything else in the row", rest, kind="value"))
        card = self.card(row, i, roles)
        return {"title": card["title"], "id": card["id"], "chips": card["chips"], "sections": sections, "withheld": gone,
                "glance": [["Row", f"{i:,}"], *([["Id", card["id"]]] if card["id"] else []),
                           ["Task from", t or "–"], ["Graded by", ", ".join(grading) or "not in the row"]]}


def section(id: str, title: str, value: Any, kind: str | None = None, note: str = "") -> dict[str, Any]:
    """A view section: `kind` says how the page draws `body` (markdown, messages, code, value, kv, files, or blocks: a
    list of contract blocks, for a reader that builds its own)."""
    if kind is None:
        v = parse_json(value)
        kind = "messages" if is_messages(v) else "markdown" if isinstance(v, str) else "value"
        value = v
    return {"id": id, "title": title, "kind": kind, "body": value, "note": note}
