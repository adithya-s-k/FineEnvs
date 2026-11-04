"""Read openai/math's Comparator challenges and decide how each one is graded.

Each challenge is ``lean/ComparatorChallenges/<Name>.lean`` (the statement, with ``sorry``) plus
``<Name>.json`` (Comparator's config). Two things are added here:

* the family it belongs to (number, title, subject), from ``lean/docs/NNN.md`` and ``overview.tex``;
* a hole policy. Comparator checks only the name and type of a ``definition_names`` entry, never
  its body (comparator ``Compare.lean``: ``definitionHoleMatches``). openai/math lists fixed
  definitions as holes in 9 configs, so a solver could redefine e.g. ``ProperColoring := False`` and
  prove the theorem trivially. Holes are therefore rewritten:

  - a definition with a complete body is removed from ``definition_names``, so Comparator requires
    the solution's definition to be identical;
  - a definition whose whole body is ``sorry`` and whose name appears nowhere else in the file stays
    a hole: its type is the claim (e.g. a witness structure carrying the proof obligations);
  - anything else (a ``sorry`` inside a body, or an open definition that statements depend on) cannot
    be graded safely as written, and the challenge is excluded until it is restated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_DECL_START = re.compile(
    r"(?m)^[ \t]*(?:@\[[^\]]*\][ \t]*\n?[ \t]*)?(?:(?:noncomputable|private|protected|partial|unsafe)[ \t]+)*"
    r"(?:def|abbrev|theorem|lemma|instance|structure|class|inductive|opaque|axiom|example)\b"
)
_BLOCK_END = re.compile(
    r"(?m)^[ \t]*(?:end\b|namespace\b|section\b|run_cmd\b|open\b|variable\b|set_option\b|/--|#)"
)


@dataclass
class Family:
    number: str
    title: str
    subject: str
    summary: str


@dataclass
class Challenge:
    name: str
    lean: str
    config: dict
    family: Family | None
    theorem_names: list[str]
    definition_holes: list[str]
    dropped_holes: list[str] = field(default_factory=list)
    excluded: str | None = None

    @property
    def comparator(self) -> dict:
        return {
            "theorem_names": self.theorem_names,
            "definition_names": self.definition_holes,
            "permitted_axioms": self.config["permitted_axioms"],
        }


def _declaration_block(source: str, short_name: str) -> str | None:
    """Text of the first ``def``-like declaration named ``short_name``, up to the next one."""
    pattern = re.compile(
        r"(?m)^[ \t]*(?:@\[[^\]]*\][ \t]*\n?[ \t]*)?(?:(?:noncomputable|private|protected)[ \t]+)*"
        r"(?:def|abbrev|structure|class|inductive|opaque|instance)[ \t]+" + re.escape(short_name) + r"\b"
    )
    m = pattern.search(source)
    if not m:
        return None
    rest = source[m.end():]
    ends = [x.start() for x in (_DECL_START.search(rest), _BLOCK_END.search(rest)) if x]
    return source[m.start(): m.end() + (min(ends) if ends else len(rest))]


def _mentions(source: str, short_name: str) -> int:
    """Occurrences of ``short_name`` as a whole identifier anywhere in the file (comments included).

    Qualified uses count too (``RowColumn.Foo`` mentions ``Foo``): theorems often refer to a
    definition through its namespace, and missing one would keep an unsafe hole open.
    """
    return len(re.findall(r"(?<![\w'])" + re.escape(short_name) + r"(?![\w'])", source))


def hole_policy(source: str, config: dict) -> tuple[list[str], list[str], str | None]:
    """Return (kept holes, dropped holes, exclusion reason) for one challenge."""
    kept, dropped, problems = [], [], []
    for name in config.get("definition_names") or []:
        short = name.split(".")[-1]
        block = _declaration_block(source, short)
        if block is None:
            problems.append(f"{name}: definition not found in the challenge")
            continue
        body = block.split(":=", 1)[1] if ":=" in block else ""
        has_sorry = re.search(r"\bsorry\b", block) is not None
        if not has_sorry:
            dropped.append(name)
        elif re.fullmatch(r"\s*(?:by\s+)?sorry\s*", body) and _mentions(source, short) == 1:
            # The hole's type is the claim and nothing else in the file refers to it.
            kept.append(name)
        else:
            problems.append(f"{name}: open definition the theorem statements depend on")
    return kept, dropped, ("; ".join(problems) or None)


def load_families(overview_tex: Path) -> dict[str, Family]:
    src = overview_tex.read_text(encoding="utf-8")

    def group(i: int) -> tuple[str, int]:
        depth, j = 0, i
        while True:
            c = src[j]
            if c == "\\":
                j += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return src[i + 1: j], j + 1
            j += 1

    families, subject = {}, ""
    for m in re.finditer(r"\\(cataloguesection|resultentry)\{", src):
        if src[max(0, m.start() - 12): m.start()].endswith("newcommand{"):
            continue
        i = m.end() - 1
        if m.group(1) == "cataloguesection":
            subject, _ = group(i)
            continue
        args = []
        for _ in range(3):
            arg, i = group(i)
            args.append(arg)
            while src[i] in " \n%":
                i += 1
        families[args[0]] = Family(args[0], _plain(args[1]), subject, _plain(args[2]))
    return families


def _plain(tex: str) -> str:
    """Light LaTeX cleanup for markdown: keep $math$, drop accents and \\-escapes."""
    tex = re.sub(r"\\[\"'`^~]\{?([A-Za-z])\}?", r"\1", tex)
    tex = tex.replace("--", "–").replace("``", '"').replace("''", '"').replace("~", " ")
    tex = re.sub(r"\\(?:emph|textit|textbf)\{([^{}]*)\}", r"\1", tex)
    return re.sub(r"\s+", " ", tex).strip()


def family_links(docs_dir: Path) -> dict[str, str]:
    """Challenge name -> family number, from the Comparator tables in lean/docs/NNN.md."""
    links = {}
    for doc in sorted(docs_dir.glob("*.md")):
        for m in re.finditer(r"\(\.\./ComparatorChallenges/([A-Za-z0-9_]+)\.lean\)", doc.read_text()):
            links.setdefault(m.group(1), doc.stem)
    return links


def load_challenges(openai_math: Path) -> list[Challenge]:
    lean = openai_math / "lean"
    families = load_families(openai_math / "overview.tex")
    links = family_links(lean / "docs")
    out = []
    for cfg_path in sorted((lean / "ComparatorChallenges").glob("*.json")):
        name = cfg_path.stem
        config = json.loads(cfg_path.read_text())
        source = (lean / "ComparatorChallenges" / f"{name}.lean").read_text(encoding="utf-8")
        kept, dropped, excluded = hole_policy(source, config)
        if not config["theorem_names"] and not kept:
            excluded = excluded or "no theorem and no open definition to check"
        out.append(Challenge(
            name=name, lean=source, config=config, family=families.get(links.get(name, "")),
            theorem_names=list(config["theorem_names"]), definition_holes=kept,
            dropped_holes=dropped, excluded=excluded,
        ))
    return out
