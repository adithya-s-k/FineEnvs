#!/usr/bin/env python3
"""Place openai/math's reference proof as a submission (used by the oracle's solve.sh).

Standalone: the adapter copies this file into every task's solution/ directory.

    reference.py <openai-math lean dir> <closure.txt> <solution module> <workspace>

Each module in closure.txt (``OAI.A.B``) is copied from ``<lean>/OAI/A/B.lean`` to
``<workspace>/Submission/OAI/A/B.lean`` with ``import OAI.X`` rewritten to
``import Submission.OAI.X``; ``Submission.lean`` imports the solution module. Declaration names
are namespaces, not module paths, so the theorems keep the names the challenge expects.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_IMPORT = re.compile(r"^(\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s+(?:all\s+)?)(OAI(?:\.|\s|$))")
_HEADER_LINE = re.compile(r"^\s*(?:module\b|prelude\b|--|$)")


def rename_imports(text: str) -> str:
    """Rewrite ``import OAI...`` to ``import Submission.OAI...`` in the import header only."""
    out, in_header, in_comment = [], True, False
    for line in text.splitlines(keepends=True):
        if in_header:
            stripped = line.strip()
            if in_comment:
                in_comment = "-/" not in stripped
            elif stripped.startswith("/-"):
                in_comment = "-/" not in stripped[2:]
            elif _IMPORT.match(line):
                line = _IMPORT.sub(lambda m: m.group(1) + "Submission." + m.group(2), line, count=1)
            elif not _HEADER_LINE.match(line) and not re.match(r"^\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s", line):
                in_header = False
        out.append(line)
    return "".join(out)


def module_path(root: Path, module: str) -> Path:
    return root.joinpath(*module.split(".")).with_suffix(".lean")


def place(lean_dir: Path, modules: list[str], solution_module: str, workspace: Path) -> None:
    for module in modules:
        dst = module_path(workspace / "Submission", module)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(rename_imports(module_path(lean_dir, module).read_text(encoding="utf-8")), encoding="utf-8")
    (workspace / "Submission.lean").write_text(f"import Submission.{solution_module}\n", encoding="utf-8")


if __name__ == "__main__":
    lean_dir, closure_txt, solution_module, workspace = sys.argv[1:5]
    modules = [m.strip() for m in Path(closure_txt).read_text().splitlines() if m.strip()]
    place(Path(lean_dir), modules, solution_module, Path(workspace))
    print(f"placed {len(modules)} reference modules for {solution_module}")
