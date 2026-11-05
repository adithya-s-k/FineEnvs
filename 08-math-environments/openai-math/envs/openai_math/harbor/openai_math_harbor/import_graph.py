"""Build the module import graph of the openai/math Lean library.

Reads every ``.lean`` file under ``<repo>/lean`` (OAI library and ComparatorChallenges) and writes one
JSON object: ``{"modules": {module: {"imports": [...], "bytes": n, "lines": n}}}``. Only the import
header is parsed; Lean requires imports to come before any other command.

    python -m openai_math_harbor.import_graph /path/to/openai-math/lean graph.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# `import X`, plus the module-system forms `public import X`, `meta import X`, `import all X`.
_IMPORT = re.compile(r"^\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s+(?:all\s+)?([A-Za-z_«][^\s]*)")
_HEADER_ONLY = re.compile(r"^\s*(?:module\b|prelude\b|--|/-|$)")


def header_imports(text: str) -> list[str]:
    imports, in_comment = [], False
    for line in text.splitlines():
        if in_comment:
            in_comment = "-/" not in line
            continue
        if line.lstrip().startswith("/-") and "-/" not in line:
            in_comment = True
            continue
        m = _IMPORT.match(line)
        if m:
            imports.append(m.group(1).replace("«", "").replace("»", ""))
        elif not _HEADER_ONLY.match(line):
            break
    return imports


def build(lean_root: Path) -> dict:
    modules = {}
    for path in sorted(lean_root.rglob("*.lean")):
        rel = path.relative_to(lean_root)
        if rel.parts[0] in (".lake", "lakefile.lean") or rel.name == "lakefile.lean":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        name = ".".join(rel.with_suffix("").parts)
        modules[name] = {"imports": header_imports(text), "bytes": len(text.encode()), "lines": text.count("\n") + 1}
    return {"modules": modules}


if __name__ == "__main__":
    graph = build(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(graph, separators=(",", ":")))
    print(len(graph["modules"]), "modules")
