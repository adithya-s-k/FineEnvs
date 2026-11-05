"""Reference-proof closures for the oracle agent.

A challenge's reference proof is ``solution_module`` plus every ``OAI.*`` module it imports,
transitively (from the import graph built by ``import_graph.py``). The oracle submits exactly those
files under the ``Submission`` root agents use (see ``reference.py``), so the verifier grades the
reference proof the same way it grades an agent.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path

from .reference import place

CORE_ROOTS = {"Init", "Lean", "Std", "Lake", "Mathlib"}


def closure(graph: dict, root: str) -> list[str]:
    """Every OAI module reachable from ``root`` through imports, sorted."""
    seen, stack = set(), [root]
    while stack:
        module = stack.pop()
        if module in seen or module not in graph:
            continue
        seen.add(module)
        stack.extend(i for i in graph[module]["imports"] if i == "OAI" or i.startswith("OAI."))
    return sorted(seen)


def external_packages(graph: dict, modules: list[str]) -> list[str]:
    """Import roots outside OAI, Mathlib and the Lean core that the closure needs."""
    roots = {i.split(".")[0] for m in modules for i in graph[m]["imports"] if i not in graph}
    return sorted(roots - CORE_ROOTS)


def bundle(lean_dir: Path, modules: list[str], solution_module: str) -> bytes:
    """A .tar.gz of the placed reference submission (Submission.lean + Submission/OAI/**)."""
    with tempfile.TemporaryDirectory() as tmp:
        place(lean_dir, modules, solution_module, Path(tmp))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path in sorted(Path(tmp).rglob("*.lean")):
                tar.add(path, arcname=str(path.relative_to(tmp)))
        return buf.getvalue()
