"""Environments, whatever their format, behind one contract.

    contract.py   what every environment answers: tasks, environment, harness, rollouts, MCP (start here)
    registry.py   which adapter reads which source, and the entry points routes, runner and MCP use
    harbor.py     Harbor task folders, indexed (app/catalog.py does the indexing)
    rows.py       tasks that are dataset rows, read through the Hub's dataset viewer or the files themselves
    processors.py the row readers rows.py picks from: packed Harbor, MiMo, NeMo Gym, verl, traces, Verifiers, generic

Spaces (live servers) have their own module, app/spaces_live.py, and MCP bridge, app/mcp_bridge.py.
"""

from . import contract, direct, registry, rows, viewer  # noqa: F401
from .direct import DirectError  # noqa: F401
from .rows import (  # noqa: F401  (the rows machinery, used by tests and the old routes)
    LOCAL_MAX, PAGE, Files, Viewer, _all_rows, _backend, _broken, _choice, _dataset, _where, describe, materialize, packed, pick,
    rows as list_rows, task, task_file,
)
from .viewer import ViewerError  # noqa: F401

rows_list = list_rows
