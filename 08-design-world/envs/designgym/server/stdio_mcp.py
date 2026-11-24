"""A stdio MCP server, for pointing Claude (or any MCP client) at the environment.

    claude mcp add designgym -- uvx --from designgym designgym mcp

The client gets the same tools a trainer sees, starting with `start_task`. The
live editor is served on a local port alongside, so every call can be watched
as it happens: open the `editor_url` that `start_task` returns.

stdout belongs to the MCP protocol here, so the editor server logs nothing to it.
"""

from __future__ import annotations

import logging
import os
import sys
import threading


def _serve_editor(port: int) -> None:
    import uvicorn

    from .app import create_server

    config = uvicorn.Config(create_server(), host="127.0.0.1", port=port, log_level="warning",
                            access_log=False, log_config=None)
    uvicorn.Server(config).run()


def run(editor_port: int = 8007) -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    from ..core.pack import load_pack
    from .environment import DesignCanvasEnvironment

    if editor_port:
        os.environ.setdefault("PORT", str(editor_port))
        os.environ.setdefault("DESIGN_CANVAS_PUBLIC_URL", f"http://127.0.0.1:{editor_port}")
        threading.Thread(target=_serve_editor, args=(editor_port,), daemon=True).start()
    env = DesignCanvasEnvironment(load_pack())
    env.mcp_server.run(transport="stdio", show_banner=False)
