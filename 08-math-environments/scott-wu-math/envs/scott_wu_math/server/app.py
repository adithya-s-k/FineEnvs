"""FastAPI app: the OpenEnv API (/reset, /step, /state, /ws, /schema, task routes) plus the play page at /.

    uvicorn scott_wu_math.server.app:app --port 8012
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app

from ..models import CardAction, CardObservation, CardState
from .environment import InvalidReset, ScottWuMathEnvironment

WEB = Path(__file__).resolve().parent / "web"
README = Path(__file__).resolve().parents[1] / "README.md"

# The play page replaces OpenEnv's generic Gradio playground.
os.environ["ENABLE_WEB_INTERFACE"] = "false"

app = create_app(
    ScottWuMathEnvironment,
    CardAction,
    CardObservation,
    env_name="scott_wu_math",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "256")),
    state_cls=CardState,
)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.exception_handler(InvalidReset)
async def invalid_reset(request: Request, exc: InvalidReset) -> JSONResponse:
    """A reset the server cannot deal (index out of range, target too big, ...) is the client's error."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def play_page() -> FileResponse:
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/readme.md", include_in_schema=False)
def readme() -> FileResponse:
    """The environment card, rendered by /static/card.html."""
    return FileResponse(README, media_type="text/markdown; charset=utf-8", headers={"Cache-Control": "no-cache"})


@app.get("/web", include_in_schema=False)
def web_alias() -> RedirectResponse:
    return RedirectResponse("/")


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port or int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
