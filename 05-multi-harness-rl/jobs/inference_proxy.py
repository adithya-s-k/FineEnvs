"""Expose inference to native OpenCode sandboxes without exposing weight-update routes."""

import argparse
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse


def create_app(key, upstream="http://127.0.0.1:8000"):
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=660) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        if not secrets.compare_digest(
            request.headers.get("authorization", ""), "Bearer " + key
        ):
            raise HTTPException(401)
        if (request.method, path) not in {
            ("GET", "models"),
            ("POST", "chat/completions"),
        }:
            raise HTTPException(404)
        client = app.state.client
        message = client.build_request(
            request.method,
            upstream + "/v1/" + path,
            content=await request.body(),
            headers={"content-type": "application/json"},
        )
        response = await client.send(message, stream=True)
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            background=BackgroundTask(response.aclose),
        )

    return app


def main():
    from gradio.networking import setup_tunnel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    app = create_app(os.environ["SANDBOX_VLLM_KEY"], args.upstream)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("Inference proxy failed to start")
        time.sleep(0.1)
    url = setup_tunnel("127.0.0.1", args.port, secrets.token_urlsafe(24), None, None)
    Path(args.output).write_text(url)
    thread.join()


if __name__ == "__main__":
    main()
