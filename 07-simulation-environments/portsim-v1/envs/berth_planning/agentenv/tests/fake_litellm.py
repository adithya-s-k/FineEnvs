"""A stand-in for agent-env's LiteLLM proxy on the three routes portsim-llm and upstream's harness call: request i
gets scripted turn i, in each provider's wire format, streamed when the request asks for a stream."""

import hashlib
import json
import threading
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

CHUNK = 4000


def digest(body: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in body.items() if k not in ("model", "stream")}, sort_keys=True,
                                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def hf_turns(episode: dict) -> list[dict]:
    """A published episode's assistant messages as turns, its recorded token totals split evenly across them."""
    messages = [m for m in json.loads(episode["messages"]) if m["role"] == "assistant"]
    n = len(messages)

    def share(total: int, i: int) -> int:
        return total // n + (total % n if i == n - 1 else 0)

    return [{"content": m["content"], "reasoning": m["reasoning"], "stop": m["stop"], "tool_calls": m["tool_calls"],
             "usage": {"input": share(episode["input_tokens"], i), "output": share(episode["output_tokens"], i),
                       "cached": 0, "cache_write": 0}}
            for i, m in enumerate(messages)]


def _sse(events: list[dict], named: bool) -> Response:
    body = "".join((f"event: {e['type']}\n" if named else "") + f"data: {json.dumps(e)}\n\n" for e in events)
    return Response(body, media_type="text/event-stream")


def _usage(turn: dict) -> dict:
    return {"input": 0, "output": 0, "cached": 0, "cache_write": 0, **turn.get("usage", {})}


def messages(i: int, turn: dict, body: dict) -> Response:
    u = _usage(turn)
    blocks = []
    if turn.get("reasoning"):
        blocks.append(({"type": "thinking", "thinking": "", "signature": ""},
                       [{"type": "thinking_delta", "thinking": turn["reasoning"]},
                        {"type": "signature_delta", "signature": f"sig-{i}"}]))
    if turn.get("content"):
        blocks.append(({"type": "text", "text": ""}, [{"type": "text_delta", "text": turn["content"]}]))
    for call in turn.get("tool_calls", []):
        blocks.append(({"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}},
                       [{"type": "input_json_delta", "partial_json": call["arguments"]}]))
    events = [{"type": "message_start", "message": {
        "id": f"msg_{i}", "type": "message", "role": "assistant", "model": body["model"], "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": u["input"] - u["cached"] - u["cache_write"], "output_tokens": 0,
                  "cache_read_input_tokens": u["cached"], "cache_creation_input_tokens": u["cache_write"]}}}]
    for index, (start, deltas) in enumerate(blocks):
        events += [{"type": "content_block_start", "index": index, "content_block": start},
                   *({"type": "content_block_delta", "index": index, "delta": d} for d in deltas),
                   {"type": "content_block_stop", "index": index}]
    events += [{"type": "message_delta", "delta": {"stop_reason": turn["stop"], "stop_sequence": None},
                "usage": {"output_tokens": u["output"]}},
               {"type": "message_stop"}]
    return _sse(events, named=True)


def responses(i: int, turn: dict, body: dict) -> Response:
    u = _usage(turn)
    output = []
    if turn.get("reasoning"):
        output.append({"type": "reasoning", "id": f"rs_{i}", "encrypted_content": f"enc-{i}",
                       "summary": [{"type": "summary_text", "text": turn["reasoning"]}]})
    if turn.get("content"):
        output.append({"type": "message", "id": f"msg_{i}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": turn["content"], "annotations": []}]})
    for j, call in enumerate(turn.get("tool_calls", [])):
        output.append({"type": "function_call", "id": f"fc_{i}_{j}", "call_id": call["id"], "name": call["name"],
                       "arguments": call["arguments"], "status": "completed"})
    response = {"id": f"resp_{i}", "object": "response", "created_at": 0, "model": body["model"],
                "status": turn["stop"], "output": output,
                "usage": {"input_tokens": u["input"], "input_tokens_details": {"cached_tokens": u["cached"]},
                          "output_tokens": u["output"], "output_tokens_details": {"reasoning_tokens": 0},
                          "total_tokens": u["input"] + u["output"]}}
    if not body.get("stream"):
        return JSONResponse(response)
    end = "response.incomplete" if turn["stop"] == "incomplete" else "response.completed"
    return _sse([{"type": "response.created", "sequence_number": 0, "response": {**response, "status": "in_progress",
                                                                                   "output": []}},
                 {"type": end, "sequence_number": 1, "response": response}], named=True)


def chat(i: int, turn: dict, body: dict) -> Response:
    u = _usage(turn)
    reasoning, content = turn.get("reasoning") or "", turn.get("content") or ""
    deltas = [{"role": "assistant", "reasoning_content": reasoning[k:k + CHUNK]}
              for k in range(0, len(reasoning), CHUNK)]
    deltas += [{"content": content[k:k + CHUNK]} for k in range(0, len(content), CHUNK)]
    deltas += [{"tool_calls": [{"index": j, "id": call["id"], "type": "function",
                                "function": {"name": call["name"], "arguments": call["arguments"]}}]}
               for j, call in enumerate(turn.get("tool_calls", []))]

    def chunk(choices: list, **extra) -> dict:
        return {"id": f"chatcmpl-{i}", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                "choices": choices, **extra}

    events = [chunk([{"index": 0, "delta": d, "finish_reason": None}]) for d in deltas]
    events += [chunk([{"index": 0, "delta": {}, "finish_reason": turn["stop"]}]),
               chunk([], usage={"prompt_tokens": u["input"], "completion_tokens": u["output"],
                                "total_tokens": u["input"] + u["output"],
                                "prompt_tokens_details": {"cached_tokens": u["cached"]}})]
    return Response("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n",
                    media_type="text/event-stream")


class FakeLiteLLM:
    """Runs in a thread on a free port: `.url`, and `.requests` as (path, headers, body). A turn {"status": 500}
    answers with that error, and {"stream_error": "..."} with a 200 stream that fails, as a proxy reports a provider
    cutting a reply; past the last turn every request gets a 500. Errors ask for a 1 ms retry delay, so the SDKs'
    retries run at once."""

    def __init__(self, turns: list[dict], host: str = "127.0.0.1"):
        self.turns, self.host, self.requests = turns, host, []
        routes = [Route(f"/v1/{name}", self._route(build), methods=["POST"])
                  for name, build in [("messages", messages), ("responses", responses), ("chat/completions", chat)]]
        self.server = uvicorn.Server(uvicorn.Config(Starlette(routes=routes), host=host, port=0, log_level="warning"))

    def _route(self, build):
        async def endpoint(request: Request) -> Response:
            body = await request.json()
            i = len(self.requests)
            self.requests.append((request.url.path, dict(request.headers), body))
            turn = self.turns[i] if i < len(self.turns) else {"status": 500, "message": "no scripted turn"}
            if "status" in turn:
                return JSONResponse({"error": {"message": turn.get("message", "scripted error"), "type": "fake"}},
                                    status_code=turn["status"], headers={"retry-after-ms": "1"})
            if "stream_error" in turn:
                error = {"error": {"message": turn["stream_error"], "type": "fake"}}
                return Response(f"data: {json.dumps(error)}\n\n", media_type="text/event-stream")
            return build(i, turn, body)
        return endpoint

    def __enter__(self) -> "FakeLiteLLM":
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        self.url = f"http://{self.host}:{self.server.servers[0].sockets[0].getsockname()[1]}"
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join()
