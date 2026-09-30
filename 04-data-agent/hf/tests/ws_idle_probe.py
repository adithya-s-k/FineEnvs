"""Compare idle HF Space WebSockets with/without keepalive; creates no sandboxes."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from dotenv import dotenv_values
from websockets.asyncio.client import connect


async def main(args):
    token = dotenv_values(args.env_file)["HF_API_KEY"]
    url = args.url.replace("https://", "wss://").rstrip("/") + "/ws"
    async def one(interval):
        row = {"ping_interval_s": interval, "idle_seconds": args.seconds, "started_at": time.time()}
        try:
            async with connect(url, additional_headers={"Authorization": "Bearer " + token},
                               max_size=104857600, ping_interval=interval, ping_timeout=None,
                               open_timeout=30) as ws:
                await ws.send('{"type":"state"}')
                initial = json.loads(await asyncio.wait_for(ws.recv(), 30))
                assert initial.get("type") != "error", "Native state request rejected"
                print(json.dumps({"connected": True, "ping_interval_s": interval}), flush=True)
                await asyncio.sleep(args.seconds)
                await ws.send('{"type":"state"}')
                after = json.loads(await asyncio.wait_for(ws.recv(), 30))
                assert after.get("type") != "error", "State request rejected after idle"
                row["passed"] = True
        except Exception as exc:
            row.update(passed=False, error_type=type(exc).__name__)
        row["finished_at"] = time.time()
        print(json.dumps(row), flush=True)
        return row
    rows = await asyncio.gather(one(None), one(20))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"url": args.url, "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--env-file", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=720)
    asyncio.run(main(p.parse_args()))
