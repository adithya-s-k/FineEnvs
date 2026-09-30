"""Regression checks for simultaneous train/eval admission and capture budgets."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import unittest

from common import configure
configure()
from service_policy import Admission, output_limit


class SharedServiceTest(unittest.TestCase):
    def test_reserved_train_slot_and_cancelled_waiter(self):
        async def run():
            gate = Admission(3, 1)
            async with gate.slot("test"), gate.slot("test"):
                blocked = asyncio.create_task(enter(gate, "test"))
                await asyncio.sleep(0.15)
                self.assertFalse(blocked.done())
                async with gate.slot("train"):
                    self.assertEqual(gate.snapshot()["active"], {"train": 1, "eval": 2})
                blocked.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await blocked
            self.assertEqual(gate.snapshot()["active"], {"train": 0, "eval": 0})
            self.assertEqual(gate.snapshot()["waiting"], {"train": 0, "eval": 0})
            async with gate.slot("test"):
                pass
        async def enter(gate, role):
            async with gate.slot(role):
                pass
        asyncio.run(run())

    def test_simultaneous_split_caps_reach_engine_unchanged(self):
        from fastapi.testclient import TestClient
        from openenv.core.harness.capture.server import create_app
        class Engine:
            served_model = "test-model"
            param_fixes = {}
            capture_level = "text"
            async def completion(self, request):
                return {"id": "cap-test", "object": "chat.completion", "model": self.served_model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": str(request['max_tokens'])},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        app = create_app(llm_url="http://unused.invalid/v1", model="test-model", capture_level="text", max_output_tokens=16384)
        engine = Engine()
        app.state.inference = engine
        app.state.upstreams._default = (engine, "text")
        sessions = {split: app.state.registry.create(dataset=split, max_output_tokens=output_limit(split))
                    for split in ("train", "test")}
        with TestClient(app) as client:
            def call(split):
                r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + sessions[split].session_id},
                                json={"model": "test-model", "messages": [{"role": "user", "content": split}], "max_tokens": 32768})
                self.assertEqual(r.status_code, 200, r.text)
                return split, int(r.json()["choices"][0]["message"]["content"])
            with ThreadPoolExecutor(max_workers=2) as pool:
                for split, cap in pool.map(call, ["train", "test"] * 4):
                    self.assertEqual(cap, 16384 if split == "train" else 4096)


if __name__ == "__main__":
    unittest.main()
