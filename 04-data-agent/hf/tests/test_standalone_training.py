"""Exercise native scheduling, reward semantics and the exact capture admission audit."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "envs/blackbox-opencode"
sys.path.insert(0, str(ROOT / "train"))
spec = importlib.util.spec_from_file_location("data_agent_env", NATIVE / "__init__.py", submodule_search_locations=[str(NATIVE)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from data_agent_env.models import DataAgentRolloutResult
from data_agent_env.tasks import DataAgentTaskProvider
from standalone_comparison import ScheduledOpenCodeFactory, ComparisonSession
from data_agent_env.client import DataAgentEnv


class StandaloneTrainingTests(unittest.TestCase):
    def test_sampling_policy_reaches_the_wire_and_native_capture_registry(self):
        from data_agent_env.server.capture import mint_session
        from openenv.core.harness.capture.sessions import SessionRegistry
        from openenv.core.harness.capture.upstream import training_sampling
        policy = {"temperature": .8, "top_p": 1., "top_k": 0}
        factory = self.factory()
        client = object.__new__(DataAgentEnv)
        calls = []
        def call(name, **kwargs):
            calls.append((name, kwargs))
            return DataAgentRolloutResult().model_dump_json()
        client._call = call
        session = ComparisonSession(client, "train", 0, "fixture", **factory._rollout_kwargs)
        session.wait_for_completion(timeout_s=123)
        self.assertEqual(calls[0][1]["sampling"], policy)
        self.assertEqual(calls[0][1]["_timeout_s"], 123)
        server = Mock(registry=SessionRegistry())
        sid, _ = mint_session(server, llm_url="http://inference", model="fixture",
            rollout_id="fixture", capture_level="tokens", sampling=calls[0][1]["sampling"])
        captured = server.registry.get(sid)
        self.assertEqual(captured.sampling, training_sampling(policy))
        self.assertEqual(captured.sampling["top_k"], -1)

    def factory(self):
        from harness_schedule import make_schedule
        self.schedule = make_schedule([{ "name": f"fixture-{i}", "task_index": i, "difficulty": "easy"} for i in range(12)], ["opencode"], easy_start=4)
        return ScheduledOpenCodeFactory("http://fixture", harnesses=["opencode"], schedule=self.schedule,
            llm_url="http://inference", model="Qwen/Qwen3.5-2B", sampling={"temperature": .8, "top_p": 1., "top_k": 0})

    def test_all_scheduled_groups_and_resume_preserve_task_identity(self):
        factory = self.factory()
        tasks = [{"index": t["task_index"], "task_id": t["name"], "instruction": t["name"]} for t in self.schedule["tasks"]]
        client = Mock(); client.get_task_range.return_value = tasks
        with patch.object(ScheduledOpenCodeFactory, "_new_client", return_value=client):
            rows = factory.prompt_rows()
            restored = pickle.loads(pickle.dumps(factory))
            for offset in (0, 997, 3999):
                restored.group_offset = offset
                for group_id in range(20):
                    expected = self.schedule["groups"][(offset + group_id) % len(self.schedule["groups"])]
                    row = rows[expected["task_row"]]
                    for generation in range(8):
                        session = restored.create(row, seed=group_id, episode_id=str(generation))
                        self.assertEqual(session._task_index, expected["task_index"])
            restored.group_offset = 0
            with self.assertRaises(ValueError):
                restored.create({"prompt": [{"role": "user", "content": "unknown task"}]}, seed=0)

    def test_reward_matches_binary_comparison_and_preserves_ungraded(self):
        for correctness, raw_reward, expected in [(None, None, None), (0., 0., 0.), (.3, .3, 0.), (1., 1.1, 1.)]:
            session = ComparisonSession(Mock(), "train", 0, "task")
            session._result = DataAgentRolloutResult(correctness=correctness, reward=raw_reward)
            self.assertEqual(session.verify([]).env_reward, expected)
            self.assertEqual(session.result.reward, raw_reward)

    def test_changed_server_task_identity_is_rejected(self):
        factory = self.factory()
        tasks = [{"index": t["task_index"], "task_id": t["name"], "instruction": t["name"]} for t in self.schedule["tasks"]]
        tasks[0]["task_id"] = "wrong"
        client = Mock(); client.get_task_range.return_value = tasks
        with patch.object(ScheduledOpenCodeFactory, "_new_client", return_value=client):
            with self.assertRaises(ValueError): factory.prompt_rows()

if __name__ == "__main__": unittest.main()
