"""Regression checks for the standalone arm's token and task boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
NATIVE=Path(__file__).resolve().parents[2]/'envs/blackbox-opencode'
sys.path.insert(0,str(ROOT/'OpenEnv/src'))
spec=importlib.util.spec_from_file_location('data_agent_env',NATIVE/'__init__.py',submodule_search_locations=[str(NATIVE)])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from data_agent_env.models import DataAgentRolloutResult
from data_agent_env.server.rollout import turns_from_capture, _stage_inputs, _run_agent
from data_agent_env.harness import to_trace_entries
from data_agent_env.config import DataAgentConfig
from data_agent_env.tasks import _frozen_rows
from data_agent_env.task import DataAgentTask

class StandaloneTests(unittest.TestCase):
    def test_agent_deadline_terminates_background_process(self):
        from unittest.mock import Mock
        sandbox=Mock()
        sandbox.start_bg.return_value.wait.side_effect=TimeoutError('agent budget expired')
        code=_run_agent(sandbox,'http://capture','session','model',DataAgentConfig(),'solve this')
        self.assertEqual(code,124)
        sandbox.start_bg.return_value.kill.assert_called_once()
        sandbox.exec.assert_not_called()
    def test_background_transport_failure_remains_ungraded(self):
        from unittest.mock import Mock
        sandbox=Mock()
        sandbox.start_bg.return_value.wait.side_effect=ConnectionError('transport unavailable')
        with self.assertRaises(ConnectionError):
            _run_agent(sandbox,'http://capture','session','model',DataAgentConfig(),'solve this')
    def test_partial_and_zero_masks_survive_wire_boundary(self):
        for mask in ([0,0,1,0],[0,0,0,0]):
            raw={'prompt_token_ids':[1,2],'completion_token_ids':[3,4],
                 'per_token_logps':[-.1,-.2],'loss_mask':mask}
            result=DataAgentRolloutResult.model_validate_json(DataAgentRolloutResult(
                rollout_type='train',turns=turns_from_capture([raw])).model_dump_json())
            entries=to_trace_entries(result)
            self.assertEqual(entries[0]['loss_mask'] if entries else [],mask if any(mask) else [])
    def test_bad_logprobs_cannot_become_trainable(self):
        raw={'prompt_token_ids':[1,2],'completion_token_ids':[3,4],
             'per_token_logps':[-.1],'loss_mask':[0,0,1,1]}
        with self.assertRaises(ValueError):
            to_trace_entries(DataAgentRolloutResult(turns=turns_from_capture([raw])))
    def test_stage_token_is_transient_and_absent_from_shell(self):
        from unittest.mock import Mock
        task=DataAgentTask(task_id='fixture',instruction='question',answer='a',hf_bucket='owner/bucket',bucket_prefix='data')
        sandbox=Mock();sandbox.exec.return_value.exit_code=0
        _stage_inputs(sandbox,task,'fixture-secret',DataAgentConfig())
        call=sandbox.exec.call_args
        self.assertNotIn('fixture-secret',call.args[0])
        self.assertEqual(call.kwargs['envs']['HF_TOKEN'],'fixture-secret')
        self.assertNotIn('HF_TOKEN',task.env(None))
    def test_fixed_catalog_preserves_identity_and_difficulty(self):
        root=ROOT/'experiments/daytona_harness_comparison/logs/20260915/datasets'
        if not root.exists():self.skipTest('frozen comparison dataset is unavailable')
        from collections import Counter
        rows=_frozen_rows(str(root),'test')
        self.assertEqual(len(rows),250)
        self.assertEqual(Counter(r['difficulty_tier'] for r in rows),{'easy':33,'medium':118,'hard':99})
        manifest=json.loads((root.parent/'test_manifest.json').read_text())
        self.assertEqual([r['task_id'] for r in rows],sorted(t['name'] for t in manifest['tasks']))

if __name__=='__main__':unittest.main()
