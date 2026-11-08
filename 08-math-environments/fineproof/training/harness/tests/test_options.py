"""FineProofOptions: kwargs, and the FINEPROOF_* fallbacks for launchers that cannot pass kwargs."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from fineproof_agent.agent import FineProofOptions


class EnvDefaults(unittest.TestCase):
    def test_env_fills_missing_options(self):
        env = {"FINEPROOF_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": false}}',
               "FINEPROOF_ROUNDS": "2", "FINEPROOF_STREAM": "true"}
        with mock.patch.dict(os.environ, env, clear=True):
            o = FineProofOptions.model_validate({"api_base": "http://proxy/v1"})
        self.assertEqual((o.rounds, o.stream), (2, True))
        self.assertIn("enable_thinking", o.extra_body)

    def test_kwargs_win(self):
        with mock.patch.dict(os.environ, {"FINEPROOF_ROUNDS": "3"}, clear=True):
            o = FineProofOptions.model_validate({"api_base": "http://proxy/v1", "rounds": 0})
        self.assertEqual(o.rounds, 0)

    def test_defaults_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            o = FineProofOptions.model_validate({"api_base": "http://proxy/v1"})
        self.assertEqual((o.rounds, o.max_tokens, o.extra_body), (1, 8192, None))


if __name__ == "__main__":
    unittest.main()
