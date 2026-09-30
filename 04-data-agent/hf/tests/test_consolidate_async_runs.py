import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import consolidate_async_runs as c


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.score = {"complete": True, "comparison_ready": True, "tito_pass": True,
            "harness_versions_match_baseline": True, "graded_cells": 1000,
            "average_pass_at_1": 33 / 250, "harnesses": {
                h: {"graded": 250, "pass_at_1": 33 / 250, "difficulty": {
                    d: {"graded": n, "correct": n if d == 'easy' else 0}
                    for d, n in c.COUNTS.items()}} for h in c.HARNESSES}}

    def test_weighting_uses_counts_not_mean_of_difficulty_rates(self):
        c.validated_score(self.score)
        m = c.score_metrics(self.score, self.score)
        self.assertEqual(m['eval/pass_at_1'], .132)
        self.assertEqual(m['eval/difficulty/easy/pass_at_1'], 1)
        self.assertEqual(m['eval/difficulty/medium/pass_at_1'], 0)
        self.assertEqual(len([k for k in m if k.startswith('eval/harness_difficulty/')]), 12)

    def test_partial_or_bad_audit_cannot_be_published(self):
        for field, value in [('graded_cells',999),('tito_pass',False),('comparison_ready',False)]:
            s=copy.deepcopy(self.score);s[field]=value
            with self.assertRaises(ValueError):c.validated_score(s)
        s=copy.deepcopy(self.score);s['harnesses']['codex']['difficulty']['hard']['graded']=98
        with self.assertRaises(ValueError):c.validated_score(s)

    def test_training_and_eval_share_run_but_have_distinct_replay_ids(self):
        train=c.native.event(c.PROJECT,c.ARMS[0],100,{'train/reward':.2},{},identity='training')
        again=c.native.event(c.PROJECT,c.ARMS[0],100,{'train/reward':.2},{},identity='training')
        evaluation=c.native.event(c.PROJECT,c.ARMS[0],100,{'eval/pass_at_1':.248},{},identity='evaluation')
        other=c.native.event(c.PROJECT,c.ARMS[1],100,{'train/reward':.2},{},identity='training')
        self.assertEqual(train['log_id'],again['log_id'])
        self.assertNotEqual(train['log_id'],evaluation['log_id'])
        self.assertEqual(train['run_id'],evaluation['run_id'])
        self.assertNotEqual(train['run_id'],other['run_id'])


if __name__=='__main__':unittest.main()
