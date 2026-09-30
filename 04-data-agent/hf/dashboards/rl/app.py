import os

os.environ["TRACKIO_PLOT_ORDER"] = 'eval_observed/pass_at_1,eval_observed/combined_reward_normalized,eval_observed/tool_call_savings_pct,train_verified/reward_mean50,train_sft/loss_mean50,train_sft/*,train_summary/*,eval_coverage/graded_cells,eval_coverage/missing_cells,eval_observed/harness/*,eval_observed/difficulty/*,eval_efficiency/*,eval_usage/overall/*,eval_usage/harness/*,train_verified/reward,train_verified/tools/*,train_verified/completions/*,train_verified/batch/trained_tokens_per_step,train_verified/batch/forwarded_tokens_per_step,eval_observed/harness_difficulty/*,train_verified/*,eval_coverage/*,eval/pass_at_1,eval/provisional_pass_at_1,train/reward_rolling20,eval/harness/*,eval/difficulty/*'

import trackio
from trackio import server
from trackio.sqlite_storage import SQLiteStorage
from comparison_reads import install

install(server, SQLiteStorage, "data-agent-rl-comparison", hidden_run_ids=['7e03598b74c5944b42002d1759e0184c', 'd189e77a4f134099c308ed223b840405', '2419d3a7bf21850a46903d5df20f3d59', '60cde9f411f365b5c634ee3de1351505', 'a9a405588dd8646aea613c02bf479c8b', '736a2514f326ea6119699671e61bb0ef'])

from dashboard_ui import prepare_frontend

trackio.show(project='data-agent-rl-comparison', open_browser=False, frontend_dir=prepare_frontend())
