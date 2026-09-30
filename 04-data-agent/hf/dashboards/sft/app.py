import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TRACKIO_DIR", "/tmp/data-agent-sft-trackio")
os.environ["TRACKIO_PLOT_ORDER"] = "eval_observed/pass_at_1,train_sft/loss_mean50,eval_observed/tool_call_savings_pct,eval_coverage/graded_cells,eval_observed/harness/*,eval_observed/difficulty/*,eval_efficiency/*,eval_usage/*,train_sft/*,train_summary/*,eval_coverage/*"

import trackio
from trackio import server
from trackio.sqlite_storage import SQLiteStorage
from comparison_reads import install

PROJECT = "data-agent-sft-comparison"
groups = defaultdict(list)
with gzip.open(Path(__file__).with_name("sft_events.json.gz"), "rt") as source:
    for entry in json.load(source):
        groups[entry["run_id"]].append(entry)
for run_id, entries in groups.items():
    SQLiteStorage.bulk_log(project=PROJECT, run=entries[0]["run"], run_id=run_id,
        metrics_list=[e["metrics"] for e in entries], steps=[e["step"] for e in entries],
        log_ids=[e["log_id"] for e in entries], config=entries[0]["config"])
print(f"Restored {sum(map(len, groups.values()))} scalar events across {len(groups)} SFT runs.", flush=True)
install(server, SQLiteStorage, PROJECT)
trackio.show(project=PROJECT, open_browser=False)
