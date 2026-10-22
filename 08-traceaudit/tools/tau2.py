"""Prepare a collection plan or convert native results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_ENV_ROOT = str(ROOT / "envs")
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from adapters.tau2 import convert_results  # noqa: E402
from tools.data import admission_report, write_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument(
        "--output", type=Path, default=ROOT / "data/generated/tau2_collection_plan.json"
    )
    plan.add_argument("--agent-model")
    plan.add_argument("--user-model")
    plan.add_argument("--budget-usd", type=float)
    convert = sub.add_parser("convert")
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--revision", required=True)
    convert.add_argument("--domain", choices=["airline", "telecom", "retail"], required=True)
    convert.add_argument("--policy", type=Path, required=True)
    convert.add_argument("--tools", type=Path, required=True)
    convert.add_argument(
        "--agent-model", help="Captured collector provenance; must agree with Results metadata"
    )
    convert.add_argument(
        "--user-model", help="Captured user-simulator provenance; must agree with Results metadata"
    )
    args = parser.parse_args()
    if args.command == "plan":
        checks = json.loads((ROOT / "data/source_checks.json").read_text(encoding="utf-8"))
        data = {
            "revision": checks["tau_revision"],
            "agent_model": args.agent_model,
            "user_model": args.user_model,
            "budget_usd": args.budget_usd,
            "status": "not_launched",
            "requires_budget_enforcement_by_collector": True,
            "domains": {
                key: value["db_change_task_ids"] for key, value in checks["tau_domains"].items()
            },
            "collection_requirements": {
                "unit": "task/agent_model/user_model",
                "repeated_trials": True,
                "target_unique_pairs_per_unit": 10,
                "target_written_successes_per_unit": 10,
                "target_written_failures_per_unit": 10,
                "stopping_rule": (
                    "Stop at collector budget or configured trial limit, "
                    "even if mixed outcomes are missing; never manufacture labels."
                ),
                "retain": (
                    "All raw trials, including unlabelled and unpaired records; "
                    "sampling never deletes source data."
                ),
                "coverage_report": [
                    "raw trials and failures/successes per task/model pair",
                    "paired retained count and retention fraction",
                    "unique written successes/failures",
                    "task-family coverage and estimated collection shortfall",
                ],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Unlaunched collection plan: {args.output}")
    else:
        rows = convert_results(
            json.loads(args.input.read_text(encoding="utf-8")),
            revision=args.revision,
            domain=args.domain,
            agent_model=args.agent_model,
            user_model=args.user_model,
            policy=args.policy.read_text(encoding="utf-8"),
            tools=json.loads(args.tools.read_text(encoding="utf-8")),
        )
        report = admission_report(rows)
        write_jsonl(args.output, rows)
        Path(str(args.output) + ".report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Converted {len(rows)} runs; criteria-ineligible runs remain unlabelled")


if __name__ == "__main__":
    main()
