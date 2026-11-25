"""Offline baseline evaluation on synthetic fixtures or imported traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ENV_ROOT = str(Path(__file__).resolve().parents[2])
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from trace_audit.core import load_records  # noqa: E402
from trace_audit.core.evaluation import calibration_svg, evaluate  # noqa: E402
from trace_audit.core.tokenization import token_counter  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/sample.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local/baselines.json")
    parser.add_argument("--allow-fixtures", action="store_true")
    parser.add_argument(
        "--encoding", help="Target tiktoken encoding, installed/cached during setup"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="JSONL {record_id, response} from a separately authorized judge",
    )
    parser.add_argument("--training-size", type=int)
    parser.add_argument("--source-cap", type=float, default=0.5)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--max-record-reuse", type=int, default=1)
    args = parser.parse_args()
    records = load_records(args.data)
    fixture = args.allow_fixtures or args.data.resolve() == (ROOT / "data/sample.jsonl").resolve()
    predictions = None
    if args.predictions:
        predictions = {}
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["record_id"] in predictions:
                raise ValueError("Duplicate judge prediction")
            predictions[row["record_id"]] = row["response"]
    report = evaluate(
        records,
        allow_fixtures=fixture,
        predictions=predictions,
        token_counter=token_counter(args.encoding),
        training_size=args.training_size,
        source_cap=args.source_cap,
        sampling_seed=args.sampling_seed,
        max_record_reuse=args.max_record_reuse,
        validated=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    calibration_svg(report, args.output.with_suffix(".svg"))
    episode = report["episode"]
    print("System prompt:\n" + episode["system_prompt"])
    print("Prompt:\n" + episode["prompt"])
    print("Response: " + json.dumps(episode["response"]))
    print("Reward: " + str(episode["reward"]))
    print(
        json.dumps(
            {
                "fixture_only": report["fixture_only"],
                "baselines": list(report["baselines"]),
                "output": str(args.output),
                "gate": report["rl_gate"]["comparison_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
