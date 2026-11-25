"""Offline imports, admission reports and human review."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_ENV_ROOT = str(ROOT / "envs")
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from trace_audit.core import DataError, load_records  # noqa: E402
from trace_audit.core.provenance import generator_key, resolve_model  # noqa: E402
from trace_audit.core.records import scoring_exclusion_reason, validate_collection  # noqa: E402
from trace_audit.core.review import apply_reviews, make_queue  # noqa: E402

from adapters import abcd, external, swe_rebench, tau2  # noqa: E402


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def admission_report(records, *, max_tokens=32000, token_counter=None, validated=False):
    if not validated:
        validate_collection(records)
    counts = Counter(
        (
            row["source"]["dataset"],
            row["split"],
            row["outcome"]["label_source"],
            str(row["outcome"]["failed"]),
        )
        for row in records
    )
    exclusions, eligible = [], {}
    # Rendering/admission is exactly the runtime path, including conservative byte budget.
    from trace_audit.core.render import SYSTEM_PROMPT, render_record

    count = token_counter or (lambda text: len(text.encode()))
    for row in records:
        reason = scoring_exclusion_reason(row)
        try:
            rendered = render_record(row)
            if reason is None and count(SYSTEM_PROMPT + "\n" + rendered) > max_tokens:
                reason = "over_render_budget"
        except ValueError:
            reason = "unsupported_multimodal"
        if reason:
            exclusions.append((row["source"]["dataset"], row["split"], reason))
        else:
            eligible[row["split"]] = eligible.get(row["split"], 0) + 1
    return {
        "records": len(records),
        "counts": [
            {"source": a, "split": b, "label_source": c, "failed": d, "n": n}
            for (a, b, c, d), n in sorted(counts.items())
        ],
        "eligible": eligible,
        "exclusions": [
            {"source": a, "split": b, "reason": c, "n": n}
            for (a, b, c), n in sorted(Counter(exclusions).items())
        ],
        "max_tokens": max_tokens,
        "counting": getattr(
            token_counter, "encoding_name", "Conservative UTF-8 byte upper bound; no truncation"
        ),
        "model_provenance": {
            "known_records": sum(generator_key(row) is not None for row in records),
            "missing_records": sum(generator_key(row) is None for row in records),
        },
    }


def import_source(args):
    if args.source == "abcd":
        if args.license != "MIT":
            raise DataError("Pinned ABCD source licence is MIT; conflicting --license rejected")
        opener = gzip.open if str(args.input).endswith(".gz") else open
        with opener(args.input, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        rows = [
            abcd.convert(row, revision=args.revision, split=split)
            for split, subset in data.items()
            for row in subset
        ]
    else:
        rows = []
        _import_rows(args, rows)
    for row in rows:
        for field in ("agent_model", "user_model"):
            row["source"][field] = resolve_model(
                row["source"].get(field), getattr(args, field, None)
            )
    from trace_audit.core.tokenization import token_counter

    report = admission_report(rows, token_counter=token_counter(args.encoding))
    write_jsonl(args.output, rows)
    report["input_sha256"] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    report["revision"] = args.revision
    Path(str(args.output) + ".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"records": len(rows), "output": str(args.output), "eligible": report["eligible"]}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument(
        "--source",
        required=True,
        choices=[
            "abcd",
            "swe",
            "tau2",
            "microsoft/AgentRx",
            "PatronusAI/TRAIL",
            "Kevin355/Who_and_When",
            "mcemri/MAST-Data",
        ],
    )
    imp.add_argument("--input", type=Path, required=True)
    imp.add_argument("--output", type=Path, required=True)
    imp.add_argument("--revision", required=True)
    imp.add_argument(
        "--license",
        required=True,
        help="Verified licence for this pinned data export; not inherited from framework code",
    )
    imp.add_argument("--encoding", help="Target tiktoken encoding; install/cache it during setup")
    imp.add_argument(
        "--agent-model",
        help="Captured generating-model provenance; must agree with record metadata",
    )
    imp.add_argument("--user-model", help="Captured simulator identity, or explicit not_applicable")
    check = sub.add_parser("validate")
    check.add_argument("input", type=Path)
    check.add_argument("--encoding", help="Target tiktoken encoding; install/cache it during setup")
    queue = sub.add_parser("review-queue")
    queue.add_argument("input", type=Path)
    queue.add_argument("output", type=Path)
    queue.add_argument("--per-source", type=int, default=50)
    apply = sub.add_parser("apply-reviews")
    apply.add_argument("input", type=Path)
    apply.add_argument("reviews", type=Path)
    apply.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "import":
            import_source(args)
        elif args.command == "validate":
            from trace_audit.core.tokenization import token_counter

            print(
                json.dumps(
                    admission_report(
                        load_records(args.input),
                        token_counter=token_counter(args.encoding),
                        validated=True,
                    ),
                    indent=2,
                )
            )
        elif args.command == "review-queue":
            records = load_records(args.input)
            queue = make_queue(records, per_source=args.per_source)
            write_jsonl(args.output, queue)
            from trace_audit.core.render import render_record

            selected = {r["record_id"]: r for r in records}
            packet = []
            for item in queue:
                row = selected[item["record_id"]]
                try:
                    trace = render_record(row)
                except ValueError:
                    trace = (
                        "Unsupported multimodal trace: review the original record; "
                        "not admitted to v1."
                    )
                packet.append(
                    {
                        "record_id": item["record_id"],
                        "record_sha256": item["record_sha256"],
                        "source": row["source"],
                        "existing_outcome": row["outcome"],
                        "observation": trace,
                    }
                )
            write_jsonl(str(args.output) + ".traces.jsonl", packet)
            print(f"Pending review queue: {args.output}")
        else:
            rows, summary = apply_reviews(load_records(args.input), read_jsonl(args.reviews))
            write_jsonl(args.output, rows)
            Path(str(args.output) + ".reviews.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary))
    except (DataError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


def _import_rows(args, rows):
    for index, row in enumerate(read_jsonl(args.input), 1):
        try:
            if args.source == "swe":
                converted = swe_rebench.convert(row, revision=args.revision, license=args.license)
            elif args.source == "tau2":
                converted = tau2.convert(row, revision=args.revision, license=args.license)
            else:
                converted = external.convert(
                    row, dataset=args.source, revision=args.revision, license=args.license
                )
            rows.append(converted)
        except (ValueError, KeyError, TypeError) as exc:
            raise DataError(f"Import rejected at line {index}; no output written: {exc}") from exc


if __name__ == "__main__":
    main()
