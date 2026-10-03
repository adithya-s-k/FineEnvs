"""Streaming raw-reaction normalization with provenance-preserving deduplication."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from rdkit.Chem import rdChemReactions

from .chemistry import (
    ChemistryError,
    audit_atom_mapping,
    canonicalize_reaction,
    primary_product,
    stable_hash,
)


SCHEMA_VERSION = "litefold-reaction-v1"
EVIDENCE_KINDS = {"observed", "generated", "simulated"}


def iter_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield one-based input positions and rows from JSONL, CSV, or TSV."""
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8", newline="") as handle:
        if suffix in {".jsonl", ".ndjson"}:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row is not an object")
                yield line_number, value
            return
        if suffix not in {".csv", ".tsv"}:
            raise ValueError(f"unsupported input extension {suffix!r}: {path}")
        delimiter = "\t" if suffix == ".tsv" else ","
        for line_number, row in enumerate(csv.DictReader(handle, delimiter=delimiter), 2):
            yield line_number, dict(row)


def normalize_row(
    row: dict[str, Any],
    *,
    source_name: str | None = None,
    source_license: str | None = None,
    evidence_kind: str = "observed",
    reaction_column: str = "reaction_smiles",
) -> dict[str, Any]:
    """Normalize one source row or raise a reason suitable for a rejects file."""
    raw_reaction = _text(row.get(reaction_column))
    if not raw_reaction:
        reactants = _component_text(row.get("reactants"))
        reagents = _component_text(row.get("reagents"))
        products = _component_text(row.get("products"))
        raw_reaction = f"{reactants}>{reagents}>{products}"

    canonical, reactants, reagents, products = canonicalize_reaction(raw_reaction)
    mapped_canonical, _, _, _ = canonicalize_reaction(
        raw_reaction, keep_atom_maps=True
    )
    mapping = audit_atom_mapping(raw_reaction)
    if mapping["status"] == "invalid":
        raise ChemistryError("invalid atom mapping: " + "; ".join(mapping["errors"]))

    name = _text(row.get("source_name")) or source_name
    license_id = _text(row.get("source_license")) or source_license
    if not name:
        raise ValueError("source_name is required per row or via --source-name")
    if not license_id:
        raise ValueError("source_license is required per row or via --source-license")

    kind = (_text(row.get("evidence_kind")) or evidence_kind).lower()
    if kind not in EVIDENCE_KINDS:
        raise ValueError(
            f"evidence_kind must be one of {sorted(EVIDENCE_KINDS)}, got {kind!r}"
        )

    reaction_smarts = _text(row.get("reaction_smarts"))
    template_hash = None
    if reaction_smarts:
        try:
            reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
            rdChemReactions.SanitizeRxn(reaction)
        except Exception as exc:
            raise ChemistryError(f"invalid reaction_smarts: {exc}") from exc
        template_hash = stable_hash(reaction_smarts, prefix="tpl_")

    reaction_hash = stable_hash(canonical, prefix="crh_", length=32)
    source = {
        "name": name,
        "license": license_id,
        "record_id": _text(row.get("source_record_id") or row.get("id")),
        # Patent/application/route grouping is required for split leakage audits.
        # It is deliberately distinct from the per-reaction record identifier.
        "group_id": _text(
            row.get("source_group_id")
            or row.get("patent_id")
            or row.get("route_group_id")
        ),
        "url": _text(row.get("source_url")),
        "publication_year": _integer_or_none(row.get("publication_year")),
    }
    source = {key: value for key, value in source.items() if value is not None}

    confidence = _float_or_none(row.get("confidence"))
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    conditions = _jsonish(row.get("conditions"))
    reaction_class = _text(row.get("reaction_class"))
    return {
        "schema_version": SCHEMA_VERSION,
        "reaction_id": stable_hash(canonical, prefix="rxn_", length=24),
        "canonical_reaction_hash": reaction_hash,
        "reaction_smiles": canonical,
        "mapped_reaction_smiles": (
            mapped_canonical if mapping["status"] != "unmapped" else None
        ),
        "reactants": list(reactants),
        "reagents": list(reagents),
        "products": list(products),
        "primary_product": primary_product(products),
        "atom_mapping": mapping,
        "reaction_class": reaction_class,
        "reaction_class_candidates": [reaction_class] if reaction_class else [],
        "reaction_smarts": reaction_smarts,
        "reaction_smarts_candidates": [reaction_smarts] if reaction_smarts else [],
        "template_hash": template_hash,
        "conditions": [conditions] if conditions not in (None, "", {}) else [],
        "confidence": confidence,
        "confidence_values": [confidence] if confidence is not None else [],
        "annotations": [
            annotations
        ] if (annotations := _jsonish(row.get("annotations"))) not in (None, "", {}) else [],
        "evidence_kind": kind,
        "evidence_kinds": [kind],
        "source": [source],
    }


def merge_duplicate(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Merge mutable metadata while leaving canonical chemistry untouched."""
    if existing["canonical_reaction_hash"] != incoming["canonical_reaction_hash"]:
        raise ValueError("attempted to merge different reactions")
    _extend_unique(existing["source"], incoming["source"])
    _extend_unique(existing["reaction_class_candidates"], incoming["reaction_class_candidates"])
    _extend_unique(existing["reaction_smarts_candidates"], incoming["reaction_smarts_candidates"])
    _extend_unique(existing["conditions"], incoming["conditions"])
    _extend_unique(existing["confidence_values"], incoming["confidence_values"])
    _extend_unique(existing["annotations"], incoming["annotations"])
    _extend_unique(existing["evidence_kinds"], incoming["evidence_kinds"])

    classes = existing["reaction_class_candidates"]
    existing["reaction_class"] = classes[0] if len(classes) == 1 else None
    templates = existing["reaction_smarts_candidates"]
    existing["reaction_smarts"] = templates[0] if len(templates) == 1 else None
    existing["template_hash"] = (
        stable_hash(templates[0], prefix="tpl_") if len(templates) == 1 else None
    )
    kinds = existing["evidence_kinds"]
    existing["evidence_kind"] = kinds[0] if len(kinds) == 1 else "mixed"
    confidences = existing["confidence_values"]
    existing["confidence"] = max(confidences) if confidences else None

    # Prefer the strongest mapping evidence available among identical chemistry.
    rank = {"unmapped": 0, "partial": 1, "complete": 2}
    old = existing["atom_mapping"]["status"]
    new = incoming["atom_mapping"]["status"]
    if rank.get(new, -1) > rank.get(old, -1):
        existing["atom_mapping"] = incoming["atom_mapping"]
        existing["mapped_reaction_smiles"] = incoming["mapped_reaction_smiles"]


def build_corpus(
    inputs: Iterable[Path],
    output: Path,
    rejects: Path,
    *,
    source_name: str | None,
    source_license: str | None,
    evidence_kind: str = "observed",
    reaction_column: str = "reaction_smiles",
    report_path: Path | None = None,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    reject_rows: list[dict[str, Any]] = []

    for input_path in inputs:
        for line_number, row in iter_rows(input_path):
            counts["rows_read"] += 1
            try:
                record = normalize_row(
                    row,
                    source_name=source_name,
                    source_license=source_license,
                    evidence_kind=evidence_kind,
                    reaction_column=reaction_column,
                )
            except Exception as exc:
                counts["rows_rejected"] += 1
                reject_rows.append(
                    {
                        "input": str(input_path),
                        "line": line_number,
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                        "row": row,
                    }
                )
                continue
            key = record["canonical_reaction_hash"]
            if key in records:
                counts["duplicates_merged"] += 1
                merge_duplicate(records[key], record)
            else:
                records[key] = record
                counts["unique_reactions"] += 1
            counts[f"mapping_{record['atom_mapping']['status']}"] += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": [str(path) for path in inputs],
        "counts": dict(sorted(counts.items())),
    }
    _write_jsonl(output, (records[key] for key in sorted(records)))
    _write_jsonl(rejects, reject_rows)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _extend_unique(target: list[Any], values: Iterable[Any]) -> None:
    seen = {json.dumps(value, sort_keys=True, default=str) for value in target}
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key not in seen:
            target.append(value)
            seen.add(key)
    target.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))


def _component_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ".".join(str(item).strip() for item in value if str(item).strip())
    return _text(value) or ""


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _integer_or_none(value: Any) -> int | None:
    text = _text(value)
    return int(text) if text is not None else None


def _float_or_none(value: Any) -> float | None:
    text = _text(value)
    return float(text) if text is not None else None


def _jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] in "[{\"" or text in {"true", "false", "null"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--source-name")
    parser.add_argument("--source-license")
    parser.add_argument("--evidence-kind", choices=sorted(EVIDENCE_KINDS), default="observed")
    parser.add_argument("--reaction-column", default="reaction_smiles")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report or args.output.with_suffix(".report.json")
    report = build_corpus(
        args.inputs,
        args.output,
        args.rejects,
        source_name=args.source_name,
        source_license=args.source_license,
        evidence_kind=args.evidence_kind,
        reaction_column=args.reaction_column,
        report_path=report_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["counts"].get("rows_rejected") else 2


if __name__ == "__main__":
    sys.exit(main())
