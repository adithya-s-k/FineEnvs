"""Stream downloaded source formats into RetroEnv's flat normalization contract."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from .chemistry import audit_atom_mapping


SOURCE_META = {
    "crd": ("crd-1976-2024", "CC-BY-4.0"),
    "uspto_lowe": ("uspto-lowe-1976-2016", "CC0-1.0"),
    "creed": ("creed", "CC-BY-NC-4.0"),
    "frea": ("frea-rxnverif", "CC-BY-4.0"),
    "ord": ("ord-data", "CC-BY-SA-4.0"),
    "paroutes_v2": ("paroutes-v2-benchmark", "CC-BY-4.0"),
    "uspto_llm": ("uspto-llm", "CC-BY-4.0"),
}


def adapt_crd(path: Path) -> Iterator[dict[str, Any]]:
    archive = _one(path, "*.zip") if path.is_dir() else path
    with zipfile.ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"expected one data file in {archive}, found {names}")
        with zipped.open(names[0]) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8", errors="strict")
            for line_number, line in enumerate(text, 1):
                reaction = line.strip()
                if reaction:
                    yield {
                        "reaction_smiles": reaction,
                        "source_name": SOURCE_META["crd"][0],
                        "source_license": SOURCE_META["crd"][1],
                        "source_record_id": f"line-{line_number}",
                        "evidence_kind": "observed",
                    }


def adapt_frea(path: Path) -> Iterator[dict[str, Any]]:
    csv_path = _one(path, "**/rxnverif_v*.csv") if path.is_dir() else path
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            feasible = str(row.get("feasible", "")).strip() == "1"
            method = row.get("method") or ("positive" if feasible else "generated_negative")
            yield {
                "reaction_smiles": f"{row['reactants']}>>{row['product']}",
                "source_name": SOURCE_META["frea"][0],
                "source_license": SOURCE_META["frea"][1],
                "source_record_id": f"row-{row_number}",
                "source_group_id": f"product:{row['product']}",
                "evidence_kind": "observed" if feasible else "generated",
                "annotations": {"feasible": feasible, "generation_method": method},
            }


def adapt_creed(path: Path, *, release: str = "CREED-CCV") -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("install the 'sources' extra to adapt CREED") from exc
    release_dir = path / release if path.is_dir() else path
    for parquet_path in sorted(release_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        row_offset = 0
        for batch in parquet.iter_batches(
            batch_size=1024, columns=["product_smiles", "reactant_candidates"]
        ):
            for local_index, row in enumerate(batch.to_pylist()):
                product_group = f"{parquet_path.name}:{row_offset + local_index}"
                for candidate_index, candidate in enumerate(row["reactant_candidates"] or []):
                    product = candidate.get("product_isomeric_smiles") or row["product_smiles"]
                    reactants = candidate.get("reactants_isomeric_smiles") or []
                    if not product or not reactants:
                        continue
                    score = candidate.get("chemcensor_score")
                    yield {
                        "reaction_smiles": f"{'.'.join(reactants)}>>{product}",
                        "source_name": SOURCE_META["creed"][0],
                        "source_license": SOURCE_META["creed"][1],
                        "source_record_id": f"{product_group}:{candidate_index}",
                        "source_group_id": product_group,
                        "evidence_kind": "generated",
                        "annotations": {
                            "release": release,
                            "chemcensor_score": score,
                        },
                    }
            row_offset += batch.num_rows


def adapt_uspto_lowe(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError("install the 'sources' extra to adapt Lowe USPTO archives") from exc
    archives = sorted(path.glob("*_smiles.7z")) if path.is_dir() else [path]
    if not archives:
        raise FileNotFoundError(f"no *_smiles.7z files under {path}")
    for archive_path in archives:
        with tempfile.TemporaryDirectory(prefix="retroenv-lowe-") as temporary:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                archive.extractall(temporary)
            for extracted in sorted(Path(temporary).rglob("*")):
                if not extracted.is_file():
                    continue
                with extracted.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        fields = line.rstrip("\n").split("\t")
                        reaction_index = next(
                            (index for index, value in enumerate(fields) if value.count(">") == 2),
                            None,
                        )
                        if reaction_index is None:
                            continue
                        reaction = fields[reaction_index]
                        tail = fields[reaction_index + 1 :]
                        patent = tail[0].strip() if tail else ""
                        year = next(
                            (
                                int(value)
                                for value in tail
                                if value.isdigit() and 1970 <= int(value) <= 2030
                            ),
                            None,
                        )
                        yield {
                            "reaction_smiles": reaction,
                            "source_name": SOURCE_META["uspto_lowe"][0],
                            "source_license": SOURCE_META["uspto_lowe"][1],
                            "source_record_id": f"{archive_path.name}:{extracted.name}:{line_number}",
                            "source_group_id": patent or None,
                            "publication_year": year,
                            "evidence_kind": "observed",
                        }


def adapt_uspto_llm(path: Path, *, release: str = "single_step") -> Iterator[dict[str, Any]]:
    """Stream mapped single-step or step-divided USPTO-LLM CSV from its zip."""
    root = path if path.is_dir() else path.parent
    if release == "single_step":
        archive_path = root / "single_step.zip" if path.is_dir() else path
        member = "single_step/HGAR/uspto_llm_atommap.csv"
    elif release == "multi_step":
        archive_path = root / "multi_step.zip" if path.is_dir() else path
        member = "multi_step/uspto_multiple_step.csv"
    else:
        raise ValueError("USPTO-LLM release must be 'single_step' or 'multi_step'")
    with zipfile.ZipFile(archive_path) as zipped:
        if member not in zipped.namelist():
            raise FileNotFoundError(f"{member!r} is absent from {archive_path}")
        with zipped.open(member) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="strict", newline="")
            for row_number, row in enumerate(csv.DictReader(text), 1):
                record_id = str(row.get("id") or row.get("reaction_id") or f"row-{row_number}")
                raw_reaction = str(row.get("rs>>ps") or row.get("reactants>>products") or "")
                parts = raw_reaction.split(">")
                if len(parts) == 2:
                    reaction = f"{parts[0]}>>{parts[1]}"
                    middle = ""
                elif len(parts) == 3:
                    reaction = f"{parts[0]}>>{parts[2]}"
                    middle = parts[1]
                else:
                    # Let the generic normalizer reject malformed chemistry with
                    # a source record attached instead of silently dropping it.
                    reaction = raw_reaction
                    middle = ""
                tokens = record_id.split("-")
                patent_id = "-".join(tokens[1:-1]) if len(tokens) >= 3 else record_id
                year = int(tokens[0][:4]) if tokens and tokens[0][:4].isdigit() else None
                conditions = {
                    key: value
                    for key, value in {
                        "solvents": row.get("solvents"),
                        "catalyst": row.get("catalyst"),
                        "temperature": row.get("temperature"),
                        "time": row.get("time"),
                        "multi_step_context": middle or None,
                    }.items()
                    if value not in (None, "", "[]")
                }
                yield {
                    "reaction_smiles": reaction,
                    "source_name": SOURCE_META["uspto_llm"][0],
                    "source_license": SOURCE_META["uspto_llm"][1],
                    "source_record_id": record_id,
                    "source_group_id": patent_id,
                    "publication_year": year,
                    "evidence_kind": "observed",
                    "conditions": conditions,
                    "annotations": {"release": release, "llm_extracted": True},
                }


def adapt_paroutes_v2(path: Path, *, release: str = "n1") -> Iterator[dict[str, Any]]:
    """Convert PaRoutes molecule/reaction trees to explicit provenance-backed routes."""
    if release not in {"n1", "n5"}:
        raise ValueError("PaRoutes release must be 'n1' or 'n5'")
    route_path = path / f"ref_routes_{release}.json" if path.is_dir() else path
    with route_path.open("r", encoding="utf-8") as handle:
        routes = json.load(handle)
    if not isinstance(routes, list):
        raise ValueError(f"expected a route list in {route_path}")
    for route_index, root in enumerate(routes):
        route_id = f"paroutes-v2:{release}:{route_index}"
        steps: list[dict[str, Any]] = []

        def visit(molecule: dict[str, Any]) -> None:
            if molecule.get("type") != "mol" or not molecule.get("smiles"):
                raise ValueError(f"{route_id}: malformed molecule node")
            reactions = [
                child for child in molecule.get("children") or []
                if child.get("type") == "reaction"
            ]
            if not reactions:
                return
            if len(reactions) != 1:
                raise ValueError(
                    f"{route_id}: molecule node contains {len(reactions)} reaction alternatives"
                )
            reaction = reactions[0]
            precursors = [
                child for child in reaction.get("children") or []
                if child.get("type") == "mol" and child.get("smiles")
            ]
            if not precursors:
                raise ValueError(f"{route_id}: reaction has no precursor molecule nodes")
            metadata = reaction.get("metadata") or {}
            mapped_reaction = str(metadata.get("smiles") or "")
            try:
                mapping_status = audit_atom_mapping(mapped_reaction)["status"]
            except Exception:
                mapping_status = "unmapped"
            steps.append(
                {
                    "product": str(molecule["smiles"]),
                    "reactants": [str(precursor["smiles"]) for precursor in precursors],
                    "reaction_id": str(
                        metadata.get("reaction_hash")
                        or metadata.get("ID")
                        or f"{route_id}:step:{len(steps)}"
                    ),
                    "mapping_status": mapping_status,
                }
            )
            for precursor in precursors:
                visit(precursor)

        visit(root)
        if not steps:
            raise ValueError(f"{route_id}: route contains no reaction steps")
        yield {
            "route_id": route_id,
            "target_smiles": str(root["smiles"]),
            "steps": steps,
            "source": [
                {
                    "name": SOURCE_META["paroutes_v2"][0],
                    "license": SOURCE_META["paroutes_v2"][1],
                    "record_id": f"{release}:{route_index}",
                    "group_id": route_id,
                }
            ],
            "annotations": {"release": release, "source_format": "paroutes_tree"},
        }


def adapt_ord(path: Path) -> Iterator[dict[str, Any]]:
    """Adapt ORD protobuf-in-Parquet while preserving reaction roles and patent IDs."""
    try:
        from ord_schema.datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the 'sources' extra to adapt ORD") from exc
    files = sorted(path.glob("data/**/*.parquet")) if path.is_dir() else [path]
    for parquet_path in files:
        dataset = load_dataset(str(parquet_path))
        for row_number, reaction in enumerate(dataset.reactions, 1):
            reactants: list[str] = []
            reagents: list[str] = []
            for reaction_input in reaction.inputs.values():
                for component in reaction_input.components:
                    smiles = _compound_smiles(component)
                    if not smiles:
                        continue
                    role = _enum_name(component, "reaction_role")
                    (reactants if "REACTANT" in role else reagents).append(smiles)
            products: list[str] = []
            for outcome in reaction.outcomes:
                for product in outcome.products:
                    compound = getattr(product, "compound", product)
                    smiles = _compound_smiles(compound)
                    if smiles:
                        products.append(smiles)
            if not reactants or not products:
                continue
            patent = getattr(getattr(reaction, "provenance", None), "patent", None)
            patent_id = _first_text_attr(
                patent, ("document_id", "patent_number", "title")
            )
            yield {
                "reactants": reactants,
                "reagents": reagents,
                "products": products,
                "source_name": SOURCE_META["ord"][0],
                "source_license": SOURCE_META["ord"][1],
                "source_record_id": str(getattr(reaction, "reaction_id", "") or f"{parquet_path.name}:{row_number}"),
                "source_group_id": patent_id,
                "evidence_kind": "observed",
                "annotations": {"ord_dataset_file": str(parquet_path.relative_to(path) if path.is_dir() else parquet_path.name)},
            }


def _compound_smiles(compound: Any) -> str | None:
    for identifier in getattr(compound, "identifiers", ()):
        name = _enum_name(identifier, "type")
        if "SMILES" in name and getattr(identifier, "value", ""):
            return str(identifier.value)
    return None


def _enum_name(message: Any, field: str) -> str:
    try:
        descriptor = message.DESCRIPTOR.fields_by_name[field]
        value = getattr(message, field)
        return descriptor.enum_type.values_by_number[value].name
    except Exception:
        return ""


def _first_text_attr(value: Any, names: Iterable[str]) -> str | None:
    if value is None:
        return None
    for name in names:
        result = getattr(value, name, None)
        if result:
            return str(result)
    return None


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern!r} under {root}, found {len(matches)}")
    return matches[0]


def write_jsonl(rows: Iterable[dict[str, Any]], output: Path, limit: int | None = None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if limit is not None and count >= limit:
                break
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=sorted(SOURCE_META))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release", help="CREED release or USPTO-LLM mode")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    adapters = {
        "crd": lambda: adapt_crd(args.input),
        "frea": lambda: adapt_frea(args.input),
        "creed": lambda: adapt_creed(args.input, release=args.release or "CREED-CCV"),
        "uspto_lowe": lambda: adapt_uspto_lowe(args.input),
        "uspto_llm": lambda: adapt_uspto_llm(args.input, release=args.release or "single_step"),
        "paroutes_v2": lambda: adapt_paroutes_v2(args.input, release=args.release or "n1"),
        "ord": lambda: adapt_ord(args.input),
    }
    count = write_jsonl(adapters[args.source](), args.output, args.limit)
    print(json.dumps({"source": args.source, "rows": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
