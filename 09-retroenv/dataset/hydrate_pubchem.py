#!/usr/bin/env python3
"""Hydrate name/CAS queries into a frozen PubChem cache before rollouts."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


PROPERTIES = "CanonicalSMILES,ConnectivitySMILES,IsomericSMILES,InChIKey,MolecularFormula,MolecularWeight"


def fetch(query: str) -> dict:
    encoded = urllib.parse.quote(query, safe="")
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}"
        f"/property/{PROPERTIES}/JSON"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "RetroEnv/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    row = payload["PropertyTable"]["Properties"][0]
    smiles = (
        row.get("ConnectivitySMILES")
        or row.get("CanonicalSMILES")
        or row.get("SMILES")
        or row.get("IsomericSMILES")
    )
    if not smiles:
        raise ValueError("PubChem response contains no canonical SMILES")
    return {
        "canonical_smiles": smiles,
        "cid": row.get("CID"),
        "inchikey": row.get("InChIKey"),
        "formula": row.get("MolecularFormula"),
        "molecular_weight": row.get("MolecularWeight"),
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{row.get('CID')}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path, help="One molecule name or CAS number per line")
    parser.add_argument("output", type=Path)
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()
    cache = {}
    if args.output.exists():
        cache = json.loads(args.output.read_text(encoding="utf-8"))
    failures = {}
    for line in args.queries.read_text(encoding="utf-8").splitlines():
        query = line.strip()
        if not query or query.startswith("#") or query.casefold() in cache:
            continue
        try:
            cache[query.casefold()] = fetch(query)
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            failures[query] = str(exc)
        time.sleep(max(0.0, args.delay))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cached": len(cache), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
