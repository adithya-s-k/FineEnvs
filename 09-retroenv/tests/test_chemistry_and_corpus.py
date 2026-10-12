from __future__ import annotations

import json
import zipfile

import pytest
from retroenv.adapters import adapt_crd, adapt_frea, adapt_paroutes_v2, adapt_uspto_llm
from retroenv.chemistry import (
    ChemistryError,
    audit_atom_mapping,
    canonicalize_smiles,
    inspect_molecule,
)
from retroenv.corpus import build_corpus, normalize_row


def test_canonicalization_preserves_stereochemistry():
    left = canonicalize_smiles("N[C@@H](C)C(=O)O")
    right = canonicalize_smiles("N[C@H](C)C(=O)O")
    assert left != right


def test_invalid_valence_is_rejected():
    with pytest.raises(ChemistryError):
        canonicalize_smiles("C(C)(C)(C)(C)C")


def test_molecule_inspection_is_deterministic():
    result = inspect_molecule("Oc1ccccc1")
    assert result["valid"] is True
    assert result["formula"] == "C6H6O"
    assert result["murcko_scaffold"] == "c1ccccc1"


def test_complete_atom_mapping_is_audited():
    reaction = "[CH3:1][CH2:2][OH:3].[CH3:4][C:5](=[O:6])[OH:7]>>[CH3:1][CH2:2][O:3][C:5](=[O:6])[CH3:4]"
    audit = audit_atom_mapping(reaction)
    assert audit["status"] == "complete"
    assert audit["product_coverage"] == 1.0


def test_changed_mapped_element_is_rejected():
    audit = audit_atom_mapping("[CH3:1]>>[NH2:1]")
    assert audit["status"] == "invalid"
    assert "element/isotope" in audit["errors"][0]


def test_normalization_requires_a_data_license():
    with pytest.raises(ValueError, match="source_license"):
        normalize_row(
            {"reaction_smiles": "CCO.CC(=O)O>>CCOC(C)=O"},
            source_name="test",
        )


def test_corpus_merges_duplicate_provenance(tmp_path):
    source = tmp_path / "raw.jsonl"
    rows = [
        {
            "reaction_smiles": "CCO.CC(=O)O>>CCOC(C)=O",
            "source_name": "one",
            "source_license": "CC0-1.0",
            "source_record_id": "a",
        },
        {
            "reaction_smiles": "CC(=O)O.CCO>>CCOC(C)=O",
            "source_name": "two",
            "source_license": "CC-BY-4.0",
            "source_record_id": "b",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output, rejects = tmp_path / "corpus.jsonl", tmp_path / "rejects.jsonl"
    report = build_corpus(
        [source],
        output,
        rejects,
        source_name=None,
        source_license=None,
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["counts"]["duplicates_merged"] == 1
    assert len(records) == 1
    assert {item["name"] for item in records[0]["source"]} == {"one", "two"}


def test_crd_adapter_streams_zip_rows(tmp_path):
    archive = tmp_path / "crd.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("reactions.txt", "CCO.CC(=O)O>>CCOC(C)=O\n")
    rows = list(adapt_crd(archive))
    assert rows[0]["source_license"] == "CC-BY-4.0"
    assert rows[0]["reaction_smiles"].endswith(">CCOC(C)=O")


def test_frea_adapter_keeps_generated_negatives_labelled(tmp_path):
    source = tmp_path / "rxnverif_v1.0.csv"
    source.write_text("reactants,product,method,feasible\nCCO,CC=O,positive,1\nCCN,CC=O,RR,0\n")
    rows = list(adapt_frea(source))
    assert rows[0]["evidence_kind"] == "observed"
    assert rows[1]["evidence_kind"] == "generated"
    assert rows[1]["annotations"]["generation_method"] == "RR"


def test_uspto_llm_adapter_streams_mapped_zip_and_groups_patents(tmp_path):
    archive = tmp_path / "single_step.zip"
    member = "single_step/HGAR/uspto_llm_atommap.csv"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            member,
            "id,rs>>ps,solvents,catalyst,temperature,time\n"
            "20150101-US123A1-0042,[CH3:1][OH:2]>>[CH2:1]=[O:2],['O'],[],['20'],['30']\n",
        )
    rows = list(adapt_uspto_llm(archive, release="single_step"))
    assert rows[0]["source_group_id"] == "US123A1"
    assert rows[0]["publication_year"] == 2015
    assert rows[0]["source_license"] == "CC-BY-4.0"
    assert rows[0]["annotations"]["llm_extracted"] is True


def test_paroutes_adapter_preserves_tree_connectivity_and_route_provenance(tmp_path):
    root = {
        "type": "mol",
        "smiles": "CCOC(C)=O",
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "metadata": {
                    "reaction_hash": "fixture-hash",
                    "smiles": "[CH3:1][CH2:2][OH:3].[CH3:4][C:5](=[O:6])[OH:7]>>"
                    "[CH3:1][CH2:2][O:3][C:5](=[O:6])[CH3:4]",
                },
                "children": [
                    {"type": "mol", "smiles": "CCO", "in_stock": True},
                    {"type": "mol", "smiles": "CC(=O)O", "in_stock": True},
                ],
            }
        ],
    }
    (tmp_path / "ref_routes_n1.json").write_text(json.dumps([root]))
    rows = list(adapt_paroutes_v2(tmp_path, release="n1"))
    assert rows[0]["target_smiles"] == "CCOC(C)=O"
    assert rows[0]["steps"][0]["reactants"] == ["CCO", "CC(=O)O"]
    assert rows[0]["steps"][0]["mapping_status"] == "complete"
    assert rows[0]["source"][0]["license"] == "CC-BY-4.0"
