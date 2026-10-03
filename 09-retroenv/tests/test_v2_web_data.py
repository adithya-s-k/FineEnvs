from __future__ import annotations

import json

from v2.web.build_data import legacy_calls, legacy_tree_to_dag


def test_legacy_tree_display_conversion_preserves_branch_and_reasoning():
    submission = {
        "routes": [
            {
                "type": "mol",
                "smiles": "CCOC(C)=O",
                "in_stock": False,
                "children": [
                    {
                        "type": "reaction",
                        "is_reaction": True,
                        "metadata": {
                            "explanation": "Esterification of ethanol and acetic acid.",
                            "reaction_class": "esterification",
                            "confidence": 0.9,
                            "precursor_roles": {
                                "CCO": "alcohol",
                                "CC(=O)O": "carboxylic acid",
                            },
                        },
                        "children": [
                            {
                                "type": "mol",
                                "smiles": "CCO",
                                "in_stock": True,
                                "children": [],
                            },
                            {
                                "type": "mol",
                                "smiles": "CC(=O)O",
                                "in_stock": True,
                                "children": [],
                            },
                        ],
                    }
                ],
            }
        ]
    }
    graph = legacy_tree_to_dag(submission, "CCOC(C)=O")
    assert len(graph["routes"]) == 1
    assert len(graph["nodes"]) == 4
    assert len(graph["edges"]) == 3
    reactant_edges = [edge for edge in graph["edges"] if edge["type"] == "reactant"]
    assert {edge["metadata"]["role"] for edge in reactant_edges} == {
        "alcohol",
        "carboxylic acid",
    }
    assert all(edge["metadata"]["evidence_ids"] == [] for edge in graph["edges"])


def test_legacy_call_recovery_keeps_arguments_and_excludes_terminal_call():
    transcript = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "one",
                    "function": {
                        "name": "stock_retrieve",
                        "arguments": json.dumps({"query": "CCO", "mode": "exact"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "stock_retrieve",
            "tool_call_id": "one",
            "content": json.dumps({"results": [{"smiles": "CCO"}]}),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "two",
                    "function": {
                        "name": "emit_routes",
                        "arguments": json.dumps({"submission": {"routes": []}}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "emit_routes",
            "tool_call_id": "two",
            "content": json.dumps({"done": True}),
        },
    ]
    rows, terminal_calls = legacy_calls(transcript)
    assert terminal_calls == 1
    assert len(rows) == 1
    assert rows[0]["arguments"] == {"query": "CCO", "mode": "exact"}
    assert rows[0]["summary"]["hits"] == ["CCO"]

