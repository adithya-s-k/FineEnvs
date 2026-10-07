"""Tool names, descriptions and JSON schemas shared by every RetroEnv front-end.

The OpenEnv server advertises these schemas over MCP, the v1 runner sends them
to OpenAI-compatible endpoints, and a Harbor CLI can document them, so one
definition keeps the action surface identical everywhere. Validation stays in
``RetroRouteSession`` and the verifier: a malformed submission is scored, not
rejected at the schema boundary, because parse and graph validity are part of
the dense reward.
"""

from __future__ import annotations

import copy
from typing import Any


def _function(
    name: str,
    description: str,
    properties: dict,
    required: list[str] | None = None,
    defs: dict[str, Any] | None = None,
) -> dict:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
    if defs:
        parameters["$defs"] = defs
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


SMILES = {"type": "string", "description": "A SMILES string."}
REACTANTS = {"type": "array", "items": SMILES, "minItems": 1}
ROUTE_GRAPH_DEFS: dict[str, Any] = {
    "metadata": {
        "type": "object",
        "description": "Evidence and confidence for this disconnection.",
        "properties": {
            "source": {"type": "string"},
            "explanation": {"type": "string"},
            "reaction_class": {"type": "string"},
            "classification": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "policy_probability": {"type": "number", "minimum": 0, "maximum": 1},
            "literature": {"type": "array", "items": {"type": "object"}},
            "conditions": {"type": "array", "items": {"type": "object"}},
            "precursor_roles": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": [
            "explanation",
            "reaction_class",
            "confidence",
            "literature",
            "precursor_roles",
        ],
        "additionalProperties": True,
    },
    "reaction": {
        "type": "object",
        "description": "A reaction whose children are its precursor molecules.",
        "properties": {
            "type": {"const": "reaction"},
            "is_reaction": {"const": True},
            "metadata": {"$ref": "#/$defs/metadata"},
            "children": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/mol"},
            },
        },
        "required": ["type", "is_reaction", "metadata", "children"],
        "additionalProperties": False,
    },
    "mol": {
        "type": "object",
        "description": "A molecule node. Expanded nodes have one reaction child and in_stock=false; leaves have no children.",
        "properties": {
            "type": {"const": "mol"},
            "smiles": {"type": "string", "minLength": 1},
            "in_stock": {"type": "boolean"},
            "children": {
                "type": "array",
                "maxItems": 1,
                "items": {"$ref": "#/$defs/reaction"},
            },
        },
        "required": ["type", "smiles", "in_stock", "children"],
        "additionalProperties": False,
    },
}

TOOLS = [
    _function("inspect_molecule", "Inspect a molecule with RDKit.", {"smiles": SMILES}, ["smiles"]),
    _function(
        "pubchem_lookup",
        "Canonicalize a SMILES or query the frozen molecule cache.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _function(
        "stock_retrieve",
        "The only stock access. Search exact SMILES/InChIKey, class, SMARTS, or similarity; at most 20 results.",
        {
            "query": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["auto", "exact", "inchikey", "class", "substructure", "similarity"],
                "default": "auto",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        },
        ["query"],
    ),
    _function(
        "reaction_precedent_search",
        "Find similar reactions in the train-visible corpus, ranked by product similarity.",
        {
            "product_smiles": SMILES,
            "reaction_class": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        },
    ),
    _function(
        "validate_disconnection",
        "Check whether train-visible reactions or frequent templates support a proposed product-to-reactants cut.",
        {"product_smiles": SMILES, "reactants": REACTANTS},
        ["product_smiles", "reactants"],
    ),
    _function(
        "reaction_class_lookup",
        "Name the reaction class of any proposed cut with the verifier's own labeller.",
        {"product_smiles": SMILES, "reactants": REACTANTS},
        ["product_smiles", "reactants"],
    ),
    _function(
        "reaction_conditions_search",
        "Reagent sets reported for train-visible analogues of a cut or product.",
        {
            "product_smiles": SMILES,
            "reactants": REACTANTS,
            "reaction_class": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
    ),
    _function(
        "search_literature",
        "Patent identifiers attached to train-visible analogue reactions.",
        {
            "product_smiles": SMILES,
            "reaction_class": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
    ),
    _function(
        "emit_routes",
        "Terminal call. Emit the required renderable molecule/reaction trees.",
        {
            "submission": {
                "type": "object",
                "properties": {
                    "schema_version": {"type": "string"},
                    "routes": {
                        "type": "array",
                        "description": "Root molecule nodes directly; do not wrap them in route/root/tree objects.",
                        "items": {"$ref": "#/$defs/mol"},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                },
                "required": ["routes"],
            }
        },
        ["submission"],
        defs=ROUTE_GRAPH_DEFS,
    ),
]
EMIT_TOOL = next(tool for tool in TOOLS if tool["function"]["name"] == "emit_routes")

ALL_TOOL_NAMES = tuple(tool["function"]["name"] for tool in TOOLS)
# No tool reads a task's hidden routes. "unaided" is an ablation without the step checker.
ASSIST_TOOLS = ("validate_disconnection",)
TOOLSETS: dict[str, tuple[str, ...]] = {
    "full": ALL_TOOL_NAMES,
    "unaided": tuple(name for name in ALL_TOOL_NAMES if name not in ASSIST_TOOLS),
}


def tool_names(toolset: str = "full") -> tuple[str, ...]:
    try:
        return TOOLSETS[toolset]
    except KeyError:
        raise ValueError(f"unknown toolset {toolset!r}; choose one of {sorted(TOOLSETS)}") from None


def openai_tools(toolset: str = "full") -> list[dict[str, Any]]:
    """OpenAI function-calling definitions for the tools in ``toolset``."""
    names = set(tool_names(toolset))
    return [copy.deepcopy(tool) for tool in TOOLS if tool["function"]["name"] in names]


def tool_spec(name: str) -> dict[str, Any]:
    """The ``{"name", "description", "parameters"}`` block of one tool."""
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return copy.deepcopy(tool["function"])
    raise KeyError(f"unknown tool {name!r}")
