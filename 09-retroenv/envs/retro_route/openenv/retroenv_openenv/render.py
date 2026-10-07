"""HTML for the playground: molecule drawings, route trees and score tables."""

from __future__ import annotations

import html
from functools import lru_cache
from typing import Any

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

STYLE = """
<style>
.gradio-container .retro { font-size: 13px; line-height: 1.5; color: var(--body-text-color); }
.gradio-container .retro .muted { color: var(--body-text-color-subdued); }
.gradio-container .retro .mono { font-family: var(--font-mono); font-size: 12px; overflow-wrap: anywhere; }
.gradio-container .retro .mol { background: #fff; border: 1px solid var(--border-color-primary);
  border-radius: 6px; display: inline-block; line-height: 0; }
.gradio-container .retro table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
.gradio-container .retro th, .gradio-container .retro td { text-align: left; padding: 4px 8px;
  border-bottom: 1px solid var(--border-color-primary); vertical-align: top; }
.gradio-container .retro th { color: var(--body-text-color-subdued); font-weight: 500; }
.gradio-container .retro td.num { text-align: right; }
.gradio-container .retro .kv { display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; }
.gradio-container .retro .kv dt { color: var(--body-text-color-subdued); }
.gradio-container .retro .kv dd { margin: 0; min-width: 0; }
.gradio-container .retro .tree { margin-left: 18px; border-left: 1px solid var(--border-color-primary); padding-left: 12px; }
.gradio-container .retro .node { display: flex; gap: 10px; align-items: center; margin: 6px 0; }
.gradio-container .retro .rxn { margin: 4px 0 4px 6px; color: var(--body-text-color-subdued); }
.gradio-container .retro .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  margin-right: 5px; vertical-align: 1px; background: var(--body-text-color-subdued); }
.gradio-container .retro .dot.ok { background: #15803d; }
.gradio-container .retro .dot.bad { background: #c2410c; }
</style>
"""


@lru_cache(maxsize=512)
def molecule_svg(smiles: str, width: int = 240, height: int = 160) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return '<span class="muted">unparsable SMILES</span>'
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return f'<span class="mol">{svg[svg.find("<svg") :]}</span>'


def wrap(body: str) -> str:
    return f'<div class="retro">{body}</div>'


def key_values(rows: list[tuple[str, str]]) -> str:
    items = "".join(f"<dt>{html.escape(k)}</dt><dd>{v}</dd>" for k, v in rows)
    return f'<dl class="kv">{items}</dl>'


def target_panel(opening: dict[str, Any] | None, state: Any = None) -> str:
    if not opening:
        return wrap('<p class="muted">Choose a split and task, then start an episode.</p>')
    remaining = opening["max_tool_calls"] - (state.tool_calls if state else 0)
    rows = [
        ("Task", f'<span class="mono">{html.escape(opening["task_id"])}</span>'),
        ("Target", f'<span class="mono">{html.escape(opening["target_smiles"])}</span>'),
        ("Depth", f"at most {opening['max_depth']} reactions (longest linear sequence)"),
        ("Variant", html.escape(opening.get("variant", "standard"))),
        ("Routes", f"{opening['min_routes']} to {opening['max_routes']}"),
        ("Stock", html.escape(opening["stock_id"])),
        ("Tool calls left", f"{remaining} of {opening['max_tool_calls']}"),
    ]
    if state is not None and state.done:
        rows.append(("Reward", f"{state.reward:.3f}" if state.reward is not None else "not scored"))
    return wrap(molecule_svg(opening["target_smiles"], 300, 200) + key_values(rows))


def _route_node(node: Any, depth: int = 0) -> str:
    if not isinstance(node, dict) or depth > 12:
        return '<div class="muted">not a molecule node</div>'
    smiles = str(node.get("smiles", ""))
    stock = node.get("in_stock")
    label = "in stock" if stock else ("intermediate" if node.get("children") else "leaf, not claimed in stock")
    out = (
        f'<div class="node">{molecule_svg(smiles, 150, 100)}'
        f'<div><div class="mono">{html.escape(smiles)}</div><div class="muted">{label}</div></div></div>'
    )
    for reaction in node.get("children") or []:
        if not isinstance(reaction, dict):
            continue
        metadata = reaction.get("metadata") or {}
        title = html.escape(str(metadata.get("reaction_class") or "reaction"))
        note = html.escape(str(metadata.get("explanation") or "")[:240])  # slice first: never split an entity
        children = "".join(_route_node(child, depth + 1) for child in reaction.get("children") or [])
        out += f'<div class="tree"><div class="rxn">{title}{" · " + note if note else ""}</div>{children}</div>'
    return out


def routes_panel(submission: Any) -> str:
    routes = submission.get("routes") if isinstance(submission, dict) else None
    if not routes:
        return wrap('<p class="muted">Submitted routes appear here.</p>')
    blocks = "".join(f"<h4>Route {i + 1}</h4>{_route_node(route)}" for i, route in enumerate(routes))
    return wrap(blocks)


def score_panel(score: dict[str, Any] | None) -> str:
    if not score:
        return wrap('<p class="muted">Emit routes to score them.</p>')
    valid = score.get("valid")
    status = f'<span class="dot {"ok" if valid else "bad"}"></span>{"Pass" if valid else "Fail"}'
    rows = "".join(
        f"<tr><td>{html.escape(name.replace('_', ' '))}</td><td class='num'>{value:.3f}</td></tr>"
        for name, value in sorted((score.get("components") or {}).items())
    )
    failures = "".join(f"<li>{html.escape(str(f))}</li>" for f in (score.get("hard_failures") or [])[:6])
    return wrap(
        key_values(
            [
                ("Result", status),
                ("Reward", f"{score.get('reward', 0.0):.3f}"),
                ("Tier", html.escape(str(score.get("verification_tier")))),
            ]
        )
        + f"<table><thead><tr><th>Component</th><th class='num'>Score</th></tr></thead><tbody>{rows}</tbody></table>"
        + (f"<p class='muted'>Why it failed</p><ul>{failures}</ul>" if failures else "")
    )


def history_panel(history: list[dict[str, Any]]) -> str:
    if not history:
        return wrap('<p class="muted">Tool calls appear here.</p>')
    rows = "".join(
        f"<tr><td class='num'>{i + 1}</td><td class='mono'>{html.escape(item['tool'])}</td>"
        f"<td class='mono'>{html.escape(item['arguments'][:120])}</td><td class='mono'>{html.escape(item['summary'][:160])}</td></tr>"
        for i, item in enumerate(history)
    )
    return wrap(
        f"<table><thead><tr><th class='num'>#</th><th>Tool</th><th>Arguments</th><th>Result</th></tr></thead><tbody>{rows}</tbody></table>"
    )
