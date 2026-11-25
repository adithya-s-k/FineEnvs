"""HTML for the playground: molecule drawings, route trees and score tables."""

from __future__ import annotations

import html
from functools import lru_cache
from typing import Any

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .molecule3d import viewer_url

STYLE = """
<style>
/* OpenEnv mounts the playground with its own Gradio theme (a green primary and accent);
   Gradio 6 ignores a theme passed to Blocks, so these variables give it neutral greys. */
body .gradio-container {
  --button-primary-background-fill: #171717; --button-primary-background-fill-hover: #404040;
  --button-primary-border-color: #171717; --button-primary-text-color: #ffffff;
  --color-accent: #404040; --color-accent-soft: #f5f5f5; --border-color-accent: #a3a3a3;
  --checkbox-background-color-selected: #262626; --loader-color: #525252; --slider-color: #525252;
}
body.dark .gradio-container {
  --button-primary-background-fill: #30363d; --button-primary-background-fill-hover: #484f58;
  --button-primary-border-color: #484f58; --button-primary-text-color: #e6edf3;
  --color-accent: #c9d1d9; --color-accent-soft: #30363d; --border-color-accent: #6e7781;
  --checkbox-background-color-selected: #8b949e; --loader-color: #8b949e; --slider-color: #8b949e;
}
/* Gradio's own chrome: its footer advertises Gradio rather than the environment. */
footer { display: none !important; }
.gradio-container { max-width: 1280px !important; }
.gradio-container .block { box-shadow: none; }
.gradio-container .tabs { border: none; }
.gradio-container .tab-nav button { font-weight: 500; }
.gradio-container .retro { font-size: 13px; line-height: 1.5; color: var(--body-text-color); }
.gradio-container .retro .muted { color: var(--body-text-color-subdued); }
.gradio-container .retro a { color: inherit; text-decoration: underline; text-underline-offset: 2px;
  text-decoration-color: var(--body-text-color-subdued); }
.gradio-container .retro .mono { font-family: var(--font-mono); font-size: 12px; overflow-wrap: anywhere; }
.gradio-container .retro .mol { border: 1px solid var(--border-color-primary); border-radius: 6px;
  display: inline-block; line-height: 0; padding: 2px; color: var(--body-text-color); }
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
.gradio-container .retro .views { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.gradio-container .retro .views .mol { display: block; }
.gradio-container .retro .views .mol svg { width: 100%; height: auto; }
.gradio-container .retro .frame { border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 4px; }
.gradio-container .retro .frame iframe { display: block; width: 100%; border: 0; background: transparent; }
.gradio-container .retro .caption { margin: 2px 0 6px; color: var(--body-text-color-subdued); font-size: 12px; }
</style>
"""


# Heteroatoms in mid-tone element colours that read on light and dark backgrounds; carbon
# and every bond stay black in the SVG and are then switched to the page's text colour.
ATOM_COLOURS = {
    5: (0.83, 0.48, 0.48),  # B
    7: (0.24, 0.44, 0.85),  # N
    8: (0.88, 0.28, 0.23),  # O
    9: (0.23, 0.65, 0.33),  # F
    15: (0.88, 0.54, 0.12),  # P
    16: (0.79, 0.64, 0.15),  # S
    17: (0.23, 0.65, 0.33),  # Cl
    35: (0.71, 0.40, 0.23),  # Br
    53: (0.56, 0.36, 0.71),  # I
}


@lru_cache(maxsize=512)
def molecule_svg(smiles: str, width: int = 240, height: int = 160) -> str:
    """A 2D drawing whose bonds follow the page theme and whose heteroatoms keep element colours."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return '<span class="muted">unparsable SMILES</span>'
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    options.bondLineWidth = 1.4
    options.useBWAtomPalette()
    options.updateAtomPalette(ATOM_COLOURS)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("#000000", "currentColor")
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
    return wrap(key_values(rows))


def structure_2d(smiles: str | None, width: int = 420, height: int = 280) -> str:
    if not smiles:
        return wrap('<p class="muted">Start an episode to see the target.</p>')
    return wrap(f'<div class="views">{molecule_svg(smiles, width, height)}</div>')


def structure_3d(smiles: str | None, height: int = 300) -> str:
    """The 3D view: an iframe onto the server's conformer page, loaded when it is shown."""
    if not smiles:
        return wrap('<p class="muted">Start an episode to see the target.</p>')
    return wrap(
        f'<div class="frame"><iframe title="3D structure" loading="lazy" height="{height}" '
        f'src="{viewer_url(smiles)}"></iframe></div>'
        '<p class="caption">One RDKit conformer (ETKDG, then MMFF), for shape only.</p>'
    )


def molecule_views(smiles: str | None) -> str:
    """The molecule under study, 2D beside 3D."""
    if not smiles:
        return wrap("")
    return wrap(
        f'<p class="caption mono">{html.escape(smiles)}</p>'
        f'<div class="views">{molecule_svg(smiles, 360, 240)}'
        f'<div class="frame"><iframe title="3D structure" loading="lazy" height="240" '
        f'src="{viewer_url(smiles)}"></iframe></div></div>'
    )


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


def plan_panel(target: str, plan: dict[str, tuple[str, ...]], stock: dict[str, bool]) -> str:
    """The route being assembled by hand, with each precursor's stock status as the tools reported it."""
    if not target:
        return wrap('<p class="muted">Start an episode to plan a route.</p>')

    def node(smiles: str, depth: int = 0) -> str:
        precursors = plan.get(smiles) if depth <= 12 else None
        if precursors:
            label = "to make"
        elif stock.get(smiles):
            label = '<span class="dot ok"></span>in stock'
        elif smiles in stock:
            label = '<span class="dot bad"></span>not in stock, not yet cut'
        else:
            label = '<span class="dot"></span>stock not checked'
        out = (
            f'<div class="node">{molecule_svg(smiles, 140, 95)}'
            f'<div><div class="mono">{html.escape(smiles)}</div><div class="muted">{label}</div></div></div>'
        )
        for precursor in precursors or ():
            out += f'<div class="tree">{node(precursor, depth + 1)}</div>'
        return out

    return wrap(node(target))


def verdict_panel(result: dict[str, Any]) -> str:
    """What validate_disconnection said about one proposed cut."""
    if not isinstance(result, dict) or not result:
        return wrap("")
    if result.get("error"):
        return wrap(f'<p><span class="dot bad"></span>{html.escape(str(result["error"]))}</p>')
    supported = bool(result.get("supported"))
    kind = {"known_precedent": "a recorded reaction", "reaction_template": "a frequent reaction template"}.get(
        str(result.get("support")), str(result.get("support"))
    )
    head = (
        f'<span class="dot ok"></span>Supported by {html.escape(kind)}'
        if supported
        else '<span class="dot bad"></span>The train-visible evidence does not support this cut'
    )
    rows = [("Reaction class", html.escape(str(result.get("reaction_class") or "unclassified")))]
    if result.get("template_frequency"):
        rows.append(("Template seen", f"{result['template_frequency']} times"))
    if result.get("stereo_consistent") is not None:
        rows.append(("Stereochemistry", "agrees" if result["stereo_consistent"] else "differs"))
    return wrap(
        f"<p style='margin:0 0 6px'>{head}</p>"
        + key_values(rows)
        + '<p class="muted" style="margin:6px 0 0">The final verifier uses a larger library, so it may still '
        "accept a cut this call rejects.</p>"
    )


def stock_panel(found: list[tuple[str, dict[str, Any]]]) -> str:
    """Stock lookups for the precursors of one proposed cut."""
    rows = []
    for smiles, result in found:
        held = bool(result.get("results"))
        status = '<span class="dot ok"></span>in stock' if held else '<span class="dot bad"></span>not in stock'
        rows.append(
            f"<tr><td>{molecule_svg(smiles, 110, 70)}</td>"
            f'<td class="mono">{html.escape(smiles)}</td><td>{status}</td></tr>'
        )
    return wrap(f"<table><tbody>{''.join(rows)}</tbody></table>")


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


DATASET = "LiteFold/RetroEnv"
DATASET_URL = "https://huggingface.co/datasets/LiteFold/RetroEnv"


def _link(url: str, text: str) -> str:
    return f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(text)}</a>'


def credit_line() -> str:
    """The one-line credit under the title: whose dataset this environment serves."""
    return (
        f"Tasks and known routes from {_link(DATASET_URL, DATASET)}, the dataset built and released by "
        f"{_link('https://huggingface.co/LiteFold', 'LiteFold')} from PaRoutes v2 (CC BY 4.0). Credits at the "
        "bottom of the page."
    )


def credits_panel(source: dict[str, Any] | None, snapshot: dict[str, Any] | None) -> str:
    """Where the data comes from, for the playground's footer; values come from the served manifests."""
    source, snapshot = source or {}, snapshot or {}
    revision = (snapshot.get("source") or {}).get("revision")
    dataset = _link(DATASET_URL, DATASET) + (
        f" at revision <span class='mono'>{revision[:12]}</span>" if revision else ""
    )
    rows = [
        (
            "Dataset",
            f"{dataset}, built and released by {_link('https://huggingface.co/LiteFold', 'LiteFold')}: every "
            "task, known route and difficulty label, the stock and the reaction library served here",
        ),
        (
            "Routes",
            f"{html.escape(source.get('name', 'PaRoutes v2'))}: S. Genheden and E. Bjerrum, PaRoutes: towards a "
            "framework for benchmarking retrosynthesis route predictions, <i>Digital Discovery</i> 2022 "
            f"({_link('https://doi.org/10.1039/D2DD00015F', 'doi:10.1039/D2DD00015F')}); data "
            f"{_link(source.get('url', 'https://zenodo.org/records/7341155'), 'Zenodo 7341155')}, "
            f"{html.escape(source.get('license', 'CC-BY-4.0'))}",
        ),
        (
            "Reactions",
            "D. Lowe, Chemical reactions from US patents (1976 to Sep 2016), "
            f"{_link('https://doi.org/10.6084/m9.figshare.5104873.v1', 'figshare')}, CC0",
        ),
        ("Software", "RDKit, rdchiral, 3Dmol.js and OpenEnv"),
    ]
    if snapshot.get("bucket_id"):
        bucket = snapshot["bucket_id"]
        rows.insert(
            3,
            (
                "Served from",
                f"{_link(f'https://huggingface.co/buckets/{bucket}', bucket)}, snapshot "
                f"<span class='mono'>{html.escape(snapshot.get('snapshot_id', '')[:12])}</span>",
            ),
        )
    return wrap("<h4 style='margin:0 0 6px'>Data and credits</h4>" + key_values(rows))


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
