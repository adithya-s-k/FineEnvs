"""3D molecule views for the playground: one RDKit conformer per molecule, drawn by 3Dmol.js.

The server embeds a single deterministic conformer (ETKDG, then MMFF) and returns a small
self-contained page that the playground shows in an iframe. 3Dmol.js is vendored under
``static/`` (BSD-3-Clause), so the view works on a laptop with no network. A conformer shows
a plausible shape; nothing in the environment or its reward depends on it.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from html import escape
from pathlib import Path
from urllib.parse import quote

from rdkit import Chem
from rdkit.Chem import AllChem

STATIC = Path(__file__).resolve().parent / "static"
MAX_SMILES = 500
MAX_HEAVY_ATOMS = 120
SEED = 0xF00D


@lru_cache(maxsize=512)
def conformer_block(smiles: str) -> str | None:
    """A MOL block with 3D coordinates and explicit hydrogens, or None when RDKit cannot embed it."""
    if not smiles or len(smiles) > MAX_SMILES:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True  # strained or caged systems often need random starts
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return Chem.MolToMolBlock(mol)


_PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D view</title>
<style>
  :root { color-scheme: light dark; --text: #262626; --muted: #737373; --line: #d4d4d4; }
  @media (prefers-color-scheme: dark) { :root { --text: #e5e5e5; --muted: #a3a3a3; --line: #404040; } }
  html, body { margin: 0; height: 100%; overflow: hidden; background: transparent; color: var(--text);
    font: 12px/1.4 Inter, ui-sans-serif, system-ui, sans-serif; }
  #view { position: absolute; inset: 0 0 26px 0; }
  #bar { position: absolute; left: 0; right: 0; bottom: 0; height: 26px; display: flex; gap: 12px;
    align-items: center; color: var(--muted); }
  #bar label, #bar span { display: flex; gap: 4px; align-items: center; white-space: nowrap; }
  @media (max-width: 380px) { #bar .hint { display: none; } }
  select { font: inherit; color: inherit; background: transparent; border: 1px solid var(--line);
    border-radius: 4px; padding: 1px 4px; }
  #empty { padding: 12px 0; color: var(--muted); }
</style>
<body>
__BODY__
</body>
</html>
"""

_VIEWER = """<div id="view" aria-label="3D structure of __SMILES__"></div>
<div id="bar">
  <label>Style
    <select id="style">
      <option value="ballstick">Ball and stick</option>
      <option value="stick">Stick</option>
      <option value="sphere">Space filling</option>
    </select>
  </label>
  <label><input type="checkbox" id="hydrogens"> Hydrogens</label>
  <span class="hint">Drag to rotate, scroll to zoom</span>
</div>
<script src="/molecule/static/3Dmol-min.js"></script>
<script>
  const block = __BLOCK__;
  const viewer = $3Dmol.createViewer("view", { backgroundAlpha: 0, antialias: true });
  viewer.addModel(block, "mol");
  const styles = {
    ballstick: { stick: { radius: 0.14 }, sphere: { scale: 0.24 } },
    stick: { stick: { radius: 0.18 } },
    sphere: { sphere: { scale: 1.0 } },
  };
  function draw() {
    const style = styles[document.getElementById("style").value];
    const showH = document.getElementById("hydrogens").checked;
    viewer.setStyle({}, style);
    if (!showH) viewer.setStyle({ elem: "H" }, {});
    viewer.render();
  }
  document.getElementById("style").addEventListener("change", draw);
  document.getElementById("hydrogens").addEventListener("change", draw);
  draw();
  viewer.zoomTo();
  viewer.render();
  // A tab that starts hidden gives the canvas no size; refit once it is shown.
  window.addEventListener("resize", () => { viewer.resize(); viewer.zoomTo(); viewer.render(); });
</script>"""


# Pages are cached by browsers for a day; changing the template changes this, which changes the URL.
VERSION = hashlib.sha256((_PAGE + _VIEWER).encode()).hexdigest()[:8]


def viewer_url(smiles: str) -> str:
    return f"/molecule/3d?smiles={quote(smiles, safe='')}&v={VERSION}"


def viewer_page(smiles: str) -> str:
    block = conformer_block(smiles)
    if block is None:
        body = '<p id="empty">No 3D conformer for this molecule.</p>'
    else:
        # json.dumps quotes the block for JavaScript; "</" is split so it cannot end the script.
        body = _VIEWER.replace("__BLOCK__", json.dumps(block).replace("</", "<\\/")).replace(
            "__SMILES__", escape(smiles)
        )
    return _PAGE.replace("__BODY__", body)
