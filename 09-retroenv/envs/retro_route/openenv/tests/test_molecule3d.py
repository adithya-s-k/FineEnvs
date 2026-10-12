"""The playground's 3D conformers: deterministic, bounded, and safe to embed in a page."""

from __future__ import annotations

from retroenv_openenv.molecule3d import MAX_HEAVY_ATOMS, conformer_block, viewer_page


def test_a_conformer_is_deterministic_and_three_dimensional():
    conformer_block.cache_clear()
    first = conformer_block("CC(=O)Nc1ccc(O)cc1")
    conformer_block.cache_clear()
    assert conformer_block("CC(=O)Nc1ccc(O)cc1") == first
    coordinates = [line.split()[:3] for line in first.splitlines()[4:24]]
    assert any(abs(float(z)) > 0.1 for _, _, z in coordinates)  # not a flat 2D layout


def test_unparsable_oversized_and_empty_inputs_get_no_conformer():
    assert conformer_block("not a molecule") is None
    assert conformer_block("") is None
    assert conformer_block("C" * (MAX_HEAVY_ATOMS + 1)) is None


def test_the_page_cannot_be_broken_out_of():
    page = viewer_page("C")
    script = page.split("const block = ", 1)[1].split("\n", 1)[0]
    assert "</script" not in script
    assert "&lt;img" in viewer_page("C<img src=x onerror=alert(1)>") or "No 3D conformer" in viewer_page(
        "C<img src=x onerror=alert(1)>"
    )
