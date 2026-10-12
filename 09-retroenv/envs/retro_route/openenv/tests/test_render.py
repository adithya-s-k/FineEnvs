"""The playground credits LiteFold's dataset, and what it derives from, on every page."""

from __future__ import annotations

from retroenv_openenv import render

SOURCE = {"name": "PaRoutes v2", "url": "https://zenodo.org/records/7341155", "license": "CC-BY-4.0"}
SNAPSHOT = {
    "bucket_id": "FineEnvs/retroenv-bucket",
    "snapshot_id": "350672d1218e" + "0" * 52,
    "source": {"repo": "LiteFold/RetroEnv", "revision": "7ca4c3ad54ce6223435fd6b77c79018641c8606b"},
}


def test_the_title_line_credits_litefolds_dataset():
    line = render.credit_line()
    assert "https://huggingface.co/datasets/LiteFold/RetroEnv" in line and "LiteFold" in line
    assert "PaRoutes v2" in line and "CC BY 4.0" in line


def test_the_credits_name_the_dataset_first_with_what_is_served():
    panel = render.credits_panel(SOURCE, SNAPSHOT)
    assert panel.index("LiteFold/RetroEnv") < panel.index("PaRoutes")
    assert "7ca4c3ad54ce" in panel and "350672d1218e" in panel
    assert "doi.org/10.1039/D2DD00015F" in panel and "zenodo.org/records/7341155" in panel


def test_a_local_release_without_a_snapshot_still_credits_the_dataset():
    panel = render.credits_panel(SOURCE, None)
    assert "LiteFold/RetroEnv" in panel and "Served from" not in panel
