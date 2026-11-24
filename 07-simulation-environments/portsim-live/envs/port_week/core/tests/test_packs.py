from berth_core import load_pack

from port_week_core import marine, packs, wind


def test_the_live_weeks_are_berth_cores_packs():
    assert packs.live_pack() is load_pack()
    assert [r.name for r in packs.live_pack().roots] == ["dock-v1-eval", "dock-v1-train"]


def test_the_other_packs_and_the_references_sit_under_port_week_tasks_dir_else_the_tasks_root(monkeypatch, tmp_path):
    monkeypatch.setenv("PORT_WEEK_TASKS_DIR", str(tmp_path))
    assert (packs.tasks_root(), packs.references("live"), marine.pack_dir(), wind.pack_dir()) == (
        tmp_path, tmp_path / "live" / "references.jsonl", tmp_path / "dock-v1-marine", tmp_path / "dock-v1-wind")
    for unset in (monkeypatch.delenv, lambda name: monkeypatch.setenv(name, "")):
        unset("PORT_WEEK_TASKS_DIR")
        assert packs.tasks_root() == packs.TASKS_ROOT
        assert packs.references("marine") == packs.TASKS_ROOT / "marine" / "references.jsonl"
