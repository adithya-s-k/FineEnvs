import importlib.util
import json

from program_helpers import TOOLS

SPEC = importlib.util.spec_from_file_location("passenger_plans", TOOLS / "passenger_plans.py")
pax_step0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pax_step0)


def row(date: str, v2: float, pax: float, split: str = "eval") -> dict:
    return {"date": date, "split": split, "ratio": v2 / pax, "G1": v2 >= pax_step0.G1_RATIO * pax, "won": "incumbent",
            "rebooking": {"share": 0.02}, "sensitivity": {"0.5": {"ratio": 1.01}, "2": {"ratio": 1.1}},
            "mip": {"status": "Time limit reached", "gap_pct": 3.0, "wall_s": 1790.0}}


def test_g1_and_its_ratios_are_recomputed_from_the_per_date_files_alone(tmp_path):
    days = tmp_path / "dates"
    days.mkdir()
    for n in range(24):
        date = f"2024-07-{n + 1:02d}"
        pax_step0.path(days, date).write_text(json.dumps(row(date, 106 if n < 12 else 104, 100)))
    pax_step0.path(days, "2024-06-01").write_text(json.dumps(row("2024-06-01", 200, 100, "train")))
    g = pax_step0.guards(tmp_path)
    assert (g["G1"]["dates"], g["G1"]["passing"], g["G1"]["pass"]) == (24, 12, True)
    assert g["G1"]["ratios"]["2024-07-01"] == 1.06 and "2024-06-01" not in g["G1"]["ratios"]
    assert g["G2_reported"]["2024-06-01"]["gap_pct"] == 3.0 and g["won"] == {"incumbent": 25}
    pax_step0.path(days, "2024-07-01").write_text(json.dumps(row("2024-07-01", 105, 101)))
    assert not pax_step0.guards(tmp_path)["G1"]["pass"]
    assert "G1: FAIL, 11 of 24" in pax_step0.report(pax_step0.guards(tmp_path))
