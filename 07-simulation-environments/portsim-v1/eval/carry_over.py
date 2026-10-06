"""Copy episodes from an old run into a new one for tasks whose public content and references are unchanged.

    python eval/carry_over.py OLD_RUN NEW_RUN
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envs" / "berth_planning" / "core"))
from berth_core import load_pack  # noqa: E402


def main(old: str, new: str, old_pack: str):
    pack = load_pack()
    before = {t.task_id: t for t in load_pack(old_pack).tasks}
    src, dst = ROOT / "results" / "rollouts" / old, ROOT / "results" / "rollouts" / new
    n = 0
    for f in src.glob("*/*.json"):
        t, o = pack._by_id.get(f.stem), before.get(f.stem)
        if t is None or o is None or t.to_dict() != o.to_dict():
            continue
        rec = json.loads(f.read_text())
        if rec.get("errors"):
            continue
        rec["run"] = new
        (dst / f.parent.name).mkdir(parents=True, exist_ok=True)
        (dst / f.parent.name / f.name).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        n += 1
    print(f"carried {n} episodes into {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
