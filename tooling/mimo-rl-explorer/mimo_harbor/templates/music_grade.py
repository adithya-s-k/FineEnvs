"""Score a Music task as the explorer does (app/runner/domains.py Music._score, Xiaomi's music scorer).

The reply is /app/answer.md; if the agent wrote none, its final message from the OpenCode log stands in, so an
agent that answered in text is graded like the single-completion original."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
from music_scorer.pipeline import do, extract_abc  # noqa: E402

V = Path("/logs/verifier")
V.mkdir(parents=True, exist_ok=True)


def reply() -> str:
    a = Path("/app/answer.md")
    if a.exists() and a.read_text(errors="replace").strip():
        return a.read_text(errors="replace")
    text = ""
    for log in sorted(Path("/logs/agent").glob("*.txt")):
        for line in log.read_text(errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "text" and (ev.get("part") or {}).get("text"):
                text = ev["part"]["text"]
    return text


def done(reward: float, why: str, **extra) -> None:
    (V / "reward.txt").write_text(f"{reward}\n")
    (V / "result.json").write_text(json.dumps({"reward": reward, "summary": why, **extra}, indent=1, default=str))
    print(f"reward {reward}: {why}")
    sys.exit(0)


text = reply()
if not text.strip():
    done(0.0, "The agent wrote no answer.")
abc = extract_abc(text)
if not abc:
    done(0.0, "No ABC notation found.")
(V / "piece.abc").write_text(abc)
res = do({"key": "rollout", "id": 0, "rep": 0, "abc": abc, "tag": None, "lang": None, "abc_len": len(abc),
          "nvoice": 0, "latency": None})
if res.get("skip"):
    done(0.0, {"no_midi": "abc2midi could not turn the notation into MIDI."}.get(res["skip"], str(res["skip"])))
gate = {"notation_errors": res.get("err", 0), "bad_bars": res.get("bar", 0), "blank_lines": bool(res.get("blank")),
        "channel_conflicts": res.get("ch_conflict", 0)}
total = res.get("total")
if res.get("reject"):
    done(0.0, "Rejected by the validity gate", gate=gate, quality=total, groups=res.get("groups"))
if total is None:   # pipeline.compute_score: no total -> 0.0
    done(0.0, "The scorer could not measure this piece (no total).", gate=gate, groups=res.get("groups"))
done(max(0.0, min(1.0, float(total) / 100.0)), f"human-likeness {total:.1f}/100", gate=gate, groups=res.get("groups"))
