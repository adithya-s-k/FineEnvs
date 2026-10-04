"""Grade a General (simulated workplace) task the way verl's general_agent environment does.

1. Every system must still be up; a dead backend is a testbed failure, not the model's (not scored).
2. The agent's final message is handed to the verifier in the session shape run_verify.py reads, and becomes
   answer.md if the agent wrote none (_write_answer_md).
3. The task's verifier files are put in place only now, and run_verify.py runs with the judge settings from
   [verifier.env].
4. Masking: a reward that is missing, not a number, or outside [0, 1] means the testbed failed, so no reward is
   written and Harbor records the trial as an error rather than a 0.
"""
import json
import math
import re
import os
import shutil
import subprocess
import sys
from pathlib import Path

T = Path("/tests")
cfg = json.loads((T / "grade.json").read_text())
V = Path("/logs/verifier")
V.mkdir(parents=True, exist_ok=True)


# run_verify.py prints the first characters of the judge key; Harbor scrubs whole secrets only, so redact here
# (the explorer's Rollout._redact pattern).
SECRET = re.compile(r"\b(hf_|sk-|sk_|api_key=|Bearer )(?=[A-Za-z0-9\-]*[A-Z0-9])[A-Za-z0-9\-]{4,}(?![a-z_])")
KEY_ECHO = re.compile(r"key=\S+")


def redact(text: str) -> str:
    return KEY_ECHO.sub("key=[redacted]", SECRET.sub(lambda m: m.group(1) + "[redacted]", text))


def scrub_outputs() -> None:
    for f in V.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".txt", ".log", ".md"):
            try:
                t = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if redact(t) != t:
                f.write_text(redact(t))


def port_up(p: int) -> bool:
    return subprocess.run(["bash", "-c", f"(echo > /dev/tcp/127.0.0.1/{int(p)}) 2>/dev/null"]).returncode == 0


def final_message() -> str:
    """The agent's last text, from Harbor's OpenCode log (/logs/agent/opencode.txt, one JSON event per line)."""
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


def fail(why: str) -> None:
    scrub_outputs()
    for f in ("reward.json", "reward.txt"):
        (V / f).unlink(missing_ok=True)
    print(f"not scored: {why}", file=sys.stderr)
    sys.exit(1)


dead = [p for p in cfg["wait_ports"] if not port_up(p)]
if dead:
    fail(f"a system stopped responding (port {dead[0]}): mcp_backend_down")
final = final_message()
os.makedirs("/tmp/agent_output/sessions", exist_ok=True)
Path("/tmp/agent_output/sessions/opencode.jsonl").write_text(
    json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": final}]}}) + "\n")
answer = Path(cfg["cwd"]) / "answer.md"
if final and not (answer.exists() and answer.stat().st_size):
    answer.write_text(final)
    shutil.chown(answer, "agent", "agent")
for up in cfg["uploads"]:   # the verifier's own files, hidden until now
    src, dst = T / "verifier" / up["source"], Path(up["target"])
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
rf, rd = Path(cfg["reward_file"]), Path(cfg["reward_detail_file"])
rf.parent.mkdir(parents=True, exist_ok=True)
rf.unlink(missing_ok=True)
rd.unlink(missing_ok=True)
p = subprocess.run(["sh", "-c", cfg["command"]], cwd="/work", timeout=cfg["timeout_sec"], capture_output=True, text=True)
print(redact(p.stdout + "\n" + p.stderr)[-6000:])
try:
    raw = json.loads(rf.read_text())
    reward = float(raw["reward"])
except (OSError, ValueError, KeyError, TypeError):
    detail = json.loads(rd.read_text()) if rd.exists() else {}
    fail(detail.get("reward_error") or "missing_or_invalid_reward_json")
if not (math.isfinite(reward) and 0.0 <= reward <= 1.0):
    fail("reward_out_of_range")
if rd.exists() and rd.resolve() != (V / "reward_detail.json").resolve():
    shutil.copy2(rd, V / "reward_detail.json")
(V / "reward.json").write_text(json.dumps({"reward": reward}))   # Harbor takes numbers only; details stay in reward_detail.json
scrub_outputs()
print(f"reward {reward}")
