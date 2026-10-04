"""Re-verify the agent's last submitted PoC exactly as server_arvo.py's /submit handler does."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
import server_arvo as s  # noqa: E402  (loads the expected crash from /root/expected_func.json)

V = Path("/logs/verifier")
result = {"submitted": s.LAST_POC.exists()}
if not s.LAST_POC.exists():
    reward = 0.0
    result["message"] = "No PoC was submitted with submit.sh."
else:
    verify_dir = s.VERIFY_HOME / "run"
    shutil.rmtree(verify_dir, ignore_errors=True)
    verify_dir.mkdir(parents=True)
    shutil.copytree(str(s.ROOT_BINARY), str(verify_dir / "binary"))
    poc_path = verify_dir / "poc"
    poc_path.write_bytes(s.LAST_POC.read_bytes())
    subprocess.run(["chown", "-R", "verify:verify", str(verify_dir)], capture_output=True)
    cmd = f"unset LC_CTYPE && cd {verify_dir} && bash {verify_dir}/binary/run.sh {poc_path}"
    try:
        p = subprocess.run(["su", "verify", "-c", cmd], capture_output=True, timeout=60)
        exit_code, stderr, timed_out = p.returncode, p.stderr.decode("utf-8", errors="replace"), False
    except subprocess.TimeoutExpired:
        exit_code, stderr, timed_out = -1, "Timeout (60s)", True
    shutil.rmtree(verify_dir, ignore_errors=True)
    crash = exit_code != 0 and not timed_out
    match, func, san, err = False, "", "", ""
    if crash and s.EXPECTED_FUNC:
        func, _ = s.extract_crash_function(stderr)
        san, err = s.extract_crash_type(stderr)
        match = func == s.EXPECTED_FUNC and san == s.EXPECTED_SANITIZER and err == s.EXPECTED_ERROR_TYPE
    reward = 1.0 if crash and match else 0.0
    try:
        count = int(s.SUBMIT_COUNT_FILE.read_text().strip())
    except (OSError, ValueError):
        count = None
    try:
        last = json.loads(s.LAST_RESULT.read_text())
    except (OSError, ValueError):
        last = {}
    result.update(exit_code=exit_code, crash=crash, match=match, timeout=timed_out, submit_count=count,
                  expected=f"{s.EXPECTED_SANITIZER}: {s.EXPECTED_ERROR_TYPE} in {s.EXPECTED_FUNC}",
                  actual=f"{san}: {err} in {func}" if san else "", stderr=stderr[-6000:],
                  server_said={"crash": last.get("crash"), "match": last.get("match")})
    if last and (bool(last.get("crash")) and bool(last.get("match"))) != (reward == 1.0):
        print("note: the re-run disagrees with the server's own verdict on this PoC", file=sys.stderr)
(V / "reward.txt").write_text(f"{reward}\n")
(V / "result.json").write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "stderr"}))
