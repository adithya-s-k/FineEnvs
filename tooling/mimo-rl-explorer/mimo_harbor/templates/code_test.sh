#!/bin/bash
# Code grading, mimoagent opensource_code.py: reset the files the hidden test patch touches, apply it, run the
# task's test command; reward 1 when it exits 0. A testbed failure (reset or apply) writes no reward: not scored.
M=/var/lib/mimo
CWD=@@CWD@@
V=/logs/verifier
mkdir -p "$V"
BASE=$(cat "$M/base" 2>/dev/null) || { echo "setup never ran (no $M/base)" >&2; exit 1; }
cd "$CWD" || exit 1
if [ -d "$M/git-hidden" ]; then rm -rf "$CWD/.git"; mv "$M/git-hidden" "$CWD/.git"; fi
# the agent's change against the base, so its own commits count too
git add -A >/dev/null 2>&1 && git -c core.fileMode=false diff --cached "$BASE" > "$V/agent.diff"
git reset -q
if ! (
@@RESET_CMD@@
); then echo "could not reset the test files (testbed problem, not scored)" >&2; exit 1; fi
if ! git apply --verbose /tests/test.patch > "$V/apply.log" 2>&1; then
  cat "$V/apply.log" >&2
  echo "the hidden tests could not be applied (testbed problem, not scored)" >&2
  exit 1
fi
timeout @@TEST_TIMEOUT@@ sh -c "$(cat /tests/test_command.sh)" > "$V/test_output.log" 2>&1
RC=$?
tail -c 12000 "$V/test_output.log"
if [ $RC -eq 0 ]; then echo 1 > "$V/reward.txt"; else echo 0 > "$V/reward.txt"; fi
echo "test command exited $RC"
