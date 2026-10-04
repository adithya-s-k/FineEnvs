#!/bin/bash
# Terminal-Bench grading: the task's own test script (task_test.sh here) writes /logs/verifier/reward.txt.
# "-1" there is its crash sentinel: a testbed failure, so no reward is written (not scored).
CWD=@@CWD@@
mkdir -p /logs/verifier
# the task's anti-hack guard rejects files planted in the workspace: drop what agent harnesses leave there
for d in .opencode .claude .codex .gemini .goose .aider.tags.cache.v4; do
  grep -qx "$d" /var/lib/mimo/before 2>/dev/null || rm -rf "$CWD/$d"
done
cd "$CWD" && bash /tests/task_test.sh
R=$(cat /logs/verifier/reward.txt 2>/dev/null | tr -d '[:space:]')
if [ -z "$R" ] || [ "$R" = "-1" ]; then
  rm -f /logs/verifier/reward.txt
  echo "the tests crashed before grading (reward '${R}'): not scored" >&2
  exit 1
fi
