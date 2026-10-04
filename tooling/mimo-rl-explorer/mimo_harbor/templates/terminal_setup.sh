# Terminal-Bench rows (in the General domain): the Harbor convention itself. Scrub test leftovers from the image and
# remember what was in the workspace, so harness residue can be told apart from the agent's work at grading time.
rm -rf /tests
find /logs/verifier -mindepth 1 -delete 2>/dev/null || true
ls -A @@CWD@@ 2>/dev/null > "$M/before" || true
