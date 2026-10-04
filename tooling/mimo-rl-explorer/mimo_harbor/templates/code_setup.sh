# Code: mimoagent environments/datasets/opensource_code.py + base.py, as the explorer runs it (app/runner/domains.py Code).
CWD=@@CWD@@
git config --global --add safe.directory "$CWD"
cd "$CWD" || fail "no working directory $CWD"
if ! git rev-parse --git-dir >/dev/null 2>&1; then   # a bare source tree: give it a baseline commit
  git init -q && git add -A && git -c user.name=mimo -c user.email=mimo@localhost commit -q -m baseline --allow-empty
fi
BASE=$(git rev-parse HEAD 2>/dev/null)
[ ${#BASE} -eq 40 ] || fail "could not resolve the base commit in $CWD"
echo "$BASE" > "$M/base"
# Images are built with history truncated at the base. If one is not, the fix could be read out of git log,
# so .git is hidden while the agent works and put back for grading.
LATER=$(git rev-list --all --not "$BASE" 2>/dev/null | head -n 5 | grep -c . || true)
@@RESIDUE_SCRUB@@
git clean -fdx @@CLEAN_EXCLUDES@@ >/dev/null 2>&1
@@GLOBAL_CACHE_SCRUB@@
if [ "$LATER" -gt 0 ]; then
  mv "$CWD/.git" "$M/git-hidden"
  echo "history not truncated at ${BASE:0:12}: .git hidden while the agent works"
fi
write_blocklist
