#!/usr/bin/env bash
# Stop the env server and any rollout.
#
# MATCHES ON argv[0] BEING A PYTHON INTERPRETER, not on the command line containing a string. A
# substring match also hits every bash wrapper whose command line happens to quote the same command --
# including the agent harness's own shells. That has killed this session's shell once and deadlocked a
# restart loop for 11 minutes, and CLAUDE.md warns about exactly it. Ancestors are skipped too, so
# this can never kill whatever launched it.
cd "$(dirname "$0")/.."

ancestors=" "
p=$$
while [ "$p" -gt 1 ] 2>/dev/null; do
  ancestors="$ancestors$p "
  p=$(awk '{print $4}' /proc/$p/stat 2>/dev/null) || break
done

for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  case "$ancestors" in *" $pid "*) continue;; esac
  [ -r "/proc/$pid/cmdline" ] || continue
  argv0=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | head -1)
  case "$argv0" in *python*) ;; *) continue;; esac
  rest=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  case "$rest" in
    *"uvicorn data_agent_env.server.app"*|*"rollout.py --server"*)
      echo "  stopping $pid: $(echo "$rest" | cut -c1-90)"
      kill "$pid" 2>/dev/null ;;
  esac
done
