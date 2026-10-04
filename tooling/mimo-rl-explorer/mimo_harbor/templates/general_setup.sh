# General (simulated workplace): verl general_agent + mimoagent's sidecar, as the explorer runs it (domains.py General).
# The workplace files come from the benchmark at a pinned revision, each checked against its hash; the systems then
# start on 127.0.0.1 and the agent works through their MCP servers.
python3 "$M/files/fetch.py" "$M/files/fetch.json" || fail "could not fetch the workplace files"
# the sidecar needs a venv with mcp<2 (the image's own python has mcp 2.x, which renamed FastMCP)
if [ ! -x /opt/openai-agents-venv/bin/python ]; then
  python3 -m venv /opt/openai-agents-venv && /opt/openai-agents-venv/bin/pip install -q @@MCP_PIN@@ \
    || fail "could not prepare the MCP sidecar"
fi
timeout @@SETUP_TIMEOUT@@ sh -c @@SETUP_CMD@@ > "$M/systems.log" 2>&1 || { tail -c 1500 "$M/systems.log" >&2; fail "the systems did not start"; }
for p in @@PORTS@@; do
  for i in $(seq 60); do (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null && break; sleep 1; done
  (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null || fail "system on port $p did not come up"
done
# Upstream these systems run in a separate sidecar pod. Here they share the container, so the agent is an
# unprivileged user and the systems' data and code are root-only: MCP is the way in.
id -u agent >/dev/null 2>&1 || useradd -u 500 -m -s /bin/bash agent
chown -R agent:agent @@CWD@@
chmod 700 /work/system /work/tools /installed-agent /work/_setup 2>/dev/null
if runuser -u agent -- ls /work/system >/dev/null 2>&1; then fail "could not isolate the systems' data from the agent"; fi
echo agent > "$M/agent_user"
chmod 777 /logs/agent /logs/artifacts 2>/dev/null || true
write_blocklist
