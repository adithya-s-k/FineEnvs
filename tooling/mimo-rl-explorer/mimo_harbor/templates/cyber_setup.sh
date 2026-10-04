# Cyber (ARVO): mimoagent environments/datasets/arvo.py with the image's own verify server (app/vendor/server_arvo.py),
# as the explorer runs it. The expected crash comes from the task description, which the agent also reads.
install -m 600 "$M/files/expected_func.json" /root/expected_func.json
install -m 700 "$M/files/server_arvo.py" /root/server.py
(setsid python3 /root/server.py </dev/null >/tmp/.srv.log 2>&1 &)
for i in $(seq 40); do curl -s -o /dev/null http://127.0.0.1:8666/ && break; sleep 0.5; done
curl -s -o /dev/null http://127.0.0.1:8666/ || fail "the PoC verification server did not start: $(tail -c 400 /tmp/.srv.log)"
id -u agent >/dev/null 2>&1 || fail "the image has no agent user"
echo agent > "$M/agent_user"
chmod 777 /logs/agent /logs/artifacts 2>/dev/null || true
write_blocklist
