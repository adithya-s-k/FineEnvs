// Connecting a coding agent to an environment over MCP (the explorer's bridge, app/mcp_bridge.py): the URL, and how
// to add it to Claude Code, Codex, Cursor, VS Code, Gemini CLI, Windsurf, any client, or Python. Used by Space pages
// (play the environment) and environment pages (browse its tasks).
import { esc, storage, toast } from "./util.js";
import { icon } from "./icons.js";

export const agentName = (spec) => spec.split("/")[1].toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "env";
export const copyBtn = (text, label = "Copy") => `<button class="btn sm ghost sp-copy" type="button" data-copy="${esc(text)}" title="Copy">${icon("copy", 13)}${label ? `<span>${label}</span>` : ""}</button>`;
export const snippet = (text) => `<div class="sp-snip"><pre class="code-block"><code>${esc(text)}</code></pre>${copyBtn(text, "")}</div>`;

const mcpPython = (u) => `# pip install mcp
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("${u}") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print([t.name for t in (await session.list_tools()).tools])

asyncio.run(main())`;

// [id, label, (name, url, extra) -> {text, also?, file?}]; extra.python(url) gives a Python snippet of the page's own
export const AGENTS = [
  ["claude", "Claude Code", (n, u) => ({ text: `claude mcp add --transport http ${n} ${u}` })],
  ["codex", "Codex", (n, u) => ({ text: `codex mcp add ${n} --url ${u}`, also: `# or in ~/.codex/config.toml\n[mcp_servers.${n.replace(/-/g, "_")}]\nurl = "${u}"` })],
  ["cursor", "Cursor", (n, u) => ({ text: JSON.stringify({ mcpServers: { [n]: { url: u } } }, null, 2), file: "In ~/.cursor/mcp.json, or .cursor/mcp.json in a project:" })],
  ["vscode", "VS Code", (n, u) => ({ text: `code --add-mcp '${JSON.stringify({ name: n, type: "http", url: u })}'`, also: `// or in .vscode/mcp.json\n${JSON.stringify({ servers: { [n]: { type: "http", url: u } } }, null, 2)}` })],
  ["gemini", "Gemini CLI", (n, u) => ({ text: `gemini mcp add --transport http ${n} ${u}`, also: `// or in ~/.gemini/settings.json\n${JSON.stringify({ mcpServers: { [n]: { httpUrl: u } } }, null, 2)}` })],
  ["windsurf", "Windsurf", (n, u) => ({ text: JSON.stringify({ mcpServers: { [n]: { serverUrl: u } } }, null, 2), file: "In ~/.codeium/windsurf/mcp_config.json:" })],
  ["other", "Other clients", (n, u) => ({ text: JSON.stringify({ mcpServers: { [n]: { type: "http", url: u } } }, null, 2), file: "Any client that speaks Streamable HTTP. No sign-in needed." })],
  ["python", "Python", (n, u, extra) => ({ text: extra?.python ? extra.python(u) : mcpPython(u) })],
];

export const chosenAgent = () => (AGENTS.some((a) => a[0] === storage.get("sp-agent")) ? storage.get("sp-agent") : "claude");

export function agentHtml(id, name, url, extra) {
  const a = AGENTS.find((x) => x[0] === id) || AGENTS[0];
  const c = a[2](name, url, extra);
  return `${c.file ? `<p class="fine sp-file">${esc(c.file)}</p>` : ""}${snippet(c.text)}${c.also ? snippet(c.also) : ""}`;
}

// the whole panel: the URL, what it offers (`note`, HTML), the agent tabs and the snippet; wire it with wireConnect
export function connectPanel({ url, name, note = "" }) {
  const chosen = chosenAgent();
  return `<div class="cn" data-cn-url="${esc(url)}" data-cn-name="${esc(name)}"><div class="sp-url"><div class="sp-url-row"><span class="sp-url-k">MCP</span><code>${esc(url)}</code>${copyBtn(url)}</div>
    ${note ? `<p class="fine">${note}</p>` : ""}</div>
    <div class="seg sp-agents" role="tablist" aria-label="Agent">${AGENTS.map(([id, label]) => `<button type="button" role="tab" data-agent="${id}" aria-pressed="${id === chosen}">${esc(label)}</button>`).join("")}</div>
    <div class="cn-agent">${agentHtml(chosen, name, url)}</div></div>`;
}

export function wireConnect(root) {
  root.addEventListener("click", (e) => {
    const c = e.target.closest(".cn [data-copy]");
    if (c) { navigator.clipboard?.writeText(c.dataset.copy).then(() => toast("Copied"), () => toast("Couldn't copy")); return; }
    const ag = e.target.closest(".cn [data-agent]");
    if (!ag) return;
    const box = ag.closest(".cn");
    storage.set("sp-agent", ag.dataset.agent);
    box.querySelectorAll("[data-agent]").forEach((x) => x.setAttribute("aria-pressed", String(x === ag)));
    box.querySelector(".cn-agent").innerHTML = agentHtml(ag.dataset.agent, box.dataset.cnName, box.dataset.cnUrl);
  });
}
