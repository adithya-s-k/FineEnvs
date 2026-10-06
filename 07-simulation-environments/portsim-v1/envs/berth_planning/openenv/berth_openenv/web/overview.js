// Overview (#/): what the environment is, how to use it, and the eval, on one page. The Space opens here.
import { getRun } from "./api.js";
import { escapeHtml, fmtNum } from "./model.js";

const RUN = "dock-eval50";
const SHOWCASE = { model: "openai:gpt-6.1-sol", task: "dock-24B-w35x2-storm-1" };
const NAMES = {
  "openai:gpt-6.1-sol": "GPT-6.1 Sol",
  "anthropic:claude-sonnet-5-5": "Claude Sonnet 5.5",
  "hf:zai-org/GLM-5.3-Flash:baseten": "GLM-5.3-Flash",
  "hf:Qwen/Qwen3.8-2.4T-A95B:together": "Qwen3.8-2.4T",
  "hf:zai-org/GLM-5.3:together": "GLM-5.3",
  "hf:Qwen/Qwen3.8-27B:cerebras|ovhcloud": "Qwen3.8-27B",
};
const LINKS = [
  ["Article", "https://huggingface.co/spaces/FineEnvs/simulation-rl-environments", "Simulation RL Environments, part 1"],
  ["Dataset", "https://huggingface.co/datasets/FineEnvs/PortSimEnv", "tasks, the source port calls, eval rollouts"],
  ["Bucket", "https://huggingface.co/buckets/FineEnvs/PortSimEnv", "3D twin data, raw eval rollouts"],
  ["Code", "https://github.com/adithya-s-k/FineEnvs", "environment, grader, viewer"],
  ["Discussion", "https://github.com/adithya-s-k/FineEnvs/discussions/36", "ideas for v2, v3, post-training, data"],
];
const TIERS = ["standard", "busy", "storm", "extreme"];
const enc = encodeURIComponent;
const runHref = (model, task) => `#/run/${RUN}/${enc(model)}/${enc(task)}`;

function board(episodes) {
  const by = new Map();
  for (const e of episodes) {
    if (!by.has(e.model)) by.set(e.model, []);
    by.get(e.model).push(e);
  }
  const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  return [...by.entries()].map(([model, eps]) => ({
    model,
    name: NAMES[model] || model,
    n: eps.length,
    mean: mean(eps.map((e) => e.reward || 0)),
    tiers: Object.fromEntries(TIERS.map((t) => [t, mean(eps.filter((e) => e.difficulty === t).map((e) => e.reward || 0))])),
    submitted: eps.filter((e) => e.submitted).length,
    valid: eps.filter((e) => e.feasible).length,
    optimal: eps.filter((e) => (e.reward || 0) >= 0.999).length,
  })).sort((a, b) => b.mean - a.mean);
}

export async function overviewPage({ app, setCrumbs, isCurrent }) {
  setCrumbs([]);
  const url = location.origin;
  const showcase = runHref(SHOWCASE.model, SHOWCASE.task);
  app.innerHTML = `
  <div class="page ov-page">
    <div class="ov-grid">
      <div class="ov-main">
        <section class="ov-intro">
          <h1>Re-plan a week of container-ship dockings at the Port of Barcelona</h1>
          <p class="muted">Real 2024 port calls, the terminal's real cranes and the port's rules, and a week that has just gone wrong.
          The agent decides when, where and with how many cranes every ship docks. One graded submit per episode, scored
          deterministically against the plan a CP-SAT solver proved optimal.</p>
          <div class="ov-acts">
            <a class="btn primary" href="#/play">Play an episode</a>
            <a class="btn" href="${showcase}">Watch a rollout in 3D</a>
            <a class="btn" href="#/tasks">Tasks and rollouts</a>
          </div>
        </section>
        <a class="ov-shot" href="${showcase}" title="GPT-6.1 Sol's plan for storm week 35 at APM Terminals, in 3D">
          <img src="img/overview.webp" alt="The 3D twin of APM Terminals Barcelona: a container ship berthing with tugs under the quay cranes, the container yard and the tank farm behind." loading="eager">
        </a>
        <section>
          <div class="sec-head"><h2>Eval</h2><span class="muted small">${RUN} · 50 held-out weeks · one rollout per model · 12 turns, 32k output tokens per turn</span></div>
          <table class="tbl click" id="ov-board">
            <thead><tr><th>Model</th><th class="num">Mean</th>${TIERS.map((t) => `<th class="num opt">${t[0].toUpperCase() + t.slice(1)}</th>`).join("")}<th class="num" title="Episodes that ended with submit_plan">Submitted</th><th class="num" title="Plans that break no rule">Valid</th><th class="num" title="Plans that match the proven optimum (reward 1.0)">Optimal</th></tr></thead>
            <tbody>${Array.from({ length: 6 }, () => `<tr class="skel">${"<td><i></i></td>".repeat(9)}</tr>`).join("")}</tbody>
          </table>
          <p class="muted small ov-note">Click a model to watch its rollout of storm week 35 in 3D. Every rollout is under <a href="#/tasks">Tasks</a>.</p>
        </section>
        <section>
          <div class="sec-head"><h2>Connect an agent</h2><span class="muted small">OpenEnv · MCP tools over a WebSocket session</span></div>
          <pre class="ov-code"><code>from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

env = MCPToolClient("${escapeHtml(url)}").sync()
obs = env.reset(task_id="dock-24B-w07x1-busy-0")   # or reset(split="train", index=0)
rules = obs.observation.metadata["instructions"]     # the system prompt

step = env.step(CallToolAction(tool_name="get_situation", arguments={}))
plan = [{"ship": 0, "berth_hour": 0, "section": 9, "cranes": 3}, ...]
step = env.step(CallToolAction(tool_name="check_plan", arguments={"plan": plan}))
step = env.step(CallToolAction(tool_name="submit_plan", arguments={"plan": plan}))
print(step.reward)   # 0..1, graded once</code></pre>
        </section>
      </div>
      <aside class="ov-side">
        <section>
          <h2>Environment</h2>
          <dl class="kv" id="ov-env">
            <dt>Tasks</dt><dd id="ov-tasks">train 1,050 · eval 50</dd>
            <dt>Quays</dt><dd>BEST 36A (13 cranes) · APM 24B (9 cranes)</dd>
            <dt>Episode</dt><dd>10 checks · 24 tool calls · 1 submit</dd>
            <dt>Grader</dt><dd>deterministic, no LLM judge</dd>
            <dt>Data</dt><dd>Port of Barcelona open data, 2024 (CC BY-SA 4.0)</dd>
          </dl>
        </section>
        <section>
          <h2>Tools</h2>
          <dl class="kv ov-tools">
            <dt><code>get_situation()</code></dt><dd>the quay, notices, closures, ships to berth</dd>
            <dt><code>check_plan(plan)</code></dt><dd>rule breaks and cost per ship; 10 per episode</dd>
            <dt><code>submit_plan(plan)</code></dt><dd>ends the episode with one grade</dd>
          </dl>
        </section>
        <section>
          <h2>Reward</h2>
          <dl class="kv">
            <dt>No plan</dt><dd>0</dd>
            <dt title="0.2 × the share of ships placed without a violation">A rule broken</dt><dd>≤ 0.2</dd>
            <dt title="gap = (cost − optimum) / (optimum − unavoidable + 100)">Valid plan</dt><dd>0.2 + 0.8·e<sup>−gap/0.5</sup></dd>
            <dt>The optimum</dt><dd>1.0</dd>
          </dl>
        </section>
        <section>
          <h2>Links</h2>
          <dl class="kv">
            ${LINKS.map(([k, href, what]) => `<dt><a class="ext" href="${href}" target="_blank" rel="noopener">${k} ↗</a></dt><dd class="muted">${what}</dd>`).join("")}
            <dt><a class="ext" href="/web" target="_blank" rel="noopener">OpenEnv UI ↗</a></dt><dd class="muted">the standard OpenEnv web interface and playground</dd>
            <dt><a class="ext" href="/docs" target="_blank" rel="noopener">API ↗</a></dt><dd class="muted">reset, step, state, schema, MCP, Task API</dd>
          </dl>
        </section>
      </aside>
    </div>
  </div>`;

  fetch("/healthz").then((r) => (r.ok ? r.json() : null)).then((h) => {
    if (!isCurrent() || !h || !h.tasks) return;
    const el = app.querySelector("#ov-tasks");
    if (el) el.textContent = Object.entries(h.tasks).map(([s, n]) => `${s} ${n.toLocaleString()}`).join(" · ");
  }).catch(() => {});

  const tbody = app.querySelector("#ov-board tbody");
  let rows;
  try {
    rows = board((await getRun(RUN)).episodes || []);
  } catch (err) {
    if (!isCurrent()) return;
    tbody.innerHTML = `<tr><td colspan="9" class="muted">No eval rollouts on this server. <span class="small">${escapeHtml(err.message || String(err))}</span></td></tr>`;
    return;
  }
  if (!isCurrent()) return;
  tbody.innerHTML = rows.map((r) => `<tr data-row="${escapeHtml(r.model)}">
      <td>${escapeHtml(r.name)}</td><td class="num"><b>${fmtNum(r.mean, 3)}</b></td>
      ${TIERS.map((t) => `<td class="num opt">${r.tiers[t] == null ? "–" : fmtNum(r.tiers[t], 2)}</td>`).join("")}
      <td class="num">${r.submitted}/${r.n}</td><td class="num">${r.valid}/${r.n}</td><td class="num">${r.optimal}/${r.n}</td></tr>`).join("");
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-row]");
    if (tr) location.hash = runHref(tr.dataset.row, SHOWCASE.task);
  });
}
