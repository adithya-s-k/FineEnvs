// Overview (#/): what the environment is, how to use it, and the eval, on one page. Both Spaces open here: the
// environment's server (env mode) and the eval Space (explorer mode: the eval first, playing links out).
import { getRun, getRuns } from "./api.js";
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
  ["Code", "https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments/portsim-v1", "07-simulation-environments/portsim-v1 on GitHub"],
  ["Discussion", "https://github.com/adithya-s-k/FineEnvs/discussions/36", "ideas for v2, v3, post-training, data"],
];
const ENV_SPACE = "https://huggingface.co/spaces/FineEnvs/PortSimEnv";
const ENV_URL = "https://fineenvs-portsimenv.hf.space";
const EVAL_VIEWER = "https://fineenvs-portsimenv-eval.hf.space/viewer/"; // the eval Space, for servers without rollouts
const TIERS = ["standard", "busy", "storm", "extreme"];
const TIER_TIPS = {  // core/berth_core/dock.py TIERS
  standard: "1 week · 1 closure · 1–2 late ships",
  busy: "1–2 weeks · 1–2 closures · 2–3 late ships · a crane outage · priority cargo",
  storm: "2 weeks · a gale · an emergency · 2–4 late ships · a crane outage",
  extreme: "2–3 weeks · 2–3 closures · gales · a diverted quay · bunched and late ships · emergencies",
};
const enc = encodeURIComponent;
const runHref = (model, task) => `#/run/${RUN}/${enc(model)}/${enc(task)}`;
export const nameOf = (model) => NAMES[model] || model;
const cap = (s) => s[0].toUpperCase() + s.slice(1);

function board(episodes) {
  const by = new Map();
  for (const e of episodes) {
    if (!by.has(e.model)) by.set(e.model, []);
    by.get(e.model).push(e);
  }
  const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  return [...by.entries()].map(([model, eps]) => ({
    model,
    name: nameOf(model),
    eps,
    n: eps.length,
    mean: mean(eps.map((e) => e.reward || 0)),
    tiers: Object.fromEntries(TIERS.map((t) => [t, mean(eps.filter((e) => e.difficulty === t).map((e) => e.reward || 0))])),
    submitted: eps.filter((e) => e.submitted).length,
    valid: eps.filter((e) => e.feasible).length,
    optimal: eps.filter((e) => (e.reward || 0) >= 0.999).length,
  })).sort((a, b) => b.mean - a.mean);
}

// Outcome of one rollout, for the bars and the grid.
const OUTCOMES = [
  { k: "optimal", label: "optimal plan (1.0)" },
  { k: "valid", label: "valid, not optimal" },
  { k: "broken", label: "broke a rule (≤ 0.2)" },
  { k: "none", label: "no plan submitted (0)" },
];
const outcomeOf = (e) => (!e.submitted ? "none" : !e.feasible ? "broken" : (e.reward || 0) >= 0.999 ? "optimal" : "valid");

function outcomeBar(r) {
  const n = r.n || 1;
  const c = { optimal: r.optimal, valid: r.valid - r.optimal, broken: r.submitted - r.valid, none: r.n - r.submitted };
  return `<span class="ov-obar" title="${OUTCOMES.map((o) => `${c[o.k]} ${o.label}`).join(" · ")}">${OUTCOMES.filter((o) => c[o.k] > 0)
    .map((o) => `<i class="oc-${o.k}" style="width:${((100 * c[o.k]) / n).toFixed(2)}%"></i>`).join("")}</span>`;
}

function resultHtml(e) {
  if (!e.submitted) return `<span class="muted" title="${escapeHtml(e.end_reason || "")}">not submitted</span>`;
  if (!e.feasible) return '<i class="dot bad"></i>rule broken';
  return (e.reward || 0) >= 0.999 ? '<i class="dot ok"></i>optimal' : '<i class="dot ok"></i>valid';
}

// One model's rollouts: every eval week, each row opening the 3D replay.
const EP_COLS = [
  { k: "task_id", label: "Task", get: (r) => r.task_id, html: (r) => `<a href="${runHref(r.model, r.task_id)}">${escapeHtml(r.task_id)}</a>` },
  { k: "quay", label: "Quay", get: (r) => r.quay, opt: true },
  { k: "difficulty", label: "Tier", get: (r) => r.difficulty, sort: (r) => TIERS.indexOf(r.difficulty), opt: true },
  { k: "ships", label: "Ships", get: (r) => r.ships, num: true, opt: true },
  { k: "result", label: "Result", get: (r) => r.end_reason, html: resultHtml, sort: (r) => (r.submitted ? 1 : 0) + (r.feasible ? 1 : 0) + ((r.reward || 0) >= 0.999 ? 1 : 0) },
  { k: "cost", label: "Cost", get: (r) => (r.feasible ? r.cost : "–"), sort: (r) => (r.feasible ? r.cost : null), num: true, title: "Cost of the submitted plan (valid plans only)" },
  { k: "optimal_cost", label: "Optimum", get: (r) => r.optimal_cost, num: true, opt: true, title: "Proven optimal cost (CP-SAT)" },
  { k: "naive_cost", label: "Naive", get: (r) => r.naive_cost, num: true, opt: true, title: "Cost of the naive re-plan: keep sections, push conflicting ships to the next free hour" },
  { k: "checks", label: "Checks", get: (r) => r.checks, num: true, opt: true, title: "check_plan calls (10 allowed)" },
  { k: "reward", label: "Reward", get: (r) => fmtNum(r.reward, 3), sort: (r) => r.reward, num: true },
];

export async function overviewPage({ app, setCrumbs, isCurrent, params, config = {}, sortableTable }) {
  setCrumbs([]);
  const explorer = !!config.explorer;
  const playUrl = config.playUrl || ENV_SPACE;
  const envUrl = explorer ? config.envUrl || ENV_URL : location.origin;
  const showcase = runHref(SHOWCASE.model, SHOWCASE.task);
  const shot = `<a class="ov-shot" href="${showcase}" title="GPT-6.1 Sol's plan for storm week 35 at APM Terminals, in 3D">
          <img src="img/overview.webp" alt="The 3D twin of APM Terminals Barcelona: a container ship berthing with tugs under the quay cranes, the container yard and the tank farm behind." loading="eager">
        </a>`;
  const intro = explorer
    ? `<section class="ov-intro">
          <h1>PortSimEnv v1 eval: re-planning a broken week at the Port of Barcelona</h1>
          <p class="muted">50 held-out tasks on real 2024 container calls at two quays, one to three weeks each, every one with
          something gone wrong: closures, late and bunched ships, crane outages, gales, diverted traffic, emergencies. One
          rollout per model and task, one graded submit, scored against the plan a CP-SAT solver proved optimal. Every
          rollout replays here in 3D with its dock chart, grade and transcript. To play a task yourself or connect an
          agent, use the environment.</p>
          <div class="ov-acts">
            <a class="btn primary" href="${showcase}">Watch a rollout in 3D</a>
            <a class="btn" href="#/tasks">All tasks and rollouts</a>
            <a class="btn" href="${escapeHtml(playUrl)}" target="_blank" rel="noopener">Play an episode ↗</a>
          </div>
        </section>`
    : `<section class="ov-intro">
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
        ${shot}`;
  const evalSec = `<section>
          <div class="sec-head"><h2>Eval</h2><span class="muted small">${RUN} · 50 held-out tasks · one rollout per model and task · 12 turns, 32k output tokens per turn</span></div>
          <table class="tbl click" id="ov-board">
            <thead><tr><th>Model</th><th class="ov-oth" title="The 50 eval weeks by outcome">Outcomes</th><th class="num">Mean</th>${TIERS.map((t) => `<th class="num opt" title="${escapeHtml(TIER_TIPS[t])}">${cap(t)}</th>`).join("")}<th class="num" title="Episodes that ended with submit_plan">Submitted</th><th class="num" title="Plans that break no rule">Valid</th><th class="num" title="Plans that match the proven optimum (reward 1.0)">Optimal</th></tr></thead>
            <tbody>${Array.from({ length: 6 }, () => `<tr class="skel">${"<td><i></i></td>".repeat(10)}</tr>`).join("")}</tbody>
          </table>
          <div class="ov-legend">${OUTCOMES.map((o) => `<span><i class="oc-${o.k}"></i>${o.label}</span>`).join("")}</div>
          <p class="muted small ov-note" id="ov-board-note">Mean reward per model and tier. Click a model to list its rollouts; each one replays in 3D.</p>
        </section>
        <section class="ov-grid-sec" hidden>
          <div class="sec-head"><h2>Every rollout</h2><span class="muted small">one square per model and eval week, coloured by outcome (fainter = lower reward) · click a square to replay it in 3D</span></div>
          <div class="ov-mx" id="ov-mx"></div>
        </section>
        <section class="ov-eps" hidden>
          <div class="sec-head"><h2 id="ov-eps-h">Rollouts</h2><span class="muted small" id="ov-eps-note"></span></div>
          <table class="tbl click" id="ov-eps"><thead></thead><tbody></tbody></table>
        </section>`;
  const code = `<section>
          <div class="sec-head"><h2>${explorer ? "Evaluate your own model" : "Connect an agent"}</h2><span class="muted small">OpenEnv · MCP tools over a WebSocket session${explorer ? ` · <a href="${escapeHtml(playUrl)}" target="_blank" rel="noopener">the environment Space</a>` : ""}</span></div>
          <pre class="ov-code"><code>from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

env = MCPToolClient("${escapeHtml(envUrl)}").sync()
obs = env.reset(task_id="dock-24B-w07x1-busy-0")   # or reset(split="${explorer ? "eval" : "train"}", index=0)
rules = obs.observation.metadata["instructions"]     # the system prompt

step = env.step(CallToolAction(tool_name="get_situation", arguments={}))
plan = [{"ship": 0, "berth_hour": 0, "section": 9, "cranes": 3}]   # one entry per ship
step = env.step(CallToolAction(tool_name="check_plan", arguments={"plan": plan}))
step = env.step(CallToolAction(tool_name="submit_plan", arguments={"plan": plan}))
print(step.reward)   # 0..1, graded once</code></pre>
        </section>`;
  const links = explorer
    ? `<dt><a class="ext" href="${escapeHtml(playUrl)}" target="_blank" rel="noopener">Environment ↗</a></dt><dd class="muted">PortSimEnv: play an episode, connect an agent</dd>
            ${LINKS.map(([k, href, what]) => `<dt><a class="ext" href="${href}" target="_blank" rel="noopener">${k} ↗</a></dt><dd class="muted">${what}</dd>`).join("")}`
    : `${LINKS.map(([k, href, what]) => `<dt><a class="ext" href="${href}" target="_blank" rel="noopener">${k} ↗</a></dt><dd class="muted">${what}</dd>`).join("")}
            <dt><a class="ext" href="/web" target="_blank" rel="noopener">OpenEnv UI ↗</a></dt><dd class="muted">the standard OpenEnv web interface and playground</dd>
            <dt><a class="ext" href="/docs" target="_blank" rel="noopener">API ↗</a></dt><dd class="muted">reset, step, state, schema, MCP, Task API</dd>`;
  app.innerHTML = `
  <div class="page ov-page${explorer ? " ov-explorer" : ""}">
    <div class="ov-grid">
      <div class="ov-main">
        ${intro}
        ${evalSec}
        ${code}
      </div>
      <aside class="ov-side">
        ${explorer ? shot : ""}
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
            ${links}
          </dl>
        </section>
      </aside>
    </div>
  </div>`;

  fetch(new URL("/healthz", location.origin).toString()).then((r) => (r.ok ? r.json() : null)).then((h) => {
    if (!isCurrent() || !h || !h.tasks) return;
    const el = app.querySelector("#ov-tasks");
    if (el) el.textContent = Object.entries(h.tasks).map(([s, n]) => `${s} ${n.toLocaleString()}`).join(" · ");
  }).catch(() => {});

  const tbody = app.querySelector("#ov-board tbody");
  const noRollouts = () => {
    tbody.innerHTML = `<tr><td colspan="10" class="muted">No eval rollouts on this server.${explorer ? "" : ` They are in the <a href="${EVAL_VIEWER}" target="_blank" rel="noopener">eval Space ↗</a>.`}</td></tr>`;
    app.querySelector("#ov-board-note").hidden = true;
    if (explorer) return;
    for (const a of app.querySelectorAll('a[href^="#/run/"]')) { // the showcase rollout: watch it in the eval Space
      a.href = EVAL_VIEWER + a.getAttribute("href");
      a.target = "_blank";
      a.rel = "noopener";
    }
  };
  let rows;
  try {
    if (!(await getRuns()).some((r) => r.run === RUN)) throw new Error("no run");
    rows = board((await getRun(RUN)).episodes || []);
  } catch {
    if (isCurrent()) noRollouts();
    return;
  }
  if (!isCurrent()) return;
  if (!rows.length) return noRollouts();
  tbody.innerHTML = rows.map((r) => `<tr data-row="${escapeHtml(r.model)}" title="List ${escapeHtml(r.name)}'s rollouts">
      <td>${escapeHtml(r.name)}</td><td class="ov-otd">${outcomeBar(r)}</td><td class="num"><b>${fmtNum(r.mean, 3)}</b></td>
      ${TIERS.map((t) => `<td class="num opt">${r.tiers[t] == null ? "–" : fmtNum(r.tiers[t], 2)}</td>`).join("")}
      <td class="num">${r.submitted}/${r.n}</td><td class="num">${r.valid}/${r.n}</td><td class="num">${r.optimal}/${r.n}</td></tr>`).join("");

  const perTier = new Map();
  for (const e of rows[0].eps) perTier.set(e.difficulty, (perTier.get(e.difficulty) || 0) + 1);
  app.querySelector("#ov-board-note").textContent = `Mean reward per model and tier (${TIERS.filter((t) => perTier.has(t)).map((t) => `${t} ${perTier.get(t)}`).join(", ")} tasks). Click a model to list its rollouts; each one replays in 3D.`;

  // Every rollout at a glance: models down, eval weeks across (grouped by tier), each square a link to its 3D replay.
  {
    const tasks = [...new Map(rows.flatMap((r) => r.eps).map((e) => [e.task_id, e])).values()]
      .sort((a, b) => TIERS.indexOf(a.difficulty) - TIERS.indexOf(b.difficulty) || a.task_id.localeCompare(b.task_id));
    const byKey = new Map(rows.flatMap((r) => r.eps.map((e) => [`${r.model}|${e.task_id}`, e])));
    const groups = TIERS.map((t) => ({ t, tasks: tasks.filter((e) => e.difficulty === t) })).filter((g) => g.tasks.length);
    const cell = (r, task) => {
      const e = byKey.get(`${r.model}|${task.task_id}`);
      if (!e) return '<span class="mx-c mx-missing"></span>';
      const oc = outcomeOf(e);
      const op = oc === "valid" ? (0.35 + 0.65 * Math.max(0, ((e.reward || 0) - 0.2) / 0.8)).toFixed(2) : "1";
      return `<a class="mx-c oc-${oc}" style="opacity:${op}" href="${runHref(r.model, task.task_id)}" title="${escapeHtml(r.name)} · ${escapeHtml(task.task_id)} · ${task.ships} ships · ${oc === "none" ? "no plan" : `reward ${fmtNum(e.reward, 2)}`}"></a>`;
    };
    app.querySelector("#ov-mx").innerHTML = `<div class="mx-row mx-head"><span class="mx-name"></span>${groups.map((g) => `<span class="mx-g" style="--n:${g.tasks.length}" title="${escapeHtml(TIER_TIPS[g.t])}">${cap(g.t)} · ${g.tasks.length}</span>`).join("")}</div>` +
      rows.map((r) => `<div class="mx-row"><span class="mx-name">${escapeHtml(r.name)}</span>${groups.map((g) => `<span class="mx-g" style="--n:${g.tasks.length}">${g.tasks.map((t) => cell(r, t)).join("")}</span>`).join("")}<span class="mx-mean">${fmtNum(r.mean, 2)}</span></div>`).join("");
    app.querySelector(".ov-grid-sec").hidden = false;
  }

  // The selected model's rollouts, under the board (#/?model=<id> keeps the choice in shared links).
  const epsSec = app.querySelector(".ov-eps");
  let epsTable = null;
  function select(model, scroll) {
    const r = rows.find((x) => x.model === model) || rows[0];
    for (const tr of tbody.querySelectorAll("tr[data-row]")) tr.classList.toggle("sel", tr.dataset.row === r.model);
    epsSec.hidden = false;
    app.querySelector("#ov-eps-h").textContent = `Rollouts · ${r.name}`;
    app.querySelector("#ov-eps-note").textContent = `${r.model} · ${r.n} tasks · click one to replay it in 3D`;
    if (!sortableTable) {
      app.querySelector("#ov-eps tbody").innerHTML = r.eps.map((e) => `<tr><td><a href="${runHref(e.model, e.task_id)}">${escapeHtml(e.task_id)}</a></td><td class="num">${fmtNum(e.reward, 3)}</td></tr>`).join("");
    } else if (!epsTable) {
      epsTable = sortableTable(app.querySelector("#ov-eps"), EP_COLS, r.eps, {
        initial: { key: "task_id", dir: 1 },
        rowAttrs: (e) => `data-row="${escapeHtml(e.task_id)}" data-model="${escapeHtml(e.model)}"`,
        onRow: (taskId) => (location.hash = runHref(cur, taskId)),
      });
    } else epsTable.setRows(r.eps);
    cur = r.model;
    if (scroll && epsSec.getBoundingClientRect().top > window.innerHeight - 120) epsSec.scrollIntoView({ block: "start", behavior: "smooth" });
  }
  let cur = null;
  select(params && params.get("model"), false);
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-row]");
    if (!tr) return;
    select(tr.dataset.row, true);
    history.replaceState(null, "", `${location.pathname}${location.search}#/?model=${enc(cur)}`);
  });
}
