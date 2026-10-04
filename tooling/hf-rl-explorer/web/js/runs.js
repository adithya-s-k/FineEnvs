// Rollouts: everything you've started, live or finished. Refreshes itself while anything is running.
import { $, api, esc, dur, ago, statusPill, rewardBadge, LIVE_STATUSES, sk, emptyState, visibilityBadge } from "./util.js";
import { icon } from "./icons.js";
import { getSession } from "./session.js";

let timer = null, alive = true;
const FAILED = ["failed", "cancelled", "interrupted"];
export function unmount() { alive = false; clearTimeout(timer); }

const head = (stats = "") => `<div class="page-h"><div><h1>My rollouts</h1><p>Only yours, public and private. They keep going when you close the page. Everyone's public rollouts are under <a class="u" href="/community">Community</a>.</p></div>${stats}</div>`;

export async function mount(el) {
  alive = true;
  if (!getSession().user) {
    el.innerHTML = `<div class="wrap page">${head()}<div class="panel">${emptyState("user", "Sign in to run rollouts",
      "Pick a task, choose an agent and a model from HF Inference Providers, and watch it work in an HF Sandbox. Runs are billed to your Hugging Face account.",
      `<button class="btn primary" type="button" data-signin>${icon("user", 15)}Sign in</button>`)}</div></div>`;
    return;
  }
  el.innerHTML = `<div class="wrap page">${head(`<div class="ex-stats" id="rs-stats"></div>`)}
    <div class="seg" id="rs-filters" role="group" aria-label="Filter rollouts"></div>
    <div id="rs-list"><div class="rs-table">${Array.from({ length: 5 }, () => `<div class="rs-row">${sk.box(22, "width:90px;border-radius:99px")}
      <div style="display:grid;gap:6px">${sk.line(70)}${sk.line(40, 10)}</div>${sk.line(60)}${sk.line(60)}${sk.line(70)}${sk.line(80)}</div>`).join("")}</div></div></div>`;
  let filter = "all";
  $("#rs-filters", el).addEventListener("click", (e) => { const b = e.target.closest("[data-f]"); if (b) { filter = b.dataset.f; draw(); } });
  let runs = [];
  const draw = () => {
    const counts = { all: runs.length, live: runs.filter((r) => LIVE_STATUSES.includes(r.status)).length, done: runs.filter((r) => r.status === "done").length,
                     failed: runs.filter((r) => FAILED.includes(r.status)).length };
    $("#rs-filters", el).innerHTML = [["all", "All"], ["live", "Running"], ["done", "Finished"], ["failed", "Failed or stopped"]]
      .map(([k, l]) => `<button data-f="${k}" aria-pressed="${filter === k}">${k === "live" && counts.live ? '<i class="pulse" style="color:var(--live)"></i>' : ""}${l}<span>${counts[k]}</span></button>`).join("");
    const shown = runs.filter((r) => filter === "all" || (filter === "live" ? LIVE_STATUSES.includes(r.status) : filter === "done" ? r.status === "done" : FAILED.includes(r.status)));
    $("#rs-list", el).innerHTML = shown.length
      ? `<div class="rs-table"><div class="rs-row head"><span>Status</span><span>Task</span><span>Model</span><span>Reward</span><span>Agent</span><span style="text-align:right">Started</span></div>${shown.map(row).join("")}</div>`
      : `<div class="panel">${runs.length ? emptyState("filter", "Nothing in this view", "Try another filter.")
        : emptyState("play", "No rollouts yet", "Open a task and run one. It shows up here with its live trajectory and reward.", `<a class="btn primary" href="/">${icon("grid", 15)}Browse environments</a>`)}</div>`;
    const scored = runs.filter((r) => r.reward != null);
    $("#rs-stats", el).innerHTML = `<div><b>${runs.length}</b><span>rollouts</span></div>
      <div><b>${scored.length}</b><span>scored rollouts</span></div>`;
  };
  const load = async () => {
    if (!alive) return;
    try { runs = (await api("/api/runs")).runs; if (alive) draw(); }
    catch (e) { $("#rs-list", el).innerHTML = `<div class="panel">${emptyState("alert", "Couldn't load your rollouts", esc(e.message))}</div>`; }
    if (alive && runs.some((r) => LIVE_STATUSES.includes(r.status))) timer = setTimeout(load, 3000);
  };
  load();
}

const COLL = { mimo: "grid", terminal: "terminal", data: "table", swe: "code", mixed: "flask" };
function row(r) {
  const end = r.finished_at || Date.now() / 1000;
  const model = r.endpoint ? r.model : (r.model || "").split("/")[1] || r.model;
  return `<a class="rs-row" href="/run/${esc(r.id)}" style="--dc:var(--c-${r.collection || "hub"})">
    <span class="rs-status">${statusPill(r.status)} ${visibilityBadge(r.visibility)}</span>
    <span class="rs-task"><b>${esc(r.title)}</b><em>${icon(COLL[r.collection] || "database", 12)}${esc(r.dataset)}</em></span>
    <span class="rs-model">${esc(model)}</span>
    <span class="rs-reward">${rewardBadge(r.reward, r.status)}</span>
    <span class="rs-cost">${esc(r.runner === "nemo-gym" ? "NeMo Gym prediction" : r.runner === "mimo" ? (r.domain === "music" ? "MiMo scorer" : "MiMo harness") : r.harness)}</span>
    <span class="rs-when">${ago(r.created_at)}<em>${dur(end - (r.started_at || r.created_at))}</em></span></a>`;
}
