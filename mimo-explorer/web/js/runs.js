// Rollouts: everything you've started, live or finished. Refreshes itself while anything is running.
import { $, api, esc, money, dur, ago, statusPill, rewardBadge, DOMAIN_NAME, LIVE_STATUSES, sk, emptyState } from "./util.js";
import { icon, DOMAIN_ICON } from "./icons.js";
import { getSession } from "./session.js";

let timer = null, alive = true;
const FAILED = ["failed", "cancelled", "interrupted"];
export function unmount() { alive = false; clearTimeout(timer); }

const head = (stats = "") => `<div class="page-h"><div><h1>Rollouts</h1><p>Everything you've run. They keep going when you close the page.</p></div>${stats}</div>`;

export async function mount(el) {
  alive = true;
  if (!getSession().user) {
    el.innerHTML = `<div class="wrap page">${head()}<div class="panel">${emptyState("user", "Sign in to run rollouts",
      "Pick an environment, choose a model from HF Inference Providers, and watch it work in an HF Sandbox. Runs are billed to your Hugging Face account.",
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
      ? `<div class="rs-table"><div class="rs-row head"><span>Status</span><span>Task</span><span>Model</span><span>Reward</span><span style="text-align:right">Cost</span><span style="text-align:right">Started</span></div>${shown.map(row).join("")}</div>`
      : `<div class="panel">${runs.length ? emptyState("filter", "Nothing in this view", "Try another filter.")
        : emptyState("play", "No rollouts yet", "Pick an environment and run one. It shows up here with its live progress, grade and cost.", `<a class="btn primary" href="#/">${icon("grid", 15)}Browse environments</a>`)}</div>`;
    const spent = runs.reduce((s, r) => s + ((r.cost || {}).total || 0), 0);
    const scored = runs.filter((r) => r.reward != null);
    const full = scored.filter((r) => r.reward >= 0.999).length;
    $("#rs-stats", el).innerHTML = `<div><b>${runs.length}</b><span>rollouts</span></div><div><b>${money(spent)}</b><span>spent</span></div>
      <div><b>${scored.length ? `${full}/${scored.length}` : "–"}</b><span>full marks</span></div>`;
  };
  const load = async () => {
    if (!alive) return;
    try { runs = (await api("/api/runs")).runs; if (alive) draw(); }
    catch (e) { $("#rs-list", el).innerHTML = `<div class="panel">${emptyState("alert", "Couldn't load your rollouts", esc(e.message))}</div>`; }
    if (alive && runs.some((r) => LIVE_STATUSES.includes(r.status))) timer = setTimeout(load, 3000);
  };
  load();
}

function row(r) {
  const end = r.finished_at || Date.now() / 1000;
  return `<a class="rs-row" href="#/run/${esc(r.id)}" style="--dc:var(--c-${r.domain})">
    <span class="rs-status">${statusPill(r.status)}</span>
    <span class="rs-task"><b>${esc(r.title)}</b><em><span class="dot"></span>${esc(DOMAIN_NAME[r.domain])} · ${esc(r.task_id)}</em></span>
    <span class="rs-model">${esc(r.model.split("/")[1] || r.model)}</span>
    <span class="rs-reward">${rewardBadge(r.reward, r.status)}</span>
    <span class="rs-cost">${money((r.cost || {}).total)}</span>
    <span class="rs-when">${ago(r.created_at)}<em>${dur(end - (r.started_at || r.created_at))}</em></span></a>`;
}
