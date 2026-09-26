// My rollouts: everything you've started, live or finished. Refreshes itself while anything is running.
import { $, api, esc, money, dur, ago, statusPill, rewardBadge, DOMAIN_NAME } from "./util.js";
import { getSession } from "./session.js";

let timer = null, alive = true;
const LIVE = ["queued", "starting", "setup", "running", "verifying"];
export function unmount() { alive = false; clearTimeout(timer); }

export async function mount(el) {
  alive = true;
  if (!getSession().user) {
    el.innerHTML = `<div class="wrap page"><h1 class="page-h">My rollouts</h1><p class="muted">Sign in with Hugging Face to run rollouts and see them here.</p>
      <p><a class="btn primary" href="/login">Sign in with Hugging Face</a></p></div>`;
    return;
  }
  el.innerHTML = `<div class="wrap page"><div class="page-top"><h1 class="page-h">My rollouts</h1><span class="muted sm" id="rs-note"></span></div>
    <div class="rs-filters" id="rs-filters"></div><div id="rs-list"><div class="skel-list"><i></i><i></i><i></i><i></i></div></div></div>`;
  let filter = "all";
  $("#rs-filters", el).addEventListener("click", (e) => { const b = e.target.closest("[data-f]"); if (b) { filter = b.dataset.f; draw(); } });
  let runs = [];
  const draw = () => {
    const counts = { all: runs.length, live: runs.filter((r) => LIVE.includes(r.status)).length, done: runs.filter((r) => r.status === "done").length,
                     failed: runs.filter((r) => ["failed", "cancelled", "interrupted"].includes(r.status)).length };
    $("#rs-filters", el).innerHTML = [["all", "All"], ["live", "Running"], ["done", "Finished"], ["failed", "Failed or stopped"]]
      .map(([k, l]) => `<button data-f="${k}" aria-pressed="${filter === k}">${l} <span>${counts[k]}</span></button>`).join("");
    const shown = runs.filter((r) => filter === "all" || (filter === "live" ? LIVE.includes(r.status) : filter === "done" ? r.status === "done"
      : ["failed", "cancelled", "interrupted"].includes(r.status)));
    $("#rs-list", el).innerHTML = shown.length ? `<div class="rs-table">${shown.map(row).join("")}</div>`
      : `<p class="empty">${runs.length ? "Nothing in this view." : `No rollouts yet. <a href="#/">Pick a task</a> and run one.`}</p>`;
    const spent = runs.reduce((s, r) => s + ((r.cost || {}).total || 0), 0);
    const scored = runs.filter((r) => r.reward != null);
    const full = scored.filter((r) => r.reward >= 0.999).length;
    $("#rs-note", el).innerHTML = [`${runs.length} rollout${runs.length === 1 ? "" : "s"}`, `${money(spent)} spent`,
      scored.length ? `${full} of ${scored.length} graded scored full marks` : "", counts.live ? `<b>${counts.live} running</b> · updates live` : ""]
      .filter(Boolean).join(" · ");
  };
  const load = async () => {
    if (!alive) return;
    try { runs = (await api("/api/runs")).runs; draw(); } catch (e) { $("#rs-list", el).innerHTML = `<p class="empty">${esc(e.message)}</p>`; }
    if (alive && runs.some((r) => LIVE.includes(r.status))) timer = setTimeout(load, 3000);
  };
  load();
}

function row(r) {
  const end = r.finished_at || Date.now() / 1000;
  return `<a class="rs-row" href="#/run/${esc(r.id)}" style="--dc:var(--c-${r.domain})">
    <span class="rs-status">${statusPill(r.status)}</span>
    <span class="rs-task"><i class="dot"></i><b>${esc(r.title)}</b><em>${esc(DOMAIN_NAME[r.domain])} · ${esc(r.task_id)}</em></span>
    <span class="rs-model">${esc(r.model.split("/")[1] || r.model)}</span>
    <span class="rs-reward">${rewardBadge(r.reward, r.status)}</span>
    <span class="rs-cost">${money((r.cost || {}).total)}</span>
    <span class="rs-when">${ago(r.created_at)}<em>${dur(end - (r.started_at || r.created_at))}</em></span></a>`;
}
