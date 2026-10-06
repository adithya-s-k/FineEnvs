// Router + pages: #/ (overview), #/tasks, #/task/<id>, #/run/<run>/<model>/<task_id>, #/live/<id>, #/play.
// /api/config sets the mode: "env" (the environment's server: Play and live episodes here) or "explorer" (the eval
// Space: read-only, no env session; playing links out to the environment Space).
import { getAllEpisodes, getEpisode, getLiveEpisode, getLiveEpisodes, getReference, getTask, getTasks } from "./api.js";
import { BerthChart } from "./chart.js";
import { DIFF_RANK, ago, callKinds, clockAt, divertWindows, escapeHtml, evaluatePlan, fmtNum, fmtPct, hasCranes, horizonOf, movesOf, publishedPlan, sectionsLabel, badgesHtml } from "./model.js";
import { createStage, stageMarkup } from "./stage.js";
import { checkSummary, renderTranscript } from "./transcript.js";
import { playPage } from "./play.js";
import { nameOf, overviewPage } from "./overview.js";

const EMBED = new URLSearchParams(location.search).get("embed") === "1";
document.documentElement.classList.toggle("embed", EMBED);

const CONFIG = await fetch(new URL("/api/config", location.origin).toString(), { headers: { Accept: "application/json" } })
  .then((r) => (r.ok ? r.json() : {}))
  .catch(() => ({}));
const EXPLORER = CONFIG.mode === "explorer";
const PLAY_URL = CONFIG.play_url || "https://huggingface.co/spaces/FineEnvs/PortSimEnv";
if (EXPLORER) {
  document.documentElement.classList.add("explorer");
  document.title = "PortSimEnv v1 eval · Port of Barcelona";
  const brand = document.querySelector("header.top .brand");
  if (brand) brand.textContent = "PortSimEnv v1 eval";
  const play = document.querySelector('.topnav [data-nav="play"]');
  if (play) {
    play.href = PLAY_URL;
    play.target = "_blank";
    play.rel = "noopener";
    play.title = "Play an episode in the environment Space";
    play.textContent = "Play ↗";
    play.removeAttribute("data-nav");
    play.parentElement.appendChild(play); // after Explorer: the only link that leaves this Space
  }
}

const app = document.getElementById("app");
const crumbs = document.getElementById("crumbs");
let teardown = null;
let routeSeq = 0;
const sceneMod = () => import("./scene.js");

const enc = encodeURIComponent;
const taskHref = (id, plan) => `#/task/${enc(id)}${plan ? `?plan=${enc(plan)}` : ""}`;
const runHref = (run, model, taskId) => `#/run/${enc(run)}/${enc(model)}/${enc(taskId)}`;
const liveHref = (id) => `#/live/${enc(id)}`;

function setCrumbs(items) {
  crumbs.innerHTML = items.map((it) => (it.href ? `<a href="${it.href}">${escapeHtml(it.label)}</a>` : `<span>${escapeHtml(it.label)}</span>`)).join('<span class="sep">/</span>');
}

function errorBox(err) {
  return `<div class="error">Could not load data. <span class="muted">${escapeHtml(err && err.message ? err.message : String(err))}</span></div>`;
}

function skeletonRows(cols, n) {
  return Array.from({ length: n }, () => `<tr class="skel">${Array.from({ length: cols }, () => '<td><i></i></td>').join("")}</tr>`).join("");
}


function disruptionSummary(list) {
  const counts = {};
  for (const d of list || []) counts[d] = (counts[d] || 0) + 1;
  return Object.entries(counts)
    .map(([k, v]) => (v > 1 ? `${v} ${k}` : k))
    .join(", ");
}

/* ------------------------------------------------------------------ */
/* Sortable table helper                                               */
/* ------------------------------------------------------------------ */

function sortableTable(table, cols, rows, { initial, rowAttrs, onRow }) {
  let sortKey = initial ? initial.key : cols[0].k;
  let dir = initial ? initial.dir : 1;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  function head() {
    thead.innerHTML = `<tr>${cols
      .map((c) => {
        const on = c.k === sortKey;
        return `<th class="${c.num ? "num" : ""} ${c.opt ? "opt" : ""}" ${c.title ? `title="${escapeHtml(c.title)}"` : ""}><button type="button" class="th-b${on ? " on" : ""}" data-k="${c.k}">${escapeHtml(c.label)}<span class="arr">${on ? (dir > 0 ? "▲" : "▼") : ""}</span></button></th>`;
      })
      .join("")}</tr>`;
  }
  function body() {
    const col = cols.find((c) => c.k === sortKey) || cols[0];
    const val = col.sort || col.get;
    const sorted = [...rows].sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
    });
    tbody.innerHTML = sorted.length
      ? sorted.map((r) => `<tr ${rowAttrs ? rowAttrs(r) : ""}>${cols.map((c) => `<td class="${c.num ? "num" : ""} ${c.opt ? "opt" : ""}">${c.html ? c.html(r) : escapeHtml(c.get(r) ?? "–")}</td>`).join("")}</tr>`).join("")
      : `<tr><td colspan="${cols.length}" class="muted">No rows.</td></tr>`;
  }
  thead.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-k]");
    if (!b) return;
    if (b.dataset.k === sortKey) dir = -dir;
    else {
      sortKey = b.dataset.k;
      dir = cols.find((c) => c.k === sortKey).num ? -1 : 1;
    }
    head();
    body();
  });
  if (onRow) {
    tbody.addEventListener("click", (e) => {
      if (e.target.closest("a")) return;
      const tr = e.target.closest("tr[data-row]");
      if (tr) onRow(tr.dataset.row);
    });
  }
  head();
  body();
  return {
    setRows(r) {
      rows = r;
      body();
    },
  };
}

/* ------------------------------------------------------------------ */
/* Tasks page                                                          */
/* ------------------------------------------------------------------ */

async function tasksPage(seq) {
  setCrumbs([{ label: "Tasks" }]);
  app.innerHTML = `
  <div class="page tasks-page">
    <section class="live-sec" hidden>
      <div class="sec-head"><h2>Live</h2><span class="muted small" title="Episodes running or recently finished on this server; refreshes every 5 s">this server</span></div>
      <table class="tbl click" id="l-table"><thead><tr><th>Episode</th><th class="opt">Task</th><th class="opt">Started</th><th>Status</th><th class="num opt">Plans checked</th><th class="num">Reward</th></tr></thead><tbody></tbody></table>
    </section>
    <section class="tasks-sec">
      <div class="sec-head">
        <h2>Tasks <span class="muted" id="t-count"></span></h2>
        <div class="filters">
          <input id="t-q" type="search" placeholder="Filter" aria-label="Filter tasks" autocomplete="off">
          <select id="t-split" aria-label="Split"><option value="">All splits</option></select>
          <select id="t-diff" aria-label="Difficulty"><option value="">All difficulties</option></select>
        </div>
      </div>
      <table class="tbl click" id="t-table"><thead><tr><th>Task</th><th class="opt">Quay</th><th class="opt">Terminal</th><th class="num opt">Week</th><th>Difficulty</th><th class="num">Ships</th><th class="opt">Disruptions</th></tr></thead><tbody>${skeletonRows(7, 10)}</tbody></table>
    </section>
    <div class="side-col">
      <section class="board-sec">
        <div class="sec-head"><h2>Models</h2></div>
        <table class="tbl" id="b-table"><thead><tr><th>Model</th><th class="num">Episodes</th><th class="num">Reward</th><th class="num">Feasible</th><th class="num">Submitted</th></tr></thead><tbody>${skeletonRows(5, 3)}</tbody></table>
      </section>
    </div>
  </div>`;
  if (!EXPLORER) startLiveList(seq); // live episodes exist only on the environment's server
  let tasks;
  let eps;
  try {
    [tasks, eps] = await Promise.all([getTasks(), getAllEpisodes().catch(() => ({ runs: [], episodes: [] }))]);
  } catch (err) {
    if (seq !== routeSeq) return;
    app.querySelector(".tasks-sec").innerHTML = errorBox(err);
    app.querySelector(".board-sec").remove();
    return;
  }
  if (seq !== routeSeq) return;
  const byTask = new Map();
  for (const e of eps.episodes) {
    if (!byTask.has(e.task_id)) byTask.set(e.task_id, []);
    byTask.get(e.task_id).push(e);
  }
  const hasRuns = eps.episodes.length > 0;
  const rows = tasks.map((t) => {
    const es = byTask.get(t.task_id) || [];
    const rw = es.filter((e) => e.reward != null);
    return { ...t, n_ep: es.length, reward: rw.length ? rw.reduce((a, e) => a + e.reward, 0) / rw.length : null };
  });
  const cols = [
    { k: "task_id", label: "Task", get: (r) => r.task_id, html: (r) => `<a href="${taskHref(r.task_id)}">${escapeHtml(r.task_id)}</a>` },
    { k: "quay", label: "Quay", get: (r) => r.quay, opt: true },
    { k: "terminal", label: "Terminal", get: (r) => r.terminal, opt: true },
    { k: "week", label: "Week", get: (r) => r.week, num: true, opt: true },
    { k: "difficulty", label: "Difficulty", get: (r) => r.difficulty, sort: (r) => DIFF_RANK[r.difficulty] ?? 9 },
    { k: "ships", label: "Ships", get: (r) => r.ships, num: true },
    { k: "disruptions", label: "Disruptions", get: (r) => disruptionSummary(r.disruptions), sort: (r) => (r.disruptions || []).length, opt: true, title: "Closures, late arrivals, crane overruns, extra calls, calls diverted from the other quay" },
  ];
  if (hasRuns) {
    cols.push({ k: "reward", label: "Reward", get: (r) => (r.reward == null ? "–" : r.reward.toFixed(2)), sort: (r) => r.reward, num: true, title: "Mean reward over all episodes of this task" });
    cols.push({ k: "n_ep", label: "Episodes", get: (r) => r.n_ep, num: true, opt: true });
  }
  const splits = [...new Set(tasks.map((t) => t.split).filter(Boolean))].sort();
  const diffs = [...new Set(tasks.map((t) => t.difficulty).filter(Boolean))].sort((a, b) => (DIFF_RANK[a] ?? 9) - (DIFF_RANK[b] ?? 9));
  const splitSel = app.querySelector("#t-split");
  const diffSel = app.querySelector("#t-diff");
  splitSel.insertAdjacentHTML("beforeend", splits.map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join(""));
  diffSel.insertAdjacentHTML("beforeend", diffs.map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join(""));
  if (splits.length < 2) splitSel.hidden = true;
  if (diffs.length < 2) diffSel.hidden = true;
  if (EXPLORER && splits.includes("eval")) splitSel.value = "eval"; // the eval Space: the eval split first
  const q = app.querySelector("#t-q");
  const count = app.querySelector("#t-count");
  const table = sortableTable(app.querySelector("#t-table"), cols, rows, {
    initial: { key: "task_id", dir: 1 },
    rowAttrs: (r) => `data-row="${escapeHtml(r.task_id)}"`,
    onRow: (id) => (location.hash = taskHref(id)),
  });
  const filter = () => {
    const s = q.value.trim().toLowerCase();
    const f = rows.filter((r) => (!splitSel.value || r.split === splitSel.value) && (!diffSel.value || r.difficulty === diffSel.value) && (!s || `${r.task_id} ${r.quay} ${r.terminal} ${r.week} ${(r.disruptions || []).join(" ")}`.toLowerCase().includes(s)));
    table.setRows(f);
    count.textContent = f.length === rows.length ? `${rows.length}` : `${f.length} of ${rows.length}`;
  };
  q.addEventListener("input", filter);
  splitSel.addEventListener("change", filter);
  diffSel.addEventListener("change", filter);
  filter();

  // model board
  const board = app.querySelector(".board-sec");
  if (!hasRuns) {
    board.innerHTML = `<div class="sec-head"><h2>Models</h2></div><p class="muted small">No runs yet.</p>`;
    return;
  }
  const multiRun = new Set(eps.episodes.map((e) => e.run)).size > 1;
  const groups = new Map();
  for (const e of eps.episodes) {
    const k = `${e.run}\u0000${e.model}`;
    if (!groups.has(k)) groups.set(k, { run: e.run, model: e.model, eps: [] });
    groups.get(k).eps.push(e);
  }
  const brows = [...groups.values()].map((g) => {
    const n = g.eps.length;
    const mean = (f) => g.eps.reduce((a, e) => a + (f(e) ? 1 : 0), 0) / n;
    const rw = g.eps.filter((e) => e.reward != null);
    return { run: g.run, model: g.model, n, reward: rw.length ? rw.reduce((a, e) => a + e.reward, 0) / rw.length : null, feasible: mean((e) => e.feasible), submitted: mean((e) => e.submitted) };
  });
  const bcols = [
    { k: "model", label: "Model", get: (r) => r.model, html: (r) => `<span title="${escapeHtml(r.model)}">${escapeHtml(nameOf(r.model))}</span>${multiRun ? `<div class="muted small">${escapeHtml(r.run)}</div>` : ""}` },
    { k: "n", label: "Episodes", get: (r) => r.n, num: true, opt: true },
    { k: "reward", label: "Reward", get: (r) => fmtNum(r.reward, 3), sort: (r) => r.reward, num: true, title: "Mean reward" },
    { k: "feasible", label: "Feasible", get: (r) => fmtPct(r.feasible), sort: (r) => r.feasible, num: true, title: "Share of episodes whose final plan had no conflicts" },
    { k: "submitted", label: "Submitted", get: (r) => fmtPct(r.submitted), sort: (r) => r.submitted, num: true, title: "Share of episodes that called submit_plan" },
  ];
  sortableTable(app.querySelector("#b-table"), bcols, brows, { initial: { key: "reward", dir: -1 } });
}

/* ------------------------------------------------------------------ */
/* Shared bits for task + rollout pages                                */
/* ------------------------------------------------------------------ */

function shipPhase(row, t) {
  const s = row.ship;
  if (!row.placed) return t < s.arrival - 2.5 ? "at sea" : t < s.arrival ? "approaching" : "waiting";
  if (t < Math.min(s.arrival, row.berth) - 3) return "at sea";
  if (t < s.arrival) return "approaching";
  if (t < row.berth - 0.6) return "waiting";
  if (t < row.berth) return "docking";
  if (t < row.dep) return row.hold > 0 && t >= row.finish ? "held by wind" : "alongside";
  if (t < row.dep + 3.6) return "departing";
  return "sailed";
}
const PHASE_DOT = { alongside: "ok", "held by wind": "warn", waiting: "warn", docking: "ok", departing: "", approaching: "", "at sea": "", sailed: "" };

function shipsTableHtml(task, ev) {
  const kinds = callKinds(task);
  const cr = hasCranes(task);
  const head = `<thead><tr><th class="num">#</th><th>Ship</th><th class="num" title="Arrival hour">Arrives</th><th class="num opt" title="${cr ? "Hours of work at the plan's crane count" : "Hours alongside needed"}">Hours</th>${cr ? '<th class="num opt" title="Quay cranes in this plan (min–max in the tooltip)">Cranes</th>' : ""}<th class="opt" title="Published plan: docking hour @ sections">Planned</th><th title="This plan: docking hour @ sections">This plan</th><th class="num opt" title="Departure hour in this plan">Departs</th><th class="num opt" title="Due departure hour">Due</th><th class="num" title="Hours late">Delay</th><th class="opt">Moved</th><th class="num" title="sections × hours late (× weight) + emergency penalty + 5 if moved">Cost</th><th class="opt now-h" title="Submitted schedule at this hour; physical traffic in 3D may wait for clearance">Plan status</th></tr></thead>`;
  const rows = ev.rows
    .map((r) => {
      const s = r.ship;
      const planned = s.planned_hour == null ? '<span class="muted" title="Unscheduled call: no published slot">–</span>' : `h ${s.planned_hour} @ ${sectionsLabel(s.planned_section, s.planned_section + s.sections - 1)}`;
      let mine = '<span class="bad-t">not placed</span>';
      if (r.placed) {
        const hChg = s.planned_hour != null && r.berth !== s.planned_hour;
        const sChg = r.moved;
        mine = `<span class="${hChg ? "chg" : ""}">h ${r.berth}</span> @ <span class="${sChg ? "chg" : ""}">${sectionsLabel(r.section, r.last)}</span>`;
      }
      const conf = r.conflicts.length ? ` <span class="bad-t" title="${escapeHtml(r.conflicts.join("\n"))}">${r.conflicts.length} conflict${r.conflicts.length > 1 ? "s" : ""}</span>` : "";
      const tip = `${s.from_port || "?"} → ${s.to_port || "?"} · ${Math.round(s.length_m)} m · ${s.sections} sections${s.draught_m ? ` · draught ${s.draught_m} m` : ""}${s.imo ? ` · IMO ${s.imo}` : ""}${cr ? ` · ${movesOf(task, s)} moves, ${s.min_cranes}–${s.max_cranes} cranes` : ""}`;
      return `<tr data-row="${s.id}" class="${r.conflicts.length || !r.placed ? "has-bad" : ""}">
        <td class="num muted">${s.id}</td>
        <td><span class="sname" title="${escapeHtml(tip)}">${escapeHtml(s.name)}</span>${badgesHtml(s)}${kindTag(kinds.get(s.id))}${conf}</td>
        <td class="num" title="${escapeHtml(fmtClock(task, s.arrival))}">${s.arrival}</td>
        <td class="num opt"${r.hold ? ` title="then held by the wind ${r.hold} h"` : ""}>${r.placed && r.work != null ? r.work : s.handling}${r.hold ? '<span class="muted">+w</span>' : ""}</td>
        ${cr ? `<td class="num opt" title="${s.min_cranes}–${s.max_cranes} cranes · ${movesOf(task, s)} moves">${r.placed ? r.cranes : "–"}</td>` : ""}
        <td class="opt">${planned}</td>
        <td>${mine}</td>
        <td class="num opt">${r.placed ? r.dep : "–"}</td>
        <td class="num opt">${s.due}</td>
        <td class="num ${r.delay > 0 ? "bad-t" : "muted"}">${r.placed ? (r.delay > 0 ? `+${r.delay}` : "0") : "–"}</td>
        <td class="opt">${r.moved ? "moved" : ""}</td>
        <td class="num">${r.placed ? r.cost : "–"}</td>
        <td class="opt now"></td>
      </tr>`;
    })
    .join("");
  return `<table class="tbl ships-tbl click">${head}<tbody>${rows}</tbody></table>`;
}

function kindTag(kind) {
  if (!kind) return "";
  if (kind.kind === "divert") return ` <span class="muted small" title="Call diverted from quay ${escapeHtml(kind.from)} while it is closed; no published slot">from ${escapeHtml(kind.from)}</span>`;
  return ' <span class="muted small" title="Unscheduled extra call; no published slot">extra</span>';
}

function noticesHtml(task) {
  const items = (task.notices || []).map((n) => `<li>${escapeHtml(n)}</li>`);
  for (const d of divertWindows(task)) {
    if ((task.notices || []).some((n) => /divert/i.test(n))) break;
    const names = (d.ships || []).map((id) => (task.ships.find((s) => s.id === Number(id)) || {}).name || `ship ${id}`);
    items.push(`<li>Quay ${escapeHtml(d.from_quay)} is closed from hour ${d.start} to ${d.end} (${escapeHtml(fmtClock(task, d.start))} – ${escapeHtml(fmtClock(task, d.end))}); diverted here: ${escapeHtml(names.join(", "))}.</li>`);
  }
  return items.join("") || '<li class="muted">None.</li>';
}

function fmtClock(task, h) {
  const c = clockAt(task, h);
  return `${c.day} ${c.date} ${c.hm} UTC`;
}

function bindShipsTable(container, ev, stage) {
  const tbody = container.querySelector("tbody");
  if (!tbody) return { setTime() {}, mark() {} };
  tbody.addEventListener("mouseover", (e) => {
    const tr = e.target.closest("tr[data-row]");
    stage.hover(tr ? Number(tr.dataset.row) : null);
  });
  tbody.addEventListener("mouseleave", () => stage.hover(null));
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-row]");
    if (!tr) return;
    const id = Number(tr.dataset.row);
    stage.select(stage.selected === id ? null : id, { focus: stage.selected !== id, from: "table" });
  });
  const cells = new Map();
  for (const tr of tbody.querySelectorAll("tr[data-row]")) cells.set(Number(tr.dataset.row), { tr, now: tr.querySelector(".now") });
  return {
    setTime(t) {
      for (const r of ev.rows) {
        const c = cells.get(r.id);
        if (!c) continue;
        const ph = shipPhase(r, t);
        if (c.now._ph !== ph) {
          c.now._ph = ph;
          const dot = PHASE_DOT[ph];
          c.now.innerHTML = `${dot ? `<i class="dot ${dot}"></i>` : ""}${ph}`;
          c.now.classList.toggle("muted", !dot);
        }
      }
    },
    mark(hover, selected) {
      for (const [id, c] of cells) {
        c.tr.classList.toggle("hov", id === hover);
        c.tr.classList.toggle("sel", id === selected);
      }
    },
  };
}

function statusHtml(ev) {
  if (ev.feasible) return `<i class="dot ok"></i>Feasible`;
  const bits = [];
  if (ev.issues.length) bits.push(`${ev.issues.length} conflict${ev.issues.length > 1 ? "s" : ""}`);
  if (ev.missing.length) bits.push(`${ev.missing.length} not placed`);
  return `<i class="dot bad"></i>Infeasible · ${bits.join(" · ")}`;
}

function costKv(task, ev, ref, extra = []) {
  const late = ev.rows.filter((r) => r.placed && r.delay > 0).length;
  const waiting = ev.rows.filter((r) => r.placed && r.wait > 0).length;
  const rows = [
    ["Status", statusHtml(ev)],
    ["Delay cost", `${ev.delayCost}`, "Sum over ships of sections × hours late (× 3 for priority ships), plus the per-hour penalty for an emergency docking after its deadline"],
    ["Moves", `${ev.moves} × 5 = ${ev.moves * 5}`, "Ships docked at different sections than published"],
    ["Total cost", `<b>${ev.cost}</b>${ev.missing.length ? ' <span class="muted">(placed ships only)</span>' : ""}`],
  ];
  if (ref) {
    rows.push(["Naive re-plan", `${ref.naive_cost}`, "Keep sections, push conflicting ships to the next free hour"]);
    rows.push(["Optimum", `${ref.optimal_cost}${ref.proven_optimal ? ' <span class="muted">proven</span>' : ""}`, ref.solver ? `Solver: ${ref.solver}` : ""]);
    if (ev.feasible) {
      const gap = ev.cost - ref.optimal_cost;
      rows.push(["Gap to optimum", `${gap}${ref.optimal_cost > 0 ? ` <span class="muted">(${Math.round((gap / ref.optimal_cost) * 100)}%)</span>` : ""}`]);
    }
  }
  rows.push(["Late ships", `${late} of ${task.ships.length}`]);
  rows.push(["Waited to dock", `${waiting}`]);
  for (const e of extra) rows.push(e);
  return `<dl class="kv">${rows.map(([k, v, tip]) => `<dt${tip ? ` title="${escapeHtml(tip)}"` : ""}>${escapeHtml(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

function taskMeta(task) {
  const c = clockAt(task, 0);
  const r = task.rules || {};
  const rules = hasCranes(task) ? ` · ${r.crane_pool} cranes · ≤${r.max_moves_per_hour} moves/h` : "";
  return `Quay ${escapeHtml(task.quay)} · ${escapeHtml(task.terminal)} · week ${task.week} (from ${c.day} ${c.date}) · sections ${task.first_section}–${task.last_section} · ${escapeHtml(task.difficulty)} · ${task.ships.length} ships${rules}`;
}

/* ------------------------------------------------------------------ */
/* Task page                                                           */
/* ------------------------------------------------------------------ */

async function taskPage(seq, id, params) {
  setCrumbs([{ label: "Tasks", href: "#/tasks" }, { label: id }]);
  app.innerHTML = `
  <div class="page task-page">
    <div class="page-head"><h1>${escapeHtml(id)}</h1><span class="meta muted" id="meta">&nbsp;</span></div>
    ${stageMarkup()}
    <div class="cols">
      <section class="ships-sec"><div class="sec-head"><h2>Ships</h2><span class="muted small" id="ships-note"></span></div><div id="ships"><table class="tbl"><tbody>${skeletonRows(6, 12)}</tbody></table></div></section>
      <aside class="side">
        <section><div class="sec-head"><h2>Plan</h2></div><div id="cost"><dl class="kv skel-kv">${"<dt><i></i></dt><dd><i></i></dd>".repeat(7)}</dl></div></section>
        <section><div class="sec-head"><h2>Notices</h2></div><ul class="notices" id="notices"><li class="skel"><i></i></li><li class="skel"><i></i></li></ul></section>
        <section id="eps-sec" hidden><div class="sec-head"><h2>Rollouts</h2></div><div id="eps"></div></section>
      </aside>
    </div>
  </div>`;
  const modP = sceneMod();
  let task, ref, eps, mod;
  try {
    [task, ref, eps, mod] = await Promise.all([getTask(id), getReference(id).catch(() => null), getAllEpisodes().catch(() => ({ episodes: [] })), modP]);
  } catch (err) {
    if (seq !== routeSeq) return;
    app.querySelector(".task-page").innerHTML = errorBox(err);
    return;
  }
  if (seq !== routeSeq) return;
  const metaEl = app.querySelector("#meta");
  metaEl.innerHTML = taskMeta(task);
  metaEl.title = metaEl.textContent;
  app.querySelector("#notices").innerHTML = noticesHtml(task);

  const episodes = eps.episodes.filter((e) => e.task_id === task.task_id);
  const options = [{ key: "published", label: "Published (broken)", plan: publishedPlan(task) }];
  if (ref) {
    options.push({ key: "naive", label: "Naive re-plan", plan: ref.naive_plan });
    options.push({ key: "optimal", label: "Optimal", plan: ref.optimal_plan });
  }
  for (const e of episodes) options.push({ key: `ep:${e.run}:${e.model}`, label: `${nameOf(e.model)}${new Set(episodes.map((x) => x.run)).size > 1 ? ` · ${e.run}` : ""}`, episode: e, plan: null });

  if (episodes.length) {
    app.querySelector("#eps-sec").hidden = false;
    app.querySelector("#eps").innerHTML = `<table class="tbl"><thead><tr><th>Model</th><th class="num">Reward</th><th class="num">Cost</th><th>Final plan</th></tr></thead><tbody>${episodes
      .map((e) => `<tr><td><a href="${runHref(e.run, e.model, e.task_id)}" title="${escapeHtml(e.model)}">${escapeHtml(nameOf(e.model))}</a><div class="muted small">${escapeHtml(e.run)}</div></td><td class="num">${fmtNum(e.reward, 2)}</td><td class="num">${fmtNum(e.cost)}</td><td>${e.submitted ? (e.feasible ? '<i class="dot ok"></i>feasible' : '<i class="dot bad"></i>infeasible') : '<span class="muted">not submitted</span>'}</td></tr>`)
      .join("")}</tbody></table>`;
  }

  const allPlans = () => options.filter((o) => o.plan).map((o) => o.plan);
  let H = horizonOf(task, allPlans());
  const stageRoot = app.querySelector(".task-page");
  let shipsCtl = null;
  let curHover = null;
  const stage = createStage(stageRoot, {
    task,
    horizon: H,
    sceneMod: mod,
    onTime: (t) => shipsCtl && shipsCtl.setTime(t),
    onHover: (hid) => {
      curHover = hid;
      if (shipsCtl) shipsCtl.mark(curHover, stage.selected);
    },
    onSelect: () => shipsCtl && shipsCtl.mark(curHover, stage.selected),
  });

  stage.tl.innerHTML = `<label class="ovbox plan-l">Plan <select class="ovs" id="plan-sel" aria-label="Plan shown">${options.map((o) => `<option value="${escapeHtml(o.key)}">${escapeHtml(o.label)}</option>`).join("")}</select></label><span class="ovbox status" id="plan-status"></span>`;
  const sel = stage.tl.querySelector("#plan-sel");
  const statusEl = stage.tl.querySelector("#plan-status");
  const want = params.get("plan");
  sel.value = options.some((o) => o.key === want) ? want : options.some((o) => o.key === "optimal") ? "optimal" : "published";

  async function showPlan(key) {
    const opt = options.find((o) => o.key === key) || options[0];
    let extra = [];
    if (opt.episode && !opt.plan) {
      statusEl.innerHTML = "Loading plan…";
      try {
        const ro = await getEpisode(opt.episode.run, opt.episode.model, opt.episode.task_id);
        opt.rollout = ro;
        opt.plan = (ro.final && ro.final.plan) || (ro.steps && ro.steps.length ? ro.steps[ro.steps.length - 1].plan : []) || [];
      } catch (err) {
        statusEl.textContent = "Could not load this rollout";
        return;
      }
      if (seq !== routeSeq) return;
    }
    if (opt.episode) {
      const g = (opt.rollout && opt.rollout.final && opt.rollout.final.grade) || {};
      extra = [
        ["Reward", fmtNum(opt.rollout ? opt.rollout.reward : opt.episode.reward, 3)],
        ["Submitted", opt.episode.submitted ? "yes" : "no"],
        ["Rollout", `<a href="${runHref(opt.episode.run, opt.episode.model, opt.episode.task_id)}">open transcript</a>`],
      ];
      if (g.clean_fraction != null) extra.splice(1, 0, ["Clean fraction", fmtNum(g.clean_fraction, 2)]);
    }
    const ev = evaluatePlan(task, opt.plan);
    const newH = horizonOf(task, allPlans());
    if (newH !== H) H = newH;
    stage.setEvaluation(ev, H);
    statusEl.innerHTML = statusHtml(ev) + (ev.feasible || !ev.issues.length ? ` · cost ${ev.cost}` : "");
    app.querySelector("#cost").innerHTML = costKv(task, ev, ref, extra);
    app.querySelector("#ships").innerHTML = shipsTableHtml(task, ev);
    app.querySelector("#ships-note").textContent = `${opt.label}`;
    shipsCtl = bindShipsTable(app.querySelector("#ships"), ev, stage);
    shipsCtl.setTime(stage.time);
    shipsCtl.mark(curHover, stage.selected);
    const p = new URLSearchParams(location.hash.split("?")[1] || ""); // keep t= and cam= for shared views
    p.set("plan", opt.key);
    history.replaceState(null, "", `${location.pathname}${location.search}#/task/${enc(task.task_id)}?${p.toString()}`);
  }
  sel.addEventListener("change", () => showPlan(sel.value));
  const t0 = Number(params.get("t"));
  if (Number.isFinite(t0) && t0 > 0) stage.setTime(t0);
  await showPlan(sel.value);

  teardown = () => stage.destroy();
}

/* ------------------------------------------------------------------ */
/* Rollout page                                                        */
/* ------------------------------------------------------------------ */

async function rolloutPage(seq, run, model, taskId) {
  setCrumbs([{ label: "Tasks", href: "#/tasks" }, { label: taskId, href: taskHref(taskId) }, { label: nameOf(model) }]);
  app.innerHTML = `
  <div class="page ro-page">
    <div class="page-head"><h1 title="${escapeHtml(model)}">${escapeHtml(nameOf(model))}</h1><span class="meta muted">${escapeHtml(run)} · <a href="${taskHref(taskId)}">${escapeHtml(taskId)}</a></span></div>
    <div class="ro-grid">
      <div class="ro-right">
        ${stageMarkup({ chartTitle: "Dock chart of the selected step" })}
      </div>
      <div class="ro-left">
        <section><div class="sec-head"><h2>Final grade</h2></div><div id="grade"><dl class="kv skel-kv">${"<dt><i></i></dt><dd><i></i></dd>".repeat(6)}</dl></div></section>
        <section><div class="sec-head"><h2>Transcript</h2></div><div id="transcript" class="transcript"><div class="msg skel"><i></i></div><div class="msg skel"><i></i></div><div class="msg skel"><i></i></div></div></section>
      </div>
    </div>
  </div>`;
  let task, ref, ro, mod;
  try {
    [task, ref, ro, mod] = await Promise.all([getTask(taskId), getReference(taskId).catch(() => null), getEpisode(run, model, taskId), sceneMod()]);
  } catch (err) {
    if (seq !== routeSeq) return;
    app.querySelector(".ro-page").innerHTML = errorBox(err);
    return;
  }
  if (seq !== routeSeq) return;
  let steps = ro.steps || [];
  if (!steps.length && ro.final && Array.isArray(ro.final.plan)) steps = [{ tool: "final", plan: ro.final.plan }];
  const H = horizonOf(task, [publishedPlan(task), ...(ref ? [ref.naive_plan, ref.optimal_plan] : []), ...steps.map((st) => st.plan)]);

  // grade
  const g = (ro.final && ro.final.grade) || {};
  const u = ro.usage || {};
  const gradeRows = [
    ["Reward", `<b>${fmtNum(ro.reward ?? g.reward, 3)}</b>`],
    ["Submitted", ro.final && ro.final.submitted ? "yes" : `no${ro.end_reason ? ` <span class="muted">(${escapeHtml(ro.end_reason)})</span>` : ""}`],
    ["Feasible", g.feasible == null ? "–" : g.feasible ? '<i class="dot ok"></i>yes' : '<i class="dot bad"></i>no'],
    ["Cost", `${fmtNum(g.cost)}${g.delay_cost != null ? ` <span class="muted">= delay ${g.delay_cost} + ${g.moves} moves × 5</span>` : ""}`],
    ["Naive re-plan", fmtNum(g.naive_cost ?? (ref && ref.naive_cost))],
    ["Optimum", fmtNum(g.optimal_cost ?? (ref && ref.optimal_cost))],
  ];
  if (g.quality != null) gradeRows.push(["Quality", fmtNum(g.quality, 3), "Where the cost falls between naive (0) and optimum (1)"]);
  if (g.clean_fraction != null) gradeRows.push(["Clean fraction", fmtNum(g.clean_fraction, 2), "Share of ships without a violation"]);
  gradeRows.push(["Tool calls", `${steps.filter((s) => s.tool === "check_plan").length} checks · ${steps.filter((s) => s.tool === "submit_plan").length} submit`]);
  if (ro.seconds != null) gradeRows.push(["Time", `${fmtNum(ro.seconds, 0)} s`]);
  if (u.input_tokens != null) gradeRows.push(["Tokens", `${u.input_tokens.toLocaleString()} in · ${(u.output_tokens || 0).toLocaleString()} out${u.cost_usd != null ? ` · $${u.cost_usd}` : ""}`]);
  app.querySelector("#grade").innerHTML = `<dl class="kv">${gradeRows.map(([k, v, tip]) => `<dt${tip ? ` title="${escapeHtml(tip)}"` : ""}>${escapeHtml(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;

  const right = app.querySelector(".ro-right");
  const wide = () => window.matchMedia("(min-width: 1000px)").matches;
  const n = task.last_section - task.first_section + 1;
  const chartMax = () => (wide() ? Math.max(120, Math.min(n * 11 + 30, window.innerHeight * 0.36)) : null);
  const stage = createStage(right, { task, horizon: H, sceneMod: mod, chartMaxHeight: chartMax() });
  const onResize = () => stage.setChartMaxHeight(chartMax());
  window.addEventListener("resize", onResize);

  let tr = null;
  const viewer = stepViewer(stage, task, { emptyNote: "the model never proposed a plan", onShow: (i, scroll) => {
    if (!tr) return;
    tr.select(i);
    if (scroll) tr.scrollTo(i);
  } });
  viewer.setSteps(steps, H);
  tr = renderTranscript(app.querySelector("#transcript"), ro, task, { onStep: (i) => viewer.show(i, false), horizon: H });
  viewer.show(steps.length - 1, false);

  teardown = () => {
    window.removeEventListener("resize", onResize);
    tr.destroy();
    stage.destroy();
  };
}

/* ------------------------------------------------------------------ */
/* Step viewer (rollout + live): prev/next over the agent's plans      */
/* ------------------------------------------------------------------ */

function stepViewer(stage, task, { onShow, emptyNote = "no plan checked yet" }) {
  stage.tl.innerHTML = `<div class="ovbox stepnav"><button type="button" class="ovb" data-d="-1" aria-label="Previous step">‹</button><span class="step-l"></span><button type="button" class="ovb" data-d="1" aria-label="Next step">›</button></div><span class="ovbox status" id="step-status"></span>`;
  const stepL = stage.tl.querySelector(".step-l");
  const statusEl = stage.tl.querySelector("#step-status");
  const prevB = stage.tl.querySelector('[data-d="-1"]');
  const nextB = stage.tl.querySelector('[data-d="1"]');
  let steps = [];
  let H = 0;
  let k = -1;
  const readable = (i) => (steps[i] && Array.isArray(steps[i].plan) ? steps[i].plan : null);
  function planFor(i) {
    if (readable(i)) return { plan: readable(i), note: "" };
    for (let j = i - 1; j >= 0; j--) if (readable(j)) return { plan: readable(j), note: `showing step ${j + 1}` };
    return { plan: publishedPlan(task), note: "showing the published plan" };
  }
  function show(i, scroll = false) {
    if (!steps.length) {
      k = -1;
      stepL.textContent = "No plan";
      prevB.disabled = nextB.disabled = true;
      stage.setEvaluation(evaluatePlan(task, publishedPlan(task)), H);
      statusEl.innerHTML = `<span class="muted">${escapeHtml(emptyNote)} · showing the published plan</span>`;
      return;
    }
    k = Math.max(0, Math.min(steps.length - 1, i));
    const st = steps[k];
    const { plan, note } = planFor(k);
    const ev = evaluatePlan(task, plan);
    stage.setEvaluation(ev, H);
    stepL.textContent = st.tool === "final" ? "Final plan" : `Step ${k + 1} of ${steps.length} · ${st.tool}`;
    statusEl.innerHTML = note ? `<i class="dot bad"></i>Unreadable plan · ${escapeHtml(note)}` : statusHtml(ev) + (ev.feasible ? ` · cost ${ev.cost}` : "");
    prevB.disabled = k <= 0;
    nextB.disabled = k >= steps.length - 1;
    if (onShow) onShow(k, scroll);
  }
  prevB.addEventListener("click", () => show(k - 1, true));
  nextB.addEventListener("click", () => show(k + 1, true));
  return {
    setSteps(s, h) {
      steps = s || [];
      H = h;
    },
    show,
    get k() {
      return k;
    },
  };
}

/* ------------------------------------------------------------------ */
/* Live episodes                                                       */
/* ------------------------------------------------------------------ */

function liveStatus(e) {
  if (!e.done) return '<i class="dot warn"></i>running';
  return `<i class="dot ${e.grade && e.grade.feasible ? "ok" : e.end_reason === "submitted" ? "bad" : ""}"></i>${escapeHtml(e.end_reason || "done")}`;
}

function startLiveList(seq) {
  let timer = 0;
  const load = async () => {
    let eps;
    try {
      eps = await getLiveEpisodes();
    } catch {
      eps = [];
    }
    if (seq !== routeSeq) return;
    const sec = app.querySelector(".live-sec");
    if (!sec) return;
    sec.hidden = !eps.length;
    sec.querySelector("tbody").innerHTML = eps
      .slice(0, 8)
      .map((e) => `<tr data-row="${escapeHtml(e.episode_id)}"><td><a href="${liveHref(e.episode_id)}" title="${escapeHtml(e.episode_id)}">${escapeHtml(String(e.episode_id).slice(0, 12))}</a></td><td class="opt">${escapeHtml(e.task_id)}</td><td class="opt muted" title="${escapeHtml(new Date(e.started * 1000).toISOString())}">${escapeHtml(ago(e.started))}</td><td>${liveStatus(e)}</td><td class="num opt">${e.checks ?? "–"}</td><td class="num">${e.grade && e.grade.reward != null ? fmtNum(e.grade.reward, 2) : "–"}</td></tr>`)
      .join("");
  };
  app.querySelector("#l-table tbody").addEventListener("click", (e) => {
    if (e.target.closest("a")) return;
    const tr = e.target.closest("tr[data-row]");
    if (tr) location.hash = liveHref(tr.dataset.row);
  });
  load();
  timer = setInterval(load, 5000);
  teardown = () => clearInterval(timer);
}

async function livePage(seq, id) {
  setCrumbs([{ label: "Tasks", href: "#/tasks" }, { label: "Live" }, { label: id }]);
  app.innerHTML = `
  <div class="page ro-page live-page">
    <div class="page-head"><h1>Live episode</h1><span class="meta muted" id="meta">${escapeHtml(id)}</span></div>
    <div class="ro-grid">
      <div class="ro-right">${stageMarkup({ chartTitle: "Dock chart of the selected step" })}</div>
      <div class="ro-left">
        <section><div class="sec-head"><h2>Episode</h2><span class="muted small" id="poll" title="Refreshes every 2 s while running"></span></div><div id="ep-kv"><dl class="kv skel-kv">${"<dt><i></i></dt><dd><i></i></dd>".repeat(5)}</dl></div></section>
        <section><div class="sec-head"><h2>Steps</h2></div><div id="steps" class="transcript"><div class="msg skel"><i></i></div><div class="msg skel"><i></i></div></div></section>
      </div>
    </div>
  </div>`;
  let ep, task, ref, mod;
  try {
    ep = await getLiveEpisode(id);
    [task, ref, mod] = await Promise.all([getTask(ep.task_id), getReference(ep.task_id).catch(() => null), sceneMod()]);
  } catch (err) {
    if (seq !== routeSeq) return;
    app.querySelector(".live-page").innerHTML = err && err.status === 404 ? `<div class="error">This episode is unknown or has expired. <a href="#/tasks">Back to tasks</a></div>` : errorBox(err);
    return;
  }
  if (seq !== routeSeq) return;
  const meta = app.querySelector("#meta");
  meta.innerHTML = `${escapeHtml(id)} · <a href="${taskHref(task.task_id)}">${escapeHtml(task.task_id)}</a> · quay ${escapeHtml(task.quay)} · ${escapeHtml(task.difficulty)}`;
  const right = app.querySelector(".ro-right");
  const wide = () => window.matchMedia("(min-width: 1000px)").matches;
  const n = task.last_section - task.first_section + 1;
  const chartMax = () => (wide() ? Math.max(120, Math.min(n * 11 + 30, window.innerHeight * 0.36)) : null);
  const basePlans = [publishedPlan(task), ...(ref ? [ref.naive_plan, ref.optimal_plan] : [])];
  let H = horizonOf(task, basePlans);
  const stage = createStage(right, { task, horizon: H, sceneMod: mod, chartMaxHeight: chartMax() });
  const onResize = () => stage.setChartMaxHeight(chartMax());
  window.addEventListener("resize", onResize);
  const stepsEl = app.querySelector("#steps");
  let charts = [];
  let follow = true;
  const viewer = stepViewer(stage, task, {
    onShow: (k, scroll) => {
      for (const c of stepsEl.querySelectorAll(".call")) c.classList.toggle("sel", Number(c.dataset.step) === k);
      if (scroll) {
        const c = stepsEl.querySelector(`.call[data-step="${k}"]`);
        if (c) c.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    },
  });
  stepsEl.addEventListener("click", (e) => {
    if (e.target.closest("summary, table")) return;
    const c = e.target.closest(".call[data-step]");
    if (!c) return;
    const k = Number(c.dataset.step);
    follow = k === (ep.steps || []).length - 1;
    viewer.show(k);
  });

  function renderSteps(steps) {
    for (const c of charts) c.destroy();
    charts = [];
    if (!steps.length) {
      stepsEl.innerHTML = `<p class="muted small">No plan checked yet.</p>`;
      return;
    }
    stepsEl.innerHTML = "";
    steps.forEach((st, k) => {
      const d = document.createElement("div");
      d.className = "call pick";
      d.dataset.step = k;
      const ok = Array.isArray(st.plan);
      let head = `<span class="fn">${escapeHtml(st.tool)}</span>${st.turn != null ? ` <span class="muted">turn ${escapeHtml(st.turn)}</span>` : ""}`;
      if (!ok) head += ' <span class="bad-t">unreadable plan</span>';
      head += `<span class="step-no muted">step ${k + 1}</span>`;
      d.innerHTML = `<div class="call-h">${head}</div>${ok ? '<div class="mini-chart"></div>' : ""}<div class="result">${checkSummary(task, st.result) || ""}</div>`;
      stepsEl.appendChild(d);
      if (ok) {
        const ch = new BerthChart(d.querySelector(".mini-chart"), { mini: true, onPick: () => {} });
        ch.setData(task, evaluatePlan(task, st.plan), H);
        charts.push(ch);
      }
    });
  }

  function renderKv(e) {
    const g = e.grade || {};
    const rows = [
      ["Status", liveStatus(e)],
      ["Task", `<a href="${taskHref(task.task_id)}">${escapeHtml(task.task_id)}</a>`],
      ["Started", `<span title="${escapeHtml(new Date(e.started * 1000).toISOString())}">${escapeHtml(ago(e.started))}</span>`],
      ["Plans checked", `${(e.steps || []).filter((st) => st.tool === "check_plan").length}`],
    ];
    if (e.done && e.grade) {
      rows.push(["Reward", `<b>${fmtNum(g.reward, 3)}</b>`]);
      rows.push(["Feasible", g.feasible ? '<i class="dot ok"></i>yes' : '<i class="dot bad"></i>no']);
      rows.push(["Cost", `${fmtNum(g.cost)}${g.delay_cost != null ? ` <span class="muted">= delay ${g.delay_cost} + ${g.moves} moves × 5</span>` : ""}`]);
      rows.push(["Naive re-plan", fmtNum(g.naive_cost ?? (ref && ref.naive_cost))]);
      rows.push(["Optimum", fmtNum(g.optimal_cost ?? (ref && ref.optimal_cost))]);
    } else if (ref) {
      rows.push(["Naive re-plan", fmtNum(ref.naive_cost)]);
      rows.push(["Optimum", fmtNum(ref.optimal_cost)]);
    }
    app.querySelector("#ep-kv").innerHTML = `<dl class="kv">${rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
  }

  let lastSig = "";
  function apply(e) {
    ep = e;
    const steps = e.steps || [];
    const sig = `${steps.length}|${e.done}|${e.end_reason}`;
    renderKv(e);
    app.querySelector("#poll").textContent = e.done ? "finished" : "updating every 2 s";
    if (sig === lastSig) return;
    lastSig = sig;
    H = horizonOf(task, [...basePlans, ...steps.map((st) => st.plan)]);
    viewer.setSteps(steps, H);
    renderSteps(steps);
    viewer.show(follow || viewer.k < 0 ? steps.length - 1 : viewer.k, false);
  }
  apply(ep);
  let timer = 0;
  let busy = false;
  const poll = async () => {
    if (busy || ep.done) return;
    busy = true;
    try {
      const e = await getLiveEpisode(id);
      if (seq === routeSeq) apply(e);
    } catch (err) {
      if (seq === routeSeq) app.querySelector("#poll").textContent = err && err.status === 404 ? "expired on the server" : "connection lost, retrying";
    } finally {
      busy = false;
    }
    if (ep.done) clearInterval(timer);
  };
  timer = setInterval(poll, 2000);
  teardown = () => {
    clearInterval(timer);
    window.removeEventListener("resize", onResize);
    for (const c of charts) c.destroy();
    stage.destroy();
  };
}

/* ------------------------------------------------------------------ */
/* Explorer mode: pages that need the environment's server             */
/* ------------------------------------------------------------------ */

function elsewherePage(what) {
  document.documentElement.classList.add("on-play"); // no section of this Space is current
  setCrumbs([{ label: what === "play" ? "Play" : "Live" }]);
  const why = what === "play" ? "Playing an episode needs a live environment session" : "Live episodes run on the environment's server";
  app.innerHTML = `
  <div class="page elsewhere">
    <p>${why}; this Space only replays the eval.</p>
    <p><a class="btn primary" href="${escapeHtml(PLAY_URL)}" target="_blank" rel="noopener">Open PortSimEnv ↗</a> <a class="btn" href="#/">Back to the eval</a></p>
  </div>`;
}

/* ------------------------------------------------------------------ */
/* Router                                                              */
/* ------------------------------------------------------------------ */

function route() {
  if (teardown) {
    try {
      teardown();
    } catch (e) {
      console.error(e);
    }
    teardown = null;
  }
  const seq = ++routeSeq;
  const raw = location.hash.replace(/^#/, "") || "/";
  const qi = raw.indexOf("?");
  const path = qi >= 0 ? raw.slice(0, qi) : raw;
  const params = new URLSearchParams(qi >= 0 ? raw.slice(qi + 1) : "");
  const parts = path.split("/").filter(Boolean);
  const dec = (s) => {
    try {
      return decodeURIComponent(s);
    } catch {
      return s;
    }
  };
  window.scrollTo(0, 0);
  document.documentElement.classList.remove("on-play", "on-overview");
  if (parts[0] === "task" && parts[1]) return taskPage(seq, dec(parts.slice(1).join("/")), params);
  if (EXPLORER && (parts[0] === "play" || parts[0] === "live")) return elsewherePage(parts[0]);
  if (parts[0] === "live" && parts[1]) return livePage(seq, dec(parts.slice(1).join("/")));
  if (parts[0] === "play") {
    setCrumbs([]);
    document.documentElement.classList.add("on-play");
    return playPage({ app, params, isCurrent: () => seq === routeSeq, setTeardown: (f) => (teardown = f), sceneMod });
  }
  if (!parts.length) {
    document.documentElement.classList.add("on-overview");
    return overviewPage({ app, setCrumbs, isCurrent: () => seq === routeSeq, params, config: { explorer: EXPLORER, playUrl: PLAY_URL, envUrl: CONFIG.env_url }, sortableTable });
  }
  if (parts[0] === "run" && parts.length >= 4) {
    const run = dec(parts[1]);
    const taskId = dec(parts[parts.length - 1]);
    const model = dec(parts.slice(2, -1).join("/"));
    return rolloutPage(seq, run, model, taskId);
  }
  return tasksPage(seq);
}

window.addEventListener("hashchange", route);
route();
