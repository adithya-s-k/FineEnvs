// Community: everyone's public rollouts, and the tasks nobody has tried yet. Anonymous by construction: the API
// never returns who ran what. Filters and the view live in the URL, so a filtered page can be shared.
import { $, $$, api, esc, money, ago, rewardBadge, rewardText, DOMAIN_NAME, vals, sk, emptyState, REPORT_URL, fmt } from "./util.js";
import { icon, DOMAIN_ICON } from "./icons.js";
import * as explore from "./explore.js";

let alive = false;
export function unmount() { alive = false; }

const REWARD = [["", "Any reward"], ["full", "Full marks"], ["partial", "Partial"], ["zero", "Zero"]];
const THINK = { default: "model default", none: "off", low: "low", medium: "medium", high: "high" };
const SORT = [["new", "Newest"], ["reward_desc", "Highest reward"], ["reward_asc", "Lowest reward"], ["cost_asc", "Cheapest"], ["cost_desc", "Most expensive"]];
const short = (m) => (m || "").split("/")[1] || m || "";

export async function mount(el, qs) {
  alive = true;
  const p = new URLSearchParams(qs || "");
  const f = { view: p.get("view") === "tasks" ? "tasks" : "rollouts", domain: p.get("domain") || "", model: p.get("model") || "",
              served: p.get("served") || "", judge: p.get("judge") || "", reward: p.get("reward") || "", thinking: p.get("thinking") || "",
              sort: p.get("sort") || "new", q: p.get("q") || "", has: p.get("has") || "none" };
  el.innerHTML = `<div class="wrap page">
    <div class="page-h"><div><h1>Community</h1><p>Public rollouts from everyone, shown without who ran them. Compare models on the same tasks, read their traces, or pick a task nobody has tried yet.</p></div>
      <div class="ex-stats" id="cm-top">${sk.line(100, 30)}</div></div>
    <div class="seg" id="cm-view" role="tablist"><button data-view="rollouts">${icon("list", 13)}Rollouts</button><button data-view="tasks">${icon("grid", 13)}Tasks</button></div>
    <div id="cm-body"></div>
    <p class="fine" style="margin-top:18px">Public rollouts may later be released as an open dataset (for example SFT traces) for research.
      Something wrong? <a href="${REPORT_URL}" target="_blank" rel="noopener">Report an issue</a>.</p></div>`;
  const sync = () => {
    const q = new URLSearchParams(Object.entries(f).filter(([k, v]) => v && !(k === "view" && v === "rollouts") && !(k === "sort" && v === "new") && !(k === "has" && v === "none")));
    history.replaceState(null, "", "#/community" + (q.toString() ? "?" + q : ""));
    $$("#cm-view [data-view]", el).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === f.view)));
  };
  $("#cm-view", el).addEventListener("click", (e) => { const b = e.target.closest("[data-view]"); if (b) { f.view = b.dataset.view; render(); } });
  const render = () => { sync(); (f.view === "tasks" ? tasksView : rolloutsView)(el, f, sync); };
  render();
}

// ── rollouts ─────────────────────────────────────────────────────────────────
const DOMS = Object.keys(DOMAIN_NAME);
const num = (x) => (x == null ? "–" : x.toFixed(2));

function row(r) {
  const thumb = r.shot ? `<img class="cm-shot" loading="lazy" src="/api/runs/${esc(r.id)}/artifacts/screenshot.jpg" alt="">` : "";
  return `<a class="runrow" href="#/run/${esc(r.id)}" style="--dc:var(--c-${r.domain})">
    ${rewardBadge(r.reward, r.status)}<span class="m"><b style="font-weight:500">${esc(r.title)}</b><br><span class="faint xs">${icon(DOMAIN_ICON[r.domain], 11)} ${esc(DOMAIN_NAME[r.domain])}
      · ${esc(r.endpoint ? "own endpoint" : r.provider || "auto")}${r.judge ? ` · judge ${esc(short(r.judge))}` : ""}</span></span>
    <span class="t sm-hide">${thumb}</span>
    <span class="c sm-hide">${esc(r.endpoint ? r.model : short(r.model))}</span><span class="c">${money((r.cost || {}).total)}</span>
    <span class="w2">${ago(r.created_at)}</span>${icon("chevronRight", 14)}</a>`;
}

function rolloutsView(el, f, sync) {
  const body = $("#cm-body", el);
  body.innerHTML = `
    <div class="cm-figs" id="cm-figs">${sk.line(60, 22)}</div>
    <section class="panel cm-board"><div class="panel-h"><h2>Leaderboard</h2><span class="aside">mean reward over the rollouts shown · best per column in bold</span></div>
      <div class="panel-b" id="cm-board">${sk.lines(90, 80, 85)}</div></section>
    <div class="cm-filters" id="cm-filters"></div>
    <div class="cm-active" id="cm-active"></div>
    <section class="panel"><div class="panel-h"><h2>Rollouts</h2><span class="aside" id="cm-count"></span></div>
      <div class="panel-b"><div class="runlist cm-list" id="cm-rows">${sk.lines(95, 90, 92, 88)}</div>
      <div class="more-row" id="cm-more" hidden><span class="muted sm"></span><button class="btn" type="button">Load more</button></div></div></section>`;
  let offset = 0;
  const LABEL = { model: (v) => short(v), served: (v) => "served by " + v, judge: (v) => "judge " + short(v), thinking: (v) => "thinking " + (THINK[v] || v),
                  reward: (v) => (REWARD.find(([k]) => k === v) || [v, v])[1].toLowerCase(), q: (v) => `“${v}”`, domain: (v) => DOMAIN_NAME[v] };
  const load = async (append = false) => {
    const q = new URLSearchParams({ limit: "40", offset: String(append ? offset : 0) });
    for (const k of ["domain", "model", "served", "judge", "reward", "thinking", "sort", "q"]) if (f[k]) q.set(k, f[k]);
    let d;
    try { d = await api(`/api/community?${q}`); } catch (e) { $("#cm-rows", el).innerHTML = `<p class="err-text sm">${esc(e.message)}</p>`; return; }
    if (!alive) return;
    const s = d.stats, fc = d.facets;
    if (!append) {
      const all = fc.domain.reduce((n, [, c]) => n + c, 0);
      const covered = Object.values(d.coverage).reduce((n, c) => n + c.tasks, 0);
      $("#cm-top", el).innerHTML = "";
      const fig = (v, l) => `<div><b>${v}</b><span>${l}</span></div>`;
      $("#cm-figs", el).innerHTML = fig(fmt.format(all), "public rollouts") + fig(`${fmt.format(covered)} <small>of ${fmt.format(d.all_tasks)}</small>`, "tasks tried")
        + fig(fc.model.length, "models") + fig(num(s.mean), "mean reward") + fig(s.scored ? `${Math.round(s.full / s.scored * 100)}%` : "–", "full marks");
      const mx = s.matrix || {};
      const best = Object.fromEntries(DOMS.map((k) => [k, Math.max(-1, ...s.by_model.map((m) => ((mx[m.model] || {})[k] || {}).mean ?? -1))]));
      const bestAll = Math.max(-1, ...s.by_model.map((m) => m.mean));
      $("#cm-board", el).innerHTML = s.by_model.length ? `<div class="cm-tablewrap"><table class="mtable cm-lb"><thead><tr><th class="r">#</th><th>Model</th><th class="r">Runs</th><th class="r">Mean</th><th class="r">Full</th>
        ${DOMS.map((k) => `<th class="r">${esc(DOMAIN_NAME[k])}</th>`).join("")}</tr></thead><tbody>
        ${s.by_model.map((m, i) => `<tr><td class="r faint">${i + 1}</td><td title="${esc(m.model)}"><button class="link" data-pick-model="${esc(m.model)}">${esc(short(m.model))}</button>${m.custom ? ' <span class="chip">own endpoint</span>' : ""}</td>
          <td class="r">${m.runs}</td><td class="r ${m.mean === bestAll ? "best" : ""}">${num(m.mean)}</td><td class="r">${m.full}</td>
          ${DOMS.map((k) => { const c = (mx[m.model] || {})[k];
            return c ? `<td class="r"><button class="link cm-cell ${c.mean === best[k] ? "best" : ""}" data-cell="${esc(m.model)}|${k}" title="${c.runs} run${c.runs === 1 ? "" : "s"}">${num(c.mean)}</button></td>`
                     : `<td class="r faint">–</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`
        : `<p class="muted sm">No scored rollouts for these filters.</p>`;
      const opt = (key, label, list, fmtv = (x) => x) => `<select class="input" data-f="${key}" aria-label="${label}"><option value="">${label}</option>
        ${list.map(([v, c]) => `<option value="${esc(v)}" ${f[key] === v ? "selected" : ""}>${esc(fmtv(v))} (${c})</option>`).join("")}</select>`;
      $("#cm-filters", el).innerHTML = `
        <div class="seg" role="group" aria-label="Domain">${[["", "All"], ...Object.entries(DOMAIN_NAME)].map(([k, l]) =>
          `<button type="button" data-dom="${k}" aria-pressed="${f.domain === k}">${l}</button>`).join("")}</div>
        <div class="search">${icon("search", 15)}<input id="cm-q" type="search" placeholder="Search tasks" value="${esc(f.q)}"></div>
        ${opt("model", "Any model", fc.model, short)}
        <select class="input" data-f="reward" aria-label="Reward">${REWARD.map(([v, l]) => `<option value="${v}" ${f.reward === v ? "selected" : ""}>${l}</option>`).join("")}</select>
        <details class="cm-more"><summary class="btn">${icon("filter", 14)}More</summary><div class="cm-pop">
          ${opt("served", "Served by: any", fc.served)}${opt("judge", "Any judge", fc.judge, short)}${opt("thinking", "Any thinking", fc.thinking, (v) => "thinking " + (THINK[v] || v))}</div></details>
        <select class="input cm-sort" data-f="sort" aria-label="Sort">${SORT.map(([v, l]) => `<option value="${v}" ${f.sort === v ? "selected" : ""}>${l}</option>`).join("")}</select>`;
      const active = ["domain", "model", "served", "judge", "reward", "thinking", "q"].filter((k) => f[k]);
      $("#cm-active", el).innerHTML = active.length ? `<span class="faint xs">Filtered by</span>` + active.map((k) => `<button type="button" class="cm-pill" data-unset="${k}">${esc(LABEL[k](f[k]))}${icon("x", 11)}</button>`).join("")
        + `<button class="link xs" id="cm-clear" type="button">Clear</button>` : "";
      $("#cm-rows", el).innerHTML = "";
    }
    $("#cm-rows", el).insertAdjacentHTML("beforeend", d.runs.map(row).join(""));
    const n = $$("#cm-rows .runrow", el).length;
    offset = n;
    $("#cm-count", el).textContent = d.total ? `${n} of ${d.total}` : "";
    $("#cm-more", el).hidden = n >= d.total;
    $("#cm-more span", el).textContent = `Showing ${n} of ${d.total}`;
    if (!d.total && !append) $("#cm-rows", el).innerHTML = emptyState("filter", "Nothing matches", "Try fewer filters, or switch to Tasks to find one nobody has run.");
  };
  let t;
  const reload = () => { sync(); load(); };
  // a screenshot that won't load just disappears (no inline onerror: the CSP forbids it)
  body.addEventListener("error", (e) => { if (e.target.classList?.contains("cm-shot")) e.target.remove(); }, true);
  body.addEventListener("click", (e) => {
    const b = e.target.closest("[data-dom]");
    if (b) { f.domain = b.dataset.dom; reload(); return; }
    const pm = e.target.closest("[data-pick-model]");
    if (pm) { f.model = pm.dataset.pickModel; reload(); return; }
    const cell = e.target.closest("[data-cell]");
    if (cell) { [f.model, f.domain] = cell.dataset.cell.split("|"); reload(); return; }
    const u = e.target.closest("[data-unset]");
    if (u) { f[u.dataset.unset] = ""; reload(); return; }
    if (e.target.closest("#cm-clear")) { for (const k of ["model", "served", "judge", "reward", "thinking", "q", "domain"]) f[k] = ""; reload(); return; }
    if (e.target.closest("#cm-more button")) load(true);
  });
  body.addEventListener("change", (e) => { const s = e.target.closest("[data-f]"); if (s) { f[s.dataset.f] = s.value; reload(); } });
  body.addEventListener("input", (e) => { if (e.target.id === "cm-q") { clearTimeout(t); t = setTimeout(() => { f.q = e.target.value.trim(); reload(); }, 250); } });
  load();
}

// ── tasks ────────────────────────────────────────────────────────────────────
async function tasksView(el, f, sync) {
  const body = $("#cm-body", el);
  body.innerHTML = `<div class="cm-filters" id="ct-filters"></div><div class="panel"><div class="panel-h"><h2>${icon("grid", 15)}Tasks</h2><span class="aside" id="ct-count"></span></div>
    <div class="panel-b"><div class="runlist cm-list" id="ct-rows">${sk.lines(95, 90, 92, 88, 94)}</div>
    <div class="more-row" id="ct-more" hidden><span class="muted sm"></span><button class="btn" type="button">Load more</button></div></div></div>`;
  let idx, counts;
  try { [idx, counts] = await Promise.all([explore.load(), api("/api/community/tasks")]); }
  catch (e) { $("#ct-rows", el).innerHTML = `<p class="err-text sm">${esc(e.message)}</p>`; return; }
  if (!alive) return;
  const doms = Object.fromEntries(idx.domains.map((d) => [d.id, d]));
  const tried = counts.tasks;
  let list = [], shown = 0;
  const seed = Math.random();
  const rank = (id) => { let h = 2166136261; for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619); return ((h >>> 0) ^ (seed * 4294967295)) >>> 0; };
  const filter = () => {
    const q = f.q.toLowerCase();
    list = idx.envs.filter((e) => (!f.domain || e.d === f.domain) && (f.has === "all" || (f.has === "none" ? !tried[e.id] : !!tried[e.id]))
      && (!q || (e.t + " " + e.id).toLowerCase().includes(q)));
    if (f.has === "some") list.sort((a, b) => (tried[b.id].last || 0) - (tried[a.id].last || 0));
    else list.sort((a, b) => rank(a.id) - rank(b.id));   // shuffled, so people don't all pick the same untried task
    shown = 0;
    $("#ct-rows", el).innerHTML = "";
    more();
    const none = idx.envs.filter((e) => (!f.domain || e.d === f.domain) && !tried[e.id]).length;
    $("#cm-top", el).innerHTML = `<div><b>${fmt.format(Object.keys(tried).length)}</b><span>tasks with public rollouts</span></div>
      <div><b>${fmt.format(none)}</b><span>${f.domain ? DOMAIN_NAME[f.domain] + " tasks" : "tasks"} not tried yet</span></div>`;
  };
  const more = () => {
    const next = list.slice(shown, shown + 40);
    $("#ct-rows", el).insertAdjacentHTML("beforeend", next.map((e) => {
      const c = tried[e.id], d = doms[e.d];
      return `<a class="runrow" href="#/task/${encodeURIComponent(e.id)}" style="--dc:var(--c-${e.d})">
        ${c ? `<span class="reward ${c.mean == null ? "none" : c.mean >= 0.999 ? "full" : c.mean > 0 ? "part" : "zero"}" title="mean reward">${c.mean == null ? "–" : rewardText(c.mean)}</span>` : `<span class="status">${icon("play", 11)}New</span>`}
        <span class="m"><b style="font-weight:500">${esc(e.t)}</b><br><span class="faint xs">${icon(DOMAIN_ICON[e.d], 11)} ${esc(d.name)} · ${esc(vals(e.f[d.main]).join(", "))}</span></span>
        <span class="c">${c ? `${c.runs} rollout${c.runs === 1 ? "" : "s"}` : "no rollouts yet"}</span><span class="w2">${c ? "best " + rewardText(c.best) : ""}</span>
        ${icon("chevronRight", 14)}</a>`;
    }).join(""));
    shown += next.length;
    $("#ct-count", el).textContent = `${fmt.format(list.length)} tasks`;
    $("#ct-more", el).hidden = shown >= list.length;
    $("#ct-more span", el).textContent = `Showing ${fmt.format(shown)} of ${fmt.format(list.length)}`;
    if (!list.length) $("#ct-rows", el).innerHTML = emptyState("check", f.has === "none" ? "Every task here has a public rollout" : "No tasks match", "Try another domain or filter.");
  };
  $("#ct-filters", el).innerHTML = `
    <div class="seg" role="group" aria-label="Domain">${[["", "All"], ...Object.entries(DOMAIN_NAME)].map(([k, l]) => `<button type="button" data-dom="${k}" aria-pressed="${f.domain === k}">${l}</button>`).join("")}</div>
    <div class="seg" role="group" aria-label="Rollouts">${[["none", "No rollouts yet"], ["some", "Has rollouts"], ["all", "All tasks"]].map(([k, l]) => `<button type="button" data-has="${k}" aria-pressed="${f.has === k}">${l}</button>`).join("")}</div>
    <div class="search" style="flex:1;min-width:200px">${icon("search", 15)}<input id="ct-q" type="search" placeholder="Search tasks" value="${esc(f.q)}" style="height:34px"></div>`;
  let t;
  body.addEventListener("click", (e) => {
    const b = e.target.closest("[data-dom]");
    if (b) { f.domain = b.dataset.dom; $$("[data-dom]", body).forEach((x) => x.setAttribute("aria-pressed", String(x === b))); sync(); filter(); return; }
    const h = e.target.closest("[data-has]");
    if (h) { f.has = h.dataset.has; $$("[data-has]", body).forEach((x) => x.setAttribute("aria-pressed", String(x === h))); sync(); filter(); return; }
    if (e.target.closest("#ct-more button")) more();
  });
  body.addEventListener("input", (e) => { if (e.target.id === "ct-q") { clearTimeout(t); t = setTimeout(() => { f.q = e.target.value.trim(); sync(); filter(); }, 200); } });
  filter();
}
