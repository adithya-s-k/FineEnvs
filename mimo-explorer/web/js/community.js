// Community: everyone's public rollouts, and the tasks nobody has tried yet. Anonymous by construction: the API
// never returns who ran what. Filters and the view live in the URL, so a filtered page can be shared.
import { $, $$, api, esc, money, ago, rewardBadge, rewardText, DOMAIN_NAME, vals, sk, emptyState, REPORT_URL, fmt, storage } from "./util.js";
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
const pct = (x) => `${Math.round(x * 100)}%`;
// mean reward -> a colour between the "bad" and "ok" tokens, so the heatmap reads at a glance in both themes
const heat = (m) => `color-mix(in srgb, color-mix(in srgb, var(--ok) ${Math.round(m * 100)}%, var(--bad)) ${18 + Math.round(m * 22)}%, transparent)`;
function ring(r, size = 40) {
  const k = r == null ? 0 : Math.max(0, Math.min(1, Number(r))), c = 2 * Math.PI * 16;
  const cls = r == null ? "none" : k >= 0.999 ? "full" : k > 0 ? "part" : "zero";
  return `<span class="cm-ring ${cls}" style="--s:${size}px" title="Reward ${r == null ? "–" : Number(r).toFixed(3)}">
    <svg viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="16" class="bg"/><circle cx="20" cy="20" r="16" class="fg"
      stroke-dasharray="${(k * c).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 20 20)"/></svg><b>${rewardText(r)}</b></span>`;
}
function card(r) {
  const thumb = r.shot
    ? `<img loading="lazy" src="/api/runs/${esc(r.id)}/artifacts/screenshot.jpg" alt="">`
    : `<span class="cm-glyph">${icon(DOMAIN_ICON[r.domain], 26)}</span>`;
  return `<a class="cm-card" href="#/run/${esc(r.id)}" data-cd="${esc(r.domain)}" style="--dc:var(--c-${r.domain})">
    <span class="cm-thumb${r.shot ? " shot" : ""}">${thumb}<span class="cm-ringpos">${ring(r.reward)}</span></span>
    <span class="cm-cb"><span class="cm-dtag">${icon(DOMAIN_ICON[r.domain], 12)}${esc(DOMAIN_NAME[r.domain])}</span>
      <b class="cm-title">${esc(r.title)}</b>
      <span class="cm-meta"><span title="${esc(r.model)}">${esc(r.endpoint ? r.model : short(r.model))}</span>
        <span>${esc(r.endpoint ? "own endpoint" : r.provider || "auto")}</span><span>${money((r.cost || {}).total)}</span><span>${ago(r.created_at)}</span></span></span></a>`;
}
function row(r) {
  return `<a class="runrow" href="#/run/${esc(r.id)}" style="--dc:var(--c-${r.domain})">
    ${rewardBadge(r.reward, r.status)}<span class="m"><b style="font-weight:500">${esc(r.title)}</b><br><span class="faint xs">${icon(DOMAIN_ICON[r.domain], 11)} ${esc(DOMAIN_NAME[r.domain])}
      · ${esc(r.endpoint ? "own endpoint" : r.provider || "auto")}${r.judge ? ` · judge ${esc(short(r.judge))}` : ""}</span></span>
    <span class="c sm-hide">${esc(r.endpoint ? r.model : short(r.model))}</span><span class="c">${money((r.cost || {}).total)}</span>
    <span class="w2">${ago(r.created_at)}</span>${icon("chevronRight", 14)}</a>`;
}

function rolloutsView(el, f, sync) {
  const body = $("#cm-body", el);
  let layout = storage.get("cm-layout") || "grid";
  body.innerHTML = `
    <div class="cm-hero" id="cm-hero">${sk.line(100, 70)}</div>
    <div class="cm-overview">
      <section class="panel"><div class="panel-h"><h2>${icon("gauge", 15)}Rewards</h2><span class="aside">rollouts shown, by reward</span></div>
        <div class="panel-b" id="cm-dist">${sk.lines(90, 80)}</div></section>
      <section class="panel"><div class="panel-h"><h2>${icon("grid", 15)}Coverage</h2><span class="aside">tasks with a public rollout</span></div>
        <div class="panel-b" id="cm-cov">${sk.lines(90, 80, 85)}</div></section>
    </div>
    <section class="panel cm-board"><div class="panel-h"><h2>${icon("flag", 15)}Leaderboard</h2><span class="aside">mean reward by model and domain · click a cell to filter</span></div>
      <div class="panel-b" id="cm-board">${sk.lines(90, 80, 85, 70)}</div></section>
    <div class="cm-toolbar" id="cm-filters"></div>
    <div class="cm-active" id="cm-active"></div>
    <div id="cm-rows">${sk.lines(95, 90, 92)}</div>
    <div class="more-row" id="cm-more" hidden><span class="muted sm"></span><button class="btn" type="button">Load more</button></div>`;
  let offset = 0, facets = null;
  const LABEL = { model: (v) => short(v), served: (v) => v, judge: (v) => "judge " + short(v), thinking: (v) => "thinking " + (THINK[v] || v),
                  reward: (v) => (REWARD.find(([k]) => k === v) || [v, v])[1], q: (v) => `"${v}"`, domain: (v) => DOMAIN_NAME[v] };
  const load = async (append = false) => {
    const q = new URLSearchParams({ limit: "36", offset: String(append ? offset : 0) });
    for (const k of ["domain", "model", "served", "judge", "reward", "thinking", "sort", "q"]) if (f[k]) q.set(k, f[k]);
    let d;
    try { d = await api(`/api/community?${q}`); } catch (e) { $("#cm-rows", el).innerHTML = `<p class="err-text sm">${esc(e.message)}</p>`; return; }
    if (!alive) return;
    const s = d.stats, fc = (facets = d.facets);
    if (!append) {
      const all = fc.domain.reduce((n, [, c]) => n + c, 0);
      const covered = Object.values(d.coverage).reduce((n, c) => n + c.tasks, 0);
      $("#cm-top", el).innerHTML = "";
      $("#cm-hero", el).innerHTML = `
        <div class="cm-tile"><span>Public rollouts</span><b>${fmt.format(all)}</b><em>${fmt.format(d.total)} match these filters</em></div>
        <div class="cm-tile"><span>Tasks covered</span><b>${fmt.format(covered)}<small> / ${fmt.format(d.all_tasks)}</small></b>
          <i class="cm-meter"><i style="width:${Math.max(0.5, covered / d.all_tasks * 100).toFixed(2)}%"></i></i></div>
        <div class="cm-tile"><span>Models</span><b>${fc.model.length}</b><em>${fc.served.length} ways of serving them</em></div>
        <div class="cm-tile"><span>Mean reward</span><b>${s.mean == null ? "–" : s.mean.toFixed(2)}</b><em>over ${fmt.format(s.scored)} scored</em></div>
        <div class="cm-tile"><span>Full marks</span><b>${s.scored ? pct(s.full / s.scored) : "–"}</b><em>${fmt.format(s.full)} of ${fmt.format(s.scored)} rollouts</em></div>`;
      const top = Math.max(1, ...s.histogram);
      $("#cm-dist", el).innerHTML = s.scored ? `<div class="cm-hist">${s.histogram.map((n, i) => `<span title="${n} rollout${n === 1 ? "" : "s"} scoring ${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}"
          style="--h:${(n / top * 100).toFixed(1)}%;--m:${(i + 0.5) / 10}"><i></i>${n ? `<em>${n}</em>` : ""}</span>`).join("")}</div>
        <div class="cm-axis"><span>0</span><span>0.5</span><span>1</span></div>` : `<p class="muted sm">No scored rollouts for these filters.</p>`;
      $("#cm-cov", el).innerHTML = DOMS.map((k) => { const c = d.coverage[k] || { tasks: 0, total: 0, runs: 0 };
        return `<button type="button" class="cm-covrow" data-dom="${k}" style="--dc:var(--c-${k})" aria-pressed="${f.domain === k}">
          <span class="n">${icon(DOMAIN_ICON[k], 13)}${esc(DOMAIN_NAME[k])}</span>
          <i class="cm-meter"><i style="width:${c.total ? Math.max(c.tasks ? 1.5 : 0, c.tasks / c.total * 100).toFixed(2) : 0}%"></i></i>
          <span class="v">${fmt.format(c.tasks)}<small> / ${fmt.format(c.total)}</small></span><span class="r">${fmt.format(c.runs)} run${c.runs === 1 ? "" : "s"}</span></button>`; }).join("");
      const mx = s.matrix || {};
      $("#cm-board", el).innerHTML = s.by_model.length ? `<div class="cm-heatwrap"><table class="cm-heat"><thead><tr><th>#</th><th>Model</th><th>Overall</th>
        ${DOMS.map((k) => `<th style="--dc:var(--c-${k})">${icon(DOMAIN_ICON[k], 12)}${esc(DOMAIN_NAME[k])}</th>`).join("")}</tr></thead><tbody>
        ${s.by_model.map((m, i) => `<tr><td class="rank">${i + 1}</td><td class="mname" title="${esc(m.model)}"><button class="link" data-pick-model="${esc(m.model)}">${esc(short(m.model))}</button>${m.custom ? ' <span class="chip">own endpoint</span>' : ""}</td>
          <td class="cell all" style="background:${heat(m.mean)}"><b>${m.mean.toFixed(2)}</b><small>${m.runs} run${m.runs === 1 ? "" : "s"}${m.full ? ` · ${m.full} full` : ""}</small></td>
          ${DOMS.map((k) => { const c = (mx[m.model] || {})[k];
            return c ? `<td class="cell"><button type="button" data-cell="${esc(m.model)}|${k}" style="background:${heat(c.mean)}"><b>${c.mean.toFixed(2)}</b><small>${c.runs}</small></button></td>`
                     : `<td class="cell empty"><span>–</span></td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`
        : `<p class="muted sm">No scored rollouts for these filters.</p>`;
      const opt = (key, label, list, fmtv = (x) => x) => `<label class="cm-sel"><span>${label}</span><select class="input" data-f="${key}"><option value="">Any</option>
        ${list.map(([v, c]) => `<option value="${esc(v)}" ${f[key] === v ? "selected" : ""}>${esc(fmtv(v))} (${c})</option>`).join("")}</select></label>`;
      const dcount = Object.fromEntries(fc.domain);
      const nMore = ["served", "judge", "thinking", "reward"].filter((k) => f[k]).length;
      $("#cm-filters", el).innerHTML = `
        <div class="cm-chips" role="group" aria-label="Domain">${[["", "All", all], ...DOMS.map((k) => [k, DOMAIN_NAME[k], dcount[k] || 0])].map(([k, l, n]) =>
          `<button type="button" class="cm-chip" data-dom="${k}" aria-pressed="${f.domain === k}" style="--dc:var(--c-${k || "code"})">${k ? icon(DOMAIN_ICON[k], 13) : ""}${l}<small>${fmt.format(n)}</small></button>`).join("")}</div>
        <div class="cm-tools">
          <div class="search">${icon("search", 15)}<input id="cm-q" type="search" placeholder="Search tasks" value="${esc(f.q)}"></div>
          ${opt("model", "Model", fc.model, short)}
          <details class="cm-more" ${nMore ? "open" : ""}><summary class="btn">${icon("filter", 14)}Filters${nMore ? `<span class="count">${nMore}</span>` : ""}</summary>
            <div class="cm-pop">${opt("served", "Served by", fc.served)}${opt("judge", "Judge", fc.judge, short)}
              <label class="cm-sel"><span>Reward</span><select class="input" data-f="reward">${REWARD.map(([v, l]) => `<option value="${v}" ${f.reward === v ? "selected" : ""}>${v ? l : "Any"}</option>`).join("")}</select></label>
              ${opt("thinking", "Thinking", fc.thinking, (v) => THINK[v] || v)}</div></details>
          <label class="cm-sel"><span>Sort</span><select class="input" data-f="sort">${SORT.map(([v, l]) => `<option value="${v}" ${f.sort === v ? "selected" : ""}>${l}</option>`).join("")}</select></label>
          <div class="seg cm-layout" role="group" aria-label="Layout"><button type="button" data-layout="grid" aria-pressed="${layout === "grid"}" title="Cards">${icon("grid", 14)}</button><button type="button" data-layout="list" aria-pressed="${layout === "list"}" title="List">${icon("list", 14)}</button></div>
        </div>`;
      const active = ["domain", "model", "served", "judge", "reward", "thinking", "q"].filter((k) => f[k]);
      $("#cm-active", el).innerHTML = active.length ? active.map((k) => `<button type="button" class="cm-pill" data-unset="${k}">${esc(LABEL[k](f[k]))}${icon("x", 12)}</button>`).join("")
        + `<button class="link sm" id="cm-clear" type="button">Clear all</button>` : "";
      $("#cm-rows", el).innerHTML = `<div class="${layout === "grid" ? "cm-cards" : "runlist cm-list"}"></div>`;
    }
    const box = $("#cm-rows > div", el);
    box.insertAdjacentHTML("beforeend", d.runs.map(layout === "grid" ? card : row).join(""));
    const n = box.children.length;
    offset = n;
    $("#cm-more", el).hidden = n >= d.total;
    $("#cm-more span", el).textContent = `Showing ${n} of ${d.total}`;
    if (!d.total && !append) $("#cm-rows", el).innerHTML = emptyState("filter", "Nothing matches", "Try fewer filters, or switch to Tasks to find one nobody has run.");
  };
  let t;
  const reload = () => { sync(); load(); };
  // a screenshot that won't load falls back to the domain tile (no inline onerror: the CSP forbids it)
  body.addEventListener("error", (e) => {
    const img = e.target;
    if (img.tagName !== "IMG" || !img.closest(".cm-thumb")) return;
    const th = img.closest(".cm-thumb"), dom = th.closest(".cm-card").dataset.cd;
    th.classList.remove("shot");
    img.replaceWith(Object.assign(document.createElement("span"), { className: "cm-glyph", innerHTML: icon(DOMAIN_ICON[dom], 26) }));
  }, true);
  body.addEventListener("click", (e) => {
    const b = e.target.closest("[data-dom]");
    if (b) { f.domain = f.domain === b.dataset.dom && b.classList.contains("cm-covrow") ? "" : b.dataset.dom; reload(); return; }
    const pm = e.target.closest("[data-pick-model]");
    if (pm) { f.model = pm.dataset.pickModel; reload(); return; }
    const cell = e.target.closest("[data-cell]");
    if (cell) { [f.model, f.domain] = cell.dataset.cell.split("|"); reload(); return; }
    const u = e.target.closest("[data-unset]");
    if (u) { f[u.dataset.unset] = ""; reload(); return; }
    const lay = e.target.closest("[data-layout]");
    if (lay) { layout = lay.dataset.layout; storage.set("cm-layout", layout); load(); return; }
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
