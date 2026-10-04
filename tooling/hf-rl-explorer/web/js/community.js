// Public, completed rollouts across all saved runners. Filters live in the URL.
import { $, $$, api, esc, money, ago, rewardBadge, DOMAIN_NAME, sk, emptyState, fmt } from "./util.js";
import { icon } from "./icons.js";

let current = null;
export function unmount() {
  if (current) { current.controller?.abort(); current.events.abort(); clearTimeout(current.timer); }
  current = null;
}
const REWARD = [["", "Any reward"], ["full", "Reward ≥ 0.999"], ["partial", "0 < reward < 0.999"], ["zero", "Reward ≤ 0"]];
const SORT = [["new", "Newest"], ["reward_desc", "Highest reward"], ["reward_asc", "Lowest reward"], ["cost_asc", "Cheapest"], ["cost_desc", "Most expensive"]];
const THINK = { default: "model default", none: "off", low: "low", medium: "medium", high: "high" };
const RUNNER = { harbor: "Harbor", mimo: "MiMo", "nemo-gym": "NeMo Gym", openenv: "OpenEnv", verifiers: "Verifiers", verl: "verl", skyrl: "SkyRL", mcp: "MCP", ors: "ORS" };
const runnerName = (value) => RUNNER[value] || value || "Harbor";
const short = (value) => (value || "").split("/").at(-1) || "Unknown model";
const num = (value) => value == null ? "–" : new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(value);
const KEYS = ["dataset", "runner", "group", "model", "served", "judge", "reward", "thinking", "q"];
const LABEL = { model: short, served: (v) => "served by " + v, judge: (v) => "judge " + short(v), thinking: (v) => "thinking " + (THINK[v] || v),
  reward: (v) => (REWARD.find(([k]) => k === v) || [v, v])[1], q: (v) => `“${v}”`, group: (v) => v, dataset: (v) => v, runner: runnerName };

function row(r) {
  const title = r.title || r.path || r.task_id || "Rollout";
  const details = [r.dataset, runnerName(r.runner), r.domain && DOMAIN_NAME[r.domain],
    r.harness && !["mimo", "nemo-gym"].includes(r.runner) ? r.harness : null,
    r.endpoint ? "own endpoint" : r.provider, r.judge ? `judge ${short(r.judge)}` : null].filter(Boolean).join(" · ");
  return `<a class="runrow" href="/run/${encodeURIComponent(r.id)}">
    ${rewardBadge(r.reward, r.status)}<span class="m"><b title="${esc(title)}">${esc(title)}</b><br><span class="faint xs" title="${esc(details)}">${esc(details)}</span></span>
    <span class="t sm-hide">${r.shot ? `<img class="cm-shot" loading="lazy" src="/api/runs/${encodeURIComponent(r.id)}/artifacts/screenshot.jpg" alt="">` : ""}</span>
    <span class="c sm-hide" title="${esc(r.model || 'custom endpoint')}">${esc(short(r.model))}</span><span class="c">${money(r.cost?.total)}</span>
    <span class="w2">${ago(r.created_at)}</span>${icon("chevronRight", 14)}</a>`;
}

function modelResults(results) {
  return `<div class="cm-tablewrap"><table class="mtable cm-lb"><thead><tr><th>Model</th><th>Environment</th><th>Runner</th><th>Metric</th><th class="r">Runs</th><th class="r">Mean reward</th></tr></thead>
    <tbody>${results.map((r) => `<tr><td><button class="link cm-clip" data-pick-model="${esc(r.model)}" title="${esc(r.model)}">${esc(short(r.model))}</button><small>${esc(r.served)}</small></td>
      <td><button class="link cm-clip" data-pick-dataset="${esc(r.dataset)}" title="${esc(r.dataset)}">${esc(r.dataset)}</button>
      <small>${esc([DOMAIN_NAME[r.domain] || r.domain, r.revision ? `revision ${r.revision.slice(0, 8)}` : ""].filter(Boolean).join(" · "))}</small></td>
      <td>${esc(runnerName(r.runner))}</td><td><span class="cm-clip" title="${esc(r.metric)}">${esc(r.metric)}</span>${r.judge ? `<small title="${esc(r.judge)}">judge ${esc(short(r.judge))}</small>` : ""}</td>
      <td class="r">${fmt.format(r.runs)}</td><td class="r" title="Observed range ${esc(num(r.min))} to ${esc(num(r.max))}">${num(r.mean)}</td></tr>`).join("")}</tbody></table></div>`;
}

export async function mount(el, { qs }) {
  unmount();
  const page = current = { controller: null, events: new AbortController(), timer: null, sequence: 0, pending: false };
  const p = qs || new URLSearchParams();
  const f = Object.fromEntries(KEYS.map((k) => [k, p.get(k) || ""]));
  f.dataset ||= p.get("d") || "";
  f.group ||= DOMAIN_NAME[p.get("domain")] || "";
  f.sort = SORT.some(([k]) => k === p.get("sort")) ? p.get("sort") : "new";
  // Legacy Tasks URLs become rollout filters; no implicit MiMo dataset selection.
  const sync = () => {
    const q = new URLSearchParams(Object.entries(f).filter(([k, v]) => v && !(k === "sort" && v === "new")));
    history.replaceState(null, "", "/community" + (q.size ? `?${q}` : ""));
  };
  el.innerHTML = `<div class="wrap page fade-in">
    <div class="page-h"><div><h1>Community</h1><p>Public, graded rollouts across Harbor, MiMo, NeMo Gym and other supported runners. Explore their traces and compare results within an environment. Your public and private runs are in <a class="u" href="/runs">My rollouts</a>.</p></div></div>
    <div class="cm-figs" id="cm-figs">${sk.line(60, 22)}</div>
    <div class="cm-filters" id="cm-filters"></div><div class="cm-active" id="cm-active"></div>
    <p class="muted xs">Reward scales differ by environment. Values below use each rollout’s recorded scoring metric.</p>
    <section class="panel" aria-label="Public rollouts"><div class="panel-h"><h2>Rollouts</h2><span class="aside" id="cm-count"></span></div>
      <div class="panel-b"><div id="cm-error" role="status"></div><div class="runlist cm-list" id="cm-rows">${sk.lines(95, 90, 92, 88)}</div>
      <div class="more-row" id="cm-more" hidden><span class="muted sm"></span><button class="btn" type="button">Load more</button></div></div></section>
    <details class="panel cm-board" id="cm-result-panel" hidden><summary class="panel-h"><h2>Model results</h2><span class="aside">Grouped by environment, runner and scoring metric</span></summary>
      <div class="panel-b" id="cm-board"></div><p class="fine">Observed results can use different tasks, revisions and judges; they are not a controlled model ranking.</p></details>
    <p class="fine" style="margin-top:18px">Public rollouts hide account identities and may later be released as an open dataset for research.</p></div>`;
  let offset = 0;
  const seen = new Set();
  const option = (key, label, entries = [], labelOf = (v) => v) => {
    const list = entries.slice();
    if (f[key] && !list.some(([v]) => v === f[key])) list.push([f[key], 0]);
    return `<select class="input" data-f="${key}" aria-label="${label}"><option value="">${label}</option>${list.map(([v, n]) =>
      `<option value="${esc(v)}" ${f[key] === v ? "selected" : ""}>${esc(labelOf(v))} (${fmt.format(n)})</option>`).join("")}</select>`;
  };
  const renderHead = (d) => {
    const s = d.stats, fc = d.facets;
    const fig = (v, label) => `<div><b>${fmt.format(v)}</b><span>${label}</span></div>`;
    $("#cm-figs", el).innerHTML = fig(d.total, "public rollouts") + fig(s.datasets, "environments") + fig(s.runners, "runners") + fig(s.models, "models");
    $("#cm-result-panel", el).hidden = !s.results.length;
    $("#cm-board", el).innerHTML = s.results.length ? modelResults(s.results) : "";
    const focus = document.activeElement, focusedQuery = focus?.id === "cm-q", focusedFilter = focus?.dataset?.f;
    const selection = focusedQuery ? [focus.selectionStart, focus.selectionEnd] : null;
    const moreOpen = $(".cm-more", el)?.open;
    $("#cm-filters", el).innerHTML = `<div class="search">${icon("search", 15)}<input id="cm-q" type="search" aria-label="Search rollouts" placeholder="Search rollouts, environments or models" value="${esc(f.q)}"></div>
      ${option("runner", "Any runner", fc.runner, runnerName)}${option("dataset", "Any environment", fc.dataset)}${option("model", "Any model", fc.model, short)}
      <select class="input" data-f="reward" aria-label="Reward">${REWARD.map(([v, l]) => `<option value="${v}" ${f.reward === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      <details class="cm-more" ${moreOpen ? 'open' : ''}><summary class="btn">${icon("filter", 14)}More</summary><div class="cm-pop">
      ${option("group", "Any group", fc.group)}${option("served", "Any provider", fc.served)}${option("judge", "Any judge", fc.judge, short)}${option("thinking", "Any thinking", fc.thinking, v => THINK[v] || v)}</div></details>
      <select class="input cm-sort" data-f="sort" aria-label="Sort">${SORT.map(([v, l]) => `<option value="${v}" ${f.sort === v ? 'selected' : ''}>${l}</option>`).join("")}</select>`;
    if (focusedQuery) { const input = $("#cm-q", el); input.focus({preventScroll:true}); input.setSelectionRange(...selection); }
    else if (focusedFilter) $$('[data-f]', el).find(x => x.dataset.f === focusedFilter)?.focus({preventScroll:true});
    const active = KEYS.filter(k => f[k]);
    $("#cm-active", el).innerHTML = active.length ? '<span class="faint xs">Filtered by</span>' + active.map(k => `<button type="button" class="cm-pill" data-unset="${k}" title="${esc(LABEL[k](f[k]))}"><span>${esc(LABEL[k](f[k]))}</span>${icon("x", 11)}</button>`).join("") + '<button class="link xs" id="cm-clear" type="button">Clear filters</button>' : "";
  };
  const load = async (append = false) => {
    if (append && page.pending) return;
    page.controller?.abort();
    const controller = page.controller = new AbortController(), sequence = ++page.sequence;
    page.pending = true;
    $("#cm-more button", el).disabled = true;
    $("#cm-rows", el).setAttribute("aria-busy", "true");
    $("#cm-error", el).innerHTML = "";
    const q = new URLSearchParams({limit:"40",offset:String(append ? offset : 0)});
    for (const [key, value] of Object.entries(f)) if (value) q.set(key, value);
    try {
      const d = await api(`/api/community?${q}`, {signal:controller.signal});
      if (current !== page || sequence !== page.sequence) return;
      if (!append) { renderHead(d); seen.clear(); offset = 0; $("#cm-rows", el).innerHTML = ""; }
      const next = d.runs.filter(r => { if (seen.has(r.id)) return false; seen.add(r.id); return true; });
      $("#cm-rows", el).insertAdjacentHTML("beforeend", next.map(row).join(""));
      offset += d.runs.length;
      $("#cm-count", el).textContent = d.total ? `${fmt.format(seen.size)} of ${fmt.format(d.total)}` : "";
      $("#cm-more", el).hidden = offset >= d.total || !d.runs.length;
      $("#cm-more span", el).textContent = `Showing ${fmt.format(seen.size)} of ${fmt.format(d.total)}`;
      if (!d.total) $("#cm-rows", el).innerHTML = emptyState("globe", KEYS.some(k => f[k]) ? "No rollouts match" : "No public rollouts yet",
        KEYS.some(k => f[k]) ? "Clear a filter to see more public rollouts." : "Completed, graded runs appear here when shared publicly. Private runs remain in My rollouts.",
        '<a class="btn" href="/runs">My rollouts</a><a class="btn" href="/">Explore environments</a>');
    } catch (e) {
      if (current === page && sequence === page.sequence && e.name !== "AbortError") {
        if (!append) $("#cm-rows", el).innerHTML = "";
        $("#cm-error", el).innerHTML = `<p class="err-text">Couldn't load rollouts: ${esc(e.message)}</p><button class="btn" data-retry="${append ? 'more' : 'all'}">Try again</button>`;
      }
    } finally {
      if (current === page && sequence === page.sequence) { page.pending = false; $("#cm-more button", el).disabled = false; $("#cm-rows", el).setAttribute("aria-busy", "false"); }
    }
  };
  const reload = () => { clearTimeout(page.timer); sync(); load(); };
  el.addEventListener("error", e => { if (e.target.classList?.contains("cm-shot")) e.target.remove(); }, {capture:true, signal:page.events.signal});
  el.addEventListener("click", e => {
    const unset = e.target.closest("[data-unset]"), model = e.target.closest("[data-pick-model]"), dataset = e.target.closest("[data-pick-dataset]"), retry = e.target.closest("[data-retry]");
    if (unset) { f[unset.dataset.unset] = ""; reload(); }
    else if (model) { f.model = model.dataset.pickModel; reload(); }
    else if (dataset) { f.dataset = dataset.dataset.pickDataset; reload(); }
    else if (retry) load(retry.dataset.retry === 'more');
    else if (e.target.closest("#cm-clear")) { for (const key of KEYS) f[key] = ""; reload(); }
    else if (e.target.closest("#cm-more button")) load(true);
  }, {signal:page.events.signal});
  // Tie every listener to this mount, even when the router reuses the main node.
  const change = e => { const select = e.target.closest("[data-f]"); if (select) { f[select.dataset.f] = select.value; reload(); } };
  const input = e => { if (e.target.id !== "cm-q") return; f.q = e.target.value.trim(); page.controller?.abort(); ++page.sequence; clearTimeout(page.timer); page.timer = setTimeout(reload, 250); };
  el.addEventListener("change", change, {signal:page.events.signal});
  el.addEventListener("input", input, {signal:page.events.signal});
  sync(); load();
}
