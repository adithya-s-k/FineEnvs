// Explore: domain cards, the treemap, search, facets and the task list.
import { $, esc, fmt, vals, getJSONgz, sk, progress, emptyState } from "./util.js";
import { icon, DOMAIN_ICON } from "./icons.js";

const CJK = /[㐀-鿿豈-﫿]/;
const PAGE = 30;
const AUTO_PAGES = 3;   // pages that load by themselves as you scroll; after that a button, so the footer stays reachable
const MAP_TOP = { code: 7, webdev: 9, cyber: 7, music: 9, general: 9 };
const LEFTOVER = /^(Other|Unknown|unknown|None named|Unspecified|Unrated)$/;
const num = (v) => { const m = String(v).match(/\d+/); return m ? +m[0] : 99; };
const ORDINAL = { tier: num, voices: num, tempo: (v) => (/^Slow/.test(v) ? 0 : /^Moderate/.test(v) ? 1 : /^Fast/.test(v) ? 2 : 3), meter: (v) => ["2/4", "3/4", "4/4", "6/8", "7/8"].indexOf(v) };

let DATA, DOMS, ENVS, SEARCH;
const state = { q: "", dom: null, sel: {}, seed: Math.random() };
let matches = [], shown = 0, searchIds = null, root = null, observer = null, autoLoaded = 0;
const expanded = new Set();

export async function load() {
  if (DATA) return DATA;
  DATA = await getJSONgz("/data/index.json.gz");
  DOMS = Object.fromEntries(DATA.domains.map((d) => [d.id, d]));
  ENVS = DATA.envs;
  ENVS.forEach((e) => { e._blob = (e.t + " " + e.s).toLowerCase(); e._rand = hash(e.id); });
  SEARCH = new MiniSearch({
    fields: ["t", "s", "x", "id"], storeFields: [],
    extractField: (doc, f) => (f === "x" ? Object.values(doc.f).flat().join(" ") + " " + DOMS[doc.d].name : doc[f]),
    searchOptions: { boost: { t: 3, x: 2 }, prefix: true, fuzzy: 0.15, combineWith: "AND" },
  });
  SEARCH.addAll(ENVS);
  return DATA;
}
export const record = (id) => ENVS && ENVS.find((e) => e.id === id);
export const domains = () => DOMS;

// Shuffle per page load, so people who arrive together don't all open (and run) the same first task.
function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return h >>> 0; }
const order = (e) => ((e._rand ^ Math.floor(state.seed * 4294967295)) >>> 0);

function skeleton() {
  return `<div class="wrap">
    <div class="ex-head"><div style="flex:1">${sk.line(28, 24)}<div style="height:10px"></div>${sk.line(52)}</div></div>
    <div class="domains">${Array.from({ length: 6 }, () => sk.card(sk.line(50) + sk.line(35, 20) + sk.line(70))).join("")}</div>
    <div class="panel" style="padding:16px;margin-bottom:18px">${sk.box(220)}</div>
    <div class="layout"><div>${sk.lines(60, 90, 80, 70, 85, 60)}</div>
      <div class="list">${Array.from({ length: 6 }, () => `<li>${sk.card(sk.line(30) + sk.line(90, 16) + sk.lines(100, 70))}</li>`).join("")}</div></div></div>`;
}

export async function mount(el, params) {
  root = el;
  if (!DATA) {
    el.innerHTML = skeleton();
    try { await progress.wrap(load()); }
    catch (e) { el.innerHTML = `<div class="wrap page">${emptyState("alert", "Couldn't load the environments", esc(e.message), `<button class="btn" onclick="location.reload()">${icon("refresh")}Try again</button>`)}</div>`; return; }
    if (root !== el) return;
  }
  if (params) {
    state.q = params.get("q") || "";
    state.dom = DOMS[params.get("d")] ? params.get("d") : null;
    state.sel = {};
    (params.get("f") || "").split(";").filter(Boolean).forEach((pair) => {
      const [k, v] = pair.split(":");
      if (k && v) state.sel[k] = new Set(v.split("|").map(decodeURIComponent));
    });
  }
  el.innerHTML = `
    <div class="wrap fade-in">
      <section class="ex-head">
        <div><h1>RL environments</h1>
          <p>Every task in Xiaomi's MiMo-V2.6 RL release. Open one to see exactly what the agent gets and how it is graded, then run a rollout with a model of your choice.</p></div>
      </section>
      <div class="domains" id="domains" role="tablist" aria-label="Domains"></div>
      <details class="panel map-card" id="map-card" ${matchMedia("(max-width: 760px)").matches ? "" : "open"}>
        <summary>${icon("chevronRight", 15, "chev")}<h2 id="map-title"></h2><span class="hint">Area is the number of environments. Click a block to filter.</span></summary>
        <div id="map" class="map"></div>
      </details>
      <div class="toolbar" id="toolbar">
        <div class="search">${icon("search", 16)}
          <input id="q" type="search" placeholder="Search titles, briefs, categories…" autocomplete="off" spellcheck="false" aria-label="Search environments"><kbd>/</kbd>
        </div>
        <button class="btn" id="surprise" type="button" title="Open a random environment from the current results">${icon("shuffle", 15)}<span class="long">Random</span></button>
        <button class="btn only-mobile" id="open-filters" type="button">${icon("filter", 15)}Filters<span id="nfilt"></span></button>
      </div>
      <div class="layout">
        <aside class="filters" id="filters" aria-label="Filters">
          <div class="sheet-head"><b>Filters</b><button class="btn sm" id="close-filters" type="button">Done</button></div>
          <div class="filters-head"><span>Filters</span><button class="link" id="clear" type="button">Clear all</button></div>
          <div id="facets"></div>
        </aside>
        <section class="results" aria-live="polite">
          <div class="results-head"><span class="count" id="count"></span><span class="order" id="order-note"></span><div class="active" id="active"></div></div>
          <ol class="list" id="list"></ol>
          <div class="sentinel" id="sentinel"></div>
          <div class="more-row" id="more-row" hidden><span class="muted sm" id="more-count"></span>
            <button class="btn" id="more" type="button">Load more</button></div>
          <div id="empty" hidden>${emptyState("search", "No environments match", "Try fewer filters or a different word.", `<button class="btn" id="clear2" type="button">Clear search and filters</button>`)}</div>
        </section>
      </div>
    </div>`;
  $("#q", el).value = state.q;
  wire(el);
  run();
}

export function unmount() { if (observer) observer.disconnect(); observer = null; }

function writeURL() {
  const p = new URLSearchParams();
  if (state.q) p.set("q", state.q);
  if (state.dom) p.set("d", state.dom);
  const f = Object.entries(state.sel).filter(([, s]) => s.size).map(([k, s]) => k + ":" + [...s].map(encodeURIComponent).join("|")).join(";");
  if (f) p.set("f", f);
  const qs = p.toString();
  history.replaceState(null, "", "#/" + (qs ? "?" + qs : ""));
}

// ── filtering ────────────────────────────────────────────────────────────────
const facetsFor = (dom) => (dom ? DOMS[dom].facets : [["d", "Domain"], ["language", "Brief language"]]);
const valueOf = (e, key) => (key === "d" ? DOMS[e.d].name : e.f[key]);
function passes(e, skipKey) {
  if (state.dom && e.d !== state.dom) return false;
  if (searchIds && !searchIds.has(e.id)) return false;
  for (const [k, set] of Object.entries(state.sel)) {
    if (k === skipKey || !set.size) continue;
    if (!vals(valueOf(e, k)).some((v) => set.has(v))) return false;
  }
  return true;
}
function computeSearch() {
  const q = state.q.trim();
  if (!q) { searchIds = null; return null; }
  if (CJK.test(q)) {
    const needle = q.toLowerCase();
    const hit = ENVS.filter((e) => e._blob.includes(needle));
    searchIds = new Set(hit.map((e) => e.id));
    return new Map(hit.map((e, i) => [e.id, -i]));
  }
  const res = SEARCH.search(q);
  searchIds = new Set(res.map((r) => r.id));
  return new Map(res.map((r) => [r.id, r.score]));
}
function run() {
  const scores = computeSearch();
  const allowed = new Set(facetsFor(state.dom).map(([k]) => k));
  Object.keys(state.sel).forEach((k) => { if (!allowed.has(k)) delete state.sel[k]; });
  matches = ENVS.filter((e) => passes(e));
  if (scores) matches.sort((a, b) => (scores.get(b.id) ?? 0) - (scores.get(a.id) ?? 0));
  else matches.sort((a, b) => order(a) - order(b));
  $("#order-note", root).innerHTML = scores ? "best match first" : `${icon("shuffle", 13)}shuffled each visit`;
  const nf = Object.values(state.sel).reduce((n, s) => n + s.size, 0);
  $("#nfilt", root).textContent = nf ? ` (${nf})` : "";
  renderDomains(); renderFacets(); renderActive(); renderMap(); renderList(true); writeURL();
}

// ── domain cards ─────────────────────────────────────────────────────────────
function renderDomains() {
  const inSearch = (d) => (searchIds ? ENVS.filter((e) => (!d || e.d === d) && searchIds.has(e.id)).length : null);
  const cards = [{ id: null, name: "All domains", verifier: "5 kinds of grader", count: DATA.total }, ...DATA.domains];
  $("#domains", root).innerHTML = cards.map((d) => {
    const hit = inSearch(d.id);
    return `<button class="dom" role="tab" data-dom="${d.id ?? ""}" aria-selected="${state.dom === d.id}" style="--dc:${d.id ? `var(--c-${d.id})` : "var(--text)"}" title="${esc(d.task || "")}">
      <span class="name">${icon(d.id ? DOMAIN_ICON[d.id] : "grid", 14)}${esc(d.name)}</span>
      <span class="n">${fmt.format(hit ?? d.count)}${hit != null ? `<span class="of"> / ${fmt.format(d.count)}</span>` : ""}</span>
      <span class="ver">${d.id ? `Graded by ${esc(d.verifier)}` : esc(d.verifier)}</span></button>`;
  }).join("");
}

// ── facets ───────────────────────────────────────────────────────────────────
function renderFacets() {
  $("#facets", root).innerHTML = facetsFor(state.dom).map(([key, label]) => {
    const counts = new Map();
    ENVS.forEach((e) => { if (passes(e, key)) vals(valueOf(e, key)).forEach((v) => counts.set(v, (counts.get(v) || 0) + 1)); });
    const sel = state.sel[key] || new Set();
    sel.forEach((v) => { if (!counts.has(v)) counts.set(v, 0); });
    if (counts.size <= 1 && !sel.size && key !== "d") return "";
    // ordinal facets keep their natural order (Tier 1 → 5, 1 voice → 6, slow → fast); the rest go by count
    const ordinal = ORDINAL[key];
    let items = [...counts.entries()].sort(ordinal ? (a, b) => ordinal(a[0]) - ordinal(b[0])
      : (a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
    const limit = expanded.has(key) ? Infinity : 8;
    const hidden = Math.max(0, items.length - limit);
    items = items.slice(0, limit);
    return `<div class="facet"><h3>${esc(label)}</h3>
      ${items.map(([v, c]) => `<button class="opt ${c ? "" : "zero"}" data-k="${esc(key)}" data-v="${esc(v)}" aria-pressed="${sel.has(v)}">
        <span class="box">${sel.has(v) ? icon("check", 11) : ""}</span><span class="lab" title="${esc(v)}">${esc(v)}</span><span class="c">${fmt.format(c)}</span></button>`).join("")}
      ${hidden || expanded.has(key) ? `<button class="link more" data-more="${esc(key)}">${icon(expanded.has(key) ? "chevronDown" : "chevronRight", 13)}${expanded.has(key) ? "Show fewer" : `Show ${hidden} more`}</button>` : ""}</div>`;
  }).join("");
}
function toggle(key, v) {
  if (key === "d") { const id = DATA.domains.find((d) => d.name === v)?.id; if (id) { setDomain(id); return; } }
  const set = (state.sel[key] ||= new Set());
  set.has(v) ? set.delete(v) : set.add(v);
  if (!set.size) delete state.sel[key];
  run();
}
function setDomain(id) { state.dom = id; state.sel = {}; run(); }
function renderActive() {
  const labels = Object.fromEntries(facetsFor(state.dom));
  const pills = [];
  Object.entries(state.sel).forEach(([k, set]) => set.forEach((v) =>
    pills.push(`<button class="pill" data-k="${esc(k)}" data-v="${esc(v)}"><span>${esc(labels[k] || k)}</span>${esc(v)}${icon("x", 13)}</button>`)));
  $("#active", root).innerHTML = pills.join("");
  $("#count", root).textContent = `${fmt.format(matches.length)} environment${matches.length === 1 ? "" : "s"}`;
}

// ── map ──────────────────────────────────────────────────────────────────────
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function mapData() {
  if (!state.dom) {
    const children = DATA.domains.map((d) => {
      const counts = new Map();
      ENVS.forEach((e) => { if (e.d === d.id && passes(e)) vals(e.f[d.main]).forEach((v) => counts.set(v, (counts.get(v) || 0) + 1)); });
      const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
      const top = sorted.slice(0, MAP_TOP[d.id]);
      const rest = sorted.slice(MAP_TOP[d.id]).reduce((s, [, c]) => s + c, 0);
      const kids = top.map(([name, value], rank) => ({ name, value, rank, dom: d.id, other: LEFTOVER.test(name) }));
      if (rest) kids.push({ name: `${sorted.length - top.length} more`, value: rest, rank: top.length, dom: d.id, other: true });
      return { name: d.name, dom: d.id, children: kids };
    }).filter((d) => d.children.length);
    return { name: "all", children };
  }
  const d = DOMS[state.dom];
  const counts = new Map();
  ENVS.forEach((e) => { if (passes(e, d.main)) vals(e.f[d.main]).forEach((v) => counts.set(v, (counts.get(v) || 0) + 1)); });
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  return { name: d.name, children: [{ name: d.name, dom: d.id, children: sorted.map(([name, value], rank) => ({ name, value, rank, dom: d.id, other: LEFTOVER.test(name) })) }] };
}
export function renderMap() {
  const el = root && $("#map", root);
  if (!el) return;
  const W = el.clientWidth, H = el.clientHeight;
  $("#map-title", root).textContent = state.dom
    ? `${DOMS[state.dom].name}, by ${DOMS[state.dom].facets.find(([k]) => k === DOMS[state.dom].main)[1].replace(" (mentioned)", "").toLowerCase()}`
    : "Everything, by domain and category";
  if (!W || !H) return;   // the overview is collapsed
  const data = mapData();
  el.innerHTML = "";
  if (!data.children.length) { el.innerHTML = `<p class="empty">Nothing to map.</p>`; return; }
  const h = d3.hierarchy(data).sum((d) => d.value || 0).sort((a, b) => (!!a.data.other - !!b.data.other) || b.value - a.value);
  d3.treemap().size([W, H]).paddingInner(2).paddingTop((n) => (n.depth === 1 && !state.dom ? 20 : 0)).round(true).tile(d3.treemapSquarify.ratio(1.35))(h);
  const surface = cssVar("--surface");
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img").attr("aria-label", "Treemap of environments by category");
  if (!state.dom) {
    svg.selectAll("g.dh").data(h.children).join("g").attr("class", "cell dh").attr("transform", (d) => `translate(${d.x0},${d.y0})`).call((g) => {
      g.append("rect").attr("width", (d) => d.x1 - d.x0).attr("height", 18).attr("fill", "transparent").on("click", (_, d) => setDomain(d.data.dom));
      g.append("text").attr("class", "dname").attr("x", 1).attr("y", 13).attr("fill", (d) => cssVar(`--c-${d.data.dom}`))
        .text((d) => (d.x1 - d.x0 > 90 ? `${d.data.name} · ${fmt.format(d.value)}` : d.x1 - d.x0 > 34 ? d.data.name : ""));
    });
  }
  const leaves = h.leaves();
  const maxRank = d3.max(leaves, (d) => d.data.rank) || 1;
  const sel = state.dom ? state.sel[DOMS[state.dom].main] : null;
  const g = svg.selectAll("g.leaf").data(leaves).join("g").attr("class", (d) => "cell leaf" + (sel && sel.size && !sel.has(d.data.name) ? " dim" : ""))
    .attr("transform", (d) => `translate(${d.x0},${d.y0})`);
  g.append("rect").attr("width", (d) => Math.max(0, d.x1 - d.x0)).attr("height", (d) => Math.max(0, d.y1 - d.y0)).attr("rx", 4)
    .attr("fill", (d) => d3.interpolateRgb(surface, cssVar(`--c-${d.data.dom}`))(d.data.other ? 0.42 : 0.95 - 0.38 * (d.data.rank / Math.max(maxRank, 6))))
    .on("click", (_, d) => clickCell(d.data))
    .on("mousemove", (ev, d) => tip(ev, `<b>${esc(d.data.name)}</b><br>${fmt.format(d.value)} environments · ${DOMS[d.data.dom].name}`))
    .on("mouseleave", () => tip());
  g.each(function (d) {
    const w = d.x1 - d.x0, hh = d.y1 - d.y0;
    if (w < 44 || hh < 26) return;
    const color = d3.lab(this.querySelector("rect").getAttribute("fill")).l > 64 ? "#17181c" : "#ffffff";
    const max = Math.floor((w - 14) / 6.6);
    const label = d.data.name.length > max ? d.data.name.slice(0, Math.max(1, max - 1)) + "…" : d.data.name;
    const t = d3.select(this);
    t.append("text").attr("class", "lbl").attr("x", 8).attr("y", 18).attr("fill", color).text(label);
    if (hh > 42) t.append("text").attr("class", "cnt").attr("x", 8).attr("y", 34).attr("fill", color).text(fmt.format(d.value));
  });
  animateMap(svg, g, h, W, H);
}

// Drill-down motion: blocks that stay glide from where they were; new blocks grow out of the region they came
// from (the domain you clicked), or out of their own centre. Layout and clicks are unchanged; this is only motion.
let lastMap = null;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const leafKey = (d) => d.data.dom + "|" + d.data.name;
function animateMap(svg, g, h, W, H) {
  const now = { W, H, leaves: new Map(), doms: new Map() };
  g.each((d) => now.leaves.set(leafKey(d), [d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0]));
  (h.children || []).forEach((c) => now.doms.set(c.data.dom, [c.x0, c.y0, c.x1 - c.x0, c.y1 - c.y0]));
  const prev = lastMap;
  lastMap = now;
  if (!prev || prev.W !== W || prev.H !== H || reduceMotion.matches) return;   // first paint or a resize: no motion
  const T = d3.transition().duration(560).ease(d3.easeCubicInOut);
  g.each(function (d) {
    const to = [d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0];
    const from = prev.leaves.get(leafKey(d)) || prev.doms.get(d.data.dom) || [to[0] + to[2] / 2, to[1] + to[3] / 2, 0, 0];
    if (from.every((v, i) => Math.abs(v - to[i]) < 0.5)) return;
    const sel = d3.select(this);
    sel.attr("transform", `translate(${from[0]},${from[1]})`).transition(T).attr("transform", `translate(${to[0]},${to[1]})`);
    sel.select("rect").attr("width", from[2]).attr("height", from[3]).transition(T).attr("width", to[2]).attr("height", to[3]);
    sel.selectAll("text").attr("opacity", 0).transition(T).delay(260).duration(260).attr("opacity", 1);
  });
  svg.selectAll("g.dh").attr("opacity", 0).transition(T).delay(200).attr("opacity", 1);
}
function clickCell(d) {
  tip();
  const dom = DOMS[d.dom];
  if (state.dom !== d.dom) { state.dom = d.dom; state.sel = {}; }
  if (!d.other) {
    const set = (state.sel[dom.main] ||= new Set());
    set.has(d.name) ? set.delete(d.name) : set.add(d.name);
    if (!set.size) delete state.sel[dom.main];
  }
  run();
  // let the drill-down play, then bring the matching environments into view
  setTimeout(() => $("#toolbar", root)?.scrollIntoView({ behavior: "smooth", block: "start" }), reduceMotion.matches ? 0 : 620);
}
function tip(ev, html) {
  const el = $("#tip");
  if (!ev) { el.hidden = true; return; }
  el.innerHTML = html; el.hidden = false;
  el.style.left = Math.min(ev.clientX + 14, innerWidth - el.offsetWidth - 8) + "px";
  el.style.top = ev.clientY + 16 + "px";
}

// ── list ─────────────────────────────────────────────────────────────────────
function terms() {
  if (!state.q) return [];
  if (CJK.test(state.q)) return [state.q.trim()];
  return state.q.toLowerCase().split(/[^\p{L}\p{N}_+#.-]+/u).filter((t) => t.length > 1);
}
function highlight(text) {
  const html = esc(text), ts = terms();
  if (!ts.length) return html;
  const re = new RegExp("(" + ts.map((t) => esc(t).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")", "gi");
  return html.replace(re, "<mark>$1</mark>");
}
function card(e) {
  const d = DOMS[e.d];
  const chips = d.facets.filter(([k]) => k !== d.main && k !== "source")
    .flatMap(([k]) => vals(e.f[k]).filter((v) => !LEFTOVER.test(v)).slice(0, 2)).slice(0, 4);
  const stats = e.n ? [[`${e.n.systems}`, "systems"], [`${e.n.files}`, "files"], [`${e.n.checks}`, "checks"]] : [];
  return `<li><a class="card" href="#/task/${encodeURIComponent(e.id)}" style="--dc:var(--c-${e.d})">
    <div class="row1"><span class="dn">${icon(DOMAIN_ICON[e.d], 13)}${esc(d.name)}</span><span class="faint">/</span><span class="cat">${esc(vals(e.f[d.main]).join(", "))}</span></div>
    <p class="t">${highlight(e.t)}</p><p class="s">${highlight(e.s)}</p>
    ${chips.length || stats.length ? `<div class="chips">${chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}${stats.map(([n, l]) => `<span class="chip outline"><b>${n}</b> ${l}</span>`).join("")}</div>` : ""}
  </a></li>`;
}
function renderList(reset) {
  const list = $("#list", root);
  if (reset) { list.innerHTML = ""; shown = 0; autoLoaded = 0; }
  const next = matches.slice(shown, shown + PAGE);
  list.insertAdjacentHTML("beforeend", next.map(card).join(""));
  shown += next.length;
  $("#empty", root).hidden = matches.length > 0;
  $("#list", root).hidden = !matches.length;
  const left = matches.length - shown;
  $("#more-row", root).hidden = left <= 0 || autoLoaded < AUTO_PAGES;
  $("#more-count", root).textContent = `Showing ${fmt.format(shown)} of ${fmt.format(matches.length)}`;
  $("#more", root).textContent = `Load ${fmt.format(Math.min(PAGE, left))} more`;
}

function wire(el) {
  let t;
  $("#q", el).addEventListener("input", (ev) => { clearTimeout(t); t = setTimeout(() => { state.q = ev.target.value; run(); }, 120); });
  $("#domains", el).addEventListener("click", (ev) => { const b = ev.target.closest(".dom"); if (b) setDomain(b.dataset.dom || null); });
  $("#facets", el).addEventListener("click", (ev) => {
    const o = ev.target.closest(".opt");
    if (o) return toggle(o.dataset.k, o.dataset.v);
    const m = ev.target.closest("[data-more]");
    if (m) { expanded.has(m.dataset.more) ? expanded.delete(m.dataset.more) : expanded.add(m.dataset.more); renderFacets(); }
  });
  $("#active", el).addEventListener("click", (ev) => { const p = ev.target.closest(".pill"); if (p) toggle(p.dataset.k, p.dataset.v); });
  const clearAll = () => { state.sel = {}; state.q = ""; $("#q", el).value = ""; state.dom = null; run(); };
  $("#clear", el).addEventListener("click", clearAll);
  $("#clear2", el).addEventListener("click", clearAll);
  $("#close-filters", el).addEventListener("click", () => $("#filters", el).classList.remove("open"));
  $("#map-card", el).addEventListener("toggle", () => renderMap());
  $("#surprise", el).addEventListener("click", () => {
    const pool = matches.length ? matches : ENVS;
    location.hash = "#/task/" + encodeURIComponent(pool[Math.floor(Math.random() * pool.length)].id);
  });
  $("#open-filters", el).addEventListener("click", () => $("#filters", el).classList.toggle("open"));
  observer = new IntersectionObserver((es) => {
    if (es.some((x) => x.isIntersecting) && shown < matches.length && autoLoaded < AUTO_PAGES) { autoLoaded++; renderList(false); }
  }, { rootMargin: "600px" });
  $("#more", el).addEventListener("click", () => renderList(false));
  observer.observe($("#sentinel", el));
}

export function focusSearch() { const q = root && $("#q", root); if (q) q.focus(); }
