// Explore: domain cards, the treemap, search, facets and the task list.
import { $, esc, fmt, vals, getJSONgz, storage } from "./util.js";

const CJK = /[㐀-鿿豈-﫿]/;
const PAGE = 40;
const MAP_TOP = { code: 7, webdev: 9, cyber: 7, music: 9, general: 9 };
const LEFTOVER = /^(Other|Unknown|unknown|None named|Unspecified|Unrated)$/;

let DATA, DOMS, ENVS, SEARCH;
const state = { q: "", dom: null, sel: {}, seed: Math.random() };
let matches = [], shown = 0, searchIds = null, root = null, observer = null;
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

export async function mount(el, params) {
  root = el;
  await load();
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
    <div class="wrap">
      <section class="intro">
        <p class="lede"><strong>${fmt.format(DATA.total)}</strong> environments an agent can be trained in, across <strong>5 domains</strong>,
          each graded a different way. Browse them, open one to see everything inside it, and run a rollout with the model of your choice.</p>
        <div class="domains" id="domains" role="tablist" aria-label="Domains"></div>
      </section>
      <section class="map-card" aria-label="Map of environments by category">
        <div class="map-head"><h2 id="map-title"></h2><span class="hint">Block area = number of environments · click to filter</span></div>
        <div id="map" class="map"></div>
      </section>
      <div class="toolbar" id="toolbar">
        <div class="search">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M10 4a6 6 0 1 0 3.87 10.59l4.27 4.27 1.41-1.41-4.27-4.27A6 6 0 0 0 10 4zm0 2a4 4 0 1 1 0 8 4 4 0 0 1 0-8z"/></svg>
          <input id="q" type="search" placeholder="Search environments…  (press / )" autocomplete="off" spellcheck="false" aria-label="Search environments">
        </div>
        <button class="btn" id="surprise" type="button" title="Open a random task">🎲 Surprise me</button>
        <button class="btn ghost only-mobile" id="open-filters" type="button">Filters</button>
      </div>
      <div class="layout">
        <aside class="filters" id="filters" aria-label="Filters">
          <div class="filters-head"><span>Filters</span><button class="link" id="clear" type="button">Clear all</button></div>
          <div id="facets"></div>
        </aside>
        <section class="results" aria-live="polite">
          <div class="results-head"><span id="count"></span><span class="muted sm" id="order-note"></span><div class="active" id="active"></div></div>
          <ol class="list" id="list"></ol>
          <div class="sentinel" id="sentinel"></div>
          <p class="empty" id="empty" hidden>No environments match. Try fewer filters or a different word.</p>
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
  $("#order-note", root).textContent = scores ? "best match first" : "shuffled each visit";
  renderDomains(); renderFacets(); renderActive(); renderMap(); renderList(true); writeURL();
}

// ── domain cards ─────────────────────────────────────────────────────────────
function renderDomains() {
  const inSearch = (d) => (searchIds ? ENVS.filter((e) => (!d || e.d === d) && searchIds.has(e.id)).length : null);
  const cards = [{ id: null, name: "All domains", task: "Every environment in the release", verifier: "5 kinds", count: DATA.total }, ...DATA.domains];
  $("#domains", root).innerHTML = cards.map((d) => {
    const hit = inSearch(d.id);
    return `<button class="dom" role="tab" data-dom="${d.id ?? ""}" aria-selected="${state.dom === d.id}" style="--dc:${d.id ? `var(--c-${d.id})` : "var(--accent)"}">
      <div class="name"><span class="dot"></span>${esc(d.name)}</div>
      <div class="n">${fmt.format(hit ?? d.count)}${hit != null ? `<span class="of"> / ${fmt.format(d.count)}</span>` : ""}</div>
      <div class="task">${esc(d.task)}</div><div class="ver">Graded by <b>${esc(d.verifier)}</b></div></button>`;
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
    let items = [...counts.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
    const limit = expanded.has(key) ? Infinity : 8;
    const hidden = Math.max(0, items.length - limit);
    items = items.slice(0, limit);
    return `<div class="facet"><h3>${esc(label)}</h3>
      ${items.map(([v, c]) => `<button class="opt ${c ? "" : "zero"}" data-k="${esc(key)}" data-v="${esc(v)}" aria-pressed="${sel.has(v)}">
        <span class="box"></span><span class="lab" title="${esc(v)}">${esc(v)}</span><span class="c">${fmt.format(c)}</span></button>`).join("")}
      ${hidden || expanded.has(key) ? `<button class="link more" data-more="${esc(key)}">${expanded.has(key) ? "Show fewer" : `Show ${hidden} more`}</button>` : ""}</div>`;
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
    pills.push(`<button class="pill" data-k="${esc(k)}" data-v="${esc(v)}"><span>${esc(labels[k] || k)}</span>${esc(v)}<i aria-hidden="true">×</i></button>`)));
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
        .text((d) => (d.x1 - d.x0 > 70 ? `${d.data.name} · ${fmt.format(d.value)}` : d.x1 - d.x0 > 34 ? d.data.name : ""));
    });
  }
  const leaves = h.leaves();
  const maxRank = d3.max(leaves, (d) => d.data.rank) || 1;
  const sel = state.dom ? state.sel[DOMS[state.dom].main] : null;
  const g = svg.selectAll("g.leaf").data(leaves).join("g").attr("class", (d) => "cell leaf" + (sel && sel.size && !sel.has(d.data.name) ? " dim" : ""))
    .attr("transform", (d) => `translate(${d.x0},${d.y0})`);
  g.append("rect").attr("width", (d) => Math.max(0, d.x1 - d.x0)).attr("height", (d) => Math.max(0, d.y1 - d.y0)).attr("rx", 4)
    .attr("fill", (d) => d3.interpolateRgb(surface, cssVar(`--c-${d.data.dom}`))(d.data.other ? 0.28 : 0.95 - 0.55 * (d.data.rank / Math.max(maxRank, 6))))
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
    if (hh > 42) t.append("text").attr("class", "num").attr("x", 8).attr("y", 34).attr("fill", color).text(fmt.format(d.value));
  });
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
  $("#toolbar", root).scrollIntoView({ behavior: "smooth", block: "start" });
}
function tip(ev, html) {
  const el = $("#tip");
  if (!ev) { el.hidden = true; return; }
  el.innerHTML = html; el.hidden = false;
  el.style.left = Math.min(ev.clientX + 14, innerWidth - el.offsetWidth - 8) + "px";
  el.style.top = ev.clientY + 16 + "px";
}

// ── list ─────────────────────────────────────────────────────────────────────
const CJKsafe = (s) => s;
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
    .flatMap(([k]) => vals(e.f[k]).filter((v) => !LEFTOVER.test(v)).slice(0, 2)).slice(0, 5);
  const stats = e.n ? [`${e.n.systems} systems`, `${e.n.files} files`, `${e.n.checks} checks`] : [];
  return `<li><a class="card" href="#/task/${encodeURIComponent(e.id)}" style="--dc:var(--c-${e.d})">
    <div class="row1"><span class="dot"></span><span class="dn">${esc(d.name)}</span><span class="sep">/</span><span>${esc(vals(e.f[d.main]).join(", "))}</span></div>
    <p class="t">${highlight(CJKsafe(e.t))}</p><p class="s">${highlight(e.s)}</p>
    ${chips.length || stats.length ? `<div class="chips">${chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}${stats.map((s) => `<span class="chip stat">${s}</span>`).join("")}</div>` : ""}
  </a></li>`;
}
function renderList(reset) {
  const list = $("#list", root);
  if (reset) { list.innerHTML = ""; shown = 0; }
  const next = matches.slice(shown, shown + PAGE);
  list.insertAdjacentHTML("beforeend", next.map(card).join(""));
  shown += next.length;
  $("#empty", root).hidden = matches.length > 0;
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
  $("#clear", el).addEventListener("click", () => { state.sel = {}; state.q = ""; $("#q", el).value = ""; state.dom = null; run(); });
  $("#surprise", el).addEventListener("click", () => {
    const pool = matches.length ? matches : ENVS;
    location.hash = "#/task/" + encodeURIComponent(pool[Math.floor(Math.random() * pool.length)].id);
  });
  $("#open-filters", el).addEventListener("click", () => $("#filters", el).classList.toggle("open"));
  observer = new IntersectionObserver((es) => { if (es.some((x) => x.isIntersecting) && shown < matches.length) renderList(false); }, { rootMargin: "600px" });
  observer.observe($("#sentinel", el));
}

export function focusSearch() { const q = root && $("#q", root); if (q) q.focus(); }
