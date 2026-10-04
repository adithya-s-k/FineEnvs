// Explore: FineEnvs at the top with the numbers that matter, the trending environments in a row you slide
// through, then everything: Harbor task datasets and OpenEnv Spaces, by collection, searchable and filterable.
// The server filters, counts and sorts (/api/search, over the catalog snapshot: app/search_api.py); the page asks
// for one page of cards at a time, and the first request also brings the header numbers and the trending row.
import { $, api, esc, fmt, ago, sk, progress, emptyState, toast, openDialog, closeModal, storage, nav, setMeta, ownerLink } from "./util.js";
import { icon } from "./icons.js";
import { myDatasets } from "./data.js";
import { countUp } from "./count-up.js";

const PAGE = 40, TRENDING = 24;
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const enc = (s) => s.split("/").map(encodeURIComponent).join("/");
const href = (d) => `/${d.kind === "space" ? "s" : "d"}/${enc(d.id)}`;
const SORTS = [["trending", "Hub interest, then likes and downloads"], ["downloads", "most downloaded first"], ["likes", "most liked first"],
               ["tasks", "most tasks first"], ["rollouts", "most rollouts first"], ["updated", "recently updated first"], ["new", "newest first"]];
const time = (iso) => (iso ? Date.parse(iso) : 0);
// what kind of environment each is: Harbor task folders, an OpenEnv (or other) Space, or rows a processor reads
// Spaces: API-checked OpenEnv servers, unverified discovery candidates, other environment servers; datasets: by the Hub's
// framework tags, Harbor whenever its index found task folders (the server works it out: `fw` on each card)
const KINDS = [["all", "Everything", "All"], ["harbor", "Harbor datasets", "Harbor"], ["openenv", "OpenEnv · API checked", "OpenEnv"], ["unverified-space", "Unverified Space candidates", "Unverified Spaces"],
               ["other-space", "Other environment Spaces", "Other Spaces"], ["verifiers", "Verifiers", "Verifiers"],
               ["nemo-gym", "NeMo Gym", "NeMo Gym"], ["openenv-data", "OpenEnv datasets", "OpenEnv data"], ["verl", "verl and SkyRL", "verl"],
               ["rows", "Other RL datasets", "Other"]];
// ORS servers are few (one, so far): they sit with the other environment Spaces, their cards still say ORS
const fw = (d) => d.fw || (d.kind === "space" ? (d.framework === "openenv" || d.openenv ? "openenv" : "other-space") : d.framework || "harbor");
const FW = { harbor: ["database", "Harbor", "dataset", "A dataset of tasks in Harbor's format"], verifiers: ["check", "Verifiers", "dataset", "Data for a Verifiers environment"],
             "nemo-gym": ["cpu", "NeMo Gym", "dataset", "A NeMo Gym environment's data"], "openenv-data": ["table", "OpenEnv", "dataset", "Data for an OpenEnv environment"],
             verl: ["table", "verl", "dataset", "RL rows in verl's format"], rows: ["table", "RL", "dataset", "A dataset of RL tasks, read row by row"] };
const OTHER = "Not in a collection";
const SIZE_ORDER = ["Under 100", "100 to 1,000", "1,000 or more", "Not indexed yet"];
const MINE = { id: "mine", group: "Your datasets", icon: "user", color: null, about: "yours and your organizations', private ones too" };
const FACETS = [["health", "OpenEnv API availability"], ["mode", "OpenEnv interface"], ["oe", "OpenEnv version (declared)"],
                ["oe_source", "Version evidence"], ["tools", "MCP tool discovery"], ["type", "Labelled on the Hub as"], ["size", "Tasks"],
                ["stage", "Space status at snapshot"], ["mcp", "MCP metadata"], ["evidence", "OpenEnv evidence"], ["tags", "Tags"]];
const FACET_HELP = {
  health: "API checked means running with healthy OpenEnv endpoints checked within the last hour. Episodes and rewards are not tested.",
  mode: "Simulation exposes reset, step and state. MCP service exposes tools without episode control routes.",
  oe: "Repository dependency or Hub tag. This does not confirm the installed runtime version. Ranges and Git references are kept as declared.",
  tools: "Tools returned by the server within the last hour. Discovery does not test tool execution.",
};
const versionGroup = (v) => /^\d/.test(v) ? 0 : /^[<>=!~]/.test(v) ? 1 : v.startsWith("git:") ? 2 : v === "Unknown" ? 4 : 3;

let root = null, COLLECTIONS = [], COLL = {}, STATS = null, TREND = null, FEATURED = [], PROMOTED = [], SNAPSHOT = null, RANKING = null, MINE_N = 0;
const state = { q: "", owner: null, coll: null, kind: "all", sel: {}, sort: "trending", mine: false, candidates: false };
let res = null, rows = [], page = 1, seq = 0, ctrl = null, typing = null;
let refreshTimer = null, refreshing = false, lastAnswer = 0;
const known = new Map();   // every card seen, by key: what the quick look opens
const expanded = new Set(), facetOpen = new Map();

// ── the API ──────────────────────────────────────────────────────────────────
const cache = new Map();   // recent answers by URL: back to Explore renders at once
async function get(url, signal) {
  const hit = cache.get(url);
  if (hit && Date.now() - hit.at < 30000) return hit.d;
  const d = await api(url, { signal });
  cache.set(url, { at: Date.now(), d });
  if (cache.size > 40) cache.delete(cache.keys().next().value);
  return d;
}
function fString() {
  return Object.entries(state.sel).filter(([, s]) => s.size).map(([k, s]) => k + ":" + [...s].map(encodeURIComponent).join("|")).join(";");
}
function query({ page: pg = 1, facets = true, trending = 0 } = {}) {
  const p = new URLSearchParams();
  p.set("scope", state.candidates || state.kind === "unverified-space" ? "all" : "ready");
  if (state.q) p.set("q", state.q);
  if (state.owner) p.set("owner", state.owner);
  if (state.kind !== "all") p.set("kind", state.kind);
  if (state.coll) p.set("collection", state.coll);
  if (state.sort !== "trending") p.set("sort", state.sort);
  const f = fString();
  if (f) p.set("f", f);
  if (pg > 1) p.set("page", pg);
  if (!facets) p.set("facets", "0");
  if (trending) p.set("trending", trending);
  if (state.mine) p.set("mine", "1");
  return `/api/search?${p}`;
}
function remember(list) { (list || []).forEach((d) => known.set(d.key, d)); }
function apply(d) {
  if (d.collections) {
    COLLECTIONS = d.collections;
    COLL = Object.fromEntries(COLLECTIONS.map((c) => [c.id, c]));
    if (MINE_N) COLL.mine = { ...MINE, ids: [] };
  }
  if (d.stats) STATS = d.stats;
  if (d.snapshot) SNAPSHOT = d.snapshot;
  if (d.ranking) RANKING = d.ranking;
  if (d.featured) FEATURED = d.featured;
  if (d.promoted) { PROMOTED = d.promoted; remember(PROMOTED); }
  if (d.trending) { TREND = d.trending; remember(TREND); }
  remember(d.rows);
  // what the server applied: an unknown collection is no filter (but "Yours" waits for the visitor's datasets)
  if (d.query && !(state.coll === "mine" && !MINE_N)) state.coll = d.query.collection;
  if (d.query) state.owner = d.query.owner || null;
  if (d.query?.f) state.sel = Object.fromEntries(Object.entries(d.query.f).map(([k, v]) => [k, new Set(v)]));
}
const color = (id) => (COLL[id]?.color ? `var(--c-${COLL[id].color})` : "var(--text)");

// the page's own containers with placeholders in them, so nothing moves when the data arrives
export function skeleton() {
  const trCard = () => `<li><div class="tr-card sk-in">${sk.line(42, 11)}${sk.line(78, 14)}${sk.lines(100, 64)}${sk.line(36, 11)}</div></li>`;
  const card = () => `<li><div class="card sk-in">${sk.line(34, 11)}${sk.line(56, 16)}${sk.lines(96, 72)}${sk.line(28, 11)}</div></li>`;
  const facet = (n) => `<div class="sk-in sk-facet">${sk.line(45, 10)}${Array.from({ length: n }, (_, i) => sk.line([80, 64, 72, 56, 68][i % 5], 12)).join("")}</div>`;
  return `<div class="wrap" aria-busy="true">
    <section class="ex-head"><div class="sk-in sk-hero">${sk.line(62, 28)}${sk.line(88, 13)}${sk.line(46, 13)}</div>
      <div class="ex-stats fe-stats">${[0, 1, 2].map(() => `<div class="sk-in sk-stat">${sk.line(100, 22)}${sk.line(80, 10)}</div>`).join("")}</div></section>
    <section class="fe-trend"><div class="fe-trend-h">${sk.box(16, "width:96px")}<span class="grow"></span>${sk.box(16, "width:120px")}</div>
      <ol class="tr-grid">${Array.from({ length: 8 }, trCard).join("")}</ol></section>
    <div class="toolbar">${sk.box(40, "flex:1;border-radius:10px")}${sk.box(40, "width:128px;border-radius:10px")}</div>
    <div class="layout">
      <aside class="filters sk-filters">${facet(6)}${facet(4)}${facet(5)}</aside>
      <section class="results"><div class="results-head">${sk.box(14, "width:160px")}</div>
        <ol class="list">${Array.from({ length: 6 }, card).join("")}</ol></section>
    </div></div>`;
}

export async function mount(el, { qs }) {
  root = el;
  state.q = qs.get("q") || "";
  state.owner = qs.get("owner")?.trim().slice(0, 200) || null;
  const k = { dataset: "harbor", space: "openenv", ors: "other-space" }[qs.get("k")] || qs.get("k");   // old links
  state.kind = KINDS.some(([x]) => x === k) ? k : "all";
  state.sort = SORTS.some(([x]) => x === qs.get("s")) ? qs.get("s") : "trending";
  state.coll = qs.get("c") || null;
  state.mine = MINE_N > 0;
  state.candidates = qs.get("scope") === "all" || state.kind === "unverified-space";
  state.sel = {};
  (qs.get("f") || "").split(";").filter(Boolean).forEach((pair) => {
    const [k, v] = pair.split(":");
    if (k && v) state.sel[k] = new Set(v.split("|").map(decodeURIComponent));
  });
  const url = query({ trending: TRENDING });
  const hit = cache.get(url);
  if (!hit || Date.now() - hit.at >= 30000) el.innerHTML = skeleton();
  let d;
  try { d = await progress.wrap(get(url)); } catch (e) {
    if (root !== el) return;
    el.innerHTML = `<div class="wrap page">${emptyState("alert", "Couldn't load the environments", esc(e.message), `<button class="btn" id="reload" type="button">${icon("refresh")}Try again</button>`)}</div>`;
    $("#reload", el)?.addEventListener("click", () => location.reload());
    return;
  }
  if (root !== el) return;
  apply(d);
  el.innerHTML = `
    <div class="wrap fade-in">
      <section class="ex-head">
        <div><h1>Explore RL environments on Hugging Face</h1>
          <p>Across OpenEnv, Harbor, Verifiers and more. Open one to see what it asks and how it's graded, then run an agent on it.</p></div>
        <div class="ex-stats fe-stats" id="stats"></div>
      </section>
      <details class="catalog-method"><summary>How counts and trending work <span class="faint" id="catalog-freshness"></span></summary><div id="catalog-context"></div></details>
      <section class="fe-trend" id="fineenvs-featured" aria-label="FineEnvs environments" hidden>
        <div class="fe-trend-h"><h2>FineEnvs environments</h2><span class="grow"></span><a class="link" href="/?owner=FineEnvs">See all from FineEnvs ${icon("chevronRight", 13)}</a></div>
        <ul class="tr-grid" id="promoted"></ul>
      </section>
      <div class="catalog-featured" id="featured" aria-label="Featured by maintainers"></div>
      <section class="fe-trend" aria-label="Trending environments">
        <div class="fe-trend-h"><h2>${icon("flame", 15)}Trending</h2>
          <span class="muted xs" id="tr-scope"></span>
          <span class="grow"></span><span class="tr-pager" id="tr-pager"><button class="icon-btn" id="tr-prev" type="button" aria-label="Previous">${icon("chevronRight", 16, "flip")}</button>
          <span id="tr-page"></span><button class="icon-btn" id="tr-next" type="button" aria-label="Next">${icon("chevronRight", 16)}</button></span>
          <button class="link" id="tr-all" type="button"></button></div>
        <ol class="tr-grid" id="trending"></ol>
      </section>
      <div class="toolbar" id="toolbar">
        <div class="search">${icon("search", 16)}
          <input id="q" data-search type="search" placeholder="Search environments, owners, tags…" autocomplete="off" spellcheck="false" aria-label="Search environments"><kbd>/</kbd>
        </div>
        <button class="btn" id="random" type="button" title="Open a random task from the collections">${icon("shuffle", 15)}<span class="long">Random task</span></button>
        <button class="btn only-mobile" id="open-filters" type="button" aria-controls="filters" aria-expanded="false">${icon("filter", 15)}Filters<span id="nfilt"></span></button>
      </div>
      <div class="layout">
        <div class="sheet-scrim" id="sheet-scrim" hidden></div>
        <aside class="filters" id="filters" aria-label="Filters">
          <div class="sheet-head"><b>Filters</b><button class="icon-btn" id="close-filters" type="button" aria-label="Close filters">${icon("x", 18)}</button></div>
          <div class="filters-head"><span>Filters</span><button class="link" id="clear" type="button">Clear all</button></div>
          <label class="candidate-option"><input type="checkbox" id="include-candidates">Include unverified or offline Spaces</label>
          <div id="facets"></div>
          <div class="sheet-foot"><button class="btn" id="clear3" type="button">Clear</button><button class="btn primary grow" id="show-results" type="button">Show results</button></div>
        </aside>
        <section class="results" aria-live="polite">
          <div class="results-head"><span class="count" id="count"></span>
            <label class="order sortsel"><span id="order-icon"></span><select id="sort" aria-label="Sort">${SORTS.map(([k, l]) => `<option value="${k}">${l}</option>`).join("")}</select></label>
            <div class="active" id="active"></div></div>
          <ol class="list" id="list"></ol>
          <div class="more-row" id="more-row" hidden><span class="muted sm" id="more-count"></span><button class="btn" id="more" type="button">Show more</button></div>
          <div id="empty" hidden>${emptyState("search", "Nothing matches", "Try fewer filters or a different word.", `<button class="btn" id="clear2" type="button">Clear search and filters</button>`)}</div>
        </section>
      </div>
    </div>`;
  $("#q", el).value = state.q;
  $("#sort", el).value = state.sort;
  renderStats();
  renderTrending();
  wire(el);
  show(d);
  writeURL();
  refreshTimer = setInterval(refreshCatalog, 45000);
  // the signed-in visitor's own datasets: "Yours" in Collection, their private ones among the cards
  if (!MINE_N) myDatasets().then((mine) => {
    if (root !== el) return;
    if (!mine.length) { if (state.coll === "mine") setColl(null); return; }   // a "Yours" link, signed out: no filter
    MINE_N = mine.length;
    COLL.mine = { ...MINE, ids: [] };
    state.mine = true;
    run({ trending: true });
  });
}

export function unmount() { document.body.classList.remove("noscroll"); ctrl?.abort(); clearTimeout(typing); clearInterval(refreshTimer); root = null; }

function writeURL() {
  const p = new URLSearchParams();
  if (state.candidates) p.set("scope", "all");
  if (state.q) p.set("q", state.q);
  if (state.owner) p.set("owner", state.owner);
  if (state.kind !== "all") p.set("k", state.kind);
  if (state.coll) p.set("c", state.coll);
  if (state.sort !== "trending") p.set("s", state.sort);
  const f = fString();
  if (f) p.set("f", f);
  history.replaceState(null, "", "/" + (p.toString() ? `?${p}` : ""));
}

function renderStats() {
  if (!STATS) return;
  const box = $("#stats", root), previous = box._counts || {};
  const coverage = STATS.space_task_coverage || {};
  const numbers = { datasets: STATS.datasets, spaces: STATS.spaces, tasks: STATS.dataset_tasks ?? STATS.tasks, reported: STATS.space_tasks || 0 };
  const metrics = [["datasets", "datasets"], ["spaces", state.candidates ? "Space candidates" : "checked Spaces"], ["tasks", "tasks indexed"], ["reported", "Space task entries"]];
  box.innerHTML = metrics.map(([key, label]) => `<div role="group" aria-label="${fmt.format(numbers[key])} ${label}" title="${fmt.format(numbers[key])} ${label}${key === "tasks" ? '; dataset records in the explorer index' : key === "reported" ? '; reported by Space APIs, loaded on demand, may overlap indexed datasets' : ''}">
    <b data-count="${key}" aria-hidden="true">${fmt.format(numbers[key])}</b><span>${label}</span>
    ${key === "reported" ? `<small>reported by ${fmt.format(coverage.spaces || 0)} Spaces</small>` : ""}</div>`).join("");
  for (const [key] of metrics) countUp(box.querySelector(`[data-count="${key}"]`), numbers[key], (n) => fmt.format(n), previous[key] ?? 0);
  box._counts = numbers;
  const at = SNAPSHOT?.built_at;
  $("#catalog-freshness", root).textContent = `· updates automatically${at ? ` · catalog ${ago(time(at) / 1000)}` : ""}`;
  $("#catalog-context", root).innerHTML = `<p>All four counts follow your current search, organization and filters.
    By default, datasets and running Spaces with a recently checked environment API are shown. Offline and unverified candidates are available in Filters.
    Checks run automatically; OpenEnv labels require passing OpenEnv API checks within the last hour. An API check does not certify episode or reward quality.</p>
    <p><b>${fmt.format(numbers.tasks)} tasks indexed</b> are dataset records loaded into this explorer's index.
    <b>${fmt.format(numbers.reported)} Space task entries</b> are reported by ${fmt.format(coverage.spaces || 0)} running Spaces within the last hour; individual task records are fetched when you open them.
    These are entries across repositories and splits, not unique problems: forks, conversions and shared datasets can overlap. The two task counts can also overlap and are not added together.
    ${coverage.partial_spaces ? `${fmt.format(coverage.partial_spaces)} Space catalogs are partial; only known split counts are included.` : ""}
    Spaces without a recent task count are uncounted, not assumed empty. Space totals describe their Task API catalogs; their task bodies are loaded on demand.
    ${coverage.oldest_check ? `Oldest included Space count: ${esc(new Date(coverage.oldest_check * 1000).toLocaleString())}.` : "No recent Space counts yet."}</p>
    <p>Repository names, tags and an <code>openenv.yaml</code> filename are discovery hints only. Versions describe repository dependencies, not a confirmed installed runtime.
    Automatic indexing refreshes public repositories and supported dataset indexes in batches; full task records from row datasets and Spaces are loaded on demand.</p>
    <p>${esc(RANKING?.description || "Hub trending score, then total likes and monthly dataset downloads.")} ${esc(RANKING?.quality || "")}</p>
    <p>Counts cover the public, tagged environment catalog and curated repositories, not a census of all Hub Spaces.
    ${at ? `Catalog snapshot: <time datetime="${esc(at)}" title="${esc(at)}">${esc(new Date(at).toLocaleString())}</time>. Runtime status and metrics can change.` : "Snapshot time unavailable."}</p>`;
}

// ── trending: a grid of cards, a page at a time ──────────────────────────────
let trPage = 0;
function trendCard(d, i = null) {
  const [org, name] = d.id.split("/");
  const c = COLL[d.collection];
  const space = d.kind === "space";
  const nums = space
    ? [[icon("heart", 12), compact.format(d.likes), "likes"], ...(d.mcp ? [[icon("plug", 12), "MCP", "tagged for MCP"]] : [])]
    : [[icon("download", 12), compact.format(d.downloads), "downloads last month"], [icon("heart", 12), compact.format(d.likes), "likes"],
       ...(d.indexed?.tasks ? [[icon("list", 12), compact.format(d.indexed.tasks), "tasks"]] : [])];
  nums.unshift([icon("flame", 12), compact.format(d.trending || 0), "Hub trending score; 0 means ordering falls back to likes and dataset downloads"]);
  const kind = kindTag(d, true);
  return `<li><div class="tr-card${space ? " is-space" : ""}" data-open="${href(d)}" style="--dc:${c ? color(c.id) : "var(--faint)"}">
    <div class="tr-row1">${i === null ? "" : `<b>${i + 1}</b>`}${kind}${c ? `<em>${esc(c.group)}</em>` : ""}</div>
    <div class="tr-name">${ownerLink(org, "/")}<a href="${href(d)}" title="${esc(d.id)}">${esc(name)}</a></div>
    <p>${esc(d.brief || d.heading || (c ? c.about : "") || "")}</p>
    <div class="tr-foot">${d.api_status === "API checked" ? `<span title="${esc(checkTitle(d))}">${icon("check", 12)}API checked</span>` : ""}${nums.map(([ic, n, l]) => `<span title="${esc(l)}">${ic}<b>${n}</b></span>`).join("")}</div>
    ${space ? `<img class="tr-thumb sp-thumb" src="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/${enc(d.id)}.png" alt="" loading="lazy">` : ""}
    <button class="ql-btn" type="button" data-ql="${esc(d.key)}" title="Quick look" aria-label="Quick look at ${esc(d.id)}">${icon("eye", 14)}</button></div></li>`;
}
// the framework an environment is built for, and what it is: a Harbor dataset of tasks, or an OpenEnv Space (a server)
function kindTag(d, dot = false) {
  if (d.kind === "space") {
    const k = d.framework === "ors" ? "ors" : fw(d), name = { openenv: "OpenEnv", "unverified-space": "Unverified", ors: "ORS" }[k];
    return `<span class="kind-tag sp" title="${k === "openenv" ? `OpenEnv API checks passed within the last hour. This does not verify episode or reward quality.` : k === "unverified-space" ? "Repository metadata only. An openenv.yaml filename or Hub tag does not establish OpenEnv compatibility." : k === "ors" ? "Declares Open Reward Standard support" : "An environment on a Space"}">${icon("globe", 11)}${name ? `${name}<span class="k2">Space</span>` : "Space"}${dot && d.stage === "RUNNING" ? '<i class="dot" style="--dc:var(--ok)" title="Running at snapshot time"></i>' : ""}</span>`;
  }
  const [ic, a, b, title] = FW[d.framework] || FW.harbor;
  return `<span class="kind-tag ds ${esc(d.framework || "harbor")}" title="${title}">${icon(ic, 11)}${a}<span class="k2">${b}</span></span>`;
}
function pageSize() {
  const w = $("#trending", root)?.clientWidth || 1200;
  const cols = w >= 1100 ? 4 : w >= 760 ? 3 : w >= 520 ? 2 : 1;
  return cols === 1 ? 4 : cols * 2;   // two rows (four cards stacked on a phone)
}
function renderTrending() {
  if (!TREND) return;
  $("#fineenvs-featured", root).hidden = !PROMOTED.length;
  $("#promoted", root).innerHTML = PROMOTED.map((d) => trendCard(d)).join("");
  $("#featured", root).hidden = !FEATURED.length;
  $("#featured", root).innerHTML = `<span class="muted xs">Featured by maintainers</span>${FEATURED.map((d) => `<a class="chip" href="${href(d)}">${esc(d.id)}</a>`).join("")}`;
  const filtered = state.owner || state.kind !== "all" || state.coll || state.q || Object.keys(state.sel).length;
  $("#tr-scope", root).textContent = filtered ? "Within these filters" : "Across the catalog";
  const all = storage.get("trend-view") === "all";
  const top = TREND;
  const n = pageSize(), pages = Math.max(1, Math.ceil(top.length / n));
  trPage = Math.max(0, Math.min(trPage, pages - 1));
  const shown = all ? top : top.slice(trPage * n, trPage * n + n);
  const grid = $("#trending", root);
  grid.innerHTML = shown.length ? shown.map((d) => trendCard(d, top.indexOf(d))).join("") : '<li class="muted sm">No environments match these filters.</li>';
  grid.classList.remove("turn");
  void grid.offsetWidth;   // replay the page-turn fade
  grid.classList.add("turn");
  $("#tr-all", root).textContent = all ? "Show fewer" : `See all ${top.length}`;
  $("#tr-all", root).hidden = top.length <= n;
  $("#tr-pager", root).hidden = all || top.length <= n;
  $("#tr-page", root).textContent = top.length ? `${trPage * n + 1}–${Math.min(top.length, trPage * n + n)} of ${top.length}` : "0 results";
  $("#tr-prev", root).disabled = trPage === 0;
  $("#tr-next", root).disabled = trPage >= pages - 1;
}

// ── quick look: an environment's numbers and facts in a popup, without leaving the page ─
function quickLook(d) {
  const [org, name] = d.id.split("/");
  const c = COLL[d.collection];
  const ix = d.indexed;
  const kinds = ix?.graded ? Object.entries(ix.graded).filter(([k]) => k !== "unknown").map(([k]) => (k === "json" ? "named scores" : "one score")) : [];
  const facts = d.kind === "space" ? [
    ["Status at snapshot", esc(d.stage_label || d.stage || "Unknown")],
    d.api_status ? ["OpenEnv API", `${esc(d.api_status)}${d.api_check?.checked_at ? ` · ${esc(ago(d.api_check.checked_at))}` : ""}`] : null,
    d.api_mode ? ["Interface", esc(d.api_mode)] : null,
    d.declared_version ? ["OpenEnv version (declared)", `${esc(d.declared_version)} · ${esc(d.version_source)}`] : null,
    d.api_check?.version?.path ? ["Version source", `<a class="u" href="https://huggingface.co/spaces/${enc(d.id)}/blob/${esc(d.api_check.version.revision)}/${enc(d.api_check.version.path)}" target="_blank" rel="noopener">${esc(d.api_check.version.path)}</a>`] : null,
    d.tools_status ? ["MCP tool discovery", esc(d.tools_status)] : null,
    ["Likes", fmt.format(d.likes)], ["Hub trending score", fmt.format(d.trending || 0)],
    ["Kind", fw(d) === "openenv" ? `API-checked OpenEnv${d.openenv_version ? ` ${esc(d.openenv_version)}` : ""}${d.manifest ? ` (<code>${esc(d.manifest)}</code>)` : ""}` : fw(d) === "unverified-space" ? "Unverified Space candidate; metadata is not API verification" : d.framework === "ors" ? "ORS declared" : "Docker Space"],
    d.evidence ? ["OpenEnv evidence", esc(d.evidence)] : null,
    ["MCP metadata", d.mcp ? "tagged; inspect live tools when opened" : "not tagged"], d.hardware ? ["Hardware", esc(d.hardware)] : null,
    d.updated ? ["Updated", ago(time(d.updated) / 1000)] : null,
  ] : [
    ["Tasks", ix?.tasks != null ? fmt.format(ix.tasks) : "not indexed yet"], ["Downloads", `${compact.format(d.downloads)} <span class="faint">last month</span>`],
    ["Likes", fmt.format(d.likes)], ["Hub trending score", fmt.format(d.trending || 0)],
    kinds.length ? ["Graded by", esc(kinds.join(", "))] : null, ix?.tasks ? ["Prebuilt image", `${Math.round((100 * (ix.image || 0)) / ix.tasks)}% of tasks`] : null,
    d.rollouts ? ["Public rollouts", fmt.format(d.rollouts)] : null, d.updated ? ["Updated", ago(time(d.updated) / 1000)] : null,
  ];
  const dlg = openDialog(`<div class="ql">
    ${d.kind === "space" ? `<img class="ql-shot" src="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/${enc(d.id)}.png" alt="">` : ""}
    <div class="dialog-h"><div><div class="ql-kick">${c ? `<span class="badge" style="--dc:${color(c.id)}">${icon(c.icon, 12)}${esc(c.group)}</span>` : ""}
        <span class="chip">${icon(d.kind === "space" ? "globe" : "database", 11)}${d.kind === "space" ? "Space" : "Dataset"}</span>${d.badges.map((b) => `<span class="chip">${esc(b)}</span>`).join("")}</div>
      <h2>${ownerLink(org, "/")}${esc(name)}</h2>${d.heading && d.heading !== name ? `<p>${esc(d.heading)}</p>` : ""}</div>
      <button class="icon-btn" type="button" data-close aria-label="Close">${icon("x", 18)}</button></div>
    <div class="dialog-b">${d.brief ? `<p class="ql-brief">${esc(d.brief)}</p>` : ""}
      <dl class="ql-facts">${facts.filter(Boolean).map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
      ${d.tags.length ? `<div class="chips">${d.tags.slice(0, 10).map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div>` : ""}
      <div class="ql-actions"><a class="btn primary" href="${href(d)}">${icon(d.kind === "space" ? "globe" : "list", 14)}${d.kind === "space" ? "Open the Space" : "Open the tasks"}</a>
        <a class="btn" href="https://huggingface.co/${d.kind === "space" ? "spaces" : "datasets"}/${enc(d.id)}" target="_blank" rel="noopener">${icon("external", 14)}On the Hub</a></div></div></div>`);
  dlg.classList.add("wide");
  dlg.addEventListener("click", (e) => { if (e.target.closest("[data-close]") || e.target.closest(".ql-actions a[href^='#']")) closeModal(); });
}

// ── asking the server ────────────────────────────────────────────────────────
// Any change of search, kind, collection, facet or order: the controls answer at once, the cards and counts when the
// server does. A newer question cancels an older one still on its way; a slow answer shows placeholder cards.
function run({ debounce = 0, trending = true } = {}) {
  writeURL();
  renderFacets(); renderActive();
  clearTimeout(typing);
  if (debounce) { typing = setTimeout(() => ask(trending), debounce); return; }
  ask(trending);
}
async function ask(withTrending) {
  ctrl?.abort();
  const mine = (ctrl = new AbortController()), my = ++seq;
  const slow = setTimeout(() => { if (my === seq && root) loading(true); }, 150);
  try {
    const d = await get(query({ trending: withTrending ? TRENDING : 0 }), mine.signal);
    if (my !== seq || !root) return;
    apply(d);
    show(d);
    if (d.trending) { trPage = 0; renderStats(); renderTrending(); }
    writeURL();
  } catch (e) {
    if (e.name === "AbortError" || my !== seq || !root) return;
    loading(false);
    toast(`Couldn't load the environments: ${e.message}`);
  } finally { clearTimeout(slow); }
}

async function refreshCatalog() {
  // Refresh an idle first page. Never replace expanded results or a focused
  // card/filter while someone is browsing, and never race a newer query.
  if (!root || document.hidden || refreshing || page !== 1 || Date.now() - lastAnswer < 40000) return;
  const active = document.activeElement;
  if (active?.closest(".results, .filters, .fe-trend, .modal")) return;
  const el = root, version = seq, url = query({ trending: TRENDING });
  refreshing = true;
  try {
    const d = await get(url);
    if (root !== el || seq !== version || page !== 1 || url !== query({ trending: TRENDING })) return;
    apply(d);
    show(d);
    renderStats();
    renderTrending();
  } catch { /* Keep the last successful catalog; retry on the next tick. */ }
  finally { refreshing = false; }
}

function loading(on) {
  const list = $("#list", root);
  $(".results", root)?.setAttribute("aria-busy", on ? "true" : "false");
  if (!on) { renderList(); return; }
  list.hidden = false;
  $("#empty", root).hidden = true;
  $("#more-row", root).hidden = true;
  list.innerHTML = sk.cards(6);
}
function show(d) {
  lastAnswer = Date.now();
  res = d;
  rows = d.rows.slice();
  page = 1;
  $(".results", root)?.setAttribute("aria-busy", "false");
  $("#order-icon", root).innerHTML = icon(state.sort === "trending" ? "flame" : "list", 13);
  renderFacets(); renderActive(); renderList();
}
async function more() {
  const btn = $("#more", root), list = $("#list", root), my = seq;
  btn.disabled = true;
  btn.textContent = "Loading…";
  list.insertAdjacentHTML("beforeend", sk.cards(4).replaceAll("<li>", '<li class="sk-more">'));
  try {
    const d = await get(query({ page: page + 1, facets: false }));
    if (my !== seq || !root) return;
    page += 1;
    remember(d.rows);
    rows.push(...d.rows);
    list.querySelectorAll(".sk-more").forEach((x) => x.remove());
    list.insertAdjacentHTML("beforeend", d.rows.map(card).join(""));
    moreRow();
  } catch (e) {
    if (my !== seq || !root) return;
    toast(`Couldn't load more: ${e.message}`);
  } finally {
    if (root && my === seq) { list.querySelectorAll(".sk-more").forEach((x) => x.remove()); btn.disabled = false; moreRow(); }
  }
}

// ── filters ──────────────────────────────────────────────────────────────────
const phone = () => matchMedia("(max-width: 760px)").matches;
function renderFacets() {
  if (!res?.facets || !root) return;
  const active = document.activeElement;
  const focus = active?.closest('#facets') && active.matches('button') ? { ...active.dataset } : null;
  const F = res.facets;
  const collFacet = (() => {
    const counts = F.collection || {};
    const items = [...(MINE_N ? [["mine", "Yours"]] : []), ...COLLECTIONS.map((c) => [c.id, c.group]), ["other", OTHER]];
    return `<details class="facet" data-facet="coll" open><summary><h3>Collection</h3>${state.coll ? `<span class="fsel">1 selected</span>` : ""}${icon("chevronDown", 14, "chev")}</summary>
      ${items.map(([id, label]) => `<button class="opt ${counts[id] ? "" : "zero"}" data-coll="${id}" aria-pressed="${state.coll === id}">
        <span class="box">${state.coll === id ? icon("check", 11) : ""}</span><span class="lab">${esc(label)}</span><span class="c">${fmt.format(counts[id] || 0)}</span></button>`).join("")}</details>`;
  })();
  const kindFacet = `<details class="facet" data-facet="kind" open><summary><h3>Kind</h3>${icon("chevronDown", 14, "chev")}</summary>
    ${KINDS.map(([k, l]) => { const n = F.kind?.[k] || 0;
      if (!n && k !== "all" && state.kind !== k) return "";
      return `<button class="opt radio" data-kind="${k}" aria-pressed="${state.kind === k}"><span class="box">${state.kind === k ? '<i></i>' : ""}</span><span class="lab">${l}</span><span class="c">${fmt.format(n)}</span></button>`; }).join("")}</details>`;
  $("#facets", root).innerHTML = kindFacet + collFacet + FACETS.map(([key, label]) => {
    const counts = new Map(F[key] || []);
    if (key === "health" && counts.size && !counts.has("API checked")) counts.set("API checked", 0);
    if (key === "tools" && counts.size && !counts.has("Tools discovered")) counts.set("Tools discovered", 0);
    const sel = state.sel[key] || new Set();
    sel.forEach((v) => { if (!counts.has(v)) counts.set(v, 0); });
    if (counts.size <= 1 && !sel.size && !(["health", "oe", "tools", "mode"].includes(key) && counts.size)) return "";
    let items = [...counts.entries()].sort(key === "size" ? (a, b) => SIZE_ORDER.indexOf(a[0]) - SIZE_ORDER.indexOf(b[0])
      : key === "oe" ? (a, b) => versionGroup(a[0]) - versionGroup(b[0]) || String(b[0]).localeCompare(String(a[0]), undefined, { numeric: true })
      : (a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
    const limit = expanded.has(key) ? Infinity : (phone() ? 5 : 8);
    const hidden = Math.max(0, items.length - limit);
    items = items.slice(0, limit);
    const open = facetOpen.has(key) ? facetOpen.get(key) : (!phone() || sel.size > 0);
    return `<details class="facet" data-facet="${key}" ${open ? "open" : ""}><summary><h3>${esc(label)}</h3>${sel.size ? `<span class="fsel">${sel.size} selected</span>` : ""}${icon("chevronDown", 14, "chev")}</summary>
      ${FACET_HELP[key] ? `<p class="facet-help">${esc(FACET_HELP[key])}</p>` : ""}
      ${items.map(([v, c]) => `<button class="opt ${c ? "" : "zero"}" data-k="${key}" data-v="${esc(v)}" aria-pressed="${sel.has(v)}">
        <span class="box">${sel.has(v) ? icon("check", 11) : ""}</span><span class="lab" title="${esc(v)}">${esc(v)}</span><span class="c">${fmt.format(c)}</span></button>`).join("")}
      ${hidden || expanded.has(key) ? `<button class="link more" data-more="${key}">${icon(expanded.has(key) ? "chevronDown" : "chevronRight", 13)}${expanded.has(key) ? "Show fewer" : `Show ${hidden} more`}</button>` : ""}</details>`;
  }).join("");
  const nf = Object.values(state.sel).reduce((n, s) => n + s.size, 0);
  $("#nfilt", root).textContent = nf ? ` (${nf})` : "";
  if (focus) [...$("#facets", root).querySelectorAll('button')].find((b) => Object.entries(focus).every(([k, v]) => b.dataset[k] === v))?.focus({ preventScroll: true });
}
function toggle(key, v) {
  const set = (state.sel[key] ||= new Set());
  set.has(v) ? set.delete(v) : set.add(v);
  if (!set.size) delete state.sel[key];
  run();
}
function setColl(id) { state.coll = id || null; run(); }
function renderActive() {
  if (!res || !root) return;
  const labels = Object.fromEntries(FACETS);
  $("#include-candidates", root).checked = state.candidates;
  const pills = [];
  const heading = $(".ex-head h1", root), description = $(".ex-head p", root);
  heading.textContent = state.owner ? `Environments by ${state.owner}` : "Explore RL environments on Hugging Face";
  description.textContent = state.owner ? "Datasets and Spaces from this owner in the explorer catalog. Search or filter to find an environment." : "Across OpenEnv, Harbor, Verifiers and more. Open one to see what it asks and how it's graded, then run an agent on it.";
  $("#q", root).placeholder = state.owner ? `Search ${state.owner} environments…` : "Search environments, owners, tags…";
  setMeta({ title: state.owner ? `${state.owner} environments` : undefined });
  if (state.owner) pills.push(`<button class="pill" data-clear-owner aria-label="Remove owner filter"><span>Owner</span>${esc(state.owner)}${icon("x", 13)}</button>`);
  if (state.kind !== "all") pills.push(`<button class="pill" data-kind="all"><span>Kind</span>${esc(KINDS.find(([k]) => k === state.kind)?.[2] || state.kind)}${icon("x", 13)}</button>`);
  if (state.coll) pills.push(`<button class="pill" data-coll=""><span>Collection</span>${esc(COLL[state.coll]?.group || OTHER)}${icon("x", 13)}</button>`);
  Object.entries(state.sel).forEach(([k, set]) => set.forEach((v) =>
    pills.push(`<button class="pill" data-k="${k}" data-v="${esc(v)}"><span>${esc(labels[k] || k)}</span>${esc(v)}${icon("x", 13)}</button>`)));
  $("#active", root).innerHTML = pills.join("");
  const noun = ["openenv", "unverified-space", "other-space"].includes(state.kind) ? "Space" : state.kind !== "all" ? "dataset" : "environment";
  $("#count", root).textContent = `${fmt.format(res.total)} ${noun}${res.total === 1 ? "" : "s"}`;
  const sr = $("#show-results", root);
  if (sr) sr.textContent = `Show ${fmt.format(res.total)} ${noun}${res.total === 1 ? "" : "s"}`;
}

// ── list ─────────────────────────────────────────────────────────────────────
function checkTitle(d) {
  const when = d.api_check?.checked_at ? new Date(d.api_check.checked_at * 1000).toLocaleString() : "Never checked";
  return `${when}. API checks only; episode execution and rewards are not validated.`;
}
function card(d) {
  const [org, name] = d.id.split("/");
  const c = COLL[d.collection];
  const chips = d.kind === "space" ? [
    d.api_status ? `<span class="chip outline" title="${esc(checkTitle(d))}">${icon(d.api_status === "API checked" ? "check" : "info", 11)}${esc(d.api_status)}${d.api_check?.checked_at ? ` · ${esc(ago(d.api_check.checked_at))}` : ""}</span>` : "",
    `<span class="chip outline" title="Hub runtime at snapshot time; open the Space to check now"><span class="dot" style="--dc:${d.stage === "RUNNING" ? "var(--ok)" : "var(--faint)"}"></span>${esc(d.stage_label || d.stage || "Unknown")}</span>`,
    d.likes ? `<span class="chip outline"><b>${compact.format(d.likes)}</b> likes</span>` : "",
    d.mcp ? `<span class="chip outline">${icon("plug", 11)}MCP tagged</span>` : "",
    d.declared_version && d.declared_version !== "Unknown" ? `<span class="chip" title="${esc(d.version_source)}; repository declaration, not installed runtime">Declared OpenEnv ${esc(d.declared_version)}</span>` : "",
    d.evidence ? `<span class="chip">${esc(d.evidence)}</span>` : "",
  ] : [
    d.indexed?.tasks ? `<span class="chip outline"><b>${fmt.format(d.indexed.tasks)}</b> tasks</span>` : "",
    `<span class="chip outline" title="downloads in the last month"><b>${compact.format(d.downloads)}</b> downloads</span>`,
    d.likes ? `<span class="chip outline"><b>${compact.format(d.likes)}</b> likes</span>` : "",
    d.rollouts ? `<span class="chip outline"><b>${fmt.format(d.rollouts)}</b> rollouts</span>` : "",
    d.private ? `<span class="chip">${icon("shield", 11)}Private</span>` : "",
    ...d.badges.map((b) => `<span class="chip">${esc(b)}</span>`),
  ];
  // a search that matched a tag the card doesn't otherwise show: show that tag, so the match makes sense
  const q = state.q.toLowerCase();
  if (q) (d.tags || []).filter((t) => t.toLowerCase().includes(q)).slice(0, 2).forEach((t) => chips.push(`<span class="chip match">${icon("search", 11)}${esc(t)}</span>`));
  return `<li><div class="card${d.kind === "space" ? " sp" : ""}" data-open="${href(d)}" style="--dc:${c ? color(c.id) : "var(--c-hub)"}">
    ${d.kind === "space" ? `<img class="sp-thumb" loading="lazy" alt="" src="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/${enc(d.id)}.png">` : ""}
    <div class="row1">${kindTag(d)}
      ${c ? `<span class="dn">${icon(c.icon, 13)}${esc(c.group)}</span><span class="faint">/</span>` : ""}${ownerLink(org)}
      ${d.pinned ? `<span class="chip">${icon("flag", 11)}Pinned</span>` : ""}${d.updated ? `<span class="when" title="last updated">${esc(ago(time(d.updated) / 1000))}</span>` : ""}</div>
    <p class="t repo"><a href="${href(d)}" title="${esc(d.id)}">${esc(name)}</a></p>
    <button class="ql-btn" type="button" data-ql="${esc(d.key)}" title="Quick look" aria-label="Quick look at ${esc(d.id)}">${icon("eye", 14)}</button>
    ${d.brief || d.heading ? `<p class="s">${esc(d.brief || d.heading)}</p>` : ""}
    <div class="chips">${chips.filter(Boolean).join("")}</div></div></li>`;
}
function renderList() {
  const list = $("#list", root);
  list.innerHTML = rows.map(card).join("");
  $("#empty", root).hidden = res.total > 0;
  list.hidden = !res.total;
  moreRow();
}
function moreRow() {
  const left = res.total - rows.length;
  $("#more-row", root).hidden = left <= 0;
  $("#more-count", root).textContent = `Showing ${fmt.format(rows.length)} of ${fmt.format(res.total)}`;
  $("#more", root).textContent = `Show ${fmt.format(Math.min(PAGE, left))} more`;
}

function wire(el) {
  el.addEventListener("click", (e) => {
    const card = e.target.closest("[data-open]");
    if (!card || e.defaultPrevented || e.target.closest("a,button,input,select") || e.button || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey || window.getSelection()?.toString()) return;
    nav(card.dataset.open);
  });
  $("#q", el).addEventListener("input", (ev) => { state.q = ev.target.value.trim(); run({ debounce: 160 }); });
  $("#sort", el).addEventListener("change", (ev) => { state.sort = ev.target.value; run(); });
  $("#facets", el).addEventListener("click", (ev) => { const b = ev.target.closest("[data-kind]"); if (b) { state.kind = b.dataset.kind; state.coll = null; run(); } });
  $("#tr-all", el).addEventListener("click", () => { storage.set("trend-view", storage.get("trend-view") === "all" ? "row" : "all"); trPage = 0; renderTrending(); });
  $("#tr-prev", el).addEventListener("click", () => { trPage = Math.max(0, trPage - 1); renderTrending(); });
  $("#tr-next", el).addEventListener("click", () => { trPage += 1; renderTrending(); });
  // the quick look opens instead of following the card's link
  el.addEventListener("click", (e) => {
    const q = e.target.closest("[data-ql]");
    if (!q) return;
    e.preventDefault();
    e.stopPropagation();
    const d = known.get(q.dataset.ql);
    if (d) quickLook(d);
  }, true);
  $("#facets", el).addEventListener("click", (ev) => {
    const c = ev.target.closest("[data-coll]");
    if (c) return setColl(state.coll === c.dataset.coll ? null : c.dataset.coll);
    const o = ev.target.closest(".opt:not([data-kind])");
    if (o) return toggle(o.dataset.k, o.dataset.v);
    const m = ev.target.closest("[data-more]");
    if (m) { expanded.has(m.dataset.more) ? expanded.delete(m.dataset.more) : expanded.add(m.dataset.more); renderFacets(); }
  });
  $("#facets", el).addEventListener("toggle", (e) => { const d = e.target.closest?.("[data-facet]"); if (d) facetOpen.set(d.dataset.facet, d.open); }, true);
  $("#active", el).addEventListener("click", (ev) => {
    const p = ev.target.closest(".pill");
    if (!p) return;
    if (p.hasAttribute("data-clear-owner")) { state.owner = null; run(); }
    else if (p.dataset.kind) { state.kind = "all"; run(); } else if (p.dataset.coll != null) setColl(null); else toggle(p.dataset.k, p.dataset.v);
  });
  const clearAll = () => { state.owner = null; state.candidates = false; state.sel = {}; state.q = ""; $("#q", el).value = ""; state.coll = null; state.kind = "all"; run(); };
  $("#include-candidates", el).addEventListener("change", (e) => {
    state.candidates = e.target.checked;
    if (!state.candidates && state.kind === "unverified-space") state.kind = "all";
    run();
  });
  $("#clear", el).addEventListener("click", clearAll);
  $("#clear2", el).addEventListener("click", clearAll);
  const sheet = (on) => {
    $("#filters", el).classList.toggle("open", on);
    syncFilters();
    $(on ? "#close-filters" : "#open-filters", el).focus({ preventScroll: true });
  };
  syncFilters();
  el.addEventListener("keydown", (e) => {
    const drawer = $("#filters", el);
    if (!phone() || !drawer.classList.contains("open")) return;
    if (e.key === "Escape") { e.preventDefault(); sheet(false); }
    if (e.key === "Tab") {
      const items = [...drawer.querySelectorAll('button:not(:disabled), summary, input, select, a[href]')].filter((x) => x.getClientRects().length);
      const first = items[0], last = items.at(-1);
      if (e.shiftKey && (document.activeElement === first || !drawer.contains(document.activeElement))) { e.preventDefault(); last?.focus(); }
      else if (!e.shiftKey && (document.activeElement === last || !drawer.contains(document.activeElement))) { e.preventDefault(); first?.focus(); }
    }
  });
  $("#open-filters", el).addEventListener("click", () => sheet(true));
  $("#close-filters", el).addEventListener("click", () => sheet(false));
  $("#sheet-scrim", el).addEventListener("click", () => sheet(false));
  $("#show-results", el).addEventListener("click", () => { sheet(false); $("#toolbar", el).scrollIntoView({ block: "start" }); });
  $("#clear3", el).addEventListener("click", () => { state.sel = {}; state.coll = null; state.candidates = false; state.kind = "all"; run(); });
  $("#more", el).addEventListener("click", more);
  $("#random", el).addEventListener("click", async () => {
    // a task from the chosen collection's indexed datasets, each equally likely; else the server picks from all of them
    try {
      let pick = "";
      if (state.owner || (state.coll && COLL[state.coll] && state.coll !== "mine")) {
        const p = new URLSearchParams({ size: "100", facets: "0" });
        if (state.owner) p.set("owner", state.owner);
        if (state.coll) p.set("collection", state.coll);
        const d = await get(`/api/search?${p}`);
        const pool = d.rows.filter((x) => x.kind === "dataset" && x.indexed?.tasks);
        if (!pool.length && state.owner) { toast("No indexed dataset tasks are available for this owner yet."); return; }
        pick = pool.length ? pool[Math.floor(Math.random() * pool.length)].id : "";
      }
      const r = await api(`/api/random${pick ? `?d=${encodeURIComponent(pick)}` : ""}`);
      nav(`/t/${enc(r.dataset)}/${enc(r.path)}`);
    } catch (e) { toast(e.message); }
  });
  if (phone()) $("#q", el).placeholder = "Search environments…";
}

function syncFilters() {
  const drawer = root && $("#filters", root);
  if (!drawer) return;
  const mobile = phone();
  if (!mobile) drawer.classList.remove("open");
  const open = mobile && drawer.classList.contains("open");
  drawer.inert = mobile && !open;
  if (mobile && !open) drawer.setAttribute("aria-hidden", "true"); else drawer.removeAttribute("aria-hidden");
  if (open) { drawer.setAttribute("role", "dialog"); drawer.setAttribute("aria-modal", "true"); }
  else { drawer.removeAttribute("role"); drawer.removeAttribute("aria-modal"); }
  $("#open-filters", root).setAttribute("aria-expanded", String(open));
  $("#sheet-scrim", root).hidden = !open;
  document.body.classList.toggle("noscroll", open);
}

let resizeT;   // the page size follows the width: re-lay the trending grid when it changes
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => { syncFilters(); if (root && TREND && $("#trending", root)) renderTrending(); }, 150); });
