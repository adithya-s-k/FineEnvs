/* MiMo RL Environments Explorer. Plain JS: d3 (treemap) and MiniSearch (search) from the CDN. */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const fmt = new Intl.NumberFormat("en-US");
  const CJK = /[㐀-鿿豈-﫿]/;
  const PAGE = 40;
  const MAP_TOP = { code: 7, webdev: 9, cyber: 7, music: 9, general: 9 };   // blocks per domain in the overview
  const LEFTOVER = /^(Other|Unknown|unknown|None named|Unspecified|Unrated)$/;       // real buckets, but not categories

  const state = { q: "", dom: null, sel: {}, open: null };
  let DATA, DOMS, ENVS, SEARCH, matches = [], shown = 0, searchIds = null;
  const detailCache = {};

  // ── utilities ──────────────────────────────────────────────────────────────
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const vals = (v) => (Array.isArray(v) ? v : v == null ? [] : [v]);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const domColor = (d) => cssVar(`--c-${d}`);
  const storage = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: fine */ } },
  };

  function terms() {
    if (!state.q) return [];
    if (CJK.test(state.q)) return [state.q.trim()];
    return state.q.toLowerCase().split(/[^\p{L}\p{N}_+#.-]+/u).filter((t) => t.length > 1);
  }
  function highlight(text) {
    const html = esc(text);
    const ts = terms();
    if (!ts.length) return html;
    const re = new RegExp("(" + ts.map((t) => esc(t).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")", "gi");
    return html.replace(re, "<mark>$1</mark>");
  }

  // ── load ───────────────────────────────────────────────────────────────────
  // Data ships gzipped because Spaces serve static files uncompressed. If a server does decompress
  // it on the way (Content-Encoding), the bytes arrive plain, so check the gzip magic, not the name.
  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    if (buf[0] !== 0x1f || buf[1] !== 0x8b) return JSON.parse(new TextDecoder().decode(buf));
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).json();
  }

  async function boot() {
    initTheme();
    DATA = await getJSON("data/index.json.gz");
    DOMS = Object.fromEntries(DATA.domains.map((d) => [d.id, d]));
    ENVS = DATA.envs;
    ENVS.forEach((e, i) => { e._i = i; e._blob = (e.t + " " + e.s).toLowerCase(); });

    SEARCH = new MiniSearch({
      fields: ["t", "s", "x", "id"],
      storeFields: [],
      extractField: (doc, f) => (f === "x" ? Object.values(doc.f).flat().join(" ") + " " + DOMS[doc.d].name : doc[f]),
      searchOptions: { boost: { t: 3, x: 2 }, prefix: true, fuzzy: 0.15, combineWith: "AND" },
    });
    SEARCH.addAll(ENVS);

    $("#total").textContent = fmt.format(DATA.total);
    readURL();
    wire();
    run();
    if (state.open) openEnv(state.open, false);
  }

  // ── URL state ──────────────────────────────────────────────────────────────
  function readURL() {
    const p = new URLSearchParams(location.search);
    state.q = p.get("q") || "";
    state.dom = DOMS[p.get("d")] ? p.get("d") : null;
    state.sel = {};
    (p.get("f") || "").split(";").filter(Boolean).forEach((pair) => {
      const [k, v] = pair.split(":");
      if (k && v) state.sel[k] = new Set(v.split("|").map(decodeURIComponent));
    });
    state.open = p.get("env");
    $("#q").value = state.q;
  }
  function writeURL() {
    const p = new URLSearchParams();
    if (state.q) p.set("q", state.q);
    if (state.dom) p.set("d", state.dom);
    const f = Object.entries(state.sel).filter(([, s]) => s.size)
      .map(([k, s]) => k + ":" + [...s].map(encodeURIComponent).join("|")).join(";");
    if (f) p.set("f", f);
    if (state.open) p.set("env", state.open);
    const qs = p.toString();
    history.replaceState(null, "", qs ? "?" + qs : location.pathname);
  }

  // ── filtering ──────────────────────────────────────────────────────────────
  function facetsFor(dom) {
    return dom ? DOMS[dom].facets : [["d", "Domain"], ["language", "Brief language"]];
  }
  function valueOf(e, key) { return key === "d" ? DOMS[e.d].name : e.f[key]; }
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
    // facets from other domains would never match; drop them when the domain changes
    const allowed = new Set(facetsFor(state.dom).map(([k]) => k));
    Object.keys(state.sel).forEach((k) => { if (!allowed.has(k)) delete state.sel[k]; });

    matches = ENVS.filter((e) => passes(e));
    if (scores) matches.sort((a, b) => (scores.get(b.id) ?? 0) - (scores.get(a.id) ?? 0));
    renderDomains();
    renderFacets();
    renderActive();
    renderMap();
    renderList(true);
    writeURL();
  }

  // ── domain cards ───────────────────────────────────────────────────────────
  function renderDomains() {
    const box = $("#domains");
    const inSearch = (d) => (searchIds ? ENVS.filter((e) => (!d || e.d === d) && searchIds.has(e.id)).length : null);
    const cards = [{ id: null, name: "All domains", task: "Every environment in the release", verifier: "5 kinds", count: DATA.total }, ...DATA.domains];
    box.innerHTML = cards.map((d) => {
      const hit = inSearch(d.id);
      return `<button class="dom" role="tab" data-dom="${d.id ?? ""}" aria-selected="${state.dom === d.id}"
        style="--dc:${d.id ? `var(--c-${d.id})` : "var(--accent)"}">
        <div class="name"><span class="dot"></span>${esc(d.name)}</div>
        <div class="n">${fmt.format(hit ?? d.count)}${hit != null ? `<span style="font-size:12px;font-weight:500;color:var(--faint)"> / ${fmt.format(d.count)}</span>` : ""}</div>
        <div class="task">${esc(d.task)}</div>
        <div class="ver">Checked by <b>${esc(d.verifier)}</b></div>
      </button>`;
    }).join("");
  }

  // ── facets ─────────────────────────────────────────────────────────────────
  const expanded = new Set();
  function renderFacets() {
    const html = facetsFor(state.dom).map(([key, label]) => {
      const counts = new Map();
      ENVS.forEach((e) => {
        if (!passes(e, key)) return;
        vals(valueOf(e, key)).forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
      });
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
        ${hidden || expanded.has(key) ? `<button class="link more" data-more="${esc(key)}">${expanded.has(key) ? "Show fewer" : `Show ${hidden} more`}</button>` : ""}
      </div>`;
    }).join("");
    $("#facets").innerHTML = html;
  }
  function toggle(key, v) {
    if (key === "d") {   // choosing a domain in the "All" view means switching to it
      const id = DATA.domains.find((d) => d.name === v)?.id;
      if (id) { setDomain(id); return; }
    }
    const set = (state.sel[key] ||= new Set());
    set.has(v) ? set.delete(v) : set.add(v);
    if (!set.size) delete state.sel[key];
    run();
  }
  function setDomain(id) {
    state.dom = id;
    state.sel = {};
    run();
  }
  function renderActive() {
    const labels = Object.fromEntries(facetsFor(state.dom));
    const pills = [];
    Object.entries(state.sel).forEach(([k, set]) => set.forEach((v) =>
      pills.push(`<button class="pill" data-k="${esc(k)}" data-v="${esc(v)}"><span>${esc(labels[k] || k)}</span>${esc(v)}<i aria-hidden="true">×</i></button>`)));
    $("#active").innerHTML = pills.join("");
    $("#count").textContent = `${fmt.format(matches.length)} environment${matches.length === 1 ? "" : "s"}`;
  }

  // ── map ────────────────────────────────────────────────────────────────────
  function mapData() {
    // The map ignores its own facet's selection, so a picked block stays visible (and highlighted).
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

  function renderMap() {
    const el = $("#map");
    const W = el.clientWidth, H = el.clientHeight;
    $("#map-title").textContent = state.dom ? `${DOMS[state.dom].name}, by ${DOMS[state.dom].facets.find(([k]) => k === DOMS[state.dom].main)[1].replace(" (mentioned)", "").toLowerCase()}` : "Everything, by domain and category";
    const data = mapData();
    el.innerHTML = "";
    if (!data.children.length) { el.innerHTML = `<p class="empty">Nothing to map.</p>`; return; }

    // squarify lays out in order, so leftovers sort last and land bottom-right, out of the way
    const root = d3.hierarchy(data).sum((d) => d.value || 0)
      .sort((a, b) => (!!a.data.other - !!b.data.other) || b.value - a.value);
    d3.treemap().size([W, H]).paddingInner(2).paddingTop((n) => (n.depth === 1 && !state.dom ? 20 : 0)).round(true)
      .tile(d3.treemapSquarify.ratio(1.35))(root);

    const surface = cssVar("--surface");
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img")
      .attr("aria-label", "Treemap of environments by category");

    if (!state.dom) {
      svg.selectAll("g.dh").data(root.children).join("g").attr("class", "cell dh")
        .attr("transform", (d) => `translate(${d.x0},${d.y0})`)
        .call((g) => {
          g.append("rect").attr("width", (d) => d.x1 - d.x0).attr("height", 18).attr("fill", "transparent")
            .on("click", (_, d) => setDomain(d.data.dom));
          g.append("text").attr("class", "dname").attr("x", 1).attr("y", 13).attr("fill", (d) => domColor(d.data.dom))
            .text((d) => (d.x1 - d.x0 > 70 ? `${d.data.name} · ${fmt.format(d.value)}` : d.x1 - d.x0 > 34 ? d.data.name : ""));
        });
    }

    const leaves = root.leaves();
    const maxRank = d3.max(leaves, (d) => d.data.rank) || 1;
    const sel = state.dom ? state.sel[DOMS[state.dom].main] : null;
    const g = svg.selectAll("g.leaf").data(leaves).join("g")
      .attr("class", (d) => "cell leaf" + (sel && sel.size && !sel.has(d.data.name) ? " dim" : ""))
      .attr("transform", (d) => `translate(${d.x0},${d.y0})`);

    g.append("rect")
      .attr("width", (d) => Math.max(0, d.x1 - d.x0)).attr("height", (d) => Math.max(0, d.y1 - d.y0)).attr("rx", 4)
      .attr("fill", (d) => {
        const t = d.data.other ? 0.28 : 0.95 - 0.55 * (d.data.rank / Math.max(maxRank, 6));
        return d3.interpolateRgb(surface, domColor(d.data.dom))(t);
      })
      .on("click", (_, d) => clickCell(d.data))
      .on("mousemove", (ev, d) => tip(ev, `<b>${esc(d.data.name)}</b><br>${fmt.format(d.value)} environments · ${DOMS[d.data.dom].name}`))
      .on("mouseleave", () => tip());

    g.each(function (d) {
      const w = d.x1 - d.x0, h = d.y1 - d.y0;
      if (w < 44 || h < 26) return;
      const rect = this.querySelector("rect");
      const light = d3.lab(rect.getAttribute("fill")).l > 64;
      const color = light ? "#17181c" : "#ffffff";
      const t = d3.select(this);
      const maxChars = Math.floor((w - 14) / 6.6);
      const label = d.data.name.length > maxChars ? d.data.name.slice(0, Math.max(1, maxChars - 1)) + "…" : d.data.name;
      t.append("text").attr("class", "lbl").attr("x", 8).attr("y", 18).attr("fill", color).text(label);
      if (h > 42) t.append("text").attr("class", "num").attr("x", 8).attr("y", 34).attr("fill", color).text(fmt.format(d.value));
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
    $("#toolbar").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function tip(ev, html) {
    const el = $("#tip");
    if (!ev) { el.hidden = true; return; }
    el.innerHTML = html;
    el.hidden = false;
    const x = Math.min(ev.clientX + 14, innerWidth - el.offsetWidth - 8);
    el.style.left = x + "px";
    el.style.top = ev.clientY + 16 + "px";
  }

  // ── list ───────────────────────────────────────────────────────────────────
  function card(e) {
    const d = DOMS[e.d];
    const main = vals(e.f[d.main]).join(", ");
    const chips = d.facets.filter(([k]) => k !== d.main && k !== "source")
      .flatMap(([k]) => vals(e.f[k]).filter((v) => !/^(None named|Unspecified|Unknown|Unrated)$/.test(v)).slice(0, 2))
      .slice(0, 5);
    const stats = e.n ? [`${e.n.systems} systems`, `${e.n.files} files`, `${e.n.checks} checks`] : [];
    return `<li><button class="card" data-id="${esc(e.id)}" style="--dc:var(--c-${e.d})">
      <div class="row1"><span class="dot"></span><span class="dn">${esc(d.name)}</span><span class="sep">/</span><span>${esc(main)}</span></div>
      <p class="t">${highlight(e.t)}</p>
      <p class="s">${highlight(e.s)}</p>
      ${chips.length || stats.length ? `<div class="chips">${chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}${stats.map((s) => `<span class="chip stat">${s}</span>`).join("")}</div>` : ""}
    </button></li>`;
  }
  function renderList(reset) {
    const list = $("#list");
    if (reset) { list.innerHTML = ""; shown = 0; }
    const next = matches.slice(shown, shown + PAGE);
    list.insertAdjacentHTML("beforeend", next.map(card).join(""));
    shown += next.length;
    $("#empty").hidden = matches.length > 0;
  }

  // ── detail drawer ──────────────────────────────────────────────────────────
  function detailFor(dom) {
    return (detailCache[dom] ||= getJSON(`data/${dom}.json.gz`));
  }
  async function openEnv(id, push = true) {
    const e = ENVS.find((x) => x.id === id);
    if (!e) { state.open = null; writeURL(); return; }
    state.open = id;
    if (push) writeURL();
    const d = DOMS[e.d];
    const drawer = $("#drawer");
    drawer.style.setProperty("--dc", `var(--c-${e.d})`);
    $("#d-domain").innerHTML = `<span class="dot" style="width:7px;height:7px;border-radius:50%;background:var(--dc)"></span>${esc(d.name)}`;
    $("#d-id").textContent = e.id;
    $("#d-body").innerHTML = `<h2>${esc(e.t)}</h2><p class="loading">Loading…</p>`;
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    $("#scrim").hidden = false;
    $("#d-close").focus({ preventScroll: true });

    const all = await detailFor(e.d);
    if (state.open !== id) return;
    const x = all[id] || {};
    const parts = [`<h2>${esc(e.t)}</h2>`];
    if (x.meta) parts.push(`<dl class="meta">${x.meta.filter(([, v]) => v).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`);
    parts.push(section("The task", "", `<div class="brief">${md(x.brief || e.s)}</div>`));
    if (x.extra) parts.push(section("Crash report", "", `<div class="brief"><pre><code>${esc(x.extra)}</code></pre></div>`));
    if (x.systems?.length) parts.push(section("Systems the agent works through", `${x.systems.length} MCP servers, each backed by its own database`,
      `<div class="tags">${x.systems.map((s) => `<span class="tag">${esc(s)}</span>`).join("")}</div>`));
    if (x.files && Object.keys(x.files).length) {
      const n = Object.values(x.files).reduce((s, v) => s + v.length, 0);
      parts.push(section(e.d === "general" && x.systems ? "Workspace" : "Files", `${n} file${n === 1 ? "" : "s"}`,
        `<div class="files">${Object.entries(x.files).map(([k, fs]) => `<div class="fgroup"><h4>${esc(k)} <span>${fs.length}</span></h4>
          <ul class="flist">${fs.slice(0, 14).map((f) => `<li>${esc(f)}</li>`).join("")}${fs.length > 14 ? `<li>… ${fs.length - 14} more</li>` : ""}</ul></div>`).join("")}</div>`));
    }
    if (x.rubric?.length) {
      const total = x.rubric.reduce((s, r) => s + (r.weight || 0), 0) || 1;
      parts.push(section("How it is graded", `${x.rubric.length} checks · answers not shown`,
        `<ol class="rubric">${x.rubric.map((r) => `<li><div class="rtop"><span class="tier ${esc(r.tier)}">${esc(r.tier)}</span><span>${r.method === "llm" ? "LLM judge" : "Rule"}</span>
          <span class="weight"><span class="track"><span class="bar" style="display:block;width:${Math.round((r.weight / total) * 100)}%"></span></span>${Math.round((r.weight / total) * 100)}%</span></div>${esc(r.q)}</li>`).join("")}</ol>`));
    }
    const links = [`<a class="out" href="https://huggingface.co/datasets/${DATA.source}/viewer/${e.d}" target="_blank" rel="noopener">Open ${esc(d.name)} in the dataset viewer ↗</a>`];
    if (x.tree) links.unshift(`<a class="out" href="${esc(x.tree)}" target="_blank" rel="noopener">Browse this environment's files on the Hub ↗</a>`);
    parts.push(`<div class="sec" style="display:grid;gap:8px">${links.join("")}</div>`);
    $("#d-body").innerHTML = parts.join("");
    $("#d-body").scrollTop = 0;
  }
  function section(title, note, body) {
    return `<div class="sec"><h3>${esc(title)}${note ? `<span class="k">${esc(note)}</span>` : ""}</h3>${body}</div>`;
  }
  function closeEnv() {
    state.open = null;
    writeURL();
    $("#drawer").classList.remove("open");
    $("#drawer").setAttribute("aria-hidden", "true");
    $("#scrim").hidden = true;
  }

  // A small markdown renderer: escape first, then headings, lists, fences, inline code and bold.
  function md(src) {
    const out = [];
    const lines = String(src).replace(/\r/g, "").split("\n");
    let i = 0, para = [], list = null;
    const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    const flush = () => {
      if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
      if (list) { out.push(`<${list.tag}>${list.items.map((x) => `<li>${inline(x)}</li>`).join("")}</${list.tag}>`); list = null; }
    };
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*```/.test(line)) {
        flush();
        const buf = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
        out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
        i++;
        continue;
      }
      const h = line.match(/^\s*(#{1,4})\s+(.*)$/);
      const ul = line.match(/^\s*[-*•]\s+(.*)$/);
      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (h) { flush(); out.push(`<h4>${inline(h[2])}</h4>`); }
      else if (ul || ol) {
        if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
        const tag = ul ? "ul" : "ol";
        if (!list || list.tag !== tag) { if (list) flush(); list = { tag, items: [] }; }
        list.items.push((ul || ol)[1]);
      } else if (!line.trim()) flush();
      else { if (list) flush(); para.push(line.trim()); }
      i++;
    }
    flush();
    return out.join("");
  }

  // ── theme ──────────────────────────────────────────────────────────────────
  function initTheme() {
    const saved = storage.get("theme");
    if (saved) document.documentElement.dataset.theme = saved;
  }
  function flipTheme() {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    storage.set("theme", document.documentElement.dataset.theme);
    renderMap();
  }

  // ── events ─────────────────────────────────────────────────────────────────
  function wire() {
    let t;
    $("#q").addEventListener("input", (ev) => { clearTimeout(t); t = setTimeout(() => { state.q = ev.target.value; run(); }, 120); });
    $("#domains").addEventListener("click", (ev) => {
      const b = ev.target.closest(".dom");
      if (b) setDomain(b.dataset.dom || null);
    });
    $("#facets").addEventListener("click", (ev) => {
      const o = ev.target.closest(".opt");
      if (o) return toggle(o.dataset.k, o.dataset.v);
      const m = ev.target.closest("[data-more]");
      if (m) { expanded.has(m.dataset.more) ? expanded.delete(m.dataset.more) : expanded.add(m.dataset.more); renderFacets(); }
    });
    $("#active").addEventListener("click", (ev) => {
      const p = ev.target.closest(".pill");
      if (p) toggle(p.dataset.k, p.dataset.v);
    });
    $("#clear").addEventListener("click", () => { state.sel = {}; state.q = ""; $("#q").value = ""; state.dom = null; run(); });
    $("#list").addEventListener("click", (ev) => {
      const c = ev.target.closest(".card");
      if (c) openEnv(c.dataset.id);
    });
    $("#d-close").addEventListener("click", closeEnv);
    $("#scrim").addEventListener("click", closeEnv);
    $("#d-copy").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(location.href); } catch { /* clipboard blocked: the URL bar has it */ }
      const toast = $("#toast");
      toast.hidden = false;
      setTimeout(() => (toast.hidden = true), 1400);
    });
    $("#theme").addEventListener("click", flipTheme);
    $("#open-filters").addEventListener("click", () => $("#filters").classList.toggle("open"));
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "/" && document.activeElement !== $("#q")) { ev.preventDefault(); $("#q").focus(); }
      if (ev.key === "Escape") { if (state.open) closeEnv(); else $("#filters").classList.remove("open"); }
    });
    new IntersectionObserver((es) => { if (es.some((x) => x.isIntersecting) && shown < matches.length) renderList(false); },
      { rootMargin: "600px" }).observe($("#sentinel"));
    let rt;
    addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderMap, 150); });
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", renderMap);
    addEventListener("popstate", () => { readURL(); run(); });
  }

  boot().catch((err) => {
    console.error(err);
    $("#list").innerHTML = `<li class="empty">Could not load the data (${esc(err.message)}).</li>`;
  });
})();
