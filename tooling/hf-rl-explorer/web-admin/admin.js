// The admin dashboard, on its own private Space, for FineEnvs members: what's happening (rollouts, people, new
// environments), and the controls (what trends at the top, what's hidden, the collections, indexing, the rollout
// switches). The server checks membership on every call. Links to rollouts, datasets and Spaces open the explorer.
import { $, api, esc, fmt, ago, dur, sk, emptyState, toast, statusPill, rewardBadge, spinner } from "/js/util.js";
import { icon } from "/js/icons.js";
import { getSession } from "/js/session.js";

const TABS = [["overview", "Overview", "grid"], ["rollouts", "Rollouts", "play"], ["people", "People", "users"], ["environments", "Environments", "database"],
              ["collections", "Collections", "layout"], ["indexes", "Indexes", "archive"], ["settings", "Settings", "gauge"], ["audit", "Audit log", "list"]];
const enc = (s) => s.split("/").map(encodeURIComponent).join("/");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const bytes = (n) => (n == null ? "–" : n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} KB`);
let EXPLORER = "";   // the public explorer: datasets, Spaces and rollouts open there
const envHref = (e) => `${EXPLORER}/${e.kind === "space" ? "s" : "d"}/${enc(e.id)}`;
const runHref = (id) => `${EXPLORER}/run/${encodeURIComponent(id)}`;
const thumb = (id) => `https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/${enc(id)}.png`;
let alive = true, timer = null, counts = {};
export function unmount() { alive = false; clearTimeout(timer); }

// a person: their Hub picture (or their initial), their name, a link to their profile
const pfp = (name, src, size = 22) => (src ? `<img class="pfp" src="${esc(src)}" alt="" width="${size}" height="${size}" loading="lazy">`
  : `<span class="pfp none" style="width:${size}px;height:${size}px">${esc((name || "?").slice(0, 1).toUpperCase())}</span>`);
const person = (name, src, { link = true, filter = false } = {}) => `<span class="person">${pfp(name, src)}${filter
  ? `<button class="link" data-user="${esc(name)}" title="Only ${esc(name)}'s rollouts">${esc(name)}</button>`
  : link ? `<a href="https://huggingface.co/${encodeURIComponent(name)}" target="_blank" rel="noopener">${esc(name)}</a>` : `<b>${esc(name)}</b>`}</span>`;

export async function mount(el, { qs, explorer }) {
  alive = true;
  EXPLORER = explorer || "";
  const s = getSession();
  if (!s.user || !s.is_admin) {
    el.innerHTML = `<div class="wrap page">${emptyState("shield", "Admins only", s.user ? `The admin dashboard is for members of the ${esc(s.admin_org)} organization on Hugging Face.`
      : `Sign in with a ${esc(s.admin_org || "FineEnvs")} account.`, s.user ? "" : `<a class="btn primary" href="/login">${icon("user", 15)}Sign in with Hugging Face</a>`)}</div>`;
    return;
  }
  let tab = TABS.some(([k]) => k === qs.get("tab")) ? qs.get("tab") : "overview";
  const preset = { user: qs.get("user") || "" };
  el.innerHTML = `<div class="wrap ad-shell fade-in">
    <nav class="ad-nav" id="ad-nav" aria-label="Admin sections"></nav>
    <section class="ad-main"><div id="ad-body"></div></section></div>`;
  const nav = () => {
    $("#ad-nav", el).innerHTML = TABS.map(([k, l, ic]) => `<a href="#/${k === "overview" ? "" : `?tab=${k}`}" data-tab="${k}" class="${k === tab ? "on" : ""}">
      ${icon(ic, 15)}<span>${l}</span>${counts[k] ? `<em class="${k === "rollouts" ? "live" : ""}">${counts[k]}</em>` : ""}</a>`).join("");
  };
  const show = () => {
    clearTimeout(timer);
    nav();
    history.replaceState(null, "", `#/${tab === "overview" ? "" : `?tab=${tab}${tab === "rollouts" && preset.user ? `&user=${encodeURIComponent(preset.user)}` : ""}`}`);
    const body = $("#ad-body", el);
    body.innerHTML = `<div class="panel"><div class="panel-b">${sk.lines(80, 60, 90, 70)}</div></div>`;
    ({ overview, rollouts, people, environments, collections, indexes, settings, audit })[tab](body, { go, preset, s }).catch((e) => {
      if (alive) body.innerHTML = `<div class="note-box err">${icon("alert")}<span>${esc(e.message)}</span></div>`;
    });
    window.scrollTo(0, 0);
  };
  const go = (t, opts = {}) => { tab = t; Object.assign(preset, opts); show(); };
  el.addEventListener("click", (e) => {
    const t = e.target.closest("[data-tab]");
    if (t) { e.preventDefault(); preset.user = ""; go(t.dataset.tab); }
    const u = e.target.closest("[data-user]");
    if (u) { e.preventDefault(); go("rollouts", { user: u.dataset.user }); }
  });
  show();
  // live counts beside the sections: running rollouts, new environments
  api("/api/admin/overview").then((o) => { counts = { rollouts: o.rollouts.live || "", environments: o.new.length || "" }; if (alive) nav(); }).catch(() => {});
}

const head = (title, sub, actions = "") => `<div class="ad-head"><div><h1>${title}</h1>${sub ? `<p>${sub}</p>` : ""}</div><div class="ad-actions">${actions}</div></div>`;
const stat = (ic, n, label, sub = "", tone = "") => `<div class="ad-stat ${tone}"><span class="ic-wrap">${icon(ic, 16)}</span><div><b>${n}</b><span>${label}</span>${sub ? `<em>${sub}</em>` : ""}</div></div>`;
const runRow = (r, actions = false) => `<div class="ad-run${r.live ? " is-live" : ""}">
  <span>${statusPill(r.status)}</span>
  <a class="ad-task" href="${runHref(r.id)}" target="_blank" rel="noopener"><b>${esc(r.title)}</b><em>${esc(r.dataset)}</em></a>
  ${person(r.user, r.avatar, { filter: true })}
  <span class="ad-model">${esc((r.model || "").split("/")[1] || r.model || "endpoint")}<em>${esc(r.harness)}</em></span>
  <span>${rewardBadge(r.reward, r.status)}</span>
  <span class="ad-when">${ago(r.created_at)}<em>${dur(r.wall_s || ((r.finished_at || Date.now() / 1000) - (r.started_at || r.created_at)))}</em></span>
  ${actions ? `<span class="ad-act">${r.live ? `<button class="btn sm danger" data-cancel="${esc(r.id)}">${icon("stop", 12)}Stop</button>` : ""}
    ${r.moderated ? `<button class="btn sm" data-unhide="${esc(r.id)}">Back on Community</button>` : r.visibility === "public" ? `<button class="btn sm ghost" data-hide="${esc(r.id)}">Hide from Community</button>` : ""}</span>` : ""}</div>`;

async function runActions(body, reload) {
  body.addEventListener("click", async (e) => {
    const c = e.target.closest("[data-cancel]"), h = e.target.closest("[data-hide]"), u = e.target.closest("[data-unhide]");
    try {
      if (c) { if (!confirm("Stop this rollout? The sandbox shuts down and it isn't graded.")) return;
        await api(`/api/admin/rollouts/${c.dataset.cancel}/cancel`, { method: "POST" }); toast("Asked the explorer to stop it: a few seconds"); }
      else if (h) { if (!confirm("Take this rollout off Community? Its owner still sees it.")) return;
        await api(`/api/admin/rollouts/${h.dataset.hide}/moderate`, { method: "POST", body: { hidden: true } }); toast("Hidden from Community"); }
      else if (u) { await api(`/api/admin/rollouts/${u.dataset.unhide}/moderate`, { method: "POST", body: { hidden: false } }); toast("Back on Community"); }
      else return;
      reload();
    } catch (err) { toast(err.message); }
  });
}

// ── overview ─────────────────────────────────────────────────────────────────
async function overview(body, { go, s }) {
  const o = await api("/api/admin/overview");
  if (!alive) return;
  const r = o.rollouts, e = o.environments, on = o.settings.rollouts_enabled;
  const peak = Math.max(1, ...r.by_day.map(([, n]) => n));
  const live = o.recent.filter((x) => x.live);
  const last = o.recent[0];
  body.innerHTML = `
    <div class="ad-hello">${pfp(s.user.name, s.user.avatar, 44)}<div><h1>Hi, ${esc(s.user.name)}</h1>
      <p><span class="dot" style="--dc:${on ? "var(--ok)" : "var(--warn)"}"></span> Rollouts are ${on ? "on" : "<b>paused</b>"} · ${fmt.format(r.live)} running now${last ? ` · last one ${ago(last.created_at)}` : ""}</p></div>
      <div class="ad-actions"><button class="btn ${on ? "" : "primary"}" id="ov-toggle" type="button">${icon(on ? "stop" : "play", 14)}${on ? "Pause rollouts" : "Resume rollouts"}</button>
        <button class="btn" type="button" data-go="settings">${icon("message", 14)}Banner</button><button class="btn" type="button" data-go="indexes">${icon("archive", 14)}Index a dataset</button></div></div>
    <div class="ad-stats">
      ${stat("play", fmt.format(r.live), "running now", `${fmt.format(r.today)} started today`, r.live ? "live" : "")}
      ${stat("list", fmt.format(r.total), "rollouts in all", `${fmt.format(r.week)} this week`)}
      ${stat("users", fmt.format(r.users), "people", "who ran one")}
      ${stat("alert", r.finished ? `${Math.round((100 * r.failed) / r.finished)}%` : "–", "failed or stopped", `of ${fmt.format(r.finished)} finished`, r.finished && r.failed / r.finished > 0.3 ? "warn" : "")}
      ${stat("database", fmt.format(e.datasets), "datasets", `${fmt.format(e.indexed)} Harbor indexed`)}
      ${stat("globe", fmt.format(e.spaces), "environment Spaces", `${fmt.format(e.openenv)} OpenEnv`)}
    </div>
    ${live.length ? `<section class="panel ad-live"><div class="panel-h"><h2><i class="pulse" style="color:var(--live)"></i>Running now</h2><a class="aside" href="#" data-go="rollouts">All rollouts</a></div>
      <div class="panel-b ad-runs">${live.map((x) => runRow(x, true)).join("")}</div></section>` : ""}
    <div class="ad-grid">
      <section class="panel"><div class="panel-h"><h2>${icon("play", 15)}Rollouts, last 14 days</h2><span class="aside">${fmt.format(r.week)} this week</span></div>
        <div class="panel-b"><div class="ad-bars">${r.by_day.map(([d, n]) => `<div title="${esc(d)}: ${n} rollout${n === 1 ? "" : "s"}"><i style="height:${Math.max(2, (100 * n) / peak)}%"></i><span>${d.slice(8)}</span></div>`).join("")}</div>
        ${r.by_model.length ? `<div class="sec-label" style="margin-top:16px">Most-used models</div><ul class="ad-models">${r.by_model.map(([m, n]) => `<li><code>${esc(m)}</code>
          <span class="share"><i style="width:${(100 * n) / r.by_model[0][1]}%"></i></span><b>${fmt.format(n)}</b></li>`).join("")}</ul>` : ""}</div></section>
      <section class="panel"><div class="panel-h"><h2>${icon("flame", 15)}New on the Hub</h2><a class="aside" href="#" data-go="environments">Curate</a></div>
        <div class="panel-b">${o.new.length ? `<ul class="ad-new">${o.new.slice(0, 9).map((x) => `<li>
          ${x.kind === "space" ? `<img class="ad-th" src="${thumb(x.id)}" alt="" loading="lazy">` : `<span class="ad-th ic-only">${icon("database", 14)}</span>`}
          <a href="${envHref(x)}" target="_blank" rel="noopener"><code>${esc(x.id)}</code><em>${x.kind === "space" ? "Space" : "Dataset"} · ${ago(Date.parse(x.created) / 1000)}${x.collection ? ` · in ${esc(x.collection)}` : ""}</em></a></li>`).join("")}</ul>`
          : `<p class="muted sm">Nothing new in the last two weeks.</p>`}</div></section>
    </div>
    <section class="panel" style="margin-top:16px"><div class="panel-h"><h2>${icon("clock", 15)}Latest rollouts</h2><a class="aside" href="#" data-go="rollouts">All rollouts</a></div>
      <div class="panel-b ad-runs">${o.recent.length ? o.recent.map((x) => runRow(x)).join("") : emptyState("play", "No rollouts yet", "They show up here as soon as someone runs one.")}</div></section>`;
  body.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", (ev) => { ev.preventDefault(); go(b.dataset.go); }));
  $("#ov-toggle", body).addEventListener("click", async () => {
    if (on && !confirm("Pause rollouts for everyone? Running ones finish; new ones are refused until you resume.")) return;
    try { await api("/api/admin/settings", { method: "PUT", body: { ...pick(o.settings), rollouts_enabled: !on } }); toast(on ? "Rollouts paused" : "Rollouts resumed"); overview(body, { go, s }); }
    catch (err) { toast(err.message); }
  });
  runActions(body, () => overview(body, { go, s }));
}
const pick = (st) => ({ rollouts_enabled: st.rollouts_enabled, max_active: st.max_active, max_per_user: st.max_per_user, agents: st.agents, announcement: st.announcement });

// ── rollouts ─────────────────────────────────────────────────────────────────
async function rollouts(body, { preset }) {
  const st = { status: "", q: "", user: preset.user || "" };
  body.innerHTML = `${head("Rollouts", "Everyone's, private ones too, with who ran them. Click a person to see only theirs.")}
    <div class="ad-tools"><div class="search">${icon("search", 16)}<input id="ad-q" type="search" placeholder="Search task, dataset, model, person, id…"></div>
      <div class="seg" id="ad-st">${[["", "All"], ["live", "Running"], ["done", "Done"], ["failed", "Failed"], ["cancelled", "Stopped"], ["interrupted", "Interrupted"]].map(([k, l]) => `<button type="button" data-st="${k}" aria-pressed="${k === ""}">${l}</button>`).join("")}</div>
      <span id="ad-who"></span></div>
    <div class="panel"><div class="panel-b ad-runs" id="ad-runs">${sk.lines(80, 70, 90)}</div></div>`;
  const load = async () => {
    const d = await api(`/api/admin/rollouts?status=${st.status}&q=${encodeURIComponent(st.q)}&user=${encodeURIComponent(st.user)}&limit=200`);
    if (!alive) return;
    $("#ad-who", body).innerHTML = st.user ? `<button class="pill" id="ad-clear-user" type="button"><span>by</span>${esc(st.user)}${icon("x", 13)}</button>` : "";
    $("#ad-runs", body).innerHTML = d.runs.length ? `<p class="muted sm" style="margin-bottom:6px">${fmt.format(d.total)} rollout${d.total === 1 ? "" : "s"}</p>${d.runs.map((r) => runRow(r, true)).join("")}`
      : emptyState("filter", "No rollouts here", "Try another filter.");
    clearTimeout(timer);
    if (d.runs.some((r) => r.live)) timer = setTimeout(load, 4000);
  };
  let t;
  $("#ad-q", body).addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { st.q = e.target.value.trim(); load(); }, 200); });
  $("#ad-st", body).addEventListener("click", (e) => { const b = e.target.closest("[data-st]"); if (!b) return; st.status = b.dataset.st;
    body.querySelectorAll("#ad-st [data-st]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); load(); });
  body.addEventListener("click", (e) => { if (e.target.closest("#ad-clear-user")) { st.user = ""; preset.user = ""; load(); } });
  runActions(body, load);
  await load();
}

// ── people ───────────────────────────────────────────────────────────────────
async function people(body, { go }) {
  const d = await api("/api/admin/people");
  if (!alive) return;
  body.innerHTML = `${head("People", "Everyone who has run a rollout, most recent first.")}
    ${d.people.length ? `<div class="ad-people">${d.people.map((p) => `<article class="panel ad-person">
      <header>${pfp(p.name, p.avatar, 40)}<div><a href="https://huggingface.co/${encodeURIComponent(p.name)}" target="_blank" rel="noopener"><b>${esc(p.name)}</b> ${icon("external", 12)}</a>
        <em>active ${ago(p.last)}${p.live ? ` · <span class="live-txt">${p.live} running</span>` : ""}</em></div></header>
      <div class="ad-pstats"><div><b>${fmt.format(p.runs)}</b><span>rollouts</span></div><div><b>${fmt.format(p.solved)}</b><span>full marks</span></div>
        <div><b>${p.runs ? `${Math.round((100 * p.done) / p.runs)}%` : "–"}</b><span>finished</span></div><div><b>${fmt.format(p.public)}</b><span>public</span></div></div>
      <dl class="kv sm"><dt>Models</dt><dd>${p.models.map(([m, n]) => `<code>${esc(m.split("/")[1] || m)}</code> <span class="faint">${n}</span>`).join(", ")}</dd>
        <dt>Datasets</dt><dd>${p.datasets.map(([x, n]) => `${esc(x)} <span class="faint">${n}</span>`).join(", ")}</dd>
        <dt>Since</dt><dd>${new Date(p.first * 1000).toLocaleDateString()}</dd></dl>
      <button class="btn sm" type="button" data-user="${esc(p.name)}">${icon("play", 12)}Their rollouts</button></article>`).join("")}</div>`
      : `<div class="panel">${emptyState("users", "Nobody yet", "People show up here after their first rollout.")}</div>`}`;
}

// ── environments: what trends, what's hidden, what's featured ────────────────
async function environments(body) {
  const d = await api("/api/admin/environments");
  if (!alive) return;
  const st = { q: "", kind: "", flag: "", shown: 60 };
  const cols = d.collections;
  const now = Date.now();
  const isNew = (x) => x.created && now - Date.parse(x.created) < 14 * 864e5;
  const n = (f) => d.environments.filter(f).length;
  body.innerHTML = `${head("Environments", "Pin to lead the trending row on Explore; hide to take off every listing (the page still opens by link); put in a collection to feature it.")}
    <div class="ad-tools"><div class="search">${icon("search", 16)}<input id="ad-q" type="search" placeholder="Search environments…"></div>
      <div class="seg" id="ad-flag">${[["", "All", d.environments.length], ["new", "New", n(isNew)], ["pinned", "Pinned", n((x) => x.pinned)], ["hidden", "Hidden", n((x) => x.hidden)],
        ["loose", "In no collection", n((x) => !x.collection)], ["unindexed", "Not indexed", n((x) => x.kind === "dataset" && !x.indexed)]]
        .map(([k, l, c]) => `<button type="button" data-flag="${k}" aria-pressed="${k === ""}">${l}<span>${fmt.format(c)}</span></button>`).join("")}</div>
      <select class="hf-select-mimo" id="ad-kind"><option value="">Datasets and Spaces</option><option value="dataset">Datasets</option><option value="space">Spaces</option></select></div>
    <div class="panel"><div class="ad-envs" id="ad-envs"></div></div>
    <div class="more-row"><button class="btn" id="ad-more" type="button">Show more</button></div>`;
  const draw = () => {
    const q = st.q.toLowerCase();
    const rows = d.environments.filter((x) => (!st.kind || x.kind === st.kind) && (!q || `${x.id} ${x.heading || ""} ${x.tags.join(" ")}`.toLowerCase().includes(q))
      && (!st.flag || (st.flag === "new" ? isNew(x) : st.flag === "pinned" ? x.pinned : st.flag === "hidden" ? x.hidden
        : st.flag === "loose" ? !x.collection : x.kind === "dataset" && !x.indexed)))
      .sort((a, b) => (b.pinned - a.pinned) || (b.trending * 1e9 + b.likes * 1e4 + Math.min(b.downloads, 9999)) - (a.trending * 1e9 + a.likes * 1e4 + Math.min(a.downloads, 9999)));
    $("#ad-envs", body).innerHTML = rows.length ? `<div class="ad-env head"><span>Environment</span><span>Collection</span><span class="num">Trending</span><span class="num">Downloads</span>
      <span class="num">Tasks</span><span class="num">Rollouts</span><span></span></div>` + rows.slice(0, st.shown).map((x) => `<div class="ad-env${x.hidden ? " gone" : ""}">
      <span class="ad-name">${x.kind === "space" ? `<img class="ad-th" src="${thumb(x.id)}" alt="" loading="lazy">` : `<span class="ad-th ic-only">${icon("database", 14)}</span>`}
        <a href="${envHref(x)}" target="_blank" rel="noopener"><code>${esc(x.id)}</code></a>
        ${x.pinned ? `<span class="chip">${icon("flag", 11)}Pinned</span>` : ""}${x.hidden ? `<span class="chip">Hidden</span>` : ""}${isNew(x) ? `<span class="chip new">New</span>` : ""}</span>
      <span><select class="hf-select-mimo sm" data-feature="${esc(x.key)}" aria-label="Collection"><option value="">No collection</option>${cols.map((c) => `<option value="${c.id}" ${x.collection === c.id ? "selected" : ""}>${esc(c.group)}</option>`).join("")}</select></span>
      <span class="num">${fmt.format(x.trending || 0)}</span><span class="num">${x.kind === "dataset" ? compact.format(x.downloads) : `${compact.format(x.likes)} ♥`}</span>
      <span class="num">${x.indexed?.tasks != null ? fmt.format(x.indexed.tasks) : x.kind === "dataset" ? `<button class="link" data-index="${esc(x.key)}">Index</button>` : "–"}</span>
      <span class="num">${fmt.format(x.rollouts || 0)}</span>
      <span class="ad-act"><button class="btn sm ${x.pinned ? "" : "ghost"}" data-act="${x.pinned ? "unpin" : "pin"}" data-key="${esc(x.key)}" title="${x.pinned ? "Unpin" : "Pin to the top of trending"}">${icon("flag", 12)}${x.pinned ? "Unpin" : "Pin"}</button>
        <button class="btn sm ${x.hidden ? "" : "ghost"}" data-act="${x.hidden ? "unhide" : "hide"}" data-key="${esc(x.key)}">${x.hidden ? "Unhide" : "Hide"}</button></span></div>`).join("")
      : emptyState("search", "Nothing here", "Try another filter.");
    $("#ad-more", body).hidden = rows.length <= st.shown;
  };
  const act = async (key, action, extra = {}) => {
    if (action === "hide" && !confirm(`Hide ${key.replace(/^space:/, "")} from every listing?`)) return draw();
    try {
      await api("/api/admin/environments", { method: "POST", body: { key, action, ...extra } });
      const x = d.environments.find((y) => y.key === key);
      if (x) {
        if (action === "pin" || action === "unpin") x.pinned = action === "pin";
        if (action === "hide" || action === "unhide") x.hidden = action === "hide";
        if (action === "feature" || action === "unfeature") x.collection = extra.collection || null;
      }
      toast({ pin: "Pinned: it leads the trending row", unpin: "Unpinned", hide: "Hidden from listings", unhide: "Back in listings", feature: "Added to the collection",
              unfeature: "Taken out of its collection", index: "Indexing started" }[action]);
      draw();
    } catch (e) { toast(e.message); }
  };
  let t;
  $("#ad-q", body).addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { st.q = e.target.value.trim(); st.shown = 60; draw(); }, 150); });
  $("#ad-kind", body).addEventListener("change", (e) => { st.kind = e.target.value; draw(); });
  $("#ad-flag", body).addEventListener("click", (e) => { const b = e.target.closest("[data-flag]"); if (!b) return; st.flag = b.dataset.flag;
    body.querySelectorAll("#ad-flag [data-flag]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw(); });
  $("#ad-more", body).addEventListener("click", () => { st.shown += 100; draw(); });
  body.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (b) act(b.dataset.key, b.dataset.act);
    const i = e.target.closest("[data-index]");
    if (i) { i.outerHTML = spinner(); act(i.dataset.index, "index"); }
  });
  body.addEventListener("change", (e) => {
    const s = e.target.closest("[data-feature]");
    if (s) act(s.dataset.feature, s.value ? "feature" : "unfeature", s.value ? { collection: s.value } : {});
  });
  draw();
}

// ── collections ──────────────────────────────────────────────────────────────
async function collections(body) {
  const d = await api("/api/admin/collections");
  if (!alive) return;
  let cols = d.collections.map((c) => ({ ...c, ids: [...c.ids] }));
  let dirty = false;
  const draw = () => {
    body.innerHTML = `${head("Collections", "The tiles on Explore, in this order. One environment per line: a dataset id, or <code>space:org/name</code> for a Space.",
      `<button class="btn" id="ad-add" type="button">${icon("plus", 14)}Add a collection</button><button class="btn primary" id="ad-save" type="button" ${dirty ? "" : "disabled"}>${icon("check", 14)}Save</button>`)}
      <div class="ad-cols">${cols.map((c, i) => `<section class="panel" data-i="${i}"><div class="panel-h"><h2><span class="ad-swatch" style="--dc:var(--c-${esc(c.color)})">${icon(c.icon, 14)}</span>${esc(c.group || "New collection")}
        <span class="faint xs">${c.ids.length} environment${c.ids.length === 1 ? "" : "s"}</span></h2>
        <span class="aside"><button class="icon-btn" data-up="${i}" ${i ? "" : "disabled"} aria-label="Move up" title="Move up">${icon("chevronDown", 14, "up")}</button>
        <button class="icon-btn" data-down="${i}" ${i < cols.length - 1 ? "" : "disabled"} aria-label="Move down" title="Move down">${icon("chevronDown", 14)}</button>
        <button class="icon-btn" data-del="${i}" aria-label="Delete" title="Delete">${icon("x", 14)}</button></span></div>
        <div class="panel-b ad-form">
          <label class="field"><span>Name</span><input class="input" data-f="group" value="${esc(c.group)}" maxlength="40"></label>
          <label class="field"><span>Id <em>lowercase, for links</em></span><input class="input mono" data-f="id" value="${esc(c.id)}" maxlength="31"></label>
          <label class="field wide"><span>About</span><input class="input" data-f="about" value="${esc(c.about || "")}" maxlength="80"></label>
          <label class="field"><span>Icon</span><select class="hf-select-mimo" data-f="icon">${d.icons.map((x) => `<option ${x === c.icon ? "selected" : ""}>${x}</option>`).join("")}</select></label>
          <label class="field"><span>Colour</span><select class="hf-select-mimo" data-f="color">${d.colors.map((x) => `<option ${x === c.color ? "selected" : ""}>${x}</option>`).join("")}</select></label>
          <label class="field wide"><span>Environments</span><textarea class="input mono" data-f="ids" rows="${Math.min(10, c.ids.length + 2)}">${esc(c.ids.join("\n"))}</textarea></label>
        </div></section>`).join("")}</div>`;
  };
  const read = () => body.querySelectorAll("[data-i]").forEach((sec) => {
    const c = cols[+sec.dataset.i];
    sec.querySelectorAll("[data-f]").forEach((f) => { c[f.dataset.f] = f.dataset.f === "ids" ? f.value.split("\n").map((x) => x.trim()).filter(Boolean) : f.value.trim(); });
  });
  const touch = () => { dirty = true; const b = $("#ad-save", body); if (b) b.disabled = false; };
  body.addEventListener("input", touch);
  body.addEventListener("change", touch);
  body.addEventListener("click", async (e) => {
    const up = e.target.closest("[data-up]"), down = e.target.closest("[data-down]"), del = e.target.closest("[data-del]");
    if (up || down || del) { read(); dirty = true; }
    if (up) { const i = +up.dataset.up; [cols[i - 1], cols[i]] = [cols[i], cols[i - 1]]; draw(); }
    else if (down) { const i = +down.dataset.down; [cols[i + 1], cols[i]] = [cols[i], cols[i + 1]]; draw(); }
    else if (del) { if (confirm(`Delete the collection “${cols[+del.dataset.del].group}”? Its environments stay on the Hub.`)) { cols.splice(+del.dataset.del, 1); draw(); } }
    else if (e.target.closest("#ad-add")) { read(); dirty = true; cols.push({ id: `new-${cols.length + 1}`, group: "New collection", about: "", icon: "grid", color: "code", ids: [] }); draw(); }
    else if (e.target.closest("#ad-save")) {
      read();
      try { cols = (await api("/api/admin/collections", { method: "PUT", body: { collections: cols } })).collections.map((c) => ({ ...c, ids: [...c.ids] })); dirty = false; draw(); toast("Saved: Explore shows them within seconds"); }
      catch (err) { toast(err.message); }
    }
  });
  draw();
}

// ── indexes ──────────────────────────────────────────────────────────────────
async function indexes(body) {
  const d = await api("/api/admin/indexes");
  if (!alive) return;
  const jobs = Object.entries(d.jobs);
  body.innerHTML = `${head("Indexes", "Each dataset's index and pack: what makes its pages instant. Rebuild one after its dataset changes, or index a new one.")}
    <div class="ad-tools"><div class="search">${icon("database", 16)}<input id="ad-spec" placeholder="A public dataset: org/name"></div>
      <button class="btn primary" id="ad-go" type="button">${icon("play", 14)}Index it</button></div>
    ${jobs.length ? `<section class="panel" style="margin-bottom:12px"><div class="panel-h"><h2>${spinner()}Indexing now</h2></div><div class="panel-b"><dl class="kv">${jobs.map(([k, j]) =>
      `<dt><code>${esc(k)}</code></dt><dd>${esc(j.state)}${j.total ? ` · ${fmt.format(j.done || 0)} of ${fmt.format(j.total)}` : ""}${j.error ? ` · <span class="err-text">${esc(j.error)}</span>` : ""}</dd>`).join("")}</dl></div></section>` : ""}
    <div class="panel"><div class="tbl" style="max-height:none;border:0"><table><thead><tr><th>Dataset</th><th class="num">Tasks</th><th>Built</th><th class="num">Index</th><th class="num">Pack</th><th></th></tr></thead><tbody>
    ${d.indexes.map((x) => `<tr><td><a class="u" href="${EXPLORER}/d/${enc(x.spec)}" target="_blank" rel="noopener"><code>${esc(x.spec)}</code></a></td><td class="num">${x.tasks != null ? fmt.format(x.tasks) : "–"}</td>
      <td>${x.built ? ago(x.built) : "–"}${x.current ? "" : ` <span class="chip">old format</span>`}</td><td class="num">${bytes(x.bytes)}</td><td class="num">${bytes(x.pack_bytes)}</td>
      <td><button class="btn sm ghost" data-reindex="${esc(x.spec)}">${icon("refresh", 12)}Rebuild</button></td></tr>`).join("")}</tbody></table></div></div>`;
  const go = async (spec) => {
    try { await api("/api/admin/environments", { method: "POST", body: { key: spec, action: "index" } }); toast(`Indexing ${spec}`); timer = setTimeout(() => alive && indexes(body), 1500); }
    catch (e) { toast(e.message); }
  };
  $("#ad-go", body).addEventListener("click", () => { const v = $("#ad-spec", body).value.trim(); if (v) go(v); });
  $("#ad-spec", body).addEventListener("keydown", (e) => { if (e.key === "Enter") $("#ad-go", body).click(); });
  body.addEventListener("click", (e) => { const b = e.target.closest("[data-reindex]"); if (b) go(b.dataset.reindex); });
  if (jobs.length) timer = setTimeout(() => alive && indexes(body), 3000);
}

// ── settings ─────────────────────────────────────────────────────────────────
async function settings(body) {
  const s = await api("/api/admin/settings");
  if (!alive) return;
  body.innerHTML = `${head("Settings", "They apply to the explorer within seconds.")}
    <section class="panel"><div class="panel-b ad-settings">
      <div class="ad-set"><div><b>Rollouts</b><span>Off pauses new rollouts for everyone; running ones finish.</span></div>
        <label class="toggle"><input type="checkbox" id="st-on" ${s.rollouts_enabled ? "checked" : ""}><i></i></label></div>
      <div class="ad-set"><div><b>Rollouts at once</b><span>The whole explorer, then each person.</span></div>
        <div class="ad-nums"><input class="input" type="number" id="st-max" min="1" max="200" value="${s.max_active}" aria-label="The whole explorer"><span class="faint">/</span>
          <input class="input" type="number" id="st-user" min="1" max="20" value="${s.max_per_user}" aria-label="Each person"></div></div>
      <div class="ad-set"><div><b>Agents offered</b><span>What the run panel lists.</span></div>
        <div class="ad-checks">${s.all_agents.map((a) => `<label><input type="checkbox" data-agent="${a.id}" ${s.agents.includes(a.id) ? "checked" : ""}> ${esc(a.name)}</label>`).join("")}</div></div>
      <div class="ad-set col"><div><b>Announcement</b><span>One line above every page of the explorer; empty for none.</span></div>
        <input class="input" id="st-ann" maxlength="280" value="${esc(s.announcement)}" placeholder="e.g. Rollouts are slow today: a provider is overloaded.">
        <div class="announce ad-preview" id="st-prev" ${s.announcement ? "" : "hidden"}><div class="wrap">${icon("info", 14)}<span>${esc(s.announcement)}</span></div></div></div>
      <div class="ad-save-row"><span class="faint sm" id="st-dirty"></span><button class="btn primary" id="st-save" type="button">${icon("check", 14)}Save settings</button></div>
    </div></section>`;
  const mark = () => { $("#st-dirty", body).textContent = "Unsaved changes"; };
  body.addEventListener("input", (e) => {
    mark();
    if (e.target.id === "st-ann") { $("#st-prev", body).hidden = !e.target.value.trim(); $("#st-prev span", body).textContent = e.target.value; }
  });
  body.addEventListener("change", mark);
  $("#st-save", body).addEventListener("click", async () => {
    const on = $("#st-on", body).checked;
    if (!on && s.rollouts_enabled && !confirm("Pause rollouts for everyone?")) return;
    try {
      const r = await api("/api/admin/settings", { method: "PUT", body: { rollouts_enabled: on, max_active: +$("#st-max", body).value,
        max_per_user: +$("#st-user", body).value, agents: [...body.querySelectorAll("[data-agent]:checked")].map((x) => x.dataset.agent), announcement: $("#st-ann", body).value.trim() } });
      Object.assign(s, r);
      $("#st-dirty", body).textContent = "";
      toast("Saved");
    } catch (e) { toast(e.message); }
  });
}

// ── audit ────────────────────────────────────────────────────────────────────
async function audit(body) {
  const [d, ppl] = await Promise.all([api("/api/admin/audit"), api("/api/admin/people").catch(() => ({ people: [] }))]);
  if (!alive) return;
  const pics = Object.fromEntries(ppl.people.map((p) => [p.name, p.avatar]));
  const me = getSession().user;
  if (me) pics[me.name] = pics[me.name] || me.avatar;
  body.innerHTML = `${head("Audit log", "Every change made here, newest first.")}<div class="panel"><div class="panel-b">${d.entries.length ? `<ol class="ad-audit">${d.entries.map((x) => `<li>
    ${person(x.user, pics[x.user])}<span>${esc(x.note || Object.keys(x.after || {}).join(", ") || "changed settings")}</span><span class="faint" title="${new Date(x.at * 1000).toLocaleString()}">${ago(x.at)}</span></li>`).join("")}</ol>`
    : emptyState("list", "Nothing yet", "Changes made here show up in this log.")}</div></div>`;
}
