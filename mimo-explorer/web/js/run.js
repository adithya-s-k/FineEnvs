// A rollout, live: phases across the top, the agent's steps as a timeline, and the grade when it lands.
import { $, api, esc, md, money, dur, ago, tokensShort, statusPill, toast, DOMAIN_NAME, LIVE_STATUSES, rewardClass, rewardText, sk, spinner, emptyState } from "./util.js";
import { icon, DOMAIN_ICON } from "./icons.js";
import { refreshActive } from "./session.js";

const PHASES = [["sandbox", "Sandbox"], ["setup", "Setup"], ["agent", "Agent"], ["verify", "Grading"], ["done", "Done"]];
const TOOL_ICON = { bash: "terminal", read: "file", write: "pencil", edit: "pencil", grep: "search", glob: "search", todowrite: "check", list: "list" };
let timer = null, tick = null, alive = true;

export function unmount() { alive = false; clearTimeout(timer); clearInterval(tick); }
const isLive = (r) => r && LIVE_STATUSES.includes(r.status);

function skeleton() {
  return `<div class="wrap page"><div class="crumbs">${sk.line(18)}</div>
    <div class="rp-head">${sk.line(22, 22)}${sk.line(55, 24)}${sk.line(45)}</div>
    <div class="stepper">${PHASES.map(() => `<div class="step">${sk.box(20, "width:20px;border-radius:50%")}${sk.line(60)}</div>`).join("")}</div>
    <div class="rp-grid"><div style="display:grid;gap:10px">${[70, 90, 55, 85, 60].map((w) => `<div style="display:grid;grid-template-columns:28px 1fr;gap:10px">${sk.box(28, "border-radius:8px")}${sk.box(30, `width:${w}%;border-radius:10px`)}</div>`).join("")}</div>
      <div style="display:grid;gap:12px">${sk.card(sk.line(30, 14) + sk.line(25, 40) + sk.lines(90, 70))}${sk.card(sk.line(30, 14) + sk.line(35, 28) + sk.lines(80, 60))}</div></div></div>`;
}

export async function mount(el, id) {
  alive = true;
  el.innerHTML = skeleton();
  const st = { id, after: 0, events: [], run: null, follow: true, steps: 0 };
  let first;
  try { first = await api(`/api/runs/${encodeURIComponent(id)}`); }
  catch (e) {
    if (!alive) return;
    el.innerHTML = `<div class="wrap page">${emptyState("search", "Rollout not found", e.status === 404 ? "It may belong to another account, or the link is wrong." : esc(e.message),
      `<a class="btn" href="#/runs">${icon("arrowLeft")}Your rollouts</a>`)}</div>`;
    return;
  }
  if (!alive) return;
  el.innerHTML = `<div class="wrap page rp fade-in">
    <nav class="crumbs" aria-label="Breadcrumb"><a href="#/runs">Rollouts</a>${icon("chevronRight", 13)}<span>${esc(id)}</span></nav>
    <header class="rp-head" id="rp-head"></header>
    <div class="stepper" id="rp-phases"></div>
    <div class="rp-grid">
      <div class="rp-main"><div class="tl-bar"><h2>What the agent did</h2><span class="n" id="rp-n"></span>
        <div class="tools-right"><button class="btn sm ghost" id="rp-expand" type="button">Expand all</button>
        <label class="switch" id="rp-follow-wrap"><input type="checkbox" id="rp-follow" checked> Follow live</label></div></div>
        <ol class="timeline" id="rp-tl"></ol><div id="rp-tail"></div></div>
      <aside class="rp-side"><section class="panel grade" id="rp-grade"></section><section class="panel costcard" id="rp-cost"></section></aside>
    </div></div>`;
  $("#rp-follow", el).addEventListener("change", (e) => (st.follow = e.target.checked));
  $("#rp-expand", el).addEventListener("click", (e) => {
    const open = e.target.textContent === "Expand all";
    el.querySelectorAll("#rp-tl details").forEach((d) => (d.open = open));
    e.target.textContent = open ? "Collapse all" : "Expand all";
  });
  el.addEventListener("click", onClick(st));
  apply(el, st, first);
  const poll = async () => {
    if (!alive) return;
    try { apply(el, st, await api(`/api/runs/${encodeURIComponent(id)}?after=${st.after}`)); } catch { /* retry next tick */ }
    if (alive && isLive(st.run)) timer = setTimeout(poll, 1500);
    else refreshActive();
  };
  if (isLive(st.run)) timer = setTimeout(poll, 1500);
  tick = setInterval(() => { if (isLive(st.run)) { renderHead(el, st); renderTail(el, st); } }, 1000);
}

function apply(el, st, d) {
  st.run = d.run;
  const fresh = d.events.filter((e) => e.i >= st.after);
  st.events.push(...fresh);
  st.after = st.events.length ? st.events[st.events.length - 1].i + 1 : st.after;
  renderHead(el, st);
  renderPhases(el, st);
  appendTimeline(el, st, fresh);
  renderGrade(el, st);
  renderCost(el, st);
  renderTail(el, st);
  $("#rp-follow-wrap", el).hidden = !isLive(st.run);
  if (st.follow && fresh.length && isLive(st.run)) $("#rp-tail", el).scrollIntoView({ behavior: "smooth", block: "end" });
}

function renderTail(el, st) {
  const t = $("#rp-tail", el);
  if (!t) return;
  t.innerHTML = isLive(st.run) ? `<ol class="timeline"><li class="ev ev-live"><span class="ei">${spinner()}</span><div class="eb">${esc(liveHint(st))}</div></li></ol>` : "";
}

function liveHint(st) {
  const s = st.run.status;
  if (s === "queued") return "Queued…";
  if (s === "starting") return "Starting a sandbox from the task's image. Big images take a minute or two.";
  if (s === "setup") return "Preparing the environment…";
  if (s === "verifying") return "Grading…";
  const last = st.events[st.events.length - 1];
  if (last?.kind === "partial") return `Writing… ${last.chars.toLocaleString()} characters${last.thinking_chars ? `, ${last.thinking_chars.toLocaleString()} thinking` : ""}`;
  const since = last ? Math.max(0, (Date.now() / 1000 - (st.run.started_at || st.run.created_at)) - last.t) : 0;
  return since > 45 ? `The model is working · ${dur(since)} since the last step. Long files take a while to write.` : "The model is working…";
}

function renderHead(el, st) {
  const r = st.run;
  if (!r) return;
  const start = r.started_at || r.created_at, end = r.finished_at || Date.now() / 1000;
  const action = isLive(r)
    ? (st.stopping ? `<button class="btn danger sm" disabled>${spinner()}Stopping</button>` : `<button class="btn danger sm" id="rp-cancel">${icon("stop", 13)}Stop</button>`)
    : `<a class="btn sm primary" href="#/task/${encodeURIComponent(r.task_id)}">${icon("refresh", 13)}Run again</a>`;
  $("#rp-head", el).innerHTML = `
    <div class="kick"><span class="badge" style="--dc:var(--c-${r.domain})">${icon(DOMAIN_ICON[r.domain], 13)}${esc(DOMAIN_NAME[r.domain])}</span>${statusPill(r.status)}
      <span class="when">${icon("clock", 13)}${dur(end - start)} · started ${ago(r.created_at)}</span></div>
    <h1><a href="#/task/${encodeURIComponent(r.task_id)}">${esc(r.title)}</a></h1>
    <div class="rp-bar"><div class="facts">
      <span>${icon("cpu", 14)}<b>${esc(r.model.split("/")[1] || r.model)}</b> via ${esc(r.provider || "auto")}</span>
      ${r.judge ? `<span>${icon("scale", 14)}judge <b>${esc(r.judge.split("/")[1] || r.judge)}</b></span>` : ""}
      ${r.domain === "music" ? `<span>${icon("message", 14)}single model call</span>` : `<span>${icon("terminal", 14)}OpenCode</span>`}${r.flavor ? `<span>${icon("box", 14)}${esc(r.flavor)}</span>` : ""}</div>
      <div class="rp-actions"><a class="btn sm" href="#/task/${encodeURIComponent(r.task_id)}">${icon("file", 13)}View task</a>${action}</div></div>
    ${r.error ? `<div class="note-box err">${icon("alert")}<span>${esc(r.error)}</span></div>` : ""}`;
}

function renderPhases(el, st) {
  const seen = {};
  st.events.filter((e) => e.kind === "phase").forEach((e) => { (seen[e.name] ||= {})[e.status] = e; });
  const failed = ["failed", "cancelled", "interrupted"].includes(st.run.status);
  const noSandbox = st.run.domain === "music";
  const list = PHASES.filter(([k]) => !(noSandbox && (k === "sandbox" || k === "setup")));
  const box = $("#rp-phases", el);
  box.style.setProperty("--n", list.length);
  box.innerHTML = list.map(([k, label]) => {
    const p = seen[k] || {};
    const state = p.done ? (p.done.status === "error" ? "bad" : "ok") : p.start ? (failed ? "bad" : "on") : "off";
    const t = p.start && p.done ? dur(p.done.t - p.start.t) : "";
    const detail = (p.done || p.start)?.detail || "";
    const mk = state === "ok" ? icon("check", 12) : state === "bad" ? icon("x", 12) : "";
    return `<div class="step ${state}" title="${esc(detail)}"><span class="mk">${mk}</span><b>${label}</b><span class="t">${esc(t)}</span><em>${esc(detail)}</em></div>`;
  }).join("");
}

function appendTimeline(el, st, events) {
  const tl = $("#rp-tl", el);
  for (const e of events) {
    if (e.kind === "phase" && e.status === "start") st.phase = e.name;
    const html = item(e, st);
    if (html) { tl.insertAdjacentHTML("beforeend", html); if (e.kind === "tool" || e.kind === "text") st.steps++; }
  }
  $("#rp-n", el).textContent = st.steps ? `${st.steps} step${st.steps === 1 ? "" : "s"}` : "";
  const empty = $(".tl-empty", el);
  if (!tl.children.length && !isLive(st.run)) { if (!empty) tl.insertAdjacentHTML("afterend", `<p class="tl-empty">No agent steps were recorded.</p>`); }
  else if (empty) empty.remove();
}

const row = (cls, ic, t, body) => `<li class="ev ${cls}"><span class="ei">${icon(ic, 14)}</span><div class="eb">${body}</div><span class="ts">${t}</span></li>`;

function item(e, st) {
  const t = dur(e.t);
  const summary = (inner) => `<summary>${icon("chevronRight", 13, "chev")}${inner}</summary>`;
  switch (e.kind) {
    case "text": return row("ev-text", "message", t, `<div class="msg prose">${md(e.text)}</div>`);
    case "thinking": return row("ev-think", "brain", t, `<details>${summary(`<b>Thinking</b><code>${e.chars.toLocaleString()} characters</code>`)}<pre class="code-block wrap">${esc(e.text)}</pre></details>`);
    case "tool": {
      const inp = e.input || {};
      const head = inp.command || inp.filePath || inp.pattern || inp.path || e.title || Object.values(inp).map(String).join(" ").slice(0, 160);
      const mcp = !(e.tool in TOOL_ICON) && /_/.test(e.tool || "");
      const cls = `ev-tool${mcp ? " mcp" : ""}${e.status === "error" ? " bad" : ""}`;
      return row(cls, mcp ? "plug" : TOOL_ICON[e.tool] || "terminal", t, `<details>${summary(`<b>${esc(mcp ? prettyTool(e.tool) : e.tool)}</b>
        <code>${esc(String(head).split("\n")[0].slice(0, 200))}</code>${e.ms >= 1000 ? `<span class="ms">${dur(e.ms / 1000)}</span>` : ""}`)}
        <div class="io">${Object.keys(inp).length ? `<h5>Input</h5><pre>${esc(JSON.stringify(inp, null, 2))}</pre>` : ""}
          <h5>Output${e.truncated ? " (truncated)" : ""}</h5><pre>${esc(e.output || "(empty)")}</pre></div></details>`);
    }
    case "phase": return e.status === "error" && e.name !== "done" ? row("ev-err", "alert", t, `${esc(e.name)} failed: ${esc(e.detail)}`) : "";
    case "error": return row("ev-err", "alert", t, `<b>Error</b> ${esc(e.text)}`);
    case "log": return row("ev-log", "terminal", t, `<details>${summary(`<b>${({ sandbox: "Sandbox", setup: "Setup output", agent: "Agent output", verify: "Grader output" })[st.phase] || "Output"}</b><code>${esc((e.text || "").split("\n")[0].slice(0, 160))}</code>`)}<pre class="code-block">${esc(e.text)}</pre></details>`);
    case "diff": return row("ev-diff", "diff", t, `<details open>${summary(`<b>The agent's change</b><code>${e.files.length} file${e.files.length === 1 ? "" : "s"}</code>`)}<pre class="code-block diff">${diffHtml(e.text)}</pre></details>`);
    case "files": return row("ev-log", "folder", t, `<details>${summary(`<b>${esc(e.title)}</b><code>${e.files.length} files</code>`)}<pre class="code-block">${esc(e.files.join("\n"))}</pre></details>`);
    case "image": {
      const src = `/api/runs/${encodeURIComponent(st.id)}/artifacts/${encodeURIComponent(e.name)}`;
      return row("ev-img", "image", t, `<b>Rendered page</b> <span class="muted sm">${e.width}×${e.height}</span>
        <a href="${src}" target="_blank" rel="noopener"><img loading="lazy" src="${src}" alt="Full-page screenshot of what the agent built"></a>`);
    }
    default: return "";
  }
}

const humanize = (id) => { const t = String(id || "").replace(/[_-]+/g, " ").trim(); return t.charAt(0).toUpperCase() + t.slice(1); };
const prettyTool = (name) => { const parts = name.split("_"); return parts.length > 3 ? parts.slice(-3).join("_") : name; };

function diffHtml(text) {
  return (text || "").split("\n").slice(0, 3000).map((l) => {
    const c = l.startsWith("+++") || l.startsWith("---") || l.startsWith("diff ") ? "dh" : l.startsWith("+") ? "da" : l.startsWith("-") ? "dd" : l.startsWith("@@") ? "dc" : "";
    return `<span class="${c}">${esc(l)}</span>`;
  }).join("\n");
}

function renderGrade(el, st) {
  const g = [...st.events].reverse().find((e) => e.kind === "checks");
  const box = $("#rp-grade", el);
  const h = `<div class="panel-h"><h3>${icon("scale", 14)}Grade</h3></div>`;
  if (!g) {
    box.innerHTML = `${h}<div class="panel-b"><p class="muted sm">${isLive(st.run) ? "Appears here once the agent finishes and the task's grader has run." : "This rollout was not graded."}</p></div>`;
    return;
  }
  const rw = g.reward;
  const img = [...st.events].reverse().find((e) => e.kind === "image");
  const src = img && `/api/runs/${encodeURIComponent(st.id)}/artifacts/${encodeURIComponent(img.name)}`;
  const long = (g.summary || "").length > 180;
  const zh = /[一-鿿]/.test(g.summary || "");
  const cls = rewardClass(rw);
  box.innerHTML = `${h}<div class="panel-b">
    ${src ? `<a class="shot" href="${src}" target="_blank" rel="noopener" title="Open the full-page render"><img src="${src}" alt="What the judge saw"></a>` : ""}
    <div class="score"><span class="big ${cls}">${rewardText(rw)}</span><span class="of">reward, out of 1</span></div>
    ${rw != null ? `<div class="meter big ${cls}" style="font-size:0"><i style="width:${Math.max(2, rw * 100)}%"></i></div>` : ""}
    ${g.formula ? `<p class="muted xs">${esc(g.formula)}</p>` : ""}
    ${long || !g.summary ? "" : `<p class="gsum">${esc(g.summary)}</p>`}
    ${(g.checks || []).length ? `<ul class="gchecks">${g.checks.map((c) => `<li class="${c.passed === true ? "ok" : c.passed === false ? "bad" : "na"}">
      <span class="mk">${icon(c.passed === true ? "check" : c.passed === false ? "x" : "more", 14)}</span>
      <div><b>${esc(c.question || humanize(c.id))}</b>
      ${c.tier || c.method ? `<span class="meta">${esc([c.tier, c.method === "llm" ? "model-judged" : c.method].filter(Boolean).join(" · "))}</span>` : ""}
      ${c.message ? `<p>${esc(String(c.message).slice(0, 300))}</p>` : ""}</div>
      ${c.score != null && c.passed == null ? `<span class="sc">${Number(c.score).toFixed(2)}</span>` : "<span></span>"}</li>`).join("")}</ul>` : ""}
    ${long ? `<details class="notes"><summary class="disclose">${icon("chevronRight", 14, "chev")}Judge's notes</summary><p>${esc(g.summary)}</p>
      ${zh ? `<p class="xs faint">In Chinese because Xiaomi's grading rubric is written in Chinese; it is used unchanged so scores stay comparable with theirs.</p>` : ""}</details>` : ""}
  </div>`;
}

function renderCost(el, st) {
  const r = st.run, c = r.cost || {}, t = r.tokens || {};
  const tot = (c.model || 0) + (c.sandbox || 0) || 1;
  $("#rp-cost", el).innerHTML = `<div class="panel-h"><h3>${icon("coins", 14)}Cost</h3>${isLive(r) ? `<span class="aside"><i class="pulse" style="color:var(--live)"></i> live</span>` : ""}</div>
    <div class="panel-b">
    <div class="big-cost">${money(c.total)}</div>
    <div class="cost-split"><i class="m" style="width:${((c.model || 0) / tot) * 100}%"></i><i class="s" style="width:${((c.sandbox || 0) / tot) * 100}%"></i></div>
    <div class="legend">
      <div><span class="sw" style="background:var(--c-code)"></span><span>Model tokens</span><b>${money(c.model)}</b></div>
      <div><span class="sw" style="background:var(--c-cyber)"></span><span>Sandbox time</span><b>${money(c.sandbox)}</b></div>
      ${r.judge ? `<div><span class="sw" style="background:var(--surface-3)"></span><span>Judge calls</span><b class="faint">not metered</b></div>` : ""}
    </div>
    <dl class="kv sm"><dt>Tokens in</dt><dd class="num">${tokensShort((t.input || 0) + (t.cache_read || 0))}${t.cache_read ? ` <span class="faint">(${tokensShort(t.cache_read)} cached)</span>` : ""}</dd>
      <dt>Tokens out</dt><dd class="num">${tokensShort(t.output)}${t.reasoning ? ` <span class="faint">+ ${tokensShort(t.reasoning)} thinking</span>` : ""}</dd></dl>
    <p class="fine">Billed to your Hugging Face account at ${esc(r.provider || "the provider")}'s price.</p></div>`;
}

function onClick(st) {
  return async (ev) => {
    const b = ev.target.closest("#rp-cancel");
    if (!b) return;
    st.stopping = true;
    b.disabled = true;
    b.innerHTML = `${spinner()}Stopping`;
    try { await api(`/api/runs/${encodeURIComponent(st.id)}/cancel`, { method: "POST" }); toast("Stopping the rollout…"); }
    catch (e) { st.stopping = false; toast(e.message); b.disabled = false; b.innerHTML = `${icon("stop", 13)}Stop`; }
  };
}
