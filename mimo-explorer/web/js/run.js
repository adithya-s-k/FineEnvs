// A rollout, live: phases across the top, the agent's steps as a timeline, and the grade when it lands.
import { $, $$, api, esc, md, money, dur, ago, tokensShort, statusPill, rewardBadge, toast, DOMAIN_NAME } from "./util.js";
import { refreshActive } from "./session.js";

const PHASES = [["sandbox", "Sandbox"], ["setup", "Setup"], ["agent", "Agent"], ["verify", "Grading"], ["done", "Done"]];
const TOOL_ICON = { bash: "$", read: "📄", write: "✎", edit: "✎", grep: "⌕", glob: "⌕", todowrite: "☑", list: "▤" };
let timer = null, tick = null, alive = true;

export function unmount() { alive = false; clearTimeout(timer); clearInterval(tick); }

export async function mount(el, id) {
  alive = true;
  el.innerHTML = `<div class="wrap page"><div class="skel-page"><i></i><i></i><i></i></div></div>`;
  const st = { id, after: 0, events: [], run: null, follow: true };
  let first;
  try { first = await api(`/api/runs/${encodeURIComponent(id)}`); }
  catch (e) { el.innerHTML = `<div class="wrap page"><p class="empty">Could not open this rollout: ${esc(e.message)}</p><p><a href="#/runs">← My rollouts</a></p></div>`; return; }
  el.innerHTML = `<div class="wrap page rp">
    <a class="back" href="#/runs">← My rollouts</a>
    <header class="rp-head" id="rp-head"></header>
    <div class="phases" id="rp-phases"></div>
    <div class="rp-grid">
      <div class="rp-main"><div class="tl-bar"><h2>What the agent did</h2><label class="follow"><input type="checkbox" id="rp-follow" checked> Follow live</label></div>
        <ol class="timeline" id="rp-tl"></ol><div id="rp-tail"></div></div>
      <aside class="rp-side"><div id="rp-grade" class="grade"></div><div id="rp-cost" class="costcard"></div></aside>
    </div></div>`;
  $("#rp-follow", el).addEventListener("change", (e) => (st.follow = e.target.checked));
  el.addEventListener("click", onClick(st));
  apply(el, st, first);
  const poll = async () => {
    if (!alive) return;
    try { apply(el, st, await api(`/api/runs/${encodeURIComponent(id)}?after=${st.after}`)); } catch { /* retry */ }
    if (alive && isLive(st.run)) timer = setTimeout(poll, 1500);
    else refreshActive();
  };
  if (isLive(st.run)) timer = setTimeout(poll, 1500);
  tick = setInterval(() => {
    renderHead(el, st);
    const t = $("#rp-tail", el);
    if (t && isLive(st.run)) t.innerHTML = `<div class="thinking"><span class="spinner"></span>${esc(liveHint(st))}</div>`;
  }, 1000);
}

const isLive = (r) => r && ["queued", "starting", "setup", "running", "verifying"].includes(r.status);

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
  const tail = $("#rp-tail", el);
  tail.innerHTML = isLive(st.run) ? `<div class="thinking"><span class="spinner"></span>${esc(liveHint(st))}</div>` : "";
  if (st.follow && fresh.length && isLive(st.run)) tail.scrollIntoView({ behavior: "smooth", block: "end" });
}

function liveHint(st) {
  const s = st.run.status;
  if (s === "queued") return "Queued…";
  if (s === "starting") return "Starting a sandbox from the task's image (big images take a minute or two)…";
  if (s === "setup") return "Preparing the environment…";
  if (s === "verifying") return "Grading…";
  const last = st.events[st.events.length - 1];
  if (last?.kind === "partial") return `Writing… ${last.chars.toLocaleString()} characters${last.thinking_chars ? `, ${last.thinking_chars.toLocaleString()} thinking` : ""}`;
  const since = last ? Math.max(0, (Date.now() / 1000 - (st.run.started_at || st.run.created_at)) - last.t) : 0;
  const tail = since > 45 ? ` · ${dur(since)} since the last step (long files take a while to write)` : "";
  return `The model is working…${tail}`;
}

function renderHead(el, st) {
  const r = st.run;
  if (!r) return;
  const start = r.started_at || r.created_at, end = r.finished_at || Date.now() / 1000;
  $("#rp-head", el).innerHTML = `
    <div class="kick"><span class="badge" style="--dc:var(--c-${r.domain})"><i class="dot"></i>${esc(DOMAIN_NAME[r.domain])}</span>${statusPill(r.status)}
      <span class="muted sm">${dur(end - start)} · started ${ago(r.created_at)}</span></div>
    <h1><a href="#/task/${encodeURIComponent(r.task_id)}">${esc(r.title)}</a></h1>
    <div class="facts"><span>agent <b>${esc(r.model)}</b> via ${esc(r.provider || "auto")}</span>${r.judge ? `<span>judge <b>${esc(r.judge)}</b></span>` : ""}
      <span>harness <b>OpenCode</b></span>${r.flavor ? `<span>sandbox <b>${esc(r.flavor)}</b></span>` : ""}
      ${isLive(r) ? (st.stopping ? `<button class="btn danger sm" disabled><span class="spinner sm"></span> Stopping</button>` : `<button class="btn danger sm" id="rp-cancel">Stop</button>`) : `<a class="btn sm" href="#/task/${encodeURIComponent(r.task_id)}">Run again</a>`}</div>
    ${r.error ? `<p class="err">${esc(r.error)}</p>` : ""}`;
}

function renderPhases(el, st) {
  const seen = {};
  st.events.filter((e) => e.kind === "phase").forEach((e) => { (seen[e.name] ||= {})[e.status] = e; });
  const failed = ["failed", "cancelled", "interrupted"].includes(st.run.status);
  const noSandbox = st.run.domain === "music";
  $("#rp-phases", el).innerHTML = PHASES.filter(([k]) => !(noSandbox && (k === "sandbox" || k === "setup"))).map(([k, label]) => {
    const p = seen[k] || {};
    const state = p.done ? (p.done.status === "error" ? "bad" : "ok") : p.start ? (failed ? "bad" : "on") : "off";
    const t = p.start && p.done ? dur(p.done.t - p.start.t) : p.start && isLive(st.run) ? "…" : "";
    const detail = (p.done || p.start)?.detail || "";
    return `<div class="ph ph-${state}"><i></i><b>${label}</b><span>${esc(t)}</span><em title="${esc(detail)}">${esc(detail)}</em></div>`;
  }).join("");
}

function appendTimeline(el, st, events) {
  const tl = $("#rp-tl", el);
  for (const e of events) {
    const html = item(e, st);
    if (html) tl.insertAdjacentHTML("beforeend", html);
  }
  if (!tl.children.length && !isLive(st.run)) tl.innerHTML = `<li class="muted">No agent steps were recorded.</li>`;
}

function item(e, st) {
  const t = `<span class="ts">${dur(e.t)}</span>`;
  switch (e.kind) {
    case "text": return `<li class="ev ev-text">${t}<div class="msg">${md(e.text)}</div></li>`;
    case "thinking": return `<li class="ev ev-think">${t}<details><summary>Thinking (${e.chars.toLocaleString()} characters)</summary><pre class="wrap">${esc(e.text)}</pre></details></li>`;
    case "tool": {
      const inp = e.input || {};
      const head = inp.command || inp.filePath || inp.pattern || inp.path || e.title || Object.values(inp).map(String).join(" ").slice(0, 160);
      const mcp = !(e.tool in TOOL_ICON) && /_/.test(e.tool || "");
      const icon = mcp ? "⇄" : TOOL_ICON[e.tool] || "•";
      const ok = e.status === "error" ? " bad" : "";
      return `<li class="ev ev-tool${mcp ? " mcp" : ""}${ok}">${t}<details><summary><span class="ti">${icon}</span><b>${esc(mcp ? prettyTool(e.tool) : e.tool)}</b>
        <code>${esc(String(head).split("\n")[0].slice(0, 200))}</code>${e.ms ? `<span class="ms">${dur(e.ms / 1000)}</span>` : ""}</summary>
        <div class="io">${Object.keys(inp).length ? `<h5>Input</h5><pre>${esc(JSON.stringify(inp, null, 2))}</pre>` : ""}
          <h5>Output${e.truncated ? " (truncated)" : ""}</h5><pre>${esc(e.output || "(empty)")}</pre></div></details></li>`;
    }
    case "phase": return e.status === "error" ? `<li class="ev ev-err">${t}${esc(e.name)} failed: ${esc(e.detail)}</li>` : "";
    case "error": return `<li class="ev ev-err">${t}<b>Error</b> ${esc(e.text)}</li>`;
    case "log": return `<li class="ev ev-log">${t}<details><summary>Output</summary><pre>${esc(e.text)}</pre></details></li>`;
    case "diff": return `<li class="ev ev-diff">${t}<details open><summary>The agent's change: ${e.files.length} file${e.files.length === 1 ? "" : "s"}</summary><pre class="diff">${diffHtml(e.text)}</pre></details></li>`;
    case "files": return `<li class="ev ev-log">${t}<details><summary>${esc(e.title)}: ${e.files.length} files</summary><pre>${esc(e.files.join("\n"))}</pre></details></li>`;
    case "image": return `<li class="ev ev-img">${t}<div><b>Rendered page</b> <span class="muted sm">${e.width}×${e.height}</span>
      <a href="/api/runs/${encodeURIComponent(st.id)}/artifacts/${esc(e.name)}" target="_blank" rel="noopener"><img loading="lazy" src="/api/runs/${encodeURIComponent(st.id)}/artifacts/${esc(e.name)}" alt="Full-page screenshot"></a></div></li>`;
    default: return "";
  }
}

const prettyTool = (name) => {
  const parts = name.split("_");
  return parts.length > 3 ? parts.slice(-3).join("_") : name;
};

function diffHtml(text) {
  return (text || "").split("\n").slice(0, 3000).map((l) => {
    const c = l.startsWith("+++") || l.startsWith("---") || l.startsWith("diff ") ? "dh" : l.startsWith("+") ? "da" : l.startsWith("-") ? "dd" : l.startsWith("@@") ? "dc" : "";
    return `<span class="${c}">${esc(l)}</span>`;
  }).join("\n");
}

function renderGrade(el, st) {
  const g = [...st.events].reverse().find((e) => e.kind === "checks");
  const box = $("#rp-grade", el);
  if (!g) {
    box.innerHTML = `<h3>Grade</h3><p class="muted sm">${isLive(st.run) ? "Appears here once the agent finishes and the task's grader has run." : "This rollout was not graded."}</p>`;
    return;
  }
  const rw = g.reward;
  const img = [...st.events].reverse().find((e) => e.kind === "image");
  const src = img && `/api/runs/${encodeURIComponent(st.id)}/artifacts/${encodeURIComponent(img.name)}`;
  const long = (g.summary || "").length > 180;
  const zh = /[\u4e00-\u9fff]/.test(g.summary || "");
  box.innerHTML = `<h3>Grade</h3>
    ${src ? `<a class="shot" href="${src}" target="_blank" rel="noopener" title="Open the full-page render"><img src="${src}" alt="What the judge saw"></a>` : ""}
    <div class="big ${rw == null ? "none" : rw >= 0.999 ? "full" : rw > 0 ? "part" : "zero"}">${rw == null ? "–" : rw % 1 === 0 ? rw.toFixed(0) : rw.toFixed(3)}</div>
    ${g.formula ? `<p class="muted xs">${esc(g.formula)}</p>` : ""}
    ${long ? "" : `<p class="gsum">${esc(g.summary || "")}</p>`}
    <ul class="gchecks">${(g.checks || []).map((c) => `<li class="${c.passed === true ? "ok" : c.passed === false ? "bad" : "na"}">
      <span class="mk">${c.passed === true ? "✓" : c.passed === false ? "✗" : "·"}</span>
      <div><b>${esc(c.question || c.id)}</b>${c.score != null && c.passed == null ? ` <span class="sc">${Number(c.score).toFixed(2)}</span>` : ""}
      ${c.tier || c.method ? `<span class="meta">${esc([c.tier, c.method === "llm" ? "model-judged" : c.method].filter(Boolean).join(" · "))}</span>` : ""}
      ${c.message ? `<p>${esc(String(c.message).slice(0, 300))}</p>` : ""}</div></li>`).join("")}</ul>
    ${long ? `<details class="notes"><summary>Judge's notes</summary><p>${esc(g.summary)}</p>
      ${zh ? `<p class="muted xs">In Chinese because Xiaomi's grading rubric is written in Chinese; it is used unchanged so scores stay comparable with theirs.</p>` : ""}</details>` : ""}`;
}

function renderCost(el, st) {
  const r = st.run, c = r.cost || {}, t = r.tokens || {};
  $("#rp-cost", el).innerHTML = `<h3>Cost ${isLive(r) ? '<span class="muted xs">live</span>' : ""}</h3>
    <div class="big-cost">${money(c.total)}</div>
    <dl class="kv sm"><dt>Model</dt><dd>${money(c.model)}</dd>${r.judge ? `<dt>Judge</dt><dd class="muted">not metered (small)</dd>` : ""}
      <dt>Sandbox</dt><dd>${money(c.sandbox)}</dd>
      <dt>Tokens</dt><dd>${tokensShort((t.input || 0) + (t.cache_read || 0))} in · ${tokensShort(t.output)} out${t.reasoning ? ` · ${tokensShort(t.reasoning)} thinking` : ""}</dd></dl>
    <p class="muted xs">Billed to your Hugging Face account at ${esc(r.provider || "the provider")}'s price.</p>`;
}

function onClick(st) {
  return async (ev) => {
    if (ev.target.id === "rp-cancel") {
      st.stopping = true;
      ev.target.disabled = true;
      ev.target.innerHTML = `<span class="spinner sm"></span> Stopping`;
      try { await api(`/api/runs/${encodeURIComponent(st.id)}/cancel`, { method: "POST" }); toast("Stopping the rollout…"); }
      catch (e) { st.stopping = false; toast(e.message); ev.target.disabled = false; ev.target.textContent = "Stop"; }
    }
  };
}
