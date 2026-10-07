"use strict";
// Eval runs: a leaderboard and task grid per eval set, run pages, and episode trajectories.

const pct = (x) => (x == null ? "–" : `${Math.round(100 * x)}%`);
const secs = (x) => (x == null ? "–" : x < 90 ? `${Math.round(x)} s` : `${(x / 60).toFixed(1)} min`);
const evalHref = (path, params) => `#/evals${path}?${new URLSearchParams(params)}`;
const runOrder = (a, b) => (b.pass_at_1 ?? -1) - (a.pass_at_1 ?? -1) || (b.mean_reward ?? -1) - (a.mean_reward ?? -1);
const REFRESH_MS = 15000;
let refreshTimer = null;

// Re-render in place while a run is still writing episodes, keeping the scroll position.
function refreshWhileLive(live, render) {
  clearTimeout(refreshTimer);
  if (!live) return;
  const hash = location.hash;
  refreshTimer = setTimeout(async () => {
    if (location.hash !== hash) return;
    const y = window.scrollY;
    await render();
    hydrate($("#view"));
    window.scrollTo(0, y);
  }, REFRESH_MS);
}

function statusHtml(s) {
  if (s.status === "done") return `<span class="ok sm"><i class="dot"></i>done</span>`;
  if (s.status === "running") return `<span class="sm live"><span class="spinner"></span>${s.graded} / ${s.expected}</span>`;
  return `<span class="warn sm"><i class="dot"></i>stopped at ${s.graded} / ${s.expected}</span>`;
}

function meter(x) {
  return `<span class="meter"><i style="width:${(100 * (x || 0)).toFixed(0)}%"></i></span>`;
}

// The newest run of each model is current; older runs of the same model are earlier attempts.
function markEarlier(runs) {
  const latest = new Map();
  runs.forEach((r) => {
    if (!latest.has(r.model) || r.updated > latest.get(r.model).updated) latest.set(r.model, r);
  });
  return runs.map((r) => ({ ...r, earlier: latest.get(r.model) !== r }));
}

function cellHtml(run, task, cell) {
  const href = evalHref("/episode", { run: run.run, task: task.id });
  const title = `${run.label} · ${task.id}`;
  if (!cell) return `<span class="cell none" title="${esc(title)}: not run"></span>`;
  const tip = `${title} · reward ${num(cell.reward, 2)}${cell.failure ? ` · ${cell.failure}` : ""}`;
  if (!cell.graded) return `<a class="cell ungraded" href="${href}" title="${esc(tip)} (provider error)">·</a>`;
  if (cell.valid)
    return `<a class="cell pass${cell.exact ? " exact" : ""}" href="${href}" title="${esc(tip)}">${cell.exact ? "★" : "✓"}</a>`;
  if (!cell.reward) return `<a class="cell zero" href="${href}" title="${esc(tip)}">0</a>`;
  return `<a class="cell part" style="--a:${Math.round(15 + 45 * cell.reward)}%" href="${href}" title="${esc(tip)}">${Math.round(100 * cell.reward)}</a>`;
}

function leaderboardHtml(runs) {
  return `<div class="tbl"><table>
    <tr><th class="num">#</th><th>Model</th><th>Status</th><th class="num">Solved</th><th class="num">Exact route</th><th>Mean reward</th>
      <th class="num">Steps supported</th><th class="num">Stock claims right</th><th class="num">Tool calls</th><th class="num">Time / episode</th><th class="num">Repaired JSON</th><th class="num">Cost</th></tr>
    ${runs
      .map((r, i) => {
        const c = r.components || {};
        const ci = r.pass_at_1_ci95;
        return `<tr class="click${r.earlier ? " earlier" : ""}" data-run="${esc(r.run)}">
        <td class="num faint">${r.earlier ? "" : i + 1}</td>
        <td><div>${esc(r.label)}${r.earlier ? ` <span class="tag">earlier attempt</span>` : ""}</div><div class="mono xs faint">${esc(r.folder)}</div></td>
        <td>${statusHtml(r)}</td>
        <td class="num"><b>${pct(r.pass_at_1)}</b> <span class="faint sm">${r.pass_at_1 == null ? "" : `${Math.round(r.pass_at_1 * r.graded)}/${r.graded}`}</span>
          ${ci && r.graded ? `<div class="faint xs">95% CI ${pct(ci[0])}–${pct(ci[1])}</div>` : ""}</td>
        <td class="num">${pct(r.exact_route_rate)}</td>
        <td>${meter(r.mean_reward)} <span class="sm">${num(r.mean_reward, 2)}</span></td>
        <td class="num">${pct(c.steps)}</td><td class="num">${pct(c.stock)}</td>
        <td class="num">${num(r.mean_tool_calls, 1)}</td><td class="num">${secs(r.mean_latency_seconds)}</td>
        <td class="num">${pct(r.coerced_submission_rate)}</td>
        <td class="num">${r.cost_usd ? `$${num(r.cost_usd, 2)}` : `<span class="faint">–</span>`}</td></tr>`;
      })
      .join("")}
  </table></div>`;
}

function gridHtml(runs, data, sort) {
  const solvedBy = (task) => runs.filter((r) => data.cells[r.run]?.[task.id]?.valid).length;
  const tasks = [...data.tasks];
  if (sort === "hard") tasks.sort((a, b) => solvedBy(a) - solvedBy(b));
  return `<div class="tbl egrid"><table>
    <tr><th>Task</th>${runs.map((r) => `<th class="vert" title="${esc(r.folder)}"><a href="${evalHref("/run", { run: r.run })}">${esc(r.label)}</a></th>`).join("")}<th class="num">Solved by</th></tr>
    ${tasks
      .map(
        (t) => `<tr><td><div class="gtask">${t.smiles ? mol(t.smiles, 160, 90) : ""}<div><a class="link mono xs" href="#/task/${encodeURIComponent(t.id)}">${esc(t.id)}</a>
          <div class="faint xs">${esc(t.tier || "")}${t.max_depth ? ` · depth ≤ ${t.max_depth}` : ""}</div></div></div></td>
        ${runs.map((r) => `<td class="c">${cellHtml(r, t, data.cells[r.run]?.[t.id])}</td>`).join("")}
        <td class="num">${solvedBy(t)} / ${runs.length}</td></tr>`,
      )
      .join("")}
  </table></div>
  <p class="faint xs" style="margin-top:6px">★ passed and matched the patent route · ✓ passed · number: partial reward × 100 · 0: no reward · dot: provider error · empty: not run yet. Click a cell to read the episode.</p>`;
}

async function evalsView(view, params) {
  const { sets } = await api("evals");
  if (!sets.length) {
    view.innerHTML = `<p class="note">No eval runs under <span class="mono">runs/</span> yet. Start one with <span class="mono">uv run python eval/run_eval.py --output runs/&lt;set&gt;/&lt;model&gt; …</span></p>`;
    return;
  }
  const saved = store.get("evalSet", "");
  const name = [params.set, saved].find((s) => sets.some((x) => x.name === s)) || sets[0].name;
  store.set("evalSet", name);
  const showAll = params.all === "1";
  const render = async () => {
    const data = await api("evals/board", { name });
    const marked = markEarlier(data.runs).sort((a, b) => a.earlier - b.earlier || runOrder(a, b));
    const shown = showAll ? marked : marked.filter((r) => !r.earlier);
    const live = data.runs.filter((r) => r.status === "running");
    view.innerHTML = `
      <div class="toolbar">
        <select id="evalset" aria-label="Eval set">${sets.map((s) => `<option value="${esc(s.name)}"${s.name === name ? " selected" : ""}>${esc(s.name)} (${s.runs} runs)</option>`).join("")}</select>
        <label class="sm"><input type="checkbox" id="all"${showAll ? " checked" : ""}> show earlier attempts (${marked.filter((r) => r.earlier).length})</label>
        <span class="grow"></span>
        ${live.length ? `<span class="sm live"><span class="spinner"></span>${live.length} run${live.length > 1 ? "s" : ""} in progress · refreshing every ${REFRESH_MS / 1000} s</span>` : ""}
      </div>
      <div class="section" style="margin-top:0"><h2>Leaderboard <span class="faint sm">· ${data.tasks.length} tasks · click a model for its episodes</span></h2>${leaderboardHtml(shown)}</div>
      <div class="section"><h2>Which model solved which task
        <select id="gsort" aria-label="Task order" style="margin-left:8px"><option value="">task order</option><option value="hard"${params.sort === "hard" ? " selected" : ""}>fewest solvers first</option></select></h2>
        ${gridHtml(shown, data, params.sort)}</div>`;
    $("#evalset").addEventListener("change", (e) => go("/evals", { ...params, set: e.target.value }));
    $("#all").addEventListener("change", (e) => go("/evals", { ...params, set: name, all: e.target.checked ? 1 : "" }));
    $("#gsort").addEventListener("change", (e) => go("/evals", { ...params, set: name, sort: e.target.value }));
    view
      .querySelectorAll("tr[data-run]")
      .forEach((row) => row.addEventListener("click", () => go("/evals/run", { run: row.dataset.run })));
    refreshWhileLive(live.length > 0, render);
  };
  await render();
}

function evalCrumbs(summary, tail = "") {
  const set = summary.run.split("/").slice(0, -1).join("/");
  const runLink = tail ? `<a href="${evalHref("/run", { run: summary.run })}">${esc(summary.label)}</a>` : `<span>${esc(summary.label)}</span>`;
  return `<div class="crumbs"><a href="${evalHref("", { set })}">Evals</a><span>/</span><a href="${evalHref("", { set })}">${esc(set)}</a><span>/</span>${runLink}${tail ? `<span>/</span>${tail}` : ""}</div>`;
}

function runFactsHtml(s) {
  const sampling = Object.entries(s.sampling || {})
    .filter(([, v]) => v != null)
    .map(([k, v]) => `${k} ${v}`)
    .join(" · ");
  return `<dl class="kv">
    <dt>Model</dt><dd class="mono">${esc(s.model)} <span class="faint">via ${esc(s.provider)}</span></dd>
    <dt>Status</dt><dd>${statusHtml(s)}</dd>
    <dt>Solved</dt><dd><b>${pct(s.pass_at_1)}</b>${s.pass_at_1_ci95 && s.graded ? ` <span class="faint">(95% CI ${pct(s.pass_at_1_ci95[0])}–${pct(s.pass_at_1_ci95[1])})</span>` : ""} · exact patent route ${pct(s.exact_route_rate)}</dd>
    <dt>Mean reward</dt><dd>${num(s.mean_reward, 3)}</dd>
    <dt>Per episode</dt><dd>${num(s.mean_tool_calls, 1)} tool calls · ${secs(s.mean_latency_seconds)}</dd>
    <dt>Submissions</dt><dd>${pct(s.coerced_submission_rate)} needed JSON repair · ${pct(s.no_emit_rate)} never submitted</dd>
    <dt>Budget</dt><dd>${s.max_turns} model turns · ${s.max_tool_calls} tool calls</dd>
    <dt>Sampling</dt><dd class="sm">${esc(sampling || "provider defaults")}</dd>
    <dt>Cost</dt><dd>${s.cost_usd ? `$${num(s.cost_usd, 2)}` : `<span class="faint">not reported by the provider</span>`}</dd>
  </dl>`;
}

function outcomeHtml(e) {
  if (!e.graded) return `<span class="faint sm"><i class="dot"></i>provider error</span>`;
  if (e.valid) return `<span class="ok sm"><i class="dot"></i>${e.exact ? "passed · patent route" : "passed"}</span>`;
  return `<span class="${e.reward ? "warn" : "err"} sm"><i class="dot"></i>failed</span>`;
}

async function evalRunView(view, params) {
  const render = async () => {
    const data = await api("evals/run", { run: params.run });
    const s = data.summary;
    const parts = Object.entries(s.components || {}).sort((a, b) => (META.weights[b[0]] || 0) - (META.weights[a[0]] || 0));
    view.innerHTML = `${evalCrumbs(s)}
      <div class="grid" style="grid-template-columns:minmax(0,5fr) minmax(0,4fr) minmax(0,5fr)">
        <div class="panel"><div class="panel-h"><h2>${esc(s.label)}</h2></div><div class="panel-b">${runFactsHtml(s)}</div></div>
        <div class="panel"><div class="panel-h"><h3>Reward components <span class="faint sm">(mean score × weight)</span></h3></div><div class="panel-b">${
          parts.length ? bars(parts.map(([k, v]) => [`${k} × ${num(META.weights[k], 2)}`, v, num(v, 2)])) : `<p class="faint sm">No graded episodes yet.</p>`
        }</div></div>
        <div class="panel"><div class="panel-h"><h3>Why the ${data.failed} failed episodes failed</h3></div><div class="panel-b">${
          data.failures.length ? bars(data.failures.map(([k, n]) => [k, n])) : `<p class="faint sm">No failed episodes.</p>`
        }</div></div>
      </div>
      <div class="section"><h2>Episodes <span class="faint sm">· click one to read the trajectory</span></h2>
        <div class="tbl"><table>
          <tr><th></th><th>Task</th><th>Tier</th><th class="num">Depth budget</th><th>Outcome</th><th class="num">Reward</th><th class="num">Tool calls</th><th class="num">Turns</th><th class="num">Time</th><th>First problem</th></tr>
          ${data.episodes
            .map(
              (e) => `<tr class="click" data-task="${esc(e.task_id)}" data-attempt="${e.attempt}"><td>${mol(e.smiles, 96, 52, "thumb")}</td><td class="mono sm">${esc(e.task_id)}</td>
              <td class="sm">${esc(e.tier || "–")}</td><td class="num">${e.max_depth ?? "–"}</td><td>${outcomeHtml(e)}</td><td class="num">${num(e.reward, 2)}</td>
              <td class="num">${e.tool_calls ?? "–"}</td><td class="num">${e.turns ?? "–"}</td><td class="num">${secs(e.latency)}</td>
              <td class="sm">${esc(e.failure)}${e.coerced ? ` <span class="tag">JSON repaired</span>` : ""}${e.auto_emitted ? ` <span class="tag">never submitted</span>` : ""}</td></tr>`,
            )
            .join("")}
        </table></div></div>`;
    view
      .querySelectorAll("tr[data-task]")
      .forEach((row) =>
        row.addEventListener("click", () =>
          go("/evals/episode", { run: params.run, task: row.dataset.task, attempt: row.dataset.attempt }),
        ),
      );
    refreshWhileLive(s.status === "running", render);
  };
  await render();
}

// A submitted molecule/reaction tree in the shape route-graph.js draws.
function treeToRoute(tree, stock) {
  const steps = [];
  const walk = (node) => {
    const reaction = (node?.children || [])[0];
    if (!reaction) return;
    const md = reaction.metadata || {};
    const kids = (reaction.children || []).filter((k) => typeof k?.smiles === "string");
    const label = md.reaction_class || md.classification || "unlabelled";
    steps.push({
      product: node.smiles,
      reactants: kids.map((k) => k.smiles),
      family: label,
      phrase: label,
      text: md.explanation || "The model gave no explanation.",
    });
    kids.forEach(walk);
  };
  walk(tree);
  return { kind: "submitted", steps, patents: [], in_stock: stock };
}

function longText(text, limit = 700) {
  if (text.length <= limit) return `<p class="said">${esc(text)}</p>`;
  return `<p class="said">${esc(text.slice(0, limit))}…</p><details><summary>Show all ${fmt(text.length)} characters</summary><p class="said">${esc(text)}</p></details>`;
}

function parseJson(text, fallback) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function blockHtml(kind, label, body, extra = "") {
  return `<div class="block ${kind}"><div class="btag">${esc(label)}${extra}</div>${body}</div>`;
}

function argsBlock(name, args) {
  return `<div class="block call"><div class="btag">tool call <span class="mono">${esc(name)}</span></div>
    <ul class="calls" style="margin:0"><li>${callLabel(name, args)}</li></ul>
    <details><summary>Arguments</summary><pre>${esc(JSON.stringify(args, null, 2))}</pre></details></div>`;
}

function resultBlock(name, out) {
  const summary = out ? resultLine(name, out) : `<li class="faint">no result recorded</li>`;
  return `<div class="block result"><div class="btag">tool result <span class="mono">${esc(name)}</span></div>
    <ul class="calls" style="margin:0">${summary}</ul>
    ${out ? `<details><summary>Full result</summary><pre>${esc(JSON.stringify(out, null, 2))}</pre></details>` : ""}</div>`;
}

function routeSummary(submission) {
  const routes = Array.isArray(submission?.routes) ? submission.routes : null;
  if (!routes) return `<span class="warn">not a route object</span>`;
  if (!routes.length) return `<span class="warn">empty submission</span>`;
  return `${routes.length} route tree${routes.length === 1 ? "" : "s"} <span class="faint">· drawn below</span>`;
}

// Reshape the stored transcript into turn-level blocks: thinking, text, tool calls, their results, and the final answer.
function turnBlocks(transcript) {
  const results = new Map(transcript.filter((m) => m.role === "tool").map((m) => [m.tool_call_id, m]));
  const turns = [];
  let turn = 0;
  const nudges = [];
  for (const m of transcript) {
    if (m.role === "tool") continue;
    if (m.role !== "assistant") {
      if (m.role === "user" && turn > 0) nudges.push(m.content || "");
      continue;
    }
    turn += 1;
    const text = typeof m.content === "string" ? m.content.trim() : "";
    const calls = (m.tool_calls || []).map((call) => ({
      name: call.function?.name || "?",
      args: parseJson(call.function?.arguments || "{}", { raw: call.function?.arguments }),
      result: results.get(call.id),
    }));
    turns.push({ turn, thinking: m.reasoning || "", text, calls, nudges: nudges.splice(0) });
  }
  return turns;
}

function turnHtml(t) {
  const blocks = [];
  t.nudges.forEach((n) => blocks.push(blockHtml("nudge", "harness nudge", longText(n))));
  if (t.thinking) {
    const chars = fmt(t.thinking.length);
    const brief = t.thinking.length > 320 ? `<p class="said">${esc(t.thinking.slice(0, 320))}…</p><details><summary>Show all ${chars} characters</summary><p class="said">${esc(t.thinking)}</p></details>` : `<p class="said">${esc(t.thinking)}</p>`;
    blocks.push(blockHtml("think", "model thinking", brief, ` <span class="faint xs">· ${chars} chars</span>`));
  }
  const emit = t.calls.find((c) => c.name === "emit_routes");
  const regular = t.calls.filter((c) => c.name !== "emit_routes");
  if (t.text) blocks.push(blockHtml("say", "model reply", longText(t.text)));
  for (const call of regular) {
    blocks.push(argsBlock(call.name, call.args));
    blocks.push(
      resultBlock(call.name, call.result ? parseJson(call.result.content, { result: call.result.content }) : null),
    );
  }
  if (!t.calls.length && !t.text && !t.thinking)
    blocks.push(blockHtml("empty", "empty reply", `<p class="faint sm">Nothing: no thinking, no text, no tool call.</p>`));
  if (emit) {
    const parsed = emit.result ? parseJson(emit.result.content, {}) : {};
    blocks.push(
      blockHtml(
        "answer",
        "answer · emit_routes",
        `<p class="sm" style="margin:0">${routeSummary(emit.args.submission)}</p>${
          parsed.score ? `<p class="sm" style="margin:4px 0 0">Scored ${num(parsed.score.reward)} · ${parsed.score.valid ? "passed" : "did not pass"}</p>` : ""
        }<details><summary>Submission</summary><pre>${esc(JSON.stringify(emit.args, null, 2))}</pre></details>`,
      ),
    );
  }
  const toolCount = regular.length + (emit ? 1 : 0);
  const tag = toolCount ? `· ${toolCount} tool call${toolCount === 1 ? "" : "s"}` : "· no tool call";
  return `<article class="turn"><header class="turn-h"><span class="turn-n">Turn ${t.turn}</span><span class="faint sm">${tag}${t.thinking ? ` · ${fmt(t.thinking.length)} chars of thinking` : ""}</span></header>
    <div class="turn-b">${blocks.join("")}</div></article>`;
}

function transcriptHtml(transcript, prompt) {
  const turns = turnBlocks(transcript);
  const promptBlock = prompt
    ? `<article class="turn"><header class="turn-h"><span class="turn-n">Prompt</span><span class="faint sm">· what the model started with</span></header>
       <div class="turn-b">${blockHtml("prompt", "task prompt", longText(prompt, 1200))}</div></article>`
    : "";
  return `<div class="trajectory">${promptBlock}${turns.map(turnHtml).join("")}</div>`;
}

function sameTaskHtml(others, current) {
  if (!others?.length) return "";
  return `<div class="panel"><div class="panel-h"><h3>Same task across models <span class="faint sm">· newest run per model</span></h3></div>
    <div class="tbl" style="border:0"><table>${others
      .map(
        (o) => `<tr class="click${o.current ? " on" : ""}" data-run="${esc(o.run)}">
          <td>${esc(o.label)}${o.current ? ` <span class="tag">this one</span>` : ""}</td>
          <td class="num"><span class="meter"><i style="width:${(100 * (o.reward || 0)).toFixed(0)}%"></i></span></td>
          <td class="num sm">${num(o.reward, 2)}</td>
          <td class="sm">${o.valid ? `<span class="ok"><i class="dot"></i>${o.exact ? "patent" : "passed"}</span>` : o.graded ? `<span class="warn"><i class="dot"></i>failed</span>` : `<span class="faint"><i class="dot"></i>error</span>`}</td></tr>`,
      )
      .join("")}</table></div></div>`;
}

async function evalEpisodeView(view, params) {
  const attempt = params.attempt || 0;
  const data = await api("evals/episode", { run: params.run, task: params.task, attempt });
  const s = data.summary,
    e = data.episode,
    usage = e.usage || {};
  const routes = Array.isArray(e.submission?.routes) ? e.submission.routes : null;
  const nav = (task, label) =>
    task ? `<a class="btn" href="${evalHref("/episode", { run: params.run, task, attempt })}">${label}</a>` : `<span class="btn" aria-disabled="true">${label}</span>`;
  const score = {
    reward: e.reward,
    valid: e.valid,
    components: e.components,
    hard_failures: e.hard_failures,
    verification_tier: e.verification_tier,
  };
  const turns = turnBlocks(e.transcript);
  const thinkChars = turns.reduce((n, t) => n + (t.thinking?.length || 0), 0);
  view.innerHTML = `${evalCrumbs(s, `<span class="mono">${esc(e.task_id)}</span>`)}
    <div class="toolbar">${nav(data.prev, "← Previous episode")}${nav(data.next, "Next episode →")}<span class="grow"></span>
      ${data.has_task ? `<a class="btn" href="#/task/${encodeURIComponent(e.task_id)}?reveal=1">Task page with the known routes</a>` : ""}</div>
    <div class="split">
      <div class="panel"><div class="panel-h"><h2>Task</h2><span class="grow"></span><span class="faint sm mono">${esc(e.variant || "")}</span></div>
        <div class="panel-b"><div class="molbox">${mol(e.target_smiles, 360, 170)}</div>
          <dl class="kv" style="margin-top:12px">
            <dt>Target</dt><dd class="mono">${esc(e.target_smiles)}</dd>
            <dt>Model</dt><dd>${esc(s.label)} <span class="faint">(${esc(s.model)})</span></dd>
            <dt>Split · tier</dt><dd>${esc(e.split)} · ${esc(e.tier || "–")}</dd>
            <dt>Depth budget</dt><dd>${e.max_depth ?? "–"} reactions</dd>
            <dt>Outcome</dt><dd>${outcomeHtml({ ...e, exact: e.exact_match })} · reward ${num(e.reward, 3)}</dd>
            <dt>Effort</dt><dd>${e.tool_calls} tool calls over ${e.turns} turns · ${secs(usage.latency_seconds)} · ${fmt((usage.prompt_tokens || 0) + (usage.completion_tokens || 0))} tokens${thinkChars ? ` · ${fmt(thinkChars)} chars of thinking` : ""}</dd>
            <dt>Submission</dt><dd>${routes ? `${routes.length} route tree${routes.length === 1 ? "" : "s"}` : "not a route object"}${e.submission_coerced ? " · JSON repaired by the harness" : ""}${e.auto_emitted ? " · the model never submitted; the harness closed the episode" : ""}</dd>
          </dl>
          ${(e.errors || []).length ? `<h3 style="margin:14px 0 6px">Harness notes</h3><ul class="calls">${e.errors.map((x) => `<li class="warn"><i class="dot"></i>${esc(x.slice(0, 400))}</li>`).join("")}</ul>` : ""}
        </div></div>
      <div class="stack">
        ${rewardHtml({ score })}
        ${sameTaskHtml(data.same_task, params.run)}
      </div>
    </div>
    <div class="section"><h2>Submitted routes <span class="faint sm">· leaf colours show the real stock, whatever the model claimed</span></h2>
      ${
        routes?.length
          ? routes.map((_, i) => `<h3 style="margin:${i ? "16px" : "0"} 0 8px">Route ${i + 1}</h3><div data-sroute="${i}"></div>`).join("")
          : `<div class="note">${routes ? "The submission had no route trees." : `The submission was not a route object.<details><summary>Raw submission</summary><pre>${esc(JSON.stringify(e.submission, null, 2))}</pre></details>`}</div>`
      }</div>
    <div class="section"><h2>Trajectory <span class="faint sm">· ${turns.length} turn${turns.length === 1 ? "" : "s"}</span></h2>
      ${transcriptHtml(e.transcript, e.prompt)}</div>`;
  view
    .querySelectorAll("tr[data-run]")
    .forEach((row) => row.addEventListener("click", () => go("/evals/episode", { run: row.dataset.run, task: params.task, attempt })));
  view.querySelectorAll("[data-sroute]").forEach((host) => {
    const tree = routes[Number(host.dataset.sroute)];
    if (typeof tree?.smiles === "string") mountRouteGraph(host, treeToRoute(tree, data.stock), tree.smiles);
    else host.innerHTML = `<pre>${esc(JSON.stringify(tree, null, 2))}</pre>`;
  });
}
