// The MiMo release's own views (app/envs/mimo.py), as the MiMo RL Environment Explorer draws them: the agent's prompt,
// the systems a General task works through and their data, its workspace files with previews, the grader in full
// (hidden tests, the crash to reproduce, rubric checks and the judge prompt, the screenshot judge's bands, the music
// scorer's features), and the reward design across the release.
//   render(kind, data, ctx) -> HTML        wire(root, ctx): the clicks (previews, tables, the prompt's fold and copy)
import { $, $$, api, esc, fmt, md, bytes, sk, toast, openModal, progress, emptyState, table, sheetGrid } from "../util.js";
import { icon, DOMAIN_ICON, FILE_ICON } from "../icons.js";

const enc = (s) => s.split("/").map(encodeURIComponent).join("/");
const KIND_LABEL = { spreadsheet: "Excel", document: "Word", pdf: "PDF", slides: "PowerPoint", image: "image", web: "web page", text: "text", archive: "archive" };
const SHAPE_IN = { both: "the systems' databases and the workspace files", db: "the systems' databases", workspace: "the workspace files" };
const SHAPE_ACT = { none: "change nothing: its final answer is what gets graded", mutate_db: "update records in the systems' databases",
  edit_workspace: "edit files in the workspace", both: "update the databases and edit workspace files" };
const pre = (t) => `<pre class="code-block wrap"><code>${esc(t)}</code></pre>`;
const pct = (x) => `${Math.round(x * 100)}%`;
const disclose = (label, inner) => `<details style="margin-top:12px"><summary class="disclose">${icon("chevronRight", 14, "chev")}${esc(label)}</summary>${inner}</details>`;
const fmtNum = (x) => (x == null ? "" : Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(2) : x.toFixed(3));

export function render(kind, data, ctx) {
  return ({ prompt, systems, workspace, grading, rewards }[kind] || (() => ""))(data, ctx);
}

// ── the agent's first message ────────────────────────────────────────────────
function prompt(p) {
  const chars = p.parts.reduce((n, [, t]) => n + t.length, 0);
  const rows = [p.harness
    ? ["Harness", `OpenCode ${esc(p.harness.version)}, which adds its own system prompt and tool definitions`]
    : ["Harness", "None. One chat completion: this message is the whole conversation, with no system prompt and no tools"]];
  if (p.harness) rows.push(["Tools", `OpenCode's own${p.systems ? `, plus the tools of the ${p.systems} systems over MCP` : ""}; web search and web fetch are denied`]);
  if (p.steps) rows.push(["Step limit", `${fmt.format(p.steps)} model calls`]);
  if (p.timeout_min) rows.push(["Time limit", `${p.timeout_min} minutes`]);
  if (p.max_tokens) rows.push(["Reply length", `up to ${fmt.format(p.max_tokens)} tokens${p.harness ? " per model call" : ""}`]);
  const body = p.parts.map(([kind, text]) => kind === "task" && p.task_is_brief
    ? `<button class="pr-task" type="button" data-pr-expand title="Show it here">‹the task above, ${fmt.format(text.length)} characters›</button><span hidden>${esc(text)}</span>`
    : esc(text)).join("");
  return `<dl class="kv">${rows.map(([k, x]) => `<dt>${k}</dt><dd>${x}</dd>`).join("")}</dl>
    <div class="pr-h"><span class="sec-label">The message · ${fmt.format(chars)} characters</span>
      <button class="btn sm" type="button" data-pr-copy>${icon("copy", 13)}Copy</button></div>
    <pre class="code-block wrap pr" data-pr-text="${esc(p.parts.map(([, t]) => t).join(""))}">${body}</pre>
    <p class="muted xs" style="margin-top:8px">Limits are the training harness's defaults; the run panel can change them.</p>`;
}

// ── systems: MCP servers and their databases ─────────────────────────────────
function systems(list) {
  return `<p class="muted sm" style="margin-bottom:12px">The agent can only reach this data through the tools below. Open a table to see the rows it starts with.</p>
    <div class="systems">${list.map((s, i) => `
    <details class="sys" ${i === 0 ? "open" : ""}>
      <summary>${icon("chevronRight", 14, "chev")}<b>${esc(s.label)}</b><span class="meta">${s.tools.length} tools · ${s.tables.length} tables · ${s.tables.reduce((n, t) => n + t.rows, 0).toLocaleString()} rows</span></summary>
      <div class="sys-in">
        <div><div class="sec-label">Tools</div>
          <ul class="tools">${s.tools.map((t) => `<li><code class="sig">${esc(t.name)}(<span>${t.params.map((p) => esc(p.name) + (p.default != null ? "?" : "")).join(", ")}</span>)</code>
            ${t.doc ? `<p>${esc(t.doc)}</p>` : ""}</li>`).join("")}</ul></div>
        <div><div class="sec-label">Data</div>
          <div class="tables">${s.tables.map((t) => `<button class="tbl-btn" type="button" data-sys="${esc(s.name)}" data-label="${esc(s.label)}" data-table="${esc(t.table)}">${icon("table", 15)}
            <b>${esc(t.table)}</b><span>${t.rows.toLocaleString()} × ${t.columns.length}</span></button>`).join("")}</div></div>
      </div>
    </details>`).join("")}</div>`;
}

// ── workspace files ──────────────────────────────────────────────────────────
function workspace(files) {
  return `<div class="files-grid">${files.map((f) => `<button class="file k-${esc(f.kind)}" type="button" data-path="${esc(f.path)}" data-kind="${esc(f.kind)}" title="${esc(f.path)}">
    <span class="fi">${icon(FILE_ICON[f.kind] || "file", 16)}</span><span class="fx"><span class="fn">${esc(f.path).replace(/([_/.-])/g, "$1<wbr>")}</span><span class="fs">${esc(KIND_LABEL[f.kind] || f.kind)} · ${bytes(f.size)}</span></span></button>`).join("")}</div>`;
}

// ── the grader ───────────────────────────────────────────────────────────────
function diffHtml(patch) {
  return patch.split("\n").slice(0, 4000).map((l) => {
    const cls = l.startsWith("+++") || l.startsWith("---") || l.startsWith("diff ") ? "dh" : l.startsWith("+") ? "da" : l.startsWith("-") ? "dd" : l.startsWith("@@") ? "dc" : "";
    return `<span class="${cls}">${esc(l)}</span>`;
  }).join("\n");
}

function grading(g) {
  let body = `<p class="lead">${esc(g.summary)}</p>`;
  if (g.steps) body += `<ol class="steps">${g.steps.map((s) => `<li><span>${md(s).replace(/^<p>|<\/p>$/g, "")}</span></li>`).join("")}</ol>`;
  if (g.formula) body += `<p class="formula"><code>${esc(g.formula)}</code></p>`;
  if (g.kind === "tests") {
    body += `<div class="sec-label" style="margin-top:18px">Hidden tests</div><div class="tbl"><table><thead><tr><th>File</th><th>Lines</th><th></th></tr></thead><tbody>${g.files.map((f) =>
      `<tr><td><code>${esc(f.path)}</code></td><td class="pm"><span class="a">+${f.added}</span> <span class="d">−${f.removed}</span></td><td>${f.new ? '<span class="chip">new</span>' : ""}</td></tr>`).join("")}</tbody></table></div>`;
    if (g.script) body += disclose("Test command script", `<pre class="code-block"><code>${esc(g.script)}</code></pre>`);
    if (g.patch) body += disclose(`Full hidden test patch (${g.patch.length.toLocaleString()} characters)`, `<pre class="code-block diff">${diffHtml(g.patch)}</pre>`);
  }
  if (g.kind === "crash" && g.expected) {
    const x = g.expected;
    body += `<div class="target">${[["Sanitizer", esc(x.sanitizer)], ["Bug class", esc(x.error_type)], ["Function", `<code>${esc(x.function)}</code>`], ["File", `<code>${esc(x.file)}</code>`]]
      .map(([k, val]) => `<div class="stat"><span>${k}</span><b>${val}</b></div>`).join("")}</div>`;
  }
  if (g.kind === "rubric") {
    const shape = [["Facts come from", SHAPE_IN[g.shape?.input]], ["The agent must", SHAPE_ACT[g.shape?.act]]].filter(([, x]) => x);
    if (shape.length) body += `<dl class="kv" style="margin-top:14px">${shape.map(([k, x]) => `<dt>${k}</dt><dd>${esc(x)}</dd>`).join("")}</dl>`;
    if (g.rules) body += `<ul class="rules">${g.rules.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    // verify.py counts a check with no weight as weight 1 (weights are relative, not percentages)
    const wt = (c) => (c.weight == null ? 1 : Number(c.weight));
    const tot = g.checks.reduce((s, c) => s + wt(c), 0) || 1;
    const top = Math.max(...g.checks.map(wt)) || 1;
    body += `<div class="sec-label" style="margin-top:18px">${g.checks.length} checks, weighted</div><ol class="checks">${g.checks.map((c) => {
      const share = Math.round((wt(c) / tot) * 100);
      return `<li><div class="ck-top"><span class="tier t-${esc(c.tier)}">${esc(c.tier || "")}</span>
      <span>${c.method === "llm" ? `judged by a model${c.files?.length ? ` · reads ${esc(c.files.join(", "))}` : ""}` : "checked by code"}</span>
      <span class="w" title="Share of the reward"><i><b style="width:${Math.round((wt(c) / top) * 100)}%"></b></i>${share}%</span></div>
      <p>${esc(c.question || c.id)}</p></li>`; }).join("")}</ol><p class="muted xs" style="margin-top:10px">The answer each check expects is not shown.</p>`;
    const bare = g.checks.filter((c) => c.weight == null);
    if (bare.length && bare.length < g.checks.length) {
      const set = g.checks.filter((c) => c.weight != null).reduce((s, c) => s + Number(c.weight), 0);
      body += `<p class="muted xs" style="margin-top:6px">The weights given add up to ${set.toFixed(2).replace(/\.?0+$/, "")}, and ${bare.length === 1 ? "one check has none" : `${bare.length} checks have none`}.
        verify.py counts a missing weight as 1, so ${bare.length === 1 ? "that check decides" : "those checks decide"} ${pct(bare.length / tot)} of the reward.</p>`;
    }
    const j = g.judge;
    if (j) {
      const how = `<p class="muted xs" style="margin:8px 0">The judge gets one such message per model-judged check. This is the one for <code>${esc(j.check)}</code>, with the expected answer withheld.</p>`;
      if (j.english) body += disclose("The judge prompt, in English", how + pre(j.english));
      body += disclose("The judge prompt as sent, in Chinese", (j.english ? "" : how) + pre(j.original));
    }
    if (g.grader) body += `<p class="muted xs" style="margin-top:10px">${esc(g.grader)}</p>`;
  }
  if (g.kind === "terminal") {
    body += `<div class="sec-label" style="margin-top:18px">Hidden test files</div>${table([["File", "Size"], ...g.files.map((f) => [f.path, bytes(f.size)])])}`;
    if (g.script) body += disclose("tests/test.sh", `<pre class="code-block"><code>${esc(g.script)}</code></pre>`);
    if (g.tests) body += disclose("tests/test_outputs.py", `<pre class="code-block"><code>${esc(g.tests)}</code></pre>`);
  }
  if (g.kind === "visual") {
    const parts = [["visual", "Visual quality", "mean of five criteria"], ["query_fulfillment", "Brief fulfilment", ""], ["premium_assets", "Asset quality", ""]];
    body += `<div class="parts">${parts.map(([key, name, how]) => {
      const ds = g.dims.filter((d) => d.group === key);
      return ds.length ? `<div class="rpart"><div class="rpart-h"><b>${name}</b><span>⅓ of the score${how ? ` · ${how}` : ""}</span></div>
        <dl>${ds.map((d) => `<div><dt>${esc(d.label)}</dt><dd>${esc(d.desc)}</dd></div>`).join("")}</dl></div>` : "";
    }).join("")}</div>`;
    const j = g.judge;
    if (j?.rules) body += `<div class="sec-label">The judge's instructions</div><ul class="rules" style="margin-top:0">${j.rules.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    if (j?.bands) body += disclose("Scoring bands for each criterion, from the judge prompt", `<div class="bands">${g.dims.map((d) => j.bands[d.key]
      ? `<div><b>${esc(d.label)}</b><ol>${j.bands[d.key].map((t, i) => `<li><span>${j.edges[i]}</span><span>${esc(t)}</span></li>`).join("")}</ol></div>` : "").join("")}</div>`);
    if (j?.why) body += `<div class="sec-label">Why these weights</div><ul class="rules" style="margin-top:0">${j.why.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    body += `<div class="sec-label">In training</div><p class="sm muted">${esc(j?.training || g.note)}</p>`;
    if (j?.original) body += disclose("The judge prompt as sent, in Chinese, with this task's brief", `<p class="muted xs" style="margin:8px 0">Sent with the full-page screenshot. The brief is cut to its first ${fmt.format(j.query_cap)} characters.</p>${pre(j.original)}`);
  }
  if (g.kind === "music") {
    const groups = {};
    g.features.forEach((f) => (groups[f.group] ||= []).push(f));
    body += `<div class="note-box gated">${icon("alert")}<span><b>Validity gate first.</b> No notation errors, fewer than 10 bars of the wrong length, no blank lines in the tune,
      and one instrument per MIDI channel. A piece that fails any of these scores 0.</span></div>
      <div class="feat-groups">${Object.entries(groups).map(([gname, fs]) => `<div><div class="sec-label">${esc(gname)}${g.weights?.[gname] != null ? ` · ${pct(g.weights[gname])}` : ""}</div>
      ${fs.map((f) => `<div class="feat"><b>${esc(f.label || f.name)}</b><span>${f.rule === "band" ? "within the human range" : f.rule === "high" ? "the higher the better" : "the lower the better"} · <code>${esc(f.name)}</code></span>
        ${f.band ? `<em>${fmtNum(f.band[0])} – ${fmtNum(f.band[1])}</em>` : ""}</div>`).join("")}</div>`).join("")}</div>`;
    if (g.curves) body += `<div class="sec-label">How a feature scores</div><dl class="kv">${[["Within the human range", g.curves.band], ["The higher the better", g.curves.high],
      ["The lower the better", g.curves.low], ["Histogram similarity", g.histograms]].filter(([, x]) => x).map(([k, x]) => `<dt>${k}</dt><dd>${esc(x)}</dd>`).join("")}</dl>
      <p class="muted xs" style="margin-top:8px">Group weights are the share of the 85% that the six groups make up together.</p>`;
  }
  if (g.not_scored) body += `<dl class="kv" style="margin-top:16px"><dt>Not scored</dt><dd>${esc(g.not_scored)}${/^Never/.test(g.not_scored) ? "" : " A rollout that isn't scored has no reward, rather than 0."}</dd></dl>`;
  body += `<p class="xs" style="margin-top:14px"><a href="/d/XiaomiMiMo/MiMo-V2.6-RL-oss?rewards=1">Every kind of task side by side, and how the rubrics are built across the dataset</a></p>`;
  return body;
}

// ── reward design, across the release ────────────────────────────────────────
const VERL = "https://github.com/XiaomiMiMo/verl/tree/a2ad9f6";
const MIMOAGENT = "https://github.com/XiaomiMiMo/mimoagent/tree/467f0a1";
const KINDS = [
  { k: "code", d: "code", name: "Code", reward: "1 if the hidden tests pass, otherwise 0", range: "0 or 1",
    by: "The task's hidden tests, applied after the agent finishes and run in its own image", judge: "–",
    ns: "The tests can't be applied, or the files they touch can't be reset", train: "Same" },
  { k: "cyber", d: "cyber", name: "Cyber", reward: "1 if the last proof of concept crashes in the expected function, otherwise 0", range: "0 or 1",
    by: "A root-owned server runs the real fuzz target on it, under the sanitizer", judge: "–", ns: "–", train: "Same" },
  { k: "general-workplace", d: "general", name: "General, workplace", reward: "Σ weight × score ÷ Σ weight, over the task's rubric checks", range: "0 to 1",
    by: "Code checks on the systems' databases and the workspace, and a text judge: one 0-or-1 call per check", judge: "text",
    ns: "The judge can't be reached, or one of the task's systems stops responding", train: "The same verify.py; it first looks for Xiaomi's internal grader, not released" },
  { k: "general-terminal", d: "general", name: "General, terminal", reward: "1 if every test passes, otherwise 0", range: "0 or 1",
    by: "The task's pytest suite, copied in after the agent finishes", judge: "–",
    ns: "The tests crash before grading, or write no reward", train: "No reference harness in the release; follows the Terminal-Bench convention" },
  { k: "webdev", d: "webdev", name: "Webdev", reward: "mean(visual, brief fulfilment, asset quality); visual is the mean of five criteria", range: "0 to 1",
    by: "A vision judge scores a full-page render against a five-band scale per criterion", judge: "vision",
    ns: "The page can't be rendered, or the judge can't be reached", train: "Different: a relative pick among up to 8 sibling rollouts, minus a deduction for missing the brief" },
  { k: "music", d: "music", name: "Music", reward: "(0.85 × weighted feature groups + 0.15 × histogram similarity) ÷ 100", range: "0 to 1",
    by: "abc2midi, then 18 features compared with the range of human music", judge: "–",
    ns: "Never; a piece that can't be read or measured scores 0", train: "Same" },
];

function rewards({ kinds, stats }, ctx) {
  const share = (x, of) => (of ? `${Math.round((x / of) * 100)}%` : "–");
  const n = (x) => fmt.format(x || 0);
  const g = stats.general;
  const rows = KINDS.map((k) => {
    const c = kinds[k.k] || { count: 0 };
    return `<tr><td><span class="rw-dom">${icon(DOMAIN_ICON[k.d], 13)}${esc(k.name)}</span>${c.example ? `<br><a class="xs" href="/t/${enc(ctx.spec)}/${encodeURIComponent(c.example)}">an example</a>` : ""}</td>
      <td class="num nw">${n(c.count)}</td><td>${esc(k.reward)}</td><td class="nw">${esc(k.range)}</td><td>${esc(k.by)}</td><td class="nw">${esc(k.judge)}</td>
      <td>${esc(k.ns)}</td><td>${esc(k.train)}</td></tr>`;
  }).join("");
  const m = g.method || {}, t = g.tier || {}, mix = g.mix || {}, inp = g.input || {}, act = g.act || {};
  const judged = m.llm || 0;
  const facts = [
    ["Checks", `${n(g.checks)} across ${n(g.tasks)} tasks, ${g.per_task.min} to ${g.per_task.max} per task (median ${g.per_task.median})`],
    ["Judged by a model", `${n(judged)} (${share(judged, g.checks)}), each in its own call, answered 0 or 1`],
    ["Checked by code", `${n(m.rule)} (${share(m.rule, g.checks)}), scored 0 to 1 against the databases and the workspace`],
    ["Tasks", `${n(mix.judge_only)} graded by the judge alone, ${n(mix.both)} by code and the judge, ${n(mix.code_only)} by code alone`],
    ["Judge's share of a task's reward", `median ${share(g.judge_share_median, 1)}`],
    ["Tiers", `critical ${n(t.critical)} (${share(t.critical, g.checks)}), important ${n(t.important)} (${share(t.important, g.checks)}), sanity ${n(t.sanity)} (${share(t.sanity, g.checks)}). A label only: tiers don't enter the formula`],
    ["Weights", `set in ${n(g.weighted_tasks)} tasks; the rest weight every check equally (a check without a weight counts 1)`],
    ...(g.mixed_weights ? [["Mixed weights", `${n(g.mixed_weights)} tasks give some checks weights that add up to about 1 and leave others without one, which count 1 each.
      In ${n(g.mixed_heavy)} of them the unweighted checks decide 40% or more of the reward`]] : []),
    ["Facts come from", `the systems' databases in ${n(inp.db)} tasks, the workspace files in ${n(inp.workspace)}, both in ${n(inp.both)}`],
    ["The agent must", `change nothing in ${n(act.none)} tasks (its answer is graded), update databases in ${n(act.mutate_db)}, edit workspace files in ${n(act.edit_workspace)}, both in ${n(act.both)}`],
    ["What the judge reads", (g.judged_files || []).map(([f, c]) => `${esc(f)} for ${n(c)} checks`).join(", ") + ", as extracted text, up to 20,000 characters per file"],
    ["Gate and tamper checks", `supported by verify.py, used by ${n(g.uses_gates)} tasks`],
  ];
  return `<div class="sec-label">By kind of task</div>
    <div class="tbl rw-tbl"><table><thead><tr><th>Task</th><th class="num nw">Tasks</th><th>Reward</th><th class="nw">Range</th><th>Graded by</th><th class="nw">Judge</th>
      <th>Not scored when</th><th>In training</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="sec-label" style="margin-top:18px">General rubrics, across the dataset <span class="faint">counted from every task's verifier_meta.json</span></div>
    <dl class="kv">${facts.map(([k, x]) => `<dt>${esc(k)}</dt><dd>${x}</dd>`).join("")}</dl>
    <div class="sec-label" style="margin-top:18px">Everywhere</div>
    <ul class="rules" style="margin:0">
      <li>A failure that isn't the model's (the testbed, the render, the judge) is not scored: the rollout has no reward, rather than 0. Xiaomi's harness masks these the same way.</li>
      <li>Nothing that grades a task is in the sandbox while the agent works. Hidden tests, rubrics and the expected crash are uploaded after it finishes.</li>
      <li>The judge prompts are sent in Chinese, unchanged, so scores stay comparable with Xiaomi's. Each task's Grading section has its prompt with an English translation.</li>
      <li>Expected answers are never shown: not the rubric answers, not the code that checks them, not the Cyber proof of concept.</li>
      <li>Graders follow <a href="${VERL}" target="_blank" rel="noopener">XiaomiMiMo/verl a2ad9f6</a> and <a href="${MIMOAGENT}" target="_blank" rel="noopener">mimoagent 467f0a1</a>; the vendored files are unmodified.</li>
    </ul>`;
}

// ── clicks: the prompt, previews, tables ─────────────────────────────────────
export function wire(root, ctx) {
  root.addEventListener("click", async (ev) => {
    const fold = ev.target.closest("[data-pr-expand]");
    if (fold) { fold.nextElementSibling.hidden = false; fold.remove(); return; }
    const cp = ev.target.closest("[data-pr-copy]");
    if (cp) {
      const text = cp.closest(".bl-custom")?.querySelector("[data-pr-text]")?.dataset.prText || "";
      try { await navigator.clipboard.writeText(text); toast("Message copied"); } catch { toast("Couldn't copy"); }
      return;
    }
    const f = ev.target.closest(".bl-custom .file[data-path]");
    if (f) return previewFile(ctx, f.dataset.path, f.dataset.kind);
    const t = ev.target.closest(".bl-custom .tbl-btn");
    if (t) return previewTable(ctx, t.dataset.sys, t.dataset.table, t.dataset.label);
  });
}

const dataUrl = (ctx, part, params) => `/api/env/${enc(ctx.spec)}/data?${new URLSearchParams({ ref: ctx.ref, part, ...params })}`;
const rawUrl = (ctx, path) => `/api/env/${enc(ctx.spec)}/raw?${new URLSearchParams({ ref: ctx.ref, f: path })}`;
const previewSkeleton = (kind) => kind === "spreadsheet"
  ? `<div style="display:flex;gap:6px;margin-bottom:12px">${sk.box(28, "width:90px;border-radius:7px")}${sk.box(28, "width:70px;border-radius:7px")}</div>${sk.box(360, "border-radius:10px")}`
  : `<div style="max-width:860px;margin:0 auto;display:grid;gap:10px">${sk.line(40, 18)}${sk.lines(100, 96, 90, 98, 70)}<div style="height:8px"></div>${sk.lines(94, 100, 88, 60)}</div>`;

async function previewFile(ctx, path, kind) {
  const raw = rawUrl(ctx, path);
  const body = openModal(path, previewSkeleton(kind), { raw, kind });
  if (kind === "image") { body.innerHTML = `<div class="pv-img"><img src="${esc(raw)}" alt="${esc(path)}"></div>`; return; }
  if (kind === "web") { body.innerHTML = `<iframe class="pv-frame" sandbox src="${esc(raw)}" title="${esc(path)}"></iframe>`; return; }
  let p;
  try { p = await progress.wrap(api(dataUrl(ctx, "preview", { path }))); }
  catch (e) { body.innerHTML = emptyState("alert", "Couldn't open this file", esc(e.message)); return; }
  if (!body.isConnected || $("#modal").hidden) return;
  if (p.type === "sheets") {
    body.innerHTML = `<div class="pv-tabs" role="tablist">${p.sheets.map((s, i) => `<button role="tab" data-i="${i}" aria-selected="${i === 0}">${icon("sheet", 13)}${esc(s.name)}${s.dims ? `<em>${esc(s.dims)}</em>` : ""}</button>`).join("")}</div><div id="pv-sheet"></div>`;
    const show = (i) => {
      $("#pv-sheet", body).innerHTML = sheetGrid(p.sheets[i].rows) + (p.sheets[i].rows.length >= 60 ? `<p class="pv-foot">Showing the first 60 rows and 24 columns. Open the original for the whole workbook.</p>` : "");
      $$(".pv-tabs button", body).forEach((b) => b.setAttribute("aria-selected", String(b.dataset.i == i)));
    };
    body.querySelector(".pv-tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
    show(0);
  } else if (p.type === "document") {
    body.innerHTML = `<div class="pv-doc prose">${p.blocks.map((b) => b.table ? table(b.table, { header: true }) : b.h ? `<h${Math.min(b.h + 2, 5)}>${esc(b.text)}</h${Math.min(b.h + 2, 5)}>` : `<p>${esc(b.text)}</p>`).join("")}</div>`;
  } else if (p.type === "slides") {
    body.innerHTML = `${p.repeated?.length ? `<div class="note-box" style="margin-bottom:14px">${icon("info")}<span>On every slide, not repeated below: ${p.repeated.map((t) => `<b>${esc(t)}</b>`).join(" · ")}</span></div>` : ""}
      <div class="pv-slides">${p.slides.map((s) => `<article class="slide"><span class="n">${s.n}</span><div>
        <b>${esc(s.title || "Untitled slide")}</b>${s.text.map((t) => `<p>${esc(t).replace(/\n/g, "<br>")}</p>`).join("")}
        ${(s.tables || []).map((t) => table(t)).join("")}</div></article>`).join("")}</div>`;
  } else if (p.type === "pdf") {
    body.innerHTML = `<div class="pv-tabs"><button data-m="view" aria-selected="true">${icon("doc", 13)}Document</button><button data-m="text" aria-selected="false">${icon("list", 13)}Text <em>${p.page_count} pages</em></button></div>
      <div id="pv-pdf"><iframe class="pv-frame" src="${esc(raw)}" title="${esc(path)}"></iframe></div>`;
    body.querySelector(".pv-tabs").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$(".pv-tabs button", body).forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      $("#pv-pdf", body).innerHTML = b.dataset.m === "view" ? `<iframe class="pv-frame" src="${esc(raw)}" title="${esc(path)}"></iframe>`
        : `<div class="pv-doc">${p.pages.map((pg) => `<div class="sec-label">Page ${pg.n}</div><pre class="code-block wrap">${esc(pg.text)}</pre>`).join("")}</div>`;
    });
  } else if (p.type === "text") body.innerHTML = `<pre class="code-block wrap" style="max-height:none">${esc(p.text)}</pre>`;
  else if (p.type === "archive") body.innerHTML = table([["Entry", "Size"], ...p.entries.map((x) => [x.name, bytes(x.size)])]);
  else if (p.type === "error") body.innerHTML = emptyState("alert", "Couldn't preview this file", esc(p.error), `<a class="btn" href="${esc(raw)}&download=1">Download</a>`);
  else body.innerHTML = emptyState("file", "No preview for this file type", "", `<a class="btn" href="${esc(raw)}&download=1">Download</a>`);
}

async function previewTable(ctx, sys, tbl, label) {
  const body = openModal(`${label || sys} · ${tbl}`, sk.box(420, "border-radius:10px"), { kind: "table" });
  try {
    const d = await progress.wrap(api(dataUrl(ctx, "table", { system: sys, table: tbl, limit: 100 })));
    if ($("#modal").hidden) return;
    body.innerHTML = table([d.columns, ...d.rows.map((r) => r.map((c) => (c == null ? "" : String(c))))]) + `<p class="pv-foot">${d.rows.length >= 100 ? "First 100 rows" : `${d.rows.length} row${d.rows.length === 1 ? "" : "s"}`} · ${d.columns.length} columns · the state of this database when the agent starts</p>`;
  } catch (e) { body.innerHTML = emptyState("alert", "Couldn't load this table", esc(e.message)); }
}
