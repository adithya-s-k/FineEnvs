// Rows of RL training data (NeMo Gym, Verifiers, verl / SkyRL; app/envs/convo.py reads them): the conversation as the
// policy gets it, and the tools it may call. Text is shown whole and escaped; long parts (a system prompt, reasoning, a
// tool's output) fold under a line that says how long they are, so nothing scrolls inside the page.
//   transcript  {turns: [{kind: message|reasoning|call|output|other, role, text, name?, args?, output?, images?}],
//                next?: {label, text}, note?}
//   tools       {tools: [{name, kind, description, built_in, params: [{name, type, required, description, enum?,
//                default?, params?}]}], open?}
//   fields      {rows: [{key, text, code}]}: a row's other fields, label over value; long values fold
import { esc, fmt, md } from "../util.js";
import { icon } from "../icons.js";

export function render(kind, data) {
  return ({ transcript, tools, fields }[kind] || (() => ""))(data || {});
}

const LONG = 1400, TALL = 18;   // past this many characters or lines, a system prompt, reasoning or tool output folds
const long = (s) => s.length > LONG || (s.match(/\n/g) || []).length > TALL;
const chars = (s) => `${fmt.format(s.length)} characters`;
const ROLE = { system: "System", developer: "Developer", user: "User", assistant: "Assistant", tool: "Tool", other: "Other" };
const text = (s, cls = "") => (s ? `<div class="rl-text ${cls}">${esc(s)}</div>` : `<p class="rl-empty">empty</p>`);
const images = (list) => (list || []).map((u) => `<img class="rl-img" src="${esc(u)}" alt="an image in the conversation" loading="lazy">`).join("");
const peek = (s) => esc(s.replace(/\s+/g, " ").trim().slice(0, 140));
function fold(label, meta, inner, open = false) {
  return `<details class="rl-fold" ${open ? "open" : ""}><summary>${icon("chevronRight", 13, "chev")}<b>${label}</b><span class="rl-meta">${meta}</span></summary>
    <div class="rl-fold-b">${inner}</div></details>`;
}

// ── the conversation ─────────────────────────────────────────────────────────
function part(t) {
  if (t.kind === "reasoning") return fold("Reasoning", t.text ? `${chars(t.text)} · ${peek(t.text)}` : "empty", text(t.text));
  if (t.kind === "call") {
    const args = t.args && t.args !== "{}" ? `<pre class="rl-code">${esc(t.args)}</pre>` : `<p class="rl-empty">no arguments</p>`;
    const out = "output" in t ? (long(t.output || "")
      ? fold("Output", `${chars(t.output)} · ${peek(t.output)}`, `<pre class="rl-code">${esc(t.output)}</pre>${images(t.images)}`)
      : `<div class="rl-lab">Output</div>${t.output ? `<pre class="rl-code">${esc(t.output)}</pre>` : `<p class="rl-empty">empty</p>`}${images(t.images)}`) : "";
    return `<div class="rl-call"><div class="rl-lab">${icon("wrench", 12)}Calls <code>${esc(t.name)}</code></div>${args}${out}</div>`;
  }
  if (t.kind === "output") {
    const body = `<pre class="rl-code">${esc(t.text)}</pre>${images(t.images)}`;
    return long(t.text || "") ? fold("Output", `${chars(t.text)} · ${peek(t.text)}`, body) : body;
  }
  if (t.kind === "other") return `<pre class="rl-code">${esc(t.text)}</pre>`;
  if (t.role === "system" || t.role === "developer") {
    return t.text && long(t.text) ? fold(`${ROLE[t.role]} prompt`, chars(t.text), text(t.text)) + images(t.images) : text(t.text) + images(t.images);
  }
  // the assistant writes Markdown; what it's given is shown as it is
  if (t.role === "assistant" && t.text) return `<div class="rl-md prose">${md(t.text)}</div>${images(t.images)}`;
  return text(t.text) + images(t.images);
}

const KEEP = 4;   // a long conversation shows its opening and its last turns; the middle folds
function turnHtml(g) {
  return `<li class="rl-turn r-${esc(g.side)}"><div class="rl-who">${esc(ROLE[g.side] || g.side)}${g.name ? ` <span>${esc(g.name)}</span>` : ""}</div>
    <div class="rl-body">${g.items.map(part).join("")}</div></li>`;
}
function tally(groups) {
  const items = groups.flatMap((g) => g.items);
  const n = (f) => items.filter(f).length;
  return [[n((t) => t.kind === "message" && t.role === "user"), "user message"], [n((t) => t.kind === "message" && t.role === "assistant"), "assistant message"],
    [n((t) => t.kind === "call"), "tool call"]].filter(([k]) => k).map(([k, w]) => `${fmt.format(k)} ${w}${k === 1 ? "" : "s"}`).join(", ");
}

function transcript(d) {
  const ts = d.turns || [];
  // consecutive turns of one side (the assistant's reasoning, words and tool calls) are one block
  const groups = [];
  ts.forEach((t) => {
    const side = t.kind === "output" ? "tool" : t.kind === "other" ? "other" : t.role;
    const g = groups[groups.length - 1];
    if (g && g.side === side && side === "assistant") g.items.push(t);
    else groups.push({ side, name: t.name && t.kind === "message" ? t.name : "", items: [t] });
  });
  const n = (k, role) => ts.filter((t) => t.kind === k && (!role || t.role === role)).length;
  const counts = [[n("message", "user"), "user message"], [n("message", "assistant"), "assistant message"], [n("call"), "tool call"], [n("reasoning"), "reasoning step"]]
    .filter(([c]) => c).map(([c, w]) => `${fmt.format(c)} ${w}${c === 1 ? "" : "s"}`).join(" · ");
  // the opening (system prompt and the first request) and the last few turns stay open; a long middle folds
  const firstUser = groups.findIndex((g) => g.side === "user");
  const head = firstUser < 0 ? 1 : firstUser + 1;
  const tailFrom = Math.max(head, groups.length - KEEP);
  const middle = groups.slice(head, tailFrom);
  const body = middle.length >= 3
    ? [...groups.slice(0, head).map(turnHtml), `<li class="rl-gap"><details class="rl-fold rl-earlier"><summary>${icon("chevronRight", 13, "chev")}<b>${fmt.format(middle.length)} earlier turns</b><span class="rl-meta">${tally(middle)}</span></summary>
        <ol class="rl-tr rl-inner">${middle.map(turnHtml).join("")}</ol></details></li>`, ...groups.slice(tailFrom).map(turnHtml)]
    : groups.map(turnHtml);
  return `${d.note ? `<p class="rl-note">${esc(d.note)}</p>` : ""}
    ${counts && ts.length > 2 ? `<p class="rl-count">${counts}</p>` : ""}
    <ol class="rl-tr">${body.join("")}
      ${d.next ? `<li class="rl-turn rl-next"><div class="rl-who">${esc(d.next.label || "Next")}</div><div class="rl-body"><p>${esc(d.next.text || "")}</p></div></li>` : ""}</ol>`;
}

// ── the tools ────────────────────────────────────────────────────────────────
const sentence = (s) => { const m = String(s || "").replace(/\s+/g, " ").match(/^.{0,160}?[.!?](\s|$)/); return m ? m[0].trim() : String(s || "").replace(/\s+/g, " ").slice(0, 160); };
function params(list) {
  return `<dl class="rl-params">${list.map((p) => `<dt><code>${esc(p.name)}</code>${p.required ? `<span class="rl-req">required</span>` : ""}</dt>
    <dd>${p.type ? `<span class="rl-type">${esc(p.type)}</span>` : ""}${p.description ? ` ${esc(p.description)}` : ""}
      ${p.enum ? `<div class="rl-enum">One of ${p.enum.map((x) => `<code>${esc(x)}</code>`).join(" ")}</div>` : ""}
      ${p.default != null ? `<div class="rl-enum">Default <code>${esc(p.default)}</code></div>` : ""}
      ${p.params?.length ? params(p.params) : ""}</dd>`).join("")}</dl>`;
}
function tools(d) {
  const list = d.tools || [];
  if (!list.length) return `<p class="muted sm">No tools: the policy answers in words only.</p>`;
  const open = d.open ?? list.length <= 3;
  return `<div class="rl-tools">${list.map((t) => {
    const sig = t.built_in ? "" : `(<span>${t.params.map((p) => esc(p.name) + (p.required ? "" : "?")).join(", ")}</span>)`;
    return `<details class="rl-tool" ${open ? "open" : ""}><summary>${icon("chevronRight", 13, "chev")}<code class="rl-sig">${esc(t.name)}${sig}</code>
      <span class="rl-meta">${t.built_in ? `built-in ${esc(t.kind)} tool` : esc(sentence(t.description))}</span></summary>
      <div class="rl-fold-b">${t.description ? `<p class="rl-desc">${esc(t.description)}</p>` : ""}${t.params.length ? params(t.params) : `<p class="rl-empty">no parameters</p>`}</div></details>`;
  }).join("")}</div>`;
}

// ── a row's other fields ─────────────────────────────────────────────────────
function fields(d) {
  const rows = d.rows || [];
  if (!rows.length) return "";
  const val = (r) => {
    const body = r.code ? `<pre class="rl-code">${esc(r.text)}</pre>` : text(r.text);
    if (long(r.text)) return fold(r.code ? "Show" : "Show all", `${chars(r.text)} · ${peek(r.text)}`, body);
    return r.code && !r.text.includes("\n") ? `<code class="rl-val">${esc(r.text)}</code>` : body;
  };
  return `<dl class="rl-fields">${rows.map((r) => `<dt><code>${esc(r.key)}</code></dt><dd>${val(r)}</dd>`).join("")}</dl>`;
}
