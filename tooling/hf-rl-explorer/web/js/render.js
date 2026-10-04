// Show any JSON value well: images and audio carried as base64 or URLs, chat transcripts, tables for lists of
// records, long text as text, nested objects as key/value rows that fold. Used for environment observations, Task
// API tasks and the rows of custom environments. Everything is escaped; media only from data: or https: URLs.
import { esc, fmt, md } from "./util.js";

const IMG_KEY = /(image|img|png|jpe?g|frame|screenshot|pixels|photo|thumbnail|render)/i;
const AUDIO_KEY = /(audio|wav|speech|sound|waveform)/i;
const B64 = /^[A-Za-z0-9+/=\s]+$/;
const IMG_URL = /^https:\/\/[^\s"'<>]+\.(png|jpe?g|gif|webp|svg|avif)(\?[^\s"'<>]*)?$/i;
const AUDIO_URL = /^https:\/\/[^\s"'<>]+\.(wav|mp3|ogg|flac|m4a)(\?[^\s"'<>]*)?$/i;

function mediaOf(key, v, ctx) {
  if (typeof v !== "string" || v.length < 16) return null;
  if (key === "media_url" && /^https:\/\/[^\s"'<>]+$/.test(v)) {
    if (/^audio\/[\w.+-]+$/.test(ctx.mime || "")) return `<audio class="rv-audio" aria-label="Task audio" controls preload="none" src="${esc(v)}"></audio>`;
    if (/^image\/(png|jpeg|gif|webp|avif)$/.test(ctx.mime || "")) return `<img class="rv-img" src="${esc(v)}" alt="Task image" loading="lazy">`;
    if (/^video\/(mp4|webm|ogg)$/.test(ctx.mime || "")) return `<video class="rv-video" aria-label="Task video" controls preload="none" src="${esc(v)}"></video>`;
  }
  if (v.startsWith("data:image/") && /^data:image\/(png|jpe?g|gif|webp);base64,/.test(v)) return `<img class="rv-img" src="${esc(v)}" alt="${esc(key)}">`;
  if (v.startsWith("data:audio/") && /^data:audio\/[\w.+-]+;base64,/.test(v)) return `<audio class="rv-audio" controls src="${esc(v)}"></audio>`;
  if (IMG_URL.test(v)) return `<img class="rv-img" src="${esc(v)}" alt="${esc(key)}" loading="lazy">`;
  if (AUDIO_URL.test(v)) return `<audio class="rv-audio" controls preload="none" src="${esc(v)}"></audio>`;
  if (v.length > 200 && B64.test(v.slice(0, 400))) {
    if (IMG_KEY.test(key)) {
      const fmtName = String(ctx.image_format || ctx.format || "png").toLowerCase().replace("jpg", "jpeg");
      return `<img class="rv-img" src="data:image/${/^(png|jpeg|gif|webp)$/.test(fmtName) ? fmtName : "png"};base64,${esc(v.replace(/\s/g, ""))}" alt="${esc(key)}">`;
    }
    if (AUDIO_KEY.test(key)) return `<audio class="rv-audio" controls src="data:audio/${esc(ctx.mime?.split("/")[1] || "wav")};base64,${esc(v.replace(/\s/g, ""))}"></audio>`;
    return `<span class="rv-faint">base64, ${fmt.format(Math.round(v.length * 0.75 / 1024))} KB</span>`;
  }
  return null;
}

const isMessage = (x) => x && typeof x === "object" && !Array.isArray(x) && typeof x.role === "string" && ("content" in x || "tool_calls" in x);
const scalar = (x) => x == null || ["string", "number", "boolean"].includes(typeof x);

function text(s, key) {
  if (s === "") return `<span class="rv-faint">empty</span>`;
  if (s.length > 160 || s.includes("\n")) {
    const looksMd = /(^|\n)(#{1,4} |[-*] |\d+\. |```)/.test(s);
    return looksMd && s.length < 40_000 ? `<div class="rv-text prose">${md(s)}</div>` : `<pre class="rv-pre">${esc(s.length > 60_000 ? s.slice(0, 60_000) + "\n… (cut)" : s)}</pre>`;
  }
  return `<span class="rv-str">${esc(s)}</span>`;
}

// a long message shows its first lines and a "Show all" toggle (a checkbox: no script needed in a rendered string)
let clipN = 0;
const clip = (html, len) => (len < 1500 ? html : `<div class="rv-clip"><input type="checkbox" class="rv-tog" id="rv-c${++clipN}" aria-label="Show all"><div class="rv-clip-b">${html}</div>
  <label for="rv-c${clipN}" class="rv-more"><span class="o">Show all</span><span class="c">Show less</span></label></div>`);
const lenOf = (c) => (typeof c === "string" ? c.length : JSON.stringify(c ?? "").length);

function chat(msgs, depth) {
  return `<div class="rv-chat">${msgs.slice(0, 200).map((m) => {
    const body = typeof m.content === "string" ? text(m.content, "content")
      : Array.isArray(m.content) ? m.content.map((p) => (p?.type === "text" ? text(p.text || "", "text") : value(p, "part", depth + 1))).join("")
      : m.content == null ? "" : value(m.content, "content", depth + 1);
    const calls = Array.isArray(m.tool_calls) ? m.tool_calls.map((c) => `<div class="rv-call"><b>${esc(c.function?.name || c.name || "tool")}</b> <code>${esc(typeof c.function?.arguments === "string" ? c.function.arguments.slice(0, 2000) : JSON.stringify(c.arguments ?? c.function?.arguments ?? {}).slice(0, 2000))}</code></div>`).join("") : "";
    return `<div class="rv-msg ${esc(m.role)}"><span class="rv-role">${esc(m.role)}${m.name ? ` · ${esc(m.name)}` : ""}</span>${clip(body, lenOf(m.content))}${calls}</div>`;
  }).join("")}${msgs.length > 200 ? `<p class="rv-faint">and ${fmt.format(msgs.length - 200)} more messages</p>` : ""}</div>`;
}

function table(rows, depth) {
  const keys = [...new Set(rows.slice(0, 50).flatMap((r) => Object.keys(r)))].slice(0, 12);
  return `<div class="rv-table"><table><thead><tr>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${rows.slice(0, 100).map((r) =>
    `<tr>${keys.map((k) => `<td>${scalar(r[k]) ? cell(r[k]) : value(r[k], k, depth + 1)}</td>`).join("")}</tr>`).join("")}</tbody></table>
    ${rows.length > 100 ? `<p class="rv-faint">${fmt.format(rows.length - 100)} more rows</p>` : ""}</div>`;
}
const cell = (x) => (x == null ? `<span class="rv-faint">–</span>` : typeof x === "string" ? `<span title="${esc(x)}">${esc(x.length > 140 ? x.slice(0, 140) + "…" : x)}</span>` : `<code>${esc(String(x))}</code>`);

export function value(v, key = "", depth = 0, ctx = {}) {
  if (depth > 7) return `<code class="rv-faint">…</code>`;
  if (v == null) return `<span class="rv-null">null</span>`;
  if (typeof v === "boolean") return `<span class="rv-bool ${v}">${v}</span>`;
  if (typeof v === "number") return `<span class="rv-num">${esc(Number.isInteger(v) ? fmt.format(v) : String(+v.toFixed(6)))}</span>`;
  if (typeof v === "string") return mediaOf(key, v, ctx) || text(v, key);
  if (Array.isArray(v)) {
    if (!v.length) return `<span class="rv-faint">empty list</span>`;
    if (v.every(isMessage)) return chat(v, depth);
    if (v.every(scalar) && v.length <= 60 && v.every((x) => String(x).length < 60)) return `<span class="rv-list">${v.map((x) => `<span>${esc(String(x))}</span>`).join("")}</span>`;
    if (v.length > 1 && v.every((x) => x && typeof x === "object" && !Array.isArray(x)) && v.slice(0, 20).every((x) => Object.values(x).filter((y) => !scalar(y)).length <= 1)) return table(v, depth);
    return `<ol class="rv-ol">${v.slice(0, 100).map((x, i) => `<li>${value(x, key, depth + 1, ctx)}</li>`).join("")}${v.length > 100 ? `<li class="rv-faint">${fmt.format(v.length - 100)} more</li>` : ""}</ol>`;
  }
  if (typeof v === "object") return object(v, depth);
  return esc(String(v));
}

// key/value rows; a nested object folds when it's big
export function object(o, depth = 0, { order = [], hide = [] } = {}) {
  const keys = Object.keys(o).filter((k) => !hide.includes(k));
  const pri = (k) => (order.includes(k) ? order.indexOf(k) : order.length);
  keys.sort((a, b) => pri(a) - pri(b));   // stable: the rest keep the server's order
  if (!keys.length) return `<span class="rv-faint">empty</span>`;
  return `<dl class="rv-kv">${keys.map((k) => {
    const x = o[k];
    const big = x && typeof x === "object" && (Array.isArray(x) ? x.length > 8 : Object.keys(x).length > 6) && depth > 0;
    const inner = value(x, k, depth + 1, o);
    return `<dt title="${esc(k)}">${esc(k)}</dt><dd>${big ? `<details><summary>${Array.isArray(x) ? `${fmt.format(x.length)} items` : `${fmt.format(Object.keys(x).length)} fields`}</summary>${inner}</details>` : inner}</dd>`;
  }).join("")}</dl>`;
}

// An observation from reset/step: reward and done up front, the rest below
export function observation(data) {
  const obs = data && typeof data === "object" && "observation" in data ? data.observation : data;
  const reward = data?.reward ?? obs?.reward;
  const done = data?.done ?? obs?.done;
  const head = `<div class="rv-obs-h">${reward != null ? `<span class="rv-reward ${Number(reward) > 0 ? "pos" : Number(reward) < 0 ? "neg" : ""}">reward <b>${esc(typeof reward === "number" ? String(+reward.toFixed(4)) : String(reward))}</b></span>` : ""}
    ${done != null ? `<span class="rv-done ${done ? "yes" : ""}">${done ? "done" : "not done"}</span>` : ""}</div>`;
  const rest = obs && typeof obs === "object" && !Array.isArray(obs) ? object(obs, 0, { order: ["prompt", "feedback", "message", "text"], hide: ["reward", "done"] }) : value(obs, "observation");
  return head + rest;
}

// An MCP tool result: its content parts (text, images), or its raw value
export function toolResult(r) {
  if (r && Array.isArray(r.content)) {
    const parts = r.content.map((c) => (c?.type === "text" ? (() => { try { const j = JSON.parse(c.text); return typeof j === "object" && j ? value(j) : text(c.text, "text"); } catch { return text(c.text || "", "text"); } })()
      : c?.type === "image" && /^image\/(png|jpeg|gif|webp)$/.test(c.mimeType || c.mime_type) ? `<img class="rv-img" src="data:${esc(c.mimeType || c.mime_type)};base64,${esc(c.data)}" alt="">`
      : c?.type === "audio" ? `<audio class="rv-audio" controls src="data:${esc(c.mimeType || "audio/wav")};base64,${esc(c.data)}"></audio>`
      : value(c))).join("");
    return `${r.isError || r.is_error ? `<div class="rv-obs-h"><span class="rv-done err">error</span></div>` : ""}${parts || `<span class="rv-faint">no content</span>`}`;
  }
  return value(r);
}
