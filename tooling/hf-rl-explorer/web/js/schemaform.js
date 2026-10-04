// A form from a JSON schema (an MCP tool's inputSchema, an OpenEnv action schema): one field per property, typed
// inputs for strings, numbers, booleans and enums, a JSON box for anything nested. read() turns it back into an object
// (fields left empty are left out, so the server's defaults apply).
import { esc } from "./util.js";

// strings that hold prose or code get a box; the rest (a word, an id, a path) a line
const LONG = /(prompt|text|code|content|description|instruction|message|answer|transcript|latex|query|body|script|reasoning|patch)/i;

function resolve(p, defs) {
  if (p?.$ref) return defs[p.$ref.split("/").pop()] || {};
  return p || {};
}

// a property's type, looking through anyOf/oneOf with null (Optional[...])
function kind(p, defs) {
  p = resolve(p, defs);
  if (p.enum) return { t: "enum", p };
  if (p.const !== undefined) return { t: "const", p };
  const alts = (p.anyOf || p.oneOf || []).map((x) => resolve(x, defs)).filter((x) => x.type !== "null");
  if (alts.length === 1) return { ...kind({ title: p.title, ...alts[0] }, defs), nullable: true };
  if (alts.length > 1) return { t: "json", p };
  const t = Array.isArray(p.type) ? p.type.find((x) => x !== "null") : p.type;
  if (t === "string") return { t: p.maxLength > 200 || (p.maxLength == null && !p.format && !p.enum && LONG.test(p.title || "")) ? "text" : "string", p };
  if (t === "integer" || t === "number") return { t: "number", p, int: t === "integer" };
  if (t === "boolean") return { t: "bool", p };
  return { t: "json", p };
}

const label = (k, p) => esc(p.title || k);

export function schemaForm(schema, { prefix = "sf", values = {}, skip = ["metadata"] } = {}) {
  const defs = schema?.$defs || schema?.definitions || {};
  const props = Object.entries(schema?.properties || {}).filter(([k]) => !skip.includes(k));
  const req = new Set(schema?.required || []);
  if (!props.length) return `<p class="sf-none">No inputs.</p>`;
  return `<div class="sf">${props.map(([k, raw]) => {
    const { t, p, int, nullable } = kind({ title: k, ...raw }, defs);
    const id = `${prefix}-${k.replace(/[^\w-]/g, "_")}`;
    const v = values[k] ?? p.default;
    const hint = [req.has(k) ? "required" : nullable ? "optional" : "", p.description || resolve(raw, defs).description || ""].filter(Boolean).join(" · ");
    let input;
    if (t === "const") return "";
    if (t === "enum") input = `<select class="input" id="${id}" data-k="${esc(k)}" data-t="enum">${req.has(k) ? "" : `<option value="">–</option>`}${p.enum.map((x) =>
      `<option value="${esc(JSON.stringify(x))}" ${JSON.stringify(x) === JSON.stringify(v) ? "selected" : ""}>${esc(String(x))}</option>`).join("")}</select>`;
    else if (t === "bool") input = `<label class="sf-check"><input type="checkbox" id="${id}" data-k="${esc(k)}" data-t="bool" ${v ? "checked" : ""}><span>${v ? "true" : "false"}</span></label>`;
    else if (t === "number") input = `<input class="input mono" type="number" id="${id}" data-k="${esc(k)}" data-t="${int ? "int" : "num"}" ${p.minimum != null ? `min="${p.minimum}"` : ""} ${p.maximum != null ? `max="${p.maximum}"` : ""} step="${int ? 1 : "any"}" value="${v ?? ""}" placeholder="${v == null ? (nullable ? "none" : "") : ""}">`;
    else if (t === "text") input = `<textarea class="input mono" rows="3" id="${id}" data-k="${esc(k)}" data-t="str" spellcheck="false">${esc(v ?? "")}</textarea>`;
    else if (t === "string") input = `<input class="input mono" id="${id}" data-k="${esc(k)}" data-t="str" value="${esc(v ?? "")}" spellcheck="false" autocomplete="off">`;
    else input = `<textarea class="input mono" rows="3" id="${id}" data-k="${esc(k)}" data-t="json" spellcheck="false" placeholder="JSON">${v != null ? esc(JSON.stringify(v, null, 1)) : ""}</textarea>`;
    return `<label class="field sf-f" for="${id}"><span><code>${label(k, p)}</code>${hint ? `<em>${esc(hint)}</em>` : ""}</span>${input}</label>`;
  }).join("")}</div>`;
}

// the form's values as an object; throws Error(message) on bad JSON or a missing required field
export function readForm(el, schema) {
  const out = {};
  const req = new Set(schema?.required || []);
  for (const f of el.querySelectorAll("[data-k]")) {
    const k = f.dataset.k, t = f.dataset.t;
    if (t === "bool") { out[k] = f.checked; continue; }
    const raw = f.value;
    if (raw === "" || raw == null) { if (req.has(k) && t !== "str") throw new Error(`${k} is required`); if (req.has(k)) out[k] = ""; continue; }
    if (t === "int" || t === "num") { const n = Number(raw); if (!Number.isFinite(n)) throw new Error(`${k} should be a number`); out[k] = t === "int" ? Math.trunc(n) : n; }
    else if (t === "enum") out[k] = JSON.parse(raw);
    else if (t === "json") { try { out[k] = JSON.parse(raw); } catch { throw new Error(`${k} isn't valid JSON`); } }
    else out[k] = raw;
  }
  // a const field (an action's type discriminator) goes in as its value
  for (const [k, p] of Object.entries(schema?.properties || {})) if (p?.const !== undefined && !(k in out)) out[k] = p.const;
  return out;
}

export function wireForm(el) {
  el.addEventListener("change", (e) => {
    const c = e.target.closest('[data-t="bool"]');
    if (c) c.nextElementSibling.textContent = String(c.checked);
  });
}
