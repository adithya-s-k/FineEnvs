// Forms preserve the JSON schema's values and defaults. Complex unions and free-form
// objects use a JSON editor instead of guessing which action variant the user meant.
import { esc } from "./util.js";

const LONG = /(prompt|text|code|content|description|instruction|message|answer|transcript|latex|query|body|script|reasoning|patch)/i;
const own = (o, k) => Object.prototype.hasOwnProperty.call(o || {}, k);
const object = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

export function resolveSchema(schema, root = schema, seen = new Set()) {
  if (!object(schema)) return {};
  if (!schema.$ref) return schema;
  const ref = schema.$ref;
  if (typeof ref !== "string" || !ref.startsWith("#/") || seen.has(ref) || seen.size > 20) return {};
  let target = root;
  for (const part of ref.slice(2).split("/").map((s) => s.replace(/~1/g, "/").replace(/~0/g, "~"))) {
    if (!own(target, part)) return {};
    target = target[part];
  }
  const { $ref, ...rest } = schema;
  return { ...resolveSchema(target, root, new Set([...seen, ref])), ...rest };
}

function kind(raw, root) {
  const p = resolveSchema(raw, root);
  if (own(p, "const")) return { t: "const", p };
  if (Array.isArray(p.enum)) return { t: "enum", p };
  const alternatives = p.anyOf || p.oneOf;
  if (Array.isArray(alternatives)) {
    const choices = alternatives.map((x) => resolveSchema(x, root)).filter((x) => x.type !== "null");
    if (choices.length === 1 && choices.length < alternatives.length) {
      const { anyOf, oneOf, ...rest } = p;
      return { ...kind({ ...choices[0], ...rest }, root), nullable: true };
    }
    return { t: "json", p };
  }
  if (p.allOf) return { t: "json", p };
  const ts = Array.isArray(p.type) ? p.type.filter((x) => x !== "null") : [p.type];
  const t = ts.length === 1 ? ts[0] : undefined;
  return { t: t === "string" ? (LONG.test(p.title || "") ? "text" : "str") :
    t === "integer" ? "int" : t === "number" ? "num" : t === "boolean" ? "bool" : "json", p,
    nullable: Array.isArray(p.type) && p.type.includes("null") };
}

// Deliberately a bounded, conservative subset of JSON Schema: server validation
// remains authoritative for formats, patterns and custom framework constraints.
export function validateValue(value, schema, path = "Input", root = schema, depth = 0) {
  if (depth > 24) return;
  if (schema === false) throw new Error(`${path} is not allowed`);
  const p = resolveSchema(schema, root);
  const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  if (own(p, "const") && !equal(value, p.const)) throw new Error(`${path} must be ${JSON.stringify(p.const)}`);
  if (p.enum && !p.enum.some((x) => equal(x, value))) throw new Error(`${path} must be one of the listed choices`);
  for (const option of p.allOf || []) validateValue(value, option, path, root, depth + 1);
  const alternatives = p.anyOf || p.oneOf;
  if (alternatives) {
    const matches = alternatives.filter((x) => { try { validateValue(value, x, path, root, depth + 1); return true; } catch { return false; } });
    if (!matches.length || (p.oneOf && matches.length !== 1)) throw new Error(`${path} must match ${p.oneOf ? "exactly one" : "one"} of the allowed schemas`);
  }
  const types = Array.isArray(p.type) ? p.type : p.type ? [p.type] : [];
  const is = (t) => t === "null" ? value === null : t === "object" ? object(value) : t === "array" ? Array.isArray(value) :
    t === "integer" ? Number.isSafeInteger(value) : t === "number" ? typeof value === "number" && Number.isFinite(value) : typeof value === t;
  if (types.length && !types.some(is)) throw new Error(`${path} must be ${types.join(" or ")}`);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${path} must be a finite number`);
    for (const [key, bad, word] of [["minimum", (n) => value < n, "at least"], ["maximum", (n) => value > n, "at most"],
      ["exclusiveMinimum", (n) => value <= n, "greater than"], ["exclusiveMaximum", (n) => value >= n, "less than"]]) {
      if (typeof p[key] === "number" && bad(p[key])) throw new Error(`${path} must be ${word} ${p[key]}`);
    }
  }
  if (typeof value === "string") {
    const length = [...value].length;
    if (p.minLength != null && length < p.minLength) throw new Error(`${path} needs at least ${p.minLength} characters`);
    if (p.maxLength != null && length > p.maxLength) throw new Error(`${path} allows at most ${p.maxLength} characters`);
  }
  if (Array.isArray(value)) {
    if (p.minItems != null && value.length < p.minItems) throw new Error(`${path} needs at least ${p.minItems} items`);
    if (p.maxItems != null && value.length > p.maxItems) throw new Error(`${path} allows at most ${p.maxItems} items`);
    if (p.items) value.forEach((v, i) => validateValue(v, p.items, `${path}[${i}]`, root, depth + 1));
  } else if (object(value)) {
    for (const key of p.required || []) if (!own(value, key)) throw new Error(`${path}.${key} is required`);
    for (const [key, v] of Object.entries(value)) {
      if (own(p.properties, key)) validateValue(v, p.properties[key], `${path}.${key}`, root, depth + 1);
      else if (p.additionalProperties === false) throw new Error(`${path}.${key} is not an allowed field`);
      else if (object(p.additionalProperties)) validateValue(v, p.additionalProperties, `${path}.${key}`, root, depth + 1);
    }
  }
}

export function schemaForm(schema, { prefix = "sf", values = {}, skip = ["metadata"] } = {}) {
  const root = schema || {}, s = resolveSchema(root, root);
  const required = new Set(s.required || []);
  const props = Object.entries(s.properties || {}).filter(([k]) => !skip.includes(k) || required.has(k));
  const idBase = esc(prefix);
  if (s.anyOf || s.oneOf || s.allOf || !props.length && s.additionalProperties !== false) {
    const initial = Object.keys(values).length ? values : s.default;
    return `<label class="field sf-f" for="${idBase}-json"><span>Input JSON <em>Use the server's schema; nested fields and action variants are kept as written.</em></span><textarea class="input mono" id="${idBase}-json" data-root-json rows="5" spellcheck="false" placeholder="{}">${initial !== undefined ? esc(JSON.stringify(initial, null, 2)) : ""}</textarea></label>`;
  }
  if (!props.length) return '<p class="sf-none">No inputs.</p>';
  return `<div class="sf">${props.map(([k, raw], index) => {
    const { t, p, nullable } = kind({ title: k, ...raw }, root);
    // Schema property names are untrusted and can collide after sanitization.
    const id = `${idBase}-${index}`, v = own(values, k) ? values[k] : p.default;
    const hint = [required.has(k) ? "required" : "optional; blank uses the server default", p.description || ""].filter(Boolean).join(" · ");
    if (t === "const") return "";
    const attrs = `id="${id}" data-k="${esc(k)}" data-t="${t}" ${required.has(k) ? 'aria-required="true"' : ""} aria-describedby="${id}-hint"`;
    let input;
    if (t === "enum" || t === "bool") {
      const choices = t === "bool" ? [true, false, ...(nullable ? [null] : [])] : p.enum;
      input = `<select class="input" ${attrs}><option value="">${required.has(k) ? "Choose a value" : "Use server default"}</option>${choices.map((x) =>
        `<option value="${esc(JSON.stringify(x))}" ${JSON.stringify(x) === JSON.stringify(v) ? "selected" : ""}>${esc(String(x))}</option>`).join("")}</select>`;
    } else if (t === "int" || t === "num") input = `<input class="input mono" type="number" ${attrs} ${typeof p.minimum === "number" ? `min="${esc(p.minimum)}"` : ""} ${typeof p.maximum === "number" ? `max="${esc(p.maximum)}"` : ""} step="${t === "int" ? 1 : "any"}" value="${esc(v ?? "")}">`;
    else if (t === "str") input = `<input class="input mono" ${attrs} value="${esc(v ?? "")}" spellcheck="false" autocomplete="off">`;
    else input = `<textarea class="input mono" rows="3" ${attrs} spellcheck="false" ${t === "json" ? 'placeholder="JSON"' : ""}>${v !== undefined ? esc(t === "json" ? JSON.stringify(v, null, 2) : v) : ""}</textarea>`;
    return `<div class="field sf-f"><label for="${id}"><code>${esc(p.title || k)}</code></label><em class="sf-hint" id="${id}-hint">${esc(hint)}</em>${input}</div>`;
  }).join("")}</div>`;
}

export function readForm(el, schema) {
  const rootInput = el.querySelector("[data-root-json]");
  if (rootInput) {
    let value;
    try { value = JSON.parse(rootInput.value || "{}"); } catch { throw new Error("Input isn't valid JSON"); }
    validateValue(value, schema);
    return value;
  }
  const out = Object.create(null), s = resolveSchema(schema, schema);
  const required = new Set(s.required || []);
  for (const f of el.querySelectorAll("[data-k]")) {
    const k = f.dataset.k, t = f.dataset.t, raw = f.value;
    if (raw === "" || raw == null) {
      if (required.has(k)) {
        if (t !== "str" && t !== "text") throw new Error(`${k} is required`);
        out[k] = "";
      }
      continue;
    }
    if (t === "int" || t === "num") {
      const n = Number(raw);
      if (!Number.isFinite(n)) throw new Error(`${k} must be a finite number`);
      if (t === "int" && !Number.isSafeInteger(n)) throw new Error(`${k} must be a whole number within the safe integer range`);
      out[k] = n;
    } else if (["enum", "bool", "json"].includes(t)) {
      try { out[k] = JSON.parse(raw); } catch { throw new Error(`${k} isn't valid JSON`); }
    } else out[k] = raw;
  }
  for (const [k, raw] of Object.entries(s.properties || {})) {
    const p = resolveSchema(raw, schema);
    if (own(p, "const") && !own(out, k)) out[k] = p.const;
  }
  validateValue(out, schema);
  return out;
}

export function wireForm() { /* Native labelled controls handle their own state. */ }
