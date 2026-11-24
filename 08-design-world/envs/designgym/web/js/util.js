// Small DOM and maths helpers shared by every module.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "html") el.innerHTML = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const round2 = (v) => Math.round(v * 100) / 100;

export function fmt(v, digits = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  const r = Number(v.toFixed(digits));
  return String(Object.is(r, -0) ? 0 : r);
}

export function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export class Emitter {
  constructor() { this._h = new Map(); }
  on(evt, fn) {
    if (!this._h.has(evt)) this._h.set(evt, new Set());
    this._h.get(evt).add(fn);
    return () => this._h.get(evt).delete(fn);
  }
  emit(evt, ...args) {
    for (const fn of this._h.get(evt) || []) {
      try { fn(...args); } catch (e) { console.error(e); }
    }
  }
}

export const rad = (deg) => (deg * Math.PI) / 180;
export const deg = (r) => (r * 180) / Math.PI;

export function rotate(px, py, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [px * c - py * s, px * s + py * c];
}

// Axis-aligned bounds of an element's rotated box, in canvas pixels.
export function bounds(el) {
  if (!el.angle) return { x0: el.x, y0: el.y, x1: el.x + el.width, y1: el.y + el.height };
  const cx = el.x + el.width / 2, cy = el.y + el.height / 2;
  const a = rad(el.angle);
  const hw = (Math.abs(el.width * Math.cos(a)) + Math.abs(el.height * Math.sin(a))) / 2;
  const hh = (Math.abs(el.width * Math.sin(a)) + Math.abs(el.height * Math.cos(a))) / 2;
  return { x0: cx - hw, y0: cy - hh, x1: cx + hw, y1: cy + hh };
}

export function unionBounds(els) {
  const bs = els.map(bounds);
  return {
    x0: Math.min(...bs.map((b) => b.x0)), y0: Math.min(...bs.map((b) => b.y0)),
    x1: Math.max(...bs.map((b) => b.x1)), y1: Math.max(...bs.map((b) => b.y1)),
  };
}

export function cssColor(hex) {
  if (!hex) return "transparent";
  if (hex.length === 9) {
    const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16), a = parseInt(hex.slice(7, 9), 16) / 255;
    return `rgba(${r}, ${g}, ${b}, ${a.toFixed(3)})`;
  }
  return hex;
}

// An image's file name, by asset id, on a lived-in computer (core.workspace): layers and uploads are named after
// the file they came from, as in real editors (store.js fills it).
export const assetNames = new Map();

export function elementLabel(el) {
  if (el.name) return el.name;
  if (el.kind === "text") {
    const t = (el.text || "").replace(/\s+/g, " ").trim();
    return t.length > 28 ? t.slice(0, 26) + "…" : t || "Text";
  }
  if (el.kind === "shape") return el.shape === "ellipse" ? "Ellipse" : "Rectangle";
  const file = assetNames.get(el.asset_id);
  return file ? file.replace(/\.[^.]+$/, "") : `Image ${el.asset_id}`;
}

export function isMac() {
  return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
}

export function shortcut(s) {
  if (!s) return "";
  const mac = isMac();
  return s.replace(/Mod/g, mac ? "⌘" : "Ctrl").replace(/Shift/g, mac ? "⇧" : "Shift")
    .replace(/Alt/g, mac ? "⌥" : "Alt").replace(/\+/g, mac ? "" : "+");
}
