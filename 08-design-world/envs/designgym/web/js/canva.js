// Canva's own side panels and popovers, as its editor draws them: colour and font open in the
// side panel (not a popover), the colour panel has a search ("#00c4cc" or "blue"), a rainbow "+"
// for a custom colour, the design's colours and Canva's default palette as round swatches; the
// Spacing popover reads letter spacing in thousandths of an em, as Canva does.

import { clamp, fmt, h } from "./util.js";
import { icon } from "./icons.js";
import { ensureFonts, fontCatalog } from "./fonts.js";
import { hexToRgb, hsvToRgb, numberField, popover, rgbToHex, rgbToHsv } from "./widgets.js";

export const CANVA_PALETTE = [
  "#000000", "#545454", "#737373", "#a6a6a6", "#bfbfbf", "#d9d9d9", "#ffffff",
  "#ff3131", "#ff5757", "#ff66c4", "#e2a9f1", "#cb6ce6", "#8c52ff", "#5e17eb",
  "#0097b2", "#0cc0df", "#5ce1e6", "#38b6ff", "#5170ff", "#004aad", "#1800ad",
  "#00bf63", "#7ed957", "#c1ff72", "#ffde59", "#ffbd59", "#ff914d", "#ff5c00",
];

/** "#00c4cc", "00c4cc", "#0cc" or a colour name ("blue") -> "#rrggbb", or null. */
export function parseColor(text) {
  const t = (text || "").trim();
  if (!t) return null;
  const hex = t.replace(/^#?/, "#");
  if (/^#[0-9a-f]{6}$/i.test(hex)) return hex.toLowerCase();
  if (/^#[0-9a-f]{3}$/i.test(hex)) return `#${[...hex.slice(1)].map((c) => c + c).join("")}`.toLowerCase();
  if (!/^[a-z ]+$/i.test(t)) return null;
  const probe = h("span", { style: { color: "rgb(1, 2, 3)" } });
  probe.style.color = t.replace(/\s+/g, "");
  if (probe.style.color === "rgb(1, 2, 3)") return null;  // not a CSS colour name
  document.body.append(probe);
  const m = /rgba?\((\d+), (\d+), (\d+)/.exec(getComputedStyle(probe).color);
  probe.remove();
  return m ? rgbToHex(+m[1], +m[2], +m[3]) : null;
}

const swatch = (c, onPick, { on = false, cls = "" } = {}) => h("button", {
  class: `cv-swatch ${on ? "on" : ""} ${cls}`, type: "button", title: c.toUpperCase(), "aria-label": `Colour ${c}`,
  style: { background: c }, dataset: { color: c }, onclick: () => onPick(c) });

/** The mini "Solid color" picker the rainbow "+" opens: square, hue bar, hex field. */
function solidPicker(anchor, initial, onPick) {
  let [hue, sat, val] = rgbToHsv(...hexToRgb(initial || "#000000"));
  const current = () => rgbToHex(...hsvToRgb(hue, sat, val));
  const sv = h("div", { class: "cp-sv cvp-sv" }, h("div", { class: "cp-sv-knob" }));
  const hueBar = h("div", { class: "cp-hue cvp-hue" }, h("div", { class: "cp-hue-knob" }));
  const dot = h("span", { class: "cvp-dot" });
  const hex = h("input", { class: "cp-hex num-input cvp-hex", spellcheck: "false", "aria-label": "Hex colour" });
  const paint = () => {
    sv.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, hsl(${hue}, 100%, 50%))`;
    sv.firstChild.style.left = `${sat * 100}%`;
    sv.firstChild.style.top = `${(1 - val) * 100}%`;
    hueBar.firstChild.style.left = `${(hue / 360) * 100}%`;
    dot.style.background = current();
    if (document.activeElement !== hex) hex.value = current();
  };
  const drag = (el, fn) => el.addEventListener("pointerdown", (e) => {
    el.setPointerCapture(e.pointerId);
    const at = (ev) => {
      const r = el.getBoundingClientRect();
      fn(clamp((ev.clientX - r.left) / r.width, 0, 1), clamp((ev.clientY - r.top) / r.height, 0, 1));
      paint();
    };
    at(e);
    const move = (ev) => at(ev);
    const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); onPick(current()); };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
  drag(sv, (x, y) => { sat = x; val = 1 - y; });
  drag(hueBar, (x) => { hue = x * 359.9; });
  hex.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") e.stopPropagation();
    if (e.key === "Enter") {
      const c = parseColor(hex.value);
      if (c) { [hue, sat, val] = rgbToHsv(...hexToRgb(c)); paint(); onPick(c); }
      hex.select();
    }
  });
  paint();
  // Canva's popover has Solid color / Gradient tabs; solid colours are what this editor has.
  popover(anchor, h("div", { class: "cvp" }, h("div", { class: "cvp-tabs" }, h("span", { class: "cvp-tab on" }, "Solid color")),
    sv, hueBar, h("div", { class: "cvp-row" }, h("div", { class: "cvp-hexbox" }, dot, hex))), { cls: "cv-solid-pop" });
  hex.focus();
  hex.select();
}

/** The side panel colour picker. `kind` names the section ("Text colors", "Colors"). */
export function colorPanel(app, { get, set, kind = "Text colors", docColors = () => [] }) {
  const search = h("input", { class: "cv-colsearch", placeholder: "Try “blue” or “#00c4cc”", spellcheck: "false",
    "aria-label": "Search colours" });
  const body = h("div", { class: "cv-colors" });
  const pick = async (c) => { await set(c); paint(); };
  const paint = () => {
    const cur = (get() || "").toLowerCase();
    body.textContent = "";
    const add = h("button", { class: "cv-swatch cv-addcolor", type: "button", title: "Add a new color", "aria-label": "Add a new color",
      html: icon("plus", 14) });
    add.addEventListener("click", (e) => { e.stopPropagation(); solidPicker(add, cur || "#000000", pick); });
    const docs = [...new Set(docColors().map((c) => c.toLowerCase()))].filter((c) => c !== cur);
    body.append(
      h("div", { class: "cv-colhead" }, h("span", { html: icon("palette", 16) }), typeof kind === "function" ? kind() : kind),
      h("div", { class: "cv-swatches" }, add, ...(cur ? [swatch(cur, pick, { on: true })] : [])),
      ...(docs.length ? [h("div", { class: "cv-colsub" }, "Colors in this design"), h("div", { class: "cv-swatches" }, ...docs.map((c) => swatch(c, pick)))] : []),
      h("div", { class: "cv-colhead" }, h("span", { html: icon("palette", 16) }), "Default solid colors"),
      h("div", { class: "cv-swatches cv-palette" }, ...CANVA_PALETTE.map((c) => swatch(c, pick, { on: c === cur }))));
  };
  search.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") e.stopPropagation();
    if (e.key === "Enter") {
      const c = parseColor(search.value);
      search.classList.toggle("bad", !c);
      if (c) pick(c);
    }
  });
  search.addEventListener("focus", () => search.select());
  const el = h("div", { class: "cv-colorpanel" }, h("div", { class: "cv-searchbox" }, h("span", { html: icon("search", 16) }), search), body);
  return { el, refresh: paint, focus: () => {} };
}

const recent = [];

/** The side panel font list: search, the design's fonts, recently used, then every font in its own face. */
export function fontPanel(app, { get, set, docFonts = () => [] }) {
  const search = h("input", { class: "font-search cv-fontsearch", placeholder: "Try “Calligraphy” or “Open Sans”", spellcheck: "false",
    "aria-label": "Search fonts" });
  const list = h("div", { class: "font-list cv-fontlist", role: "listbox" });
  let families = [];
  const seen = new IntersectionObserver((entries) => {
    const fams = entries.filter((e) => e.isIntersecting).map((e) => e.target.dataset.family);
    if (fams.length) ensureFonts(fams);
  }, { root: list, rootMargin: "160px" });
  const row = (fam, cur) => {
    const r = h("button", { class: `font-row ${fam === cur ? "on" : ""}`, type: "button", role: "option", dataset: { family: fam },
      onclick: async () => {
        await set(fam);
        ensureFonts([fam]);
        const i = recent.indexOf(fam);
        if (i >= 0) recent.splice(i, 1);
        recent.unshift(fam);
        recent.length = Math.min(recent.length, 5);
        paint();
      } },
    h("span", { class: "cv-fontname", style: { fontFamily: `'${fam}'` } }, fam), h("span", { class: "cv-fontcheck", html: fam === cur ? icon("check", 16) : "" }));
    seen.observe(r);
    return r;
  };
  const group = (title, ic) => h("div", { class: "font-group cv-fontgroup" }, h("span", { html: icon(ic, 16) }), title);
  const paint = () => {
    const cur = get() || "";
    const q = search.value.trim().toLowerCase();
    seen.disconnect();
    list.textContent = "";
    if (q) {
      const hits = families.filter((f) => f.toLowerCase().includes(q));
      hits.forEach((f) => list.append(row(f, cur)));
      if (!hits.length) list.append(h("div", { class: "cv-fontempty" }, `No results for “${search.value.trim()}”`));
      return;
    }
    const docs = [...new Set(docFonts().filter(Boolean))];
    if (docs.length) { list.append(group("Document fonts", "file-text")); docs.forEach((f) => list.append(row(f, cur))); }
    const rec = recent.filter((f) => !docs.includes(f));
    if (rec.length) { list.append(group("Recently used", "clock")); rec.forEach((f) => list.append(row(f, cur))); }
    list.append(group("All fonts", "type"));
    families.forEach((f) => list.append(row(f, cur)));
  };
  search.addEventListener("input", paint);
  search.addEventListener("focus", () => search.select());
  search.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") ev.stopPropagation();
    if (ev.key === "Enter") list.querySelector(".font-row")?.click();
  });
  const el = h("div", { class: "font-pop cv-fontpanel" }, h("div", { class: "cv-searchbox" }, h("span", { html: icon("search", 16) }), search), list);
  return {
    el,
    refresh: async () => {
      const first = !families.length;
      if (first) families = (await fontCatalog()).map((f) => f.family);
      // Repaint for new state, but not under someone typing a search.
      if (first || !list.children.length || document.activeElement !== search || !search.value) {
        paint();
        if (first) list.querySelector(".font-row.on")?.scrollIntoView({ block: "nearest" });
      }
    },
    // Opening the panel puts the cursor in the search with any old query selected, so typing replaces it.
    focus: () => { search.focus(); search.select(); list.querySelector(".font-row.on")?.scrollIntoView({ block: "nearest" }); },
  };
}

/** Canva's Spacing popover: letter spacing (thousandths of an em) and line spacing, slider + number each. */
export function spacingPopover(anchor, m, tget) {
  const size = () => m.common("font_size", ["text"]) || 16;
  const pair = (label, name, { min, max, step, get, set, digits }) => {
    const range = h("input", { type: "range", class: "range cv-range", min, max, step, "aria-label": label });
    const field = numberField({ label: "", title: label, name, digits, min, max, step, get, set: async (v) => { await set(v); sync(); } });
    const sync = () => { const v = get(); range.value = v ?? min; field.set(); };
    range.addEventListener("input", () => { field.input.value = fmt(+range.value, digits); });
    range.addEventListener("change", async () => { await set(+range.value); sync(); });
    sync();
    return h("div", { class: "cv-spacing-item" }, h("div", { class: "cv-spacing-label" }, label), h("div", { class: "cv-spacing-row" }, range, field));
  };
  const ls = pair("Letter spacing", "letter_spacing", { min: -200, max: 800, step: 1, digits: 0,
    get: () => { const v = tget("letter_spacing")(); return v === undefined || v === null ? v : (v / size()) * 1000; },
    set: (v) => m.set((e) => ({ letter_spacing: (v / 1000) * e.font_size }), ["text"]) });
  const lh = pair("Line spacing", "line_height", { min: 0.5, max: 2.5, step: 0.01, digits: 2,
    get: tget("line_height"), set: (v) => m.set({ line_height: v }, ["text"]) });
  popover(anchor, h("div", { class: "cv-pop-body cv-spacing" }, ls, lh), { cls: "cv-spacing-pop" });
}
