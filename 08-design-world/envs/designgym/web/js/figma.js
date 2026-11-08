// Figma's own pickers and typography controls, as its current (UI3) editor draws them:
// the colour and font pickers are floating panels to the left of the properties panel,
// weight is a dropdown rather than B / I buttons, line height reads "120%" and letter
// spacing "2px", and underline / case sit behind the Type settings button.
// Only what this editor can honour is offered (solid fills, the faces a family has).

import { clamp, fmt, h } from "./util.js";
import { icon } from "./icons.js";
import { ensureFonts, fontCatalog, hasStyle } from "./fonts.js";
import {
  closePopover, floatPanel, hexToRgb, hsvToRgb, iconButton, menu, numberField, popover, rgbToHex, rgbToHsv, segmented,
} from "./widgets.js";

const SIDE = ".fg-right";

// A little list opened inside a floating panel; it flips upward when it would leave the screen.
function dropList(host, list) {
  host.append(list);
  const r = list.getBoundingClientRect();
  if (r.bottom > innerHeight - 8) list.style.top = `${-r.height - 4 + 30}px`;
  if (r.bottom > innerHeight - 8 && list.getBoundingClientRect().top < 8) list.style.top = `${8 - host.getBoundingClientRect().top}px`;
}

// ---------------------------------------------------------------- colour

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b), l = (max + min) / 2, d = max - min;
  const s = d ? d / (1 - Math.abs(2 * l - 1)) : 0;
  const [hue] = rgbToHsv(r * 255, g * 255, b * 255);
  return [hue, s, l];
}
function hslToRgb(hue, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs(((hue / 60) % 2) - 1)), m = l - c / 2;
  const [r, g, b] = hue < 60 ? [c, x, 0] : hue < 120 ? [x, c, 0] : hue < 180 ? [0, c, x]
    : hue < 240 ? [0, x, c] : hue < 300 ? [x, 0, c] : [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

const MODELS = ["Hex", "RGB", "HSL", "HSB"];
let model = "Hex";  // Figma remembers the colour model between pickers

/** Figma's colour picker: Custom panel with the square, hue bar, model + value fields, "On this page". */
export function openColorPanel(anchor, get, set, docColors = []) {
  let [hue, sat, val] = rgbToHsv(...hexToRgb(get() || "#000000"));
  const current = () => rgbToHex(...hsvToRgb(hue, sat, val));
  const sv = h("div", { class: "cp-sv fgc-sv" }, h("div", { class: "cp-sv-knob" }));
  const hueBar = h("div", { class: "cp-hue fgc-hue" }, h("div", { class: "cp-hue-knob" }));
  const modelBtn = h("button", { class: "fgc-model", type: "button", "aria-label": "Colour model" });
  const fields = h("div", { class: "fgc-fields" });
  const pick = (hex) => { [hue, sat, val] = rgbToHsv(...hexToRgb(hex)); paint(); set(hex.toLowerCase()); };

  const buildFields = () => {
    fields.textContent = "";
    modelBtn.innerHTML = `<span>${model}</span>${icon("chevron-down", 10)}`;
    if (model === "Hex") {
      const hex = h("input", { class: "cp-hex num-input", spellcheck: "false", "aria-label": "Hex colour" });
      hex.addEventListener("keydown", (e) => {
        if (e.key !== "Escape") e.stopPropagation();
        if (e.key === "Enter") {
          const v = hex.value.trim().replace(/^#?/, "#");
          if (/^#[0-9a-f]{6}$/i.test(v)) pick(v);
          else if (/^#[0-9a-f]{3}$/i.test(v)) pick(`#${[...v.slice(1)].map((c) => c + c).join("")}`);
          hex.select();
        }
      });
      hex.addEventListener("focus", () => hex.select());
      fields.append(hex);
    } else {
      const names = { RGB: ["R", "G", "B"], HSL: ["H", "S", "L"], HSB: ["H", "S", "B"] }[model];
      names.forEach((n, i) => {
        const inp = h("input", { class: "num-input fgc-part", spellcheck: "false", "aria-label": `${model} ${n}`, dataset: { part: i } });
        inp.addEventListener("keydown", (e) => {
          if (e.key !== "Escape") e.stopPropagation();
          if (e.key === "Enter") {
            const parts = [...fields.querySelectorAll(".fgc-part")].map((x) => parseFloat(x.value));
            if (parts.every(Number.isFinite)) {
              const [a, b, c] = parts;
              const rgb = model === "RGB" ? [a, b, c] : model === "HSL"
                ? hslToRgb(clamp(a, 0, 360) % 360, clamp(b, 0, 100) / 100, clamp(c, 0, 100) / 100)
                : hsvToRgb(clamp(a, 0, 360) % 360, clamp(b, 0, 100) / 100, clamp(c, 0, 100) / 100);
              pick(rgbToHex(...rgb));
            }
            inp.select();
          }
        });
        inp.addEventListener("focus", () => inp.select());
        fields.append(inp);
      });
    }
    paint();
  };
  modelBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    // A small list inside the panel (a second popover would close this one).
    const list = h("div", { class: "fgc-models" }, ...MODELS.map((m) => h("button", {
      class: `menu-item ${m === model ? "on" : ""}`, type: "button", dataset: { model: m },
      onclick: (ev) => { ev.stopPropagation(); model = m; list.remove(); buildFields(); },
    }, h("span", { class: "mi-icon", html: m === model ? icon("check", 12) : "" }), h("span", { class: "mi-label" }, m))));
    panel.querySelector(".fgc-models")?.remove();
    dropList(modelBtn.parentElement, list);
  });

  const swatches = [...new Set(docColors.map((c) => c.toLowerCase()))].slice(0, 16);
  const body = h("div", { class: "fgc-body" }, sv,
    h("div", { class: "fgc-sliders" }, hueBar),
    h("div", { class: "fgc-values" }, modelBtn, fields),
    swatches.length ? h("div", { class: "fgc-page" },
      h("div", { class: "fgc-page-title" }, "On this page"),
      h("div", { class: "fgc-swatches" }, ...swatches.map((c) => h("button", {
        class: "cp-swatch fgc-swatch", type: "button", title: c.toUpperCase(), style: { background: c }, onclick: () => pick(c) })))) : null);
  const panel = floatPanel("", body, { cls: "fg-color-panel" });
  // Figma's header is a tab strip: Custom (solid paint) is the one this editor has.
  panel.querySelector(".float-title").replaceWith(h("span", { class: "fgc-tab on" }, "Custom"));

  const paint = () => {
    sv.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, hsl(${hue}, 100%, 50%))`;
    sv.firstChild.style.left = `${sat * 100}%`;
    sv.firstChild.style.top = `${(1 - val) * 100}%`;
    hueBar.firstChild.style.left = `${(hue / 360) * 100}%`;
    hueBar.firstChild.style.background = `hsl(${hue}, 100%, 50%)`;
    const [r, g, b] = hsvToRgb(hue, sat, val);
    const vals = model === "RGB" ? [r, g, b].map(Math.round)
      : model === "HSL" ? (([a, s2, l]) => [Math.round(a), Math.round(s2 * 100), Math.round(l * 100)])(rgbToHsl(r, g, b))
        : model === "HSB" ? [Math.round(hue), Math.round(sat * 100), Math.round(val * 100)] : null;
    fields.querySelectorAll("input").forEach((inp, i) => {
      if (document.activeElement === inp) return;
      inp.value = vals ? String(vals[i]) : current().slice(1).toUpperCase();
    });
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
    // One edit per gesture (the canvas previews nothing mid-drag; the fields and knob do).
    const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); set(current()); };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
  drag(sv, (x, y) => { sat = x; val = 1 - y; });
  drag(hueBar, (x) => { hue = x * 359.9; });
  buildFields();
  popover(anchor, panel, { cls: "pop-color fg-float", beside: SIDE });
}

// ---------------------------------------------------------------- fonts

let filter = "all";

/** Figma's Fonts panel: search (pre-filled with the current font), a filter, and every font in its own face. */
export async function openFontPanel(anchor, get, set, docFonts = []) {
  const families = (await fontCatalog()).map((f) => f.family);
  const cur = get() || "";
  const inFile = [...new Set(docFonts)];
  const search = h("input", { class: "font-search fgf-search", placeholder: "Search fonts", spellcheck: "false", "aria-label": "Search fonts", value: cur });
  const filterBtn = h("button", { class: "fgf-filter", type: "button", "aria-label": "Font filter" });
  const list = h("div", { class: "font-list fgf-list", role: "listbox" });
  // Faces load as rows scroll into view, as Figma's list does.
  const seen = new IntersectionObserver((entries) => {
    const fams = entries.filter((e) => e.isIntersecting).map((e) => e.target.dataset.family);
    if (fams.length) ensureFonts(fams);
  }, { root: list, rootMargin: "120px" });
  const row = (fam) => {
    const on = fam === cur;
    const r = h("button", { class: `font-row ${on ? "on" : ""}`, type: "button", role: "option", dataset: { family: fam },
      onclick: () => { set(fam); ensureFonts([fam]); closePopover(); } },
    h("span", { class: "fgf-check", html: on ? icon("check", 12) : "" }), h("span", { class: "fgf-name", style: { fontFamily: `'${fam}'` } }, fam));
    seen.observe(r);
    return r;
  };
  const paint = (initial = false) => {
    const q = initial ? "" : search.value.trim().toLowerCase();
    filterBtn.hidden = !!q;  // Figma hides the filter while searching
    filterBtn.innerHTML = `<span>${filter === "all" ? "All fonts" : "In this file"}</span>${icon("chevron-down", 10)}`;
    list.textContent = "";
    const pool = filter === "file" ? inFile : families;
    pool.filter((f) => f.toLowerCase().includes(q)).forEach((f) => list.append(row(f)));
    if (!list.children.length) list.append(h("div", { class: "fgf-empty" }, "No fonts found"));
  };
  filterBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    panel.querySelector(".fgc-models")?.remove();
    const opts = h("div", { class: "fgc-models fgf-filters" }, ...[["all", "All fonts"], ["file", "In this file"]].map(([id, label]) => h("button", {
      class: `menu-item ${id === filter ? "on" : ""}`, type: "button", dataset: { filter: id },
      onclick: (ev) => { ev.stopPropagation(); filter = id; opts.remove(); paint(true); },
    }, h("span", { class: "mi-icon", html: id === filter ? icon("check", 12) : "" }), h("span", { class: "mi-label" }, label))));
    dropList(filterBtn.parentElement, opts);
  });
  search.addEventListener("input", () => paint());
  search.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") ev.stopPropagation();
    if (ev.key === "Enter") list.querySelector(".font-row")?.click();
  });
  const body = h("div", { class: "font-pop fgf-body" },
    h("div", { class: "fgf-searchrow" }, h("span", { class: "fgf-search-ic", html: icon("search", 12) }), search),
    h("div", { class: "fgf-filterrow" }, filterBtn), list);
  const panel = floatPanel("Fonts", body, { cls: "fg-font-panel" });
  paint(true);
  popover(anchor, panel, { cls: "pop-font fg-float", beside: SIDE });
  search.focus();
  search.select();
  list.querySelector(".font-row.on")?.scrollIntoView({ block: "center" });
}

// ---------------------------------------------------------------- typography

const STYLES = [["regular", "Regular", false, false], ["italic", "Italic", false, true], ["bold", "Bold", true, false], ["bolditalic", "Bold Italic", true, true]];
const SIZES = [10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 36, 40, 48, 64, 96, 128];

/** The Typography section's controls, given the inspector's selection model and registry. */
export function typography(app, m, reg, textOnly) {
  const tget = (p) => () => (textOnly() ? m.common(p, ["text"]) : undefined);
  const size0 = () => m.common("font_size", ["text"]) || 16;

  const weight = reg(h("button", { class: "fg-weight fg-select", type: "button", "aria-label": "Font style" }));
  const styleOf = () => {
    const b = m.common("bold", ["text"]), i = m.common("italic", ["text"]);
    if (b === null || i === null) return null;
    return STYLES.find((st) => st[2] === !!b && st[3] === !!i);
  };
  weight.addEventListener("click", (e) => {
    e.stopPropagation();
    const fam = m.common("font_family", ["text"]);
    const cur = styleOf();
    menu(weight, STYLES.filter(([id]) => id === "regular" || !fam || hasStyle(fam, id)).map(([id, label, bold, italic]) => ({
      label, icon: cur?.[0] === id ? "check" : null, run: () => m.set({ bold, italic }, ["text"]) })), { cls: "fg-weights" });
  });
  weight.set = () => {
    const st = styleOf();
    weight.innerHTML = `<span>${!textOnly() ? "" : st ? st[1] : "Mixed"}</span>${icon("chevron-down", 10)}`;
    weight.disabled = !textOnly();
  };

  const size = reg(numberField({ label: "", title: "Font size", name: "font_size", digits: 2, min: 1, max: 2000, cls: "fg-size",
    get: tget("font_size"), set: (v) => m.set({ font_size: v }, ["text"]) }));
  const caret = iconButton("chevron-down", "Font sizes", () => {
    const cur = m.common("font_size", ["text"]);
    menu(caret, [...(cur && !SIZES.includes(cur) ? [{ label: fmt(cur, 2), icon: "check", run: () => {} }, "-"] : []),
      ...SIZES.map((n) => ({ label: String(n), icon: n === cur ? "check" : null, run: () => m.set({ font_size: n }, ["text"]) }))], { cls: "fg-sizes" });
  }, { cls: "fg-size-caret", size: 10 });
  size.append(caret);

  // Line height as Figma shows it: a percentage; "120%" or "Auto", or a plain number in px.
  const lineH = reg(numberField({ label: "", title: "Line height", name: "line_height", iconName: "baseline", min: 0.1, max: 10, step: 0.01,
    get: tget("line_height"), set: (v) => m.set({ line_height: v }, ["text"]),
    format: (v) => `${fmt(v * 100, 2)}%`,
    parse: (t) => {
      if (/^auto$/i.test(t)) return 1.2;
      const n = parseFloat(t);
      if (!Number.isFinite(n)) return NaN;
      return /%\s*$/.test(t) ? n / 100 : n / size0();
    } }));
  // Letter spacing: px, or % of the font size; a bare number keeps the unit on show.
  const spacing = reg(numberField({ label: "", title: "Letter spacing", name: "letter_spacing", iconName: "letter-spacing", min: -50, max: 500, step: 0.1,
    get: tget("letter_spacing"), set: (v) => m.set({ letter_spacing: v }, ["text"]),
    format: (v) => (Math.abs(v) < 1e-9 ? "0%" : `${fmt(v, 2)}px`),
    parse: (t) => {
      const n = parseFloat(t);
      if (!Number.isFinite(n)) return NaN;
      const pct = /%\s*$/.test(t) || (!/px\s*$/i.test(t) && Math.abs(m.common("letter_spacing", ["text"]) || 0) < 1e-9);
      return pct ? (n / 100) * size0() : n;
    } }));

  const align = reg(segmented([["left", "align-left", "Align left"], ["center", "align-center", "Align center"],
    ["right", "align-right", "Align right"]], tget("align"), (v) => m.set({ align: v }, ["text"])));
  // Type settings: decoration and case, the details this editor supports.
  const typeSettings = iconButton("sliders-horizontal", "Type settings", () => {
    const deco = segmented([["none", "minus", "None"], ["underline", "underline", "Underline"]],
      () => (textOnly() ? (m.common("underline", ["text"]) ? "underline" : "none") : null),
      async (v) => { await m.set({ underline: v === "underline" }, ["text"]); deco.set(); });
    const kase = segmented([["none", "minus", "As typed"], ["upper", "case-upper", "Uppercase"]],
      () => (textOnly() ? (m.common("uppercase", ["text"]) ? "upper" : "none") : null),
      async (v) => { await m.set({ uppercase: v === "upper" }, ["text"]); kase.set(); });
    deco.dataset.field = "decoration";
    kase.dataset.field = "case";
    deco.set(); kase.set();
    const body = h("div", { class: "fgt-body" },
      h("div", { class: "fgt-row" }, h("span", { class: "fgt-label" }, "Decoration"), deco),
      h("div", { class: "fgt-row" }, h("span", { class: "fgt-label" }, "Case"), kase));
    popover(typeSettings, floatPanel("Type settings", body, { cls: "fg-type-panel" }), { cls: "fg-float fg-typeset", beside: SIDE });
  }, { cls: "fg-typeset-btn" });

  const label = (t) => h("div", { class: "insp-sub" }, t);
  return {
    rows: [
      h("div", { class: "insp-row fg-weightrow" }, weight, size),
      h("div", { class: "insp-subrow" }, label("Line height"), label("Letter spacing")),
      h("div", { class: "insp-row" }, lineH, spacing),
      label("Alignment"),
      h("div", { class: "insp-row fg-alignrow" }, align, h("span", { class: "grow" }), typeSettings),
    ],
  };
}
