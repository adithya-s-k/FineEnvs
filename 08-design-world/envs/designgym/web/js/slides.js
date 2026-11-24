// Google Slides' own menus and dialogs, as the current Slides draws them (checked in the browser):
// the colour menu (theme colours, CUSTOM with +, Google's 10 x 8 palette of round swatches) and the
// custom colour card (square, hue, Hex / R / G / B, Cancel / OK); the font menu (More fonts, the
// theme font, Recent, fonts in their own face) and the Fonts dialog (search, Sort, checkboxes,
// My fonts, Cancel / Done); line & paragraph spacing; Format options (collapsible Size & Rotation,
// Position, Adjustments); the Background dialog; the Menus search.

import { clamp, h } from "./util.js";
import { icon } from "./icons.js";
import { ensureFonts, fontCatalog } from "./fonts.js";
import { closePopover, hexToRgb, hsvToRgb, iconButton, menu, modal, numberField, popover, rgbToHex, rgbToHsv, slider } from "./widgets.js";

// Simple Light's theme colours, then Google's palette (greys, brights, then light 3..1 and dark 1..3).
const THEME = ["#000000", "#ffffff", "#595959", "#eeeeee", "#4285f4", "#212121", "#78909c", "#ffab40", "#0097a7", "#eeff41"];
const GOOGLE = [
  "#000000", "#434343", "#666666", "#999999", "#b7b7b7", "#cccccc", "#d9d9d9", "#efefef", "#f3f3f3", "#ffffff",
  "#980000", "#ff0000", "#ff9900", "#ffff00", "#00ff00", "#00ffff", "#4a86e8", "#0000ff", "#9900ff", "#ff00ff",
  "#e6b8af", "#f4cccc", "#fce5cd", "#fff2cc", "#d9ead3", "#d0e0e3", "#c9daf8", "#cfe2f3", "#d9d2e9", "#ead1dc",
  "#dd7e6b", "#ea9999", "#f9cb9c", "#ffe599", "#b6d7a8", "#a2c4c9", "#a4c2f4", "#9fc5e8", "#b4a7d6", "#d5a6bd",
  "#cc4125", "#e06666", "#f6b26b", "#ffd966", "#93c47d", "#76a5af", "#6d9eeb", "#6fa8dc", "#8e7cc3", "#c27ba0",
  "#a61c00", "#cc0000", "#e69138", "#f1c232", "#6aa84f", "#45818e", "#3c78d8", "#3d85c6", "#674ea7", "#a64d79",
  "#85200c", "#990000", "#b45f06", "#bf9000", "#38761d", "#134f5c", "#1155cc", "#0b5394", "#351c75", "#741b47",
  "#5b0f00", "#660000", "#783f04", "#7f6000", "#274e13", "#0c343d", "#1c4587", "#073763", "#20124d", "#4c1130",
];
const customs = [];  // colours added with CUSTOM + this session

/** The custom colour card: square, preview dot + hue, Hex / R / G / B, Cancel / OK. Enter is OK. */
export function customColorDialog(initial, onOK) {
  let [hue, sat, val] = rgbToHsv(...hexToRgb(initial || "#000000"));
  const current = () => rgbToHex(...hsvToRgb(hue, sat, val));
  const sv = h("div", { class: "cp-sv slc-sv" }, h("div", { class: "cp-sv-knob" }));
  const hueBar = h("div", { class: "cp-hue slc-hue" }, h("div", { class: "cp-hue-knob" }));
  const dot = h("span", { class: "slc-dot" });
  const hex = h("input", { class: "cp-hex num-input slc-hex", spellcheck: "false", "aria-label": "Hex" });
  const rgb = ["R", "G", "B"].map((n) => h("input", { class: "num-input slc-rgb", "aria-label": n, dataset: { chan: n } }));
  const paint = (skip) => {
    sv.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, hsl(${hue}, 100%, 50%))`;
    sv.firstChild.style.left = `${sat * 100}%`;
    sv.firstChild.style.top = `${(1 - val) * 100}%`;
    hueBar.firstChild.style.left = `${(hue / 360) * 100}%`;
    const c = current();
    dot.style.background = c;
    if (skip !== hex && document.activeElement !== hex) hex.value = c;
    hexToRgb(c).forEach((v, i) => { if (skip !== rgb[i] && document.activeElement !== rgb[i]) rgb[i].value = String(v); });
  };
  const ok = () => { closePopover(); card.remove(); onOK(current()); };
  const cancel = () => { card.remove(); };
  const fromHex = () => {
    let t = hex.value.trim().replace(/^#/, "");
    if (/^[0-9a-f]{3}$/i.test(t)) t = [...t].map((c) => c + c).join("");
    if (/^[0-9a-f]{6}$/i.test(t)) { [hue, sat, val] = rgbToHsv(...hexToRgb(`#${t}`)); paint(hex); return true; }
    return false;
  };
  hex.addEventListener("input", fromHex);
  hex.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") { fromHex(); ok(); } if (e.key === "Escape") cancel(); });
  hex.addEventListener("focus", () => hex.select());
  rgb.forEach((inp) => {
    inp.addEventListener("input", () => { const v = rgb.map((x) => clamp(parseInt(x.value, 10) || 0, 0, 255)); [hue, sat, val] = rgbToHsv(...v); paint(inp); });
    inp.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") ok(); if (e.key === "Escape") cancel(); });
  });
  const drag = (el, fn) => el.addEventListener("pointerdown", (e) => {
    el.setPointerCapture(e.pointerId);
    const at = (ev) => { const r = el.getBoundingClientRect(); fn(clamp((ev.clientX - r.left) / r.width, 0, 1), clamp((ev.clientY - r.top) / r.height, 0, 1)); paint(); };
    at(e);
    const move = (ev) => at(ev);
    const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
  drag(sv, (x, y) => { sat = x; val = 1 - y; });
  drag(hueBar, (x) => { hue = x * 359.9; });
  const card = h("div", { class: "slc-card", role: "dialog", "aria-label": "Custom color" }, sv,
    h("div", { class: "slc-row" }, dot, hueBar),
    h("div", { class: "slc-fields" },
      h("label", { class: "slc-f slc-f-hex" }, h("span", {}, "Hex"), hex),
      ...rgb.map((inp, i) => h("label", { class: "slc-f" }, h("span", {}, "RGB"[i]), inp))),
    h("div", { class: "slc-btns" }, h("button", { class: "btn slc-cancel", type: "button", onclick: cancel }, "Cancel"),
      h("button", { class: "btn slc-ok", type: "button", onclick: ok }, "OK")));
  document.body.append(card);
  paint();
  hex.focus();
  hex.select();
  return card;
}

const lum = (c) => { const [r, g, b] = hexToRgb(c); return (0.299 * r + 0.587 * g + 0.114 * b) / 255; };

/** The colour menu a Slides toolbar colour button opens. */
export function colorMenu(anchor, { get, set }) {
  const cur = (get() || "").toLowerCase();
  const pick = (c) => { closePopover(); set(c); };
  const sw = (c) => h("button", { class: `slc-sw ${c === cur ? "on" : ""}`, type: "button", title: c, "aria-label": c, dataset: { color: c },
    style: { background: c, color: lum(c) > 0.6 ? "#1f1f1f" : "#ffffff" }, onclick: () => pick(c) }, c === cur ? h("span", { class: "slc-tick", html: icon("check", 11) }) : null);
  const add = iconButton("circle-plus", "Add a custom color", () => {
    closePopover();
    customColorDialog(cur || "#000000", (c) => { if (!customs.includes(c)) customs.push(c); set(c); });
  }, { cls: "slc-add", size: 16 });
  popover(anchor, h("div", { class: "slc-menu" },
    h("div", { class: "slc-head" }, "SIMPLE LIGHT"), h("div", { class: "slc-grid slc-theme" }, ...THEME.map(sw)),
    h("div", { class: "slc-sep" }), h("div", { class: "slc-head" }, "CUSTOM"), h("div", { class: "slc-grid" }, add, ...customs.map(sw)),
    h("div", { class: "slc-sep" }), h("div", { class: "slc-grid slc-google" }, ...GOOGLE.map(sw))), { cls: "sl-pop" });
}

// ---------------------------------------------------------------- fonts

const MENU_FONTS = ["Amatic SC", "Arial", "Caveat", "Comfortaa", "Comic Sans MS", "Courier New", "DM Sans", "EB Garamond", "Georgia",
  "Impact", "Lexend", "Lobster", "Lora", "Merriweather", "Montserrat", "Nunito", "Open Sans", "Oswald", "Pacifico", "Playfair Display",
  "Poppins", "Raleway", "Roboto", "Roboto Mono", "Source Code Pro", "Spectral", "Times New Roman", "Trebuchet MS", "Verdana"];
const recent = [];
const myFonts = [];

function useFont(f) {
  const i = recent.indexOf(f);
  if (i >= 0) recent.splice(i, 1);
  recent.unshift(f);
  recent.length = Math.min(recent.length, 5);
}

/** The Fonts dialog (More fonts): search, Sort, checkboxes, My fonts, Cancel / Done (Done applies the last one ticked). */
export async function fontsDialog({ set }) {
  const families = (await fontCatalog()).map((f) => f.family);
  let sort = "Popularity";
  let chosen = null;
  const picked = new Set(myFonts);
  const search = h("input", { class: "font-search slf-search", placeholder: "Search", spellcheck: "false", "aria-label": "Search fonts" });
  const list = h("div", { class: "font-list slf-list", role: "listbox" });
  const mine = h("div", { class: "slf-mine" });
  const sortBtn = h("button", { class: "slf-drop", type: "button", "aria-label": "Sort" });
  const seen = new IntersectionObserver((es) => { const f = es.filter((x) => x.isIntersecting).map((x) => x.target.dataset.family); if (f.length) ensureFonts(f); }, { root: list });
  const paintMine = () => {
    mine.textContent = "";
    for (const f of picked) mine.append(h("div", { class: "slf-chip", style: { fontFamily: `'${f}'` } }, f,
      iconButton("x", `Remove ${f}`, () => { picked.delete(f); if (chosen === f) chosen = null; paint(); }, { cls: "slf-x", size: 12 })));
  };
  const paint = () => {
    seen.disconnect();
    const q = search.value.trim().toLowerCase();
    sortBtn.innerHTML = `<span class="slf-lab">Sort</span><span>${sort}</span>${icon("chevron-down", 12)}`;
    const rows = families.filter((f) => f.toLowerCase().includes(q));
    if (sort === "Alphabetical") rows.sort((a, b) => a.localeCompare(b));
    list.textContent = "";
    if (!rows.length) list.append(h("div", { class: "slf-empty" }, "No results"));
    for (const f of rows) {
      const on = picked.has(f);
      const r = h("button", { class: `font-row slf-row ${on ? "on" : ""}`, type: "button", role: "option", dataset: { family: f },
        onclick: () => { if (picked.has(f)) { picked.delete(f); if (chosen === f) chosen = null; } else { picked.add(f); chosen = f; } paint(); } },
      h("span", { class: "slf-box", html: on ? icon("check", 12) : "" }), h("span", { class: "slf-name", style: { fontFamily: `'${f}'` } }, f));
      seen.observe(r);
      list.append(r);
    }
    paintMine();
  };
  sortBtn.addEventListener("click", () => {
    const opts = h("div", { class: "slf-sorts" }, ...["Popularity", "Alphabetical"].map((o) => h("button", { class: `menu-item ${o === sort ? "on" : ""}`, type: "button",
      onclick: (e) => { e.stopPropagation(); sort = o; opts.remove(); paint(); } }, o)));
    sortBtn.parentElement.querySelector(".slf-sorts")?.remove();
    sortBtn.parentElement.append(opts);
  });
  search.addEventListener("input", paint);
  search.addEventListener("keydown", (e) => { if (e.key !== "Escape") e.stopPropagation(); if (e.key === "Enter") list.querySelector(".font-row")?.click(); });
  const body = h("div", { class: "font-pop slf" },
    h("div", { class: "slf-bar" }, h("div", { class: "slf-searchbox" }, h("span", { html: icon("search", 14) }), search), h("div", { class: "slf-sortwrap" }, sortBtn)),
    h("div", { class: "slf-cols" }, list, h("div", { class: "slf-side" }, h("div", { class: "slf-mine-title" }, "My fonts"), mine)));
  paint();
  modal("Fonts", body, {
    cls: "sl-fonts",
    actions: [{ label: "Cancel" }, { label: "Done", primary: true, run: () => {
      myFonts.splice(0, myFonts.length, ...picked);
      if (chosen) { ensureFonts([chosen]); useFont(chosen); set(chosen); }
    } }],
  });
  search.focus();
}

/** The font menu (the toolbar's font ▾): More fonts, the theme font, Recent, then the menu's fonts. */
export async function fontMenu(anchor, { get, set, docFonts = () => [] }) {
  const cur = get() || "";
  const have = new Set((await fontCatalog()).map((f) => f.family));
  const choose = (f) => { useFont(f); ensureFonts([f]); set(f); };
  const row = (f) => ({ label: f, icon: f === cur ? "check" : null, run: () => choose(f), family: f });
  const theme = [...new Set(docFonts())].filter((f) => have.has(f)).slice(0, 2);
  const menuFonts = [...new Set([...MENU_FONTS.filter((f) => have.has(f)), ...myFonts])].sort((a, b) => a.localeCompare(b));
  const items = [{ label: "More fonts", icon: "type", run: () => fontsDialog({ set: (f) => set(f) }) }, "-",
    ...(theme.length ? [{ header: "Simple light" }, ...theme.map(row), "-"] : []),
    ...(recent.length ? [{ header: "Recent" }, ...recent.map(row), "-"] : []),
    ...menuFonts.map(row)];
  const pop = menu(anchor, items, { cls: "sl-fontmenu" });
  ensureFonts([...theme, ...recent, ...menuFonts]);
  // Each font is listed in its own face, as in Slides.
  pop.querySelectorAll(".menu-item").forEach((b) => {
    const lab = b.querySelector(".mi-label");
    if (lab && lab.textContent !== "More fonts") { lab.style.fontFamily = `'${lab.textContent}'`; b.dataset.family = lab.textContent; b.classList.add("font-row"); }
  });
  pop.classList.add("font-pop");
}

// ---------------------------------------------------------------- spacing, alignment, background

/** Line & paragraph spacing: Single / 1.15 / 1.5 / Double, and Custom spacing. */
export function spacingMenu(anchor, m) {
  const cur = m.common("line_height", ["text"]);
  const opt = (label, v) => ({ label, icon: cur !== null && Math.abs((cur ?? 1) - v) < 1e-6 ? "check" : null, run: () => m.set({ line_height: v }, ["text"]) });
  menu(anchor, [opt("Single", 1), opt("1.15", 1.15), opt("1.5", 1.5), opt("Double", 2), "-",
    { label: "Custom spacing", run: () => customSpacing(m) }], { cls: "sl-spacing-menu" });
}

function customSpacing(m) {
  const lh = numberField({ label: "Line spacing", name: "line_height", digits: 2, step: 0.05, min: 0.06, max: 10,
    get: () => m.common("line_height", ["text"]), set: (v) => m.set({ line_height: v }, ["text"]) });
  const ls = numberField({ label: "Letter spacing (px)", name: "letter_spacing", digits: 2, min: -100, max: 500,
    get: () => m.common("letter_spacing", ["text"]), set: (v) => m.set({ letter_spacing: v }, ["text"]) });
  lh.set(); ls.set();
  modal("Custom spacing", h("div", { class: "sl-custom-spacing" }, lh, ls), { cls: "modal-small sl-spacing-dialog", actions: [{ label: "Cancel" }, { label: "Apply", primary: true }] });
  lh.input.focus();
}

/** Slides' alignment dropdown: left, center, right. */
export function alignMenu(anchor, m) {
  const cur = m.common("align", ["text"]) || "left";
  const pop = popover(anchor, h("div", { class: "sl-alignpop" }, ...["left", "center", "right"].map((v) => {
    const b = iconButton(`align-${v}`, `${v[0].toUpperCase()}${v.slice(1)}`, () => { closePopover(); m.set({ align: v }, ["text"]); }, { cls: v === cur ? "on" : "" });
    b.dataset.value = v;
    return b;
  })), { cls: "sl-pop" });
  return pop;
}

/** The Background dialog: Color (the colour menu), Reset to theme, Done. */
export function backgroundDialog(app) {
  const s = app.store;
  const chip = h("button", { class: "sl-bgchip", type: "button", "aria-label": "Background color" });
  const paint = () => { chip.innerHTML = `<span class="sl-bgdot" style="background:${s.doc.background}"></span><span>Color</span>${icon("chevron-down", 12)}`; };
  chip.addEventListener("click", () => colorMenu(chip, { get: () => s.doc.background, set: async (c) => { await s.op("set_background", { color: c }); paint(); } }));
  paint();
  modal("Background", h("div", { class: "sl-bgdialog" }, h("div", { class: "sl-bgrow" }, h("span", {}, "Color"), chip)), {
    cls: "modal-small sl-bg",
    actions: [{ label: "Reset to theme", run: async () => { await s.op("set_background", { color: "#ffffff" }); paint(); return true; } }, { label: "Done", primary: true }],
  });
}

// ---------------------------------------------------------------- Format options

/** Format options: collapsible Size & Rotation, Position and Adjustments (transparency), in px. */
export function formatOptions(app, m) {
  const open = { size: true, position: true, adjust: false };
  let lock = false;
  let fromCenter = false;
  let base = null;  // sizes the Scale fields are relative to (the selection when it was made)
  const root = h("div", { class: "panel-body inspector sl-format" });
  const num = (name, label, get, set, opts = {}) => {
    const f = numberField({ label, title: label, name, digits: 2, min: opts.min ?? -1e5, max: opts.max ?? 1e5, unit: opts.unit ?? "px", get, set, cls: "sl-num" });
    f.set();
    return f;
  };
  const sec = (id, title, rows, extra = null) => {
    const head = h("button", { class: "sl-fo-head", type: "button", "aria-expanded": String(open[id]) },
      h("span", { class: "sl-fo-chev", html: icon(open[id] ? "chevron-down" : "chevron-right", 14) }), title, h("span", { class: "grow" }), extra);
    head.addEventListener("click", () => { open[id] = !open[id]; paint(); });
    return h("section", { class: `sl-fo-sec ${open[id] ? "open" : ""}`, dataset: { sec: id } }, head, open[id] ? h("div", { class: "sl-fo-body" }, ...rows) : null);
  };
  const paint = () => {
    root.textContent = "";
    const els = m.els();
    if (!els.length) { root.append(h("div", { class: "sl-fo-empty" }, "Select an object to format it.")); return; }
    const e0 = els[0];
    const sid = els.map((e) => e.id).join();
    if (!base || base.id !== sid) base = { id: sid, w: e0.width, h: e0.height };
    const setSize = (prop) => (v) => m.set((e) => {
      if (lock && e.kind !== "text") { const r = e.width / e.height; return prop === "width" ? { width: v, height: v / r } : { height: v, width: v * r }; }
      return { [prop]: v };
    });
    const isText = els.every((e) => e.kind === "text");
    const w = num("width", "Width", () => m.common("width"), setSize("width"), { min: 1 });
    const hh = num("height", "Height", () => (isText ? undefined : m.common("height")), setSize("height"), { min: 1 });
    const ws = num("width_scale", "Width Scale", () => (m.common("width") ?? 0) / base.w * 100, (v) => m.set((e) => ({ width: base.w * v / 100, ...(lock && e.kind !== "text" ? { height: base.h * v / 100 } : {}) })), { unit: "%", min: 1 });
    const hs = num("height_scale", "Height Scale", () => (isText ? undefined : (m.common("height") ?? 0) / base.h * 100), (v) => m.set({ height: base.h * v / 100 }), { unit: "%", min: 1 });
    const lockBox = h("label", { class: "sl-check" }, h("input", { type: "checkbox", checked: lock, onchange: (ev) => { lock = ev.target.checked; } }), "Lock aspect ratio");
    const angle = num("angle", "Angle", () => m.common("angle"), (v) => m.set({ angle: v }), { unit: "°", min: -360, max: 360 });
    const rot90 = iconButton("rotate-cw", "Rotate clockwise 90°", () => app.commands.rotateCw.run(), { cls: "sl-fo-btn" });
    const flipH = iconButton("flip-horizontal-2", "Flip horizontally", () => app.commands.flipH.run(), { cls: "sl-fo-btn" });
    const flipV = iconButton("flip-vertical-2", "Flip vertically", () => app.commands.flipV.run(), { cls: "sl-fo-btn" });
    const cx = (e) => e.x + (fromCenter ? e.width / 2 : 0), cy = (e) => e.y + (fromCenter ? e.height / 2 : 0);
    const x = num("x", "X", () => (els.length === 1 ? cx(e0) : m.common("x")), (v) => m.set((e) => ({ x: fromCenter ? v - e.width / 2 : v })));
    const y = num("y", "Y", () => (els.length === 1 ? cy(e0) : m.common("y")), (v) => m.set((e) => ({ y: fromCenter ? v - e.height / 2 : v })));
    const from = h("button", { class: "sl-from", type: "button", "aria-label": "From" });
    from.innerHTML = `<span>${fromCenter ? "Center" : "Top left"}</span>${icon("chevron-down", 12)}`;
    from.addEventListener("click", () => menu(from, [{ label: "Top left", run: () => { fromCenter = false; paint(); } }, { label: "Center", run: () => { fromCenter = true; paint(); } }]));
    // Transparency is how Slides shows opacity (0% = fully opaque).
    const tr = slider({ label: "Transparency", name: "opacity", unit: "%", min: 0, max: 100,
      get: () => { const v = m.common("opacity"); return v === null || v === undefined ? v : Math.round((1 - v) * 100); }, set: (v) => m.set({ opacity: 1 - v / 100 }) });
    tr.set();
    root.append(
      sec("size", "Size & Rotation", [h("div", { class: "sl-fo-label" }, "Size"), h("div", { class: "sl-fo-grid" }, w, hh, ws, hs), lockBox,
        h("div", { class: "sl-fo-label" }, "Rotate"), h("div", { class: "sl-fo-rot" }, angle, h("div", { class: "sl-fo-mini" }, h("span", {}, "90°"), rot90),
          h("div", { class: "sl-fo-mini" }, h("span", {}, "Flip"), h("div", {}, flipH, flipV)))]),
      sec("position", "Position", [h("div", { class: "sl-fo-label" }, "From"), from, h("div", { class: "sl-fo-grid" }, x, y)]),
      sec("adjust", "Adjustments", [tr]));
  };
  return { el: root, refresh: () => { if (!root.contains(document.activeElement)) paint(); } };
}

/** "Search the menus": type part of a command's name, Enter runs the first match. */
export function menuSearch(app, commands) {
  const input = h("input", { class: "sl-menusearch", placeholder: "Menus", spellcheck: "false", "aria-label": "Search the menus (Alt+/)" });
  const box = h("label", { class: "sl-menusearchbox" }, h("span", { html: icon("search", 14) }), input);
  const matches = () => {
    const q = input.value.trim().toLowerCase();
    return q ? commands().filter((c) => c.label.toLowerCase().replace(/…/g, "").includes(q) && (!c.enabled || c.enabled())).slice(0, 8) : [];
  };
  const show = () => {
    const found = matches();
    if (!found.length) { closePopover(); return; }
    const pop = menu(box, found.map((c) => ({ label: c.label, run: () => { input.value = ""; c.run(); } })), { cls: "sl-menusearch-pop" });
    pop.querySelectorAll(".menu-item").forEach((b) => b.setAttribute("tabindex", "-1"));
    input.focus();
  };
  input.addEventListener("input", show);
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { const f = matches()[0]; closePopover(); input.value = ""; input.blur(); f?.run(); }
    if (e.key === "Escape") { closePopover(); input.value = ""; input.blur(); }
  });
  return box;
}

