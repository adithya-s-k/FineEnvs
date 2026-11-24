// PowerPoint for the web's own controls, as it draws them: the colour gallery (Theme Colors with their
// tints and shades, Standard Colors, Recent Colors, More Colors...) and the Colors dialog; the font and
// size comboboxes (type a name, or pick from the list); Align and Line Spacing dropdowns; the Font dialog
// (Font / Character Spacing) and the Paragraph dialog; the Format pane (Shape Options / Text Options,
// Fill & Line, Size & Properties, Picture) and Format Background; the header's Search box.

import { clamp, h } from "./util.js";
import { icon } from "./icons.js";
import { ensureFonts, fontCatalog } from "./fonts.js";
import { closePopover, hexToRgb, hsvToRgb, iconButton, menu, modal, numberField, popover, rgbToHex, rgbToHsv, slider } from "./widgets.js";

// ---------------------------------------------------------------- colours

// The Office Theme (2023) colours, in gallery order, and the ten Standard Colors.
const THEME = [["#FFFFFF", "White, Background 1"], ["#000000", "Black, Text 1"], ["#E8E8E8", "Light Gray, Background 2"],
  ["#0E2841", "Dark Blue, Text 2"], ["#156082", "Dark Teal, Accent 1"], ["#E97132", "Orange, Accent 2"], ["#196B24", "Green, Accent 3"],
  ["#0F9ED5", "Blue, Accent 4"], ["#A02B93", "Purple, Accent 5"], ["#4EA72E", "Green, Accent 6"]];
const STANDARD = [["#C00000", "Dark Red"], ["#FF0000", "Red"], ["#FFC000", "Orange"], ["#FFFF00", "Yellow"], ["#92D050", "Light Green"],
  ["#00B050", "Green"], ["#00B0F0", "Light Blue"], ["#0070C0", "Blue"], ["#002060", "Dark Blue"], ["#7030A0", "Purple"]];
const recent = [];

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  if (mx === mn) return [0, 0, l];
  const d = mx - mn, s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  const hh = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [hh * 60, s, l];
}
function hslToHex(hh, s, l) {
  const k = (n) => (n + hh / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = (n) => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1))));
  return rgbToHex(f(0), f(8), f(4));
}
// Office's tint / shade rows: which five variants a theme colour gets depends on how light it is.
function variants(hex) {
  const [hh, s, l] = rgbToHsl(...hexToRgb(hex));
  const lighter = (p) => [hslToHex(hh, s, l + (1 - l) * p), `Lighter ${Math.round(p * 100)}%`];
  const darker = (p) => [hslToHex(hh, s, l * (1 - p)), `Darker ${Math.round(p * 100)}%`];
  if (l >= 0.999) return [0.05, 0.15, 0.25, 0.35, 0.5].map(darker);
  if (l <= 0.001) return [0.5, 0.35, 0.25, 0.15, 0.05].map(lighter);
  if (l > 0.8) return [0.1, 0.25, 0.5, 0.75, 0.9].map(darker);
  if (l < 0.2) return [...[0.9, 0.75, 0.5, 0.25, 0.1].map(lighter)];
  return [lighter(0.8), lighter(0.6), lighter(0.4), darker(0.25), darker(0.5)];
}

/** The Colors dialog (More Colors...): colour field, hue slider, Hex / Red / Green / Blue, New / Current, OK / Cancel. */
export function moreColors(initial, onOK) {
  const init = (initial || "#000000").slice(0, 7);
  let [hue, sat, val] = rgbToHsv(...hexToRgb(init));
  const current = () => rgbToHex(...hsvToRgb(hue, sat, val));
  const sv = h("div", { class: "cp-sv ppc-sv" }, h("div", { class: "cp-sv-knob" }));
  const hueBar = h("div", { class: "cp-hue ppc-hue" }, h("div", { class: "cp-hue-knob" }));
  const hex = h("input", { class: "cp-hex num-input ppc-hex", spellcheck: "false", "aria-label": "Hex" });
  const rgb = ["Red", "Green", "Blue"].map((n) => h("input", { class: "num-input ppc-rgb", "aria-label": n }));
  const now = h("div", { class: "ppc-new" }), was = h("div", { class: "ppc-cur", style: { background: init } });
  const paint = (skip) => {
    sv.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, hsl(${hue}, 100%, 50%))`;
    sv.firstChild.style.left = `${sat * 100}%`;
    sv.firstChild.style.top = `${(1 - val) * 100}%`;
    hueBar.firstChild.style.left = `${(hue / 360) * 100}%`;
    const c = current();
    now.style.background = c;
    if (skip !== hex && document.activeElement !== hex) hex.value = c.slice(1).toUpperCase();
    hexToRgb(c).forEach((v, i) => { if (skip !== rgb[i] && document.activeElement !== rgb[i]) rgb[i].value = String(v); });
  };
  const fromHex = () => {
    let t = hex.value.trim().replace(/^#/, "");
    if (/^[0-9a-f]{3}$/i.test(t)) t = [...t].map((c) => c + c).join("");
    if (/^[0-9a-f]{6}$/i.test(t)) { [hue, sat, val] = rgbToHsv(...hexToRgb(`#${t}`)); paint(hex); }
  };
  let dlg = null;
  const ok = () => { dlg.close(); const c = current(); if (!recent.includes(c)) recent.unshift(c); recent.length = Math.min(recent.length, 10); onOK(c); };
  hex.addEventListener("input", fromHex);
  hex.addEventListener("focus", () => hex.select());
  const keys = (e) => { if (e.key === "Escape") return; e.stopPropagation(); if (e.key === "Enter") { e.preventDefault(); ok(); } };
  hex.addEventListener("keydown", keys);
  rgb.forEach((inp) => {
    inp.addEventListener("input", () => { [hue, sat, val] = rgbToHsv(...rgb.map((x) => clamp(parseInt(x.value, 10) || 0, 0, 255))); paint(inp); });
    inp.addEventListener("keydown", keys);
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
  const body = h("div", { class: "ppc-dialog" },
    h("div", { class: "ppc-main" }, sv, h("div", { class: "ppc-side" },
      h("label", { class: "ppc-f" }, h("span", {}, "Hex"), h("span", { class: "ppc-hash" }, "#"), hex),
      ...rgb.map((inp, i) => h("label", { class: "ppc-f" }, h("span", {}, ["Red", "Green", "Blue"][i]), inp)),
      h("div", { class: "ppc-cmp" }, h("div", { class: "ppc-cap" }, "New"), now, was, h("div", { class: "ppc-cap" }, "Current")))),
    hueBar);
  paint();
  dlg = modal("Colors", body, { cls: "pp-dialog pp-colors", actions: [{ label: "OK", primary: true, run: () => { ok(); return true; } }, { label: "Cancel" }] });
  hex.focus();
  hex.select();
  return dlg;
}

/** The colour gallery a Font Color / Shape Fill / Shape Outline dropdown opens. */
export function colorGallery(anchor, { get, set, automatic = null, none = null }) {
  const cur = (get() || "").toLowerCase();
  const pick = (c) => { closePopover(); if (c && !recent.includes(c) && !THEME.concat(STANDARD).some(([x]) => x.toLowerCase() === c)) recent.unshift(c); set(c); };
  const sw = (c, name) => h("button", { class: `ppc-sw ${c.toLowerCase() === cur ? "on" : ""}`, type: "button", title: name, "aria-label": name,
    dataset: { color: c.toLowerCase() }, style: { background: c }, onclick: () => pick(c.toLowerCase()) });
  const theme = h("div", { class: "ppc-grid ppc-theme" }, ...THEME.map(([c, n]) => sw(c, n)));
  const tints = h("div", { class: "ppc-grid ppc-tints" });
  for (let row = 0; row < 5; row++) THEME.forEach(([c, n]) => { const [v, lab] = variants(c)[row]; tints.append(sw(v, `${n}, ${lab}`)); });
  const item = (label, ic, run, cls = "") => h("button", { class: `ppc-item ${cls}`, type: "button", onclick: run },
    h("span", { class: "ppc-ic", html: ic ? icon(ic, 14) : "" }), label);
  const body = h("div", { class: "ppc-menu" },
    automatic ? item(automatic, null, () => pick("#000000"), "ppc-auto") : null,
    none ? item(none, "slash", () => pick(null), "ppc-none") : null,
    h("div", { class: "ppc-head" }, "Theme Colors"), theme, tints,
    h("div", { class: "ppc-head" }, "Standard Colors"), h("div", { class: "ppc-grid" }, ...STANDARD.map(([c, n]) => sw(c, n))),
    recent.length ? h("div", { class: "ppc-head" }, "Recent Colors") : null,
    recent.length ? h("div", { class: "ppc-grid" }, ...recent.map((c) => sw(c, c.toUpperCase()))) : null,
    h("div", { class: "ppc-sep" }),
    item("More Colors...", "palette", () => { closePopover(); moreColors(cur || "#000000", (c) => set(c)); }, "pp-morecolors"));
  popover(anchor, body, { cls: "pp-pop" });
}

// ---------------------------------------------------------------- fonts

const recentFonts = [];

/** The Font combobox: type a font name (Enter applies it) or pick from Theme / Recently Used / All Fonts. */
export function fontCombo({ get, set, docFonts = () => [] }) {
  const input = h("input", { class: "pp-fontinput font-search", spellcheck: "false", autocomplete: "off", "aria-label": "Font" });
  const caret = h("button", { class: "pp-combo-caret", type: "button", tabindex: "-1", "aria-label": "Font list", html: icon("chevron-down", 12) });
  const el = h("div", { class: "font-btn pp-font pp-combo", role: "combobox" }, input, caret);
  let families = null, list = null;
  const choose = (f) => {
    const i = recentFonts.indexOf(f);
    if (i >= 0) recentFonts.splice(i, 1);
    recentFonts.unshift(f);
    recentFonts.length = Math.min(recentFonts.length, 6);
    ensureFonts([f]);
    closePopover();
    input.value = f;
    input.blur();
    set(f);
  };
  const row = (f) => h("button", { class: `font-row pp-fontrow ${f === get() ? "on" : ""}`, type: "button", role: "option", dataset: { family: f },
    style: { fontFamily: `'${f}'` }, onpointerdown: (e) => e.preventDefault(), onclick: () => choose(f) }, f);
  const paint = (filter) => {
    const q = filter ? input.value.trim().toLowerCase() : "";
    list.textContent = "";
    const add = (title, fams) => { if (!fams.length) return; list.append(h("div", { class: "font-group" }, title)); fams.forEach((f) => list.append(row(f))); };
    const match = (f) => !q || f.toLowerCase().includes(q);
    if (!q) {
      add("Theme Fonts", [...new Set(docFonts())].filter((f) => families.includes(f)).slice(0, 2));
      add("Recently Used Fonts", recentFonts.filter(match));
    }
    add("All Fonts", families.filter(match));
    const seen = new IntersectionObserver((es) => { const f = es.filter((x) => x.isIntersecting).map((x) => x.target.dataset.family); if (f.length) ensureFonts(f); }, { root: list });
    list.querySelectorAll(".font-row").forEach((r) => seen.observe(r));
  };
  const openList = async (filter = false) => {
    families = families || (await fontCatalog()).map((f) => f.family);
    if (!list || !list.isConnected) {
      list = h("div", { class: "font-list pp-fontlist", role: "listbox" });
      popover(el, h("div", { class: "font-pop" }, list), { cls: "pop-font pp-pop" });
    }
    paint(filter);
  };
  input.addEventListener("focus", () => { input.select(); openList(); });
  input.addEventListener("input", () => openList(true));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { input.value = get() || ""; closePopover(); input.blur(); return; }
    e.stopPropagation();
    if (e.key === "Enter") {
      e.preventDefault();
      const t = input.value.trim().toLowerCase();
      const exact = (families || []).find((f) => f.toLowerCase() === t);
      const first = list?.querySelector(".font-row")?.dataset.family;
      if (exact || first) choose(exact || first);
    }
  });
  caret.addEventListener("click", () => { if (list?.isConnected) closePopover(); else { input.focus(); } });
  el.set = () => { if (document.activeElement !== input) input.value = get() || ""; const off = get() === undefined; input.disabled = off; caret.disabled = off; };
  return el;
}

const SIZES = [8, 9, 10, 10.5, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 54, 60, 66, 72, 80, 88, 96];

/** The Font Size combobox, with its list of sizes; and Increase / Decrease Font Size. */
export function sizeCombo({ get, set }) {
  const f = numberField({ label: "", title: "Font size", digits: 1, min: 1, max: 4000, cls: "pp-size", get, set });
  const caret = h("button", { class: "pp-combo-caret", type: "button", "aria-label": "Font size list", html: icon("chevron-down", 12) });
  caret.addEventListener("click", () => menu(caret, SIZES.map((v) => ({ label: String(v), icon: get() === v ? "check" : null, run: () => set(v) })), { cls: "pp-sizes" }));
  f.append(caret);
  const base = f.set;
  f.set = () => { base?.(); const off = get() === undefined; f.input.disabled = off; caret.disabled = off; f.classList.toggle("disabled", off); };
  const step = (d) => { const v = get(); if (v === undefined) return;
    const next = d > 0 ? SIZES.find((s) => s > v + 1e-6) ?? Math.round(v + 10) : [...SIZES].reverse().find((s) => s < v - 1e-6) ?? Math.max(1, v - 1); set(next); };
  return { field: f, grow: () => step(1), shrink: () => step(-1) };
}

// ---------------------------------------------------------------- paragraph + font dialogs

/** Line Spacing: 1.0 / 1.5 / 2.0 / 2.5 / 3.0, and Line Spacing Options... (the Paragraph dialog). */
export function lineSpacingMenu(anchor, app, m) {
  const cur = m.common("line_height", ["text"]);
  menu(anchor, [...[1, 1.5, 2, 2.5, 3].map((v) => ({ label: v.toFixed(1), icon: cur !== undefined && Math.abs((cur ?? 1) - v) < 1e-6 ? "check" : null,
    run: () => m.set({ line_height: v }, ["text"]) })), "-", { label: "Line Spacing Options...", run: () => paragraphDialog(app, m) }], { cls: "pp-linemenu" });
}

// A dropdown that looks like Office's: a box with the value and a chevron, a list of choices.
function dropdown(options, value, onChange, cls = "") {
  let v = value;
  const b = h("button", { class: `pp-dd ${cls}`, type: "button" });
  const paint = () => { b.innerHTML = `<span>${options.find((o) => o[0] === v)?.[1] ?? ""}</span>${icon("chevron-down", 12)}`; };
  b.addEventListener("click", () => menu(b, options.map(([k, label]) => ({ label, icon: k === v ? "check" : null, run: () => { v = k; paint(); onChange(k); } })), { cls: "pp-ddmenu" }));
  paint();
  b.value = () => v;
  return b;
}

/** Paragraph: Alignment; Line Spacing (Single / 1.5 lines / Double / Multiple) At. */
export function paragraphDialog(app, m) {
  const lh = m.common("line_height", ["text"]) ?? 1;
  let align = m.common("align", ["text"]) || "left";
  let mode = { 1: "single", 1.5: "one5", 2: "double" }[lh] || "multiple";
  const at = numberField({ label: "At", name: "line_height", digits: 2, step: 0.1, min: 0, max: 9.99, get: () => at.v, set: (x) => { at.v = x; } });
  at.v = lh;
  at.set();
  const alignDd = dropdown([["left", "Left"], ["center", "Centered"], ["right", "Right"]], align, (k) => { align = k; }, "pp-para-align");
  const modeDd = dropdown([["single", "Single"], ["one5", "1.5 lines"], ["double", "Double"], ["multiple", "Multiple"]], mode, (k) => {
    mode = k; at.v = { single: 1, one5: 1.5, double: 2 }[k] ?? at.v; at.set(); }, "pp-para-spacing");
  const body = h("div", { class: "pp-dlgbody" },
    h("div", { class: "pp-dlgtabs" }, h("span", { class: "pp-dlgtab on" }, "Indents and Spacing")),
    h("div", { class: "pp-dlgsec" }, "General"), h("div", { class: "pp-dlgrow" }, h("span", {}, "Alignment:"), alignDd),
    h("div", { class: "pp-dlgsec" }, "Spacing"), h("div", { class: "pp-dlgrow" }, h("span", {}, "Line Spacing:"), modeDd, at));
  modal("Paragraph", body, { cls: "pp-dialog pp-paragraph", actions: [{ label: "OK", primary: true, run: () => {
    at.input.blur();
    m.set({ align, line_height: at.v }, ["text"]);
  } }, { label: "Cancel" }] });
}

/** Font: the Font tab (font, style, size, colour, underline, All Caps) and Character Spacing (Normal / Expanded / Condensed, By). */
export function fontDialog(app, m, tab = "font") {
  const g = (p) => m.common(p, ["text"]);
  const st = { font_family: g("font_family"), bold: !!g("bold"), italic: !!g("italic"), font_size: g("font_size"), color: g("color") || "#000000",
    underline: !!g("underline"), uppercase: !!g("uppercase"), letter_spacing: g("letter_spacing") || 0 };
  const tabs = h("div", { class: "pp-dlgtabs" });
  const host = h("div", { class: "pp-dlgpage" });
  const fontBox = fontCombo({ get: () => st.font_family, set: (f) => { st.font_family = f; }, docFonts: () => [] });
  fontBox.set();
  const style = dropdown([["r", "Regular"], ["i", "Italic"], ["b", "Bold"], ["bi", "Bold Italic"]], `${st.bold ? "b" : ""}${st.italic ? "i" : ""}` || "r",
    (k) => { st.bold = k.includes("b"); st.italic = k.includes("i"); }, "pp-fd-style");
  const size = numberField({ label: "Size:", name: "font_size", digits: 1, min: 1, max: 4000, get: () => st.font_size, set: (v) => { st.font_size = v; } });
  size.set();
  const chip = h("button", { class: "pp-dd pp-fd-color", type: "button", "aria-label": "Font color" });
  const paintChip = () => { chip.innerHTML = `<span class="pp-fd-a">A</span><span class="pp-fd-bar" style="background:${st.color}"></span>${icon("chevron-down", 12)}`; };
  paintChip();
  chip.addEventListener("click", () => colorGallery(chip, { get: () => st.color, set: (c) => { st.color = c || "#000000"; paintChip(); } }));
  const under = dropdown([[false, "(none)"], [true, "Single"]], st.underline, (k) => { st.underline = k; }, "pp-fd-under");
  const caps = h("label", { class: "pp-check" }, h("input", { type: "checkbox", checked: st.uppercase, dataset: { field: "uppercase" },
    onchange: (e) => { st.uppercase = e.target.checked; } }), "All Caps");
  let kind = st.letter_spacing > 0 ? "expanded" : st.letter_spacing < 0 ? "condensed" : "normal";
  const by = numberField({ label: "By:", name: "letter_spacing", digits: 2, min: 0, max: 1584, get: () => Math.abs(st.letter_spacing), set: (v) => {
    st.letter_spacing = kind === "condensed" ? -v : kind === "normal" ? (v ? v : 0) : v; if (kind === "normal" && v) { kind = "expanded"; spacing.repaint(); } } });
  by.set();
  const spacing = dropdown([["normal", "Normal"], ["expanded", "Expanded"], ["condensed", "Condensed"]], kind, (k) => {
    kind = k; const v = Math.abs(st.letter_spacing); st.letter_spacing = k === "normal" ? 0 : k === "condensed" ? -v : v; by.set(); }, "pp-fd-spacing");
  spacing.repaint = () => { spacing.querySelector("span").textContent = { normal: "Normal", expanded: "Expanded", condensed: "Condensed" }[kind]; };
  const pages = {
    font: h("div", { class: "pp-fd-font" },
      h("div", { class: "pp-fd-grid" }, h("label", {}, "Latin text font:"), h("label", {}, "Font style:"), h("label", {}, "Size:"),
        fontBox, style, size),
      h("div", { class: "pp-dlgrow" }, h("span", {}, "Font color"), chip, h("span", {}, "Underline style"), under),
      h("div", { class: "pp-dlgsec" }, "Effects"), caps),
    spacing: h("div", { class: "pp-fd-spacingpage" }, h("div", { class: "pp-dlgrow" }, h("span", {}, "Spacing:"), spacing, by)),
  };
  const show = (id) => {
    tab = id;
    tabs.querySelectorAll(".pp-dlgtab").forEach((b) => b.classList.toggle("on", b.dataset.tab === id));
    host.textContent = "";
    host.append(pages[id]);
  };
  for (const [id, label] of [["font", "Font"], ["spacing", "Character Spacing"]])
    tabs.append(h("button", { class: "pp-dlgtab", type: "button", dataset: { tab: id }, onclick: () => show(id) }, label));
  show(tab);
  modal("Font", h("div", { class: "pp-dlgbody" }, tabs, host), { cls: "pp-dialog pp-fontdlg", actions: [{ label: "OK", primary: true, run: () => {
    document.activeElement?.blur?.();
    const patch = { bold: st.bold, italic: st.italic, underline: st.underline, uppercase: st.uppercase, color: st.color, letter_spacing: st.letter_spacing };
    if (st.font_family) { patch.font_family = st.font_family; ensureFonts([st.font_family]); }
    if (st.font_size) patch.font_size = st.font_size;
    m.set(patch, ["text"]);
  } }, { label: "Cancel" }] });
}

// ---------------------------------------------------------------- Format pane

/** The Format pane: Format Shape / Format Picture / Format Background, as PowerPoint lays them out. */
export function formatPane(app, m) {
  const s = app.store;
  const root = h("div", { class: "panel-body inspector pp-format" });
  const state = { top: "shape", sub: "size", open: { fill: true, line: false, size: true, position: true, textfill: true, transparency: true, bg: true } };
  let base = null, lock = false, fromCenter = false;
  const num = (name, label, get, set, opts = {}) => {
    const f = numberField({ label, title: label, name, digits: 2, min: opts.min ?? -1e5, max: opts.max ?? 1e5, unit: opts.unit ?? "px", get, set, cls: "pp-num" });
    f.set();
    return h("div", { class: "pp-frow" }, h("span", { class: "pp-flabel" }, label), f);
  };
  const sec = (id, title, ...rows) => {
    const head = h("button", { class: "pp-fsec-head", type: "button", "aria-expanded": String(state.open[id]) },
      h("span", { html: icon(state.open[id] ? "chevron-down" : "chevron-right", 14) }), title);
    head.addEventListener("click", () => { state.open[id] = !state.open[id]; paint(); });
    return h("section", { class: `pp-fsec ${state.open[id] ? "open" : ""}`, dataset: { sec: id } }, head, state.open[id] ? h("div", { class: "pp-fsec-body" }, ...rows) : null);
  };
  const radio = (label, on, run) => h("label", { class: "pp-radio" }, h("input", { type: "radio", checked: on, onchange: run }), label);
  const colorRow = (label, get, set, opts = {}) => {
    const b = h("button", { class: "pp-colorbtn", type: "button", "aria-label": label, dataset: { field: opts.name || "" } });
    const c = get();
    b.innerHTML = `<span class="pp-colorbar" style="background:${c || "transparent"}"></span>${icon("chevron-down", 12)}`;
    b.addEventListener("click", () => colorGallery(b, { get, set, automatic: opts.automatic || null }));
    return h("div", { class: "pp-frow" }, h("span", { class: "pp-flabel" }, label), b);
  };
  const transparency = () => {
    const tr = slider({ label: "", name: "opacity", unit: "%", min: 0, max: 100,
      get: () => { const v = m.common("opacity"); return v === null || v === undefined ? v : Math.round((1 - v) * 100); }, set: (v) => m.set({ opacity: 1 - v / 100 }) });
    tr.set();
    return h("div", { class: "pp-frow pp-trow" }, h("span", { class: "pp-flabel" }, "Transparency"), tr);
  };
  const iconTabs = (tabs) => h("div", { class: "pp-ftabs" }, ...tabs.map(([id, ic, label]) => {
    const b = iconButton(ic, label, () => { state.sub = id; paint(); }, { cls: `pp-ftab ${state.sub === id ? "on" : ""}`, size: 18 });
    b.dataset.sub = id;
    return b;
  }));
  const sizeSections = (els) => {
    const e0 = els[0], sid = els.map((e) => e.id).join(), isText = els.every((e) => e.kind === "text");
    if (!base || base.id !== sid) base = { id: sid, w: e0.width, h: e0.height };
    const setSize = (prop) => (v) => m.set((e) => (lock && e.kind !== "text" ? (prop === "width" ? { width: v, height: v * e.height / e.width } : { height: v, width: v * e.width / e.height }) : { [prop]: v }));
    const cx = (e) => e.x + (fromCenter ? e.width / 2 : 0), cy = (e) => e.y + (fromCenter ? e.height / 2 : 0);
    const from = () => { const b = dropdown([[false, "Top Left Corner"], [true, "Center"]], fromCenter, (k) => { fromCenter = k; paint(); }, "pp-from"); return h("div", { class: "pp-frow" }, h("span", { class: "pp-flabel" }, "From"), b); };
    return [
      sec("size", "Size",
        num("height", "Height", () => (isText ? undefined : m.common("height")), setSize("height"), { min: 1 }),
        num("width", "Width", () => m.common("width"), setSize("width"), { min: 1 }),
        num("angle", "Rotation", () => m.common("angle"), (v) => m.set({ angle: v }), { unit: "°", min: -3600, max: 3600 }),
        num("height_scale", "Scale Height", () => (isText ? undefined : (m.common("height") ?? 0) / base.h * 100), (v) => m.set({ height: base.h * v / 100 }), { unit: "%", min: 1 }),
        num("width_scale", "Scale Width", () => (m.common("width") ?? 0) / base.w * 100, (v) => m.set((e) => ({ width: base.w * v / 100, ...(lock && e.kind !== "text" ? { height: base.h * v / 100 } : {}) })), { unit: "%", min: 1 }),
        h("label", { class: "pp-check" }, h("input", { type: "checkbox", checked: lock, onchange: (ev) => { lock = ev.target.checked; } }), "Lock aspect ratio")),
      sec("position", "Position",
        num("x", "Horizontal position", () => (els.length === 1 ? cx(e0) : m.common("x")), (v) => m.set((e) => ({ x: fromCenter ? v - e.width / 2 : v }))), from(),
        num("y", "Vertical position", () => (els.length === 1 ? cy(e0) : m.common("y")), (v) => m.set((e) => ({ y: fromCenter ? v - e.height / 2 : v }))), from()),
    ];
  };
  const paint = () => {
    root.textContent = "";
    const els = m.els();
    if (!els.length) {  // Format Background
      root.append(iconTabs([["fill", "paint-bucket", "Fill"]]),
        sec("bg", "Fill", radio("Solid fill", true, () => {}),
          colorRow("Color", () => s.doc.background, (c) => s.op("set_background", { color: c || "#ffffff" }), { name: "background" })),
        h("div", { class: "pp-fbtns" }, h("button", { class: "btn pp-reset", type: "button", onclick: () => s.op("set_background", { color: "#ffffff" }) }, "Reset Background")));
      return;
    }
    const kinds = new Set(els.map((e) => e.kind));
    const text = kinds.size === 1 && kinds.has("text"), image = kinds.size === 1 && kinds.has("image"), shape = kinds.size === 1 && kinds.has("shape");
    if (text) {
      const tops = h("div", { class: "pp-ftop" }, ...[["shape", "Shape Options"], ["text", "Text Options"]].map(([id, label]) =>
        h("button", { class: `pp-ftop-tab ${state.top === id ? "on" : ""}`, type: "button", dataset: { top: id }, onclick: () => { state.top = id; paint(); } }, label)));
      root.append(tops);
      if (state.top === "text") {
        root.append(iconTabs([["textfill", "baseline", "Text Fill & Outline"]]),
          sec("textfill", "Text Fill", radio("Solid fill", true, () => {}),
            colorRow("Color", () => m.common("color", ["text"]), (c) => m.set({ color: c || "#000000" }, ["text"]), { name: "color" }), transparency()));
        return;
      }
      root.append(iconTabs([["size", "maximize-2", "Size & Properties"]]), ...sizeSections(els));
      return;
    }
    const tabs = [...(shape ? [["fill", "paint-bucket", "Fill & Line"]] : []), ["size", "maximize-2", "Size & Properties"], ...(image ? [["picture", "image", "Picture"]] : [])];
    if (!tabs.some(([id]) => id === state.sub)) state.sub = tabs[0][0];
    root.append(iconTabs(tabs));
    if (state.sub === "fill") {
      const fill = m.common("fill", ["shape"]), stroke = m.common("stroke", ["shape"]);
      root.append(
        sec("fill", "Fill", radio("No fill", !fill, () => m.set({ fill: null }, ["shape"])), radio("Solid fill", !!fill, () => m.set((e) => ({ fill: e.fill || "#156082" }), ["shape"])),
          colorRow("Color", () => fill, (c) => m.set({ fill: c }, ["shape"]), { name: "fill" }), transparency()),
        sec("line", "Line", radio("No line", !stroke, () => m.set({ stroke: null }, ["shape"])),
          radio("Solid line", !!stroke, () => m.set((e) => ({ stroke: e.stroke || "#000000", stroke_width: e.stroke_width || 1 }), ["shape"])),
          colorRow("Color", () => stroke, (c) => m.set((e) => ({ stroke: c, stroke_width: e.stroke_width || 1 }), ["shape"]), { name: "stroke" }),
          num("stroke_width", "Width", () => m.common("stroke_width", ["shape"]), (v) => m.set({ stroke_width: v }, ["shape"]), { min: 0, max: 1584 })));
    } else if (state.sub === "picture") {
      root.append(sec("transparency", "Picture Transparency", transparency()));
    } else {
      root.append(...sizeSections(els));
    }
  };
  const title = () => {
    const els = m.els();
    if (!els.length) return "Format Background";
    const k = new Set(els.map((e) => e.kind));
    return k.size === 1 && k.has("image") ? "Format Picture" : "Format Shape";
  };
  return { el: root, title, refresh: () => { if (!root.contains(document.activeElement)) paint(); } };
}

// ---------------------------------------------------------------- header search

/** The header's Search box: find a command by name; Enter runs the first match. */
export function commandSearch(app, commands) {
  const input = h("input", { class: "pp-search", placeholder: "Search", spellcheck: "false", "aria-label": "Search (Alt + Q)" });
  const box = h("label", { class: "pp-searchbox" }, h("span", { html: icon("search", 14) }), input);
  const matches = () => {
    const q = input.value.trim().toLowerCase();
    return q ? commands().filter((c) => c.label.toLowerCase().replace(/…/g, "").includes(q) && (!c.enabled || c.enabled())).slice(0, 8) : [];
  };
  input.addEventListener("input", () => {
    const found = matches();
    if (!found.length) { closePopover(); return; }
    const pop = menu(box, [{ header: "Actions" }, ...found.map((c) => ({ label: c.label, icon: c.icon, run: () => { input.value = ""; c.run(); } }))], { cls: "pp-search-pop" });
    pop.querySelectorAll(".menu-item").forEach((b) => b.setAttribute("tabindex", "-1"));
    input.focus();
  });
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { const f = matches()[0]; closePopover(); input.value = ""; input.blur(); f?.run(); }
    if (e.key === "Escape") { closePopover(); input.value = ""; input.blur(); }
  });
  return box;
}
