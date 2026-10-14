// Controls shared by all three skins. Each one is a plain function returning a
// DOM node plus a `set(value)` method, so panels can refresh them in place
// without losing focus while the user is typing.

import { clamp, fmt, h, shortcut } from "./util.js";
import { icon } from "./icons.js";
import { ensureFonts, fontCatalog } from "./fonts.js";

// ---------------------------------------------------------------- basics

export function iconButton(name, title, onClick, { cls = "", label = null, size = 16 } = {}) {
  const b = h("button", { class: `icon-btn ${cls}`, title, "aria-label": title, type: "button" });
  b.innerHTML = icon(name, size) + (label ? `<span class="lbl">${label}</span>` : "");
  b.addEventListener("click", (e) => { e.stopPropagation(); onClick?.(e); });
  return b;
}

export function commandButton(cmd, opts = {}) {
  const b = iconButton(cmd.icon || "square", `${cmd.label}${cmd.key ? ` (${shortcut(cmd.key)})` : ""}`,
    () => cmd.run(), opts);
  b.dataset.command = cmd.id;
  b.refresh = () => { b.disabled = cmd.enabled ? !cmd.enabled() : false; };
  return b;
}

export function toggle(name, title, getter, onChange, opts = {}) {
  const b = iconButton(name, title, () => onChange(!getter()), opts);
  b.set = () => { b.classList.toggle("on", !!getter()); };
  return b;
}

export function segmented(options, getter, onChange, { cls = "" } = {}) {
  const wrap = h("div", { class: `segmented ${cls}`, role: "radiogroup" });
  const buttons = options.map(([value, iconName, title]) => {
    const b = iconButton(iconName, title, () => onChange(value));
    b.dataset.value = value;
    b.setAttribute("role", "radio");
    wrap.append(b);
    return b;
  });
  wrap.set = () => {
    const v = getter();
    for (const b of buttons) {
      b.classList.toggle("on", b.dataset.value === v);
      b.setAttribute("aria-checked", b.dataset.value === v ? "true" : "false");
    }
  };
  return wrap;
}

// ---------------------------------------------------------------- number field

// A labelled numeric input. The label scrubs: drag it sideways to change the
// value, as in Figma. Enter or blur commits; arrow keys step (Shift = x10).
export function numberField({ label, title, unit = "", digits = 0, step = 1, min = -1e6, max = 1e6,
  get, set, cls = "", iconName = null, width = null, name = null, format = null, parse = null }) {
  // `format(v)` / `parse(text)` let a field show and read units the way its editor does
  // (Figma's "120%" line height, "2px" letter spacing); plain numbers otherwise.
  const input = h("input", { class: "num-input", type: "text", inputmode: "decimal",
    spellcheck: "false", "aria-label": title || label });
  const lab = h("span", { class: "num-label", title: title || label });
  if (iconName) lab.innerHTML = icon(iconName, 12);
  else lab.textContent = label;
  const wrap = h("label", { class: `num-field ${cls}`, dataset: { field: name || title || label || "" } }, lab, input, unit ? h("span", { class: "num-unit" }, unit) : null);
  if (width) wrap.style.width = width;
  let editing = false;
  let dirty = false; // the text differs from what we last displayed

  const commit = () => {
    if (!dirty) return;
    dirty = false;
    let v;
    if (parse) {
      v = parse(input.value.trim());
    } else {
      const raw = input.value.trim().replace(/[^\d.+\-*/() ]/g, "");
      if (!raw) { refresh(true); return; }
      try { v = Function(`"use strict";return (${raw})`)(); } catch { v = parseFloat(raw); }
    }
    if (!Number.isFinite(v)) { refresh(true); return; }
    v = clamp(v, min, max);
    const cur = get();
    if (cur === null || Math.abs(v - cur) > 1e-9) set(v);
    refresh(true);
  };
  const refresh = (force = false) => {
    if (editing && !force) return;
    const v = get();
    dirty = false;
    input.value = v === null || v === undefined ? "" : format ? format(v) : fmt(v, digits);
    input.placeholder = v === null ? "Mixed" : "";
    wrap.classList.toggle("disabled", v === undefined);
    input.disabled = v === undefined;
  };
  input.addEventListener("focus", () => { editing = true; input.select(); });
  input.addEventListener("input", () => { dirty = true; });
  input.addEventListener("blur", () => { editing = false; commit(); });
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { commit(); input.select(); }
    else if (e.key === "Escape") { editing = false; refresh(true); input.blur(); }
    else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      const cur = parse ? (get() ?? 0) : parseFloat(input.value) || 0;
      const d = (e.key === "ArrowUp" ? 1 : -1) * step * (e.shiftKey ? 10 : 1);
      input.value = format ? format(clamp(cur + d, min, max)) : fmt(clamp(cur + d, min, max), digits);
      dirty = true;
      commit();
    }
  });
  lab.addEventListener("pointerdown", (e) => {
    const v0 = get();
    if (v0 === null || v0 === undefined) return;
    e.preventDefault();
    lab.setPointerCapture(e.pointerId);
    const x0 = e.clientX;
    let last = v0;
    const move = (ev) => {
      last = clamp(v0 + Math.round((ev.clientX - x0) / 2) * step * (ev.shiftKey ? 10 : 1), min, max);
      input.value = format ? format(last) : fmt(last, digits);
    };
    const up = () => {
      lab.removeEventListener("pointermove", move);
      lab.removeEventListener("pointerup", up);
      if (last !== v0) set(last);
    };
    lab.addEventListener("pointermove", move);
    lab.addEventListener("pointerup", up);
  });
  wrap.set = refresh;
  wrap.input = input;
  return wrap;
}

export function slider({ get, set, min = 0, max = 100, step = 1, label = "", unit = "", name = null }) {
  const range = h("input", { type: "range", min, max, step, class: "range" });
  const out = h("input", { type: "text", class: "num-input range-num", "aria-label": label });
  const wrap = h("div", { class: "slider-field", dataset: { field: name || label } }, label ? h("span", { class: "slider-label" }, label) : null, range, out,
    unit ? h("span", { class: "num-unit" }, unit) : null);
  range.addEventListener("input", () => { out.value = range.value; });
  range.addEventListener("change", () => set(parseFloat(range.value)));
  out.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { sdirty = false; set(clamp(parseFloat(out.value) || 0, min, max)); }
  });
  let sdirty = false;
  out.addEventListener("input", () => { sdirty = true; });
  out.addEventListener("blur", () => {
    const v = parseFloat(out.value);
    if (sdirty && Number.isFinite(v) && v !== get()) set(clamp(v, min, max));
    sdirty = false;
  });
  wrap.set = () => {
    const v = get();
    if (document.activeElement !== out) out.value = v === null || v === undefined ? "" : fmt(v, step < 1 ? 2 : 0);
    range.value = v ?? min;
    range.disabled = out.disabled = v === undefined;
  };
  return wrap;
}

// ---------------------------------------------------------------- popovers

let openPop = null;

export function popoverOpen() { return !!openPop?.isConnected; }

export function closePopover() {
  if (openPop) { openPop.remove(); openPop = null; }
}

export function popover(anchor, content, { align = "start", cls = "", offset = 6, beside = null } = {}) {
  closePopover();
  const pop = h("div", { class: `popover ${cls}`, role: "dialog" }, content);
  document.body.append(pop);
  const r = anchor.getBoundingClientRect();
  const pr = pop.getBoundingClientRect();
  let left = align === "end" ? r.right - pr.width : r.left;
  let top = r.bottom + offset;
  if (top + pr.height > innerHeight - 8) top = Math.max(8, r.top - pr.height - offset);
  if (beside) {  // a floating panel to the left of the side panel, level with the control (Figma's pickers)
    const side = anchor.closest(beside)?.getBoundingClientRect() || r;
    left = side.left - pr.width - 8;
    top = clamp(r.top - 36, 8, innerHeight - pr.height - 8);
  }
  left = clamp(left, 8, innerWidth - pr.width - 8);
  Object.assign(pop.style, { left: `${left}px`, top: `${top}px` });
  openPop = pop;
  const off = (e) => {
    if (!pop.isConnected) { document.removeEventListener("pointerdown", off, true); return; }
    if (!pop.contains(e.target) && !anchor.contains(e.target)) {
      closePopover();
      document.removeEventListener("pointerdown", off, true);
    }
  };
  setTimeout(() => document.addEventListener("pointerdown", off, true), 0);
  pop.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); closePopover(); anchor.focus?.(); } });
  // Floating panels move by their header, as Figma's colour and font panels do.
  pop.addEventListener("pointerdown", (e) => {
    const grip = e.target.closest(".float-drag");
    if (!grip || e.button !== 0 || e.target.closest("button, input, select")) return;
    e.preventDefault();
    const x0 = e.clientX, y0 = e.clientY, b = pop.getBoundingClientRect();
    const move = (ev) => Object.assign(pop.style, {
      left: `${clamp(b.left + ev.clientX - x0, 0, innerWidth - 60)}px`, top: `${clamp(b.top + ev.clientY - y0, 0, innerHeight - 40)}px` });
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  });
  return pop;
}

/** A floating panel with a title bar (drag to move) and a close button. */
export function floatPanel(title, body, { cls = "", actions = [] } = {}) {
  return h("div", { class: `float-panel ${cls}` },
    h("div", { class: "float-head float-drag" }, h("span", { class: "float-title" }, title), h("span", { class: "grow" }), ...actions,
      iconButton("x", "Close", () => closePopover(), { cls: "float-x", size: 14 })),
    body);
}

// ---------------------------------------------------------------- colour

export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})/i.exec(hex || "");
  if (!m) return [0, 0, 0];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
export function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map((v) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, "0")).join("");
}
export function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  let hue = 0;
  if (d) {
    if (max === r) hue = ((g - b) / d) % 6;
    else if (max === g) hue = (b - r) / d + 2;
    else hue = (r - g) / d + 4;
    hue *= 60;
    if (hue < 0) hue += 360;
  }
  return [hue, max ? d / max : 0, max];
}
export function hsvToRgb(hue, s, v) {
  const c = v * s, x = c * (1 - Math.abs(((hue / 60) % 2) - 1)), m = v - c;
  const [r, g, b] = hue < 60 ? [c, x, 0] : hue < 120 ? [x, c, 0] : hue < 180 ? [0, c, x]
    : hue < 240 ? [0, x, c] : hue < 300 ? [x, 0, c] : [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

const PALETTE = ["#000000", "#545454", "#737373", "#a6a6a6", "#d9d9d9", "#ffffff",
  "#ff3131", "#ff5757", "#ff66c4", "#cb6ce6", "#8c52ff", "#5e17eb",
  "#0097b2", "#0cc0df", "#5ce1e6", "#38b6ff", "#5170ff", "#004aad",
  "#00bf63", "#7ed957", "#c1ff72", "#ffde59", "#ffbd59", "#ff914d"];

export function colorPicker(initial, onPick, docColors = []) {
  let [hue, sat, val] = rgbToHsv(...hexToRgb(initial));
  const sv = h("div", { class: "cp-sv" }, h("div", { class: "cp-sv-knob" }));
  const hueBar = h("div", { class: "cp-hue" }, h("div", { class: "cp-hue-knob" }));
  const hex = h("input", { class: "cp-hex num-input", value: (initial || "#000000").slice(0, 7), spellcheck: "false", "aria-label": "Hex colour" });
  const swatch = (c) => h("button", { class: "cp-swatch", style: { background: c }, title: c, type: "button",
    onclick: () => { [hue, sat, val] = rgbToHsv(...hexToRgb(c)); paint(); onPick(c); } });
  const root = h("div", { class: "color-picker" }, sv, hueBar,
    h("div", { class: "cp-row" }, h("span", { class: "cp-hash" }, "Hex"), hex),
    docColors.length ? h("div", { class: "cp-section" }, "Document colours") : null,
    docColors.length ? h("div", { class: "cp-swatches" }, docColors.slice(0, 12).map(swatch)) : null,
    h("div", { class: "cp-section" }, "Default colours"),
    h("div", { class: "cp-swatches" }, PALETTE.map(swatch)));
  const current = () => rgbToHex(...hsvToRgb(hue, sat, val));
  const paint = () => {
    sv.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, hsl(${hue}, 100%, 50%))`;
    sv.firstChild.style.left = `${sat * 100}%`;
    sv.firstChild.style.top = `${(1 - val) * 100}%`;
    sv.firstChild.style.background = current();
    hueBar.firstChild.style.left = `${(hue / 360) * 100}%`;
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
    if (e.key !== "Escape") e.stopPropagation();  // Escape bubbles to the popover, which closes it
    if (e.key === "Enter") {
      const v = hex.value.trim().replace(/^#?/, "#");
      if (/^#[0-9a-f]{6}$/i.test(v)) { [hue, sat, val] = rgbToHsv(...hexToRgb(v)); paint(); onPick(v.toLowerCase()); }
    }
  });
  paint();
  return root;
}

export function colorField({ get, set, label = "", docColors = () => [], cls = "", showHex = true, name = null, open = null }) {
  const chip = h("button", { class: "color-chip", type: "button", "aria-label": `${label || "Colour"} swatch` });
  const input = h("input", { class: "num-input hex-input", spellcheck: "false", "aria-label": label || "Colour hex" });
  const wrap = h("div", { class: `color-field ${cls}`, dataset: { field: name || label } }, label ? h("span", { class: "field-label" }, label) : null, chip, showHex ? input : null);
  chip.addEventListener("click", (e) => {
    e.stopPropagation();
    if (open) open(chip, get, set, docColors());  // a layout's own picker (Figma's floating panel)
    else popover(chip, colorPicker(get() || "#000000", (c) => set(c), docColors()), { cls: "pop-color" });
  });
  // Enter or blur (Tab, clicking away) commits a valid hex, like the number fields; Escape reverts.
  // Only typing commits: leaving a field whose value was already applied changes nothing (the selection
  // may have moved on, e.g. from a few characters of a text back to the whole box).
  let cancelled = false;
  let typed = false;
  input.addEventListener("input", () => { typed = true; });
  const commit = () => {
    const v = input.value.trim().replace(/^#?/, "#");
    if (typed && /^#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(v) && v.toLowerCase() !== (get() || "").toLowerCase()) set(v.toLowerCase());
    typed = false;
  };
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { commit(); input.select(); }
    else if (e.key === "Escape") { cancelled = true; input.blur(); }
  });
  input.addEventListener("focus", () => { cancelled = false; input.select(); });
  input.addEventListener("blur", () => { if (!cancelled) commit(); cancelled = false; wrap.set(); });
  wrap.set = () => {
    const v = get();
    chip.style.background = v ? v : "repeating-conic-gradient(#ccc 0 25%, #fff 0 50%) 0 0/8px 8px";
    chip.classList.toggle("none", !v);
    if (document.activeElement !== input) input.value = v ? v.replace("#", "").toUpperCase().slice(0, 6) : "None";
    wrap.classList.toggle("disabled", v === undefined);
  };
  wrap.chip = chip;
  return wrap;
}

// ---------------------------------------------------------------- font picker

export function fontPicker({ get, set, docFonts = () => [], cls = "", open = null, preview = true }) {
  const btn = h("button", { class: `font-btn ${cls}`, type: "button", "aria-label": "Font family" });
  const name = h("span", { class: "font-name" });
  btn.append(name);
  btn.insertAdjacentHTML("beforeend", icon("chevron-down", 14));
  btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (open) { open(btn, get, set, docFonts()); return; }
    const families = (await fontCatalog()).map((f) => f.family);
    const search = h("input", { class: "font-search", placeholder: "Search fonts", spellcheck: "false", "aria-label": "Search fonts" });
    const list = h("div", { class: "font-list", role: "listbox" });
    const docs = [...new Set(docFonts())];
    ensureFonts(docs);
    const row = (fam, preview) => {
      const r = h("button", { class: `font-row ${fam === get() ? "on" : ""}`, type: "button", role: "option", dataset: { family: fam },
        onclick: () => { set(fam); ensureFonts([fam]); closePopover(); } }, fam);
      if (preview) r.style.fontFamily = `'${fam}'`;
      return r;
    };
    const paint = () => {
      const q = search.value.trim().toLowerCase();
      list.textContent = "";
      const docMatch = docs.filter((f) => f.toLowerCase().includes(q));
      if (docMatch.length) {
        list.append(h("div", { class: "font-group" }, "Fonts in this design"));
        docMatch.forEach((f) => list.append(row(f, true)));
      }
      list.append(h("div", { class: "font-group" }, "All fonts"));
      families.filter((f) => f.toLowerCase().includes(q)).forEach((f) => list.append(row(f, false)));
    };
    search.addEventListener("input", paint);
    search.addEventListener("keydown", (ev) => {
      if (ev.key !== "Escape") ev.stopPropagation();  // Escape bubbles to the popover, which closes it
      if (ev.key === "Enter") {
        const first = list.querySelector(".font-row");
        first?.click();
      }
    });
    paint();
    popover(btn, h("div", { class: "font-pop" }, search, list), { cls: "pop-font" });
    search.focus();
    list.querySelector(".font-row.on")?.scrollIntoView({ block: "center" });
  });
  btn.set = () => {
    const v = get();
    name.textContent = v === null ? "Mixed" : v || "—";
    if (v && preview) { name.style.fontFamily = `'${v}'`; ensureFonts([v]); }
    btn.disabled = v === undefined;
  };
  return btn;
}

// ---------------------------------------------------------------- menus

// Menus only list what works: permanently disabled entries and commands hidden in this
// session are dropped, along with separators and headers they leave dangling.
export function usefulItems(items) {
  const kept = items.filter((it) => it === "-" || it.header || !(it.enabled === false || it.cmd?.hidden?.()));
  const out = [];
  for (let i = 0; i < kept.length; i++) {
    const it = kept[i];
    if (it === "-" && (!out.length || out[out.length - 1] === "-")) continue;
    if (it.header) {  // keep a header only if an item follows before the next break
      const next = kept.slice(i + 1).find((x) => x === "-" || x.header || x);
      if (!next || next === "-" || next.header) continue;
    }
    out.push(it);
  }
  while (out.length && (out[out.length - 1] === "-" || out[out.length - 1].header)) out.pop();
  return out;
}

function menuList(items, cls, host, depth = 0) {
  const list = h("div", { class: `menu ${cls}`, role: "menu" });
  // Submenus nest (Format ▸ Text ▸ Capitalization ▸): opening one closes only those deeper than this list.
  const closeSub = () => {
    host().querySelectorAll(".menu-sub").forEach((x) => { if (+x.dataset.depth > depth) x.remove(); });
    list.querySelectorAll(".has-sub.open").forEach((x) => x.classList.remove("open"));
  };
  for (const it of usefulItems(items)) {
    if (it === "-") { list.append(h("div", { class: "menu-sep" })); continue; }
    if (it.header) { list.append(h("div", { class: "menu-header" }, it.header)); continue; }
    if (it.items) {  // a submenu (File ▸, Edit ▸ ...): opens beside its row on hover or click
      const kids = usefulItems(it.items);
      if (!kids.length) continue;
      const b = h("button", { class: "menu-item has-sub", role: "menuitem", type: "button", "aria-haspopup": "menu", dataset: { submenu: it.label } });
      b.innerHTML = `<span class="mi-icon"></span><span class="mi-label">${it.label}</span><span class="mi-key">${icon("chevron-right", 12)}</span>`;
      const open = () => {
        if (b.classList.contains("open")) return;
        closeSub();
        b.classList.add("open");
        const sub = h("div", { class: "popover pop-menu menu-sub", dataset: { depth: String(depth + 1) } }, menuList(kids, "", host, depth + 1));
        host().append(sub);
        const r = b.getBoundingClientRect(), sr = sub.getBoundingClientRect();
        const left = r.right + sr.width + 4 > innerWidth ? r.left - sr.width - 4 : r.right + 4;
        Object.assign(sub.style, { left: `${left}px`, top: `${clamp(r.top - 6, 8, innerHeight - sr.height - 8)}px` });
      };
      b.addEventListener("mouseenter", open);
      b.addEventListener("click", (e) => { e.stopPropagation(); open(); });
      list.append(b);
      continue;
    }
    const cmd = it.cmd;
    const enabled = cmd ? (cmd.enabled ? cmd.enabled() : true) : it.enabled !== false;
    const b = h("button", { class: "menu-item", role: "menuitem", type: "button", disabled: !enabled,
      dataset: cmd ? { command: cmd.id } : {} });
    b.innerHTML = `<span class="mi-icon">${cmd?.icon || it.icon ? icon(cmd?.icon || it.icon, 14) : ""}</span>` +
      `<span class="mi-label">${it.label || cmd.label}</span>` +
      `<span class="mi-key">${shortcut(it.key || cmd?.key || "")}</span>`;
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      closePopover();
      (it.run || cmd.run)();
    });
    b.addEventListener("mouseenter", closeSub);
    list.append(b);
  }
  return list;
}

export function menu(anchor, items, { align = "start", cls = "" } = {}) {
  let pop = null;
  const list = menuList(items, cls, () => pop);
  pop = popover(anchor, list, { align, cls: "pop-menu", offset: 2 });
  return pop;
}

export function contextMenuAt(x, y, items) {
  const ghost = h("div", { style: { position: "fixed", left: `${x}px`, top: `${y}px`, width: "1px", height: "1px" } });
  document.body.append(ghost);
  const pop = menu(ghost, items, { cls: "context" });
  setTimeout(() => ghost.remove(), 0);
  return pop;
}

// ---------------------------------------------------------------- modal, toast

export function modal(title, body, { actions = [], cls = "", onClose = null } = {}) {
  const close = () => { back.remove(); onClose?.(); };
  const head = h("div", { class: "modal-head" }, h("div", { class: "modal-title" }, title),
    iconButton("x", "Close", close, { cls: "modal-x" }));
  const foot = actions.length ? h("div", { class: "modal-foot" }, actions.map((a) =>
    h("button", { class: `btn ${a.primary ? "btn-primary" : ""}`, type: "button",
      onclick: async () => { const keep = await a.run?.(); if (keep !== true) close(); } }, a.label))) : null;
  const box = h("div", { class: `modal ${cls}`, role: "dialog", "aria-modal": "true" }, head,
    h("div", { class: "modal-body" }, body), foot);
  const back = h("div", { class: "modal-back", onclick: (e) => { if (e.target === back) close(); } }, box);
  back.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } });
  document.body.append(back);
  return { close, box };
}

let toastTimer = null;
export function toast(msg, kind = "") {
  let t = document.querySelector(".toast");
  if (!t) { t = h("div", { class: "toast", role: "status" }); document.body.append(t); }
  t.textContent = msg;
  t.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}
