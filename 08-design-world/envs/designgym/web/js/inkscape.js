// Inkscape's own dialogs and controls, as Inkscape 1.3 / 1.4 draw them (checked against the
// Inkscape Beginners' Guide screenshots): docked dialogs are closable tabs; Fill and Stroke has
// Fill / Stroke paint / Stroke style, a paint-type row, RGB / HSL / HSV / CMYK sliders and the
// RGBA entry; Text and Font applies with its Apply button; Transform moves, scales and rotates
// by Apply; Document Properties holds the page background; Export has the Export button.
// Number entries are spin buttons with - / +. Only what this editor can honour is offered.

import { clamp, fmt, h } from "./util.js";
import { icon } from "./icons.js";
import { ensureFonts, fontCatalog, hasStyle } from "./fonts.js";
import { hexToRgb, hsvToRgb, iconButton, menu, numberField, popover, rgbToHex, rgbToHsv } from "./widgets.js";

/** An Inkscape spin button: a number entry with - and + beside it. */
export function spin(opts) {
  const f = numberField({ digits: 3, ...opts });
  const step = opts.step || 1;
  const bump = (d) => {
    const v = opts.get();
    if (v === null || v === undefined) return;
    opts.set(clamp(v + d * step, opts.min ?? -1e6, opts.max ?? 1e6));
  };
  f.classList.add("ik-spin");
  const minus = iconButton("minus", `Decrease ${opts.title || opts.label || ""}`.trim(), () => bump(-1), { cls: "ik-spin-btn", size: 12 });
  const plus = iconButton("plus", `Increase ${opts.title || opts.label || ""}`.trim(), () => bump(1), { cls: "ik-spin-btn", size: 12 });
  f.append(minus, plus);
  const set0 = f.set;
  f.set = (...a) => { set0(...a); const off = opts.get() === undefined; minus.disabled = plus.disabled = off; };  // greyed with the entry
  return f;
}

// ---------------------------------------------------------------- colour: the Flat color editor

function hsvToHsl(hh, s, v) { const l = v * (1 - s / 2); return [hh, l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l), l]; }
function hslToHsv(hh, s, l) { const v = l + s * Math.min(l, 1 - l); return [hh, v === 0 ? 0 : 2 * (1 - l / v), v]; }
const MODES = ["RGB", "HSL", "HSV", "CMYK"];
let mode = "HSL";  // Inkscape's Flat color opens in HSL

/** RGB / HSL / HSV / CMYK channel sliders with spin values, and the RGBA entry. */
export function flatColor({ get, set, onChange = null }) {
  const root = h("div", { class: "ik-flat" });
  // The colour model is a dropdown beside "Flat color", as in Inkscape 1.4.
  const modeBar = h("button", { class: "ik-mode-dd ik-drop", type: "button", "aria-label": "Color mode" });
  const rows = h("div", { class: "ik-chans" });
  const rgba = h("input", { class: "num-input ik-rgba", spellcheck: "false", "aria-label": "RGBA", dataset: { field: "rgba" } });
  let cur = get() || "#000000";
  const apply = async (hex) => { cur = hex; await set(hex); paint(); onChange?.(hex); };
  const channels = () => {
    const [r, g, b] = hexToRgb(cur);
    const [hh, s, v] = rgbToHsv(r, g, b);
    if (mode === "RGB") return [["R", r, 255], ["G", g, 255], ["B", b, 255]];
    if (mode === "HSV") return [["H", hh, 360], ["S", s * 100, 100], ["V", v * 100, 100]];
    if (mode === "HSL") { const [, s2, l] = hsvToHsl(hh, s, v); return [["H", hh, 360], ["S", s2 * 100, 100], ["L", l * 100, 100]]; }
    const k = 1 - Math.max(r, g, b) / 255;
    const c = k >= 1 ? 0 : (1 - r / 255 - k) / (1 - k), m = k >= 1 ? 0 : (1 - g / 255 - k) / (1 - k), y = k >= 1 ? 0 : (1 - b / 255 - k) / (1 - k);
    return [["C", c * 100, 100], ["M", m * 100, 100], ["Y", y * 100, 100], ["K", k * 100, 100]];
  };
  const fromChannels = (vals) => {
    if (mode === "RGB") return rgbToHex(...vals);
    if (mode === "HSV") return rgbToHex(...hsvToRgb(vals[0] % 360, vals[1] / 100, vals[2] / 100));
    if (mode === "HSL") { const [hh, s, v] = hslToHsv(vals[0] % 360, vals[1] / 100, vals[2] / 100); return rgbToHex(...hsvToRgb(hh, s, v)); }
    const [c, m, y, k] = vals.map((x) => x / 100);
    return rgbToHex(255 * (1 - c) * (1 - k), 255 * (1 - m) * (1 - k), 255 * (1 - y) * (1 - k));
  };
  const paint = () => {
    cur = get() || cur;
    modeBar.innerHTML = `<span class="ik-mode-ic">${icon("palette", 13)}</span><span>${mode}</span>${icon("chevron-down", 10)}`;
    const chans = channels();
    rows.textContent = "";
    chans.forEach(([name, val, max], i) => {
      // Each slider's track shows the colours that channel runs through, the others held.
      const vals0 = chans.map((c) => c[1]);
      // (Hue always shows the full spectrum, as Inkscape draws it.)
      const stops = name === "H" && mode !== "RGB" && mode !== "CMYK"
        ? ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#ff0000"]
        : Array.from({ length: 9 }, (_, k) => { const v = [...vals0]; v[i] = (max * k) / 8; return fromChannels(v); });
      const range = h("input", { type: "range", class: "ik-chan-range", min: 0, max, step: 1, value: String(Math.round(val)), "aria-label": name,
        style: { background: `linear-gradient(to right, ${stops.join(", ")})` } });
      const num = h("input", { class: "num-input ik-chan-num", value: String(Math.round(val)), "aria-label": `${name} value`, dataset: { chan: name } });
      const commit = (v) => {
        const vals = channels().map((c) => c[1]);
        vals[i] = clamp(v, 0, max);
        apply(fromChannels(vals));
      };
      range.addEventListener("change", () => commit(+range.value));
      range.addEventListener("input", () => { num.value = range.value; });
      num.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { num.blur(); return; }
        e.stopPropagation();
        if (e.key === "Enter" || e.key === "Tab") commit(parseFloat(num.value) || 0);
      });
      const step = (d) => commit(Math.round(val) + d);
      rows.append(h("label", { class: "ik-chan" }, h("span", { class: "ik-chan-name" }, `${name}:`), range, num,
        iconButton("minus", `Decrease ${name}`, () => step(-1), { cls: "ik-spin-btn", size: 12 }),
        iconButton("plus", `Increase ${name}`, () => step(1), { cls: "ik-spin-btn", size: 12 })));
    });
    if (document.activeElement !== rgba) rgba.value = `${cur.slice(1, 7)}ff`;
  };
  modeBar.addEventListener("click", () => menu(modeBar, MODES.map((md) => ({ label: md, icon: md === mode ? "check" : null,
    run: () => { mode = md; paint(); } })), { align: "end", cls: "ik-mode-menu" }));
  // The RGBA entry reads eight hex digits (the last two are alpha); six are accepted too.
  rgba.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { rgba.blur(); return; }  // Escape leaves the entry and goes back to the canvas
    e.stopPropagation();
    if (e.key === "Enter" || e.key === "Tab") {
      const t = rgba.value.trim().replace(/^#/, "");
      if (/^[0-9a-f]{6}([0-9a-f]{2})?$/i.test(t)) apply(`#${t.slice(0, 6).toLowerCase()}`);
      else rgba.value = `${cur.slice(1, 7)}ff`;
      if (e.key === "Enter") rgba.select();
    }
  });
  rgba.addEventListener("focus", () => rgba.select());
  root.append(h("div", { class: "ik-flat-head" }, h("span", { class: "ik-flat-title" }, "Flat color"), h("span", { class: "grow" }), modeBar), rows,
    h("div", { class: "ik-rgbarow" }, h("span", { class: "grow" }), h("label", { class: "ik-rgbalabel" }, "RGBA:", rgba)));
  paint();
  root.refresh = paint;
  return root;
}

// ---------------------------------------------------------------- the dock: closable dialog tabs

export function dock(defs, { onChange = null } = {}) {
  const open = [];
  let active = null;
  const bar = h("div", { class: "ik-docktabs", role: "tablist" });
  const host = h("div", { class: "ik-dock-body" });
  const el = h("aside", { class: "ik-dock" }, bar, host);
  const paint = () => {
    bar.textContent = "";
    for (const id of open) {
      const d = defs[id];
      const t = h("div", { class: `ik-docktab ${id === active ? "on" : ""}`, role: "tab", dataset: { tab: id } },
        h("span", { class: "ik-docktab-ic", html: icon(d.icon, 14) }), h("span", { class: "ik-docktab-label" }, d.title),
        iconButton("x", `Close ${d.title}`, () => close(id), { cls: "ik-docktab-x", size: 11 }));
      t.addEventListener("click", (e) => { if (!e.target.closest(".ik-docktab-x")) show(id); });
      bar.append(t);
    }
    el.classList.toggle("empty", !open.length);
  };
  const show = (id) => {
    if (!defs[id]) return;
    if (!open.includes(id)) open.push(id);
    active = id;
    host.textContent = "";
    host.append(defs[id].panel.el);
    defs[id].panel.refresh();
    paint();
    onChange?.();
  };
  const close = (id) => {
    const i = open.indexOf(id);
    if (i < 0) return;
    open.splice(i, 1);
    if (active === id) {
      active = open[Math.min(i, open.length - 1)] || null;
      host.textContent = "";
      if (active) { host.append(defs[active].panel.el); defs[active].panel.refresh(); }
    }
    paint();
    onChange?.();
  };
  return { el, show, close, current: () => active, isOpen: (id) => open.includes(id), refresh: () => { if (active) defs[active].panel.refresh(); } };
}

// ---------------------------------------------------------------- Fill and Stroke

export function fillAndStroke(app, m) {
  const tabs = [["fill", "Fill", "square"], ["stroke", "Stroke paint", "square-dashed"], ["style", "Stroke style", "minus"]];
  let tab = "fill";
  const bar = h("div", { class: "ik-subtabs" });
  const body = h("div", { class: "ik-fs-body" });
  const opacity = spin({ label: "", title: "Opacity (%)", name: "opacity", digits: 1, min: 0, max: 100, cls: "ik-opacity",
    get: () => { const v = m.common("opacity"); return v === null || v === undefined ? v : v * 100; }, set: (v) => m.set({ opacity: v / 100 }) });
  const foot = h("div", { class: "ik-fs-foot" }, h("span", { class: "ik-fs-label" }, "Opacity (%)"), opacity);
  const el = h("div", { class: "panel-body ik-fs" }, bar, body, foot);
  const target = () => {
    const els = m.els();
    if (!els.length) return null;
    if (els.every((e) => e.kind === "text")) return tab === "fill" ? { prop: "color", kinds: ["text"], none: false } : null;
    if (els.every((e) => e.kind === "shape")) return tab === "fill" ? { prop: "fill", kinds: ["shape"], none: true } : { prop: "stroke", kinds: ["shape"], none: true };
    return null;
  };
  let flat = null;
  const paint = () => {
    bar.textContent = "";
    for (const [id, label, ic] of tabs) {
      const b = h("button", { class: `ik-subtab ${id === tab ? "on" : ""}`, type: "button", dataset: { tab: id } }, h("span", { html: icon(ic, 13) }), label);
      b.addEventListener("click", () => { tab = id; paint(); });
      bar.append(b);
    }
    body.textContent = "";
    flat = null;
    const els = m.els();
    foot.hidden = !els.length;
    if (!els.length) { body.append(h("div", { class: "ik-fs-empty" }, "No objects selected")); return; }
    if (tab === "style") {
      if (!els.every((e) => e.kind === "shape")) { body.append(h("div", { class: "ik-fs-empty" }, "No stroke on this object")); return; }
      const w = spin({ label: "Width:", title: "Stroke width", name: "stroke_width", digits: 3, min: 0, max: 1000,
        get: () => m.common("stroke_width", ["shape"]), set: (v) => m.set({ stroke_width: v }, ["shape"]) });
      w.set();
      body.append(h("div", { class: "ik-fs-row" }, w, h("span", { class: "ik-unit" }, "px")));
      return;
    }
    const t = target();
    if (!t) { body.append(h("div", { class: "ik-fs-empty" }, "No paint")); return; }
    const value = m.common(t.prop, t.kinds);
    const isNone = !value || value === "none";
    const paintRow = h("div", { class: "ik-paints" },
      ...(t.none ? [iconButton("x", "No paint", () => m.set(t.prop === "stroke" ? { stroke: null, stroke_width: 0 } : { [t.prop]: "none" }, t.kinds), { cls: `ik-paint ${isNone ? "on" : ""}` })] : []),
      iconButton("square", "Flat color", () => { if (isNone) m.set(t.prop === "stroke" ? { stroke: "#000000", stroke_width: 4 } : { [t.prop]: "#000000" }, t.kinds); }, { cls: `ik-paint ${!isNone ? "on" : ""}` }));
    body.append(paintRow);
    if (isNone) { body.append(h("div", { class: "ik-fs-empty" }, "No paint")); return; }
    flat = flatColor({ get: () => { const v = m.common(t.prop, t.kinds); return v && v !== "none" ? v : "#000000"; },
      set: (c) => m.set(t.prop === "stroke" ? (e) => ({ stroke: c, stroke_width: e.stroke_width || 4 }) : { [t.prop]: c }, t.kinds) });
    body.append(flat);
  };
  const opBar = () => { opacity.set(); const v = m.common("opacity"); opacity.style.setProperty("--pct", `${Math.round((v ?? 1) * 100)}%`); };
  return { el, refresh: () => { if (flat && el.contains(document.activeElement)) { flat.refresh(); opBar(); return; } paint(); opBar(); },
    setTab: (id) => { tab = id; paint(); } };
}

// ---------------------------------------------------------------- Text and Font

export function textAndFont(app, m) {
  let tab = "font";
  let families = [];
  const pending = {};
  const textOnly = () => { const e = m.els(); return e.length && e.every((x) => x.kind === "text"); };
  const bar = h("div", { class: "ik-subtabs" });
  const body = h("div", { class: "ik-tf-body" });
  const applyBtn = h("button", { class: "btn btn-primary ik-apply", type: "button" }, "Apply");
  const el = h("div", { class: "panel-body ik-tf" }, bar, body, h("div", { class: "ik-tf-foot" }, h("span", { class: "grow" }), applyBtn));
  const val = (k) => (k in pending ? pending[k] : m.common(k, ["text"]));
  applyBtn.addEventListener("click", async () => {
    if (!textOnly() || !Object.keys(pending).length) return;
    const p = { ...pending };
    for (const k of Object.keys(pending)) delete pending[k];
    if (p.font_family) ensureFonts([p.font_family]);
    await m.set(p, ["text"]);
    paint();
  });
  const paintFont = () => {
    const fam = val("font_family") || "";
    const search = h("input", { class: "font-search ik-tf-search", placeholder: "Font family", spellcheck: "false", "aria-label": "Font family" });
    const list = h("div", { class: "ik-tf-list font-list", role: "listbox" });
    const seen = new IntersectionObserver((es) => { const f = es.filter((x) => x.isIntersecting).map((x) => x.target.dataset.family); if (f.length) ensureFonts(f); }, { root: list });
    const fill = () => {
      seen.disconnect();
      list.textContent = "";
      const q = search.value.trim().toLowerCase();
      families.filter((f) => f.toLowerCase().includes(q)).forEach((f) => {
        const r = h("button", { class: `font-row ik-tf-row ${f === fam ? "on" : ""}`, type: "button", role: "option", dataset: { family: f },
          onclick: () => { pending.font_family = f; paint(); } }, h("span", { class: "ik-tf-name" }, f), h("span", { class: "ik-tf-sample", style: { fontFamily: `'${f}'` } }, "AaBbCcIiPpQq12369"));
        seen.observe(r);
        list.append(r);
      });
    };
    search.addEventListener("input", fill);
    search.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") list.querySelector(".font-row")?.click(); });
    fill();
    const styles = [["Normal", false, false, "regular"], ["Italic", false, true, "italic"], ["Bold", true, false, "bold"], ["Bold Italic", true, true, "bolditalic"]]
      .filter(([, , , face]) => face === "regular" || !fam || hasStyle(fam, face));
    const b = !!val("bold"), it = !!val("italic");
    const styleList = h("div", { class: "ik-tf-styles" }, h("div", { class: "ik-tf-head" }, h("span", {}, "CSS"), h("span", {}, "Face")),
      ...styles.map(([label, bb, ii]) => h("button", { class: `ik-tf-style ${bb === b && ii === it ? "on" : ""}`, type: "button", dataset: { style: label },
        onclick: () => { pending.bold = bb; pending.italic = ii; paint(); } }, h("span", {}, label), h("span", { class: "ik-tf-face" }, label === "Normal" ? "Regular" : label))));
    const size = numberField({ label: "", title: "Font size", name: "font_size", digits: 2, min: 1, max: 2000, cls: "ik-tf-size",
      get: () => (textOnly() ? val("font_size") : undefined), set: (v) => { pending.font_size = v; paint(); } });
    size.set();
    const preview = h("div", { class: "ik-tf-preview", style: { fontFamily: `'${fam}'`, fontWeight: b ? 700 : 400, fontStyle: it ? "italic" : "normal" } },
      (m.els()[0]?.text || "AaBbCc").split("\n")[0].slice(0, 40));
    ensureFonts([fam]);
    body.append(h("div", { class: "ik-tf-cols" },
      h("fieldset", { class: "ik-tf-set" }, h("legend", {}, "Font family"), search, list),
      h("div", { class: "ik-tf-right" }, h("fieldset", { class: "ik-tf-set" }, h("legend", {}, "Style"), styleList),
        h("label", { class: "ik-tf-sizerow" }, "Font size", size))), preview);
    list.querySelector(".font-row.on")?.scrollIntoView({ block: "nearest" });
  };
  const paintFeatures = () => {
    const row = (label, key, opts) => h("div", { class: "ik-tf-feat" }, h("span", { class: "ik-tf-featlabel" }, label),
      ...opts.map(([v, text]) => {
        const id = `ik-${key}-${v}`;
        const r = h("input", { type: "radio", name: `ik-${key}`, id, checked: !!val(key) === v });
        r.addEventListener("change", () => { pending[key] = v; });
        return h("label", { for: id, class: "ik-tf-radio", dataset: { field: key, value: String(v) } }, r, text);
      }));
    body.append(row("Capitals:", "uppercase", [[false, "Normal"], [true, "All caps"]]), row("Decoration:", "underline", [[false, "None"], [true, "Underline"]]));
  };
  const paintText = () => {
    const area = h("textarea", { class: "ik-tf-text", spellcheck: "false", "aria-label": "Text" });
    area.value = val("text") ?? "";
    area.addEventListener("input", () => { pending.text = area.value; });
    area.addEventListener("keydown", (e) => e.stopPropagation());
    body.append(area);
  };
  const paint = () => {
    bar.textContent = "";
    for (const [id, label] of [["font", "Font"], ["features", "Features"], ["text", "Text"]]) {
      const b = h("button", { class: `ik-subtab ${id === tab ? "on" : ""}`, type: "button", dataset: { tab: id } }, label);
      b.addEventListener("click", () => { tab = id; paint(); });
      bar.append(b);
    }
    body.textContent = "";
    applyBtn.disabled = !textOnly();
    if (!textOnly()) { body.append(h("div", { class: "ik-fs-empty" }, "Select a text object")); return; }
    ({ font: paintFont, features: paintFeatures, text: paintText })[tab]();
  };
  let lastSel = "";
  return {
    el,
    refresh: async () => {
      if (!families.length) families = (await fontCatalog()).map((f) => f.family);
      const sel = m.els().map((e) => e.id).join();
      if (sel !== lastSel) { for (const k of Object.keys(pending)) delete pending[k]; lastSel = sel; }
      if (!el.contains(document.activeElement)) paint();
    },
    setTab: (id) => { tab = id; paint(); },
  };
}

// ---------------------------------------------------------------- Transform

export function transformDialog(app, m) {
  let tab = "move";
  const state = { dx: 0, dy: 0, relative: true, w: null, h: null, prop: true, angle: 0, dir: "cw" };
  const bar = h("div", { class: "ik-subtabs" });
  const body = h("div", { class: "ik-tr-body" });
  const applyBtn = h("button", { class: "btn btn-primary ik-apply", type: "button" }, "Apply");
  const clearBtn = h("button", { class: "btn ik-clear", type: "button" }, "Clear");
  const el = h("div", { class: "panel-body ik-tr" }, bar, body, h("div", { class: "ik-tf-foot" }, clearBtn, h("span", { class: "grow" }), applyBtn));
  const one = () => { const e = m.els(); return e.length ? e : null; };
  applyBtn.addEventListener("click", async () => {
    const els = one();
    if (!els) return;
    if (tab === "move") {
      await m.set((e) => (state.relative ? { x: e.x + state.dx, y: e.y + state.dy } : { x: state.dx, y: state.dy }));
    } else if (tab === "scale") {
      await m.set((e) => {
        let w = state.w ?? e.width, hh = state.h ?? e.height;
        if (state.prop) { if (state.w !== null) hh = e.height * (w / e.width); else if (state.h !== null) w = e.width * (hh / e.height); }
        return e.kind === "text" ? { width: w } : { width: w, height: hh };
      });
    } else {
      const d = state.dir === "cw" ? state.angle : -state.angle;
      await m.set((e) => ({ angle: (e.angle || 0) + d }));  // Inkscape rotates relative to the current angle
    }
    paint();
  });
  clearBtn.addEventListener("click", () => { Object.assign(state, { dx: 0, dy: 0, w: null, h: null, angle: 0 }); paint(); });
  const field = (label, name, key, opts = {}) => {
    const f = spin({ label, title: label.replace(":", ""), name, digits: 3, min: -1e5, max: 1e5, ...opts,
      get: () => (one() ? (state[key] ?? opts.fallback?.() ?? 0) : undefined), set: (v) => { state[key] = v; f.set(); } });
    f.set();
    return f;
  };
  const check = (label, key) => {
    const c = h("input", { type: "checkbox", checked: state[key] });
    c.addEventListener("change", () => {
      state[key] = c.checked;
      if (key === "relative") { const e = m.els()[0]; Object.assign(state, c.checked ? { dx: 0, dy: 0 } : { dx: e?.x ?? 0, dy: e?.y ?? 0 }); paint(); }
    });
    return h("label", { class: "ik-tr-check", dataset: { field: key } }, c, label);
  };
  const paint = () => {
    bar.textContent = "";
    for (const [id, label] of [["move", "Move"], ["scale", "Scale"], ["rotate", "Rotate"]]) {
      const b = h("button", { class: `ik-subtab ${id === tab ? "on" : ""}`, type: "button", dataset: { tab: id } }, label);
      b.addEventListener("click", () => { tab = id; paint(); });
      bar.append(b);
    }
    body.textContent = "";
    applyBtn.disabled = !one();
    if (!one()) { body.append(h("div", { class: "ik-fs-empty" }, "No objects selected")); return; }
    const e0 = m.els()[0];
    if (tab === "move") {
      body.append(h("div", { class: "ik-tr-grid" }, field("Horizontal:", "move_x", "dx"), h("span", { class: "ik-unit" }, "px"),
        field("Vertical:", "move_y", "dy"), h("span", { class: "ik-unit" }, "px")), check("Relative move", "relative"));
    } else if (tab === "scale") {
      body.append(h("div", { class: "ik-tr-grid" }, field("Width:", "scale_w", "w", { min: 1, fallback: () => e0.width }), h("span", { class: "ik-unit" }, "px"),
        field("Height:", "scale_h", "h", { min: 1, fallback: () => e0.height }), h("span", { class: "ik-unit" }, "px")), check("Scale proportionally", "prop"));
    } else {
      const dir = h("div", { class: "segmented ik-tr-dir" }, ...[["ccw", "rotate-ccw", "Counterclockwise"], ["cw", "rotate-cw", "Clockwise"]].map(([v, ic, t]) => {
        const b = iconButton(ic, t, () => { state.dir = v; paint(); }, { cls: state.dir === v ? "on" : "" });
        b.dataset.value = v;
        return b;
      }));
      body.append(h("div", { class: "ik-tr-grid" }, field("Angle:", "rotate_angle", "angle", { min: -360, max: 360 }), h("span", { class: "ik-unit" }, "°")), dir);
    }
  };
  let lastSel = "";
  return {
    el,
    refresh: () => {
      const sel = m.els().map((e) => e.id).join();
      if (sel !== lastSel) { Object.assign(state, { dx: 0, dy: 0, w: null, h: null, angle: 0, relative: true }); lastSel = sel; }
      if (!el.contains(document.activeElement)) paint();
    },
    setTab: (id) => { tab = id; paint(); },
  };
}

// ---------------------------------------------------------------- Document Properties, Export

export function documentProperties(app) {
  const s = app.store;
  const chip = h("button", { class: "color-chip ik-bgchip", type: "button", "aria-label": "Background color", dataset: { field: "page_background" } });
  chip.addEventListener("click", (e) => {
    e.stopPropagation();
    const ed = flatColor({ get: () => s.doc.background, set: (c) => s.op("set_background", { color: c }) });
    popover(chip, h("div", { class: "ik-colorpop" }, h("div", { class: "ik-colorpop-title" }, "Background color"), ed), { cls: "ik-pop" });
    ed.querySelector(".ik-rgba").focus();
  });
  const size = h("div", { class: "ik-dp-size" });
  const el = h("div", { class: "panel-body ik-dp" },
    h("div", { class: "ik-subtabs" }, h("span", { class: "ik-subtab on" }, "Page")),
    h("div", { class: "ik-dp-grid" }, h("span", { class: "ik-dp-label" }, "Format"), size,
      h("span", { class: "ik-dp-label" }, "Background color"), chip));
  return { el, refresh: () => { chip.style.background = s.doc?.background || "#ffffff"; size.textContent = `${s.doc.width} × ${s.doc.height} px`; } };
}

export function exportDialog(app) {
  const s = app.store;
  const name = h("span", { class: "ik-ex-name" });
  const btn = h("button", { class: "btn btn-primary ik-ex-btn", type: "button", dataset: { command: "submit" } }, "Export");
  btn.addEventListener("click", () => app.commands.submit.run());
  const el = h("div", { class: "panel-body ik-exportdlg" },
    h("div", { class: "ik-subtabs" }, h("span", { class: "ik-subtab on" }, "Single Image")),
    h("div", { class: "ik-dp-grid" }, h("span", { class: "ik-dp-label" }, "Export area"), h("span", {}, "Page"),
      h("span", { class: "ik-dp-label" }, "Image size"), h("span", { class: "ik-ex-size" }),
      h("span", { class: "ik-dp-label" }, "Filename"), name, h("span", { class: "ik-dp-label" }, "Format"), h("span", {}, "PNG")),
    h("div", { class: "ik-tf-foot" }, h("span", { class: "grow" }), btn));
  return { el, refresh: () => {
    const t = s.task || {};
    name.textContent = `${(t.title || "drawing").replace(/[^\w\- ]+/g, "").trim().slice(0, 32) || "drawing"}.png`;
    el.querySelector(".ik-ex-size").textContent = `${s.doc.width} × ${s.doc.height} px at 96 dpi`;
    btn.disabled = s.done;
  } };
}

/** The status bar's zoom entry: "Z: 52%" with - / +. */
export function zoomSpin(app) {
  const f = spin({ label: "Z:", title: "Zoom", name: "zoom", digits: 0, min: 1, max: 6400, step: 10, cls: "ik-zoom",
    get: () => Math.round(app.stage.zoom * 100), set: (v) => app.stage.setZoom(v / 100),
    format: (v) => `${fmt(v, 0)}%`, parse: (t) => parseFloat(t) });
  app.on("zoom", () => f.set());
  return f;
}

/** Inkscape's alignment dropdown in the Text tool's controls. */
export function alignDropdown(m) {
  const b = h("button", { class: "ik-align ik-drop", type: "button", "aria-label": "Alignment", title: "Alignment" });
  const opts = [["left", "align-left", "Align left"], ["center", "align-center", "Align center"], ["right", "align-right", "Align right"]];
  b.addEventListener("click", () => menu(b, opts.map(([v, ic, label]) => ({ label, icon: ic, run: () => m.set({ align: v }, ["text"]) })), { cls: "ik-align-menu" }));
  b.set = () => {
    const v = m.common("align", ["text"]) || "left";
    b.innerHTML = `${icon(`align-${v}`, 15)}${icon("chevron-down", 10)}`;
    b.disabled = !m.els().length;
  };
  return b;
}

// ---------------------------------------------------------------- the palette (Inkscape default, as share/palettes/inkscape.gpl)

function hsl(hh, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs(((hh / 60) % 2) - 1)), m0 = l - c / 2;
  const [r, g, b] = hh < 60 ? [c, x, 0] : hh < 120 ? [x, c, 0] : hh < 180 ? [0, c, x] : hh < 240 ? [0, x, c] : hh < 300 ? [x, 0, c] : [c, 0, x];
  return rgbToHex((r + m0) * 255, (g + m0) * 255, (b + m0) * 255);
}
const GREYS = [["#000000", "Black"], ...[90, 80, 70, 60, 50, 40, 30, 20, 10, 7.5, 5, 2.5].map((p) => [rgbToHex(...[0, 0, 0].map(() => 255 * (1 - p / 100))), `${p}% Gray`]),
  ["#ffffff", "White"]];
const BASIC = [["#800000", "Maroon"], ["#ff0000", "Red"], ["#808000", "Olive"], ["#ffff00", "Yellow"], ["#008000", "Green"], ["#00ff00", "Lime"],
  ["#008080", "Teal"], ["#00ffff", "Aqua"], ["#000080", "Navy"], ["#0000ff", "Blue"], ["#800080", "Purple"], ["#ff00ff", "Fuchsia"]];
function inkscapeDefault() {
  const out = [...GREYS, ...BASIC.map(([c, n]) => [c, `${n} (${c.toUpperCase()})`])];
  for (let hh = 0; hh < 360; hh += 24) {
    for (let k = 1; k <= 11; k++) out.push([hsl(hh, 1, k / 12), null]);          // full saturation, dark to light
    for (let k = 1; k <= 9; k++) out.push([hsl(hh, 0.569, k / 10), null]);        // muted
    for (let k = 1; k <= 7; k++) out.push([hsl(hh, 0.13, k / 8), null]);          // greyish
  }
  return out;
}
const TANGO = [["#fce94f", "Butter 1"], ["#edd400", "Butter 2"], ["#c4a000", "Butter 3"], ["#fcaf3e", "Orange 1"], ["#f57900", "Orange 2"], ["#ce5c00", "Orange 3"],
  ["#e9b96e", "Chocolate 1"], ["#c17d11", "Chocolate 2"], ["#8f5902", "Chocolate 3"], ["#8ae234", "Chameleon 1"], ["#73d216", "Chameleon 2"], ["#4e9a06", "Chameleon 3"],
  ["#729fcf", "Sky Blue 1"], ["#3465a4", "Sky Blue 2"], ["#204a87", "Sky Blue 3"], ["#ad7fa8", "Plum 1"], ["#75507b", "Plum 2"], ["#5c3566", "Plum 3"],
  ["#ef2929", "Scarlet Red 1"], ["#cc0000", "Scarlet Red 2"], ["#a40000", "Scarlet Red 3"], ["#eeeeec", "Aluminium 1"], ["#d3d7cf", "Aluminium 2"],
  ["#babdb6", "Aluminium 3"], ["#888a85", "Aluminium 4"], ["#555753", "Aluminium 5"], ["#2e3436", "Aluminium 6"]];
const GRAY = Array.from({ length: 21 }, (_, i) => { const v = Math.round(255 * i / 20); return [rgbToHex(v, v, v), `${100 - i * 5}% Gray`]; });
export const PALETTES = { "Inkscape default": inkscapeDefault(), "Tango icons": TANGO, Gray: GRAY };

/** The palette bar: X (no paint), pinned black / grey / white, two scrolling rows, ^ v and the palette menu. */
export function paletteBar({ onPick, onNone }) {
  let name = "Inkscape default";
  let row = 0;
  const grid = h("div", { class: "ik-pal-grid" });
  const view = h("div", { class: "ik-pal-view" }, grid);
  const tip = (c, n) => (n ? n : c.toUpperCase());
  const sw = (c, n, cls = "") => h("button", { class: `ik-swatch ${cls}`, type: "button", title: `${tip(c, n)} — click: fill, shift+click: stroke`,
    "aria-label": `Swatch ${tip(c, n)}`, style: { background: c }, dataset: { color: c }, onclick: (e) => onPick(c, e.shiftKey) });
  const none = h("button", { class: "ik-swatch ik-none", type: "button", title: "No paint — click: fill, shift+click: stroke", "aria-label": "No paint",
    onclick: (e) => onNone(e.shiftKey) });
  const pinned = h("div", { class: "ik-pal-pinned" }, none, sw("#000000", "Black", "big"), sw("#808080", "50% Gray", "big"), sw("#ffffff", "White", "big"));
  const scroll = (d) => { row = clamp(row + d, 0, Math.max(0, rows() - 2)); grid.style.transform = `translateY(${-row * 13}px)`; };
  const rows = () => Math.ceil(grid.children.length / Math.max(1, Math.floor(view.clientWidth / 13)));
  const paint = () => {
    grid.textContent = "";
    grid.append(...PALETTES[name].map(([c, n]) => sw(c, n)));
    row = 0;
    scroll(0);
  };
  const up = iconButton("chevron-up", "Scroll palette up", () => scroll(-1), { cls: "ik-pal-btn", size: 12 });
  const down = iconButton("chevron-down", "Scroll palette down", () => scroll(1), { cls: "ik-pal-btn", size: 12 });
  const more = iconButton("menu", "Palette", () => menu(more, Object.keys(PALETTES).map((p) => ({ label: p, icon: p === name ? "check" : null,
    run: () => { name = p; paint(); } })), { align: "end", cls: "ik-pal-menu" }), { cls: "ik-pal-btn", size: 14 });
  const el = h("div", { class: "ik-palette", "aria-label": "Palette" }, pinned, view, h("div", { class: "ik-pal-arrows" }, up, down), more);
  paint();
  return el;
}
