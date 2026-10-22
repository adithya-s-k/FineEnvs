// Panels: layers, assets, the design inspector, text presets.
// Skins place and label them differently; the behaviour is identical.

import { openColorPanel, openFontPanel, typography } from "./figma.js";
import { openColorDialog } from "./photoshop.js";
import { assetNames, cssColor, elementLabel, fmt, h } from "./util.js";
import { icon } from "./icons.js";
import { ensureFonts, fontCatalog, hasStyle } from "./fonts.js";
import {
  colorField, commandButton, fontPicker, iconButton, numberField, segmented, slider, toggle,
} from "./widgets.js";

// ---------------------------------------------------------------- selection values

export function selectionModel(app) {
  const s = app.store;
  const els = () => s.selected();
  const common = (prop, kinds = null) => {
    const list = els().filter((e) => !kinds || kinds.includes(e.kind));
    if (!list.length) return undefined;
    const range = app.stage?.textRange;
    if (range && list.length === 1 && list[0].id === range.id && ["color", "bold", "italic"].includes(prop)) {
      return app.stage.rangeValue(prop);
    }
    const v = list[0][prop];
    return list.every((e) => e[prop] === v) ? v : null;
  };
  const set = (props, kinds = null) => {
    const list = els().filter((e) => !e.locked && (!kinds || kinds.includes(e.kind)));
    // Characters chosen inside a text box take colour / bold / italic themselves (see runs.js).
    const range = app.stage?.textRange;
    if (range && list.length === 1 && list[0].id === range.id) {
      return app.stage.applyRangeStyle(list[0], typeof props === "function" ? props(list[0]) : props);
    }
    return s.updateMany(list.map((e) => {
      const p = typeof props === "function" ? props(e) : props;
      // A whole-element text style replaces any per-run value for that style.
      if (e.kind === "text" && e.runs && ("color" in p || "bold" in p || "italic" in p)) {
        p.runs = e.runs.map((r) => {
          const out = { text: r.text };
          for (const k of ["color", "bold", "italic"]) if (!(k in p) && r[k] !== undefined && r[k] !== null) out[k] = r[k];
          return out;
        });
      }
      return [e.id, p];
    }));
  };
  return { els, common, set };
}

function docColors(app) {
  const out = [];
  const doc = app.store.doc;
  if (!doc) return out;
  out.push(doc.background);
  for (const e of doc.elements) {
    for (const c of [e.color, e.fill, e.stroke, ...(e.runs || []).map((r) => r.color)]) if (c) out.push(c.slice(0, 7));
  }
  return [...new Set(out)];
}

// ---------------------------------------------------------------- layers

export function layersPanel(app, { thumbs = true, background = false, compact = false, frame = null } = {}) {
  const s = app.store;
  const list = h("div", { class: `layers ${compact ? "compact" : ""}`, role: "tree" });
  const root = h("div", { class: "panel-body layers-panel" }, list);
  let drag = null;

  const thumb = (el) => {
    if (!thumbs) return h("span", { class: "layer-icon", html: icon(el.kind === "text" ? "type" : el.kind === "shape" ? (el.shape === "ellipse" ? "circle" : "square") : "image", 14) });
    if (el.kind === "image") {
      const a = s.assetById.get(el.asset_id);
      return h("span", { class: "layer-thumb checker" }, h("img", { src: a?.url, alt: "", draggable: "false" }));
    }
    if (el.kind === "shape") {
      return h("span", { class: "layer-thumb" }, h("span", { class: "shape-swatch", style: {
        background: cssColor(el.fill), borderRadius: el.shape === "ellipse" ? "50%" : "2px" } }));
    }
    return h("span", { class: "layer-thumb text-thumb" }, "T");
  };

  const refresh = () => {
    if (!s.doc) return;
    list.textContent = "";
    const els = [...s.doc.elements].reverse();
    if (frame) {  // Figma: every layer sits inside the frame, which is itself selectable
      const on = !!app.frameSelected;
      const fr = h("div", { class: `layer-row fg-frame-row ${on ? "selected" : ""}`, role: "treeitem", "aria-selected": on ? "true" : "false",
        "aria-expanded": "true", dataset: { id: "frame" } },
      h("span", { class: "layer-caret", html: icon("chevron-down", 10) }), h("span", { class: "layer-icon", html: icon("hash", 14) }),
      h("span", { class: "layer-name" }, frame));
      fr.addEventListener("pointerdown", (e) => { if (e.button === 0) app.selectFrame(); });
      list.append(fr);
    }
    if (!els.length && !frame) list.append(h("div", { class: "empty" }, "No layers yet. Add an asset or some text."));
    for (const el of els) {
      const selected = s.selection.includes(el.id);
      const nameEl = h("span", { class: "layer-name", title: elementLabel(el) }, elementLabel(el));
      const row = h("div", {
        class: `layer-row ${frame ? "nested" : ""} ${selected ? "selected" : ""} ${el.visible === false ? "hidden" : ""} ${el.locked ? "locked" : ""}`,
        role: "treeitem", "aria-selected": selected ? "true" : "false", dataset: { id: el.id },
      },
      iconButton(el.visible === false ? "eye-off" : "eye", el.visible === false ? "Show layer" : "Hide layer",
        () => s.update(el.id, { visible: el.visible === false }), { cls: "layer-eye" }),
      thumb(el), nameEl,
      iconButton(el.locked ? "lock" : "lock-open", el.locked ? "Unlock layer" : "Lock layer",
        () => s.update(el.id, { locked: !el.locked }), { cls: "layer-lock" }));
      row.addEventListener("pointerdown", (e) => {
        if (e.target.closest("button") || e.button !== 0) return;
        if (e.shiftKey || e.metaKey || e.ctrlKey) s.select(el.id, { toggle: true });
        else if (!selected) s.select(el.id);
        drag = { id: el.id, y0: e.clientY, over: null, moved: false };
        row.setPointerCapture(e.pointerId);
      });
      row.addEventListener("pointermove", (e) => {
        if (!drag || drag.id !== el.id) return;
        if (Math.abs(e.clientY - drag.y0) > 4) drag.moved = true;
        if (!drag.moved) return;
        const target = document.elementFromPoint(e.clientX, e.clientY)?.closest(".layer-row");
        list.querySelectorAll(".drop-above,.drop-below").forEach((n) => n.classList.remove("drop-above", "drop-below"));
        if (target && target !== row) {
          const r = target.getBoundingClientRect();
          const above = e.clientY < r.top + r.height / 2;
          target.classList.add(above ? "drop-above" : "drop-below");
          drag.over = { id: target.dataset.id, above };
        }
      });
      row.addEventListener("pointerup", async () => {
        const d = drag;
        drag = null;
        list.querySelectorAll(".drop-above,.drop-below").forEach((n) => n.classList.remove("drop-above", "drop-below"));
        if (!d?.moved || !d.over) return;
        // Rows run top (front) to bottom (back); convert to a stacking index.
        const from = s.index(d.id);
        let to = s.index(d.over.id) + (d.over.above ? 1 : 0);
        if (from < to) to -= 1;
        if (to !== from) await s.op("reorder_element", { element_id: d.id, to: String(to) });
      });
      nameEl.addEventListener("dblclick", () => {
        const input = h("input", { class: "layer-rename", value: el.name || elementLabel(el) });
        nameEl.replaceWith(input);
        input.focus();
        input.select();
        const done = (save) => {
          if (save && input.value.trim() && input.value !== elementLabel(el)) s.update(el.id, { name: input.value.trim() });
          else refresh();
        };
        input.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); });
        input.addEventListener("blur", () => done(true));
      });
      row.addEventListener("dblclick", (e) => {
        if (el.kind === "text" && !e.target.closest(".layer-name")) app.stage.startEdit(el.id);
      });
      list.append(row);
    }
    if (background) {
      list.append(h("div", { class: "layer-row background-row", dataset: { id: "background" } },
        h("span", { class: "layer-eye-placeholder" }),
        h("span", { class: "layer-thumb" }, h("span", { class: "shape-swatch", style: { background: cssColor(s.doc.background) } })),
        h("span", { class: "layer-name bg-name" }, "Background"),
        h("span", { class: "layer-lock fixed", html: icon("lock", 13) })));
    }
  };
  return { el: root, refresh };
}

// ---------------------------------------------------------------- assets

export function assetsPanel(app, { cols = 2, heading = null, search = false, filter = null, empty = "", top = null,
  placeholder = "Search assets" } = {}) {
  const s = app.store;
  const grid = h("div", { class: "asset-grid", style: { gridTemplateColumns: `repeat(${cols}, 1fr)` } });
  const q = search ? h("input", { class: "panel-search", placeholder, spellcheck: "false", "aria-label": placeholder }) : null;
  const hint = h("div", { class: "panel-hint" }, "Click to add, or drag onto the canvas.");
  const root = h("div", { class: "panel-body assets-panel" }, heading ? h("div", { class: "panel-heading" }, heading) : null, q, top, grid, hint);
  const shown = () => (filter ? s.assets.filter(filter) : s.assets);

  const startDrag = (asset, e, tile) => {
    const x0 = e.clientX, y0 = e.clientY;
    let ghost = null;
    tile.setPointerCapture(e.pointerId);
    const move = (ev) => {
      if (!ghost && Math.hypot(ev.clientX - x0, ev.clientY - y0) > 5) {
        ghost = h("div", { class: "drag-ghost" }, h("img", { src: asset.url, alt: "" }));
        document.body.append(ghost);
        s.busy(true);
      }
      if (ghost) Object.assign(ghost.style, { left: `${ev.clientX}px`, top: `${ev.clientY}px` });
    };
    const up = async (ev) => {
      tile.removeEventListener("pointermove", move);
      tile.removeEventListener("pointerup", up);
      if (!ghost) { await app.addAsset(asset); return; }
      ghost.remove();
      s.busy(false);
      const p = app.stage.dropPoint(ev.clientX, ev.clientY);
      if (p) await app.addAsset(asset, p);
    };
    tile.addEventListener("pointermove", move);
    tile.addEventListener("pointerup", up);
  };

  const refresh = () => {
    const list = shown();
    const key = `${s.episodeId}:${list.map((a) => a.id).join(",")}`;
    hint.textContent = list.length ? "Click to add, or drag onto the canvas." : empty;
    if (grid.dataset.key === key) {
      const used = new Set((s.doc?.elements || []).filter((e) => e.kind === "image").map((e) => e.asset_id));
      for (const t of grid.children) t.classList.toggle("used", used.has(t.dataset.asset));
      return;
    }
    grid.dataset.key = key;
    grid.textContent = "";
    for (const a of list) {
      const label = assetNames.get(a.id) || a.id;  // a file's name, on a lived-in computer
      const tile = h("div", { class: "asset-tile", dataset: { asset: a.id }, title: `${label} · ${a.kind}`, tabindex: "0", role: "button",
        "aria-label": `Asset ${label} (${a.kind})` },
      h("div", { class: "asset-img checker" }, h("img", { src: a.url, alt: label, draggable: "false", loading: "lazy" })),
      h("div", { class: "asset-meta" }, h("span", { class: "asset-id" }, label), h("span", { class: "asset-kind" }, a.kind)));
      tile.addEventListener("pointerdown", (e) => { if (e.button === 0) startDrag(a, e, tile); });
      tile.addEventListener("keydown", (e) => { if (e.key === "Enter") app.addAsset(a); });
      grid.append(tile);
    }
    if (q) q.oninput = () => {
      const t = q.value.trim().toLowerCase();
      for (const tile of grid.children) tile.style.display = !t || tile.title.toLowerCase().includes(t) ? "" : "none";
    };
    refresh();
  };
  return { el: root, refresh };
}

// ---------------------------------------------------------------- inspector

export function inspector(app, { variant = "figma" } = {}) {
  const s = app.store;
  const m = selectionModel(app);
  const root = h("div", { class: `panel-body inspector inspector-${variant}` });
  const controls = [];
  const reg = (c) => { controls.push(c); return c; };
  const section = (title, ...rows) => h("section", { class: "insp-section" },
    title ? h("div", { class: "insp-title" }, title) : null, ...rows);
  const row = (...kids) => h("div", { class: "insp-row" }, ...kids);
  const C = app.commands;
  const cb = (id) => reg(commandButton(C[id]));

  let aspectLock = false;
  const geom = (prop, label, opts = {}) => reg(numberField({
    label, title: opts.title || prop, digits: opts.digits ?? 2, unit: opts.unit || "",
    min: opts.min ?? -1e5, max: opts.max ?? 1e5, step: opts.step ?? 1, cls: opts.cls || "",
    iconName: opts.iconName || null, name: prop,
    // Canva shows the unit inside the field ("721.5 px", "0°") and reads it back.
    ...(variant === "canva" ? { format: (v) => `${fmt(v, 1)}${prop === "angle" ? "°" : " px"}`, parse: (t) => parseFloat(t.replace(",", ".")) } : {}),
    get: () => {
      const els = m.els();
      if (!els.length) return undefined;
      if (opts.kinds && !els.every((e) => opts.kinds.includes(e.kind))) return undefined;
      const v = m.common(prop, opts.kinds);
      return opts.scale && v !== null && v !== undefined ? v * opts.scale : v;
    },
    set: (v) => {
      const val = opts.scale ? v / opts.scale : v;
      if ((prop === "width" || prop === "height") && aspectLock) {
        return m.set((e) => {
          const r = e.width / e.height;
          return prop === "width" ? { width: val, height: val / r } : { height: val, width: val * r };
        });
      }
      return m.set({ [prop]: val }, opts.kinds);
    },
  }));

  // alignment
  const alignRow = row(cb("alignLeft"), cb("alignCenter"), cb("alignRight"), h("span", { class: "sep" }),
    cb("alignTop"), cb("alignMiddle"), cb("alignBottom"), h("span", { class: "sep" }), cb("distributeH"), cb("distributeV"));
  alignRow.classList.add("align-row");

  const lockBtn = reg(toggle("lock", "Constrain proportions", () => aspectLock, (v) => { aspectLock = v; refresh(); }, { cls: "aspect-lock" }));
  // Section names follow each editor's own vocabulary.
  const T = {
    photoshop: { pos: "Transform", text: "Character", page: "Document" },
    canva: { pos: "Position", text: "Text", page: "Page" },
    figma: { pos: "Position", text: "Typography", page: "Page" },
    slides: { pos: "Size & Rotation · Position", text: "Text", page: "Slide background" },
    powerpoint: { pos: "Size & Position", text: "Text Options", page: "Background" },
    inkscape: { pos: "Transform", text: "Text and Font", page: "Document Properties" },
  }[variant] || { pos: "Position", text: "Text", page: "Page" };
  const ui3 = variant === "figma";  // Figma UI3: Position (x, y, rotation) and Layout (w, h) are separate
  const long = variant === "canva";  // Canva spells the labels out above each field
  const whRow = row(geom("width", long ? "Width" : "W", { min: 1 }), geom("height", long ? "Height" : "H", { min: 1, kinds: ["image", "shape"] }), lockBtn);
  const rotRow = row(geom("angle", long ? "Rotate" : "", { iconName: long ? null : "rotate-cw", title: "Rotation", unit: "°", digits: 2, min: -360, max: 360 }),
    cb("flipH"), cb("flipV"));
  const sub = (t) => (ui3 ? h("div", { class: "insp-sub" }, t) : null);  // Figma's small labels above field groups
  const posSection = long  // Canva's Advanced block: Width / Height / Ratio, then X / Y / Rotate
    ? section(T.pos, whRow, row(geom("x", "X"), geom("y", "Y"), rotRow.firstChild))
    : section(T.pos,
      sub("Alignment"), variant === "photoshop" || variant === "inkscape" ? null : alignRow,
      sub("Position"), row(geom("x", "X"), geom("y", "Y")),
      sub("Rotation"), ui3 ? rotRow : whRow, ui3 ? null : rotRow);
  const layoutSection = ui3 ? section("Layout", sub("Dimensions"), whRow) : null;

  const opGet = () => { const v = m.common("opacity"); return v === undefined || v === null ? v : Math.round(v * 100); };
  const opacity = ui3  // Figma: a typed "100%" field (1-0 on the keyboard also set it)
    ? reg(numberField({ label: "", title: "Opacity", name: "opacity", iconName: "blend", min: 0, max: 100, cls: "fg-opacity",
      get: opGet, set: (v) => m.set({ opacity: v / 100 }), format: (v) => `${fmt(v, 0)}%`, parse: (t) => parseFloat(t) }))
    : reg(slider({ label: "Opacity", name: "opacity", unit: "%", min: 0, max: 100, get: opGet, set: (v) => m.set({ opacity: v / 100 }) }));
  const radius = geom("radius", "", { iconName: "square-dashed", title: "Corner radius", min: 0, kinds: ["shape"] });
  const appearance = ui3
    ? section("Appearance", h("div", { class: "insp-subrow" }, sub("Opacity"), sub("Corner radius")), row(opacity, radius))
    : section("Appearance", row(opacity), row(radius));
  const kindHead = ui3 ? h("div", { class: "insp-kind" }, h("span", { class: "insp-kind-name" }, "")) : null;
  const extras = [];

  const picker = ui3 ? openColorPanel : variant === "photoshop"
    ? (chip, get, set) => openColorDialog(chip, { title: "Color Picker", get, set, addSwatch: (c) => app.addSwatch?.(c) }) : null;
  const fill = reg(colorField({ label: ui3 ? "" : "Fill", name: "fill", docColors: () => docColors(app), open: picker,
    get: () => { const els = m.els(); if (!els.length || !els.every((e) => e.kind === "shape")) return undefined; return m.common("fill", ["shape"]); },
    set: (c) => m.set({ fill: c }, ["shape"]) }));
  const stroke = reg(colorField({ label: ui3 ? "" : "Stroke", name: "stroke", docColors: () => docColors(app), open: picker,
    get: () => { const els = m.els(); if (!els.length || !els.every((e) => e.kind === "shape")) return undefined; return m.common("stroke", ["shape"]); },
    set: (c) => m.set((e) => ({ stroke: c, stroke_width: e.stroke_width || 4 }), ["shape"]) }));
  const strokeW = geom("stroke_width", "", { iconName: "minus", title: "Stroke width", min: 0, kinds: ["shape"] });
  const fillSection = ui3  // Figma: Fill and Stroke are their own sections
    ? h("div", {}, section("Fill", row(fill)), section("Stroke", row(stroke, strokeW)))
    : section("Fill", row(fill), row(stroke, strokeW));

  // typography
  const textOnly = () => { const els = m.els(); return els.length && els.every((e) => e.kind === "text"); };
  const tget = (p) => () => (textOnly() ? m.common(p, ["text"]) : undefined);
  const font = reg(fontPicker({ get: tget("font_family"), set: (f) => m.set({ font_family: f }, ["text"]),
    open: ui3 ? openFontPanel : null, preview: !ui3,
    docFonts: () => [...(s.task?.fonts_used || []), ...s.doc.elements.filter((e) => e.kind === "text").map((e) => e.font_family)] }));
  const size = geom("font_size", "", { iconName: "type", title: "Font size", min: 1, max: 2000, digits: 2, kinds: ["text"], unit: "px" });
  const styleBtn = (prop, iconName, title, styleName) => {
    const b = reg(toggle(iconName, title, () => m.common(prop, ["text"]) === true,
      (v) => m.set({ [prop]: v }, ["text"])));
    const set0 = b.set;
    b.set = () => {
      set0();
      const fam = textOnly() ? m.common("font_family", ["text"]) : null;
      b.disabled = !textOnly() || (styleName && fam && !hasStyle(fam, styleName) && !hasStyle(fam, "bolditalic"));
    };
    return b;
  };
  const bold = styleBtn("bold", "bold", "Bold", "bold");
  const italic = styleBtn("italic", "italic", "Italic", "italic");
  const underline = styleBtn("underline", "underline", "Underline", null);
  const upper = styleBtn("uppercase", "case-upper", "Uppercase", null);
  const align = reg(segmented([["left", "align-left", "Align left"], ["center", "align-center", "Align center"],
    ["right", "align-right", "Align right"]], tget("align"), (v) => m.set({ align: v }, ["text"])));
  const lineH = geom("line_height", "", { iconName: "list", title: "Line height", min: 0.1, max: 10, digits: 2, step: 0.05, kinds: ["text"] });
  const spacing = geom("letter_spacing", "", { iconName: "arrow-left-right", title: "Letter spacing", min: -50, max: 500, digits: 2, kinds: ["text"], unit: "px" });
  const color = reg(colorField({ label: ui3 ? "" : "Colour", name: "color", docColors: () => docColors(app), get: tget("color"), open: picker,
    set: (c) => m.set({ color: c }, ["text"]) }));
  // Figma: typography, then the text colour under Fill; elsewhere colour sits with the text.
  const textSection = ui3
    ? section("Typography", row(font), ...typography(app, m, reg, textOnly).rows)
    : section(T.text, row(font), row(size, bold, italic, underline, upper), row(align), row(lineH, spacing), row(color));
  const textFill = ui3 ? section("Fill", row(color)) : null;

  const arrange = section("Arrange", row(cb("bringFront"), cb("bringForward"), cb("sendBackward"), cb("sendBack"),
    h("span", { class: "sep" }), cb("duplicate"), cb("lock"), cb("delete")));

  // nothing selected: the page
  const bg = reg(colorField({ label: ui3 ? "" : "Background", name: "background", docColors: () => docColors(app), open: picker,
    get: () => s.doc?.background, set: (c) => s.op("set_background", { color: c }) }));
  const canvasInfo = h("div", { class: "canvas-info" });
  // Figma: with nothing selected the panel shows the page (the canvas colour around the design);
  // the design's background is the frame's Fill, reached by selecting the frame.
  const canvasColor = ui3 ? reg(colorField({ label: "", name: "canvas", open: picker,
    get: () => app.canvasColor || getComputedStyle(document.documentElement).getPropertyValue("--canvas-bg").trim(),
    set: (c) => app.setCanvasColor(c) })) : null;
  const exportBtn = ui3 ? h("button", { class: "btn fg-export-btn", type: "button", dataset: { command: "submit" }, onclick: () => C.submit.run() }) : null;
  const exportSection = ui3 ? section("Export", h("div", { class: "insp-row fg-export-row" },
    h("span", { class: "fg-export-scale" }, "1x"), h("span", { class: "fg-export-fmt" }, "PNG")), h("div", { class: "insp-row" }, exportBtn)) : null;
  const pageSection = ui3 ? section(T.page, row(canvasColor)) : section(T.page, row(bg), canvasInfo);
  const frameHead = ui3 ? h("div", { class: "insp-kind fg-frame-head" }, h("span", { class: "insp-kind-name" }, "Frame")) : null;
  const frameSize = ui3 ? h("div", { class: "insp-row fg-frame-size" }) : null;
  const frameSection = ui3 ? h("div", { class: "fg-frame" }, frameHead,
    section("Layout", h("div", { class: "insp-sub" }, "Dimensions"), frameSize), section("Fill", row(bg))) : null;

  if (ui3) root.append(pageSection, frameSection, kindHead, posSection, layoutSection, appearance, textSection, textFill, fillSection, exportSection, ...extras);
  else if (long) root.append(posSection);  // Canva: styling lives in the toolbar, not the Position panel
  else root.append(pageSection, posSection, textSection, fillSection, appearance, arrange);

  const refresh = () => {
    if (!s.doc) return;
    const els = m.els();
    const kinds = new Set(els.map((e) => e.kind));
    const frame = ui3 && !els.length && !!app.frameSelected;
    pageSection.hidden = els.length > 0 || frame;
    if (frameSection) {
      frameSection.hidden = !frame;
      frameSize.textContent = "";
      frameSize.append(h("span", { class: "fg-static" }, h("span", { class: "num-label" }, "W"), String(s.doc.width)),
        h("span", { class: "fg-static" }, h("span", { class: "num-label" }, "H"), String(s.doc.height)));
      exportBtn.textContent = s.done ? "Exported" : `Export ${app.skin.frameLabel || "Frame 1"}`;  // the design is the frame
      exportBtn.disabled = s.done;
    }
    posSection.hidden = els.length === 0;
    textSection.hidden = !(els.length && kinds.size === 1 && kinds.has("text"));
    if (layoutSection) layoutSection.hidden = els.length === 0;
    if (kindHead) {
      kindHead.hidden = els.length === 0;
      kindHead.firstChild.textContent = els.length > 1 ? "Mixed" : { text: "Text", image: "Image", shape: els[0]?.shape === "ellipse" ? "Ellipse" : "Rectangle" }[els[0]?.kind] || "";
      extras.forEach((x) => { x.hidden = els.length === 0; });
    }
    if (textFill) textFill.hidden = textSection.hidden;
    fillSection.hidden = !(els.length && kinds.size === 1 && kinds.has("shape"));
    appearance.hidden = els.length === 0;
    radius.hidden = !(kinds.size === 1 && kinds.has("shape"));
    arrange.hidden = els.length === 0;
    canvasInfo.textContent = `${s.doc.width} × ${s.doc.height} px · ${s.doc.elements.length} layers`;
    for (const c of controls) c.set?.() ?? c.refresh?.();
  };
  return { el: root, refresh };
}


// ---------------------------------------------------------------- text presets (Canva)

export function textPresets(app) {
  // Canva's Text panel: search, Add a text box, and the default heading / subheading / body styles.
  // Searching lists fonts; a click adds a text box set in that font.
  const C = app.commands;
  const search = h("input", { class: "cv-textsearch", placeholder: "Search fonts and combinations", spellcheck: "false", "aria-label": "Search fonts and combinations" });
  const results = h("div", { class: "text-results" });
  const styles = h("div", { class: "text-styles" },
    h("div", { class: "panel-heading" }, "Default text styles"),
    h("button", { class: "preset preset-h1", type: "button", onclick: () => C.addHeading.run() }, "Add a heading"),
    h("button", { class: "preset preset-h2", type: "button", onclick: () => C.addSubheading.run() }, "Add a subheading"),
    h("button", { class: "preset preset-body", type: "button", onclick: () => C.addBody.run() }, "Add a little bit of body text"));
  let families = [];
  const paint = () => {
    const q = search.value.trim().toLowerCase();
    styles.hidden = !!q;
    results.textContent = "";
    if (!q) return;
    const hits = families.filter((f) => f.toLowerCase().includes(q)).slice(0, 30);
    ensureFonts(hits);
    results.append(h("div", { class: "panel-heading" }, "Fonts"), ...hits.map((f) => h("button", { class: "font-sample", type: "button", style: { fontFamily: `'${f}'` },
      onclick: () => app.addTextPreset({ text: f, font_family: f, font_size: Math.round(app.store.doc.height / 16) }) }, f)));
    if (!hits.length) results.append(h("div", { class: "panel-hint" }, `No results for \u201c${search.value.trim()}\u201d`));
  };
  search.addEventListener("input", paint);
  search.addEventListener("keydown", (e) => { if (e.key !== "Escape") e.stopPropagation(); });
  const root = h("div", { class: "panel-body text-presets" },
    h("div", { class: "cv-searchbox" }, h("span", { html: icon("search", 16) }), search),
    // As in Canva: the text box lands on the page straight away, ready to type into.
    h("button", { class: "btn btn-primary wide", type: "button", onclick: () => {
      const d = app.store.doc;
      if (d) app.addTextAt(d.width * 0.3, d.height * 0.5);
    } }, h("span", { html: icon("type", 16) }), " Add a text box"),
    styles, results);
  const refresh = async () => { if (!families.length) families = (await fontCatalog()).map((f) => f.family); };
  return { el: root, refresh };
}

// ---------------------------------------------------------------- shapes (Canva "Elements")

export function shapesPanel(app) {
  const C = app.commands;
  const shape = (cmd, cls) => h("button", { class: `shape-tile ${cls}`, type: "button", title: cmd.label, "aria-label": cmd.label,
    onclick: () => cmd.run() }, h("span", { class: "shape-demo" }));
  return {
    el: h("div", { class: "panel-body shapes-panel" },
      h("div", { class: "panel-heading" }, "Shapes"),
      h("div", { class: "shape-grid" }, shape(C.addRect, "sq"), shape(C.addRounded, "rr"), shape(C.addEllipse, "ci"))),
    refresh: () => {},
  };
}

export function statusText(app) {
  const s = app.store;
  if (!s.doc) return "";
  const z = Math.round(app.stage.zoom * 100);
  return `${z}% · ${s.doc.width} × ${s.doc.height} px · ${s.doc.elements.length} layers`;
}

export { fmt };
