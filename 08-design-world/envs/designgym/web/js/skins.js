// Six editor skins over one set of components.
//
// A skin decides where things live (panels, toolbars, menus), what they are
// called, and a few interaction details (Canva's rotate handle and floating
// toolbar, Figma's always-on size badge). It never decides what a command
// does. Randomising the skin per episode is how a computer-use agent is kept
// from learning pixel positions instead of learning the editor.

import { fmt, h } from "./util.js";
import { preUploaded } from "./files.js";
import { CANVA_PALETTE, colorPanel, fontPanel, spacingPopover } from "./canva.js";
import { openColorDialog } from "./photoshop.js";
import { colorGallery, commandSearch, fontCombo, fontDialog, formatPane, lineSpacingMenu, sizeCombo } from "./ppt.js";
import { alignMenu, backgroundDialog, colorMenu, fontMenu, formatOptions, menuSearch, spacingMenu } from "./slides.js";
import {
  alignDropdown, dock, documentProperties, exportDialog, fillAndStroke, paletteBar, spin, textAndFont, transformDialog, zoomSpin,
} from "./inkscape.js";
import { hasStyle } from "./fonts.js";
import { icon } from "./icons.js";
import {
  assetsPanel, inspector, layersPanel, selectionModel, shapesPanel, statusText,
  textPresets,
} from "./panels.js";
import {
  closePopover, colorField, colorPicker, commandButton, fontPicker, iconButton, menu, modal, numberField, popover,
  segmented, slider, toggle,
} from "./widgets.js";

function tabs(defs, onChange, { cls = "" } = {}) {
  const bar = h("div", { class: `tabs ${cls}`, role: "tablist" });
  let current = defs[0][0];
  const btns = defs.map(([id, label]) => {
    const b = h("button", { class: "tab", role: "tab", type: "button", dataset: { tab: id } }, label);
    b.addEventListener("click", () => set(id));
    bar.append(b);
    return b;
  });
  const set = (id) => {
    current = id;
    for (const b of btns) {
      b.classList.toggle("on", b.dataset.tab === id);
      b.setAttribute("aria-selected", b.dataset.tab === id ? "true" : "false");
    }
    onChange(id);
  };
  bar.set = set;
  bar.current = () => current;
  return bar;
}

function mountSwitcher(host, panels) {
  let active = null;
  return (id) => {
    host.textContent = "";
    active = panels[id];
    if (active) { host.append(active.el); active.refresh(); }
  };
}

function zoomLabel(app) {
  const b = h("button", { class: "zoom-label", type: "button", "aria-label": "Zoom" }, "100%");
  b.addEventListener("click", () => menu(b, [
    { cmd: app.commands.zoomIn }, { cmd: app.commands.zoomOut }, "-",
    { cmd: app.commands.zoomFit }, { cmd: app.commands.zoom100 }, { cmd: app.commands.zoomSelection },
    { label: "Zoom to 50%", run: () => app.stage.setZoom(0.5) },
    { label: "Zoom to 200%", run: () => app.stage.setZoom(2) },
  ], { align: "end" }));
  app.on("zoom", (z) => { b.textContent = `${Math.round(z * 100)}%`; });
  return b;
}

function submitButton(app, cls, label = "Export") {
  const b = h("button", { class: `btn submit-btn ${cls}`, type: "button", dataset: { command: "submit" } });
  b.innerHTML = `${icon("download", 14)}<span>${label}</span>`;
  b.addEventListener("click", () => app.commands.submit.run());
  b.refresh = () => {
    b.disabled = app.store.done;
    b.querySelector("span").textContent = app.store.done ? "Exported" : label;
  };
  return b;
}

function mainMenuItems(app) {
  // Figma's main menu: File / Edit / View / Object / Text / Arrange open beside it, as submenus.
  const C = app.commands;
  return [
    { label: "File", items: [{ label: "Place image…", cmd: C.placeImage }, ...(app.showCode ? [{ cmd: C.html }] : []),
      "-", { cmd: C.submit }] },
    { label: "Edit", items: [{ cmd: C.undo }, { cmd: C.redo }, "-", { cmd: C.copy }, { cmd: C.cut }, { cmd: C.paste }, { cmd: C.duplicate },
      { cmd: C.delete }, "-", { cmd: C.selectAll }] },
    { label: "View", items: [{ label: "Zoom in", cmd: C.zoomIn }, { label: "Zoom out", cmd: C.zoomOut }, { cmd: C.zoom100 },
      { cmd: C.zoomFit }, { cmd: C.zoomSelection }] },
    { label: "Object", items: [{ cmd: C.bringFront }, { cmd: C.bringForward }, { cmd: C.sendBackward }, { cmd: C.sendBack }, "-",
      { cmd: C.flipH }, { cmd: C.flipV }, "-", { label: "Show/Hide selection", cmd: C.hide }, { label: "Lock/Unlock selection", cmd: C.lock }] },
    { label: "Text", items: [{ cmd: C.bold }, { cmd: C.italic }, { cmd: C.underline }] },
    { label: "Arrange", items: [{ cmd: C.alignLeft }, { cmd: C.alignCenter }, { cmd: C.alignRight }, "-",
      { cmd: C.alignTop }, { cmd: C.alignMiddle }, { cmd: C.alignBottom }, "-", { cmd: C.distributeH }, { cmd: C.distributeV }] },
    "-",
    { label: "Help and account", items: [{ cmd: C.shortcuts }] },
  ];
}

function avatar(app) {
  const agent = app.params.agent;
  return h("div", { class: `avatar ${agent ? "agent" : ""}`, title: agent ? "Agent session" : "You" }, agent ? "AI" : "Y");
}

// ================================================================ Figma

function figma(app, root) {
  // Modelled on Figma's current layout: a narrow navigation rail (File, Assets, ...), the file
  // panel (name, Pages, Layers), the canvas with the floating toolbar, and the properties panel
  // (Design / Prototype tabs over Position, Layout, Typography, Fill, Appearance).
  const C = app.commands;
  const s = app.store;
  const layers = layersPanel(app, { thumbs: false, frame: "Frame 1" });
  const insp = inspector(app, { variant: "figma" });

  // left: navigation rail + file panel
  const logo = h("button", { class: "fg-logo", type: "button", "aria-label": "Main menu" });
  logo.innerHTML = `<span class="brand-mark mark-studio">${icon("pen-tool", 12)}</span>${icon("chevron-down", 10)}`;
  logo.addEventListener("click", () => menu(logo, mainMenuItems(app)));
  const navItem = (id, ic, label, run) => {
    const b = h("button", { class: "fg-nav-btn", type: "button", dataset: { nav: id }, "aria-label": label },
      h("span", { class: "fg-nav-ic", html: icon(ic, 16) }), h("span", { class: "fg-nav-label" }, label));
    b.addEventListener("click", run);
    return b;
  };
  const leftHost = h("div", { class: "fg-left-body" });
  const fileName = h("div", { class: "fg-file-name" }, "Untitled");
  const fileHead = h("div", { class: "fg-filehead" },
    h("div", { class: "fg-file" }, h("div", { class: "fg-file-row" }, fileName),
      h("div", { class: "fg-file-sub" }, "Drafts")),
);
  const pages = h("div", { class: "fg-sec" },
    h("div", { class: "fg-sec-head" }, h("span", {}, "Pages")),
    h("div", { class: "fg-page on" }, h("span", { class: "fg-page-dot" }), "Page 1"));
  const layersHead = h("div", { class: "fg-sec-head fg-layers-head" }, h("span", {}, "Layers"));
  const views = {
    file: { el: h("div", { class: "fg-view" }, fileHead, pages, layersHead, layers.el), refresh: layers.refresh },
  };
  const showLeft = mountSwitcher(leftHost, views);
  let leftView = "file";
  const nav = h("nav", { class: "fg-nav", "aria-label": "Navigation" }, logo,
    navItem("file", "file", "File", () => setLeft("file")),
    h("span", { class: "grow" }),
    iconButton("help-circle", "Help and resources", () => C.shortcuts.run(), { cls: "fg-mini" }));
  const setLeft = (id) => {
    leftView = id;
    showLeft(id);
    nav.querySelectorAll(".fg-nav-btn").forEach((b) => b.classList.toggle("on", b.dataset.nav === id));
  };
  const left = h("aside", { class: "fg-panel fg-left" }, leftHost);

  // right: header (avatar, present, Export), Design / Prototype tabs + zoom, then the panel
  const rightHost = h("div", { class: "fg-right-body" });
  const showRight = mountSwitcher(rightHost, { design: insp });
  const rt = tabs([["design", "Design"]], showRight, { cls: "fg-tabs" });
  const submit = submitButton(app, "fg-share");
  const right = h("aside", { class: "fg-panel fg-right" },
    h("div", { class: "fg-head fg-head-right" }, h("span", { class: "grow" }), submit),
    h("div", { class: "fg-tabrow" }, rt, h("span", { class: "grow" }), zoomLabel(app)), rightHost);

  // canvas + floating toolbar: Move, Frame, Shape, Pen, Text, Comment, Actions | Draw / Design / Dev
  const stageHost = h("div", { class: "fg-center stage-host" });
  const toolBtn = (id, tool) => {
    const b = commandButton(C[id], { cls: "fg-tool" });
    b.refresh = () => { b.classList.toggle("on", s.tool === tool || (tool === "rect" && s.tool === "ellipse")); };
    return b;
  };
  const caret = (title, items) => {
    const b = iconButton("chevron-down", title, () => menu(b, items), { cls: "fg-caret", size: 10 });
    return b;
  };
  const moveBtn = toolBtn("toolMove", "move");
  const rectBtn = toolBtn("toolRect", "rect");
  const textBtn = toolBtn("toolText", "text");
  const tools = [moveBtn, rectBtn, textBtn];
  const toolbar = h("div", { class: "fg-toolbar", role: "toolbar", "aria-label": "Tools" },
    moveBtn, caret("Move tools", [{ cmd: C.toolMove }, { cmd: C.toolHand }, { cmd: C.zoomFit }]),
    rectBtn, caret("Shape tools", [{ cmd: C.toolRect }, { cmd: C.toolEllipse }, "-", { label: "Image/video…", cmd: C.placeImage }]),
    textBtn, ...(app.showCode ? [h("span", { class: "fg-tsep" }), commandButton(C.html, { cls: "fg-tool" })] : []));
  stageHost.append(toolbar);

  root.append(nav, left, stageHost, right);
  setLeft("file");
  rt.set("design");
  const refreshables = [...tools, submit];
  return {
    stageHost,
    panels: [layers, insp],
    // Figma: chosen images wait on the cursor for a click (or a drag to size them); dropped files land where dropped.
    placeAssets: (assets, point, how) => (how.dropped || point ? app.placeAt(assets, point) : app.stage.startPlacing(assets)),
    fileOptions: () => ({ title: "Open" }),
    // Figma's canvas menu, trimmed to what works here, in its order and with its shortcuts.
    contextItems: (onElement) => (onElement
      ? [{ cmd: C.copy }, { label: "Paste here", cmd: C.paste }, "-", { cmd: C.bringFront }, { cmd: C.sendBack }, "-",
        { label: "Show/Hide", cmd: C.hide }, { label: "Lock/Unlock", cmd: C.lock }, "-", { cmd: C.flipH }, { cmd: C.flipV }]
      : [{ label: "Paste here", cmd: C.paste }, "-", { cmd: C.selectAll }, "-", { cmd: C.zoomFit }, { cmd: C.zoom100 }]),
    refresh() {
      fileName.textContent = s.task?.title || "Untitled";
      refreshables.forEach((b) => b.refresh?.());
    },
    reveal(which) {
      if (which === "assets") app.openFiles();
      if (which === "position") rt.set("design");
    },
    leftView: () => leftView,
  };
}

// ================================================================ Canva

function canvaContextBar(app) {
  // Canva's floating toolbar: a pill over the page that appears only while something is selected,
  // with the selection's own tools (text: font, size, colour, style, spacing; image: edit, flip ...).
  const s = app.store;
  const C = app.commands;
  const m = selectionModel(app);
  const bar = h("div", { class: "cv-context", role: "toolbar", "aria-label": "Element toolbar" });
  const controls = [];
  const add = (c) => { controls.push(c); return c; };
  const textOnly = () => { const e = m.els(); return e.length && e.every((x) => x.kind === "text"); };
  const tget = (p) => () => (textOnly() ? m.common(p, ["text"]) : undefined);
  const word = (label, run, cls = "") => h("button", { class: `btn cv-textbtn ${cls}`, type: "button", onclick: run }, label);
  // Font and colour open in Canva's side panel (see canva()).
  const font = add(fontPicker({ get: tget("font_family"), set: (f) => m.set({ font_family: f }, ["text"]), cls: "cv-font",
    docFonts: () => s.task?.fonts_used || [], open: () => app.skinApi.reveal("font"), preview: false }));
  const size = add(numberField({ label: "", title: "Font size", digits: 2, min: 1, max: 2000, cls: "cv-size",
    get: tget("font_size"), set: (v) => m.set({ font_size: v }, ["text"]) }));
  const stepSize = (d) => m.set((e) => ({ font_size: Math.max(1, Math.round(e.font_size + d)) }), ["text"]);
  const sizeGroup = h("div", { class: "cv-size-group" },
    iconButton("minus", "Decrease font size", () => stepSize(-1)), size, iconButton("plus", "Increase font size", () => stepSize(1)));
  const colorBtn = h("button", { class: "icon-btn cv-textcolor", type: "button", title: "Text colour", "aria-label": "Text colour" });
  colorBtn.innerHTML = `<span class="a">A</span><span class="bar"></span>`;
  colorBtn.addEventListener("click", () => app.skinApi.reveal("color"));
  const tog = (prop, iconName, title, face = null) => {
    const b = add(toggle(iconName, title, () => m.common(prop, ["text"]) === true, (v) => m.set({ [prop]: v }, ["text"])));
    const base = b.set;
    b.set = () => {
      base();
      const fam = textOnly() ? m.common("font_family", ["text"]) : null;
      b.disabled = !!(face && fam && !hasStyle(fam, face) && !hasStyle(fam, "bolditalic"));
    };
    return b;
  };
  const bold = tog("bold", "bold", "Bold", "bold");
  const italic = tog("italic", "italic", "Italic", "italic");
  const underline = tog("underline", "underline", "Underline");
  const upper = tog("uppercase", "case-upper", "Uppercase");
  const alignOrder = ["left", "center", "right"];
  const alignBtn = iconButton("align-center", "Alignment", () => {
    const cur = m.common("align", ["text"]) || "left";
    m.set({ align: alignOrder[(alignOrder.indexOf(cur) + 1) % 3] }, ["text"]);
  });
  const spacingBtn = iconButton("letter-spacing", "Spacing", () => spacingPopover(spacingBtn, m, tget));
  const fill = add(colorField({ label: "", showHex: false, cls: "cv-fill", get: () => (m.els().every((e) => e.kind === "shape") ? m.common("fill", ["shape"]) : undefined),
    set: (c) => m.set({ fill: c }, ["shape"]), docColors: () => [], open: () => app.skinApi.reveal("color") }));
  const borderBtn = iconButton("square-dashed", "Border style", () => {
    const sc = colorField({ label: "Border colour", get: () => m.common("stroke", ["shape"]), set: (c) => m.set((e) => ({ stroke: c, stroke_width: e.stroke_width || 4 }), ["shape"]) });
    const sw = numberField({ label: "Weight", min: 0, get: () => m.common("stroke_width", ["shape"]), set: (v) => m.set({ stroke_width: v }, ["shape"]) });
    const rr = numberField({ label: "Corner rounding", min: 0, get: () => m.common("radius", ["shape"]), set: (v) => m.set({ radius: v }, ["shape"]) });
    [sc, sw, rr].forEach((c) => c.set());
    popover(borderBtn, h("div", { class: "cv-pop-body" }, h("div", { class: "cv-pop-title" }, "Border"), sc, sw, rr));
  });
  const flipBtn = word("Flip", () => menu(flipBtn, [{ cmd: C.flipH }, { cmd: C.flipV }]));
  const positionBtn = h("button", { class: "btn cv-textbtn", type: "button", "aria-label": "Position", dataset: { open: "position" } }, "Position");
  positionBtn.addEventListener("click", () => app.skinApi.reveal("position"));
  const transparency = iconButton("grid-3x3", "Transparency", () => {
    const sl = slider({ label: "Transparency", name: "opacity", unit: "%", min: 0, max: 100,
      get: () => { const v = m.common("opacity"); return v === null || v === undefined ? v : Math.round(v * 100); },
      set: (v) => m.set({ opacity: v / 100 }) });
    sl.set();
    popover(transparency, h("div", { class: "cv-pop-body" }, sl), { align: "end" });
  });
  const sep = () => h("span", { class: "cv-sep" });
  const groups = {
    text: h("div", { class: "cv-group" }, font, sizeGroup, colorBtn, bold, italic, underline, upper, alignBtn, spacingBtn),
    image: h("div", { class: "cv-group" }, flipBtn),
    shape: h("div", { class: "cv-group" }, fill, borderBtn),
    common: h("div", { class: "cv-group" }, transparency, positionBtn),
  };
  bar.append(groups.text, groups.image, groups.shape, groups.common);
  bar.refresh = () => {
    const els = m.els();
    const kinds = new Set(els.map((e) => e.kind));
    bar.hidden = els.length === 0;
    groups.text.hidden = !(els.length && kinds.size === 1 && kinds.has("text"));
    groups.shape.hidden = !(els.length && kinds.size === 1 && kinds.has("shape"));
    groups.image.hidden = !(els.length && kinds.size === 1 && kinds.has("image"));
    colorBtn.querySelector(".bar").style.background = tget("color")() || "#000";
    const al = m.common("align", ["text"]) || "left";
    alignBtn.innerHTML = icon(`align-${al}`, 16);
    for (const c of controls) c.set?.();
  };
  return bar;
}

function canvaPosition(app) {
  // Canva's Position panel: an Arrange tab (order, align to page, exact numbers) and a Layers tab.
  const C = app.commands;
  const cb = (id, label) => {
    const b = h("button", { class: "btn cv-arrange", type: "button", dataset: { command: id } }, h("span", { html: icon(C[id].icon, 16) }), label);
    b.addEventListener("click", () => C[id].run());
    return b;
  };
  const insp = inspector(app, { variant: "canva" });
  const layers = layersPanel(app, { thumbs: true });
  const arrange = { el: h("div", { class: "panel-body cv-position" },
    h("div", { class: "panel-heading" }, "Arrange"),
    h("div", { class: "cv-arrange-grid" }, cb("bringForward", "Forward"), cb("sendBackward", "Backward"),
      cb("bringFront", "To front"), cb("sendBack", "To back")),
    h("div", { class: "panel-heading" }, "Align to page"),
    h("div", { class: "cv-arrange-grid three" }, cb("alignTop", "Top"), cb("alignLeft", "Left"), cb("alignMiddle", "Middle"),
      cb("alignCenter", "Centre"), cb("alignBottom", "Bottom"), cb("alignRight", "Right")),
    h("div", { class: "panel-heading" }, "Advanced"), insp.el), refresh: insp.refresh };
  const host = h("div", {});
  const show = mountSwitcher(host, { arrange, layers });
  const bar = tabs([["arrange", "Arrange"], ["layers", "Layers"]], show, { cls: "cv-pos-tabs" });
  bar.set("arrange");
  return { el: h("div", {}, bar, host), refresh: () => { arrange.refresh(); layers.refresh(); }, tabs: bar };
}

function canva(app, root) {
  // Modelled on Canva's current editor: gradient header (menu, File, Resize, undo/redo, saved; design
  // type, people, Share), the side rail and its flyouts, the contextual toolbar, the page with its
  // header, and the footer with Notes / Duration / Timer, page count and zoom.
  const s = app.store;
  const C = app.commands;
  const flyHost = h("div", { class: "cv-fly-body" });
  const flyTitle = h("div", { class: "cv-fly-title" });
  // Canva's Background panel: a row of colours led by the palette button (which opens the colour panel).
  const bgRow = h("div", { class: "cv-bgrow" });
  const background = { el: h("div", { class: "panel-body cv-bgpanel" }, bgRow), refresh: () => {
    const cur = (s.doc?.background || "").toLowerCase();
    bgRow.textContent = "";
    const pal = h("button", { class: "cv-swatch cv-bgpalette", type: "button", title: "Background color", "aria-label": "Background color",
      html: icon("palette", 16), onclick: () => app.skinApi.reveal("bgcolor") });
    bgRow.append(pal, ...["#ffffff", "#000000", ...CANVA_PALETTE.filter((c) => !["#000000", "#ffffff"].includes(c)).slice(0, 12)].map((c) =>
      h("button", { class: `cv-swatch ${c === cur ? "on" : ""}`, type: "button", title: c.toUpperCase(), style: { background: c }, dataset: { color: c },
        onclick: () => s.op("set_background", { color: c }) })));
  } };
  const elementsPanel = shapesPanel(app);
  // Uploads holds what has been uploaded: some episodes start with the images already there,
  // others with an empty panel, where Upload files opens the computer's file window first.
  app.uploaded = new Set(preUploaded(app) ? s.assets.map((a) => a.id) : []);
  const uploadBtn = h("button", { class: "btn btn-primary cv-upload", type: "button", dataset: { open: "files" },
    onclick: () => app.openFiles() }, "Upload files");
  const uploadsAssets = assetsPanel(app, { cols: 2, search: true, placeholder: "Search keywords, tags, color",
    filter: (a) => app.uploaded.has(a.id), empty: "Media you upload will appear here.", top: uploadBtn });
  const uploads = { el: h("div", { class: "cv-uploads" }, uploadsAssets.el), refresh: uploadsAssets.refresh };
  const position = canvaPosition(app);
  const panels = {
    elements: { title: "Elements", panel: elementsPanel },
    text: { title: "Text", panel: textPresets(app) },
    uploads: { title: "Uploads", panel: uploads },
    background: { title: "Background", panel: background },
  };
  // Position, colour and font open in the side panel's place, as in the real editor.
  const m = selectionModel(app);
  const docColors = () => [...new Set([s.doc?.background, ...s.doc.elements.map((e) => (e.kind === "text" ? e.color : e.kind === "shape" ? e.fill : null))].filter(Boolean))];
  const onlyText = () => { const e = m.els(); return e.length && e.every((x) => x.kind === "text"); };
  const colors = colorPanel(app, { docColors, kind: () => (onlyText() ? "Text colors" : "Colors"),
    get: () => (onlyText() ? m.common("color", ["text"]) : m.common("fill", ["shape"])),
    set: (c) => (onlyText() ? m.set({ color: c }, ["text"]) : m.set({ fill: c }, ["shape"])) });
  const fonts = fontPanel(app, { get: () => m.common("font_family", ["text"]), set: (f) => m.set({ font_family: f }, ["text"]),
    docFonts: () => [...s.doc.elements.filter((e) => e.kind === "text").map((e) => e.font_family), ...(s.task?.fonts_used || [])] });
  const bgColors = colorPanel(app, { docColors, kind: "Colors", get: () => s.doc?.background, set: (c) => s.op("set_background", { color: c }) });
  const editViews = { position: { title: "Position", view: position }, color: { title: "Text color", view: colors }, font: { title: "Font", view: fonts },
    bgcolor: { title: "Background color", view: bgColors } };
  const editTitle = h("div", { class: "cv-edit-title" }, "Position");
  const editBody = h("div", { class: "cv-edit-body" });
  const editPanel = h("div", { class: "cv-editpanel closed", dataset: { panel: "" } },
    h("div", { class: "cv-edit-head" }, editTitle, iconButton("x", "Close", () => openEdit(null), { cls: "cv-edit-close" })), editBody);
  const openEdit = (id) => {
    if (id && current) open(null);  // the edit card takes the side panel's place
    editPanel.dataset.panel = id || "";
    editPanel.classList.toggle("closed", !id);
    editBody.textContent = "";
    if (!id) return;
    const v = editViews[id];
    editTitle.textContent = id === "color" && !onlyText() ? "Color" : v.title;
    editPanel.dataset.panel = id;
    editBody.append(v.view.el);
    v.view.refresh();
    if (id === "position") position.tabs.set(position.tabs.current());
    v.view.focus?.();
  };
  let current = null;
  let lastSel = "";
  const fly = h("div", { class: "cv-flyout", dataset: { panel: "" } }, h("div", { class: "cv-fly-head" }, flyTitle,
    iconButton("x", "Close", () => open(null), { cls: "cv-fly-close" })), flyHost,
    h("button", { class: "cv-fly-handle", type: "button", "aria-label": "Hide panel", html: icon("chevron-left", 14), onclick: () => open(null) }));
  const railBtns = {};
  const rail = h("nav", { class: "cv-rail", "aria-label": "Side panels" });
  const railDefs = [["elements", "shapes", "Elements"],
    ["text", "type", "Text"], ["uploads", "cloud-upload", "Uploads"], null, ["background", "grid-3x3", "Background"]];
  for (const d of railDefs) {
    if (!d) { rail.append(h("span", { class: "cv-rail-sep" })); continue; }
    const [id, ic, label] = d;
    const b = h("button", { class: "cv-rail-btn", type: "button", dataset: { panel: id } }, h("span", { class: "cv-rail-ic", html: icon(ic, 22) }), h("span", {}, label));
    b.addEventListener("click", () => open(current === id ? null : id));
    railBtns[id] = b;
    rail.append(b);
  }
  const codeRail = h("button", { class: "cv-rail-btn", type: "button", dataset: { command: "html" } }, h("span", { class: "cv-rail-ic", html: icon("code", 22) }), h("span", {}, "Code"));
  codeRail.addEventListener("click", () => C.html.run());
  if (app.showCode) rail.append(codeRail);
  const open = (id) => {
    if (id && editPanel.dataset.panel) { editPanel.dataset.panel = ""; editPanel.classList.add("closed"); }
    current = id;
    for (const [k, b] of Object.entries(railBtns)) b.classList.toggle("on", k === id);
    fly.classList.toggle("closed", !id);
    fly.dataset.panel = id || "";
    flyHost.textContent = "";
    if (id) {
      flyTitle.textContent = panels[id].title;
      flyHost.append(panels[id].panel.el);
      panels[id].panel.refresh();
    }
    requestAnimationFrame(() => app.stage?.drawOverlay());
  };

  const fileBtn = h("button", { class: "cv-hbtn", type: "button" }, "File");
  fileBtn.addEventListener("click", () => menu(fileBtn, [...(app.showCode ? [{ cmd: C.html }, "-"] : []),
    { label: "Download / Export…", cmd: C.submit }, "-", { cmd: C.shortcuts }]));

  const undo = commandButton(C.undo, { cls: "cv-hicon" });
  const redo = commandButton(C.redo, { cls: "cv-hicon" });
  const saved = h("span", { class: "cv-saved", title: "All changes saved" }, h("span", { html: icon("cloud-check", 18) }));
  const title = h("div", { class: "cv-title" }, "Untitled design");
  const submit = submitButton(app, "cv-share");
  const header = h("header", { class: "cv-header" }, fileBtn, h("span", { class: "cv-hsep" }), undo, redo, saved,
    h("span", { class: "grow" }), title, avatar(app), submit);

  const ctx = canvaContextBar(app);
  const stageHost = h("div", { class: "cv-stage stage-host" });
  const zoomRange = h("input", { type: "range", min: "10", max: "400", value: "100", class: "cv-zoom-range", "aria-label": "Zoom" });
  zoomRange.addEventListener("input", () => app.stage.setZoom(zoomRange.value / 100));
  const zl = zoomLabel(app);
  app.on("zoom", (z) => { zoomRange.value = String(Math.round(z * 100)); });
  const footer = h("footer", { class: "cv-footer" }, h("span", { class: "grow" }), zoomRange, zl,
    h("span", { class: "cv-pagecount" }, "Page 1 / 1"),
    iconButton("maximize", "Fit to screen", () => C.zoomFit.run(), { cls: "cv-ficon" }),
    iconButton("help-circle", "Keyboard shortcuts", () => C.shortcuts.run(), { cls: "cv-ficon" }));
  const main = h("div", { class: "cv-main" }, h("div", { class: "cv-stage-wrap" }, stageHost, ctx), footer);
  root.append(header, h("div", { class: "cv-body" }, rail, fly, editPanel, main));
  open("uploads");
  return {
    stageHost,
    panels: [...Object.values(panels).map((p) => p.panel), position, { el: colors.el, refresh: colors.refresh }, { el: fonts.el, refresh: fonts.refresh }, { el: bgColors.el, refresh: bgColors.refresh }],
    refresh() {
      // Canva names a new design after its first text, until it is renamed.
      const first = s.doc.elements.find((e) => e.kind === "text" && e.text.trim());
      title.textContent = first ? first.text.trim().split("\n")[0].slice(0, 40) : "Untitled design";
      ctx.refresh();
      [undo, redo].forEach((b) => b.refresh());
      submit.refresh();
      if (editPanel.dataset.panel && !s.selection.length) openEdit(null);
      // On narrow windows the side panel floats over the page; selecting on the canvas puts it away.
      if (s.selection.length && current && matchMedia("(max-width: 920px)").matches && s.selection.join() !== lastSel) open(null);
      lastSel = s.selection.join();
    },
    reveal(which) {
      if (which === "assets") open("uploads");
      else if (which in editViews) openEdit(editPanel.dataset.panel === which ? null : which);  // the button toggles its panel
    },
    // Canva: Upload files only uploads (the images then wait in Uploads); a file dropped on the page is placed there too.
    placeAssets: async (assets, point, how) => {
      assets.forEach((a) => app.uploaded.add(a.id));
      if (how.dropped) await app.placeAt(assets, point);
      open("uploads");
      uploadsAssets.refresh();
    },
  };
}

// ================================================================ Photoshop

function photoshop(app, root) {
  const s = app.store;
  const C = app.commands;
  const m = selectionModel(app);

  const menus = {
    File: [{ label: "Place Embedded…", cmd: C.placeImage },
      ...(app.showCode ? [{ cmd: C.html }] : []), "-",
      { label: "Export As…", cmd: C.submit }],
    Edit: [{ cmd: C.undo }, { cmd: C.redo }, "-", { cmd: C.cut }, { cmd: C.copy }, { cmd: C.paste }, { label: "Clear", cmd: C.delete }, "-",
      { cmd: C.freeTransform }, { label: "Transform: Flip Horizontal", cmd: C.flipH }, { label: "Transform: Flip Vertical", cmd: C.flipV }, "-", { cmd: C.shortcuts }],
    Layer: [{ label: "New Text Layer", cmd: C.addBody }, { label: "New Shape Layer", cmd: C.addRect }, { label: "Duplicate Layer", cmd: C.duplicate },
      { label: "Delete Layer", cmd: C.delete }, "-", { header: "Arrange" }, { cmd: C.bringFront }, { cmd: C.bringForward }, { cmd: C.sendBackward },
      { cmd: C.sendBack }, "-", { header: "Align" }, { label: "Left Edges", cmd: C.alignLeft }, { label: "Horizontal Centers", cmd: C.alignCenter },
      { label: "Right Edges", cmd: C.alignRight }, { label: "Top Edges", cmd: C.alignTop }, { label: "Vertical Centers", cmd: C.alignMiddle },
      { label: "Bottom Edges", cmd: C.alignBottom }, "-", { label: "Lock Layers…", cmd: C.lock }, { label: "Hide Layers", cmd: C.hide }],
    Type: [{ label: "Add Heading", cmd: C.addHeading }, { label: "Add Subheading", cmd: C.addSubheading }, { label: "Add Paragraph", cmd: C.addBody }],
    Select: [{ label: "All Layers", cmd: C.selectAll }, { cmd: C.deselect }],
    View: [{ cmd: C.zoomIn }, { cmd: C.zoomOut }, { label: "Fit on Screen", cmd: C.zoomFit }, { label: "100%", cmd: C.zoom100 }],
    Window: [{ label: "Color", run: () => dock0.set("color") }, { label: "Swatches", run: () => dock0.set("swatches") }, "-",
      { label: "Properties", run: () => dock1.set("properties") }, "-",
      { label: "Layers", run: () => dock2.set("layers") }],
    Help: [{ cmd: C.shortcuts }],
  };
  // Photoshop's menu bar order, less the menus with nothing that works here (Image, Filter, 3D, Plugins).
  const order = ["File", "Edit", "Layer", "Type", "Select", "View", "Window", "Help"];
  for (const k of Object.keys(menus)) if (!order.includes(k)) order.push(k);
  const sorted = Object.fromEntries(order.filter((k) => menus[k]).map((k) => [k, menus[k]]));
  for (const k of Object.keys(menus)) delete menus[k];
  Object.assign(menus, sorted);
  const menubar = h("div", { class: "ps-menubar", role: "menubar" }, h("div", { class: "ps-logo brand-mark mark-pixel", title: "Pixel", html: icon("grid-3x3", 13) }));
  for (const [name, items] of Object.entries(menus)) {
    const b = h("button", { class: "ps-menu", type: "button", role: "menuitem" }, name);
    b.addEventListener("click", () => menu(b, items));
    b.addEventListener("pointerenter", () => { if (document.querySelector(".pop-menu") && !b.matches(":focus")) { closePopover(); b.click(); } });
    menubar.append(b);
  }
  const submit = submitButton(app, "ps-share");
  menubar.append(h("span", { class: "grow" }), submit);

  // options bar: contents follow the active tool / selection
  const options = h("div", { class: "ps-options" });
  // Every colour swatch opens Photoshop's Color Picker dialog; Add to Swatches feeds the Swatches panel.
  app.psSwatches = app.psSwatches || [];
  app.addSwatch = (c) => { if (!app.psSwatches.includes(c)) app.psSwatches.push(c); swatchPanel.refresh(); };
  const dialog = (title) => (chip, get, set) => openColorDialog(chip, { title, get, set, addSwatch: app.addSwatch });
  app.autoSelect = true;          // Photoshop / Photopea: Auto-Select (Layer) on
  app.transformControls = false;  // and Show Transform Controls off: handles come with Mod+T
  const fgColor = () => app.fgColor;
  const optControls = [];
  const opt = (c) => { optControls.push(c); return c; };
  const textOnly = () => { const e = m.els(); return e.length && e.every((x) => x.kind === "text"); };
  const tget = (p) => () => (textOnly() ? m.common(p, ["text"]) : undefined);
  const psFont = opt(fontPicker({ get: tget("font_family"), set: (f) => m.set({ font_family: f }, ["text"]), cls: "ps-font", docFonts: () => s.task?.fonts_used || [] }));
  const psStyle = h("select", { class: "ps-select", "aria-label": "Font style" }, ["Regular", "Bold", "Italic", "Bold Italic"].map((o) => h("option", { value: o }, o)));
  psStyle.addEventListener("change", () => {
    const v = psStyle.value;
    m.set({ bold: v.includes("Bold"), italic: v.includes("Italic") }, ["text"]);
  });
  const psSize = opt(numberField({ label: "", iconName: "type", title: "Font size", unit: "px", digits: 2, min: 1, max: 2000, get: tget("font_size"), set: (v) => m.set({ font_size: v }, ["text"]), cls: "ps-num" }));
  const psAlign = opt(segmented([["left", "align-left", "Left align text"], ["center", "align-center", "Center text"], ["right", "align-right", "Right align text"]],
    tget("align"), (v) => m.set({ align: v }, ["text"])));
  const psColor = opt(colorField({ label: "", showHex: false, get: tget("color"), set: (c) => m.set({ color: c }, ["text"]), cls: "ps-color",
    open: dialog("Color Picker (Text Color)") }));
  // While typing, the options bar ends with Cancel (x) and Commit (check), as in Photoshop.
  const cancelEdit = iconButton("x", "Cancel any current edits (Esc)", () => app.stage.cancelEdit(), { cls: "ps-edit-x" });
  const commitEdit = iconButton("check", "Commit any current edits", () => app.stage.commitEdit(), { cls: "ps-edit-ok" });
  const editBtns = h("div", { class: "ps-editbtns" }, cancelEdit, commitEdit);
  app.on("editing", () => app.skinApi?.refresh());
  const typeOpts = h("div", { class: "ps-optgroup" }, h("span", { class: "ps-tool-ic", html: icon("type", 16) }), psFont, psStyle, psSize,
    psAlign, psColor, h("span", { class: "grow" }), editBtns);
  const alignBtns = ["alignLeft", "alignCenter", "alignRight", "alignTop", "alignMiddle", "alignBottom", "distributeH", "distributeV"].map((id) => commandButton(C[id]));
  const check = (label, get, set, cls) => {
    const box = h("input", { type: "checkbox", class: `ps-check ${cls}`, "aria-label": label });
    box.addEventListener("change", () => { set(box.checked); app.stage.drawOverlay(); app.refreshAll(); });
    const l = h("label", { class: "ps-checklabel" }, box, label);
    l.set = () => { box.checked = !!get(); };
    return opt(l);
  };
  const autoSel = check("Auto-Select", () => app.autoSelect, (v) => { app.autoSelect = v; }, "ps-autoselect");
  const showTc = check("Show Transform Controls", () => app.transformControls, (v) => { app.transformControls = v; }, "ps-transform");
  const moveOpts = h("div", { class: "ps-optgroup" }, h("span", { class: "ps-tool-ic", html: icon("move", 16) }), autoSel,
    h("span", { class: "ps-optlabel" }, "Layer"), showTc, h("span", { class: "tb-sep" }), ...alignBtns);
  const shapeFill = opt(colorField({ label: "Fill:", showHex: false, get: () => (m.els().length && m.els().every((e) => e.kind === "shape") ? m.common("fill", ["shape"]) : app.fgColor),
    set: (c) => { if (m.els().length) m.set({ fill: c }, ["shape"]); else app.setFg(c); }, cls: "ps-color", open: dialog("Color Picker (Fill Color)") }));
  const shapeStroke = opt(colorField({ label: "Stroke:", showHex: false, get: () => (m.els().every((e) => e.kind === "shape") ? m.common("stroke", ["shape"]) : undefined),
    set: (c) => m.set((e) => ({ stroke: c, stroke_width: e.stroke_width || 4 }), ["shape"]), cls: "ps-color", open: dialog("Color Picker (Stroke Color)") }));
  const shapeW = opt(numberField({ label: "", iconName: "minus", title: "Stroke width", unit: "px", min: 0, get: () => (m.els().every((e) => e.kind === "shape") ? m.common("stroke_width", ["shape"]) : undefined),
    set: (v) => m.set({ stroke_width: v }, ["shape"]), cls: "ps-num" }));
  const shapeOpts = h("div", { class: "ps-optgroup" }, h("span", { class: "ps-tool-ic", html: icon("square", 16) }),
    h("span", { class: "ps-optlabel" }, "Shape"), shapeFill, shapeStroke, shapeW);
  const handOpts = h("div", { class: "ps-optgroup" }, h("span", { class: "ps-tool-ic", html: icon("hand", 16) }),
    h("button", { class: "btn ps-btn", type: "button", onclick: () => C.zoom100.run() }, "100%"),
    h("button", { class: "btn ps-btn", type: "button", onclick: () => C.zoomFit.run() }, "Fit Screen"));
  options.append(moveOpts, typeOpts, shapeOpts, handOpts);

  // Tools column, in Photoshop's order, holding the tools that work here: Move, Type, the shape tools, Hand and Zoom.
  const toolBtns = [];
  const toolcol = h("div", { class: "ps-tools", role: "toolbar", "aria-label": "Tools" });
  const live = (id, tool, title) => {
    const b = commandButton(C[id], { cls: "ps-tool" });
    b.title = b.ariaLabel = title;
    b.setAttribute("aria-label", title);
    b.refresh = () => b.classList.toggle("on", s.tool === tool);
    toolBtns.push(b);
    return b;
  };
  const sep = () => h("span", { class: "ps-tsep" });
  toolcol.append(
    live("toolMove", "move", "Move Tool (V)"), sep(),
    live("toolText", "text", "Horizontal Type Tool (T)"), live("toolRect", "rect", "Rectangle Tool (U)"),
    live("toolEllipse", "ellipse", "Ellipse Tool (U)"), sep(),
    live("toolHand", "hand", "Hand Tool (H)"),
    iconButton("zoom-in", "Zoom Tool (Z)", () => C.zoomIn.run(), { cls: "ps-tool" }));
  const fgChip = h("button", { class: "ps-fg", type: "button", title: "Set foreground color", "aria-label": "Set foreground color" });
  fgChip.addEventListener("click", () => openColorDialog(fgChip, { title: "Color Picker (Foreground Color)", get: () => app.fgColor,
    set: (c) => app.setFg(c), addSwatch: app.addSwatch }));
  const swatches = h("div", { class: "ps-swatches" }, fgChip, h("span", { class: "ps-bgc" }));
  toolcol.append(sep(), swatches);

  // document area
  const docTab = h("div", { class: "ps-doctab on" }, h("span", { class: "ps-doctab-name" }),
  );
  const stageHost = h("div", { class: "ps-stage stage-host" });
  const status = h("div", { class: "ps-status" });
  const docArea = h("div", { class: "ps-docarea" }, h("div", { class: "ps-doctabs" }, docTab), stageHost, status);

  // Dock, as Photoshop's Essentials workspace (Color | Swatches, Properties | Libraries, Layers),
  // less the panels with nothing to do here. The Color panel sets the foreground colour.
  const colorBody = h("div", { class: "ps-colorpanel" });
  const colorPanel = { el: colorBody, refresh: () => {
    if (colorBody.dataset.fg === app.fgColor) return;
    colorBody.dataset.fg = app.fgColor;
    colorBody.textContent = "";
    colorBody.append(colorPicker(app.fgColor || "#000000", (c) => app.setFg(c)));
  } };
  const SWATCHES = ["#000000", "#ffffff", "#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#7f7f7f", "#c3c3c3",
    "#ed1c24", "#ff7f27", "#fff200", "#22b14c", "#00a2e8", "#3f48cc", "#a349a4", "#b97a57", "#ffaec9", "#ffc90e",
    "#efe4b0", "#b5e61d", "#99d9ea", "#7092be", "#c8bfe7", "#880015", "#1d1d1d", "#3d3d3d", "#5e5e5e", "#9e9e9e"];
  const swatchGrid = h("div", { class: "ps-swatch-grid" });
  const swatchPanel = { el: swatchGrid, refresh() {
    swatchGrid.textContent = "";
    swatchGrid.append(...[...SWATCHES, ...app.psSwatches].map((c) => h("button", {
      class: "ps-swatch", type: "button", title: c, "aria-label": `Swatch ${c}`, style: { background: c }, onclick: () => app.setFg(c) })));
  } };
  swatchPanel.refresh();
  const d0host = h("div", { class: "ps-dock-body" });
  const show0 = mountSwitcher(d0host, { color: colorPanel, swatches: swatchPanel });
  const dock0 = tabs([["color", "Color"], ["swatches", "Swatches"]], show0, { cls: "ps-tabs" });

  const propsPanel = inspector(app, { variant: "photoshop" });
  const d1host = h("div", { class: "ps-dock-body" });
  const group1 = { properties: propsPanel };
  const tabs1 = [["properties", "Properties"]];
  const show1 = mountSwitcher(d1host, group1);
  const dock1 = tabs(tabs1, show1, { cls: "ps-tabs" });

  const layers = layersPanel(app, { thumbs: true, background: true, compact: true });
  const opac = opt(numberField({ label: "Opacity:", unit: "%", min: 0, max: 100, cls: "ps-num",
    get: () => { const v = m.common("opacity"); return v === null || v === undefined ? v : Math.round(v * 100); },
    set: (v) => m.set({ opacity: v / 100 }) }));
  const lockBtn = commandButton(C.lock, { cls: "ps-mini" });
  const layerHead = h("div", { class: "ps-layer-head" },
    h("div", { class: "ps-row" }, h("span", { class: "ps-optlabel" }, "Lock:"), lockBtn, h("span", { class: "grow" }), opac));
  const layerFoot = h("div", { class: "ps-layer-foot" },
    commandButton(C.duplicate, { cls: "ps-mini" }), iconButton("square-plus", "New shape layer", () => C.addRect.run(), { cls: "ps-mini" }),
    commandButton(C.delete, { cls: "ps-mini" }));
  const layersWrap = { el: h("div", { class: "ps-layers" }, layerHead, layers.el, layerFoot), refresh: () => { layers.refresh(); lockBtn.refresh(); } };
  const d2host = h("div", { class: "ps-dock-body grow" });
  const show2 = mountSwitcher(d2host, { layers: layersWrap });
  const dock2 = tabs([["layers", "Layers"]], show2, { cls: "ps-tabs" });
  const dock = h("aside", { class: "ps-dock" }, h("div", { class: "ps-group ps-group-color" }, dock0, d0host),
    h("div", { class: "ps-group" }, dock1, d1host), h("div", { class: "ps-group grow" }, dock2, d2host));

  root.append(menubar, options, h("div", { class: "ps-body" }, toolcol, docArea, dock));
  dock0.set("color");
  dock1.set("properties");
  dock2.set("layers");
  return {
    stageHost,
    panels: [colorPanel, propsPanel, layersWrap],
    refresh() {
      const tool = s.tool;
      const sel = m.els();
      const kinds = new Set(sel.map((e) => e.kind));
      const textSel = sel.length && kinds.size === 1 && kinds.has("text");
      const shapeSel = sel.length && kinds.size === 1 && kinds.has("shape");
      typeOpts.hidden = !(tool === "text" || textSel);
      editBtns.hidden = !app.stage.editing;
      shapeOpts.hidden = !(tool === "rect" || tool === "ellipse" || shapeSel) || !typeOpts.hidden;
      handOpts.hidden = tool !== "hand";
      moveOpts.hidden = !typeOpts.hidden || !shapeOpts.hidden || !handOpts.hidden;
      const b = m.common("bold", ["text"]), it = m.common("italic", ["text"]);
      psStyle.value = b && it ? "Bold Italic" : b ? "Bold" : it ? "Italic" : "Regular";
      psStyle.disabled = !textSel;
      for (const c of optControls) c.set?.();
      alignBtns.forEach((x) => x.refresh());
      toolBtns.forEach((x) => x.refresh());
      fgChip.style.background = fgColor();
      submit.refresh();
      const z = Math.round(app.stage.zoom * 1000) / 10;
      const layer = sel.length === 1 ? sel[0] : null;
      const layerName = layer ? (layer.kind === "text" ? (layer.text || "").split("\n")[0].slice(0, 24) : layer.name || (layer.kind === "image" ? `Layer ${layer.asset_id}` : "Shape")) : "Background";
      docTab.querySelector(".ps-doctab-name").textContent = `${(s.task?.title || "Untitled").slice(0, 28)}.psd @ ${z}% (${layerName}, RGB/8)`;
      status.textContent = statusText(app).replace(/ · /g, "   |   ");
      colorPanel.refresh();
    },
    reveal(which) { if (which === "assets") app.openFiles(); },
  };
}

// ================================================================ shared bits for the document-style skins

// Text controls as they appear in slide and vector editors' toolbars: font, size with
// steppers, B / I / U, colour, alignment. `prefix` namespaces the classes (sl-, pp-, ik-).
function textControls(app, prefix, { steppers = true, colorButton = true, toggles = true } = {}) {
  const s = app.store;
  const m = selectionModel(app);
  const controls = [];
  const add = (c) => { controls.push(c); return c; };
  const textOnly = () => { const e = m.els(); return e.length && e.every((x) => x.kind === "text"); };
  const tget = (p) => () => (textOnly() ? m.common(p, ["text"]) : undefined);
  const font = add(fontPicker({ get: tget("font_family"), set: (f) => m.set({ font_family: f }, ["text"]), cls: `${prefix}-font`,
    docFonts: () => s.task?.fonts_used || [] }));
  const size = add(numberField({ label: "", title: "Font size", digits: 2, min: 1, max: 2000, cls: `${prefix}-size`,
    get: tget("font_size"), set: (v) => m.set({ font_size: v }, ["text"]) }));
  const step = (d) => m.set((e) => ({ font_size: Math.max(1, Math.round(e.font_size + d)) }), ["text"]);
  const sizeGroup = steppers
    ? h("div", { class: `${prefix}-sizegroup` }, iconButton("minus", "Decrease font size", () => step(-1)), size,
      iconButton("plus", "Increase font size", () => step(1)))
    : size;
  const tog = (prop, ic, title, face) => {
    const b = add(toggle(ic, title, () => m.common(prop, ["text"]) === true, (v) => m.set({ [prop]: v }, ["text"])));
    const base = b.set;
    b.set = () => {
      base();
      const fam = textOnly() ? m.common("font_family", ["text"]) : null;
      b.disabled = !textOnly() || !!(face && fam && !hasStyle(fam, face) && !hasStyle(fam, "bolditalic"));
    };
    return b;
  };
  const styleBtns = toggles ? [tog("bold", "bold", "Bold", "bold"), tog("italic", "italic", "Italic", "italic"),
    tog("underline", "underline", "Underline", null)] : [];
  let colorBtn = null;
  if (colorButton) {
    colorBtn = h("button", { class: `icon-btn ${prefix}-textcolor`, type: "button", title: "Text colour", "aria-label": "Text colour" });
    colorBtn.innerHTML = `<span class="a">A</span><span class="bar"></span>`;
    colorBtn.addEventListener("click", () => {
      const field = colorField({ get: tget("color"), set: (c) => m.set({ color: c }, ["text"]), label: "Text colour",
        docColors: () => [...new Set(s.doc.elements.map((e) => e.color || e.fill).filter(Boolean))] });
      field.set();
      document.body.append(field);
      field.style.position = "fixed";
      const r = colorBtn.getBoundingClientRect();
      Object.assign(field.style, { left: `${r.left}px`, top: `${r.top}px`, opacity: "0", pointerEvents: "none" });
      field.chip.click();
      setTimeout(() => field.remove(), 0);
    });
  }
  const align = add(segmented([["left", "align-left", "Align left"], ["center", "align-center", "Center"],
    ["right", "align-right", "Align right"]], tget("align"), (v) => m.set({ align: v }, ["text"])));
  const refresh = () => {
    for (const c of controls) c.set?.();
    if (colorBtn) colorBtn.querySelector(".bar").style.background = tget("color")() || "#000";
  };
  return { font, size, sizeGroup, styleBtns, colorBtn, align, refresh, textOnly };
}

// A right-hand task pane that shows one panel at a time (Format options, Pictures, ...).
function taskPane(app, cls, defs) {
  const host = h("div", { class: "pane-body" });
  const title = h("div", { class: "pane-title" });
  const el = h("aside", { class: `task-pane ${cls} closed`, dataset: { panel: "" } },
    h("div", { class: "pane-head" }, title, iconButton("x", "Close", () => open(null), { cls: "pane-close" })), host);
  const open = (id) => {
    if (id && !defs[id]) return;
    el.dataset.panel = id || "";
    el.classList.toggle("closed", !id);
    host.textContent = "";
    if (id) {
      title.textContent = typeof defs[id].title === "function" ? defs[id].title() : defs[id].title;
      host.append(defs[id].panel.el);
      defs[id].panel.refresh();
    }
    requestAnimationFrame(() => app.stage?.drawOverlay());
  };
  return { el, open, current: () => el.dataset.panel, retitle: () => { const id = el.dataset.panel; if (id) title.textContent = typeof defs[id].title === "function" ? defs[id].title() : defs[id].title; } };
}

function thumbStrip(app, cls, label) {
  const s = app.store;
  const img = h("img", { class: "thumb-img", alt: "Slide 1", draggable: "false" });
  const card = h("div", { class: "thumb-card on" }, h("span", { class: "thumb-n" }, "1"), h("div", { class: "thumb-frame" }, img));
  const el = h("aside", { class: `thumb-strip ${cls}`, "aria-label": label }, card);
  let last = -1, timer = null;
  const refresh = () => {
    if (!s.doc || s.version === last) return;
    last = s.version;
    clearTimeout(timer);  // a live miniature of the current slide, as slide editors show
    timer = setTimeout(() => { img.src = `/api/episodes/${s.episodeId}/render.png?size=240&v=${s.version}`; }, 400);
  };
  return { el, refresh };
}

// ================================================================ Slides (Google Slides-like)

function slides(app, root) {
  // Google Slides (checked in the browser): menus with a "Menus" search; one pill toolbar whose text
  // controls are font ▾, - size +, B I U, text colour, align ▾, line spacing ▾ and Format options;
  // Slides' colour menu, font menu and Fonts dialog; rulers; the filmstrip; Format options on demand.
  const s = app.store;
  const C = app.commands;
  const m = selectionModel(app);
  const fo = formatOptions(app, m);
  const images = assetsPanel(app, { cols: 2, search: true, heading: "Google Drive · Recent", placeholder: "Search Drive" });
  const pane = taskPane(app, "sl-side", { format: { title: "Format options", panel: fo }, images: { title: "Drive & Photos", panel: images } });
  const textOnly = () => { const e = m.els(); return e.length && e.every((x) => x.kind === "text"); };
  const shapeOnly = () => { const e = m.els(); return e.length && e.every((x) => x.kind === "shape"); };
  const tget = (p) => () => (textOnly() ? m.common(p, ["text"]) : undefined);
  const toggleText = (prop) => () => m.set({ [prop]: m.common(prop, ["text"]) !== true }, ["text"]);
  const menus = {
    File: [...(app.showCode ? [{ cmd: C.html }, "-"] : []), { label: "Download / Export…", cmd: C.submit }, "-", { cmd: C.shortcuts }],
    Edit: [{ cmd: C.undo }, { cmd: C.redo }, "-", { cmd: C.cut }, { cmd: C.copy }, { cmd: C.paste }, { cmd: C.duplicate }, { cmd: C.delete }, "-", { cmd: C.selectAll }],
    View: [{ cmd: C.zoomIn }, { cmd: C.zoomOut }, { label: "Fit", cmd: C.zoomFit }, { cmd: C.zoom100 }],
    Insert: [{ label: "Image", items: [{ label: "Upload from computer", cmd: C.placeImage }, { label: "Drive & Photos", run: () => pane.open("images") }] },
      { label: "Text box", cmd: C.toolText }, { label: "Shape", items: [{ label: "Rectangle", cmd: C.toolRect }, { label: "Oval", cmd: C.toolEllipse }] }],
    Format: [{ label: "Text", items: [{ label: "Bold", key: "Mod+B", run: toggleText("bold"), enabled: textOnly() },
      { label: "Italic", key: "Mod+I", run: toggleText("italic"), enabled: textOnly() }, { label: "Underline", key: "Mod+U", run: toggleText("underline"), enabled: textOnly() },
      "-", { label: "Capitalization", items: [{ label: "UPPERCASE", icon: m.common("uppercase", ["text"]) ? "check" : null, run: () => m.set({ uppercase: true }, ["text"]), enabled: textOnly() },
        { label: "As typed", icon: textOnly() && !m.common("uppercase", ["text"]) ? "check" : null, run: () => m.set({ uppercase: false }, ["text"]), enabled: textOnly() }] }] },
    { label: "Align & indent", items: ["left", "center", "right"].map((v) => ({ label: v[0].toUpperCase() + v.slice(1), run: () => m.set({ align: v }, ["text"]), enabled: textOnly() })) },
    { label: "Line & paragraph spacing", items: [{ label: "Single", run: () => m.set({ line_height: 1 }, ["text"]) }, { label: "1.15", run: () => m.set({ line_height: 1.15 }, ["text"]) },
      { label: "1.5", run: () => m.set({ line_height: 1.5 }, ["text"]) }, { label: "Double", run: () => m.set({ line_height: 2 }, ["text"]) }] },
    "-", { label: "Format options", run: () => pane.open("format") }],
    Slide: [{ label: "Change background", run: () => backgroundDialog(app) }],
    Arrange: [{ label: "Order", items: [{ cmd: C.bringFront }, { cmd: C.bringForward }, { cmd: C.sendBackward }, { cmd: C.sendBack }] },
      { label: "Align", items: [{ cmd: C.alignLeft }, { cmd: C.alignCenter }, { cmd: C.alignRight }, "-", { cmd: C.alignTop }, { cmd: C.alignMiddle }, { cmd: C.alignBottom }] },
      { label: "Rotate", items: [{ label: "Rotate clockwise 90°", cmd: C.rotateCw }, { label: "Rotate counter-clockwise 90°", cmd: C.rotateCcw }, "-",
        { label: "Flip horizontally", cmd: C.flipH }, { label: "Flip vertically", cmd: C.flipV }] }],
    Help: [{ cmd: C.shortcuts }],
  };
  const menubar = h("div", { class: "sl-menubar", role: "menubar" });
  for (const [name, items] of Object.entries(menus)) {
    const b = h("button", { class: "sl-menu", type: "button", role: "menuitem" }, name);
    // Menus rebuild on open so enabled / checked states follow the selection.
    b.addEventListener("click", () => menu(b, typeof items === "function" ? items() : name === "Format" ? rebuildFormat() : items));
    menubar.append(b);
  }
  const rebuildFormat = () => {
    const fmt0 = menus.Format;
    const t = textOnly();
    return fmt0.map((it) => (it && it.items && it.label === "Text" ? { ...it, items: [
      { label: "Bold", key: "Mod+B", run: toggleText("bold"), enabled: t }, { label: "Italic", key: "Mod+I", run: toggleText("italic"), enabled: t },
      { label: "Underline", key: "Mod+U", run: toggleText("underline"), enabled: t }, "-",
      { label: "Capitalization", items: [{ label: "UPPERCASE", icon: t && m.common("uppercase", ["text"]) ? "check" : null, run: () => m.set({ uppercase: true }, ["text"]), enabled: t },
        { label: "As typed", icon: t && !m.common("uppercase", ["text"]) ? "check" : null, run: () => m.set({ uppercase: false }, ["text"]), enabled: t }] }] }
      : it && it.items && it.label === "Align & indent" ? { ...it, items: it.items.map((x) => ({ ...x, enabled: t })) } : it));
  };
  const title = h("div", { class: "sl-doctitle" }, "Untitled presentation");
  const submit = submitButton(app, "sl-share");
  const header = h("header", { class: "sl-header" },
    h("span", { class: "brand-mark mark-deck", html: icon("rectangle-horizontal", 16) }),
    h("div", { class: "sl-titlebox" }, h("div", { class: "sl-titlerow" }, title,
      h("span", { class: "sl-ico", title: "All changes saved", html: icon("cloud-check", 15) })), menubar),
    h("span", { class: "grow" }), submit, avatar(app));

  // toolbar
  const controls = [];
  const add = (c) => { controls.push(c); return c; };
  const tool = (id, t) => { const b = commandButton(C[id], { cls: "sl-tool" }); b.refresh = () => b.classList.toggle("on", s.tool === t); return b; };
  const tools = [tool("toolMove", "move"), tool("toolText", "text")];
  const imageBtn = iconButton("image", "Insert image", () => menu(imageBtn, [{ label: "Upload from computer", cmd: C.placeImage },
    { label: "Drive & Photos", run: () => pane.open("images") }]), { cls: "sl-tool" });
  imageBtn.dataset.open = "images";
  const shapeBtn = iconButton("shapes", "Shape", () => menu(shapeBtn, [{ cmd: C.toolRect }, { cmd: C.toolEllipse }]), { cls: "sl-tool" });
  const undo = commandButton(C.undo, { cls: "sl-tool" });
  const redo = commandButton(C.redo, { cls: "sl-tool" });
  const zoomBtn = h("button", { class: "sl-zoom sl-drop", type: "button", "aria-label": "Zoom" });
  zoomBtn.addEventListener("click", () => menu(zoomBtn, [{ label: "Fit", run: () => C.zoomFit.run() }, "-",
    ...[50, 100, 200].map((z) => ({ label: `${z}%`, run: () => app.stage.setZoom(z / 100) })), "-", { cmd: C.zoomIn }, { cmd: C.zoomOut }]));
  // "Fit" while the slide is at the zoom Fit gives, a percentage otherwise.
  const isFit = () => {
    const st = app.stage, d = s.doc, r = st.el.getBoundingClientRect(), pad = app.skin?.fitPadding ?? 48;
    return d && Math.abs(st.zoom - Math.min((r.width - pad * 2) / d.width, (r.height - pad * 2) / d.height)) < 1e-3;
  };
  const formatBtn = h("button", { class: "btn sl-textbtn sl-formatbtn", type: "button", title: "Format options", dataset: { open: "format" },
    html: `${icon("sliders-horizontal", 16)}<span class="sl-lab">Format options</span>` });
  formatBtn.addEventListener("click", () => pane.open(pane.current() === "format" ? null : "format"));
  const bgBtn = h("button", { class: "btn sl-textbtn sl-bgbtn", type: "button" }, "Background");
  bgBtn.addEventListener("click", () => backgroundDialog(app));
  const pageGroup = h("div", { class: "sl-group sl-pagegroup" }, bgBtn);
  // text: font ▾ | - size + | B I U A | align ▾ spacing ▾
  const font = add(fontPicker({ get: tget("font_family"), set: (f) => m.set({ font_family: f }, ["text"]), cls: "sl-font", preview: false,
    docFonts: () => [...(s.task?.fonts_used || []), ...s.doc.elements.filter((e) => e.kind === "text").map((e) => e.font_family)],
    open: (btn, get, set, docFonts) => fontMenu(btn, { get, set, docFonts: () => docFonts }) }));
  const size = add(numberField({ label: "", title: "Font size", digits: 2, min: 1, max: 2000, cls: "sl-size", get: tget("font_size"), set: (v) => m.set({ font_size: v }, ["text"]) }));
  const step = (d) => m.set((e) => ({ font_size: Math.max(1, Math.round(e.font_size + d)) }), ["text"]);
  const sizeGroup = h("div", { class: "sl-sizegroup" }, iconButton("minus", "Decrease font size", () => step(-1)), size, iconButton("plus", "Increase font size", () => step(1)));
  const tog = (prop, ic, t) => add(toggle(ic, t, () => m.common(prop, ["text"]) === true, (v) => m.set({ [prop]: v }, ["text"])));
  const bold = tog("bold", "bold", "Bold"), italic = tog("italic", "italic", "Italic"), underline = tog("underline", "underline", "Underline");
  const colorBtn = h("button", { class: "icon-btn sl-textcolor", type: "button", title: "Text color", "aria-label": "Text color" });
  colorBtn.innerHTML = `<span class="a">A</span><span class="bar"></span>`;
  colorBtn.addEventListener("click", () => colorMenu(colorBtn, { get: tget("color"), set: (c) => m.set({ color: c }, ["text"]) }));
  const alignBtn = h("button", { class: "icon-btn sl-align sl-drop", type: "button", title: "Align", "aria-label": "Align" });
  alignBtn.addEventListener("click", () => alignMenu(alignBtn, m));
  const spacingBtn = iconButton("line-spacing", "Line & paragraph spacing", () => spacingMenu(spacingBtn, m), { cls: "sl-spacing" });
  const textGroup = h("div", { class: "sl-group" }, font, h("span", { class: "tb-sep" }), sizeGroup, h("span", { class: "tb-sep" }),
    bold, italic, underline, colorBtn, h("span", { class: "tb-sep" }), alignBtn, spacingBtn);
  // shapes: fill colour, border colour, border weight
  const fillBtn = iconButton("paint-bucket", "Fill color", () => colorMenu(fillBtn, { get: () => m.common("fill", ["shape"]), set: (c) => m.set({ fill: c }, ["shape"]) }), { cls: "sl-fill" });
  const borderBtn = iconButton("pencil-line", "Border color", () => colorMenu(borderBtn, { get: () => m.common("stroke", ["shape"]),
    set: (c) => m.set((e) => ({ stroke: c, stroke_width: e.stroke_width || 1 }), ["shape"]) }), { cls: "sl-border" });
  const weightBtn = iconButton("line-weight", "Border weight", () => menu(weightBtn, [1, 2, 3, 4, 8, 12, 16, 24].map((w) => ({ label: `${w}px`,
    run: () => m.set((e) => ({ stroke_width: w, stroke: e.stroke || "#000000" }), ["shape"]) }))), { cls: "sl-weight" });
  const shapeGroup = h("div", { class: "sl-group" }, fillBtn, borderBtn, weightBtn);
  const toolbar = h("div", { class: "sl-toolbar", role: "toolbar", "aria-label": "Toolbar" },
    menuSearch(app, () => Object.values(C).filter((c) => c.label && !c.hidden?.())), undo, redo, zoomBtn, h("span", { class: "tb-sep" }),
    ...tools, imageBtn, shapeBtn, h("span", { class: "tb-sep" }),
    pageGroup, shapeGroup, textGroup, h("span", { class: "grow" }), formatBtn);

  const strip = thumbStrip(app, "sl-filmstrip", "Slides");
  const stageHost = h("div", { class: "sl-stage stage-host" });
  const center = h("div", { class: "sl-center" }, h("div", { class: "sl-rulers" }, h("div", { class: "sl-ruler-corner" }),
    h("div", { class: "sl-ruler sl-ruler-h" }), h("div", { class: "sl-ruler sl-ruler-v" }), stageHost));
  root.append(header, toolbar, h("div", { class: "sl-body" }, strip.el, center, pane.el));
  return {
    stageHost,
    panels: [fo, images, strip],
    refresh() {
      title.textContent = s.task?.title || "Untitled presentation";
      const els = m.els();
      pageGroup.hidden = els.length > 0;
      textGroup.hidden = !textOnly();
      shapeGroup.hidden = !shapeOnly();
      for (const c of controls) c.set?.();
      colorBtn.querySelector(".bar").style.background = tget("color")() || "#000";
      const al = m.common("align", ["text"]) || "left";
      alignBtn.innerHTML = `${icon(`align-${al}`, 16)}${icon("chevron-down", 10)}`;
      zoomBtn.innerHTML = `<span>${isFit() ? "Fit" : `${Math.round(app.stage.zoom * 100)}%`}</span>${icon("chevron-down", 10)}`;
      [...tools, undo, redo, submit].forEach((b) => b.refresh?.());
      formatBtn.classList.toggle("on", pane.current() === "format");
      strip.refresh();
    },
    reveal(which) { if (which === "assets") pane.open("images"); else if (which === "position") pane.open("format"); },
  };
}

// ================================================================ PowerPoint-like

function powerpoint(app, root) {
  // PowerPoint for the web (checked in the browser): a white header with the file name and Search, the
  // tab row (File, Home, Insert, Design, View, and Shape Format / Picture while something is selected),
  // the simplified one-line ribbon, the thumbnail pane, the Format pane on demand, the status bar.
  const s = app.store;
  const C = app.commands;
  const m = selectionModel(app);
  const fmt = formatPane(app, m);
  const selection = layersPanel(app, { thumbs: false });
  const pane = taskPane(app, "pp-pane", { format: { title: fmt.title, panel: fmt }, selection: { title: "Selection", panel: selection } });
  const textOnly = () => { const e = m.els(); return e.length > 0 && e.every((x) => x.kind === "text"); };
  const tget = (p) => () => (textOnly() ? m.common(p, ["text"]) : undefined);
  const btn = (label, ic, attrs, onclick, { iconOnly = false, caret = false } = {}) => {
    const b = h("button", { class: `pp-btn ${iconOnly ? "pp-iconly" : ""}`, type: "button", title: label, "aria-label": label, ...attrs },
      h("span", { class: "pp-ic", html: icon(ic, 16) }), ...(iconOnly ? [] : [h("span", { class: "pp-lab" }, label)]),
      ...(caret ? [h("span", { class: "pp-caret", html: icon("chevron-down", 12) })] : []));
    b.addEventListener("click", onclick);
    return b;
  };
  const cmdBtn = (label, ic, id, opts) => {
    const b = btn(label, ic, { dataset: { command: id } }, () => C[id].run(), opts);
    b.refresh = () => { if (C[id].enabled) b.disabled = !C[id].enabled(); };
    return b;
  };
  const group = (name, ...kids) => h("div", { class: `pp-group ${name === "Clipboard" ? "pp-optional" : ""}`, "aria-label": name }, ...kids);
  const controls = [];
  const add = (c) => { controls.push(c); return c; };
  // Home: Undo | Clipboard | Font | Paragraph | Drawing | Editing
  const undo = commandButton(C.undo, { cls: "pp-btn pp-iconly" });
  const redo = commandButton(C.redo, { cls: "pp-btn pp-iconly" });
  const font = add(fontCombo({ get: tget("font_family"), set: (f) => m.set({ font_family: f }, ["text"]),
    docFonts: () => [...(s.task?.fonts_used || []), ...s.doc.elements.filter((e) => e.kind === "text").map((e) => e.font_family)] }));
  const size = sizeCombo({ get: tget("font_size"), set: (v) => m.set({ font_size: v }, ["text"]) });
  add(size.field);
  const grow = btn("Increase Font Size", "arrow-up", {}, size.grow, { iconOnly: true });
  const shrink = btn("Decrease Font Size", "arrow-down", {}, size.shrink, { iconOnly: true });
  grow.innerHTML = `<span class="pp-growA">A<sup>^</sup></span>`;
  shrink.innerHTML = `<span class="pp-growA pp-small">A<sup>ˇ</sup></span>`;
  const tog = (prop, ic, t) => add(toggle(ic, t, () => m.common(prop, ["text"]) === true, (v) => m.set({ [prop]: v }, ["text"])));
  const styleBtns = [tog("bold", "bold", "Bold"), tog("italic", "italic", "Italic"), tog("underline", "underline", "Underline")];
  let lastColor = "#c00000";
  const colorBtn = h("button", { class: "icon-btn pp-textcolor", type: "button", title: "Font Color", "aria-label": "Font Color" });
  colorBtn.innerHTML = `<span class="a">A</span><span class="bar"></span>`;
  colorBtn.addEventListener("click", () => m.set({ color: lastColor }, ["text"]));
  const colorDrop = h("button", { class: "pp-split-caret pp-textcolor-drop", type: "button", title: "Font Color options", "aria-label": "Font Color options", html: icon("chevron-down", 12) });
  colorDrop.addEventListener("click", () => colorGallery(colorDrop, { get: tget("color"), automatic: "Automatic",
    set: (c) => { lastColor = c || "#000000"; m.set({ color: lastColor }, ["text"]); } }));
  const moreFont = btn("More Font Options", "more-horizontal", {}, () => menu(moreFont, [
    { label: "Character Spacing", items: [["Very Tight", -3], ["Tight", -1.5], ["Normal", 0], ["Loose", 3], ["Very Loose", 6]].map(([l, v]) =>
      ({ label: l, run: () => m.set({ letter_spacing: v }, ["text"]) })).concat(["-", { label: "More Spacing...", run: () => fontDialog(app, m, "spacing") }]) },
    "-", { label: "Font...", run: () => fontDialog(app, m, "font") }]), { iconOnly: true });
  moreFont.classList.add("pp-morefont");
  const alignBtn = h("button", { class: "pp-btn pp-align", type: "button", title: "Align", "aria-label": "Align" });
  alignBtn.addEventListener("click", () => menu(alignBtn, [["left", "Align Left"], ["center", "Center"], ["right", "Align Right"]].map(([v, label]) =>
    ({ label, icon: `align-${v}`, run: () => m.set({ align: v }, ["text"]) })), { cls: "pp-alignmenu" }));
  const lineBtn = btn("Line Spacing", "line-spacing", {}, () => lineSpacingMenu(lineBtn, app, m), { iconOnly: true, caret: true });
  lineBtn.classList.add("pp-linespacing");
  const textBtns = [grow, shrink, ...styleBtns, colorBtn, colorDrop, moreFont, alignBtn, lineBtn];
  const shapes = btn("Shapes", "shapes", {}, () => menu(shapes, [{ header: "Basic Shapes" }, { cmd: C.toolText, label: "Text Box" }, { cmd: C.toolRect }, { cmd: C.toolEllipse }]), { caret: true });
  const shapeOnly = () => { const e = m.els(); return e.length > 0 && e.every((x) => x.kind === "shape"); };
  const fillBtn = btn("Shape Fill", "paint-bucket", {}, () => colorGallery(fillBtn, { get: () => m.common("fill", ["shape"]), none: "No Fill",
    set: (c) => m.set({ fill: c }, ["shape"]) }), { iconOnly: true, caret: true });
  const outlineBtn = btn("Shape Outline", "pencil-line", {}, () => colorGallery(outlineBtn, { get: () => m.common("stroke", ["shape"]), none: "No Outline",
    set: (c) => m.set((e) => ({ stroke: c, stroke_width: e.stroke_width || 1 }), ["shape"]) }), { iconOnly: true, caret: true });
  const arrangeItems = () => [{ header: "Order Objects" }, { cmd: C.bringFront }, { cmd: C.sendBack }, { cmd: C.bringForward }, { cmd: C.sendBackward }, "-",
    { label: "Align", items: [{ cmd: C.alignLeft }, { cmd: C.alignCenter }, { cmd: C.alignRight }, "-", { cmd: C.alignTop }, { cmd: C.alignMiddle }, { cmd: C.alignBottom }] },
    { label: "Rotate", items: [{ label: "Rotate Right 90°", cmd: C.rotateCw }, { label: "Rotate Left 90°", cmd: C.rotateCcw }, { label: "Flip Vertical", cmd: C.flipV }, { label: "Flip Horizontal", cmd: C.flipH }] },
    "-", { label: "Selection Pane...", run: () => pane.open("selection") }];
  const arrange = btn("Arrange", "bring-to-front", {}, () => menu(arrange, arrangeItems()), { caret: true });
  const select = btn("Select", "mouse-pointer-2", { dataset: { open: "selection" } }, () => menu(select, [{ cmd: C.selectAll }, { label: "Selection Pane...", run: () => pane.open("selection") }]), { caret: true });
  select.classList.add("pp-select");
  // Insert > Pictures opens PowerPoint's "Insert Picture From" menu; This Device... is the computer's file window.
  const picturesBtn = btn("Pictures", "image", { dataset: { open: "pictures" } }, () => menu(picturesBtn, [{ header: "Insert Picture From" }, { label: "This Device...", cmd: C.placeImage }]), { caret: true });
  const formatBg = btn("Format Background", "paint-bucket", { dataset: { open: "background" } }, () => { s.select([]); pane.open("format"); });
  const formatPaneBtn = btn("Format Pane", "sliders-horizontal", { dataset: { open: "format" } }, () => pane.open(pane.current() === "format" ? null : "format"));
  const ctxH = numberField({ label: "", title: "Shape Height", name: "ctx_height", digits: 2, min: 1, unit: "px", iconName: "arrow-down",
    get: () => (m.els().every((e) => e.kind !== "text") ? m.common("height") : undefined), set: (v) => m.set({ height: v }), cls: "pp-ctxnum" });
  const ctxW = numberField({ label: "", title: "Shape Width", name: "ctx_width", digits: 2, min: 1, unit: "px", iconName: "arrow-right", get: () => m.common("width"),
    set: (v) => m.set({ width: v }), cls: "pp-ctxnum" });
  const ribbons = {
    home: h("div", { class: "pp-ribbon-body" },
      group("Undo", undo, redo),
      group("Clipboard", cmdBtn("Paste", "clipboard-paste", "paste", { iconOnly: true }), cmdBtn("Cut", "scissors", "cut", { iconOnly: true }),
        cmdBtn("Copy", "copy", "copy", { iconOnly: true }), cmdBtn("Delete", "trash-2", "delete", { iconOnly: true })),
      group("Font", font, size.field, grow, shrink, ...styleBtns, h("span", { class: "pp-split" }, colorBtn, colorDrop), moreFont),
      group("Paragraph", alignBtn, lineBtn),
      group("Drawing", shapes, fillBtn, outlineBtn, arrange),
      group("Editing", select)),
    insert: h("div", { class: "pp-ribbon-body" },
      group("Images", picturesBtn),
      group("Text", cmdBtn("Text Box", "type", "toolText")),
      group("Shapes", btn("Shapes", "shapes", {}, (e) => menu(e.currentTarget, [{ cmd: C.toolRect }, { cmd: C.toolEllipse }]), { caret: true }))),
    design: h("div", { class: "pp-ribbon-body" }, group("Customize", formatBg)),
    view: h("div", { class: "pp-ribbon-body" },
      group("Zoom", cmdBtn("Zoom In", "zoom-in", "zoomIn"), cmdBtn("Zoom Out", "zoom-out", "zoomOut"), cmdBtn("Fit to Window", "maximize", "zoomFit"))),
    format: h("div", { class: "pp-ribbon-body" },
      group("Shape Styles", btn("Shape Fill", "paint-bucket", {}, (e) => colorGallery(e.currentTarget, { get: () => m.common("fill", ["shape"]), none: "No Fill",
        set: (c) => m.set({ fill: c }, ["shape"]) }), { caret: true })),
      group("Arrange", btn("Arrange", "bring-to-front", {}, (e) => menu(e.currentTarget, arrangeItems()), { caret: true }),
        btn("Rotate", "rotate-cw", {}, (e) => menu(e.currentTarget, [{ label: "Rotate Right 90°", cmd: C.rotateCw }, { label: "Rotate Left 90°", cmd: C.rotateCcw },
          { label: "Flip Vertical", cmd: C.flipV }, { label: "Flip Horizontal", cmd: C.flipH }]), { caret: true })),
      group("Size", ctxH, ctxW),
      group("Format", formatPaneBtn)),
  };
  const shapeStyles = ribbons.format.querySelector('[aria-label="Shape Styles"]');
  const ribbonHost = h("div", { class: "pp-ribbon", role: "toolbar", "aria-label": "Ribbon" });
  const tabBar = h("div", { class: "pp-tabs", role: "tablist" });
  const ctxTab = h("button", { class: "pp-tab pp-ctxtab", type: "button", role: "tab", dataset: { tab: "format" } }, "Shape Format");
  let currentTab = "home";
  const setTab = (id) => {
    if (id === "file") {
      menu(tabBar.querySelector('[data-tab="file"]'), [...(app.showCode ? [{ cmd: C.html }, "-"] : []),
        { label: "Export", items: [{ label: "Download as PNG", cmd: C.submit }] }, "-", { cmd: C.shortcuts }]);
      return;
    }
    currentTab = id;
    ribbonHost.textContent = "";
    ribbonHost.append(ribbons[id]);
    tabBar.querySelectorAll(".pp-tab").forEach((b) => b.classList.toggle("on", b.dataset.tab === id));
    ribbonHost.querySelectorAll("[data-command]").forEach((b) => b.refresh?.());
  };
  for (const [id, label] of [["file", "File"], ["home", "Home"], ["insert", "Insert"], ["design", "Design"], ["view", "View"]]) {
    const b = h("button", { class: "pp-tab", type: "button", role: "tab", dataset: { tab: id } }, label);
    b.addEventListener("click", () => setTab(id));
    tabBar.append(b);
  }
  ctxTab.addEventListener("click", () => setTab("format"));
  const submit = submitButton(app, "pp-share");
  tabBar.append(ctxTab, h("span", { class: "grow" }), submit);
  const fileName = h("span", { class: "pp-filename" }, "Presentation1");
  const titlebar = h("header", { class: "pp-titlebar" },
    h("span", { class: "brand-mark mark-present", html: icon("rectangle-horizontal", 14) }),
    fileName, h("span", { class: "pp-saved", title: "Saved to OneDrive", html: icon("cloud-check", 15) }), h("span", { class: "grow" }),
    commandSearch(app, () => Object.values(C).filter((c) => c.label && !c.hidden?.())), h("span", { class: "grow" }), avatar(app));
  const strip = thumbStrip(app, "pp-thumbs", "Slides");
  const stageHost = h("div", { class: "pp-stage stage-host" });
  const zoomRange = h("input", { type: "range", min: "10", max: "400", value: "100", class: "pp-zoom-range", "aria-label": "Zoom" });
  zoomRange.addEventListener("input", () => app.stage.setZoom(zoomRange.value / 100));
  app.on("zoom", (z) => { zoomRange.value = String(Math.round(z * 100)); });
  const statusbar = h("footer", { class: "pp-status" }, h("span", {}, "Slide 1 of 1"), h("span", {}, "English (U.S.)"), h("span", { class: "grow" }),
    iconButton("minus", "Zoom Out", () => C.zoomOut.run(), { cls: "pp-sbtn" }), zoomRange, iconButton("plus", "Zoom In", () => C.zoomIn.run(), { cls: "pp-sbtn" }),
    zoomLabel(app), iconButton("maximize", "Fit slide to current window", () => C.zoomFit.run(), { cls: "pp-sbtn" }));
  root.append(titlebar, tabBar, ribbonHost, h("div", { class: "pp-body" }, strip.el, h("div", { class: "pp-center" }, stageHost), pane.el), statusbar);
  setTab("home");
  const refreshables = [undo, redo, submit];
  return {
    stageHost,
    panels: [fmt, selection, strip],
    refresh() {
      fileName.textContent = s.task?.title ? s.task.title.slice(0, 80) : "Presentation1";
      const els = m.els(), t = textOnly();
      // The contextual tab: Shape Format for shapes and text boxes, Picture for pictures.
      ctxTab.hidden = !els.length;
      ctxTab.textContent = els.length && els.every((e) => e.kind === "image") ? "Picture" : "Shape Format";
      shapeStyles.hidden = !shapeOnly();
      if (!els.length && currentTab === "format") setTab("home");
      for (const c of controls) c.set?.();
      textBtns.forEach((b) => { b.disabled = !t; });
      fillBtn.disabled = outlineBtn.disabled = !shapeOnly();
      colorBtn.querySelector(".bar").style.background = lastColor;
      const al = m.common("align", ["text"]) || "left";
      alignBtn.innerHTML = `<span class="pp-ic">${icon(`align-${al}`, 16)}</span><span class="pp-caret">${icon("chevron-down", 12)}</span>`;
      ctxH.set(); ctxW.set();
      for (const b of ribbonHost.querySelectorAll("[data-command]")) b.refresh?.();
      refreshables.forEach((b) => b.refresh?.());
      ribbonHost.querySelectorAll("[data-open]").forEach((b) => b.classList.toggle("on", pane.current() === b.dataset.open));
      pane.retitle();
      strip.refresh();
    },
    reveal(which) {
      if (which === "assets") app.openFiles();
      else if (which === "position") pane.open("format");
    },
    currentTab: () => currentTab,
  };
}

// ================================================================ Inkscape-like

function inkscape(app, root) {
  // Inkscape 1.3 / 1.4's window (per the Inkscape Beginners' Guide): menu bar, the tool controls bar
  // under it, the toolbox on the left, rulers round the canvas, docked dialogs as closable tabs, the
  // commands bar down the right edge, the palette, and the status bar (fill / stroke, O:, layer,
  // message, cursor X / Y, Z:). Geometry lives in the Selector's controls, text styling in the Text
  // tool's controls and Text and Font, colour in Fill and Stroke, the page colour in Document Properties.
  const s = app.store;
  const C = app.commands;
  const m = selectionModel(app);
  const layers = layersPanel(app, { thumbs: false });
  const fs = fillAndStroke(app, m);
  const tf = textAndFont(app, m);
  const tr = transformDialog(app, m);
  const dp = documentProperties(app);
  const ex = exportDialog(app);
  const dialogs = {
    fill: { title: "Fill and Stroke", icon: "paint-bucket", panel: fs },
    text: { title: "Text and Font", icon: "type", panel: tf },
    layers: { title: "Layers and Objects", icon: "layers", panel: layers },
    transform: { title: "Transform", icon: "move", panel: tr },
    document: { title: "Document Properties", icon: "file", panel: dp },
    export: { title: "Export", icon: "download", panel: ex },
  };
  const docked = dock(dialogs, { onChange: () => requestAnimationFrame(() => app.stage?.fit?.()) });
  const openDialog = (id, sub) => { docked.show(id); if (sub) dialogs[id].panel.setTab?.(sub); };
  // Dialogs have their own commands (and Inkscape's shortcuts), so menus, the commands bar and keys agree.
  const dlg = (id, label, key, ic) => ({ id: `dlg_${id}`, label, key, icon: ic, run: () => openDialog(id) });
  const D = {
    fill: dlg("fill", "Fill and Stroke…", "Mod+Shift+F", "paint-bucket"), text: dlg("text", "Text and Font…", "Mod+Shift+T", "type"),
    layers: dlg("layers", "Layers and Objects…", "Mod+Shift+L", "layers"), transform: dlg("transform", "Transform…", "Mod+Shift+M", "move"),
    document: dlg("document", "Document Properties…", "Mod+Shift+D", "file"), export: dlg("export", "Export…", "Mod+Shift+E", "download"),
  };
  for (const c of Object.values(D)) C[c.id] = c;

  const menus = {
    File: [{ label: "Import…", cmd: C.placeImage }, ...(app.showCode ? [{ cmd: C.html }] : []), "-",
      { cmd: D.document }, "-", { cmd: D.export }],
    Edit: [{ cmd: C.undo }, { cmd: C.redo }, "-", { cmd: C.cut }, { cmd: C.copy }, { cmd: C.paste }, "-", { cmd: C.duplicate }, { cmd: C.delete }, "-",
      { label: "Select All", cmd: C.selectAll }, { label: "Deselect", cmd: C.deselect }],
    View: [{ header: "Zoom" }, { cmd: C.zoomIn }, { cmd: C.zoomOut }, { label: "Zoom 1:1", cmd: C.zoom100 }, { label: "Zoom Selection", cmd: C.zoomSelection },
      { label: "Zoom Page", cmd: C.zoomFit }],
    Layer: [{ cmd: D.layers }],
    Object: [{ cmd: D.fill }, { cmd: D.transform }, "-", { label: "Raise to Top", cmd: C.bringFront }, { label: "Raise", cmd: C.bringForward },
      { label: "Lower", cmd: C.sendBackward }, { label: "Lower to Bottom", cmd: C.sendBack }, "-",
      { label: "Rotate 90° CW", cmd: C.rotateCw }, { label: "Rotate 90° CCW", cmd: C.rotateCcw }, { label: "Flip Horizontal", cmd: C.flipH },
      { label: "Flip Vertical", cmd: C.flipV }, "-", { header: "Align" }, { cmd: C.alignLeft }, { cmd: C.alignCenter }, { cmd: C.alignRight },
      { cmd: C.alignTop }, { cmd: C.alignMiddle }, { cmd: C.alignBottom }, "-", { label: "Lock / Unlock", cmd: C.lock }, { label: "Hide / Unhide", cmd: C.hide }],
    Text: [{ cmd: D.text }],
    Help: [{ label: "Keys and Mouse Reference", cmd: C.shortcuts }],
  };
  const menubar = h("div", { class: "ik-menubar", role: "menubar" }, h("span", { class: "brand-mark mark-vector", html: icon("pen-tool", 13) }));
  for (const [name, items] of Object.entries(menus)) {
    const b = h("button", { class: "ik-menu", type: "button", role: "menuitem" }, name);
    b.addEventListener("click", () => menu(b, items));
    menubar.append(b);
  }

  // Tool controls bar. Selector: select all / deselect, rotate 90, flip, z-order, then X Y W (lock) H in px.
  const cmdBtn = (cmd, title) => { const b = commandButton(cmd, { cls: "ik-cmd" }); if (title) b.title = title; return b; };
  let lockRatio = false;
  const geom = (prop, label) => spin({ label, title: label, name: prop, digits: 3, cls: "ik-num", min: prop === "width" || prop === "height" ? 1 : -1e5, max: 1e5,
    get: () => { const e = m.els(); if (!e.length) return undefined; if (prop === "height" && !e.every((x) => x.kind !== "text")) return undefined; return m.common(prop); },
    set: (v) => m.set((e) => {
      if (lockRatio && (prop === "width" || prop === "height") && e.kind !== "text") {
        const r = e.width / e.height;
        return prop === "width" ? { width: v, height: v / r } : { height: v, width: v * r };
      }
      return { [prop]: v };
    }) });
  const gx = geom("x", "X"), gy = geom("y", "Y"), gw = geom("width", "W"), gh = geom("height", "H");
  const lock = toggle("lock", "When locked, change width and height by the same proportion", () => lockRatio, (v) => { lockRatio = v; lock.set(); }, { cls: "ik-cmd ik-lockratio" });
  const selOpts = h("div", { class: "ik-optgroup ik-selopts inspector" },
    cmdBtn(C.selectAll, "Select all objects"), cmdBtn(C.deselect, "Deselect any selected objects"), h("span", { class: "tb-sep" }),
    cmdBtn(C.rotateCcw, "Rotate 90° counter-clockwise"), cmdBtn(C.rotateCw, "Rotate 90° clockwise"), cmdBtn(C.flipH, "Flip horizontally"),
    cmdBtn(C.flipV, "Flip vertically"), h("span", { class: "tb-sep" }),
    cmdBtn(C.bringFront, "Raise selection to top"), cmdBtn(C.bringForward, "Raise selection one step"), cmdBtn(C.sendBackward, "Lower selection one step"),
    cmdBtn(C.sendBack, "Lower selection to bottom"), h("span", { class: "tb-sep" }), gx, gy, gw, lock, gh, h("span", { class: "ik-unit" }, "px"));
  // Text tool: font family, style, size (px), line height (lines), alignment, letter spacing.
  const tc = textControls(app, "ik", { steppers: false, colorButton: false, toggles: false });
  const FACES = [["Regular", "regular"], ["Bold", "bold"], ["Italic", "italic"], ["Bold Italic", "bolditalic"]];
  const styleSel = h("button", { class: "ik-style ik-drop", type: "button", "aria-label": "Font style" });
  styleSel.addEventListener("click", () => {
    const fam = m.common("font_family", ["text"]);
    menu(styleSel, FACES.filter(([, f]) => f === "regular" || !fam || hasStyle(fam, f)).map(([label]) => ({
      label, run: () => m.set({ bold: label.includes("Bold"), italic: label.includes("Italic") }, ["text"]) })), { cls: "ik-style-menu" });
  });
  const lineH = spin({ label: "", title: "Spacing between baselines", name: "line_height", iconName: "baseline", digits: 3, step: 0.1, min: 0.1, max: 10,
    get: () => (tc.textOnly() ? m.common("line_height", ["text"]) : undefined), set: (v) => m.set({ line_height: v }, ["text"]) });
  const letterS = spin({ label: "", title: "Spacing between letters (px)", name: "letter_spacing", iconName: "letter-spacing", digits: 2, step: 0.1, min: -100, max: 500,
    get: () => (tc.textOnly() ? m.common("letter_spacing", ["text"]) : undefined), set: (v) => m.set({ letter_spacing: v }, ["text"]) });
  const alignDd = alignDropdown(m);
  const textOpts = h("div", { class: "ik-optgroup ik-textopts" }, tc.font, styleSel, tc.size, h("span", { class: "ik-unit" }, "px"),
    lineH, h("span", { class: "ik-unit" }, "lines"), alignDd, h("span", { class: "tb-sep" }), letterS);
  const emptyOpts = h("div", { class: "ik-optgroup ik-muted" }, cmdBtn(C.selectAll, "Select all objects"), h("span", { class: "tb-sep" }),
    "No objects selected");
  const controls = h("div", { class: "ik-controls" }, selOpts, textOpts, emptyOpts);

  const toolBtns = [];
  const live = (id, tool, title) => {
    const b = commandButton(C[id], { cls: "ik-tool" });
    b.title = title;
    b.setAttribute("aria-label", title);
    b.refresh = () => b.classList.toggle("on", s.tool === tool);
    toolBtns.push(b);
    return b;
  };
  const toolbox = h("div", { class: "ik-tools", role: "toolbar", "aria-label": "Tools" },
    live("toolMove", "move", "Selector (S)"), iconButton("zoom-in", "Zoom (Z)", () => C.zoomIn.run(), { cls: "ik-tool" }), h("span", { class: "ik-tsep" }),
    live("toolRect", "rect", "Rectangle (R)"), live("toolEllipse", "ellipse", "Ellipse / Arc (E)"), h("span", { class: "ik-tsep" }),
    live("toolText", "text", "Text (T)"));
  const stageHost = h("div", { class: "ik-stage stage-host" });
  const rulerTop = h("div", { class: "ik-ruler ik-ruler-h" });
  const rulerLeft = h("div", { class: "ik-ruler ik-ruler-v" });
  const canvasWrap = h("div", { class: "ik-canvas" }, h("div", { class: "ik-ruler-corner" }), rulerTop, rulerLeft, stageHost);

  // Commands bar, down the right edge: import / export, undo / redo, clipboard, zoom, duplicate, the dialogs.
  const importBtn = iconButton("file-down", "Import a bitmap or SVG image into this document (Ctrl+I)", () => C.placeImage.run(), { cls: "ik-cmd" });
  importBtn.dataset.open = "files";
  const cbtn = (cmd, title) => { const b = commandButton(cmd, { cls: "ik-cmd" }); b.title = title; return b; };
  const commands = h("div", { class: "ik-commands", role: "toolbar", "aria-label": "Commands" },
    importBtn, cbtn(D.export, "Export this document (Shift+Ctrl+E)"), h("span", { class: "ik-csep" }),
    cbtn(C.undo, "Undo"), cbtn(C.redo, "Redo"), h("span", { class: "ik-csep" }),
    cbtn(C.copy, "Copy"), cbtn(C.cut, "Cut"), cbtn(C.paste, "Paste"), h("span", { class: "ik-csep" }),
    cbtn(C.zoomSelection, "Zoom Selection"), cbtn(C.zoomFit, "Zoom Page"), cbtn(C.zoom100, "Zoom 1:1"), h("span", { class: "ik-csep" }),
    cbtn(C.duplicate, "Duplicate"), cbtn(C.delete, "Delete"), h("span", { class: "ik-csep" }),
    cbtn(D.fill, "Fill and Stroke (Shift+Ctrl+F)"), cbtn(D.text, "Text and Font (Shift+Ctrl+T)"), cbtn(D.layers, "Layers and Objects (Shift+Ctrl+L)"),
    cbtn(D.transform, "Transform (Shift+Ctrl+M)"), cbtn(D.document, "Document Properties (Shift+Ctrl+D)"));

  // Palette: Inkscape's default palette. Click fills the selection, Shift+click strokes it (shapes);
  // X is no paint; with nothing selected a colour becomes new objects' colour.
  const palette = paletteBar({
    onPick: (c, stroke) => {
      const els = m.els();
      if (!els.length) { app.setFg(c); return; }
      if (stroke) { if (els.every((e) => e.kind === "shape")) m.set((e) => ({ stroke: c, stroke_width: e.stroke_width || 4 }), ["shape"]); return; }
      if (els.every((e) => e.kind === "text")) m.set({ color: c }, ["text"]);
      else if (els.every((e) => e.kind === "shape")) m.set({ fill: c }, ["shape"]);
    },
    onNone: (stroke) => {
      const els = m.els();
      if (els.length && els.every((e) => e.kind === "shape")) m.set(stroke ? { stroke: null, stroke_width: 0 } : { fill: "none" }, ["shape"]);
    },
  });

  // Status bar: Fill / Stroke (click opens Fill and Stroke), O: opacity, layer, message, cursor X / Y, Z: zoom.
  const fillSw = h("span", { class: "ik-ind-sw" }), strokeSw = h("span", { class: "ik-ind-sw" });
  const fillTx = h("span", { class: "ik-ind-tx" }), strokeTx = h("span", { class: "ik-ind-tx" });
  const indicator = h("button", { class: "ik-indicator", type: "button", title: "Fill and stroke of the selection; click to edit", "aria-label": "Fill and stroke" },
    h("span", { class: "ik-ind-row" }, h("span", { class: "ik-ind-k" }, "Fill:"), fillSw, fillTx),
    h("span", { class: "ik-ind-row" }, h("span", { class: "ik-ind-k" }, "Stroke:"), strokeSw, strokeTx));
  indicator.addEventListener("click", () => openDialog("fill", "fill"));
  const opac = spin({ label: "O:", title: "Opacity (%)", name: "opacity", digits: 0, min: 0, max: 100, step: 5, cls: "ik-opac",
    get: () => { const v = m.common("opacity"); return v === null || v === undefined ? v : v * 100; }, set: (v) => m.set({ opacity: v / 100 }) });
  const status = h("span", { class: "ik-msg" });
  const cursor = h("span", { class: "ik-cursor" }, h("span", {}, "X: –"), h("span", {}, "Y: –"));
  const statusbar = h("footer", { class: "ik-status" }, indicator, opac, h("span", { class: "ik-layer" }, "Layer 1"), status,
    h("span", { class: "grow" }), cursor, zoomSpin(app));

  root.append(menubar, controls, h("div", { class: "ik-body" }, toolbox, canvasWrap, docked.el, commands), palette, statusbar);
  stageHost.addEventListener("pointermove", (e) => {
    const [x, y] = app.stage.toCanvas(e.clientX, e.clientY);
    cursor.firstChild.textContent = `X: ${fmt(x, 2)}`;
    cursor.lastChild.textContent = `Y: ${fmt(y, 2)}`;
  });
  openDialog("layers");
  openDialog("fill");
  const geo = [gx, gy, gw, gh, lineH, letterS, opac];
  // Inkscape asks how to import every bitmap before it lands on the canvas. Only what this task can
  // honour is offered: images are always embedded at their file resolution, so those rows are read-only;
  // "Don't ask again" works and skips the question for the rest of the session.
  let askAgain = true;
  const row = (label, value) => [h("span", { class: "ik-import-label" }, label), h("span", { class: "ik-import-value" }, value)];
  const askImport = (assets) => new Promise((resolve) => {
    if (!askAgain) { resolve(true); return; }
    const a = assets[0], kind = a && (a.kind === "photo" || a.kind === "background") ? "jpeg" : "png";
    const never = h("input", { type: "checkbox", name: "ik-dont-ask" });
    let ok = false;
    modal(`${kind} bitmap image import`, h("div", { class: "ik-import-grid" },
      ...row("Image Import Type:", "Embed"), ...row("Image DPI:", "From file"), ...row("Image rendering mode:", "None (auto)"),
      h("span", {}), h("label", {}, never, "Don't ask again")), {
      cls: "ik-import",
      actions: [{ label: "Cancel" }, { label: "OK", primary: true, run: () => { ok = true; if (never.checked) askAgain = false; } }],
      onClose: () => resolve(ok),
    });
  });


  return {
    stageHost,
    panels: [layers, docked],
    placeAssets: async (assets, point) => { if (await askImport(assets)) app.placeAt(assets, point); },
    fileOptions: () => ({ title: "Select file to import" }),
    refresh() {
      const els = m.els();
      const textSel = tc.textOnly();
      const tool = s.tool;
      // Inkscape: the Text tool shows text controls; the Selector shows X / Y / W / H for whatever is selected.
      textOpts.hidden = tool !== "text";
      selOpts.hidden = tool === "text" || !els.length;
      emptyOpts.hidden = !(selOpts.hidden && textOpts.hidden);
      geo.forEach((f) => f.set());
      lock.set();
      alignDd.set();
      tc.refresh();
      const b = m.common("bold", ["text"]), it = m.common("italic", ["text"]);
      const fam = textSel ? m.common("font_family", ["text"]) : null;
      const want = !textSel ? "" : b === null || it === null ? "Mixed" : b && it ? "Bold Italic" : b ? "Bold" : it ? "Italic" : fam ? "Regular" : "";
      styleSel.innerHTML = `<span class="ik-style-name">${want}</span>${icon("chevron-down", 10)}`;
      styleSel.disabled = !textSel;
      toolBtns.forEach((x) => x.refresh());
      commands.querySelectorAll("[data-command]").forEach((x) => x.refresh?.());
      controls.querySelectorAll("[data-command]").forEach((x) => x.refresh?.());
      const one = els.length === 1 ? els[0] : null;
      const fillOf = (e) => (e.kind === "text" ? e.color : e.kind === "shape" ? e.fill : null);
      const f = one ? fillOf(one) : els.length ? null : app.fgColor;
      const st = one && one.kind === "shape" && one.stroke && one.stroke_width ? one.stroke : null;
      fillSw.style.background = f && f !== "none" ? f : "transparent";
      fillTx.textContent = !els.length ? "" : f && f !== "none" ? "" : one ? "None" : "Different";
      strokeSw.style.background = st || "transparent";
      strokeTx.textContent = !els.length ? "" : st ? "" : "None";
      indicator.classList.toggle("none", !els.length);
      status.textContent = !els.length ? "No objects selected. Click, Shift+click, Alt+scroll mouse on top of objects, or drag around objects to select."
        : els.length === 1 ? `${one.kind === "text" ? "Text" : one.kind === "image" ? "Image" : "Rectangle"} in layer Layer 1. Click selection to toggle scale/rotation handles.`
          : `${els.length} objects selected in layer Layer 1.`;
    },
    reveal(which) {
      if (which === "assets") app.openFiles();
      else if (which === "position") openDialog("transform");
    },
  };
}

export const SKINS = {
  figma: { name: "figma", build: figma, frameLabel: "Frame 1", fitPadding: 56, rotateHandle: false, alwaysShowSize: true, floatingToolbar: false, fill: "#d9d9d9" },
  canva: { name: "canva", build: canva, fitPadding: 48, rotateHandle: true, rotateIcon: icon("rotate-cw", 14), alwaysShowSize: false, floatingToolbar: true, fill: "#c1c1c1" },
  photoshop: { name: "photoshop", build: photoshop, fitPadding: 36, rotateHandle: false, alwaysShowSize: false, floatingToolbar: false, fill: "#5b8def" },
  slides: { name: "slides", build: slides, fitPadding: 32, rotateHandle: true, rotateIcon: "", alwaysShowSize: false, floatingToolbar: false, fill: "#9aa0a6" },
  powerpoint: { name: "powerpoint", build: powerpoint, fitPadding: 32, rotateHandle: true, rotateIcon: icon("rotate-cw", 12), alwaysShowSize: false, floatingToolbar: false, fill: "#4472c4" },
  inkscape: { name: "inkscape", build: inkscape, fitPadding: 40, rotateHandle: false, alwaysShowSize: false, floatingToolbar: false, fill: "#c3c3c3" },
};
