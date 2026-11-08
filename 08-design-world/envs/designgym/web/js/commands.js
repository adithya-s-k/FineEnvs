// Every user-facing action is a command. Menus, toolbars, context menus and
// keyboard shortcuts all point at the same registry, so the Figma, Canva and
// Photoshop skins differ only in where a command lives, never in what it does.

import { unionBounds } from "./util.js";

export function createCommands(app) {
  const s = app.store;
  const sel = () => s.selected();
  const one = () => (s.selection.length === 1 ? s.find(s.selection[0]) : null);
  const hasSel = () => s.selection.length > 0;
  const unlocked = () => sel().filter((e) => !e.locked);
  let clipboard = [];

  const addFromProps = async (el, dx = 20, dy = 20) => {
    const base = { x: el.x + dx, y: el.y + dy, angle: el.angle, opacity: el.opacity };
    let res;
    if (el.kind === "image") {
      res = await s.op("add_image", { asset_id: el.asset_id, width: el.width, height: el.height, ...base });
    } else if (el.kind === "shape") {
      res = await s.op("add_shape", {
        shape: el.shape, width: el.width, height: el.height, fill: el.fill || "none",
        stroke: el.stroke, stroke_width: el.stroke_width, radius: el.radius, ...base,
      });
    } else {
      res = await s.op("add_text", {
        text: el.text, width: el.auto_width ? undefined : el.width, font_family: el.font_family,
        font_size: el.font_size, color: el.color, align: el.align, bold: el.bold,
        italic: el.italic, underline: el.underline, uppercase: el.uppercase,
        line_height: el.line_height, letter_spacing: el.letter_spacing, runs: el.runs, ...base,
      });
    }
    return res?.element_id;
  };

  const reorder = (to) => async () => {
    const ids = [...s.selection].sort((a, b) => s.index(a) - s.index(b));
    const list = to === "front" || to === "backward" ? ids : ids.reverse();
    for (const id of list) await s.op("reorder_element", { element_id: id, to });
  };

  const align = (alignment) => async () => {
    const ids = unlocked().map((e) => e.id);
    if (!ids.length) return;
    await s.op("align_elements", {
      element_ids: ids, alignment, relative_to: ids.length > 1 ? "selection" : "canvas",
    });
  };

  // Figma: ] / [ send to the very front / back, with Mod one step; elsewhere the reverse.
  const fg = app.skin.name === "figma";
  const ik = app.skin.name === "inkscape";  // Inkscape: Home / End / PageUp / PageDown for z-order, H / V flip, S selector
  const texts = () => sel().filter((e) => e.kind === "text" && !e.locked);
  const textStyle = (prop) => () => {
    const t = texts();
    const range = app.stage?.textRange;
    if (range && t.length === 1 && t[0].id === range.id) return app.stage.applyRangeStyle(t[0], { [prop]: !app.stage.rangeHas(prop) });
    const on = !t.every((e) => e[prop]);
    return s.updateMany(t.map((e) => [e.id, { [prop]: on }]));
  };

  const cmds = {
    undo: { label: "Undo", key: "Mod+Z", icon: "undo-2", enabled: () => s.canUndo, run: () => s.undo() },
    redo: { label: "Redo", key: "Mod+Shift+Z", icon: "redo-2", enabled: () => s.canRedo, run: () => s.redo() },
    delete: {
      label: "Delete", key: "Delete", icon: "trash-2", enabled: hasSel,
      run: async () => {
        const ids = unlocked().map((e) => e.id);
        s.select([]);
        for (const id of ids) await s.op("delete_element", { element_id: id });
      },
    },
    duplicate: {
      label: "Duplicate", key: "Mod+D", icon: "copy", enabled: hasSel,
      run: async () => {
        const out = [];
        for (const el of sel()) {
          const r = await s.op("duplicate_element", { element_id: el.id, dx: 20, dy: 20 });
          if (r?.element_id) out.push(r.element_id);
        }
        s.select(out);
      },
    },
    copy: {
      label: "Copy", key: "Mod+C", icon: "copy", enabled: hasSel,
      run: () => { clipboard = sel().map((e) => JSON.parse(JSON.stringify(e))); app.toast(`Copied ${clipboard.length}`); },
    },
    cut: {
      label: "Cut", key: "Mod+X", icon: "scissors", enabled: hasSel,
      run: async () => { cmds.copy.run(); await cmds.delete.run(); },
    },
    paste: {
      label: "Paste", key: "Mod+V", icon: "clipboard-paste", enabled: () => clipboard.length > 0,
      run: async () => {
        const out = [];
        for (const el of clipboard) {
          const id = await addFromProps(el);
          if (id) out.push(id);
        }
        clipboard = clipboard.map((e) => ({ ...e, x: e.x + 20, y: e.y + 20 }));
        s.select(out);
      },
    },
    selectAll: {
      label: "Select all", key: "Mod+A", enabled: () => !!s.doc?.elements.length,
      run: () => s.select(s.doc.elements.filter((e) => e.visible !== false && !e.locked).map((e) => e.id)),
    },
    deselect: { label: "Deselect", key: "Escape", enabled: hasSel, run: () => s.select([]) },
    bold: { label: "Bold", key: "Mod+B", icon: "bold", enabled: () => texts().length > 0, run: textStyle("bold") },
    italic: { label: "Italic", key: app.skin.name === "inkscape" ? "" : "Mod+I", icon: "italic", enabled: () => texts().length > 0, run: textStyle("italic") },
    underline: { label: "Underline", key: "Mod+U", icon: "underline", enabled: () => texts().length > 0, run: textStyle("underline") },
    rotateCw: { label: "Rotate 90° clockwise", icon: "rotate-cw", enabled: hasSel,
      run: () => s.updateMany(unlocked().map((e) => [e.id, { angle: (e.angle || 0) + 90 }])) },
    rotateCcw: { label: "Rotate 90° counter-clockwise", icon: "rotate-ccw", enabled: hasSel,
      run: () => s.updateMany(unlocked().map((e) => [e.id, { angle: (e.angle || 0) - 90 }])) },
    bringFront: { label: "Bring to front", key: ik ? "Home" : fg ? "]" : "Mod+]", icon: "bring-to-front", enabled: hasSel, run: reorder("front") },
    bringForward: { label: "Bring forward", key: ik ? "PageUp" : fg ? "Mod+]" : "]", icon: "arrow-up", enabled: hasSel, run: reorder("forward") },
    sendBackward: { label: "Send backward", key: ik ? "PageDown" : fg ? "Mod+[" : "[", icon: "arrow-down", enabled: hasSel, run: reorder("backward") },
    sendBack: { label: "Send to back", key: ik ? "End" : fg ? "[" : "Mod+[", icon: "send-to-back", enabled: hasSel, run: reorder("back") },
    alignLeft: { label: "Align left", icon: "align-start-vertical", enabled: hasSel, run: align("left") },
    alignCenter: { label: "Align horizontal centers", icon: "align-center-vertical", enabled: hasSel, run: align("center") },
    alignRight: { label: "Align right", icon: "align-end-vertical", enabled: hasSel, run: align("right") },
    alignTop: { label: "Align top", icon: "align-start-horizontal", enabled: hasSel, run: align("top") },
    alignMiddle: { label: "Align vertical centers", icon: "align-center-horizontal", enabled: hasSel, run: align("middle") },
    alignBottom: { label: "Align bottom", icon: "align-end-horizontal", enabled: hasSel, run: align("bottom") },
    distributeH: {
      label: "Distribute horizontally", icon: "align-horizontal-distribute-center",
      enabled: () => s.selection.length > 2,
      run: () => s.op("distribute_elements", { element_ids: s.selection, axis: "horizontal" }),
    },
    distributeV: {
      label: "Distribute vertically", icon: "align-vertical-distribute-center",
      enabled: () => s.selection.length > 2,
      run: () => s.op("distribute_elements", { element_ids: s.selection, axis: "vertical" }),
    },
    flipH: {
      label: "Flip horizontal", key: ik ? "H" : "Shift+H", icon: "flip-horizontal-2", enabled: hasSel,
      run: () => s.updateMany(unlocked().map((e) => [e.id, { flip_x: !e.flip_x }])),
    },
    flipV: {
      label: "Flip vertical", key: ik ? "V" : "Shift+V", icon: "flip-vertical-2", enabled: hasSel,
      run: () => s.updateMany(unlocked().map((e) => [e.id, { flip_y: !e.flip_y }])),
    },
    lock: {
      label: "Lock / unlock", key: "Mod+Shift+L", icon: "lock", enabled: hasSel,
      run: () => s.updateMany(sel().map((e) => [e.id, { locked: !e.locked }])),
    },
    hide: {
      label: "Show / hide", key: "Mod+Shift+H", icon: "eye-off", enabled: hasSel,
      run: () => s.updateMany(sel().map((e) => [e.id, { visible: e.visible === false }])),
    },
    freeTransform: { label: "Free Transform", key: app.skin.name === "photoshop" ? "Mod+T" : "", enabled: () => s.selection.length > 0,
      hidden: () => app.skin.name !== "photoshop", run: () => app.setFreeTransform(!app.freeTransform) },
    toolMove: { label: "Move", key: ik ? "S" : "V", icon: "mouse-pointer-2", run: () => s.setTool("move") },
    toolHand: { label: "Hand", key: ik ? "" : "H", icon: "hand", run: () => s.setTool("hand") },
    toolText: { label: "Text", key: "T", icon: "type", run: () => s.setTool("text") },
    toolRect: { label: "Rectangle", key: "R", icon: "square", run: () => s.setTool("rect") },
    toolEllipse: { label: "Ellipse", key: ik ? "E" : "O", icon: "circle", run: () => s.setTool("ellipse") },
    zoomIn: { label: "Zoom in", key: "Mod+=", icon: "zoom-in", run: () => app.stage.zoomBy(1.25) },
    zoomOut: { label: "Zoom out", key: "Mod+-", icon: "zoom-out", run: () => app.stage.zoomBy(0.8) },
    zoomFit: { label: "Zoom to fit", key: "Shift+1", icon: "maximize", run: () => app.stage.fit() },
    zoom100: { label: "Zoom to 100%", key: "Shift+0", run: () => app.stage.setZoom(1) },
    zoomSelection: {
      label: "Zoom to selection", key: "Shift+2", enabled: hasSel,
      run: () => { app.stage.fitRect(unionBounds(sel())); app.stage.autoFit = false; },
    },
    html: { label: "Edit as HTML…", key: "Mod+Alt+H", icon: "code", enabled: () => app.showCode, run: () => app.openHtml() },
    exportPng: { label: "Export PNG", key: "Mod+Shift+S", icon: "download", run: () => app.exportPng() },
    // Export is how a design is handed in, as in any editor: it ends the episode and scores it.
    submit: { label: "Export…", key: ik ? "" : "Mod+Shift+E", icon: "download", enabled: () => !s.done, run: () => app.confirmSubmit() },
    shortcuts: { label: "Keyboard shortcuts", key: "Mod+/", icon: "keyboard", run: () => app.showShortcuts() },
    // Images come from the computer's files, the way each editor brings them in.
    placeImage: { label: { inkscape: "Import…", photoshop: "Place Embedded…", canva: "Upload files",
      slides: "Upload from computer", powerpoint: "This Device…" }[app.skin.name] || "Place image…",
      key: app.skin.name === "inkscape" ? "Mod+I" : "Mod+Shift+K", icon: "image-plus", run: () => app.openFiles() },
    addHeading: {
      label: "Add a heading", icon: "type",
      run: () => app.addTextPreset({ text: "Add a heading", font_size: Math.round(s.doc.height / 10), bold: true }),
    },
    addSubheading: {
      label: "Add a subheading", icon: "type",
      run: () => app.addTextPreset({ text: "Add a subheading", font_size: Math.round(s.doc.height / 18), bold: true }),
    },
    addBody: {
      label: "Add body text", icon: "text",
      run: () => app.addTextPreset({ text: "Add a little bit of body text", font_size: Math.round(s.doc.height / 36) }),
    },
    addRect: { label: "Rectangle", icon: "square", run: () => app.addShapePreset("rect") },
    addRounded: { label: "Rounded rectangle", icon: "rectangle-horizontal", run: () => app.addShapePreset("rect", 0.12) },
    addEllipse: { label: "Circle", icon: "circle", run: () => app.addShapePreset("ellipse") },
  };
  for (const [id, c] of Object.entries(cmds)) c.id = id;
  return cmds;
}

// Keyboard: translate an event into the "Mod+Shift+Z" spelling used above.
export function eventChord(e) {
  const parts = [];
  if (e.metaKey || e.ctrlKey) parts.push("Mod");
  if (e.shiftKey) parts.push("Shift");
  if (e.altKey) parts.push("Alt");
  let k = e.key;
  if (k === " ") k = "Space";
  if (k === "Backspace") k = "Delete";
  if (k === "+") k = "=";
  if (e.code && e.code.startsWith("Digit") && e.shiftKey) k = e.code.slice(5);
  if (k.length === 1) k = k.toUpperCase();
  if (!["Meta", "Control", "Shift", "Alt"].includes(e.key)) parts.push(k);
  return parts.join("+");
}
