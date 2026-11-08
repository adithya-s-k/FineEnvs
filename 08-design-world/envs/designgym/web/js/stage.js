// The canvas: renders the document as DOM and turns pointer input into edits.
//
// Elements are drawn with the same box model the server renderer uses (see
// core/text.py): absolute boxes at canvas pixels, rotation about the centre,
// text with `white-space: pre`, a `font_size * line_height` line box, and no
// kerning, ligatures or synthetic bold. The artboard is scaled with a CSS
// transform, so everything below is in canvas pixels except the overlay
// (selection box, handles, guides), which lives in screen space so handles
// stay the same size at every zoom level.

import { bounds, clamp, cssColor, deg, fmt, h, rad, rotate, unionBounds } from "./util.js";
import { ensureFonts } from "./fonts.js";
import { normRuns, paintRuns, rangeAll, rangeRects, readRuns, selectOffsets, selectionOffsets, styleRange } from "./runs.js";

const HANDLES = [
  ["nw", -1, -1], ["n", 0, -1], ["ne", 1, -1], ["e", 1, 0],
  ["se", 1, 1], ["s", 0, 1], ["sw", -1, 1], ["w", -1, 0],
];
const SNAP_PX = 6;
const DRAG_THRESHOLD = 3;

export class Stage {
  constructor(app, root) {
    this.app = app;
    this.store = app.store;
    this.root = root;
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.preview = new Map(); // id -> props shown during a gesture
    this.nodes = new Map();
    this.hover = null;
    this.guides = [];
    this.marquee = null;
    this.gesture = null;
    this.editing = null;
    // Characters of one text element chosen while editing, kept when the pointer goes to the editor's
    // controls so colour / Bold / Italic apply to them alone: {id, start, end}.
    this.textRange = null;
    this.space = false;
    this.fitted = false;
    // The page stays fitted to the stage as it resizes (a pane opening, the window changing) until the
    // person zooms or pans themselves, as editors do; Fit (Shift+1) turns that back on.
    this.autoFit = true;

    this.el = h("div", { class: "stage", tabindex: "-1" });
    this.artboard = h("div", { class: "artboard" });
    this.overlay = h("div", { class: "overlay" });
    this.el.append(this.artboard, this.overlay);
    root.append(this.el);

    this.el.addEventListener("pointerdown", (e) => this.onDown(e));
    this.el.addEventListener("pointermove", (e) => this.onMove(e));
    this.el.addEventListener("pointerup", (e) => this.onUp(e));
    this.el.addEventListener("pointercancel", (e) => this.onUp(e));
    this.el.addEventListener("dblclick", (e) => this.onDouble(e));
    this.el.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    this.el.addEventListener("contextmenu", (e) => this.onContext(e));
    this.el.addEventListener("pointerleave", () => { this.hover = null; this.drawOverlay(); });
    new ResizeObserver(() => {
      if ((!this.fitted || this.autoFit) && this.store.doc) this.fit();
      this.drawOverlay();
    }).observe(this.el);

    this.store.on("doc", () => this.render());
    this.store.on("selection", (ids) => {
      if (this.textRange && !(ids?.length === 1 && ids[0] === this.textRange.id)) this.textRange = null;
      this.rotateMode = false;
      this.render();
    });
    // Escape (outside text editing) drops a kept character range, as leaving text mode does.
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && this.textRange && !this.editing) { this.textRange = null; this.drawOverlay(); } });  // a new selection starts with scale handles
    this.store.on("tool", (t) => { this.el.dataset.tool = t; });
    this.el.dataset.tool = this.store.tool;
  }

  // ------------------------------------------------------------ geometry

  toScreen(x, y) { return [this.panX + x * this.zoom, this.panY + y * this.zoom]; }

  toCanvas(clientX, clientY) {
    const r = this.el.getBoundingClientRect();
    return [(clientX - r.left - this.panX) / this.zoom, (clientY - r.top - this.panY) / this.zoom];
  }

  view(el) {
    const p = this.preview.get(el.id);
    return p ? { ...el, ...p } : el;
  }

  hit(x, y) {
    const els = this.store.doc.elements;
    for (let i = els.length - 1; i >= 0; i--) {
      const el = this.view(els[i]);
      if (el.visible === false) continue;
      const cx = el.x + el.width / 2, cy = el.y + el.height / 2;
      const [lx, ly] = rotate(x - cx, y - cy, -rad(el.angle || 0));
      const pad = 2 / this.zoom;
      if (Math.abs(lx) <= el.width / 2 + pad && Math.abs(ly) <= el.height / 2 + pad) return els[i];
    }
    return null;
  }

  // ------------------------------------------------------------ zoom / pan

  setZoom(z, cx = null, cy = null) {
    this.autoFit = false;
    z = clamp(z, 0.02, 16);
    const r = this.el.getBoundingClientRect();
    cx = cx ?? r.width / 2;
    cy = cy ?? r.height / 2;
    const [wx, wy] = [(cx - this.panX) / this.zoom, (cy - this.panY) / this.zoom];
    this.zoom = z;
    this.panX = cx - wx * z;
    this.panY = cy - wy * z;
    this.applyView();
  }

  zoomBy(f) { this.setZoom(this.zoom * f); }

  fit() {
    const doc = this.store.doc;
    if (!doc) return;
    if (this.fitRect({ x0: 0, y0: 0, x1: doc.width, y1: doc.height })) { this.fitted = true; this.autoFit = true; }
  }

  fitRect(b) {
    const r = this.el.getBoundingClientRect();
    if (r.width < 10 || r.height < 10) return false;
    const pad = this.app.skin.fitPadding ?? 48;
    const w = b.x1 - b.x0, hgt = b.y1 - b.y0;
    const z = Math.min((r.width - pad * 2) / w, (r.height - pad * 2) / hgt);
    this.zoom = clamp(z, 0.02, 8);
    this.panX = (r.width - w * this.zoom) / 2 - b.x0 * this.zoom;
    this.panY = (r.height - hgt * this.zoom) / 2 - b.y0 * this.zoom;
    this.applyView();
    return true;
  }

  applyView() {
    const doc = this.store.doc;
    if (!doc) return;
    Object.assign(this.artboard.style, {
      width: `${doc.width}px`, height: `${doc.height}px`,
      transform: `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`,
    });
    this.app.emit("zoom", this.zoom);
    this.drawOverlay();
  }

  // ------------------------------------------------------------ rendering

  render() {
    const doc = this.store.doc;
    if (!doc) return;
    if (!this.fitted) this.fit();
    this.artboard.style.background = cssColor(doc.background);
    this.artboard.style.width = `${doc.width}px`;
    this.artboard.style.height = `${doc.height}px`;
    const seen = new Set();
    let prev = null;
    for (const raw of doc.elements) {
      const el = this.view(raw);
      seen.add(el.id);
      let node = this.nodes.get(el.id);
      if (!node || node.dataset.kind !== el.kind) {
        node?.remove();
        node = this.makeNode(el);
        this.nodes.set(el.id, node);
      }
      if (this.editing !== el.id) this.paint(node, el);
      // Keep DOM order equal to stacking order.
      const want = prev ? prev.nextSibling : this.artboard.firstChild;
      if (want !== node) this.artboard.insertBefore(node, want);
      prev = node;
    }
    for (const [id, node] of this.nodes) {
      if (!seen.has(id)) { node.remove(); this.nodes.delete(id); }
    }
    ensureFonts(doc.elements.filter((e) => e.kind === "text").map((e) => e.font_family))
      .then(() => this.drawOverlay());
    this.drawOverlay();
  }

  makeNode(el) {
    const node = h("div", { class: `el el-${el.kind}`, dataset: { id: el.id, kind: el.kind } });
    if (el.kind === "image") {
      node.append(h("img", { draggable: "false", alt: "" }));
    } else if (el.kind === "text") {
      node.append(h("div", { class: "txt", spellcheck: "false" }));
    }
    return node;
  }

  paint(node, el) {
    const st = node.style;
    st.left = `${el.x}px`;
    st.top = `${el.y}px`;
    st.width = `${el.width}px`;
    st.height = `${el.height}px`;
    const tf = [];
    if (el.angle) tf.push(`rotate(${el.angle}deg)`);
    if (el.flip_x || el.flip_y) tf.push(`scale(${el.flip_x ? -1 : 1}, ${el.flip_y ? -1 : 1})`);
    st.transform = tf.join(" ");
    st.opacity = el.opacity ?? 1;
    st.display = el.visible === false ? "none" : "";
    if (el.kind === "image") {
      const img = node.firstChild;
      const url = this.store.assetById.get(el.asset_id)?.url;
      if (url && img.getAttribute("src") !== url) img.src = url;
    } else if (el.kind === "shape") {
      st.background = el.fill ? cssColor(el.fill) : "transparent";
      st.border = el.stroke && el.stroke_width ? `${el.stroke_width}px solid ${cssColor(el.stroke)}` : "none";
      st.borderRadius = el.shape === "ellipse" ? "50%" : `${el.radius || 0}px`;
    } else {
      this.paintText(node.firstChild, el);
    }
  }

  paintText(t, el) {
    const lh = el.font_size * el.line_height;
    Object.assign(t.style, {
      fontFamily: `'${el.font_family}'`,
      fontSize: `${el.font_size}px`,
      lineHeight: `${lh}px`,
      letterSpacing: `${el.letter_spacing || 0}px`,
      color: cssColor(el.color || "#000000"),
      textAlign: el.align,
      fontWeight: el.bold ? "700" : "400",
      fontStyle: el.italic ? "italic" : "normal",
      textDecoration: el.underline ? "underline" : "none",
      textTransform: el.uppercase ? "uppercase" : "none",
    });
    const key = JSON.stringify([el.text, el.runs]);
    if (t.dataset.key === key) return;
    t.dataset.key = key;
    paintRuns(t, el.text, el.runs);
  }

  // ------------------------------------------------------------ overlay

  drawOverlay() {
    const o = this.overlay;
    o.textContent = "";
    const doc = this.store.doc;
    if (!doc) return;
    const skin = this.app.skin;
    const sel = this.store.selected().map((e) => this.view(e));
    if (skin.frameLabel) {  // Figma names the frame just above its top-left corner
      const [fx, fy] = this.toScreen(0, 0);
      const label = h("div", { class: "frame-label", style: { left: `${fx}px`, top: `${fy - 18}px` } }, skin.frameLabel);
      // The frame's name selects the frame, as in Figma.
      label.addEventListener("pointerdown", (e) => { if (e.button === 0) { e.stopPropagation(); e.preventDefault(); this.app.selectFrame?.(); } });
      label.classList.toggle("on", !!this.app.frameSelected);
      o.append(label);
      if (this.app.frameSelected) {
        const [x1, y1] = this.toScreen(doc.width, doc.height);
        o.append(h("div", { class: "frame-sel", style: { left: `${fx}px`, top: `${fy}px`, width: `${x1 - fx}px`, height: `${y1 - fy}px` } }),
          h("div", { class: "size-badge frame-badge", style: { left: `${(fx + x1) / 2}px`, top: `${y1 + 6}px` } }, `${doc.width} × ${doc.height}`));
      }
    }
    if (skin.pageHeader) {  // Canva names the page just above it
      const [x0, y0] = this.toScreen(0, 0);
      const [x1] = this.toScreen(doc.width, 0);
      o.append(h("div", { class: "page-header", style: { left: `${x0}px`, top: `${y0 - 30}px`, width: `${x1 - x0}px` } },
        h("span", { class: "page-header-title" }, skin.pageHeader)));
    }

    if (this.hover && !this.gesture && !this.store.selection.includes(this.hover)) {
      const el = this.store.find(this.hover);
      if (el && el.visible !== false) o.append(this.box(this.view(el), "hover-box"));
    }

    // Photoshop shows a layer's bounding box and handles only with Show Transform Controls (or Free Transform).
    const handles = this.app.showHandles ? this.app.showHandles() : true;
    if (sel.length === 1 && !this.editing && handles) {
      const el = sel[0];
      const box = this.box(el, "sel-box");
      if (el.locked) box.classList.add("locked");
      else {
        const ik = skin.name === "inkscape";
        if (ik) box.classList.add(this.rotateMode ? "ik-rotate" : "ik-scale");
        for (const [name, dx, dy] of HANDLES) {
          if (ik && this.rotateMode) break;  // rotate mode: only the corner rotate arrows
          if (el.kind === "text" && dy !== 0 && dx === 0) continue;
          box.append(h("div", { class: `handle h-${name} ${dx && dy ? "corner" : "side"}`,
            dataset: { handle: name, dx, dy } }));
        }
        if (skin.rotateHandle) {
          box.append(h("div", { class: "rot-handle", dataset: { handle: "rotate" },
            html: skin.rotateIcon || "" }));
        }
        for (const [name, dx, dy] of HANDLES) {
          if (ik && !this.rotateMode) break;  // scale mode: corners scale, no rotate zones
          if (dx && dy) box.append(h("div", { class: `rot-zone z-${name}`, dataset: { handle: "rotate" } }));
        }
      }
      o.append(box);
      if (skin.alwaysShowSize || this.gesture) o.append(this.sizeBadge(el));
      if (skin.floatingToolbar && !this.gesture && !el.locked) o.append(this.app.floatingToolbar(this.screenRect(el)));
    } else if (sel.length > 1) {
      for (const el of sel) o.append(this.box(el, "sel-box multi-member"));
      const u = unionBounds(sel);
      const group = this.rectBox(u, "sel-box group-box");
      o.append(group);
      if (this.gesture) o.append(this.sizeBadge({ x: u.x0, y: u.y0, width: u.x1 - u.x0, height: u.y1 - u.y0, angle: 0 }));
    } else if (this.editing) {
      const el = this.store.find(this.editing);
      if (el) o.append(this.box(this.view(el), "sel-box editing"));
    }

    this.drawTextRange(o);
    for (const g of this.guides) {
      if (g.axis === "x") {
        const [sx] = this.toScreen(g.pos, 0);
        const [, sy0] = this.toScreen(0, g.from);
        const [, sy1] = this.toScreen(0, g.to);
        o.append(h("div", { class: "guide guide-x", style: { left: `${sx}px`, top: `${sy0}px`, height: `${sy1 - sy0}px` } }));
      } else {
        const [, sy] = this.toScreen(0, g.pos);
        const [sx0] = this.toScreen(g.from, 0);
        const [sx1] = this.toScreen(g.to, 0);
        o.append(h("div", { class: "guide guide-y", style: { top: `${sy}px`, left: `${sx0}px`, width: `${sx1 - sx0}px` } }));
      }
    }
    if (this.marquee) {
      const m = this.marquee;
      o.append(this.rectBox({ x0: Math.min(m.x0, m.x1), y0: Math.min(m.y0, m.y1),
        x1: Math.max(m.x0, m.x1), y1: Math.max(m.y0, m.y1) }, "marquee"));
    }
    if (this.draft) o.append(this.rectBox(this.draft, `draft draft-${this.draft.shape}`));
  }

  screenRect(el) {
    const [cx, cy] = this.toScreen(el.x + el.width / 2, el.y + el.height / 2);
    const w = el.width * this.zoom, hh = el.height * this.zoom;
    const b = bounds(el);
    const [x0, y0] = this.toScreen(b.x0, b.y0);
    const [x1, y1] = this.toScreen(b.x1, b.y1);
    return { cx, cy, w, h: hh, x0, y0, x1, y1 };
  }

  box(el, cls) {
    const r = this.screenRect(el);
    return h("div", { class: cls, style: {
      left: `${r.cx - r.w / 2}px`, top: `${r.cy - r.h / 2}px`, width: `${r.w}px`, height: `${r.h}px`,
      transform: el.angle ? `rotate(${el.angle}deg)` : "",
    } });
  }

  rectBox(b, cls) {
    const [x0, y0] = this.toScreen(b.x0, b.y0);
    const [x1, y1] = this.toScreen(b.x1, b.y1);
    return h("div", { class: cls, style: { left: `${x0}px`, top: `${y0}px`, width: `${x1 - x0}px`, height: `${y1 - y0}px` } });
  }

  sizeBadge(el) {
    const r = this.screenRect(el);
    const g = this.gesture;
    let text = `${fmt(el.width)} × ${fmt(el.height)}`;
    if (g?.kind === "move") text = `X ${fmt(el.x)}  Y ${fmt(el.y)}`;
    if (g?.kind === "rotate") text = `${fmt(el.angle || 0)}°`;
    return h("div", { class: "size-badge", style: { left: `${(r.x0 + r.x1) / 2}px`, top: `${r.y1 + 8}px` } }, text);
  }

  // ------------------------------------------------------------ input

  onWheel(e) {
    e.preventDefault();
    const r = this.el.getBoundingClientRect();
    if (e.ctrlKey || e.metaKey) {
      const f = Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0025));
      this.setZoom(this.zoom * f, e.clientX - r.left, e.clientY - r.top);
    } else {
      const dx = e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX;
      const dy = e.shiftKey && !e.deltaX ? 0 : e.deltaY;
      this.autoFit = false;
      this.panX -= dx;
      this.panY -= dy;
      this.applyView();
    }
  }

  onDown(e) {
    if (this.textRange && !(this.editing && e.target.closest(".txt[contenteditable]"))) {
      // after this press's blur, so a hex typed into a field and left by clicking the canvas still lands on the range
      const kept = this.textRange;
      setTimeout(() => { if (this.textRange === kept) { this.textRange = null; this.drawOverlay(); } }, 0);
    }
    if (this.editing) {
      if (e.target.closest(".txt[contenteditable]")) return;
      this.commitEdit();
    }
    if (e.button === 2) return;
    this.el.focus({ preventScroll: true });
    const [x, y] = this.toCanvas(e.clientX, e.clientY);
    const start = { cx: e.clientX, cy: e.clientY, x, y };
    const tool = this.store.tool;

    if (this.placing && e.button === 0) {  // Figma: images chosen from the file window wait for a click
      this.begin(e, { kind: "place", start, asset: this.placing.queue[0] });
      return;
    }
    if (e.button === 1 || tool === "hand" || this.space) {
      this.begin(e, { kind: "pan", start, panX: this.panX, panY: this.panY });
      return;
    }
    const handle = e.target.closest("[data-handle]");
    if (handle && this.store.selection.length === 1) {
      const el = this.store.find(this.store.selection[0]);
      if (el && !el.locked) {
        if (handle.dataset.handle === "rotate") this.begin(e, { kind: "rotate", start, el: { ...el } });
        else this.begin(e, { kind: "resize", start, el: { ...el }, dx: +handle.dataset.dx, dy: +handle.dataset.dy });
        return;
      }
    }
    if (e.target.closest(".floating-toolbar")) return;
    if (tool === "rect" || tool === "ellipse") {
      this.begin(e, { kind: "draw", start, shape: tool === "rect" ? "rect" : "ellipse" });
      return;
    }
    if (tool === "text") {
      // Figma's, Photoshop's and Inkscape's text tools put the caret into a text they click on instead of
      // starting a new box (PowerPoint and Slides draw a new box over it).
      const hit = ["figma", "photoshop", "inkscape"].includes(this.app.skin.name) ? this.hit(x, y) : null;
      if (hit && hit.kind === "text" && !hit.locked && e.button === 0) {
        e.preventDefault();
        this.startEdit(hit.id, { selectAll: false });
        const at = document.caretRangeFromPoint?.(e.clientX, e.clientY);
        const node = this.nodes.get(hit.id);
        if (at && node?.firstChild.contains(at.startContainer)) {
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(at);
        }
        return;
      }
      this.begin(e, { kind: "text", start });
      return;
    }
    // Photoshop's Move tool with Auto-Select off moves the selected layer wherever you drag.
    if (this.app.autoSelect === false && tool === "move" && this.store.selection.length && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      const els = this.store.selected().filter((el) => !el.locked).map((el) => ({ ...el }));
      this.begin(e, { kind: "move", start, els, clicked: els[0]?.id, wasSelected: true });
      return;
    }
    const hit = this.hit(x, y);
    if (hit) {
      const selected = this.store.selection.includes(hit.id);
      if (e.shiftKey || e.metaKey || e.ctrlKey) {
        this.store.select(hit.id, { toggle: true });
        return;
      }
      if (!selected) this.store.select(hit.id);
      const els = this.store.selected().filter((el) => !el.locked).map((el) => ({ ...el }));
      this.begin(e, { kind: "move", start, els, clicked: hit.id, wasSelected: selected });
    } else {
      if (!e.shiftKey) this.store.select([]);
      this.begin(e, { kind: "marquee", start, base: e.shiftKey ? [...this.store.selection] : [] });
    }
  }

  begin(e, g) {
    this.gesture = g;
    g.moved = false;
    this.el.setPointerCapture(e.pointerId);
    this.store.busy(true);
  }

  onMove(e) {
    const g = this.gesture;
    const [x, y] = this.toCanvas(e.clientX, e.clientY);
    if (!g) {
      const hit = this.store.tool === "move" && !this.space ? this.hit(x, y) : null;
      const id = hit?.id || null;
      if (id !== this.hover) { this.hover = id; this.drawOverlay(); }
      const handle = e.target.closest?.("[data-handle]");
      this.el.dataset.cursor = handle ? handle.dataset.handle : hit ? "move" : "";
      return;
    }
    const dist = Math.hypot(e.clientX - g.start.cx, e.clientY - g.start.cy);
    if (!g.moved && dist < DRAG_THRESHOLD) return;
    g.moved = true;
    const dx = x - g.start.x, dy = y - g.start.y;
    if (g.kind === "pan") {
      this.autoFit = false;
      this.panX = g.panX + (e.clientX - g.start.cx);
      this.panY = g.panY + (e.clientY - g.start.cy);
      this.applyView();
    } else if (g.kind === "move") this.dragMove(g, dx, dy, e);
    else if (g.kind === "resize") this.dragResize(g, x, y, e);
    else if (g.kind === "rotate") this.dragRotate(g, x, y, e);
    else if (g.kind === "marquee") this.dragMarquee(g, x, y);
    else if (g.kind === "place") {  // drag out the image's box; it keeps its proportions
      const w = x - g.start.x, hh = Math.sign(y - g.start.y || 1) * Math.abs(w) / g.asset.aspect;
      this.draft = { x0: Math.min(g.start.x, g.start.x + w), y0: Math.min(g.start.y, g.start.y + hh),
        x1: Math.max(g.start.x, g.start.x + w), y1: Math.max(g.start.y, g.start.y + hh), shape: "rect" };
      this.drawOverlay();
    } else if (g.kind === "draw") {
      let x0 = g.start.x, y0 = g.start.y, w = x - x0, hh = y - y0;
      if (e.shiftKey) { const m = Math.max(Math.abs(w), Math.abs(hh)); w = Math.sign(w || 1) * m; hh = Math.sign(hh || 1) * m; }
      this.draft = { x0: Math.min(x0, x0 + w), y0: Math.min(y0, y0 + hh), x1: Math.max(x0, x0 + w), y1: Math.max(y0, y0 + hh), shape: g.shape };
      this.drawOverlay();
    }
  }

  async onUp(e) {
    const g = this.gesture;
    if (!g) return;
    this.gesture = null;
    try { this.el.releasePointerCapture(e.pointerId); } catch {}
    const [x, y] = this.toCanvas(e.clientX, e.clientY);
    try {
      if (g.kind === "move") {
        if (g.moved) {
          const changes = g.els.map((el) => [el.id, this.pick(this.preview.get(el.id), ["x", "y"])]);
          await this.store.updateMany(changes.filter(([, p]) => p));
        } else if (g.wasSelected && this.store.selection.length > 1) {
          this.store.select(g.clicked);
        } else if (g.wasSelected && this.app.skin.name === "inkscape") {
          // Inkscape: clicking the selection again toggles scale arrows <-> rotate arrows.
          this.rotateMode = !this.rotateMode;
          this.drawOverlay();
        }
      } else if (g.kind === "resize" || g.kind === "rotate") {
        if (g.moved) {
          const p = this.preview.get(g.el.id);
          if (p) await this.store.update(g.el.id, p);
        }
      } else if (g.kind === "marquee") {
        this.marquee = null;
      } else if (g.kind === "draw") {
        const d = this.draft;
        this.draft = null;
        let box = d;
        if (!g.moved || !d || d.x1 - d.x0 < 4 || d.y1 - d.y0 < 4) {
          const side = Math.round(Math.min(this.store.doc.width, this.store.doc.height) * 0.25);
          box = { x0: x - side / 2, y0: y - side / 2, x1: x + side / 2, y1: y + side / 2 };
        }
        const r = await this.store.op("add_shape", {
          shape: g.shape, x: Math.round(box.x0), y: Math.round(box.y0),
          width: Math.round(box.x1 - box.x0), height: Math.round(box.y1 - box.y0),
          fill: this.app.defaultFill(),
        });
        if (r?.element_id) this.store.select(r.element_id);
        this.store.setTool("move");
      } else if (g.kind === "text") {
        await this.app.addTextAt(x, y);
        this.store.setTool("move");
      } else if (g.kind === "place") {
        const d = this.draft;
        this.draft = null;
        const big = g.moved && d && d.x1 - d.x0 >= 4;
        await this.app.addAsset(g.asset, big ? null : [x, y],
          big ? { x: d.x0, y: d.y0, width: d.x1 - d.x0, height: d.y1 - d.y0 } : null);
        this.placing?.queue.shift();
        if (!this.placing?.queue.length) this.stopPlacing();
        else this.placing.update();
      }
    } catch (err) {
      console.warn(err);
    } finally {
      this.preview.clear();
      this.guides = [];
      this.store.busy(false);
      this.render();
    }
  }

  pick(obj, keys) {
    if (!obj) return null;
    const out = {};
    for (const k of keys) if (k in obj) out[k] = Math.round(obj[k] * 100) / 100;
    return Object.keys(out).length ? out : null;
  }

  // ------------------------------------------------------------ gestures

  snapTargets(excludeIds) {
    const doc = this.store.doc;
    const xs = [0, doc.width / 2, doc.width], ys = [0, doc.height / 2, doc.height];
    const spansX = xs.map(() => [0, doc.height]), spansY = ys.map(() => [0, doc.width]);
    for (const el of doc.elements) {
      if (excludeIds.has(el.id) || el.visible === false) continue;
      const b = bounds(el);
      for (const v of [b.x0, (b.x0 + b.x1) / 2, b.x1]) { xs.push(v); spansX.push([b.y0, b.y1]); }
      for (const v of [b.y0, (b.y0 + b.y1) / 2, b.y1]) { ys.push(v); spansY.push([b.x0, b.x1]); }
    }
    return { xs, ys, spansX, spansY };
  }

  snapAxis(values, targets) {
    const tol = SNAP_PX / this.zoom;
    let best = null;
    for (const v of values) {
      for (let i = 0; i < targets.length; i++) {
        const d = targets[i] - v;
        if (Math.abs(d) <= tol && (!best || Math.abs(d) < Math.abs(best.d))) best = { d, pos: targets[i], i };
      }
    }
    return best;
  }

  dragMove(g, dx, dy, e) {
    if (!g.els.length) return;
    if (e.shiftKey) { if (Math.abs(dx) > Math.abs(dy)) dy = 0; else dx = 0; }
    const moved = g.els.map((el) => ({ ...el, x: el.x + dx, y: el.y + dy }));
    const u = unionBounds(moved);
    this.guides = [];
    if (!e.altKey) {
      const t = this.snapTargets(new Set(g.els.map((el) => el.id)));
      const sx = this.snapAxis([u.x0, (u.x0 + u.x1) / 2, u.x1], t.xs);
      const sy = this.snapAxis([u.y0, (u.y0 + u.y1) / 2, u.y1], t.ys);
      if (sx) { dx += sx.d; }
      if (sy) { dy += sy.d; }
      const u2 = { x0: u.x0 + (sx?.d || 0), x1: u.x1 + (sx?.d || 0), y0: u.y0 + (sy?.d || 0), y1: u.y1 + (sy?.d || 0) };
      if (sx) {
        const span = t.spansX[sx.i];
        this.guides.push({ axis: "x", pos: sx.pos, from: Math.min(span[0], u2.y0) - 8, to: Math.max(span[1], u2.y1) + 8 });
      }
      if (sy) {
        const span = t.spansY[sy.i];
        this.guides.push({ axis: "y", pos: sy.pos, from: Math.min(span[0], u2.x0) - 8, to: Math.max(span[1], u2.x1) + 8 });
      }
    }
    for (const el of g.els) this.preview.set(el.id, { x: el.x + dx, y: el.y + dy });
    this.render();
  }

  dragResize(g, px, py, e) {
    const el = g.el;
    const a = rad(el.angle || 0);
    const cx = el.x + el.width / 2, cy = el.y + el.height / 2;
    const { dx, dy } = g;
    const [ax, ay] = rotate(-dx * el.width / 2, -dy * el.height / 2, a);
    const anchor = [cx + ax, cy + ay];
    let [vx, vy] = rotate(px - anchor[0], py - anchor[1], -a);
    let w = dx ? Math.max(1, vx * dx) : el.width;
    let hh = dy ? Math.max(1, vy * dy) : el.height;
    const corner = dx && dy;
    const keepAspect = el.kind === "text" ? corner : (el.kind === "image" ? corner !== e.shiftKey : e.shiftKey && corner);
    if (keepAspect && corner) {
      const s = Math.max(w / el.width, hh / el.height);
      w = el.width * s;
      hh = el.height * s;
    }
    const props = {};
    if (el.kind === "text") {
      if (corner) {
        const s = w / el.width;
        props.font_size = Math.max(1, Math.round(el.font_size * s * 100) / 100);
        props.width = w;
        props.letter_spacing = Math.round((el.letter_spacing || 0) * s * 100) / 100;
        hh = el.height * s;
      } else {
        props.width = w;
        hh = el.height;
      }
    } else {
      props.width = w;
      props.height = hh;
    }
    // Snap the moving edges of an unrotated element to guides.
    this.guides = [];
    if (!el.angle && !e.altKey) {
      const t = this.snapTargets(new Set([el.id]));
      if (dx) {
        const edge = dx > 0 ? anchor[0] + w : anchor[0] - w;
        const s = this.snapAxis([edge], t.xs);
        if (s && !keepAspect) { w += dx > 0 ? s.d : -s.d; props.width = w; this.guides.push({ axis: "x", pos: s.pos, from: 0, to: this.store.doc.height }); }
      }
      if (dy && el.kind !== "text") {
        const edge = dy > 0 ? anchor[1] + hh : anchor[1] - hh;
        const s = this.snapAxis([edge], t.ys);
        if (s && !keepAspect) { hh += dy > 0 ? s.d : -s.d; props.height = hh; this.guides.push({ axis: "y", pos: s.pos, from: 0, to: this.store.doc.width }); }
      }
    }
    const [ox, oy] = rotate(dx * w / 2, dy * hh / 2, a);
    const ncx = anchor[0] + ox, ncy = anchor[1] + oy;
    props.x = ncx - w / 2;
    props.y = ncy - hh / 2;
    if (el.kind === "text") {
      this.preview.set(el.id, { ...props, height: hh, auto_width: false });
    } else this.preview.set(el.id, props);
    this.render();
  }

  dragRotate(g, px, py, e) {
    const el = g.el;
    const cx = el.x + el.width / 2, cy = el.y + el.height / 2;
    const ang = deg(Math.atan2(py - cy, px - cx));
    if (g.a0 === undefined) g.a0 = deg(Math.atan2(g.start.y - cy, g.start.x - cx)) - (el.angle || 0);
    let a = ang - g.a0;
    if (e.shiftKey) a = Math.round(a / 15) * 15;
    else for (const s of [-180, -90, 0, 90, 180]) if (Math.abs(a - s) < 3) a = s;
    a = ((a % 360) + 540) % 360 - 180;
    this.preview.set(el.id, { angle: Math.round(a * 100) / 100 });
    this.render();
  }

  dragMarquee(g, x, y) {
    this.marquee = { x0: g.start.x, y0: g.start.y, x1: x, y1: y };
    const m = { x0: Math.min(g.start.x, x), y0: Math.min(g.start.y, y), x1: Math.max(g.start.x, x), y1: Math.max(g.start.y, y) };
    const ids = this.store.doc.elements.filter((el) => {
      if (el.visible === false || el.locked) return false;
      const b = bounds(el);
      return b.x1 >= m.x0 && b.x0 <= m.x1 && b.y1 >= m.y0 && b.y0 <= m.y1;
    }).map((el) => el.id);
    this.store.select([...new Set([...g.base, ...ids])]);
    this.drawOverlay();
  }

  // ------------------------------------------------------------ text editing

  onDouble(e) {
    const [x, y] = this.toCanvas(e.clientX, e.clientY);
    const hit = this.hit(x, y);
    if (this.editing && e.target.closest(".txt[contenteditable]")) return;  // the browser selects the word
    if (hit && hit.kind === "text" && !hit.locked) this.startEdit(hit.id, { selectAll: false });
  }

  startEdit(id, { selectAll = true } = {}) {
    const el = this.store.find(id);
    const node = this.nodes.get(id);
    if (!el || !node || el.kind !== "text") return;
    this.store.select([id]);
    this.editing = id;
    this.textRange = null;
    // Editing is a state, not an operation in flight: the store stays idle, so screenshots are not held back.
    const t = node.firstChild;
    t.dataset.key = "";
    paintRuns(t, el.text, el.runs);
    t.setAttribute("contenteditable", "plaintext-only");
    if (el.auto_width) node.style.width = "max-content";
    node.classList.add("editing");
    t.focus();
    const range = document.createRange();
    range.selectNodeContents(t);
    if (!selectAll) range.collapse(false);
    const s = window.getSelection();
    s.removeAllRanges();
    s.addRange(range);
    const lh = el.font_size * el.line_height;
    const onInput = () => {
      const lines = editedText(t).split("\n").length;
      node.style.height = `${Math.max(1, lines) * lh}px`;
      this.drawOverlay();
    };
    const onKey = (ev) => {
      ev.stopPropagation();
      const k = ev.key.toLowerCase();
      if ((ev.metaKey || ev.ctrlKey) && !ev.altKey && (k === "b" || k === "i")) {
        // Bold / Italic on the selected characters, as every editor does inside a text box
        ev.preventDefault();
        const sel = selectionOffsets(t);
        if (!sel || sel.end <= sel.start) return;
        const raw = readRuns(t);
        const text = raw.map((r) => r.text).join("");
        const cur = this.store.find(id);
        const key = k === "b" ? "bold" : "italic";
        const runs = styleRange(text, raw, sel.start, sel.end, { [key]: !rangeAll(cur, raw, sel.start, sel.end, key) });
        paintRuns(t, text, normRuns(runs, cur) || []);
        selectOffsets(t, sel.start, sel.end);
        return;
      }
      if (ev.key === "Escape" || (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey))) {
        ev.preventDefault();
        this.commitEdit();
      } else if (ev.key === "Enter") {
        ev.preventDefault();
        insertNewline(t);
        onInput();
      }
    };
    const onBlur = () => setTimeout(() => { if (this.editing === id) this.commitEdit(); }, 0);
    // A press on the editor's own controls (toolbar, panel, menu) keeps the selected characters for them.
    const onChrome = (ev) => {
      if (this.el.contains(ev.target)) return;
      const sel = selectionOffsets(t);
      this.textRange = sel && sel.end > sel.start ? { id, ...sel } : null;
    };
    // Dragging selected characters selects anew instead of moving them, as in the design editors.
    const noDrag = (ev) => ev.preventDefault();
    t.addEventListener("input", onInput);
    t.addEventListener("keydown", onKey);
    t.addEventListener("blur", onBlur);
    t.addEventListener("dragstart", noDrag);
    document.addEventListener("pointerdown", onChrome, true);
    this._editCleanup = () => {
      t.removeEventListener("input", onInput);
      t.removeEventListener("keydown", onKey);
      t.removeEventListener("blur", onBlur);
      t.removeEventListener("dragstart", noDrag);
      document.removeEventListener("pointerdown", onChrome, true);
      t.removeAttribute("contenteditable");
      node.classList.remove("editing");
    };
    this.drawOverlay();
    this.app.emit("editing", id);
  }

  /** Leave text editing without keeping the typing (Photoshop's options-bar cancel). */
  cancelEdit() {
    const id = this.editing;
    if (!id) return;
    const node = this.nodes.get(id);
    this._editCleanup?.();
    this.editing = null;
    if (node) node.firstChild.dataset.key = "";
    this.render();
    this.app.emit("editing", null);
  }

  async commitEdit() {
    const id = this.editing;
    if (!id) return;
    const node = this.nodes.get(id);
    const raw = readRuns(node.firstChild);
    const text = raw.map((r) => r.text).join("");
    this._editCleanup?.();
    this.editing = null;
    const el = this.store.find(id);
    try {
      const runs = el ? normRuns(raw, el) : null;
      const before = el ? normRuns(el.runs, el) : null;
      if (el && (text !== el.text || JSON.stringify(runs) !== JSON.stringify(before))) {
        if (!text.trim()) await this.store.op("delete_element", { element_id: id });
        else await this.store.update(id, runs || el.runs ? { text, runs } : { text });
      }
    } finally {
      if (node) node.firstChild.dataset.key = "";
      this.render();
      this.app.emit("editing", null);
    }
  }

  /**
   * Apply text styles to the kept character range: colour, bold and italic go to those characters, any
   * other property to the whole box. Returns false when no range is kept for this element.
   */
  async applyRangeStyle(el, props) {
    const r = this.textRange;
    if (!r || !el || r.id !== el.id) return false;
    if (this.editing === r.id) await this.commitEdit();
    const cur = this.store.find(r.id);
    if (!cur) return false;
    const ranged = {};
    const rest = {};
    for (const [k, v] of Object.entries(props)) (["color", "bold", "italic"].includes(k) ? ranged : rest)[k] = v;
    if (Object.keys(rest).length) await this.store.update(cur.id, rest);
    if (Object.keys(ranged).length) {
      const now = this.store.find(r.id);
      const runs = normRuns(styleRange(now.text, now.runs, r.start, r.end, ranged), now);
      await this.store.update(now.id, { text: now.text, runs });
    }
    this.drawOverlay();
    return true;
  }

  /** Whether every kept character has `key` (bold, italic) on. */
  rangeHas(key) {
    const el = this.textRange && this.store.find(this.textRange.id);
    return !!el && rangeAll(el, el.runs, this.textRange.start, this.textRange.end, key);
  }

  /** The kept characters' bold / italic (true only if all have it) or colour (null when they differ). */
  rangeValue(key) {
    if (key !== "color") return this.rangeHas(key);
    const r = this.textRange;
    const el = r && this.store.find(r.id);
    if (!el) return undefined;
    const colors = new Set();
    let pos = 0;
    for (const run of el.runs?.length ? el.runs : [{ text: el.text }]) {
      const a = pos;
      pos += run.text.length;
      if (pos > r.start && a < r.end) colors.add((run.color || el.color || "#000000").toLowerCase());
    }
    return colors.size === 1 ? [...colors][0] : null;
  }

  /** The selected characters of the text being edited ({start, end}), or null. */
  editSelection() {
    const node = this.editing && this.nodes.get(this.editing);
    return node ? selectionOffsets(node.firstChild) : null;
  }

  /** Viewport rectangles of characters [start, end) of a text element, one per character when `each`. */
  charRects(id, start, end, each = false) {
    const node = this.nodes.get(id);
    if (!node) return [];
    const box = (q) => ({ x: q.left, y: q.top, width: q.width, height: q.height });
    if (!each) return rangeRects(node.firstChild, start, end).map(box);
    const out = [];
    for (let i = start; i < end; i++) out.push(rangeRects(node.firstChild, i, i + 1).map(box)[0] || null);
    return out;
  }

  drawTextRange(o) {
    const r = this.textRange;
    const node = r && !this.editing && this.nodes.get(r.id);
    if (!node) return;
    const base = o.getBoundingClientRect();
    for (const q of rangeRects(node.firstChild, r.start, r.end)) {
      o.append(h("div", { class: "text-range", style: { left: `${q.left - base.left}px`, top: `${q.top - base.top}px`,
        width: `${q.width}px`, height: `${q.height}px` } }));
    }
  }

  // ------------------------------------------------------------ misc

  onContext(e) {
    e.preventDefault();
    const [x, y] = this.toCanvas(e.clientX, e.clientY);
    const hit = this.hit(x, y);
    if (hit && !this.store.selection.includes(hit.id)) this.store.select(hit.id);
    this.app.contextMenu(e.clientX, e.clientY, !!hit);
  }

  /** Figma's "place image" mode: each click (or drag) on the canvas places the next chosen image. */
  startPlacing(assets) {
    this.stopPlacing();
    const ghost = h("div", { class: "place-ghost" });
    const update = () => {
      const a = this.placing?.queue[0];
      if (!a) return;
      ghost.textContent = "";
      ghost.append(h("img", { src: a.url, alt: "" }));
      if (this.placing.queue.length > 1) ghost.append(h("span", { class: "fc-ghost-count" }, String(this.placing.queue.length)));
    };
    const follow = (e) => Object.assign(ghost.style, { left: `${e.clientX}px`, top: `${e.clientY}px` });
    document.body.append(ghost);
    document.addEventListener("pointermove", follow);
    this.placing = { queue: [...assets], ghost, follow, update };
    this.el.classList.add("placing");
    update();
  }

  stopPlacing() {
    if (!this.placing) return;
    this.placing.ghost.remove();
    document.removeEventListener("pointermove", this.placing.follow);
    this.placing = null;
    this.el.classList.remove("placing");
  }

  dropPoint(clientX, clientY) {
    const r = this.el.getBoundingClientRect();
    if (clientX < r.left || clientX > r.right || clientY < r.top || clientY > r.bottom) return null;
    return this.toCanvas(clientX, clientY);
  }

}

// Chromium ignores insertText("\n") in a plaintext-only editable, and may
// wrap lines in <div>/<br> on its own. Newlines are therefore inserted as
// text nodes, and the edited text is read back structurally.
function insertNewline(root) {
  const sel = window.getSelection();
  if (!sel.rangeCount) return;
  const range = sel.getRangeAt(0);
  range.deleteContents();
  const nl = document.createTextNode("\n");
  range.insertNode(nl);
  // A newline at the very end only renders once something follows it.
  if (!nl.nextSibling || (nl.nextSibling.nodeType === 3 && !nl.nextSibling.data)) {
    const pad = document.createTextNode("\u200b");
    nl.after(pad);
  }
  range.setStartAfter(nl);
  range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);
}

function editedText(root) {
  let out = "";
  const walk = (node, top) => {
    for (const c of node.childNodes) {
      if (c.nodeType === 3) out += c.data;
      else if (c.nodeName === "BR") out += "\n";
      else {
        const block = c.nodeName === "DIV" || c.nodeName === "P";
        if (block && out && !out.endsWith("\n")) out += "\n";
        walk(c, false);
      }
    }
  };
  walk(root, true);
  return out.replace(/\u200b/g, "").replace(/\r/g, "");
}
