// Editor bootstrap: builds the chosen skin around the canvas, connects to the
// episode, and wires keyboard, dialogs and the computer-use hooks.
//
//   /editor/?episode=<id>&skin=figma|canva|photoshop|slides|powerpoint|inkscape&theme=light|dark
//           &layout=swap   mirror the panels (layout randomisation)
//           &agent=1       opened by the computer-use driver

import { Emitter, h, shortcut } from "./util.js";
import { icon } from "./icons.js";
import { Store } from "./store.js";
import { Stage } from "./stage.js";
import { createCommands, eventChord } from "./commands.js";
import { SKINS } from "./skins.js";
import { openFileChooser } from "./files.js";
import { ensureFonts, fontCatalog } from "./fonts.js";
import { closePopover, contextMenuAt, iconButton, modal, popoverOpen, toast } from "./widgets.js";

// Display names are neutral on purpose: the layouts are inspired by well-known
// editors, but no product name or logo appears in the UI.
const SKIN_LABELS = { figma: "Studio layout", canva: "Template layout", photoshop: "Pixel layout" };

class App extends Emitter {
  constructor() {
    super();
    const q = new URLSearchParams(location.search);
    this.params = {
      episode: q.get("episode"),
      skin: SKINS[q.get("skin")] ? q.get("skin") : "figma",
      theme: q.get("theme") || null,
      layout: q.get("layout") || "default",
      agent: q.get("agent") === "1",
      os: q.get("os"),
      fs: q.get("fs"),
    };
    // The editor never shows the target (no reference panel, no overlay), for agents or people:
    // it comes with the task (and view_reference); the playground shows it beside the editor.
    // Nor does it offer the HTML code editor: on the GUI surface, writing the design as
    // code would skip the skill being trained. HTML is its own surface; people keep it here.
    this.showCode = !this.params.agent;
    this.skin = SKINS[this.params.skin];
    this.fgColor = this.skin.name === "inkscape" ? this.skin.fill || "#000000" : "#000000";
    document.documentElement.dataset.skin = this.skin.name;
    const theme = this.params.theme || (this.skin.name === "photoshop" ? "dark" : "light");
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.layout = this.params.layout;
  }

  async start() {
    await fontCatalog().catch(() => null);
    if (!this.params.episode) {  // no design browser (it would show every task's answer): episodes come from reset()
      this.fatal("Open an episode's editor_url (from reset / start_task) or start one in the playground at /web/.");
      return;
    }
    this.store = new Store(this.params.episode, { source: this.params.agent ? "gui-agent" : "editor" });
    this.commands = createCommands(this);
    this.build();
    this.store.on("fatal", (msg) => this.fatal(msg));
    await this.store.connect();
    await ensureFonts([...(this.store.task.fonts_used || []), ...this.store.doc.elements.filter((e) => e.kind === "text").map((e) => e.font_family)]);
    document.title = `${this.store.task.title} · DesignGym`;
    this.stage.fit();
    this.stage.render();
    this.refreshAll();
    window.designEditor = {
      ready: true,
      store: this.store,
      app: this,
      whenIdle: (t) => this.store.whenIdle(t),
    };
  }

  build() {
    const root = document.getElementById("app");
    root.textContent = "";
    root.className = `app skin-${this.skin.name}`;
    this.skinApi = this.skin.build(this, root);
    this.stage = new Stage(this, this.skinApi.stageHost);
    this.presence = h("div", { class: "presence", hidden: true }, h("span", { class: "dot" }), h("span", { class: "who" }, "Agent editing"));
    this.skinApi.stageHost.append(this.presence);
    let queued = false;
    const refresh = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; this.refreshAll(); });
    };
    this.store.on("doc", refresh);
    this.store.on("selection", () => { this.frameSelected = false; if (this.freeTransform && !this.store.selection.length) this.freeTransform = false; });  // selecting a layer leaves the frame
    this.store.on("selection", refresh);
    this.store.on("tool", refresh);
    this.on("zoom", refresh);
    this.store.on("toast", (m) => toast(m, "error"));
    this.store.on("remote", (src) => this.showPresence(src));
    this.store.on("done", (result) => { if (!this._submitting) this.showResult(result); });
    this.store.on("connection", (ok) => document.documentElement.classList.toggle("offline", !ok));
    document.addEventListener("keydown", (e) => this.onKey(e));
    document.addEventListener("keyup", (e) => { if (e.code === "Space") { this.stage.space = false; this.stage.el.dataset.space = ""; } });
  }

  refreshAll() {
    if (!this.store?.doc) return;
    this.skinApi.refresh();
    if (this.skin.frameLabel) this.stage.drawOverlay();  // the frame's selected state lives on the canvas
    for (const p of this.skinApi.panels) if (!p.el || p.el.isConnected) p.refresh?.();
    for (const b of document.querySelectorAll("[data-command]")) {
      const c = this.commands[b.dataset.command];
      if (c && b.tagName === "BUTTON" && !b.closest(".menu")) {
        if (b.refresh) b.refresh();
        else b.disabled = c.enabled ? !c.enabled() : false;
      }
    }
  }

  // ------------------------------------------------------------ keyboard

  onKey(e) {
    const t = e.target;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
    if (typing) return;
    if (document.querySelector(".modal-back")) return;
    if (e.code === "Space" && !e.repeat) {
      this.stage.space = true;
      this.stage.el.dataset.space = "1";
      e.preventDefault();
      return;
    }
    const chord = eventChord(e);
    if (chord === "Escape" && this.stage.placing) {
      this.stage.stopPlacing();
      return;
    }
    if ((chord === "Enter" || chord === "Escape") && this.freeTransform) { this.setFreeTransform(false); return; }  // commit the transform
    if (chord === "Escape") {
      if (popoverOpen()) { closePopover(); return; }  // Escape closes a picker or menu first, as in every editor
      if (this.fileChooser) { this.fileChooser.close(); return; }  // then the file window
      if (this.frameSelected) { this.frameSelected = false; this.refreshAll(); return; }
      if (this.store.tool !== "move") this.store.setTool("move");
      else this.store.select([]);
      return;
    }
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key) && this.store.selection.length) {
      e.preventDefault();
      const d = e.shiftKey ? 10 : 1;
      const dx = e.key === "ArrowLeft" ? -d : e.key === "ArrowRight" ? d : 0;
      const dy = e.key === "ArrowUp" ? -d : e.key === "ArrowDown" ? d : 0;
      this.store.updateMany(this.store.selected().filter((el) => !el.locked).map((el) => [el.id, { x: el.x + dx, y: el.y + dy }]));
      return;
    }
    if (chord === "Enter" && this.store.selection.length === 1) {
      const el = this.store.find(this.store.selection[0]);
      if (el?.kind === "text") { e.preventDefault(); this.stage.startEdit(el.id); return; }
    }
    // Figma: with a layer selected, 1-9 set its opacity to 10-90% and 0 to 100%.
    if (this.skin.name === "figma" && /^[0-9]$/.test(chord) && this.store.selection.length) {
      e.preventDefault();
      const v = chord === "0" ? 1 : Number(chord) / 10;
      this.store.updateMany(this.store.selected().filter((el) => !el.locked).map((el) => [el.id, { opacity: v }]));
      return;
    }
    const alias = { "Mod+Y": "redo", "Mod+Shift+Z": "redo", "Mod+=": "zoomIn", "Mod+-": "zoomOut", "Mod+0": "zoomFit" };
    let cmd = alias[chord] ? this.commands[alias[chord]] : Object.values(this.commands).find((c) => c.key === chord);
    if (!cmd && chord === "Delete") cmd = this.commands.delete;
    if (cmd) {
      e.preventDefault();
      if (!cmd.enabled || cmd.enabled()) cmd.run();
    }
  }

  // ------------------------------------------------------------ adding

  defaultFill() { return this.usesFg() ? this.fgColor : this.skin.fill; }
  usesFg() { return this.skin.name === "photoshop" || this.skin.name === "inkscape"; }  // new objects take the current colour
  setFg(c) { this.fgColor = c; this.refreshAll(); }

  async addAsset(asset, point = null, box = null) {
    const doc = this.store.doc;
    const side = 0.4 * Math.min(doc.width, doc.height);
    let w = asset.aspect >= 1 ? side * asset.aspect : side, hh = asset.aspect >= 1 ? side : side / asset.aspect;
    // shrink to fit the page, proportions kept: a hairline asset would otherwise be thousands of pixels long
    const fit = Math.min(1, doc.width / w, doc.height / hh);
    w = Math.max(1, w * fit);
    hh = Math.max(1, hh * fit);
    const [cx, cy] = point || [doc.width / 2, doc.height / 2];
    const b = box || { x: cx - w / 2, y: cy - hh / 2, width: w, height: hh };
    const r = await this.store.op("add_image", { asset_id: asset.id, x: Math.round(b.x), y: Math.round(b.y),
      width: Math.round(b.width), height: Math.round(b.height) });
    if (r?.element_id) this.store.select(r.element_id);
    return r;
  }

  /** The file window, opened the way this editor opens it (Insert > Pictures, Place, Import, Upload). */
  openFiles(opts = {}) {
    return openFileChooser(this, {
      onOpen: (assets) => this.placeAssets(assets),
      onDrop: (assets, point) => this.placeAssets(assets, point, { dropped: true }),
      ...this.skinApi.fileOptions?.(), ...opts,
    });
  }

  /** Put chosen or dropped images into the design: each editor's own way, else one after another at the centre. */
  async placeAssets(assets, point = null, how = {}) {
    if (this.skinApi.placeAssets) return this.skinApi.placeAssets(assets, point, how);
    return this.placeAt(assets, point);
  }

  /** Default placement: at the drop point (or the centre), each further image a little down and right. */
  async placeAt(assets, point = null) {
    const ids = [];
    for (const [i, a] of assets.entries()) {
      const p = point ? [point[0] + 24 * i, point[1] + 24 * i] : null;
      const r = await this.addAsset(a, p);
      if (r?.element_id) ids.push(r.element_id);
    }
    if (ids.length) this.store.select(ids);
  }

  defaultFont() {
    return this.store.task?.fonts_used?.[0] || "Montserrat";
  }

  async addTextAt(x, y) {
    const size = Math.max(12, Math.round(this.store.doc.height / 20));
    const r = await this.store.op("add_text", { text: "Text", x: Math.round(x), y: Math.round(y - size / 2),
      font_family: this.defaultFont(), font_size: size, color: this.skin.name === "photoshop" ? this.fgColor : "#000000" });
    if (r?.element_id) {
      this.store.select(r.element_id);
      await this.store.whenIdle(500);
      this.stage.startEdit(r.element_id, { selectAll: true });
    }
  }

  async addTextPreset(props) {
    const r = await this.store.op("add_text", { font_family: this.defaultFont(), color: "#000000", ...props });
    if (r?.element_id) this.store.select(r.element_id);
    // Canva drops a text style onto the page already editing, its words selected: typing replaces them.
    if (r?.element_id && this.skin.name === "canva") requestAnimationFrame(() => this.stage.startEdit(r.element_id, { selectAll: true }));
  }

  async addShapePreset(shape, radiusFrac = 0) {
    const doc = this.store.doc;
    const side = Math.round(Math.min(doc.width, doc.height) * 0.3);
    const r = await this.store.op("add_shape", { shape, width: side, height: side, fill: this.defaultFill(), radius: Math.round(side * radiusFrac) });
    if (r?.element_id) this.store.select(r.element_id);
  }

  // ------------------------------------------------------------ menus

  floatingToolbar(rect) {
    const C = this.commands;
    const bar = h("div", { class: "floating-toolbar" },
      iconButton("lock", "Lock", () => C.lock.run()),
      iconButton("copy", "Duplicate", () => C.duplicate.run()),
      iconButton("trash-2", "Delete", () => C.delete.run()),
      iconButton("more-horizontal", "More", (e) => this.contextMenu(e.clientX, e.clientY, true)));
    const top = Math.max(8, rect.y0 - 48);
    Object.assign(bar.style, { left: `${(rect.x0 + rect.x1) / 2}px`, top: `${top}px` });
    bar.addEventListener("pointerdown", (e) => e.stopPropagation());
    return bar;
  }

  /** Photoshop: handles show with Show Transform Controls on, or during Free Transform (Mod+T). */
  showHandles() { return this.skin.name !== "photoshop" || this.transformControls || this.freeTransform; }
  setFreeTransform(on) { this.freeTransform = !!on; this.stage.drawOverlay(); this.refreshAll(); }

  /** Figma: the frame itself is selected (its Fill is the design's background). */
  selectFrame() {
    this.store.select([]);
    this.frameSelected = true;
    this.refreshAll();
  }

  /** The colour of the canvas around the design (Figma's page colour); it is not part of the design. */
  setCanvasColor(c) {
    this.canvasColor = c;
    this.stage.el.style.background = c;
    this.refreshAll();
  }

  contextMenu(x, y, onElement) {
    const C = this.commands;
    const own = this.skinApi.contextItems?.(onElement);
    if (own) { contextMenuAt(x, y, own); return; }
    const items = onElement
      ? [{ cmd: C.copy }, { cmd: C.cut }, { cmd: C.paste }, { cmd: C.duplicate }, { cmd: C.delete }, "-",
        { cmd: C.bringFront }, { cmd: C.bringForward }, { cmd: C.sendBackward }, { cmd: C.sendBack }, "-",
        { cmd: C.flipH }, { cmd: C.flipV }, { cmd: C.lock }, { cmd: C.hide }]
      : [{ cmd: C.paste }, { cmd: C.selectAll }, "-", { cmd: C.zoomFit }, { cmd: C.zoom100 },
        ...(this.showCode ? ["-", { cmd: C.html }] : [])];
    contextMenuAt(x, y, items);
  }

  toast(m) { toast(m); }

  showPresence(src) {
    if (src === "editor" || src === "rollback" || src === "local") return;
    this.presence.hidden = false;
    this.presence.querySelector(".who").textContent = src === "agent" ? "Agent editing" : src === "gui-agent" ? "Agent (GUI) editing" : `${src} editing`;
    clearTimeout(this._presenceT);
    this._presenceT = setTimeout(() => { this.presence.hidden = true; }, 1600);
  }

  // ------------------------------------------------------------ dialogs

  async openHtml() {
    if (!this.showCode) return;
    const res = await fetch(`/api/episodes/${this.store.episodeId}/html`);
    const text = await res.text();
    const area = h("textarea", { class: "code-area", spellcheck: "false", "aria-label": "HTML source" }, text);
    const msg = h("div", { class: "code-msg" }, "Every direct child of #canvas is one element, back to front. Apply replaces the canvas.");
    area.addEventListener("keydown", (e) => e.stopPropagation());
    modal("Edit design as HTML", h("div", { class: "code-wrap" }, area, msg), {
      cls: "modal-code",
      actions: [
        { label: "Copy", run: async () => { await navigator.clipboard?.writeText(area.value).catch(() => null); toast("Copied"); return true; } },
        { label: "Cancel" },
        { label: "Apply", primary: true, run: async () => {
          const r = await fetch(`/api/episodes/${this.store.episodeId}/html`, { method: "POST", headers: { "content-type": "text/plain" }, body: area.value });
          const body = await r.json();
          if (!r.ok) { msg.textContent = body.detail || "Rejected"; msg.classList.add("err"); return true; }
          toast(body.warnings?.length ? `Applied with ${body.warnings.length} warning(s)` : "Applied");
          return false;
        } },
      ],
    });
    area.focus();
  }

  exportPng() {
    const a = h("a", { href: `/api/episodes/${this.store.episodeId}/render.png?size=2048`, download: `${(this.store.task.title || "design").replace(/\W+/g, "-")}.png` });
    document.body.append(a);
    a.click();
    a.remove();
  }

  confirmSubmit() {
    if (this.store.done) { this.showResult(this.store.result); return; }
    const t = this.store.task || {};
    const name = `${(t.title || "design").replace(/[^\w\- ]+/g, "").trim() || "design"}.png`;
    const row = (k, v) => h("div", { class: "export-row" }, h("span", { class: "export-k" }, k), h("span", { class: "export-v" }, v));
    const body = h("div", { class: "export-body" },
      row("File name", name), row("File type", "PNG"),
      row("Size", t.canvas ? `${t.canvas.width} × ${t.canvas.height} px` : "Original"),
      h("p", { class: "modal-text" }, "Exporting hands in the finished design. It can't be edited afterwards."));
    modal("Export", body, {
      cls: "modal-small modal-export",
      actions: [
        { label: "Cancel" },
        { label: "Export", primary: true, run: async () => {
          this._submitting = true;
          try {
            const result = await this.store.submit();
            if (!this.params.agent) this.exportPng();  // people also get the file
            this.showResult(result);
          } catch (e) { toast(e.message, "error"); } finally { this._submitting = false; }
        } },
      ],
    });
  }

  showResult(r) {
    if (!r) return;
    const pct = (v) => `${Math.round(v * 1000) / 10}`;
    const bar = (label, v) => h("div", { class: "score-row" }, h("span", { class: "score-label" }, label),
      h("span", { class: "score-bar" }, h("span", { style: { width: `${Math.round(v * 100)}%` } })), h("span", { class: "score-val" }, pct(v)));
    const rows = [bar("Structure", r.structure), bar("Visual", r.visual)];
    if (r.mode === "description") rows.push(bar("Coverage", r.coverage));
    rows.push(bar("Order", r.order), bar("Background", r.background));
    const worst = [...r.elements].sort((a, b) => a.score - b.score).slice(0, 5).filter((e) => e.score < 0.95);
    const body = h("div", { class: "result" },
      h("div", { class: "result-top" }, h("div", { class: "result-score" }, pct(r.total), h("small", {}, "/ 100")),
        h("div", { class: "result-sub" }, `Matched ${r.n_matched} of ${r.n_gt} target elements · ${r.n_submitted} on the canvas`)),
      h("div", { class: "result-grid" },
        h("figure", {}, h("img", { src: this.store.task.reference_url || "", alt: "Target", hidden: !this.store.task.reference_url }), h("figcaption", {}, this.store.task.reference_url ? "Target" : "")),
        h("figure", {}, h("img", { src: `/api/episodes/${this.store.episodeId}/render.png?size=768&v=${this.store.version}`, alt: "Your design" }), h("figcaption", {}, "Your design"))),
      h("div", { class: "score-rows" }, rows),
      worst.length ? h("div", { class: "result-notes" }, h("div", { class: "panel-heading" }, "Biggest differences"),
        worst.map((e) => h("div", { class: "note" }, h("b", {}, e.gt_id), ` ${pct(e.score)} · ${e.note || "position / size"}`))) : null);
    modal("Design exported", body, { cls: "modal-result", actions: [{ label: "Close", primary: true }] });
  }

  showShortcuts() {
    const rows = Object.values(this.commands).filter((c) => c.key && !(c === this.commands.html && !this.showCode)).map((c) =>
      h("div", { class: "sc-row" }, h("span", {}, c.label), h("kbd", {}, shortcut(c.key))));
    rows.push(h("div", { class: "sc-row" }, h("span", {}, "Nudge (Shift = 10px)"), h("kbd", {}, "← ↑ → ↓")),
      h("div", { class: "sc-row" }, h("span", {}, "Pan"), h("kbd", {}, "Space + drag")),
      h("div", { class: "sc-row" }, h("span", {}, "Zoom"), h("kbd", {}, `${shortcut("Mod")} + scroll`)),
      h("div", { class: "sc-row" }, h("span", {}, "Edit text"), h("kbd", {}, "Double-click / Enter")));
    modal("Keyboard shortcuts", h("div", { class: "sc-grid" }, rows), { cls: "modal-small" });
  }

  fatal(msg) {
    const root = document.getElementById("app");
    root.textContent = "";
    root.append(h("div", { class: "fatal" }, h("div", { html: icon("triangle-alert", 32) }), h("h2", {}, "This episode is not available"),
      h("p", {}, msg), h("a", { class: "btn btn-primary", href: "/web/" }, "Open the playground")));
  }
}

const app = new App();
window.__app = app;
app.start().catch((e) => { console.error(e); app.fatal?.(String(e)); });
