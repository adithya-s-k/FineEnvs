// The computer's file chooser, drawn inside the page.
//
// Real editors bring images in through the operating system: Insert > Pictures >
// This Device, File > Place Embedded, File > Import, Upload files. A headless
// browser has no native dialog to show, so the editor draws one, in the look of
// macOS, Windows or GNOME (chosen per episode), holding the task's images as
// files. Like a real file window it moves by its title bar, resizes from its
// edges, and its files can be dragged straight onto the canvas.

import { clamp, h } from "./util.js";
import { icon } from "./icons.js";

const OSES = ["mac", "windows", "linux"];
const FOLDERS = ["Downloads", "Desktop", "Pictures", "Documents"];

function hash(text) {
  let x = 2166136261;
  for (let i = 0; i < text.length; i++) x = Math.imul(x ^ text.charCodeAt(i), 16777619);
  return x >>> 0;
}

// Sidebar places, in each system's own order and wording.
const PLACES = {
  mac: [
    ["Favorites", [["recents", "Recents", "clock"], ["desktop", "Desktop", "monitor"], ["documents", "Documents", "file"],
      ["downloads", "Downloads", "download"], ["pictures", "Pictures", "image"]]],
    ["Locations", [["icloud", "iCloud Drive", "cloud"], ["disk", "Macintosh HD", "hard-drive"]]],
  ],
  windows: [
    ["", [["home", "Home", "house"]]],
    ["", [["desktop", "Desktop", "monitor"], ["downloads", "Downloads", "download"], ["documents", "Documents", "file"],
      ["pictures", "Pictures", "image"]]],
    ["", [["disk", "This PC", "hard-drive"]]],
  ],
  linux: [
    ["", [["recents", "Recent", "clock"], ["home", "Home", "house"], ["desktop", "Desktop", "monitor"],
      ["documents", "Documents", "file"], ["downloads", "Downloads", "download"], ["pictures", "Pictures", "image"]]],
    ["", [["disk", "Other Locations", "hard-drive"]]],
  ],
};
const placeNames = (os) => Object.fromEntries(PLACES[os].flatMap(([, items]) => items).map(([id, name]) => [id, name]));

/** Whether this episode starts with the task's images already uploaded (Canva's Uploads). */
export function preUploaded(app) {
  if (app.store?.workspace) return false;  // on a lived-in computer the images are still to be found and uploaded
  return hash(`${app.store?.episodeId || "episode"}:uploads`) % 2 === 0;
}

/** Which system, folder and view this episode's chooser uses (stable for the episode). */
export function chooserLook(app) {
  const key = app.params.fs || app.store?.episodeId || "episode";
  const os = OSES.includes(app.params.os) ? app.params.os : OSES[hash(`${key}:os`) % OSES.length];
  const r = hash(`${key}:files`);
  // A workspace (the env's core.workspace) says which folder holds the images; otherwise one is drawn per episode.
  const ws = app.store?.workspace;
  const base = ws ? ws.folder.split("/")[0] : null;
  const folder = base && FOLDERS.includes(base) ? base : FOLDERS[r % FOLDERS.length];
  return { os, folder, view: (r >> 4) % 2 ? "list" : "grid", offset: ((r >> 8) % 60) - 30 };
}

// A document of a workspace (a brief, an invoice): listed like any file, never opened as an image.
function docFor(path, os) {
  const name = path.split("/").pop(), ext = (name.split(".").pop() || "").toLowerCase(), r = hash(path);
  const kinds = { pdf: "PDF document", docx: "Word document", xlsx: "Spreadsheet", zip: "Zip archive", txt: "Plain text" };
  const mins = (r >> 5) % 600, hh = 8 + Math.floor(mins / 60), mm = String(mins % 60).padStart(2, "0");
  const ampm = hh >= 12 ? "PM" : "AM", h12 = ((hh + 11) % 12) + 1;
  const date = { mac: `Yesterday at ${h12}:${mm} ${ampm}`, windows: `23/09/2026 ${String(hh).padStart(2, "0")}:${mm}`, linux: `Yesterday` }[os];
  return { type: "doc", id: `doc:${path}`, name, size: `${Math.round(4 + (r % 900))} KB`, date, kind: kinds[ext] || "Document" };
}

function fileFor(asset, os) {
  const r = hash(asset.id);
  const photo = asset.kind === "photo" || asset.kind === "background";
  const ext = photo ? "jpg" : "png";
  const bytes = asset.bytes || (photo ? 900_000 + (r % 3_600_000) : 30_000 + (r % 820_000));
  const mins = (r >> 5) % 600, hh = 8 + Math.floor(mins / 60), mm = String(mins % 60).padStart(2, "0");
  const size = bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1e3)} KB`;
  const ampm = hh >= 12 ? "PM" : "AM", h12 = ((hh + 11) % 12) + 1;
  const date = { mac: `Today at ${h12}:${mm} ${ampm}`, windows: `24/09/2026 ${String(hh).padStart(2, "0")}:${mm}`, linux: `${hh}:${mm}` }[os];
  const kind = { mac: photo ? "JPEG image" : "PNG image", windows: photo ? "JPG File" : "PNG File", linux: photo ? "JPEG image" : "PNG image" }[os];
  return { type: "file", id: asset.id, name: `${asset.id}.${ext}`, asset, size, date, kind };
}

/** Open (or bring forward) the chooser. `onOpen(assets)` runs on Open; `onDrop(assets, point)` on a drop onto the canvas. */
export function openFileChooser(app, { title = "Open", action = "Open", multiple = true, onOpen, onDrop } = {}) {
  if (app.fileChooser?.el.isConnected) {
    app.fileChooser.retarget({ title, action, onOpen, onDrop });
    app.fileChooser.focus();
    return app.fileChooser;
  }
  const chooser = new FileChooser(app, { title, action, multiple, onOpen, onDrop });
  app.fileChooser = chooser;
  return chooser;
}

class FileChooser {
  constructor(app, opts) {
    this.app = app;
    this.opts = opts;
    const look = chooserLook(app);
    this.os = look.os;
    this.names = placeNames(look.os);
    this.view = look.view;
    this.home = look.folder.toLowerCase();
    this.ws = app.store.workspace || null;
    if (this.ws) {  // a lived-in computer: the window opens where such windows open, or where it was last left
      const start = { linux: ["recents", "home"], windows: ["pictures", "documents"], mac: ["recents", "documents"] }[this.os];
      this.home = start[hash(`${app.store?.episodeId || "episode"}:start`) % 2];
    }
    this.place = this.ws && app.lastPlace ? app.lastPlace : this.home;
    // the window's file type filter: images only, or everything (then documents are listed too)
    this.allFiles = !!this.ws && hash(`${app.store?.episodeId || "episode"}:filter`) % 3 === 0;
    this.history = [this.place];
    this.hi = 0;
    this.selected = new Set();
    this.anchor = null;
    this.query = "";
    if (this.ws) {  // everyday names in everyday folders, decoy pictures and documents (core.workspace)
      this.tree = {};  // folder id ("downloads/spring sale") -> its parent's id (null for a home folder)
      const addDir = (path) => {
        let id = "";
        path.split("/").forEach((part, i) => {
          const parent = i ? id : null;
          id = i ? `${id}/${part.toLowerCase()}` : part.toLowerCase();
          if (!(id in this.tree)) this.tree[id] = parent;
          if (!this.names[id]) this.names[id] = part;
        });
        return id;
      };
      FOLDERS.forEach(addDir);
      const byId = Object.fromEntries(app.store.assets.map((a) => [a.id, a]));
      this.files = this.ws.files.filter((f) => byId[f.asset]).map((f) => {
        const file = fileFor(byId[f.asset], this.os);
        const ext = (f.name.split(".").pop() || "").toLowerCase();
        const photo = ext === "jpg" || ext === "jpeg";
        const kind = { mac: photo ? "JPEG image" : "PNG image", windows: photo ? "JPG File" : "PNG File", linux: photo ? "JPEG image" : "PNG image" }[this.os];
        return { ...file, name: f.name, kind, dir: addDir(f.folder || this.ws.folder) };
      });
      this.docs = (this.ws.others || []).map((o) => ({ ...docFor(o.path, this.os), dir: addDir(o.path.split("/").slice(0, -1).join("/")) }));
    } else {
      this.files = app.store.assets.map((a) => fileFor(a, this.os));
    }
    this.build(look);
    this.render();
    this.focus();
  }

  // ------------------------------------------------------------ contents

  entries() {
    const q = this.query.trim().toLowerCase();
    let items = [];
    if (this.ws) {  // a folder lists its folders, then its files by name, as file windows do
      const byName = (a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base", numeric: true });
      if (this.place === "recents") items = this.docs.slice(0, 3);
      else if (this.place === "home" || this.place === "disk") items = FOLDERS.map((f) => ({ type: "folder", id: f.toLowerCase(), name: f }));
      else {
        const dirs = Object.keys(this.tree).filter((id) => this.tree[id] === this.place)
          .map((id) => ({ type: "folder", id, name: this.names[id] })).sort(byName);
        // an image filter hides documents (GNOME, Windows); the Mac lists them greyed out
        const docs = this.os === "mac" || this.allFiles ? this.docs.filter((d) => d.dir === this.place) : [];
        items = [...dirs, ...[...this.files.filter((f) => f.dir === this.place), ...docs].sort(byName)];
      }
      return q ? items.filter((e) => e.name.toLowerCase().includes(q)) : items;
    }
    if (this.place === this.home || this.place === "recents" || (this.os === "windows" && this.place === "home")) items = this.files;
    else if (this.place === "home" || this.place === "disk") {
      items = FOLDERS.map((f) => ({ type: "folder", id: f.toLowerCase(), name: f }));
    }
    return q ? items.filter((e) => e.name.toLowerCase().includes(q)) : items;
  }

  go(place, push = true) {
    if (!this.names[place]) return;
    this.place = place;
    this.selected.clear();
    this.query = "";
    if (this.search) this.search.value = "";
    if (push) {
      this.history = this.history.slice(0, this.hi + 1).concat(place);
      this.hi = this.history.length - 1;
    }
    this.render();
  }

  // ------------------------------------------------------------ building

  build(look) {
    const os = this.os;
    const btn = (ic, title, onclick, cls = "") => h("button", { class: `fc-btn ${cls}`, type: "button", title, "aria-label": title, onclick,
      html: icon(ic, 15) });
    this.backBtn = btn("chevron-left", "Back", () => { if (this.hi > 0) { this.hi--; this.go(this.history[this.hi], false); } });
    this.fwdBtn = btn("chevron-right", "Forward", () => { if (this.hi < this.history.length - 1) { this.hi++; this.go(this.history[this.hi], false); } });
    this.viewBtns = [["grid", "layout-grid", os === "mac" ? "as Icons" : "Large icons"], ["list", "list", os === "mac" ? "as List" : "Details"]]
      .map(([v, ic, title]) => {
        const b = btn(ic, title, () => { this.view = v; this.render(); }, "fc-view");
        b.dataset.view = v;
        return b;
      });
    this.search = h("input", { class: "fc-search", type: "search", spellcheck: "false", "aria-label": "Search" });
    this.search.addEventListener("input", () => { this.query = this.search.value; this.render(); });
    this.list = h("div", { class: "fc-files", role: "listbox", "aria-multiselectable": String(this.opts.multiple), tabindex: "0" });
    this.list.addEventListener("pointerdown", (e) => { if (e.target === this.list) { this.selected.clear(); this.sync(); } });
    this.side = h("nav", { class: "fc-side", "aria-label": "Places" });
    for (const [section, items] of PLACES[os]) {
      const group = h("div", { class: "fc-side-group" }, section ? h("div", { class: "fc-side-title" }, section) : null);
      for (const [id, name, ic] of items) {
        group.append(h("button", { class: "fc-place", type: "button", dataset: { place: id }, onclick: () => this.go(id) },
          h("span", { class: "fc-place-ic", html: icon(ic, 15) }), h("span", {}, name)));
      }
      this.side.append(group);
    }
    this.openBtn = h("button", { class: "fc-action fc-open", type: "button", dataset: { fc: "open" }, onclick: () => this.commit() }, this.opts.action);
    this.cancelBtn = h("button", { class: "fc-action fc-cancel", type: "button", dataset: { fc: "cancel" }, onclick: () => this.close() }, "Cancel");
    const grow = () => h("span", { class: "grow" });

    let top, foot;
    if (os === "mac") {
      this.search.placeholder = "Search";
      this.pathSel = h("select", { class: "fc-path", "aria-label": "Folder" });
      this.pathSel.addEventListener("change", () => this.go(this.pathSel.value));
      const lights = h("div", { class: "fc-lights" },
        h("button", { class: "fc-light close", type: "button", title: "Close", "aria-label": "Close", dataset: { fc: "close" }, onclick: () => this.close() }),
        h("span", { class: "fc-light off" }), h("span", { class: "fc-light off" }));
      top = h("div", { class: "fc-top fc-drag" }, lights, this.backBtn, this.fwdBtn, h("div", { class: "fc-seg" }, ...this.viewBtns),
        grow(), this.pathSel, grow(), h("div", { class: "fc-search-box" }, h("span", { html: icon("search", 13) }), this.search));
      foot = h("div", { class: "fc-foot" }, grow(), this.cancelBtn, this.openBtn);
    } else if (os === "windows") {
      this.crumb = h("div", { class: "fc-crumb" });
      this.nameInput = h("input", { class: "fc-name", spellcheck: "false", "aria-label": "File name" });
      this.nameInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); this.commit(); } });
      this.nameInput.addEventListener("input", () => { this.selected.clear(); this.sync(); });
      const filter = h("select", { class: "fc-filter", "aria-label": "File type" }, h("option", {}, this.allFiles ? "All files (*.*)" : "All Pictures (*.png;*.jpg;*.jpeg)"));
      top = h("div", {},
        h("div", { class: "fc-title fc-drag" }, h("span", { class: "fc-app", html: icon("app-window", 14) }), h("span", {}, this.opts.title), grow(),
          h("button", { class: "fc-winclose", type: "button", title: "Close", "aria-label": "Close", dataset: { fc: "close" }, onclick: () => this.close(),
            html: icon("x", 16) })),
        h("div", { class: "fc-nav" }, this.backBtn, this.fwdBtn, this.crumb,
          h("div", { class: "fc-search-box" }, this.search, h("span", { html: icon("search", 13) })), h("div", { class: "fc-seg" }, ...this.viewBtns)));
      foot = h("div", { class: "fc-foot" }, h("label", { class: "fc-name-label" }, "File name:", this.nameInput), filter,
        h("div", { class: "fc-foot-btns" }, this.openBtn, this.cancelBtn));
    } else {
      this.crumb = h("div", { class: "fc-crumb" });
      this.search.placeholder = "Search";
      const filter = h("select", { class: "fc-filter", "aria-label": "File type" }, h("option", {}, this.allFiles ? "All Files" : "Image Files"));
      top = h("div", {},
        h("div", { class: "fc-head fc-drag" }, this.cancelBtn, grow(), h("span", { class: "fc-head-title" }, this.opts.title === "Open" ? "Open File" : this.opts.title),
          grow(), this.openBtn),
        h("div", { class: "fc-nav" }, this.backBtn, this.fwdBtn, this.crumb, grow(),
          h("div", { class: "fc-search-box" }, h("span", { html: icon("search", 13) }), this.search), h("div", { class: "fc-seg" }, ...this.viewBtns)));
      foot = h("div", { class: "fc-foot" }, grow(), filter);
    }
    // The selected picture, large: GNOME's window shows it beside the list; Windows has a preview pane to switch on
    // (Alt+P, or its button); the Mac's has none (its icons are the pictures).
    this.preview = os === "mac" ? null : h("aside", { class: "fc-preview", "aria-label": "Preview" });
    this.showPreview = os === "linux";
    if (os === "windows") {
      this.previewBtn = btn("panel-right", "Show the preview pane", () => { this.showPreview = !this.showPreview; this.sync(); }, "fc-preview-btn");
      top.querySelector(".fc-nav").append(this.previewBtn);
    }
    const handles = ["n", "s", "e", "w", "ne", "nw", "se", "sw"].map((d) => h("div", { class: `fc-rz fc-rz-${d}`, dataset: { dir: d } }));
    this.el = h("div", { class: `fc fc-${os}`, role: "dialog", "aria-label": this.opts.title, tabindex: "-1", dataset: { os } },
      h("div", { class: "fc-frame" }, top, h("div", { class: "fc-main" }, this.side, this.list, this.preview), foot), ...handles);
    const W = Math.min(820, innerWidth - 40), H = Math.min(520, innerHeight - 60);
    this.box = { x: Math.round((innerWidth - W) / 2 + look.offset), y: Math.round(Math.max(20, (innerHeight - H) / 2 - 30 + look.offset / 2)), w: W, h: H };
    this.place_();
    document.body.append(this.el);
    this.el.addEventListener("keydown", (e) => this.onKey(e));
    for (const d of this.el.querySelectorAll(".fc-drag")) d.addEventListener("pointerdown", (e) => this.startMove(e));
    for (const r of handles) r.addEventListener("pointerdown", (e) => this.startResize(e, r.dataset.dir));
  }

  place_() {
    const b = this.box;
    Object.assign(this.el.style, { left: `${b.x}px`, top: `${b.y}px`, width: `${b.w}px`, height: `${b.h}px` });
  }

  // ------------------------------------------------------------ window: move and resize

  startMove(e) {
    if (e.button !== 0 || e.target.closest("button, input, select")) return;
    e.preventDefault();
    const x0 = e.clientX, y0 = e.clientY, b0 = { ...this.box };
    const move = (ev) => {
      this.box.x = clamp(b0.x + ev.clientX - x0, 40 - b0.w, innerWidth - 60);
      this.box.y = clamp(b0.y + ev.clientY - y0, 0, innerHeight - 40);
      this.place_();
    };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  }

  startResize(e, dir) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const x0 = e.clientX, y0 = e.clientY, b0 = { ...this.box };
    const MINW = 460, MINH = 300;
    const move = (ev) => {
      const dx = ev.clientX - x0, dy = ev.clientY - y0, b = { ...b0 };
      if (dir.includes("e")) b.w = Math.max(MINW, b0.w + dx);
      if (dir.includes("s")) b.h = Math.max(MINH, b0.h + dy);
      if (dir.includes("w")) { b.w = Math.max(MINW, b0.w - dx); b.x = b0.x + b0.w - b.w; }
      if (dir.includes("n")) { b.h = Math.max(MINH, b0.h - dy); b.y = Math.max(0, b0.y + b0.h - b.h); }
      this.box = b;
      this.place_();
    };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  }

  // ------------------------------------------------------------ rendering

  render() {
    const entries = this.entries();
    this.el.dataset.view = this.view;
    this.list.className = `fc-files fc-${this.view}`;
    this.list.textContent = "";
    if (this.view === "list") {
      this.list.append(h("div", { class: "fc-row fc-row-head", "aria-hidden": "true" },
        h("span", { class: "fc-col-name" }, "Name"), h("span", { class: "fc-col-date" }, this.os === "mac" ? "Date Modified" : "Modified"),
        h("span", { class: "fc-col-size" }, "Size"), h("span", { class: "fc-col-kind" }, this.os === "windows" ? "Type" : "Kind")));
    }
    for (const e of entries) this.list.append(this.item(e));
    if (!entries.length) this.list.append(h("div", { class: "fc-empty" }, this.query ? "No results" : this.os === "windows" ? "This folder is empty." : "No items"));
    for (const p of this.side.querySelectorAll(".fc-place")) p.classList.toggle("on", p.dataset.place === this.place);
    for (const b of this.viewBtns) b.classList.toggle("on", b.dataset.view === this.view);
    this.backBtn.disabled = this.hi <= 0;
    this.fwdBtn.disabled = this.hi >= this.history.length - 1;
    const name = this.names[this.place];
    if (this.pathSel) {
      this.pathSel.textContent = "";
      for (const [id, label] of Object.entries(this.names)) this.pathSel.append(h("option", { value: id, selected: id === this.place }, label));
    }
    if (this.crumb) {
      this.crumb.textContent = "";
      const inner = this.ws && this.place in this.tree  // a folder inside a home folder: its whole path
        ? this.place.split("/").map((_, i, a) => this.names[a.slice(0, i + 1).join("/")]) : [name];
      const trail = this.os === "windows"
        ? (this.place === "home" ? ["Home"] : this.place === "disk" ? ["This PC"] : ["This PC", ...inner])
        : (this.place === "home" ? ["user"] : this.place === "disk" ? ["Other Locations"] : ["user", ...inner]);
      trail.forEach((t, i) => {
        if (i) this.crumb.append(h("span", { class: "fc-crumb-sep", html: icon("chevron-right", 12) }));
        const home = this.os === "linux" && t === "user";
        this.crumb.append(h("span", { class: "fc-crumb-part" }, home ? h("span", { html: icon("house", 13) }) : null, t));
      });
    }
    if (this.search) this.search.placeholder = this.os === "windows" ? `Search ${name}` : "Search";
    this.sync();
  }

  // Selection changes restyle the items in place: rebuilding them under the pointer would swallow
  // the click (and so the double-click) that is still in progress.
  sync() {
    for (const el of this.list.querySelectorAll(".fc-item")) {
      const on = this.selected.has(el.dataset.file);
      el.classList.toggle("on", on);
      el.setAttribute("aria-selected", String(on));
    }
    const files = [...this.selected].map((id) => this.files.find((f) => f.id === id)).filter(Boolean);
    if (this.preview) {
      this.preview.hidden = !this.showPreview;
      this.previewBtn?.classList.toggle("on", this.showPreview);
      const one = files.length === 1 ? files[0] : null;
      if ((this.preview.dataset.file || "") !== (one?.id || "")) {
        this.preview.dataset.file = one?.id || "";
        this.preview.textContent = "";
        if (one) {
          const img = h("img", { src: one.asset.url, alt: one.name, draggable: "false" });
          const dims = h("div", { class: "fc-preview-dims" });
          if (one.asset.w) dims.textContent = `${one.asset.w} × ${one.asset.h}`;  // the file's own size, not the thumbnail's
          else img.addEventListener("load", () => { dims.textContent = `${img.naturalWidth} × ${img.naturalHeight}`; });
          this.preview.append(h("div", { class: "fc-preview-img" }, img), h("div", { class: "fc-preview-name" }, one.name), dims);
        }
      }
    }
    this.openBtn.disabled = !files.length && !(this.nameInput && this.nameInput.value.trim());
    if (this.nameInput && document.activeElement !== this.nameInput) {
      this.nameInput.value = files.length > 1 ? files.map((f) => `"${f.name}"`).join(" ") : files[0]?.name || "";
    }
  }

  item(e) {
    const sel = this.selected.has(e.id);
    const thumb = e.type === "file"
      ? h("div", { class: "fc-thumb" }, h("img", { src: e.asset.url, alt: "", draggable: "false", loading: "lazy" }))
      : e.type === "doc"
        ? h("div", { class: "fc-thumb fc-doc", html: icon("file", this.view === "grid" ? 40 : 16) })
        : h("div", { class: "fc-thumb fc-folder", html: icon("folder", this.view === "grid" ? 44 : 16) });
    const el = this.view === "grid"
      ? h("div", { class: `fc-item${sel ? " on" : ""}`, role: "option", "aria-selected": String(sel), dataset: { file: e.id, type: e.type }, title: e.name },
        thumb, h("div", { class: "fc-item-name" }, e.name))
      : h("div", { class: `fc-row fc-item${sel ? " on" : ""}`, role: "option", "aria-selected": String(sel), dataset: { file: e.id, type: e.type }, title: e.name },
        h("span", { class: "fc-col-name" }, thumb, h("span", { class: "fc-item-name" }, e.name)),
        h("span", { class: "fc-col-date" }, e.type !== "folder" ? e.date : "--"), h("span", { class: "fc-col-size" }, e.type !== "folder" ? e.size : "--"),
        h("span", { class: "fc-col-kind" }, e.type !== "folder" ? e.kind : "Folder"));
    el.addEventListener("pointerdown", (ev) => this.onItemDown(ev, e));
    el.addEventListener("dblclick", () => {
      if (e.type === "folder") this.go(e.id);
      else if (e.type === "file") { this.selected = new Set([e.id]); this.commit(); }
    });
    return el;
  }

  // ------------------------------------------------------------ selection, opening, dragging out

  onItemDown(ev, e) {
    if (ev.button !== 0) return;
    ev.stopPropagation();
    const ids = this.entries().map((x) => x.id);
    const mod = ev.metaKey || ev.ctrlKey;
    if (ev.shiftKey && this.anchor && this.opts.multiple) {
      const [a, b] = [ids.indexOf(this.anchor), ids.indexOf(e.id)].sort((p, q) => p - q);
      this.selected = new Set(ids.slice(a, b + 1));
    } else if (mod && this.opts.multiple) {
      if (this.selected.has(e.id)) this.selected.delete(e.id); else this.selected.add(e.id);
      this.anchor = e.id;
    } else if (!this.selected.has(e.id)) {
      this.selected = new Set([e.id]);
      this.anchor = e.id;
    }
    const wasOn = this.selected.has(e.id);
    this.sync();
    if (e.type !== "file") return;
    const x0 = ev.clientX, y0 = ev.clientY;
    let ghost = null;
    const dragged = () => [...this.selected].map((id) => this.files.find((f) => f.id === id)).filter(Boolean);
    const move = (m) => {
      if (!ghost && Math.hypot(m.clientX - x0, m.clientY - y0) > 5) {
        const fs = dragged();
        ghost = h("div", { class: "drag-ghost fc-ghost" }, h("img", { src: fs[0].asset.url, alt: "" }),
          fs.length > 1 ? h("span", { class: "fc-ghost-count" }, String(fs.length)) : null);
        document.body.append(ghost);
      }
      if (ghost) Object.assign(ghost.style, { left: `${m.clientX}px`, top: `${m.clientY}px` });
    };
    const up = async (u) => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      if (!ghost) {
        if (wasOn && !ev.shiftKey && !mod && this.selected.size > 1) { this.selected = new Set([e.id]); this.sync(); }
        return;
      }
      ghost.remove();
      if (this.el.contains(document.elementFromPoint(u.clientX, u.clientY))) return;  // dropped back into the window
      const point = this.app.stage.dropPoint(u.clientX, u.clientY);
      if (point) await (this.opts.onDrop || ((assets, p) => this.app.placeAssets(assets, p)))(dragged().map((f) => f.asset), point);
    };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  }

  async commit() {
    let files = [...this.selected].map((id) => this.files.find((f) => f.id === id)).filter(Boolean);
    if (!files.length && this.nameInput) {  // typed into Windows' "File name" box
      const want = this.nameInput.value.split('"').map((s) => s.trim()).filter(Boolean);
      files = this.files.filter((f) => want.includes(f.name) || want.includes(f.id));
    }
    const folder = [...this.selected].map((id) => this.entries().find((x) => x.id === id)).find((x) => x?.type === "folder");
    if (!files.length && folder) { this.go(folder.id); return; }
    if (!files.length) return;
    if (this.ws) this.app.lastPlace = this.place;  // the next file window opens here, as real ones do
    this.close();
    await (this.opts.onOpen || ((assets) => this.app.placeAssets(assets)))(files.map((f) => f.asset));
  }

  onKey(e) {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); this.close(); }
    else if (this.previewBtn && e.altKey && e.key.toLowerCase() === "p") { e.preventDefault(); this.showPreview = !this.showPreview; this.sync(); }
    else if (e.key === "Enter" && e.target !== this.nameInput) { e.preventDefault(); e.stopPropagation(); this.commit(); }
    else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "a" && e.target.tagName !== "INPUT") {
      e.preventDefault(); e.stopPropagation();
      this.selected = new Set(this.entries().filter((x) => x.type === "file").map((x) => x.id));
      this.sync();
    }
    e.stopPropagation();  // keys typed here are the chooser's, not the editor's shortcuts
  }

  retarget(opts) { Object.assign(this.opts, opts); this.openBtn.textContent = this.opts.action; }
  focus() { this.el.style.zIndex = "900"; this.el.focus({ preventScroll: true }); }
  close() {
    this.el.remove();
    if (this.app.fileChooser === this) this.app.fileChooser = null;
    this.app.refreshAll?.();
  }
}
