// The file viewer: a folder tree on the left, the open file on the right, like an editor. Used on task pages (Harbor
// and custom environments alike). Contents are fetched on click (and on hover, a moment early), never shipped up front:
// a task can hold thousands of files and a person reads three.
//
//   fileViewer(root, { files, load, initial, line, hubUrl, rawUrl, onChange })
//     files    [{path, size, withheld}]; withheld files are listed with a lock and never opened
//     load     async path -> {text, binary, withheld, truncated, size, error}
//     initial  the path to open first; line: "12" or "12-20" to highlight and scroll to
//     hubUrl   path -> link to the file on the Hub (or null); rawUrl: path -> image URL for previews (or null)
//     onChange (path, line) after the open file or the highlighted lines change, for the page's URL
//     listDir  async folder -> [{path, size, withheld, dir, lazy}]: lists a folder the index left unlisted (entries
//              with dir+lazy, path ending in "/"), when it's opened
import { esc, fmt, bytes, md, toast, storage } from "./util.js";
import { icon } from "./icons.js";
import { highlight, langOf, LANG_NAMES } from "./code.js";

const IMAGE = /\.(png|jpe?g|gif|webp|svg|bmp|ico|avif)$/i;
const HIGHLIGHT_MAX = 400_000;   // characters; past this a file is shown plain, so a huge log can't stall the page
const ICON_BY_LANG = { sh: "terminal", docker: "box", md: "doc", json: "list", toml: "list", yaml: "list", ini: "list", diff: "diff" };

const fileIcon = (f) => (f.withheld ? "lock" : IMAGE.test(f.path) ? "image" : /\.(csv|tsv|parquet|xlsx?)$/i.test(f.path) ? "sheet"
  : ICON_BY_LANG[langOf(f.path)] || (langOf(f.path) ? "code" : "file"));
const base = (p) => p.split("/").pop();
const lineCount = (t) => (t === "" ? 0 : t.split("\n").length - (t.endsWith("\n") ? 1 : 0));

// path list -> nested folders (folders first, then files, each by name, like an editor); `folders` adds known
// folders whose contents aren't listed yet
function buildTree(files, folders = []) {
  const root = { dirs: new Map(), files: [], count: 0 };
  for (const d of folders) {
    let node = root;
    for (const part of d.split("/")) {
      if (!node.dirs.has(part)) node.dirs.set(part, { dirs: new Map(), files: [], count: 0 });
      node = node.dirs.get(part);
    }
  }
  for (const f of files) {
    const parts = f.path.split("/");
    let node = root;
    node.count++;
    for (const d of parts.slice(0, -1)) {
      if (!node.dirs.has(d)) node.dirs.set(d, { dirs: new Map(), files: [], count: 0 });
      node = node.dirs.get(d);
      node.count++;
    }
    node.files.push(f);
  }
  return root;
}

// a text range "12" / "12-20" -> [from, to]
const parseLines = (s) => {
  const m = String(s || "").match(/^(\d+)(?:-(\d+))?$/);
  if (!m) return null;
  const a = +m[1], b = +(m[2] || m[1]);
  return [Math.min(a, b), Math.max(a, b)];
};

export function fileViewer(root, { files: given, load, initial, line, hubUrl = () => null, rawUrl = () => null, onChange = () => {}, listDir = null }) {
  const cache = new Map();         // path -> Promise of the loaded file
  const get = (p) => { if (!cache.has(p)) cache.set(p, load(p).catch((e) => ({ error: e.message || String(e) }))); return cache.get(p); };
  let files = given.filter((f) => !f.dir);
  const lazy = new Map(given.filter((f) => f.dir && f.lazy).map((f) => [f.path.replace(/\/$/, ""), "unlisted"]));   // folder -> unlisted | loading | listed | error
  let byPath = new Map(files.map((f) => [f.path, f]));
  let tree = buildTree(files, [...lazy.keys()]);
  const many = files.length > 40;
  // folders start open when there are few files; otherwise only the ones on the way to the open file
  const open = new Set();
  const walk = (node, prefix) => node.dirs.forEach((child, name) => { const p = prefix + name; if (!many && !lazy.has(p)) open.add(p); walk(child, p + "/"); });
  walk(tree, "");
  let current = null, lines = null, wrap = storage.get("fv-wrap") === "1", mdMode = storage.get("fv-md") || "preview", query = "";
  let full = false, hoverT = null;

  root.classList.add("fv-host");   // the viewer fits the column it's in (container queries in fv.css)
  root.innerHTML = `<div class="fv">
    <div class="fv-side">
      <button class="fv-pick" type="button" aria-expanded="false">${icon("folder", 14)}<span class="fv-pick-p"></span>${icon("chevronDown", 14, "chev")}</button>
      ${files.length > 12 ? `<label class="fv-filter">${icon("search", 13)}<input type="search" placeholder="Filter ${fmt.format(files.length)} files" aria-label="Filter files" spellcheck="false" autocomplete="off"></label>` : ""}
      <div class="fv-tree" role="tree" aria-label="Files"></div>
    </div>
    <div class="fv-main">
      <div class="fv-bar">
        <div class="fv-path"></div>
        <span class="fv-meta"></span>
        <div class="fv-tools">
          <span class="seg sm fv-mdseg" hidden><button type="button" data-md="preview">Preview</button><button type="button" data-md="code">Code</button></span>
          <button class="icon-btn sm" type="button" data-act="wrap" aria-pressed="${wrap}" title="Wrap long lines">${icon("wrap", 15)}</button>
          <button class="icon-btn sm" type="button" data-act="copy" title="Copy the file">${icon("copy", 15)}</button>
          <button class="icon-btn sm" type="button" data-act="link" title="Copy a link to this file">${icon("link", 15)}</button>
          <a class="icon-btn sm" data-act="hub" target="_blank" rel="noopener" title="Open on the Hub" hidden>${icon("external", 15)}</a>
          <button class="icon-btn sm" type="button" data-act="full" title="Full view (Esc to close)">${icon("maximize", 15)}</button>
        </div>
      </div>
      <div class="fv-body" tabindex="-1"></div>
    </div>
  </div>`;
  const el = root.querySelector(".fv"), treeEl = el.querySelector(".fv-tree"), body = el.querySelector(".fv-body");
  const filterIn = el.querySelector(".fv-filter input");
  const pick = el.querySelector(".fv-pick");

  // ── the tree ──
  function rowsOf(node, prefix, depth) {
    let html = "";
    for (const [name, child] of [...node.dirs].sort((a, b) => a[0].localeCompare(b[0]))) {
      const p = prefix + name, isOpen = open.has(p), st = lazy.get(p);
      const count = st === "loading" ? `<span class="spinner xs"></span>` : st === "unlisted" ? "" : st === "error" ? "!" : fmt.format(child.count);
      html += `<div class="fv-row dir${withheldDir(p) ? " withheld" : ""}" role="treeitem" tabindex="-1" aria-expanded="${isOpen}" data-dir="${esc(p)}" style="--d:${depth}"
        ${st === "unlisted" ? 'title="Listed when opened"' : ""}>${icon("chevronRight", 13, "chev")}${icon(isOpen ? "folderOpen" : "folder", 14)}<span class="nm">${esc(name)}</span><em>${count}</em></div>`;
      if (isOpen) html += `<div role="group">${st === "error" ? `<p class="fv-none" style="padding-left:${24 + (depth + 1) * 14}px">Couldn't list this folder.</p>` : ""}${rowsOf(child, p + "/", depth + 1)}</div>`;
    }
    for (const f of [...node.files].sort((a, b) => base(a.path).localeCompare(base(b.path)))) html += fileRow(f, depth, esc(base(f.path)));
    return html;
  }
  const fileRow = (f, depth, label) => `<div class="fv-row fv-file${f.withheld ? " withheld" : ""}" role="treeitem" tabindex="-1" aria-selected="${f.path === current}"
      data-path="${esc(f.path)}" style="--d:${depth}" ${f.withheld ? 'title="Part of the reference solution: not shown"' : ""}>
      ${icon(fileIcon(f), 14)}<span class="nm">${label}</span><em>${f.withheld ? "hidden" : bytes(f.size)}</em></div>`;
  function renderTree() {
    if (query) {
      const q = query.toLowerCase();
      const hits = files.filter((f) => f.path.toLowerCase().includes(q));
      treeEl.innerHTML = hits.length ? hits.slice(0, 500).map((f) => {
        const i = f.path.toLowerCase().indexOf(q);
        const label = `${esc(f.path.slice(0, i))}<mark>${esc(f.path.slice(i, i + q.length))}</mark>${esc(f.path.slice(i + q.length))}`;
        return fileRow(f, 0, label);
      }).join("") : `<p class="fv-none">No file matches “${esc(query)}”.</p>`;
    } else treeEl.innerHTML = rowsOf(tree, "", 0);
    const sel = treeEl.querySelector('[aria-selected="true"]') || treeEl.querySelector(".fv-row");
    if (sel) sel.tabIndex = 0;
  }
  const visibleRows = () => [...treeEl.querySelectorAll(".fv-row")];
  function focusRow(r) {
    if (!r) return;
    treeEl.querySelectorAll('.fv-row[tabindex="0"]').forEach((x) => (x.tabIndex = -1));
    r.tabIndex = 0;
    r.focus({ preventScroll: true });
    r.scrollIntoView({ block: "nearest" });
  }
  function toggleDir(p, force) {
    const on = force ?? !open.has(p);
    on ? open.add(p) : open.delete(p);
    renderTree();
    focusRow(treeEl.querySelector(`[data-dir="${CSS.escape(p)}"]`));
    if (on && lazy.get(p) === "unlisted") listFolder(p);
  }
  const withheldDir = (p) => /(^|\/)(solutions?|gold|answers?|oracle)(\/|$)/i.test(p);
  async function listFolder(p) {
    if (!listDir) return;
    lazy.set(p, "loading");
    renderTree();
    try {
      const entries = await listDir(p);
      for (const e of entries) {
        if (e.dir) { const d = e.path.replace(/\/$/, ""); if (!lazy.has(d)) lazy.set(d, "unlisted"); }
        else if (!byPath.has(e.path)) files.push({ path: e.path, size: e.size, withheld: e.withheld });
      }
      lazy.set(p, "listed");
    } catch { lazy.set(p, "error"); }
    byPath = new Map(files.map((f) => [f.path, f]));
    tree = buildTree(files, [...lazy.keys()]);
    const had = document.activeElement?.dataset?.dir === p;
    renderTree();
    if (had) focusRow(treeEl.querySelector(`[data-dir="${CSS.escape(p)}"]`));
  }
  function reveal(p) {   // open the folders on the way to a file
    const parts = p.split("/");
    for (let i = 1; i < parts.length; i++) open.add(parts.slice(0, i).join("/"));
  }

  // ── the open file ──
  function setBar(f, d) {
    const parts = f.path.split("/");
    el.querySelector(".fv-path").innerHTML = `${parts.slice(0, -1).map((x) => `<span>${esc(x)}</span><i>/</i>`).join("")}<b>${esc(parts.at(-1))}</b>`;
    el.querySelector(".fv-path").title = f.path;
    const lang = d?.text != null ? langOf(f.path, d.text.slice(0, 120)) : null;
    const meta = [d?.text != null ? `${fmt.format(lineCount(d.text))} line${lineCount(d.text) === 1 ? "" : "s"}` : null, bytes(d?.size ?? f.size), LANG_NAMES[lang]].filter(Boolean);
    el.querySelector(".fv-meta").textContent = f.withheld || d?.withheld ? "not shown" : meta.join(" · ");
    const hub = el.querySelector('[data-act="hub"]'), url = f.withheld ? null : hubUrl(f.path);
    hub.hidden = !url;
    if (url) hub.href = url;
    const isMd = /\.(md|markdown)$/i.test(f.path) && d?.text != null;
    el.querySelector(".fv-mdseg").hidden = !isMd;
    el.querySelectorAll("[data-md]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.md === mdMode)));
    const textual = d?.text != null && !(isMd && mdMode === "preview");
    el.querySelector('[data-act="wrap"]').disabled = !textual;
    el.querySelector('[data-act="copy"]').disabled = d?.text == null;
    pick.querySelector(".fv-pick-p").textContent = f.path;
  }

  function placeholder(ic, title, text = "", action = "") {
    return `<div class="fv-empty">${icon(ic, 22)}<b>${title}</b>${text ? `<p>${text}</p>` : ""}${action}</div>`;
  }

  function renderFile(f, d) {
    setBar(f, d);
    const hub = f.withheld ? null : hubUrl(f.path);
    const onHub = hub ? `<a class="btn sm" href="${esc(hub)}" target="_blank" rel="noopener">${icon("external", 13)}Open on the Hub</a>` : "";
    if (f.withheld || d.withheld) { body.innerHTML = placeholder("lock", "Not shown", "Part of the reference solution or the expected answer. A browsing tool is not the place to hand those out."); return; }
    if (d.error) { body.innerHTML = placeholder("alert", "Couldn't open this file", esc(d.error), onHub); return; }
    if (d.binary) {
      const img = IMAGE.test(f.path) && rawUrl(f.path);
      body.innerHTML = img ? `<div class="fv-img"><img src="${esc(img)}" alt="${esc(base(f.path))}"></div>`
        : placeholder("archive", "Binary file", `${bytes(d.size ?? f.size)}, not shown here.`, onHub);
      return;
    }
    const text = d.text ?? "";
    if (!text.length) { body.innerHTML = placeholder("file", "This file is empty"); return; }
    const isMd = /\.(md|markdown)$/i.test(f.path);
    const tail = d.truncated ? `<div class="fv-trunc">${icon("info", 13)}Showing the first ${bytes(text.length)} of ${bytes(d.size)}.${hub ? ` <a href="${esc(hub)}" target="_blank" rel="noopener">The whole file is on the Hub.</a>` : ""}</div>` : "";
    if (isMd && mdMode === "preview") { body.innerHTML = `<div class="fv-md prose">${md(text)}</div>${tail}`; body.scrollTop = 0; return; }
    const lang = text.length <= HIGHLIGHT_MAX ? langOf(f.path, text.slice(0, 120)) : null;
    const rows = highlight(text.endsWith("\n") ? text.slice(0, -1) : text, lang);
    body.innerHTML = `<div class="fv-code${wrap ? " fv-wrap" : ""}" role="region" aria-label="${esc(f.path)}">${rows.map((h, i) =>
      `<div class="ln" data-n="${i + 1}"><a class="n" href="#" data-line="${i + 1}" tabindex="-1" aria-hidden="true">${i + 1}</a><span class="c">${h || " "}</span></div>`).join("")}</div>${tail}`;
    paintLines(true);
  }

  function paintLines(scroll) {
    body.querySelectorAll(".ln.hl").forEach((x) => x.classList.remove("hl"));
    const r = parseLines(lines);
    if (!r) { if (scroll) body.scrollTop = 0; return; }
    const all = body.querySelectorAll(".ln");
    for (let i = r[0]; i <= Math.min(r[1], all.length); i++) all[i - 1].classList.add("hl");
    if (scroll && all[r[0] - 1]) body.scrollTop = Math.max(0, all[r[0] - 1].offsetTop - 48);
  }

  async function openFile(p, ln = null, { focus = false, silent = false } = {}) {
    const f = byPath.get(p);
    if (!f) return;
    current = p; lines = ln;
    reveal(p);
    if (!query) renderTree();
    else treeEl.querySelectorAll(".fv-row.fv-file").forEach((r) => r.setAttribute("aria-selected", String(r.dataset.path === p)));
    const row = treeEl.querySelector(`.fv-row.fv-file[data-path="${CSS.escape(p)}"]`);
    if (row) { treeEl.querySelectorAll('.fv-row[tabindex="0"]').forEach((x) => (x.tabIndex = -1)); row.tabIndex = 0; row.scrollIntoView({ block: "nearest" }); if (focus) row.focus({ preventScroll: true }); }
    el.classList.remove("picking"); pick.setAttribute("aria-expanded", "false");
    if (!silent) onChange(current, lines);   // the file opened by default isn't put in the page's URL
    if (f.withheld) { renderFile(f, { withheld: true }); return; }
    setBar(f, null);
    const t = setTimeout(() => { if (current === p) body.innerHTML = `<div class="fv-loading">${Array.from({ length: 9 }, (_, i) => `<i style="width:${[62, 40, 75, 30, 55, 68, 22, 47, 60][i]}%"></i>`).join("")}</div>`; }, 120);
    const d = await get(p);
    clearTimeout(t);
    if (current !== p) return;   // another file was picked meanwhile
    renderFile(f, d);
  }

  function setFull(on) {
    full = on;
    el.classList.toggle("full", on);
    document.body.classList.toggle("fv-locked", on);
    const b = el.querySelector('[data-act="full"]');
    b.innerHTML = icon(on ? "minimize" : "maximize", 15);
    b.title = on ? "Close full view (Esc)" : "Full view (Esc to close)";
    if (on) { el.insertAdjacentHTML("beforebegin", '<div class="fv-scrim"></div>'); }
    else root.querySelector(".fv-scrim")?.remove();
  }

  // ── events ──
  el.addEventListener("click", async (e) => {
    const t = e.target;
    const row = t.closest(".fv-row");
    if (row) { if (row.dataset.dir) toggleDir(row.dataset.dir); else openFile(row.dataset.path); return; }
    if (t.closest(".fv-pick")) { const on = !el.classList.contains("picking"); el.classList.toggle("picking", on); pick.setAttribute("aria-expanded", String(on)); return; }
    const n = t.closest("[data-line]");
    if (n) {
      e.preventDefault();
      const k = +n.dataset.line, r = parseLines(lines);
      lines = e.shiftKey && r ? `${Math.min(r[0], k)}-${Math.max(r[1], k)}` : (r && r[0] === k && r[1] === k ? null : String(k));
      paintLines(false);
      onChange(current, lines);
      return;
    }
    const m = t.closest("[data-md]");
    if (m) { mdMode = m.dataset.md; storage.set("fv-md", mdMode); renderFile(byPath.get(current), await get(current)); return; }
    const act = t.closest("[data-act]")?.dataset.act;
    if (act === "wrap") {
      wrap = !wrap; storage.set("fv-wrap", wrap ? "1" : "0");
      t.closest("[data-act]").setAttribute("aria-pressed", String(wrap));
      body.querySelector(".fv-code")?.classList.toggle("fv-wrap", wrap);
    } else if (act === "copy") {
      const d = await get(current);
      if (d?.text == null) return;
      navigator.clipboard?.writeText(d.text).then(() => toast(d.truncated ? "Copied the part shown" : "File copied"), () => toast("Couldn't copy"));
    } else if (act === "link") {
      onChange(current, lines);
      navigator.clipboard?.writeText(location.href).then(() => toast("Link copied"), () => toast("Couldn't copy"));
    } else if (act === "full") setFull(!full);
  });
  root.addEventListener("click", (e) => { if (e.target.classList.contains("fv-scrim")) setFull(false); });
  el.addEventListener("pointerover", (e) => {
    const row = e.target.closest(".fv-row.fv-file:not(.withheld)");
    clearTimeout(hoverT);
    if (row) hoverT = setTimeout(() => get(row.dataset.path), 140);
  });
  treeEl.addEventListener("keydown", (e) => {
    const rows = visibleRows(), i = rows.indexOf(document.activeElement), r = rows[i];
    if (i < 0) return;
    const k = e.key;
    if (k === "ArrowDown") focusRow(rows[i + 1]);
    else if (k === "ArrowUp") focusRow(rows[i - 1]);
    else if (k === "Home") focusRow(rows[0]);
    else if (k === "End") focusRow(rows.at(-1));
    else if (k === "ArrowRight" && r.dataset.dir) { if (!open.has(r.dataset.dir)) toggleDir(r.dataset.dir, true); else focusRow(rows[i + 1]); }
    else if (k === "ArrowLeft") {
      if (r.dataset.dir && open.has(r.dataset.dir)) toggleDir(r.dataset.dir, false);
      else { const parent = (r.dataset.dir || r.dataset.path).split("/").slice(0, -1).join("/"); if (parent) focusRow(treeEl.querySelector(`[data-dir="${CSS.escape(parent)}"]`)); }
    } else if (k === "Enter" || k === " ") { if (r.dataset.dir) toggleDir(r.dataset.dir); else openFile(r.dataset.path, null, { focus: true }); }
    else return;
    e.preventDefault();
  });
  filterIn?.addEventListener("input", () => { query = filterIn.value.trim(); renderTree(); });
  filterIn?.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); focusRow(treeEl.querySelector(".fv-row")); }
    if (e.key === "Enter") { const first = treeEl.querySelector(".fv-row.fv-file"); if (first) openFile(first.dataset.path); }
    if (e.key === "Escape" && filterIn.value) { e.stopPropagation(); filterIn.value = ""; query = ""; renderTree(); }
  });
  const onKey = (e) => {
    if (e.key === "Escape" && full) { setFull(false); return; }
    if (e.key === "/" && full && document.activeElement !== filterIn && filterIn) { e.preventDefault(); filterIn.focus(); }
  };
  document.addEventListener("keydown", onKey);

  const first = (initial && byPath.has(initial) && initial) || ["instruction.md", "README.md", "task.toml"].find((p) => byPath.has(p) && !byPath.get(p).withheld)
    || files.find((f) => !f.withheld)?.path || files[0]?.path;
  renderTree();
  if (first) openFile(first, initial === first ? line : null, { silent: initial !== first });
  else body.innerHTML = placeholder("folder", "No files");

  return {
    open: (p, ln) => openFile(p, ln),
    destroy: () => { document.removeEventListener("keydown", onKey); document.body.classList.remove("fv-locked"); },
  };
}

// A read-only highlighted block with line numbers, for code shown inline on a page (test.sh, a Dockerfile).
export function codeBlock(text, path, { tall = false } = {}) {
  const t = String(text ?? "").replace(/\n$/, "");
  const rows = highlight(t, t.length <= HIGHLIGHT_MAX ? langOf(path, t.slice(0, 120)) : null);
  return `<div class="fv-block${tall ? " tall" : ""}"><div class="fv-code">${rows.map((h, i) => `<div class="ln"><span class="n" aria-hidden="true">${i + 1}</span><span class="c">${h || " "}</span></div>`).join("")}</div></div>`;
}
